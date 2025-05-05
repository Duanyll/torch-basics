from functools import partial
import types
from typing import Any, Literal, Tuple
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.embeddings import TimestepEmbedding


def patch_time_text_embed_init(module):
    # Copy timestep_embedder as local_guidance_embedder
    embedding_dim = module.timestep_embedder.linear_1.out_features
    module.local_guidance_embedder = TimestepEmbedding(
        in_channels=256, time_embed_dim=embedding_dim
    )
    orig_state_dict = module.timestep_embedder.state_dict()
    clone_state_dict = {k: v.detach().clone() for k, v in orig_state_dict.items()}
    module.local_guidance_embedder.load_state_dict(clone_state_dict)
    module.local_guidance_embedder.linear_1.weight.data.zero_()
    module.local_guidance_embedder.linear_1.bias.data.zero_()


def patch_time_text_embed_forward(
    self,
    timestep,
    guidance,
    pooled_projection,
    local_guidance=None,
    local_guidance_pad=None,
):
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype)
    )  # (N, D)

    guidance_proj = self.time_proj(guidance)
    guidance_emb = self.guidance_embedder(
        guidance_proj.to(dtype=pooled_projection.dtype)
    )  # (N, D)

    time_guidance_emb = timesteps_emb + guidance_emb

    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = time_guidance_emb + pooled_projections

    if local_guidance is not None:
        b, n = local_guidance.shape
        local_guidance = rearrange(local_guidance, "b n -> (b n)")
        local_guidance_proj = self.time_proj(local_guidance)
        local_guidance_emb = self.local_guidance_embedder(
            local_guidance_proj.to(dtype=pooled_projection.dtype)
        )
        local_guidance_emb = rearrange(local_guidance_emb, "(b n) d -> b n d", b=b, n=n)
        if local_guidance_pad is not None:
            b, n, d = local_guidance_emb.shape
            device, dtype = local_guidance_emb.device, local_guidance_emb.dtype
            local_guidance_emb = torch.cat(
                [
                    torch.zeros((b, local_guidance_pad, d), device=device, dtype=dtype),
                    local_guidance_emb,
                ],
                dim=1,
            )
        local_conditioning = conditioning[:, None, :] + local_guidance_emb
        return conditioning, local_conditioning
    else:
        return conditioning


def patch_ada_layer_norm_forward(self, x, emb):
    if emb.dim() == 3:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=-1
        )
        x = self.norm(x) * (1 + scale_msa) + shift_msa
    else:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def patch_transformer_block_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    use_lge=True,
):
    if not use_lge and isinstance(temb, tuple):
        temb = temb[0]
    if isinstance(temb, tuple):
        temb, local_temb = temb
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=local_temb
        )
    else:
        local_temb = None
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        gate_msa = gate_msa[:, None]
        shift_mlp = shift_mlp[:, None]
        scale_mlp = scale_mlp[:, None]
        gate_mlp = gate_mlp[:, None]

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )
    joint_attention_kwargs = joint_attention_kwargs or {}
    # Attention.
    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp * ff_output

    hidden_states = hidden_states + ff_output
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + ip_attn_output

    # Process attention outputs for the `encoder_hidden_states`.

    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = (
        encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    )
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def patch_ada_layer_norm_single_forward(
    self,
    x,
    emb,
):
    if isinstance(emb, tuple):
        has_local_emb = True
        emb, local_emb = emb
        image_len = local_emb.shape[1]
        x_img = x[:, -image_len:]
        x = x[:, :-image_len]
    else:
        has_local_emb = False

    emb = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
    x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

    if has_local_emb:
        local_emb = self.linear(self.silu(local_emb))
        local_shift_msa, local_scale_msa, local_gate_msa = local_emb.chunk(3, dim=-1)
        x_img = self.norm(x_img) * (1 + local_scale_msa) + local_shift_msa
        x = torch.cat([x, x_img], dim=1)
        return x, (gate_msa, local_gate_msa)
    else:
        return x, gate_msa


def patch_single_transformer_block_forward(
    self,
    hidden_states,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    use_lge=True,
):
    if not use_lge and isinstance(temb, tuple):
        temb = temb[0]

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    proj_out = self.proj_out(hidden_states)

    if isinstance(gate, tuple):
        gate, local_gate = gate
        image_len = local_gate.shape[1]
        x_txt = proj_out[:, :-image_len]
        x_img = proj_out[:, -image_len:]
        x_txt = gate[:, None, :] * x_txt
        x_img = local_gate * x_img
        hidden_states = torch.cat([x_txt, x_img], dim=1)
    else:
        hidden_states = gate[:, None, :] * proj_out

    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states


def patch_norm_out_forward(self, x, conditioning_embedding) -> torch.Tensor:
    if isinstance(conditioning_embedding, tuple):
        conditioning_embedding, local_conditioning_embedding = conditioning_embedding
    # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
    emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = torch.chunk(emb, 2, dim=1)
    x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    return x


def patch_transformer_forward(
    self,
    hidden_states,
    encoder_hidden_states: Any,
    pooled_projections,
    timestep,
    img_ids,
    txt_ids,
    guidance=None,
    joint_attention_kwargs=None,
    return_dict: bool = True,
    src_hidden_states=None,
    local_guidance=None,
):
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """

    hidden_states = self.x_embedder(hidden_states)
    if src_hidden_states is not None:
        src_hidden_states = self.x_embedder_src(src_hidden_states)
        hidden_states = torch.cat([src_hidden_states, hidden_states], dim=1)
        src_hidden_states_len = src_hidden_states.shape[1]
        del src_hidden_states
    else:
        src_hidden_states_len = 0

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    if local_guidance is not None:
        local_guidance = local_guidance.to(hidden_states.dtype) * 1000

    temb = self.time_text_embed(
        timestep,
        guidance,
        pooled_projections,
        local_guidance,
        local_guidance_pad=src_hidden_states_len,
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if (
        joint_attention_kwargs is not None
        and "ip_adapter_image_embeds" in joint_attention_kwargs
    ):
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                use_reentrant=False,
            )  # type: ignore

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                use_reentrant=False,
            )

        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]  # type: ignore

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    output = output[:, src_hidden_states_len:, ...]

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


@torch.no_grad()
def apply_patches(
    transformer,
    double_layers: bool | list[int] = True,
    single_layers: bool | list[int] = False,
    use_src: bool = True,
):
    """
    Apply patches to the FluxTransformer2DModel class.
    """
    if use_src:
        x_emb_in = 64
        x_emb_out = transformer.x_embedder.out_features
        new_linear = nn.Linear(
            x_emb_in,
            x_emb_out,
            bias=transformer.x_embedder.bias is not None,
            dtype=transformer.dtype,
            device=transformer.device,
        )
        new_linear.weight.data = transformer.x_embedder.weight[:, :64].detach().clone()
        new_linear.bias.data = transformer.x_embedder.bias.detach().clone()
        transformer.x_embedder_src = new_linear

    patch_time_text_embed_init(transformer.time_text_embed)
    transformer.time_text_embed.forward = types.MethodType(
        patch_time_text_embed_forward, transformer.time_text_embed
    )

    def use_lge(spec, i):
        if isinstance(spec, bool):
            return spec
        return i in spec

    for i, block in enumerate(transformer.transformer_blocks):
        block.norm1.forward = types.MethodType(
            patch_ada_layer_norm_forward, block.norm1
        )
        block.norm1_context.forward = types.MethodType(
            patch_ada_layer_norm_forward, block.norm1_context
        )
        block.forward = types.MethodType(
            partial(patch_transformer_block_forward, use_lge=use_lge(double_layers, i)),
            block,
        )
    for i, block in enumerate(transformer.single_transformer_blocks):
        block.norm.forward = types.MethodType(
            patch_ada_layer_norm_single_forward, block.norm
        )
        block.forward = types.MethodType(
            partial(
                patch_single_transformer_block_forward,
                use_lge=use_lge(single_layers, i),
            ),
            block,
        )
    transformer.norm_out.forward = types.MethodType(
        patch_norm_out_forward, transformer.norm_out
    )
    transformer.forward = types.MethodType(patch_transformer_forward, transformer)
