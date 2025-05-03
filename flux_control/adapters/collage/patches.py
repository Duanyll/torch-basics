import logging
import types
from typing import Literal, Tuple
from einops import rearrange
import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.embeddings import TimestepEmbedding
import random


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


def patch_time_text_embed_forward(self, timestep, guidance, pooled_projection):
    if isinstance(pooled_projection, tuple):
        has_local_guidance = True
        pooled_projection, local_guidance = pooled_projection
        # Local guidance: [B, N, D]
    else:
        has_local_guidance = False
        local_guidance = None
        # Local guidance: None

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

    if has_local_guidance:
        b, n = local_guidance.shape
        local_guidance = rearrange(local_guidance, "b n -> (b n)")
        local_guidance_proj = self.time_proj(local_guidance)
        local_guidance_emb = self.local_guidance_embedder(
            local_guidance_proj.to(dtype=pooled_projection.dtype)
        )
        local_guidance_emb = rearrange(local_guidance_emb, "(b n) d -> b n d", b=b, n=n)
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
):
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


def patch_single_transformer_block_forward(
    self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
):
    if isinstance(temb, tuple):
        temb, local_temb = temb

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
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
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


def apply_patches(transformer):
    """
    Apply patches to the FluxTransformer2DModel class.
    """
    patch_time_text_embed_init(transformer.time_text_embed)
    transformer.time_text_embed.forward = types.MethodType(
        patch_time_text_embed_forward, transformer.time_text_embed
    )
    for block in transformer.transformer_blocks:
        block.norm1.forward = types.MethodType(
            patch_ada_layer_norm_forward, block.norm1
        )
        block.norm1_context.forward = types.MethodType(
            patch_ada_layer_norm_forward, block.norm1_context
        )
        block.forward = types.MethodType(patch_transformer_block_forward, block)
    for block in transformer.single_transformer_blocks:
        block.forward = types.MethodType(patch_single_transformer_block_forward, block)
    transformer.norm_out.forward = types.MethodType(
        patch_norm_out_forward, transformer.norm_out
    )