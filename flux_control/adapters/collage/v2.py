import logging
import pickle
from typing import Literal, Tuple
from pydantic import PositiveInt, model_validator
import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
import einops
import random

from ...utils.common import unpack_bool_tensor
from ..d_concat import DConcatAdapter
from .patches import apply_patches


class CollageAdapterV2(DConcatAdapter):
    # Latent, Coarse, Mask Corase, Foreground, Hint
    input_dimension: PositiveInt = 64 + 64 + 256 + 256 + 64
    lora_layers: Literal["all-linear"] | list[str] = [
        # Dual stream layers
        "norm1.linear",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
        # Single stream layers
        "proj_mlp",
        "proj_out",
        "norm.linear",
        # Output layers
        "norm_out.linear",
    ]
    rank: PositiveInt = 128
    gaussian_init_lora: bool = False
    use_lora_bias: bool = True
    lge_double_layers: bool | list[int] = True
    lge_single_layers: bool | list[int] = False

    use_foreground: bool = True
    use_hint: bool = True
    chance_drop_hint: float = 0.2
    chance_use_affine: float = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dimension = 384
        if self.use_foreground:
            self.input_dimension += 256
        if self.use_hint:
            self.input_dimension += 64

    def install_modules(self, transformer: FluxTransformer2DModel):
        apply_patches(
            transformer,
            double_layers=self.lge_double_layers,
            single_layers=self.lge_single_layers,
        )
        super().install_modules(transformer)

        lge_state_dict = (
            transformer.time_text_embed.local_guidance_embedder.state_dict()  # type: ignore
        )
        if not any("lora" in k for k in lge_state_dict.keys()):
            transformer.time_text_embed.local_guidance_embedder.requires_grad_(True)

    def predict_velocity(
        self,
        transformer: FluxTransformer2DModel,
        batch: dict,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:

        b, c, h, w = batch["noisy_latents"].shape
        h_len = h // self.patch_size
        w_len = w // self.patch_size

        if "coarse" not in batch:
            if "splat" in batch:
                batch["coarse"] = batch["splat"]
                batch["mask_coarse"] = batch["mask_splat"]
            else:
                batch["coarse"] = batch["affine"]
                batch["mask_coarse"] = batch["mask_affine"]

        if "hint" not in batch:
            batch["hint"] = torch.zeros_like(batch["noisy_latents"])

        inputs = [
            self._pack_latents(batch["noisy_latents"]),
            self._pack_latents(batch["coarse"]),
            self._pack_latents(1 - self._pack_mask(batch["mask_coarse"])),
        ]
        if self.use_foreground:
            inputs.append(self._pack_latents(self._pack_mask(batch["foreground"])))
        if self.use_hint:
            inputs.append(self._pack_latents(batch["hint"]))
        input_latents, _ = einops.pack(inputs, "b n *")

        if not "txt_ids" in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if not "img_ids" in batch:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        if "confidence" in batch:
            confidence = self._pack_confidence(batch["confidence"]) * 1000
            pooled_prompt_embeds = (batch["pooled_prompt_embeds"], confidence)
        else:
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]

        model_pred = transformer(
            hidden_states=input_latents,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        return self._unpack_latents(model_pred, h, w)

    def train_step(
        self,
        transformer: FluxTransformer2DModel,
        batch: dict,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        if not "coarse" in batch:
            if ("splat" in batch) and ("affine" in batch):
                use_affine = random.random() < self.chance_use_affine
            else:
                use_affine = "affine" in batch
            batch["coarse"] = batch["affine"] if use_affine else batch["splat"]
            batch["mask_coarse"] = (
                batch["mask_affine"] if use_affine else batch["mask_splat"]
            )

        if random.random() < self.chance_drop_hint:
            batch["hint"] = torch.zeros_like(batch["hint"])

        if not "clean_latents" in batch:
            batch["clean_latents"] = batch["tgt"]

        return super().train_step(transformer, batch, timestep, guidance)

    def _pack_mask(self, mask) -> torch.Tensor:
        """
        Pack the mask with PixelShuffle-like operation, as seen in Flux.1 Fill
        """
        if isinstance(mask, tuple):
            mask = unpack_bool_tensor(*mask)
        mask = mask.to(torch.bfloat16)
        return einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)

    def _pack_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        confidence = confidence.to(torch.bfloat16)
        return einops.rearrange(confidence, "b h w -> b (h w)")
