import logging
import pickle
from typing import Literal, Tuple
from pydantic import PositiveInt, model_validator
import torch
import torch.nn as nn
from diffusers import FluxTransformer2DModel, AutoencoderKL
import einops
import random

from ...utils.common import unpack_bool_tensor, ensure_trainable
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

    use_src: bool = True
    src_downscale: int = 2
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
            use_src=self.use_src
        )
        super().install_modules(transformer)
        ensure_trainable(transformer.time_text_embed.local_guidance_embedder)
        if self.use_src:
            ensure_trainable(transformer.x_embedder_src)

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

        if "confidence" in batch:
            confidence = self._pack_confidence(batch["confidence"])
        else:
            confidence = None
            
        if self.use_src:
            src_hidden_states = self._pack_latents(batch["src"])
        else:
            src_hidden_states = None

        model_pred = transformer(
            hidden_states=input_latents,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            src_hidden_states=src_hidden_states,
            local_guidance=confidence,
            return_dict=False,
        )[0]

        return self._unpack_latents(model_pred, h, w)

    def train_step(
        self,
        transformer: FluxTransformer2DModel,
        vae: AutoencoderKL,
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

        if "hint" in batch and random.random() < self.chance_drop_hint:
            batch["hint"] = torch.zeros_like(batch["hint"])

        if not "clean_latents" in batch:
            batch["clean_latents"] = batch["tgt"]

        return super().train_step(transformer, vae, batch, timestep, guidance)
    
    def prepare_sample(self, transformer: FluxTransformer2DModel, vae: AutoencoderKL, batch: dict):
        if "coarse" not in batch:
            if "splat" in batch:
                batch["coarse"] = batch["splat"]
                batch["mask_coarse"] = batch["mask_splat"]
            else:
                batch["coarse"] = batch["affine"]
                batch["mask_coarse"] = batch["mask_affine"]

        if "hint" not in batch:
            batch["hint"] = torch.zeros_like(batch["noisy_latents"])
        
        if self.use_src:
            b, c, h, w = batch["noisy_latents"].shape
            batch["src"] = self._resize_latent_maybe(vae, batch["src"], h // self.src_downscale, w // self.src_downscale)
            
        if not "txt_ids" in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if not "img_ids" in batch:
            if self.use_src:
                src_ids = self._make_img_ids(batch["src"]) * self.src_downscale
                img_ids = self._make_img_ids(batch["noisy_latents"])
                batch["img_ids"] = torch.cat([src_ids, img_ids], dim=0)
            else:
                batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])
            
        return super().prepare_sample(transformer, vae, batch)

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

    def _resize_latent_maybe(self, vae, latent, th: int, tw: int) -> torch.Tensor:
        b, c, h, w = latent.shape
        if h == th and w == tw:
            return latent
        latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latent, return_dict=False)[0]
        image = torch.nn.functional.interpolate(
            image, size=(th * 8, tw * 8), mode="bilinear", align_corners=False
        )
        latent = vae.encode(image).latent_dist.sample()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        return latent