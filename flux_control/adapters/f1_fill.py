from typing import Tuple

import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL
from pydantic import BaseModel, PositiveInt
from einops import repeat, rearrange, reduce, pack

from ..utils.common import unpack_bool_tensor
from .base import BaseAdapter


class Flux1FillAdapter(BaseAdapter):
    """
    Adapter for the FLUX.1 fill model.
    """
    enforce_mask: bool = False

    def predict_velocity(self, transformer: FluxTransformer2DModel, batch: dict, timestep: torch.Tensor, guidance: torch.Tensor | None) -> torch.Tensor:
        b, c, h, w = batch["noisy_latents"].shape
        h_len = h // self.patch_size
        w_len = w // self.patch_size

        mask = 1 - self._pack_mask(batch["mask_coarse"])
        
        inputs = pack([
            self._pack_latents(batch["noisy_latents"]),
            self._pack_latents(batch["coarse"]),
            self._pack_latents(mask),
        ], "b n *")

        model_pred = transformer(
            hidden_states=inputs,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        velocity = self._unpack_latents(model_pred, h, w)
        if self.enforce_mask:
            velocity = velocity.to(torch.float32)
            target_velocity = (batch["coarse"].float() - batch["noisy_latents"].float()) / timestep
            velocity = velocity * mask + target_velocity * (1 - mask)
        return velocity

    def prepare_sample(self, transformer: FluxTransformer2DModel, vae: AutoencoderKL, batch: dict):
        if "coarse" not in batch:
            if "affine" in batch:
                batch["coarse"] = batch["affine"]
                batch["mask_coarse"] = batch["mask_affine"]
            else:
                batch["coarse"] = batch["splat"]
                batch["mask_coarse"] = batch["mask_splat"]

        return super().prepare_sample(transformer, vae, batch)

    def _pack_mask(self, mask) -> torch.Tensor:
        """
        Pack the mask with PixelShuffle-like operation, as seen in Flux.1 Fill
        """
        if isinstance(mask, tuple):
            mask = unpack_bool_tensor(*mask)
        mask = mask.to(torch.bfloat16)
        return rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
