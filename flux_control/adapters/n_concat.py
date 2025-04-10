import logging
from typing import Literal
from pydantic import field_validator
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from einops import repeat, rearrange, reduce
from .peft_lora import PeftLoraAdapter

logger = logging.Logger(__name__)


class NConcatAdapter(PeftLoraAdapter):
    """
    Adapter for applying control to the model through concatenating the conditional image latent
    to the noisy input latent along the N dimension.

    This is used by the PhotoDoddle model.
    """

    def predict_velocity(
        self,
        transformer: FluxTransformer2DModel,
        batch: dict,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Predict velocity with the input batch at the given timestep.

        Parameters
        ----------
        transformer : FluxTransformer2DModel
            Adapted transformer model.
        batch : dict
            Input batch containing the data.

            Required keys are:
            - `noisy_latents`: `[B, C, H, W]` The pixel latents to denoise.
            - `control_latents`: `[B, C, H, W]` The VAE encoded control condition image.
            - `pooled_prompt_embeds`: `[B, D]` Pooled text embeddings from the CLIP text encoder.
            - `prompt_embeds`: `[B, N, D]` Text embeddings from triple text encoder.

            Optional keys are:
            - `txt_ids`: `[B, N, 3]` Used for adding positional embeddings to the text embeddings.
               Usually all zeros. Will be calculated if not present.
            - `img_ids`: `[B, N, 3]` Used for adding positional embeddings to the image embeddings.
               Will be calculated if not present.

        timestep : torch.Tensor([B])
            The current timestep. Range is [0, 1].
        guidance : torch.Tensor([B]) | None
            The guidance strength if required by the base model.

        Returns
        -------
        torch.Tensor([B, C, H, W])
            The predicted velocity in latent space.
        """

        b, c, h, w = batch["noisy_latents"].shape

        noisy_model_input = self._pack_latents(batch["noisy_latents"])
        control_model_input = self._pack_latents(batch["control_latents"])
        concatenated_model_input = torch.cat(
            (noisy_model_input, control_model_input), dim=1
        )
        if not "txt_ids" in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if not "img_ids" in batch:
            batch["img_ids"] = repeat(
                self._make_img_ids(batch["noisy_latents"]), "n d -> (r n) d", r=2
            )

        model_pred = transformer(
            hidden_states=concatenated_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]

        b, n, d = model_pred.shape
        model_pred = model_pred[:, : (n // 2), :]

        return self._unpack_latents(model_pred, h, w)
