from typing import Tuple

import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from pydantic import BaseModel, PositiveInt
from einops import repeat, rearrange, reduce


class BaseAdapter(BaseModel):
    """
    Base class for all control adapters.
    """

    patch_size: PositiveInt = 2

    def install_modules(self, transformer: FluxTransformer2DModel):
        """
        Create and initialize additional modules on the base model. Called after base model is
        created.

        :param transformer: The base transformer model.
        """
        pass

    def save_model(self, transformer: FluxTransformer2DModel) -> dict:
        """
        Decide which layers to save in the checkpoint. Will be wrapped and registered by
        `accelerator.register_save_state_pre_hook`.

        :param transformer: The adapted transformer model.
        :return: A state_dict containing the layers to save.
        """
        return {}

    def load_model(self, transformer: FluxTransformer2DModel, state_dict: dict):
        """
        Load the state_dict. Will be wrapped and registered by `accelerator.register_load_state_pre_hook`.

        :param transformer: The adapted transformer model.
        :param state_dict: The state_dict containing the layers to load.
        """
        pass

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
            - `pooled_prompt_embeds`: `[B, D]` Pooled text embeddings from the CLIP text encoder.
            - `prompt_embeds`: `[B, N, D]` Text embeddings from triple text encoder.

            Optional keys are:
            - `txt_ids`: `[B, N, 3]` Used for adding positional embeddings to the text embeddings.
               Usually all zeros. Will be calculated if not present.
            - `img_ids`: `[B, N, 3]` Used for adding positional embeddings to the image embeddings.
               Will be calculated if not present.

        timestep : torch.Tensor([B])
            The current timestep. Range is [0, 1], 0 for clean image, 1 for noise.
        guidance : torch.Tensor([B]) | None
            The guidance strength if required by the base model.

        Returns
        -------
        torch.Tensor([B, C, H, W])
            The predicted velocity in latent space.
        """
        b, c, h, w = batch["noisy_latents"].shape

        if not "txt_ids" in batch:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if not "img_ids" in batch:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        model_pred = transformer(
            hidden_states=self._pack_latents(batch["noisy_latents"]),
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
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
        """
        Run forward pass and compute loss for the given batch. May call `predict_velocity`
        internally.

        Parameters
        ----------
        transformer : FluxTransformer2DModel
            Adapted transformer model.
        batch : dict
            Input batch containing the data.

            Required keys are:
            - `clean_latents`: `[B, C, H, W]` The pixel latents to denoise.
            - `pooled_prompt_embeds`: `[B, D]` Pooled text embeddings from the CLIP text encoder.
            - `prompt_embeds`: `[B, N, D]` Text embeddings from triple text encoder.

            Optional keys are:
            - `txt_ids`: `[B, N, 3]` Used for adding positional embeddings to the text embeddings.
               Usually all zeros. Will be calculated if not present.
            - `img_ids`: `[B, N, 3]` Used for adding positional embeddings to the image embeddings.
               Will be calculated if not present.

        timestep : torch.Tensor([B])
            The current timestep. Range is [0, 1], 0 for clean image, 1 for noise.
        guidance : torch.Tensor([B]) | None
            The guidance strength if required by the base model.

        Returns
        -------
        torch.Tensor([B])
            Unweighted loss for each sample in the batch.
        """
        clean = batch["clean_latents"]
        noise = torch.randn_like(clean)
        t_batch = rearrange(timestep, "b -> b 1 1 1")
        noisy_latents = (1.0 - t_batch) * clean + t_batch * noise
        batch["noisy_latents"] = noisy_latents
        model_pred = self.predict_velocity(transformer, batch, timestep, guidance)
        target = noise - clean
        loss = reduce(
            (model_pred.float() - target.float()) ** 2, "b c h w -> b", reduction="mean"
        ) # Must use float() here
        return loss

    def _make_txt_ids(self, prompt_embeds):
        b, n, d = prompt_embeds.shape
        return torch.zeros(
            (n, 3), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )

    def _make_img_ids(self, pixel_latents):
        b, c, h, w = pixel_latents.shape
        h_len = h // self.patch_size
        w_len = w // self.patch_size
        img_ids = torch.zeros(
            (h_len, w_len, 3), dtype=pixel_latents.dtype, device=pixel_latents.device
        )
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.arange(
            h_len, dtype=img_ids.dtype, device=img_ids.device
        ).reshape(h_len, 1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.arange(
            w_len, dtype=img_ids.dtype, device=img_ids.device
        ).reshape(1, w_len)
        img_ids = rearrange(img_ids, "h w c -> (h w) c")
        return img_ids

    def _pack_latents(self, latents):
        return rearrange(
            latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def _unpack_latents(self, latents, h, w):
        return rearrange(
            latents,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )
