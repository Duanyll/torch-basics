import logging
from typing import Literal
from pydantic import field_validator, PositiveInt
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from einops import repeat, rearrange, reduce
from .peft_lora import PeftLoraAdapter

logger = logging.Logger(__name__)


class DConcatAdapter(PeftLoraAdapter):
    """
    Adapter for applying control to the model through concating the conditional image latent
    to the noisy input latent along the D dimension. It changes the shape of x_embedder layer.

    This is used by Flux.1 Canny and Flux.1 Depth models.
    """

    lora_layers: Literal["all-linear"] | list[str] = "all-linear"
    rank: PositiveInt = 128
    use_lora_bias: bool = True

    @field_validator("lora_layers")
    def _validate_lora_layers(cls, value):
        if value == "all-linear":
            return value
        if isinstance(value, list):
            # If x_embedder is not present, add it to the list and warn the user
            if "x_embedder" not in value:
                logger.warning(
                    "x_embedder is not present in lora_layers. Adding it to the list."
                )
                value.append("x_embedder")
            return value
        raise ValueError(
            "lora_layers must be either 'all-linear' or a list of strings."
        )

    def install_modules(self, transformer: FluxTransformer2DModel):
        # Change shape of x_embedder layer before loading LoRA
        with torch.no_grad():
            initial_input_channels: int = transformer.config.in_channels  # type: ignore
            new_linear = torch.nn.Linear(
                transformer.x_embedder.in_features * 2,
                transformer.x_embedder.out_features,
                bias=transformer.x_embedder.bias is not None,
                dtype=transformer.dtype,
                device=transformer.device,
            )
            new_linear.weight.zero_()
            new_linear.weight[:, :initial_input_channels].copy_(
                transformer.x_embedder.weight
            )
            if transformer.x_embedder.bias is not None:
                new_linear.bias.copy_(transformer.x_embedder.bias)
            transformer.x_embedder = new_linear

        super().install_modules(transformer)

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

        concatenated_noisy_model_input = torch.cat(
            (batch["noisy_latents"], batch["control_latents"]), dim=1
        )

        packed_noisy_model_input = self._pack_latents(concatenated_noisy_model_input)
        if batch["txt_ids"] is None:
            batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])
        if batch["img_ids"] is None:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=batch["prompt_embeds"],
            txt_ids=batch["txt_ids"],
            img_ids=batch["img_ids"],
            return_dict=False,
        )[0]
        
        return self._unpack_latents(model_pred, h, w)
