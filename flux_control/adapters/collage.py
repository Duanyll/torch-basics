import logging
from typing import Literal, Tuple
from pydantic import PositiveInt
import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
import einops
import random

from ..utils.common import meshgrid_to_ij
from .d_concat import DConcatAdapter

logger = logging.Logger(__name__)


class CollageContextEmbedder(nn.Module):
    def __init__(
        self,
        original_embedder: nn.Module,
        color_dim: int = 3,
        inner_dim: int = 3072,
    ):
        super().__init__()
        self.text_embedder = original_embedder
        self.color_dim = color_dim
        self.inner_dim = inner_dim
        self.color_embedder = nn.Linear(color_dim, inner_dim)
        self.color_embedder.weight.data.zero_()
        self.color_embedder.bias.data.zero_()

    def forward(self, encoder_hidden_states):
        if isinstance(encoder_hidden_states, tuple):
            x, color = encoder_hidden_states
            x = self.text_embedder(x)
            color = self.color_embedder(color)
            concated, _ = einops.pack((x, color), "b * d")
            return concated
        else:
            return self.text_embedder(encoder_hidden_states)


class CollageAdapter(DConcatAdapter):
    input_dimension: PositiveInt = 448
    lora_layers: Literal["all-linear"] | list[str] = [
        # Input layers
        "x_embedder",
        "context_embedder.text_embedder",
        # Dual stream layers
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

    chance_dropout_edge: float = 0.1
    chance_dropout_color: float = 0.1

    def install_modules(self, transformer: FluxTransformer2DModel):
        with torch.no_grad():
            orig_context_embedder = transformer.context_embedder
            new_context_embedder = CollageContextEmbedder(
                orig_context_embedder,
                color_dim=3,
                inner_dim=orig_context_embedder.out_features,
            )
            transformer.context_embedder = new_context_embedder  # type: ignore

        super().install_modules(transformer)
        transformer.context_embedder.color_embedder.requires_grad_(True)

    def save_model(self, transformer: FluxTransformer2DModel) -> dict:
        layers_to_save = super().save_model(transformer)
        for name, param in transformer.named_parameters():
            if "color_embedder" in name:
                layers_to_save[name] = param
        return layers_to_save

    def load_model(self, transformer: FluxTransformer2DModel, state_dict: dict):
        super().load_model(transformer, state_dict)
        color_dict = {
            k.replace("transformer.", ""): v
            for k, v in state_dict.items()
            if "color_embedder" in k
        }
        if len(color_dict) > 0:
            transformer.load_state_dict(color_dict)
        else:
            logger.warning(
                "No color embedder state dict found in the checkpoint. "
                "This may be due to a mismatch in the model architecture."
            )

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
            - `collage_control_latents`: `[B, C, H, W]` VAE Encoded control condition image at VAE latent resolution.
            - `collage_alpha`: `[B, H, W]` Collage alpha mask at original image resolution.
              1 for visible, 0 for invisible (to inpaint).
            - `edge_control_latents`: `[B, C, H, W]` VAE Encoded dexined control condition image at VAE latent resolution.

            Optional keys are:
            - `txt_ids`: `[B, N, 3]` Used for adding positional embeddings to the text embeddings.
               Usually all zeros. Will be calculated if not present.
            - `img_ids`: `[B, N, 3]` Used for adding positional embeddings to the image embeddings.
               Will be calculated if not present.
            - `palettes`: `[B, N, 3]` Color palettes for each text prompt.
            - `palette_locations`: `[B, N, 2]` Color palette locations for each text prompt, in F.grid_sample coordinates.

        timestep : torch.Tensor([B])
            The current timestep. Range is [0, 1].
        guidance : torch.Tensor([B]) | None
            The guidance strength if required by the base model.

        Returns
        -------
        torch.Tensor([B, C, H, W])
            The predicted velocity in latent space.
        """

        # Hotfix: Remove extra batch dim in pooled_prompt_embeds and prompt_embeds due to mistake in dataset preprocessing
        if batch["pooled_prompt_embeds"].ndim == 3:
            batch["pooled_prompt_embeds"] = batch["pooled_prompt_embeds"].squeeze(1)
        if batch["prompt_embeds"].ndim == 4:
            batch["prompt_embeds"] = batch["prompt_embeds"].squeeze(1)

        b, c, h, w = batch["noisy_latents"].shape
        h_len = h // self.patch_size
        w_len = w // self.patch_size

        input_latents, _ = einops.pack(
            (
                self._pack_latents(batch["noisy_latents"]),
                self._pack_latents(batch["collage_control_latents"]),
                self._pack_latents(self._pack_mask(1 - batch["collage_alpha"])),
                self._pack_latents(batch["edge_control_latents"]),
            ),
            "b n *",
        )

        if not "img_ids" in batch:
            batch["img_ids"] = self._make_img_ids(batch["noisy_latents"])

        if not "txt_ids" in batch:
            if "palettes" in batch:
                batch["txt_ids"], _ = einops.pack(
                    (
                        self._make_txt_ids(batch["prompt_embeds"]),
                        self._make_color_txt_ids(
                            batch["palette_locations"], h_len, w_len
                        ),
                    ),
                    "* d",
                )
            else:
                batch["txt_ids"] = self._make_txt_ids(batch["prompt_embeds"])

        if "palettes" in batch:
            encoder_hidden_states = (batch["prompt_embeds"], batch["palettes"])
        else:
            encoder_hidden_states = batch["prompt_embeds"]

        model_pred = transformer(
            hidden_states=input_latents,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"],
            encoder_hidden_states=encoder_hidden_states,
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
        # Chance to dropout edge and color
        if random.random() < self.chance_dropout_edge:
            batch["edge_control_latents"] = torch.zeros_like(
                batch["edge_control_latents"]
            )
        if random.random() < self.chance_dropout_color and "palettes" in batch:
            batch["palettes"] = torch.zeros_like(batch["palettes"])
            batch["palette_locations"] = torch.zeros_like(batch["palette_locations"])
        # Hotfix: Fill NaN values in "palettes" and "palette_locations" with zeros
        if "palettes" in batch:
            batch["palettes"] = torch.nan_to_num(batch["palettes"])
            batch["palette_locations"] = torch.nan_to_num(batch["palette_locations"])
        return super().train_step(transformer, batch, timestep, guidance)

    def _pack_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Pack the mask with PixelShuffle-like operation, as seen in Flux.1 Fill
        """
        return einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)

    def _make_color_txt_ids(
        self, palette_locations: torch.Tensor, h_len, w_len
    ) -> torch.Tensor:
        """
        Create the color text ids for the palette locations.
        """
        b, n, d = palette_locations.shape
        dtype = palette_locations.dtype
        if b != 1:
            raise ValueError(
                "CollageAdapter only supports batch size of 1. This limitation is due to the "
                "internal handling of txt_ids in FluxTransformer2DModel."
            )
        mesh = einops.rearrange(palette_locations.float(), "1 n d -> n d")
        ij = meshgrid_to_ij(mesh, h=h_len, w=w_len)
        color_txt_ids, _ = einops.pack(
            (torch.zeros((n, 1), dtype=mesh.dtype, device=mesh.device), ij), "n *"
        )
        return color_txt_ids.to(dtype=dtype)
