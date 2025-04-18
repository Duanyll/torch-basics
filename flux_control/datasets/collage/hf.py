import torch
from typing import Any
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from einops import rearrange

pipe: Any = None
pipe_device: str = "cuda"


def load_hf_pipeline(device="cuda"):
    """Initialize the VAE model."""
    global pipe, pipe_device
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, transformer=None
    )
    pipe = pipe.to(device)
    pipe_device = device


def unload_vae_model():
    """Unload the VAE model."""
    global pipe
    if pipe is not None:
        del pipe
        pipe = None
        torch.cuda.empty_cache()


@torch.no_grad()
def encode_latents(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Encode the image into latent space.

    :param image: Input image [C, H, W]
    :return: Latent representation [C, H', W']
    """

    image = rearrange(image, "c h w -> 1 c h w")
    image = image * 2 - 1
    image = image.to(torch.bfloat16)
    latent = pipe.vae.encode(image).latent_dist.sample()
    latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latent = rearrange(latent, "1 c h w -> c h w")
    return latent


@torch.no_grad()
def decode_latents(
    latents: torch.Tensor,
) -> torch.Tensor:
    """
    Decode the latent representation back to image space.

    :param latents: Latent representation [C, H', W']
    :return: Reconstructed image [C, H, W]
    """

    latents = rearrange(latents, "c h w -> 1 c h w")
    latents = (latents / pipe.config.scaling_factor) + pipe.config.shift_factor
    image = pipe.decode(latents).sample
    image = (image + 1) / 2
    image = rearrange(image, "1 c h w -> c h w")
    return image


@torch.no_grad()
def encode_prompt(prompt: str):
    prompt = prompt.replace("video", "image")
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=pipe_device
    )
    return prompt_embeds, pooled_prompt_embeds
