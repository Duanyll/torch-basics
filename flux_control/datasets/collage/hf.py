import torch
from typing import Any
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from einops import rearrange, repeat

pipe: Any = None
pipe_device: str = "cuda"


def _get_clip_prompt_embeds(
    self,
    prompt,
    num_images_per_prompt: int = 1,
    device=None,
):
    device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids

    prompt_embeds = self.text_encoder(
        text_input_ids.to(device), output_hidden_states=False
    )

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def load_hf_pipeline(device="cuda"):
    """Initialize the VAE model."""
    global pipe, pipe_device
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, transformer=None
    )
    # Patch the pipeline to remove warning
    pipe._get_clip_prompt_embeds = _get_clip_prompt_embeds.__get__(pipe)
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

    if image.ndim == 2:
        image = repeat(image, "h w -> c h w", c=3)

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
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents).sample
    image = (image + 1) / 2
    image = rearrange(image, "1 c h w -> c h w")
    return image


@torch.no_grad()
def encode_prompt(prompt: str):
    prompt = prompt.replace("video", "image")
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=pipe_device
    )
    return prompt_embeds.squeeze(0), pooled_prompt_embeds.squeeze(0)
