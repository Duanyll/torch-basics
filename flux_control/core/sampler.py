import math
from typing import Literal, cast, Optional
from einops import rearrange
from pydantic import BaseModel
from diffusers import FluxTransformer2DModel, AutoencoderKL
import torch
from PIL import Image
from rich.progress import Progress

from ..adapters import BaseAdapter


class FluxSampler(BaseModel):
    steps: int = 28
    guidance_scale: float = 3.5
    do_true_cfg: bool = False
    use_timestep_shift: bool = True
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    height: int = 1024
    width: int = 1024
    infer_size_from: str | None = None
    infer_size_ratio: int = 1
    seed: int = 0
    
    base_image_seq_len: int = 256
    base_shift: float = 0.5
    max_image_seq_len: int = 4096
    max_shift: float = 1.15
    shift: float = 3.0
    
    _device: torch.device = torch.device("cuda")
    _dtype: torch.dtype = torch.bfloat16
    _vae_dtype: torch.dtype = torch.float32
    
    def set_meta(self, device: torch.device, dtype: torch.dtype, vae_dtype: torch.dtype):
        self._device = device
        self._dtype = dtype
        self._vae_dtype = vae_dtype

    @torch.no_grad()
    def sample(
        self,
        transformer: FluxTransformer2DModel,
        vae: AutoencoderKL,
        adapter: BaseAdapter,
        batch: dict,
        progress: Optional[Progress] = None,
    ) -> Image.Image:
        if not "noisy_latents" in batch:
            if self.infer_size_from is not None:
                height, width = batch[self.infer_size_from].shape[-2:]
                height = height * self.infer_size_ratio
                width = width * self.infer_size_ratio
            else:
                height = self.height
                width = self.width
            latent = self._make_empty_latent(height, width)
        else:
            latent = batch["noisy_latents"]

        b, c, h, w = latent.shape
        latent_length = (h * w) // 256
        timesteps = self._make_timesteps(latent_length)
        guidance = torch.full(
            (1,), self.guidance_scale, device=self._device, dtype=self._dtype
        )
        if self.do_true_cfg:
            batch_uncond = self._zero_conditions(batch)

        if progress is not None:
            task = progress.add_task(
                description="[bold green]Sampling",
                total=self.steps,
                progress_type="sample",
            )
        
        batch["noisy_latents"] = latent
        adapter.prepare_sample(transformer, vae, batch)

        for i in range(self.steps):
            ti = timesteps[i : i + 1]
            noise_pred = adapter.predict_velocity(
                transformer, batch, ti, guidance
            ).float()

            if self.do_true_cfg:
                batch_uncond["noisy_latents"] = latent
                noise_pred_uncond = adapter.predict_velocity(
                    transformer, batch_uncond, ti, guidance
                ).float()
                noise_pred = (
                    noise_pred_uncond
                    + (noise_pred - noise_pred_uncond) * self.guidance_scale
                )

            latent = latent + noise_pred * (timesteps[i + 1] - timesteps[i])
            latent = latent.to(self._dtype)
            batch["noisy_latents"] = latent

            if progress is not None:
                progress.update(task, advance=1)

        image = self._latent_to_image(vae, latent)

        if progress is not None:
            progress.remove_task(task)

        return image

    def _make_empty_latent(self, height, width, channels=16):
        generator = torch.Generator(device=self._device).manual_seed(self.seed)
        return torch.randn(
            1,
            channels,
            height // 8,
            width // 8,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )

    def _latent_to_image(self, vae, latent) -> Image.Image:
        latent = latent / vae.config.scaling_factor + vae.config.shift_factor
        latent = latent.to(self._vae_dtype)
        image = vae.decode(latent, return_dict=False)[0]
        image = rearrange(image, "1 c h w -> h w c")
        image = torch.clamp(image, -1, 1)
        image = ((image + 1) / 2) * 255
        image = image.to(torch.uint8)
        image = Image.fromarray(image.cpu().numpy())
        return image

    def _make_timesteps(self, latent_len: int):
        t = torch.linspace(1.0, 0.0, self.steps + 1, device=self._device)
        if self.use_timestep_shift:
            m = (self.max_shift - self.base_shift) / (
                self.max_image_seq_len - self.base_image_seq_len
            )
            b = self.base_shift - m * self.base_image_seq_len
            mu = m * latent_len + b
            t = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
        t = t.to(self._dtype)
        return t

    def _zero_conditions(self, batch: dict):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = torch.zeros_like(v)
            else:
                new_batch[k] = v
        return new_batch
