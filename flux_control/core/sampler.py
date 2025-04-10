import math
from typing import cast, Optional
from pydantic import BaseModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
import torch
from PIL import Image
from rich.progress import Progress

from ..adapters import BaseAdapter


class FluxSampler(BaseModel):
    steps: int = 28
    guidance_scale: float = 3.5
    do_true_cfg: bool = False
    use_timestep_shift: bool = True
    vae_on_cpu: bool = True
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    height: int = 1024
    width: int = 1024
    seed: int = 0

    _pipe: FluxPipeline
    _dtype: torch.dtype
    _device: torch.device
    _vae_dtype: torch.dtype
    _vae_device: torch.device

    def load_model(self, dtype=torch.bfloat16, device=torch.device("cpu")):
        self._dtype = dtype
        self._device = device
        self._vae_dtype = torch.float32 if self.vae_on_cpu else dtype
        self._pipe = cast(
            FluxPipeline,
            FluxPipeline.from_pretrained(
                self.pretrained_model_id,
                torch_dtype=self._vae_dtype,
                transformer=None,
                text_encoder=None,
                text_encoder_2=None,
            ),
        )
        self._vae_device = torch.device("cpu") if self.vae_on_cpu else device
        self._pipe.vae.to(self._vae_device)

    @torch.no_grad()
    def sample(
        self,
        transformer: FluxTransformer2DModel,
        adapter: BaseAdapter,
        batch: dict,
        progress: Optional[Progress] = None,
    ) -> Image.Image:
        if not "noisy_latents" in batch:
            latent = self._make_empty_latent(self.height, self.width)
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

        for i in range(self.steps):
            ti = timesteps[i : i + 1]
            batch["noisy_latents"] = latent
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

            if progress is not None:
                progress.update(task, advance=1)

        image = self._latent_to_image(latent)

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

    def _latent_to_image(self, latent):
        latent = (
            latent / self._pipe.vae.config.scaling_factor
            + self._pipe.vae.config.shift_factor
        )
        latent = latent.to(device=self._vae_device, dtype=self._vae_dtype)
        image = self._pipe.vae.decode(latent, return_dict=False)[0]
        pil_image = cast(
            Image.Image,
            self._pipe.image_processor.postprocess(image, output_type="pil"),
        )[0]
        return pil_image

    def _make_timesteps(self, latent_len: int):
        t = torch.linspace(1.0, 0.0, self.steps + 1, device=self._device)
        if self.use_timestep_shift:
            scfg = self._pipe.scheduler.config
            m = (scfg.max_shift - scfg.base_shift) / (
                scfg.max_image_seq_len - scfg.base_image_seq_len
            )
            b = scfg.base_shift - m * scfg.base_image_seq_len
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
