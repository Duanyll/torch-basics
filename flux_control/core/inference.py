import logging
from math import e
from typing import Annotated, Literal, Any, cast
from pydantic import BaseModel, PlainValidator
import torch
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ..adapters import BaseAdapter, parse_adapter_config
from ..utils.upcasting import (
    apply_layerwise_upcasting,
    cast_trainable_parameters,
    LayerwiseUpcastingGranularity,
)
from ..utils.common import unpack_bool_tensor
from .sampler import FluxSampler

logger = logging.getLogger(__name__)


class FluxInference(BaseModel):
    adapter: Annotated[BaseAdapter, PlainValidator(parse_adapter_config)]
    sampler: FluxSampler
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    base_precision: Literal["fp32", "bf16", "fp8-upcast"] = "fp8-upcast"
    vae_precision: Literal["fp32", "bf16"] = "bf16"
    """
    Precision for the base transformer model.
    
    1. "fp32": Use fp32 precision, cost ~48GB VRAM for weights. This is not very useful since 
    the model is already in bf16.
    2. "bf16": Use bf16 precision, cost ~23GB VRAM. Not feasible for 24GB GPUs.
    3. "fp8-upcast": Store weights in fp8 and compute in bf16, cost ~11GB VRAM. Also speeds up
    the training.
    
    Currently quantization (store & compute in fp8) is not supported.
    """
    trainable_precision: Literal["fp32", "bf16"] = "bf16"
    """
    Precision for the trainable parameters. can be "fp32" or "bf16".
    """
    allow_tf32: bool = False

    _weight_dtype: torch.dtype = torch.bfloat16
    _trainable_dtype: torch.dtype = torch.bfloat16
    _vae_dtype: torch.dtype = torch.float32
    _transformer: Any = None
    _vae: Any = None
    _device: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._weight_dtype = (
            torch.float32 if self.base_precision == "fp32" else torch.bfloat16
        )
        self._trainable_dtype = (
            torch.float32 if self.trainable_precision == "fp32" else torch.bfloat16
        )
        self._vae_dtype = (
            torch.float32 if self.vae_precision == "fp32" else torch.bfloat16
        )

    def _info(self, message: str):
        logger.info(message)

    def _make_transformer(self) -> FluxTransformer2DModel:
        transformer = cast(
            FluxTransformer2DModel,
            FluxTransformer2DModel.from_pretrained(
                self.pretrained_model_id,
                subfolder="transformer",
                torch_dtype=self._weight_dtype,
            ),
        )
        transformer.requires_grad_(False)
        self.adapter.install_modules(transformer)
        cast_trainable_parameters(transformer, self._trainable_dtype)
        self._info(
            f"Transformer model created with {self.pretrained_model_id} and {self.adapter.__class__.__name__}"
        )
        return transformer

    def _make_vae(self) -> AutoencoderKL:
        vae = cast(
            AutoencoderKL,
            AutoencoderKL.from_pretrained(
                self.pretrained_model_id,
                subfolder="vae",
                torch_dtype=self._vae_dtype,
            ),
        )
        vae.requires_grad_(False)
        self._info(f"VAE model created with {self.pretrained_model_id}")
        return vae

    def _optimize_model(self, transformer):
        if self.base_precision == "fp8-upcast":
            apply_layerwise_upcasting(
                transformer,
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=self._weight_dtype,
                granularity=LayerwiseUpcastingGranularity.PYTORCH_LAYER,
            )
            self._info(f"Applied layerwise upcasting.")

        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            self._info(f"TF32 enabled.")

    def _load_weights(self, transformer, input_dir):
        lora_state_dict = cast(dict, FluxControlPipeline.lora_state_dict(input_dir))
        self.adapter.load_model(transformer, lora_state_dict)
        cast_trainable_parameters(transformer, self._trainable_dtype)
        self._info(f"Loaded model from {input_dir}")

    def _move_batch_to_device(self, batch, device, insert_batch_dim=True):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, tuple):
                v = unpack_bool_tensor(*v)
            if isinstance(v, torch.Tensor):
                if insert_batch_dim:
                    v = v.unsqueeze(0)
                new_batch[k] = v.to(device=device, dtype=self._weight_dtype)
            else:
                new_batch[k] = v
        return new_batch

    def load_model(
        self,
        device: torch.device,
        input_dir: str | None = None,
    ):
        """
        Load the finetuned model from the given directory.
        """
        self._info(f"Loading finetuned model from {input_dir}")
        self._transformer = self._make_transformer()
        self._optimize_model(self._transformer)
        if input_dir is not None:
            self._load_weights(self._transformer, input_dir)
        self._transformer.to(device)
        self._transformer.eval()
        self._device = device
        self.sampler.set_meta(device=self._device, dtype=self._weight_dtype, vae_dtype=self._vae_dtype)
        self._vae = self._make_vae()
        self._vae.to(device)
        self._vae.eval()

    def sample(self, batch: dict):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            image = self.sampler.sample(
                self._transformer,
                self._vae,
                self.adapter,
                self._move_batch_to_device(batch, self._device),
                progress,
            )
        return image
