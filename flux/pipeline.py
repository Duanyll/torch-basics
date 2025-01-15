import torch
from diffusers import (
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    AutoencoderKL,
    AutoencoderTiny,
)
from utils import (
    LayerwiseUpcastingGranularity,
    apply_layerwise_upcasting,
    apply_offload_all_hook,
)

def create_low_vram_flux_pipeline(
    device,
    repo="black-forest-labs/FLUX.1-dev",
    dtype=torch.bfloat16,
    storage_dtype=torch.float8_e4m3fn,
) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype)
    apply_layerwise_upcasting(
        pipe.transformer,
        storage_dtype=storage_dtype,
        compute_dtype=dtype,
        granularity=LayerwiseUpcastingGranularity.PYTORCH_LAYER,
    )
    apply_offload_all_hook(
        pipe,
        execution_device=device,
        offload_device="cpu",
        submodules=["text_encoder", "text_encoder_2", "transformer"],
    )
    pipe.vae.to(device)
    pipe.transformer.enable_gradient_checkpointing()
    
    return pipe


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
        
def freeze_pipeline(pipe):
    freeze_module(pipe.transformer)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.text_encoder_2)