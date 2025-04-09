import torch
import re
from accelerate.hooks import (
    ModelHook,
    clear_device_cache,
    send_to_device,
    add_hook_to_module,
    remove_hook_from_module,
)
from diffusers.models.attention import FeedForward, LuminaFeedForward
from diffusers.models.embeddings import (
    AttentionPooling,
    CogVideoXPatchEmbed,
    CogView3PlusPatchEmbed,
    GLIGENTextBoundingboxProjection,
    HunyuanDiTAttentionPool,
    LuminaPatchEmbed,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
)
from enum import Enum
from typing import Any, Dict, List, Tuple, Type
import warnings
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class LayerwiseUpcastingHook(ModelHook):
    r"""
    A hook that cast the input tensors and torch.nn.Module to a pre-specified dtype before the forward pass and cast
    the module back to the original dtype after the forward pass. This is useful when a model is loaded/stored in a
    lower precision dtype but performs computation in a higher precision dtype. This process may lead to quality loss
    in the output, but can significantly reduce the memory footprint.
    """

    def __init__(
        self,
        storage_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        ignore_trainable=True,
    ) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.ignore_trainable = ignore_trainable

    def _cast_storage(self, x: torch.Tensor) -> torch.Tensor:
        if self.ignore_trainable and x.requires_grad:
            return x
        if x.is_floating_point():
            return x.to(dtype=self.storage_dtype)
        return x

    def _cast_compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.ignore_trainable and x.requires_grad:
            return x
        if x.is_floating_point():
            return x.to(dtype=self.compute_dtype)
        return x

    def init_hook(self, module: torch.nn.Module):
        module._apply(self._cast_storage)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module._apply(self._cast_compute)
        # How do we account for LongTensor, BoolTensor, etc.?
        # args = tuple(align_maybe_tensor_dtype(arg, self.compute_dtype) for arg in args)
        # kwargs = {k: align_maybe_tensor_dtype(v, self.compute_dtype) for k, v in kwargs.items()}
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module._apply(self._cast_storage)
        return output


class LayerwiseUpcastingGranularity(str, Enum):
    r"""
    An enumeration class that defines the granularity of the layerwise upcasting process.
    Granularity can be one of the following:
        - `DIFFUSERS_MODEL`:
            Applies layerwise upcasting to the entire model at the highest diffusers modeling level. This will cast all
            the layers of model to the specified storage dtype. This results in the lowest memory usage for storing the
            model in memory, but may incur significant loss in quality because layers that perform normalization with
            learned parameters (e.g., RMSNorm with elementwise affinity) are cast to a lower dtype, but this is known
            to cause quality issues. This method will not reduce the memory required for the forward pass (which
            comprises of intermediate activations and gradients) of a given modeling component, but may be useful in
            cases like lowering the memory footprint of text encoders in a pipeline.
        - `DIFFUSERS_BLOCK`:
            TODO???
        - `DIFFUSERS_LAYER`:
            Applies layerwise upcasting to the lower-level diffusers layers of the model. This is more granular than
            the `DIFFUSERS_MODEL` level, but less granular than the `PYTORCH_LAYER` level. This method is applied to
            only those layers that are a group of linear layers, while excluding precision-critical layers like
            modulation and normalization layers.
        - `PYTORCH_LAYER`:
            Applies layerwise upcasting to lower-level PyTorch primitive layers of the model. This is the most granular
            level of layerwise upcasting. The memory footprint for inference and training is greatly reduced, while
            also ensuring important operations like normalization with learned parameters remain unaffected from the
            downcasting/upcasting process, by default. As not all parameters are casted to lower precision, the memory
            footprint for storing the model may be slightly higher than the alternatives. This method causes the
            highest number of casting operations, which may contribute to a slight increase in the overall computation
            time.
        Note: try and ensure that precision-critical layers like modulation and normalization layers are not casted to
        lower precision, as this may lead to significant quality loss.
    """

    DIFFUSERS_MODEL = "diffusers_model"
    DIFFUSERS_LAYER = "diffusers_layer"
    PYTORCH_LAYER = "pytorch_layer"


_SUPPORTED_DIFFUSERS_LAYERS = [
    AttentionPooling,
    HunyuanDiTAttentionPool,
    CogVideoXPatchEmbed,
    CogView3PlusPatchEmbed,
    LuminaPatchEmbed,
    TimestepEmbedding,
    GLIGENTextBoundingboxProjection,
    PixArtAlphaTextProjection,
    FeedForward,
    LuminaFeedForward,
]

_SUPPORTED_PYTORCH_LAYERS = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Linear,
]

_DEFAULT_PYTORCH_LAYER_SKIP_MODULES_PATTERN = ["pos_embed", "patch_embed", "norm"]


def apply_layerwise_upcasting_hook(
    module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype
) -> torch.nn.Module:
    r"""
    Applies a `LayerwiseUpcastingHook` to a given module.
    Args:
        module (`torch.nn.Module`):
            The module to attach the hook to.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before the forward pass.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass.
    Returns:
        `torch.nn.Module`:
            The same module, with the hook attached (the module is modified in place, so the result can be discarded).
    """
    hook = LayerwiseUpcastingHook(storage_dtype, compute_dtype)
    return add_hook_to_module(module, hook, append=True)


def apply_layerwise_upcasting(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    granularity: LayerwiseUpcastingGranularity = LayerwiseUpcastingGranularity.PYTORCH_LAYER,
    skip_modules_pattern: List[str] = [],
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    if granularity == LayerwiseUpcastingGranularity.DIFFUSERS_MODEL:
        return _apply_layerwise_upcasting_diffusers_model(
            module, storage_dtype, compute_dtype
        )
    if granularity == LayerwiseUpcastingGranularity.DIFFUSERS_LAYER:
        return _apply_layerwise_upcasting_diffusers_layer(
            module,
            storage_dtype,
            compute_dtype,
            skip_modules_pattern,
            skip_modules_classes,
        )
    if granularity == LayerwiseUpcastingGranularity.PYTORCH_LAYER:
        return _apply_layerwise_upcasting_pytorch_layer(
            module,
            storage_dtype,
            compute_dtype,
            skip_modules_pattern,
            skip_modules_classes,
        )


def _apply_layerwise_upcasting_diffusers_model(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> torch.nn.Module:
    from diffusers.models.modeling_utils import ModelMixin

    if not isinstance(module, ModelMixin):
        raise ValueError("The input module must be an instance of ModelMixin")

    # print(f'Applying layerwise upcasting to model "{module.__class__.__name__}"')
    apply_layerwise_upcasting_hook(module, storage_dtype, compute_dtype)
    return module


def _apply_layerwise_upcasting_diffusers_layer(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = _DEFAULT_PYTORCH_LAYER_SKIP_MODULES_PATTERN,
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    for name, submodule in module.named_modules():
        if (
            any(re.search(pattern, name) for pattern in skip_modules_pattern)
            or any(
                isinstance(submodule, module_class)
                for module_class in skip_modules_classes
            )
            or not isinstance(submodule, tuple(_SUPPORTED_DIFFUSERS_LAYERS))
        ):
            # print(f'Skipping layerwise upcasting for layer "{name}"')
            continue
        # print(f'Applying layerwise upcasting to layer "{name}"')
        apply_layerwise_upcasting_hook(submodule, storage_dtype, compute_dtype)
    return module


def _apply_layerwise_upcasting_pytorch_layer(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = _DEFAULT_PYTORCH_LAYER_SKIP_MODULES_PATTERN,
    skip_modules_classes: List[Type[torch.nn.Module]] = [],
) -> torch.nn.Module:
    count = 0
    for name, submodule in module.named_modules():
        if (
            any(re.search(pattern, name) for pattern in skip_modules_pattern)
            or any(
                isinstance(submodule, module_class)
                for module_class in skip_modules_classes
            )
            or not isinstance(submodule, tuple(_SUPPORTED_PYTORCH_LAYERS))
        ):
            # print(f'Skipping layerwise upcasting for layer "{name}"')
            continue
        # print(f'Applying layerwise upcasting to layer "{name}"')
        apply_layerwise_upcasting_hook(submodule, storage_dtype, compute_dtype)
        count += 1
    logger.info(f"Applied layerwise upcasting to {count} layers")
    return module


def cast_trainable_parameters(module: torch.nn.Module, dtype: torch.dtype) -> None:
    r"""
    Casts the trainable parameters of a given module to a specified dtype.
    Args:
        module (`torch.nn.Module`):
            The module to cast the parameters of.
        dtype (`torch.dtype`):
            The dtype to cast the parameters to.
    """
    for name, param in module.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=dtype)