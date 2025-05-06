from re import L
from typing import Literal
import logging
import torch
from pydantic import PositiveInt
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from .base import BaseAdapter

logger = logging.getLogger(__name__)
NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


class PeftLoraAdapter(BaseAdapter):
    """
    Adapter for LoRA fine-tuning using the PEFT library.
    """

    train_norm_layers: bool = False
    lora_layers: Literal["all-linear"] | list[str] = [
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
    ]
    rank: PositiveInt = 128
    gaussian_init_lora: bool = False
    use_lora_bias: bool = False

    def install_modules(self, transformer):
        if self.train_norm_layers:
            for name, param in transformer.named_parameters():
                if any(k in name for k in NORM_LAYER_PREFIXES):
                    param.requires_grad = True

        if self.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in self.lora_layers]
        elif self.lora_layers == "all-linear":
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)

        transformer_lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            init_lora_weights="gaussian" if self.gaussian_init_lora else True,
            target_modules=target_modules,
            lora_bias=self.use_lora_bias,
        )

        transformer.add_adapter(transformer_lora_config)

    def save_model(self, transformer: FluxTransformer2DModel) -> dict:
        layers_to_save = get_peft_model_state_dict(transformer)
        for name, param in transformer.named_parameters():
            if not "lora" in name and param.requires_grad:
                layers_to_save[name] = param.data

        return layers_to_save

    def load_model(self, transformer: FluxTransformer2DModel, state_dict: dict):
        lora_state_dict = {}
        other_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("transformer.", "")
            k = k.replace("module.", "")
            k = k.replace("default.", "")
            if "lora" in k:
                lora_state_dict[k] = v
            else:
                other_state_dict[k] = v
        incompatible_keys = set_peft_model_state_dict(
            transformer, lora_state_dict, adapter_name="default"
        )
        if incompatible_keys:
            logger.warning(
                f"Some keys in the state_dict are incompatible with the model: {incompatible_keys}"
            )
        incompatible_keys = transformer.load_state_dict(other_state_dict, strict=False)
        if incompatible_keys:
            logger.warning(
                f"Some keys in the state_dict are incompatible with the model: {incompatible_keys}"
            )
