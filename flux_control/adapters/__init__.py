from .base import BaseAdapter
from .peft_lora import PeftLoraAdapter
from .d_concat import DConcatAdapter
from .n_concat import NConcatAdapter
from .collage import CollageAdapter, CollageAdapterV2
from .f1_fill import Flux1FillAdapter

ADAPTER_MAP = {
    "peft_lora": PeftLoraAdapter,
    "d_concat": DConcatAdapter,
    "n_concat": NConcatAdapter,
    "collage": CollageAdapter,
    "collage_v2": CollageAdapterV2,
    "f1_fill": Flux1FillAdapter,
}

def parse_adapter_config(adapter_config) -> BaseAdapter:
    """
    Parse the adapter config and return an adapter instance.

    Parameters
    ----------
    adapter_config : dict
        The adapter config.

    Returns
    -------
    BaseAdapter
        The adapter instance.
    """
    if not isinstance(adapter_config, dict):
        raise ValueError("adapter_config must be a dictionary.")
    if "type" not in adapter_config:
        raise ValueError("adapter_config must contain a 'type' key.")
    adapter_type = adapter_config.pop("type")
    if adapter_type not in ADAPTER_MAP:
        raise ValueError(f"Unknown adapter type: {adapter_type}.")
    adapter_class = ADAPTER_MAP[adapter_type]
    return adapter_class(**adapter_config)