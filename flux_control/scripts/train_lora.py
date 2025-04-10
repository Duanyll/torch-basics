import argparse
import tomllib
import yaml
import json
from typing import Optional
from rich.console import Console

from ..trainers.flux_finetuner import FluxFinetuner
from ..utils.common import deep_merge_dicts


def main(config_path: str, config_overrides: Optional[str] = None):
    # Config file can be in either .toml, .yaml, or .json format
    if config_path.endswith(".toml"):
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    elif config_path.endswith(".yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .toml, .yaml, or .json.")

    # config_overrides is a json string
    if config_overrides:
        overrides = json.loads(config_overrides)
        config = deep_merge_dicts(config, overrides)

    finetuner = FluxFinetuner(**config)
    finetuner.train()


# accelerate launch -m flux_control.scripts.train_lora configs/finetune.toml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA for Flux")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument(
        "-o",
        "--config_overrides",
        type=str,
        default=None,
        help="JSON string of config overrides",
    )
    args = parser.parse_args()
    
    console = Console()
    try:
        main(args.config_path, args.config_overrides)
    except Exception:
        console.print_exception()
