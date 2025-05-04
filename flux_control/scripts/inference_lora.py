import os
import tomllib
import yaml
import json
import torch
import pickle
from typing import Optional
from rich.console import Console

if __name__ == "__main__":
    from ..utils.logging import setup_rich_logging
    setup_rich_logging()

from ..core.inference import FluxInference
from ..utils.common import deep_merge_dicts

console = Console()

def main(
    config_path: str,
    checkpoint_dir: str,
    input_path,
    output_path=None,
    device="cuda",
    config_overrides: Optional[str] = None,
):
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

    flux = FluxInference(**config)
    flux.load_finetuned_model(checkpoint_dir, device=torch.device(device))
    if os.path.isdir(input_path):
        if output_path is None:
            output_path = input_path
        files = [
            os.path.join(f, input_path)
            for f in os.listdir(input_path)
            if f.endswith(".pkl")
        ]
    else:
        output_path = os.path.dirname(input_path)
        files = [input_path]
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
        image = flux.sample(data)
        save_file = os.path.join(output_path, os.path.basename(file) + ".png")
        image.save(save_file)
        console.log(f"Saved image to {save_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference LoRA for Flux")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the checkpoint directory")
    parser.add_argument("input_path", type=str, help="Path to the input file or directory")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to the output directory",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "-o",
        "--config_overrides",
        type=str,
        default=None,
        help="JSON string of config overrides",
    )
    args = parser.parse_args()
    
    try:
        main(
            args.config_path,
            args.checkpoint_dir,
            args.input_path,
            args.output_path,
            args.device,
            args.config_overrides,
        )
    except Exception:
        console.print_exception()
        raise
    