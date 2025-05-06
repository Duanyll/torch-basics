from ast import parse
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
    input_path,
    out_name="output.png",
    checkpoint_dir=None,
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
    flux.load_model(device=torch.device(device), input_dir=checkpoint_dir)
    if os.path.isdir(input_path):
        # Recursively find all .pkl files in the directory
        files = []
        for root, _, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.endswith(".pkl"):
                    files.append(os.path.join(root, filename))
        if not files:
            raise ValueError(f"No .pkl files found in directory {input_path}.")
    else:
        files = [input_path]
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
        image = flux.sample(data)
        save_file = os.path.join(os.path.dirname(file), out_name)
        image.save(save_file)
        console.log(f"Saved image to {save_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference LoRA for Flux")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("input_path", type=str, help="Path to the input file or directory")
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="output.png",
        help="Name of the output file (default: output.png)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory (default: None, will use base model)",
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
            config_path=args.config_path,
            input_path=args.input_path,
            out_name=args.output_name,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            config_overrides=args.config_overrides,
        )
    except Exception:
        console.print_exception()
        raise
    