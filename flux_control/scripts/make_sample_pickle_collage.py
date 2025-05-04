import os
import torch
import pickle
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange

if __name__ == "__main__":
    from ..utils.logging import setup_rich_logging

    setup_rich_logging()

from ..utils.describe import describe
from ..datasets.collage.hf import load_hf_pipeline
from ..datasets.collage.pipeline import process_image_sample

logger = logging.getLogger(__name__)


def load_image_to_tensor(image_path, is_grayscale=False, has_alpha=False):
    """Load an image to a GPU tensor in CHW format, range [0,1]."""
    img = Image.open(image_path)
    if has_alpha and img.mode != "RGBA":
        logger.warning(f"Image {image_path} does not have alpha channel.")
    if is_grayscale and img.mode != "L":
        img = img.convert("L")
    elif not is_grayscale and not has_alpha and img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    if is_grayscale:
        img_tensor = img_tensor.unsqueeze(-1)  # Add channel dim for grayscale
    img_tensor = (
        img_tensor.permute(2, 0, 1) if not is_grayscale else img_tensor.permute(2, 0, 1)
    )
    return img_tensor.to(device="cuda")


def process_image_row(row, csv_dir):
    """Process a single row from the CSV using process_image_sample."""
    try:
        # Load coarse and mask_coarse from the same PNG
        coarse_path = csv_dir / row["coarse_path"]
        coarse_img = Image.open(coarse_path)
        if coarse_img.mode != "RGBA":
            raise ValueError(f"Image {coarse_path} must have an alpha channel.")
        coarse = load_image_to_tensor(coarse_path, has_alpha=True)[:3]  # RGB channels
        mask_coarse = load_image_to_tensor(coarse_path, has_alpha=True)[
            3:4
        ]  # Alpha channel

        # Load foreground (grayscale or alpha channel if RGBA)
        foreground_path = csv_dir / row["foreground_path"]
        foreground_img = Image.open(foreground_path)
        if foreground_img.mode == "RGBA":
            foreground = load_image_to_tensor(foreground_path, has_alpha=True)[
                3:4
            ]  # Use alpha channel
        else:
            foreground = load_image_to_tensor(foreground_path, is_grayscale=True)

        # Prompt from CSV
        prompt = row["prompt"]

        # Load optional src and hint (RGB)
        src = None
        if "src_path" in row and pd.notna(row["src_path"]):
            src_path = csv_dir / row["src_path"]
            src = load_image_to_tensor(src_path)

        hint = None
        if "hint_path" in row and pd.notna(row["hint_path"]):
            hint_path = csv_dir / row["hint_path"]
            hint = load_image_to_tensor(hint_path)

        # Confidence_fg from CSV
        confidence_fg = float(row["confidence_fg"])
        confidence_bg = float(row["confidence_bg"]) 

        # Call the provided function
        result = process_image_sample(
            coarse=coarse,
            mask_coarse=mask_coarse,
            foreground=foreground,
            prompt=prompt,
            src=src,
            hint=hint,
            confidence=None,  # Not provided in CSV
            confidence_fg=confidence_fg,
            confidence_bg=confidence_bg,
        )

        return result
    except Exception as e:
        logger.error(f"Error processing row {row}: {str(e)}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process images using process_image_sample."
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save output files"
    )
    args = parser.parse_args()

    load_hf_pipeline()  # Load the pipeline once at the start
    logger.info("Loaded HF pipeline.")

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    csv_path = Path(args.csv)
    csv_dir = csv_path.parent
    df = pd.read_csv(csv_path)
    required_columns = [
        "coarse_path",
        "foreground_path",
        "prompt",
        "confidence_fg",
        "confidence_bg",
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Process each row
    for idx, row in df.iterrows():
        logger.info(f"Processing row {idx + 1}/{len(df)}")
        result = process_image_row(row, csv_dir)
        describe(result, max_items=20)
        if result is not None:
            # Save output as a pickle file
            output_file = output_dir / f"sample_{idx}.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Saved result to {output_file}")


if __name__ == "__main__":
    main()
