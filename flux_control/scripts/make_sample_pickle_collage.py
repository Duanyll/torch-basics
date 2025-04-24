import os
import torch
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange

from ..utils.describe import describe
from ..datasets.collage.hf import load_hf_pipeline, encode_latents, encode_prompt
from ..datasets.collage.palette import extract_palette_from_masked_image_with_spatial

RESOLUTIONS_720P = [
    # width, height
    (768, 768),
    (832, 704),
    (896, 640),
    (960, 576),
    (1024, 512),
]

RESOLUTIONS_1080P = [
    (1024, 1024),
    (1088, 960),
    (1152, 896),
    (1216, 832),
    (1280, 768),
    (1344, 704),
]


def get_all_resolutions():
    """Include both landscape and portrait orientations."""
    all_res = []
    for res in RESOLUTIONS_720P + RESOLUTIONS_1080P:
        all_res.append(res)  # Landscape
        all_res.append((res[1], res[0]))  # Portrait
    return all_res


def find_closest_resolution(image_width, image_height):
    """Find the closest supported resolution to the image's dimensions."""
    all_resolutions = get_all_resolutions()
    aspect_ratio = image_width / image_height
    min_diff = float("inf")
    best_res = None

    for res in all_resolutions:
        res_width, res_height = res
        res_aspect = res_width / res_height
        # Calculate difference in aspect ratio and total pixels
        aspect_diff = abs(aspect_ratio - res_aspect)
        size_diff = abs(image_width * image_height - res_width * res_height)
        total_diff = aspect_diff + size_diff / (
            image_width * image_height
        )  # Normalize size difference

        if total_diff < min_diff:
            min_diff = total_diff
            best_res = (res_width, res_height)

    return best_res


def crop_and_resize_image(image, target_size):
    """Crop and resize image to target size while maintaining aspect ratio."""
    img_width, img_height = image.size
    target_width, target_height = target_size

    # Calculate aspect ratios
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height

    if img_aspect > target_aspect:
        # Image is wider than target, crop width
        new_width = int(img_height * target_aspect)
        left = (img_width - new_width) // 2
        image = image.crop((left, 0, left + new_width, img_height))
    else:
        # Image is taller than target, crop height
        new_height = int(img_width / target_aspect)
        top = (img_height - new_height) // 2
        image = image.crop((0, top, img_width, top + new_height))

    # Resize to target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def make_sample_from_rgba(image_path, prompt, device="cuda", outfile="output.pkl"):
    # Open and process image
    image = Image.open(image_path).convert("RGBA")
    img_width, img_height = image.size

    # Find closest resolution
    resize = find_closest_resolution(img_width, img_height)

    # Crop and resize image
    image = crop_and_resize_image(image, resize)

    # Convert to tensor
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to CxHxW
    image = image.float() / 255.0
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=resize, mode="bilinear", align_corners=False
    ).squeeze(0)
    image_rgb = image[:3].to(device=device, dtype=torch.bfloat16)
    image_alpha = image[3].to(device=device, dtype=torch.bfloat16)

    # Process prompt
    image_latents = encode_latents(image_rgb).cpu()
    collage_control_latents = encode_latents(image_rgb * image_alpha).cpu()
    collage_alpha = image_alpha.cpu()
    edge_control_latents = torch.zeros_like(collage_control_latents)
    palettes, locations = extract_palette_from_masked_image_with_spatial(
        image_rgb, image_alpha
    )
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)

    output = {
        "video_name": os.path.basename(image_path),
        "src": image_rgb.cpu(),
        "clean_latents": image_latents.cpu(),
        "collage_control_latents": collage_control_latents,
        "collage_alpha": collage_alpha,
        "edge_control_latents": edge_control_latents,
        "palettes": palettes.cpu(),
        "palette_locations": locations.cpu(),
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    }

    describe(output)

    with open(outfile, "wb") as f:
        pickle.dump(output, f)

    return resize


if __name__ == "__main__":
    import argparse
    from rich.console import Console

    parser = argparse.ArgumentParser(
        description="Batch process images to pickle files from CSV"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV file with image_file and caption columns",
    )
    args = parser.parse_args()

    load_hf_pipeline("cuda")
    console = Console()

    # Read CSV
    df = pd.read_csv(args.csv_path)
    required_columns = ["image_file", "caption"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV must contain 'image_file' and 'caption' columns")

    # Process each row
    for _, row in df.iterrows():
        image_path = row["image_file"]
        image_path = os.path.join(os.path.dirname(args.csv_path), image_path)
        prompt = row["caption"]

        # Generate output pickle filename (same as image filename but with .pkl extension)
        outfile = os.path.splitext(image_path)[0] + ".pkl"

        try:
            resize = make_sample_from_rgba(image_path, prompt, outfile=outfile)
            console.log(
                f"Processed {image_path} with resolution {resize}, saved to {outfile}"
            )
        except Exception as e:
            console.log(f"Error processing {image_path}: {str(e)}")
