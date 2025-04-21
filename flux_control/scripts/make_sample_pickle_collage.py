import os
import torch
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

from ..datasets.collage.hf import load_hf_pipeline, encode_latents, encode_prompt
from ..datasets.collage.palette import extract_palette_from_masked_image_with_spatial


def make_sample_from_rgba(image_path, prompt_path, device="cuda", outfile="output.pkl"):
    image = Image.open(image_path).convert("RGBA")
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to CxHxW
    image = image.float() / 255.0
    image_rgb = image[:3].to(device=device, dtype=torch.bfloat16)
    image_alpha = image[3].to(device=device, dtype=torch.bfloat16)
    
    # Load the prompt
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()
        
    collage_control_latents = encode_latents(image_rgb * image_alpha).cpu()
    collage_alpha = image_alpha.cpu()
    edge_control_latents = torch.zeros_like(collage_control_latents)
    palettes, locations = extract_palette_from_masked_image_with_spatial(image_rgb, image_alpha)
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)
    
    output = {
        "video_name": os.path.basename(image_path),
        "src": image_rgb.cpu(),
        "clean_latents": image_rgb.cpu(),
        "collage_control_latents": collage_control_latents,
        "collage_alpha": collage_alpha,
        "edge_control_latents": edge_control_latents,
        "palettes": palettes.cpu(),
        "palette_locations": locations.cpu(),
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    }
    
    with open(outfile, "wb") as f:
        pickle.dump(output, f)
        
if __name__ == "__main__":
    import argparse
    from rich.console import Console

    parser = argparse.ArgumentParser(description="Create sample pickle file")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("prompt_path", type=str, help="Path to the input prompt file")
    parser.add_argument("outfile", type=str, help="Path to the output pickle file")
    args = parser.parse_args()
    
    load_hf_pipeline("cuda")
    console = Console()
    
    make_sample_from_rgba(args.image_path, args.prompt_path, outfile=args.outfile)
    console.log(f"Sample saved to {args.outfile}")