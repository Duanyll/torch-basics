import os
import shutil
import lmdb
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from ..datasets.collage.hf import load_hf_pipeline, decode_latents
from ..utils.common import meshgrid_to_ij
from ..utils.describe import describe


def visualize_collage(sample: dict, device="cuda", output_file="output.png"):
    collage = sample["collage_control_latents"].to(device)
    alpha = sample["collage_alpha"].to(device)
    edge = sample["edge_control_latents"].to(device)
    palettes = sample["palettes"].to(device)
    locations = sample["palette_locations"].to(device)

    collage = decode_latents(collage)
    edge = decode_latents(edge)
    merged = collage * alpha + edge * (1 - alpha)
    merged = torch.clamp(merged, 0, 1)

    c, h, w = merged.shape
    merged = rearrange(merged, "c h w -> h w c")
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(merged.float().cpu().numpy())
    locations = meshgrid_to_ij(locations, h, w).float().cpu().numpy()
    palettes = palettes.float().cpu().numpy()
    for i in range(palettes.shape[0]):
        palette = palettes[i]
        loc = locations[i]
        plt.scatter(
            loc[1], loc[0], color=palette, s=100, marker="o", edgecolors="black"
        )
    plt.axis("off")
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close(fig)


def visualize_lmdb(lmdb_path, sample_count=5, device="cuda", output_dir="output"):
    # Randomly sample 5 samples from the LMDB
    load_hf_pipeline(device)
    os.makedirs(output_dir, exist_ok=True)
    env = lmdb.open(lmdb_path, readonly=True, max_dbs=16)
    db = env.open_db(b"result")
    with env.begin(write=False) as txn:
        cursor = txn.cursor(db)
        keys = [*cursor.iternext(keys=True, values=False)]
        sample_keys = np.random.choice(keys, size=sample_count, replace=False)

    for key in sample_keys:
        with env.begin(write=False) as txn:
            value = txn.get(key, db=db)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            sample = pickle.loads(value)
            with open(f"{output_dir}/{key.decode()}.pkl", "wb") as f:
                f.write(value)

        # Visualize the collage
        print(f"Visualizing sample with key: {key.decode()}")
        visualize_collage(sample, device=device, output_file=f"{output_dir}/{key.decode()}.png")
        describe(sample, max_items=20)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize collage samples from LMDB.")
    parser.add_argument("lmdb_path", type=str, help="Path to the LMDB file.")
    parser.add_argument(
        "--sample_count", type=int, default=5, help="Number of samples to visualize."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory if it exists.",
    )

    args = parser.parse_args()
    lmdb_path = args.lmdb_path
    device = args.device
    sample_count = args.sample_count
    
    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Removed existing output directory: {args.output_dir}")

    visualize_lmdb(
        lmdb_path, sample_count=sample_count, device=device, output_dir=args.output_dir
    )
    print(
        f"Visualized {sample_count} samples from {lmdb_path} and saved to {args.output_dir}."
    )
