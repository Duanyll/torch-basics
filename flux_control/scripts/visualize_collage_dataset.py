from pathlib import Path
import logging
import lmdb
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from ..datasets.collage.hf import load_hf_pipeline, decode_latents
from ..utils.describe import describe
from ..utils.common import unpack_bool_tensor

logger = logging.getLogger(__name__)


def visualize_collage(sample: dict, device="cuda", output_file="output.png"):
    """
    Visualize a sample dictionary containing images, masks, and confidence maps.

    Args:
        sample (dict): Dictionary with keys like 'src', 'mask_coarse', 'confidence', etc.
        device (str): Device to process tensors (default: 'cuda').
        output_file (str): Path to save the output image (default: 'output.png').
    """
    # Check if device is available
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        device = "cpu"
    device = torch.device(device)

    # Collect visualizable keys
    visualizable_keys = [
        "src",
        "tgt",
        "splat",
        "affine",
        "hint",
        "coarse",
        "mask_splat",
        "mask_affine",
        "mask_coarse",
        "foreground",
        "confidence",
    ]
    items_to_plot = [(k, v) for k, v in sample.items() if k in visualizable_keys]
    if not items_to_plot:
        logger.warning("No visualizable items found in sample.")
        return

    # Create dynamic subplot grid
    n_items = len(items_to_plot)
    n_cols = min(3, n_items)
    n_rows = (n_items + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle("Collage Visualization", fontsize=16)
    axs = np.array(axs).flatten() if n_items > 1 else [axs]

    for i, (key, value) in enumerate(items_to_plot):
        try:
            if key in ["src", "tgt", "splat", "affine", "hint", "coarse"]:
                # Handle images
                image = decode_latents(value.to(device)).cpu().float()
                image = torch.clamp(image, 0, 1)
                axs[i].imshow(image.permute(1, 2, 0).numpy())
                axs[i].set_title(key)
                axs[i].axis("off")
            elif key in ["mask_splat", "mask_affine", "mask_coarse", "foreground"]:
                # Handle masks
                mask = unpack_bool_tensor(*value)
                mask = rearrange(mask, "h w -> h w 1").cpu().numpy()
                axs[i].imshow(mask, cmap="gray")
                axs[i].set_title(key)
                axs[i].axis("off")
            elif key == "confidence":
                # Handle confidence map
                confidence = rearrange(value, "h w -> h w 1").float().cpu().numpy()
                im = axs[i].imshow(confidence, cmap="viridis", interpolation="nearest")
                fig.colorbar(im, ax=axs[i], orientation="vertical")
                axs[i].set_title(key)
                axs[i].axis("off")
        except Exception as e:
            logger.error(f"Error visualizing {key}: {str(e)}")
            continue

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualization to {output_file}")


def load_pkl(pkl_path: str) -> dict:
    """Load a sample dictionary from a .pkl file."""
    with open(pkl_path, "rb") as f:
        sample = pickle.load(f)
    if not isinstance(sample, dict):
        raise ValueError(f"PKL file {pkl_path} must contain a dictionary.")
    return sample


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize collage from PKL or LMDB.")
    parser.add_argument("--pkl", type=str, help="Path to input .pkl file")
    parser.add_argument("--lmdb", type=str, help="Path to input LMDB directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of samples to draw from LMDB",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to process tensors",
    )

    args = parser.parse_args()

    # Validate inputs
    if bool(args.pkl) == bool(args.lmdb):
        raise ValueError("Exactly one of --pkl or --lmdb must be provided.")

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_hf_pipeline(args.device)  # Load the pipeline once at the start
    logger.info("Loaded HF pipeline.")

    if args.pkl:
        # Process single PKL file
        pkl_path = Path(args.pkl)
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file {pkl_path} not found.")

        logger.info(f"Processing PKL file: {pkl_path}")
        sample = load_pkl(str(pkl_path))
        output_file = output_dir / f"{pkl_path.stem}.png"
        visualize_collage(sample, device=args.device, output_file=str(output_file))

    else:
        # Process LMDB
        lmdb_path = Path(args.lmdb)
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB directory {lmdb_path} not found.")

        logger.info(f"Sampling {args.sample_count} entries from LMDB: {lmdb_path}")
        env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=16)
        db = env.open_db(b"result")

        with env.begin(write=False) as txn:
            cursor = txn.cursor(db)
            keys = list(cursor.iternext(keys=True, values=False))
            if not keys:
                raise ValueError("LMDB is empty.")
            sample_keys = np.random.choice(
                keys, size=min(args.sample_count, len(keys)), replace=False
            )

        for key in sample_keys:
            with env.begin(write=False) as txn:
                value = txn.get(key, db=db)
                if value is None:
                    logger.error(f"Key {key} not found in LMDB.")
                    continue
                sample = pickle.loads(value)

                # Save as PKL
                pkl_file = output_dir / f"{key.decode()}.pkl"
                with open(pkl_file, "wb") as f:
                    f.write(value)
                logger.info(f"Saved PKL to {pkl_file}")

                # Visualize
                output_file = output_dir / f"{key.decode()}.png"
                visualize_collage(
                    sample, device=args.device, output_file=str(output_file)
                )


if __name__ == "__main__":
    main()
