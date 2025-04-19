#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import ray
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()

# Import the process_sample and load_all_models functions
# Assuming the original code is in a module named `processor`
# You might need to adjust this import based on your project structure
from .pipeline import process_sample, load_all_models


# Function to load video captions from pickle file
def load_video_captions(caption_path: str) -> Dict[str, str]:
    with open(caption_path, "rb") as f:
        captions = pickle.load(f)
    return captions


# Function to find all video files in a directory and its subdirectories
def find_video_files(input_dir: str) -> List[str]:
    video_files = []
    for ext in ["mp4", "MP4", "avi", "AVI", "mov", "MOV"]:
        video_files.extend(
            glob.glob(os.path.join(input_dir, f"**/*.{ext}"), recursive=True)
        )
    return sorted(video_files)


# Ray actor for GPU processing
@ray.remote(num_gpus=1)
class VideoProcessor:
    def __init__(self, device_id: int):
        self.device = f"cuda:{device_id}"
        # Load models for this GPU worker
        load_all_models(self.device)
        self.processed_count = 0
        logger.info(f"Initialized processor on {self.device}")

    def process_video(self, video_path: str, prompt: str) -> Optional[dict]:
        try:
            result = process_sample(video_path, prompt, device=self.device)
            self.processed_count += 1
            return {"video_path": video_path, "result": result}
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            return {"video_path": video_path, "result": None, "error": str(e)}

    def get_processed_count(self):
        return self.processed_count


# Ray task for loading video frames
@ray.remote(num_cpus=1)
def load_video_frames(video_path: str) -> Tuple[str, Optional[torch.Tensor]]:
    try:
        import torchvision

        video, _, _ = torchvision.io.read_video(
            video_path, output_format="TCHW", pts_unit="sec"
        )
        video = video.float() / 255.0
        return video_path, video
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {str(e)}")
        return video_path, None


# Writer class to handle LMDB storage
class LMDBWriter:
    def __init__(
        self, output_path: str, map_size: int = 1024 * 1024 * 1024 * 100
    ):  # 100GB default
        self.env = lmdb.open(output_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.count = 0

    def add(self, key: str, value: dict):
        try:
            self.txn.put(key.encode(), pickle.dumps(value))
            self.count += 1

            # Commit every 100 entries to avoid transaction getting too large
            if self.count % 100 == 0:
                self.txn.commit()
                self.txn = self.env.begin(write=True)
        except Exception as e:
            logger.error(f"Error adding {key} to LMDB: {str(e)}")

    def close(self):
        self.txn.commit()
        self.env.close()
        logger.info(f"LMDB closed with {self.count} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos using Ray across multiple GPUs"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save LMDB database"
    )
    parser.add_argument(
        "--captions", type=str, required=True, help="Path to video_caption_dict.pkl"
    )
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of videos to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for video loading"
    )
    parser.add_argument("--map_size", type=int, default=100, help="LMDB map size in GB")

    args = parser.parse_args()

    # Initialize Ray
    ray.init(num_gpus=args.num_gpus)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up LMDB
    lmdb_path = os.path.join(args.output_dir, "video_latents_db")
    writer = LMDBWriter(lmdb_path, map_size=args.map_size * 1024 * 1024 * 1024)

    # Load captions
    console.print(f"Loading captions from {args.captions}")
    captions = load_video_captions(args.captions)
    console.print(f"Loaded {len(captions)} captions")

    # Find all video files
    console.print(f"Finding video files in {args.input_dir}")
    all_videos = find_video_files(args.input_dir)
    console.print(f"Found {len(all_videos)} video files")

    # Limit the number of videos if specified
    if args.limit:
        all_videos = all_videos[: args.limit]
        console.print(f"Limited to {args.limit} videos")

    # Filter videos that have captions
    videos_to_process = []
    for video_path in all_videos:
        video_name = os.path.basename(video_path)
        if video_name in captions:
            videos_to_process.append(video_path)

    console.print(f"Processing {len(videos_to_process)} videos with available captions")

    # Initialize GPU processors
    processors = [
        VideoProcessor.remote(i % args.num_gpus) for i in range(args.num_gpus)
    ]

    # Process videos
    processed_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("Processed: {task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing videos...", total=len(videos_to_process))

        # Divide videos into batches
        for i in range(0, len(videos_to_process), args.batch_size):
            batch = videos_to_process[i : i + args.batch_size]

            # Start asynchronous loading of videos
            frame_futures = [
                load_video_frames.remote(video_path) for video_path in batch
            ]

            # Process loaded frames in parallel on GPUs
            processing_futures = []

            # Process each video as its frames become available
            while frame_futures:
                # Wait for the next frame to complete loading
                done_id, frame_futures = ray.wait(frame_futures, num_returns=1)
                video_path, frames = ray.get(done_id[0])

                if frames is not None:
                    video_name = os.path.basename(video_path)
                    prompt = captions.get(video_name, "")

                    # Assign to the least busy processor
                    counts = ray.get(
                        [p.get_processed_count.remote() for p in processors]
                    )
                    processor_idx = counts.index(min(counts))

                    # Process video on GPU
                    future = processors[processor_idx].process_video.remote(
                        video_path, prompt
                    )
                    processing_futures.append(future)

            # Get results from all processing futures
            results = ray.get(processing_futures)

            # Store results in LMDB
            for result_dict in results:
                video_path = result_dict["video_path"]
                result_data = result_dict["result"]
                video_name = os.path.basename(video_path)

                if result_data is not None:
                    writer.add(video_name, result_data)
                    processed_count += 1
                else:
                    error_count += 1

                progress.update(task, advance=1)

    # Close LMDB
    writer.close()

    # Print summary
    console.print(f"\nProcessing complete!")
    console.print(f"Successfully processed: {processed_count} videos")
    console.print(f"Failed: {error_count} videos")
    console.print(f"Results stored in: {lmdb_path}")


if __name__ == "__main__":
    main()
