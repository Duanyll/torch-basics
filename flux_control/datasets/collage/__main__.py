import os
import ray
import torch
import pickle
import random
import argparse
import lmdb
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# 替换成你的实际模块路径
from .pipeline import process_sample, load_all_models

logger = logging.getLogger("batch_processor")
console = Console()

logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)]
)

# Ray Actor，持久化模型
@ray.remote(num_gpus=1)
class VideoProcessor:
    def __init__(self, device="cuda"):
        self.device = device
        load_all_models(device)
        logger.info(f"[{device}] Models loaded.")

    def process(self, video_path: str, prompt: str):
        try:
            return process_sample(video_path, prompt, self.device)
        except Exception as e:
            logger.error(f"[{self.device}] Error processing {video_path}: {e}")
            return None


def find_video_files(input_dir: str, limit=None):
    paths = list(Path(input_dir).rglob("*.mp4"))
    if limit:
        paths = paths[:limit]
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Batch video processor using Ray and LMDB"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing videos and video_caption_dict.pkl",
    )
    parser.add_argument(
        "--output_lmdb", type=str, required=True, help="Path to LMDB output"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of videos to process"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel GPU workers"
    )
    parser.add_argument(
        "--map_size_gb",
        type=float,
        default=500,
        help="Max size of LMDB (default 500GB)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_lmdb)
    caption_dict_path = input_dir / "video_caption_dict.pkl"

    if not caption_dict_path.exists():
        logger.error(f"Prompt file not found: {caption_dict_path}")
        return

    logger.info("Loading video captions...")
    with open(caption_dict_path, "rb") as f:
        caption_dict = pickle.load(f)

    video_paths = find_video_files(str(input_dir), args.limit)
    logger.info(f"Found {len(video_paths)} videos to process.")

    ray.init(num_gpus=args.num_workers, include_dashboard=False)

    # 创建 VideoProcessor actor，每个绑定一个 GPU
    processors = [
        VideoProcessor.options(num_gpus=1).remote()
        for i in range(args.num_workers)
    ]

    # 分配任务
    futures = []
    for i, path in enumerate(video_paths):
        video_name = path.name
        prompt = caption_dict.get(video_name)
        if not prompt:
            logger.warning(f"No prompt for {video_name}, skipping.")
            continue
        processor = processors[i % args.num_workers]
        futures.append((video_name, processor.process.remote(str(path), prompt)))

    # 准备 LMDB 写入
    env = lmdb.open(str(output_path), map_size=int(args.map_size_gb * 1024**3))

    logger.info("Processing videos and writing to LMDB...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(futures))

        with env.begin(write=True) as txn:
            for video_name, future in futures:
                result = ray.get(future)
                if result is not None:
                    key = video_name.encode("utf-8")
                    value = pickle.dumps(result)
                    txn.put(key, value)
                progress.advance(task)

    logger.info(f"✅ Done! Processed videos saved to {output_path}")
    ray.shutdown()


if __name__ == "__main__":
    main()
