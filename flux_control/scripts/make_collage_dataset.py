from json import load
import logging
import torch
import torch.multiprocessing as mp
import torchvision.io
import pickle
import lmdb
import os
from pathlib import Path
import argparse
import queue
import time
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from ..datasets.collage.pipeline import process_sample, load_all_models

LMDB_MAP_SIZE = 256 * 1024 * 1024 * 1024  # 256 GB

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger("make_collage_dataset")
logger.setLevel(logging.DEBUG)

# Queue utility functions
def safe_put(queue, item, timeout=1):
    """Put item to queue with timeout."""
    while True:
        try:
            queue.put(item, timeout=timeout)
            return
        except queue.Full:
            time.sleep(0.1)

def safe_get(queue, timeout=1):
    """Get item from queue with timeout."""
    while True:
        try:
            return queue.get(timeout=timeout)
        except queue.Empty:
            time.sleep(0.1)

def get_video_paths(input_dir, max_samples):
    """Recursively scan input directory for mp4 files."""
    logger.debug(f"Scanning videos in {input_dir} with max_samples={max_samples}")
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(root, file))
            if len(video_paths) >= max_samples:
                logger.debug(f"Found {len(video_paths)} videos, stopping scan")
                return video_paths
    logger.debug(f"Found {len(video_paths)} videos")
    return video_paths

def load_caption_file(caption_file):
    """Load caption pickle file."""
    logger.debug(f"Loading caption file {caption_file}")
    with open(caption_file, "rb") as f:
        captions = pickle.load(f)
    logger.debug(f"Loaded {len(captions)} captions")
    return captions

def load_video(video_path):
    """Load and normalize video."""
    logger.debug(f"Loading video {video_path}")
    video, _, _ = torchvision.io.read_video(
        video_path, output_format="TCHW", pts_unit="sec", end_pts=10
    )
    logger.debug(f"Loaded video {video_path}, shape={video.shape}")
    return video.float() / 255.0

def loader(loader_queue, input_queue, progress_queue, caption_dict, num_threads):
    """Loader: Load videos and captions, put to input queue."""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    logger.debug(f"Loader {pid} started with {num_threads} threads")
    try:
        while True:
            video_path = safe_get(loader_queue)
            if video_path is None:
                logger.debug(f"Loader {pid} received None, exiting")
                safe_put(input_queue, None)
                break
            video = load_video(video_path)
            video_name = os.path.basename(video_path)
            prompt = caption_dict.get(video_name, "")
            logger.debug(f"Loader {pid} putting video {video_path} to input_queue")
            safe_put(input_queue, (video_path, video, prompt))
            safe_put(progress_queue, ("loaded", video_path))
    except Exception as e:
        logger.error(f"Loader {pid} error: {e}")
        raise

def processor(input_queue, output_queue, progress_queue, gpu_id, num_threads):
    """Processor: Process videos on GPU, put to output queue."""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    logger.debug(f"Processor {pid} started on GPU {gpu_id} with {num_threads} threads")
    torch.cuda.set_device(gpu_id)
    load_all_models(gpu_id)
    try:
        while True:
            item = safe_get(input_queue)
            if item is None:
                logger.debug(f"Processor {pid} received None, exiting")
                safe_put(output_queue, None)
                break
            video_path, video, prompt = item
            logger.debug(f"Processor {pid} processing {video_path}")
            result = process_sample(video_path, prompt, video_frames=video, device=gpu_id)
            del video
            if result is not None:
                safe_put(output_queue, (video_path, result))
                safe_put(progress_queue, ("processed", video_path))
            else:
                safe_put(progress_queue, ("skipped", video_path))
    except Exception as e:
        logger.error(f"Processor {pid} error: {e}")
        raise

def writer(output_queue, progress_queue, lmdb_path, num_threads):
    """Writer: Write results to lmdb."""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    logger.debug(f"Writer {pid} started with {num_threads} threads")
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)
    items_written = 0
    try:
        while True:
            res = safe_get(output_queue)
            if res is None:
                logger.debug(f"Writer {pid} received None, exiting")
                break
            video_path, result = res
            key = f"{items_written:08d}".encode()
            value = pickle.dumps(result)
            with env.begin(write=True) as txn:
                txn.put(key, value)
            items_written += 1
            logger.debug(f"Writer {pid} wrote {video_path} ({items_written})")
            safe_put(progress_queue, ("written", video_path))
    except Exception as e:
        logger.error(f"Writer {pid} error: {e}")
        raise
    finally:
        env.close()

def main(args):
    mp.set_start_method("spawn", force=True)
    logger.info("Main process started")

    # Load captions and videos
    caption_dict = load_caption_file(args.caption_file)
    video_paths = get_video_paths(args.input_dir, args.max_samples)
    total_items = len(video_paths)
    logger.info(f"Total videos to process: {total_items}")

    # Initialize native queues
    loader_queue = mp.Queue(maxsize=10)
    input_queue = mp.Queue(maxsize=10)
    output_queue = mp.Queue(maxsize=10)
    progress_queue = mp.Queue()
    logger.info("Queues created")

    # Initialize loader_queue
    initial_batch_size = min(10, total_items)
    for video_path in video_paths[:initial_batch_size]:
        safe_put(loader_queue, video_path)
    remaining_paths = video_paths[initial_batch_size:]
    logger.info(f"Initialized loader_queue with {initial_batch_size} paths")

    # Start processes
    processes = []
    for i in range(args.num_loaders):
        p = mp.Process(
            target=loader,
            args=(loader_queue, input_queue, progress_queue, caption_dict, args.num_threads),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started loader {i}")

    for gpu_id in range(args.num_gpus):
        c = mp.Process(
            target=processor,
            args=(input_queue, output_queue, progress_queue, gpu_id, args.num_threads),
        )
        c.start()
        processes.append(c)
        logger.info(f"Started processor on GPU {gpu_id}")

    w = mp.Process(
        target=writer,
        args=(output_queue, progress_queue, args.output_dir, total_items, args.num_threads),
    )
    w.start()
    processes.append(w)
    logger.info("Started writer")

    # Monitor progress
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("Success: {task.fields[success]} | Skipped: {task.fields[skipped]}"),
    ) as progress:
        task = progress.add_task(
            "[green]Processing videos...",
            total=total_items,
            success=0,
            skipped=0,
        )
        processed = 0
        success = 0
        skipped = 0

        try:
            while processed < total_items:
                status, video_path = safe_get(progress_queue)
                logger.debug(f"Received {status} for {video_path}")
                if status == "loaded":
                    pass
                elif status == "written":
                    success += 1
                    processed += 1
                    progress.update(task, advance=1, success=success, skipped=skipped)
                elif status == "skipped":
                    skipped += 1
                    processed += 1
                    progress.update(task, advance=1, success=success, skipped=skipped)

                # Refill loader_queue
                if remaining_paths:
                    next_path = remaining_paths[0]
                    try:
                        loader_queue.put(next_path, timeout=1)
                        remaining_paths.pop(0)
                        logger.debug(f"Loader queue refilled with {next_path}")
                    except queue.Full:
                        logger.debug("Loader queue is full, skipping refill")

        except Exception as e:
            logger.error(f"Main process error: {e}")
            # Signal termination
            for _ in range(args.num_loaders):
                safe_put(loader_queue, None)
            raise

    # Cleanup
    for _ in range(args.num_loaders):
        safe_put(loader_queue, None)
    for p in processes:
        p.join()
        logger.debug(f"Process {p.pid} joined")

    # Summary
    console.print("\n[bold green]Pipeline Completed[/bold green]")
    console.print(f"[green]Succeeded:[/green] {success}")
    console.print(f"[yellow]Skipped:[/yellow] {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory containing video files")
    parser.add_argument("--caption_file", required=True, help="Pickle file with captions")
    parser.add_argument("--output_dir", required=True, help="Output lmdb directory")
    parser.add_argument("--max_samples", type=int, default=float("inf"), help="Max number of videos to process")
    parser.add_argument("--num_loaders", type=int, default=2, help="Number of loader processes")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of CPU threads per process")
    args = parser.parse_args()
    main(args)