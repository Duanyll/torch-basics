from typing import cast
import logging
from logging.handlers import QueueHandler, QueueListener
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    MofNCompleteColumn
)

LOG_LEVEL = logging.INFO

if __name__ == "__main__":
    # Initialize rich console
    console = Console()

    handler = RichHandler(rich_tracebacks=True, console=console)
    # Configure logging
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(message)s",
        handlers=[handler],
    )
    logger = logging.getLogger("make_collage_dataset")
    logger.setLevel(LOG_LEVEL)
else:
    logger = cast(logging.Logger, None)

import torch
import torch.multiprocessing as mp
import torchvision.io
import numpy as np
import pickle
import lmdb
import os
import argparse
import queue
import time
import tomllib


def make_logger_for_worker(name, log_queue, config_hf=False):
    global logger
    """Create a logger for a worker process."""
    handler = QueueHandler(log_queue)
    logging.basicConfig(
        level=LOG_LEVEL,
        format=f"{name}: %(message)s",
        handlers=[handler],
    )
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    if config_hf:
        import transformers
        import diffusers

        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.disable_progress_bar()
        transformers.utils.logging.add_handler(handler)
        diffusers.utils.logging.disable_default_handler()
        diffusers.utils.logging.add_handler(handler)
        diffusers.utils.logging.disable_progress_bar()
    return logger


# Queue utility functions
def safe_put(q, item, timeout=1):
    """Put item to queue with timeout."""
    global logger
    total_timeout = 0
    while True:
        try:
            q.put(item, timeout=timeout)
            return
        except queue.Full:
            total_timeout += timeout
            time.sleep(0.1)
            if total_timeout > 600:
                logger.error(f"Queue put waiting too long! {total_timeout} seconds")
                return
            continue


def safe_get(q, timeout=1):
    """Get item from queue with timeout."""
    global logger
    total_timeout = 0
    while True:
        try:
            item = q.get(timeout=timeout)
            return item
        except queue.Empty:
            total_timeout += timeout
            time.sleep(0.1)
            if total_timeout > 600:
                logger.error(f"Queue get waiting too long! {total_timeout} seconds")
                return None
            continue


def get_processed_video_paths(lmdb_path):
    # Enumerate both the result and skipped databases
    video_paths = set()
    with lmdb.open(lmdb_path, max_dbs=16) as env:
        for db_name in [b"result", b"skipped"]:
            db = env.open_db(db_name)
            with env.begin(write=True) as txn:
                cursor = txn.cursor(db)
                for key in cursor.iternext(values=False):
                    video_paths.add(key.decode())
    return video_paths


def get_video_paths(input_dir, max_samples, resume):
    """Recursively scan input directory for mp4 files."""
    logger.info(f"Scanning videos in {input_dir} with max_samples={max_samples}")
    video_paths = set()
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_paths.add(os.path.join(root, file))
    logger.info(f"Found {len(video_paths)} videos")
    if resume:
        # If resuming, check if the LMDB already has processed videos
        processed_video_paths = get_processed_video_paths(args.output_dir)
        video_paths = video_paths - processed_video_paths
        logger.info(f"Resuming: {len(processed_video_paths)} videos already processed")
    result_list = list(video_paths)
    if max_samples < len(result_list):
        # Randomly sample the video paths
        np.random.seed(42)
        result_list = np.random.choice(
            result_list, size=max_samples, replace=False
        ).tolist()
    logger.info(f"Selected {len(result_list)} videos to process")
    return result_list


def load_caption_file(caption_file):
    """Load caption pickle file."""
    logger.info(f"Loading caption file {caption_file}")
    with open(caption_file, "rb") as f:
        captions = pickle.load(f)
    logger.info(f"Loaded {len(captions)} captions")
    return captions


def load_video(video_path):
    """Load and normalize video."""
    logger.debug(f"Loading video {video_path}")
    video, _, _ = torchvision.io.read_video(
        video_path, output_format="TCHW", pts_unit="sec", end_pts=10
    )
    logger.debug(f"Loaded video {video_path}, shape={video.shape}")
    return video.float() / 255.0


def load_config_file(config_path):
    from ..datasets.collage.config import CollageConfig

    if config_path is not None:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return CollageConfig(**config)
    else:
        return CollageConfig()


def loader(
    loader_queue, input_queue, progress_queue, caption_dict, log_queue, num_threads
):
    """Loader: Load videos and captions, put to input queue."""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    make_logger_for_worker(f"Loader-{pid}", log_queue)
    logger.info(f"Loader {pid} started with {num_threads} threads")
    try:
        while True:
            video_path = safe_get(loader_queue)
            if video_path is None:
                logger.info(f"Loader {pid} received None, exiting")
                break
            video = load_video(video_path)
            video.share_memory_()
            video_name = os.path.basename(video_path)
            prompt = caption_dict.get(video_name, "")
            logger.debug(f"Loader {pid} putting video {video_path} to input_queue")
            safe_put(input_queue, (video_path, video, prompt))
            safe_put(progress_queue, ("loaded", video_path))
            del video
    except Exception as e:
        logger.error(f"Loader {pid} error: {e}")
        raise


def processor(
    input_queue,
    output_queue,
    progress_queue,
    gpu_id,
    config_path,
    log_queue,
    num_threads,
):
    """Processor: Process videos on GPU, put to output queue."""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    make_logger_for_worker(f"Processor-{pid}", log_queue, config_hf=True)
    logger.info(f"Processor {pid} started on GPU {gpu_id} with {num_threads} threads")
    torch.cuda.set_device(gpu_id)

    from ..datasets.collage.pipeline import process_sample, load_all_models

    cfg = load_config_file(config_path)
    load_all_models(gpu_id)
    try:
        while True:
            item = safe_get(input_queue)
            if item is None:
                logger.info(f"Processor {pid} received None, exiting")
                break
            video_path, video, prompt = item
            logger.debug(f"Processor {pid} processing {video_path}")
            result = process_sample(
                video_path, prompt, video_frames=video, device=gpu_id, cfg=cfg
            )
            del video
            safe_put(output_queue, (video_path, result))
            if result is not None:
                safe_put(progress_queue, ("processed", video_path))
            else:
                safe_put(progress_queue, ("skipped", video_path))
    except Exception as e:
        logger.error(f"Processor {pid} error: {e}")
        raise


def writer(
    output_queue, progress_queue, lmdb_path, lmdb_map_size, log_queue, num_threads
):
    """Writer: Write results to lmdb."""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    make_logger_for_worker(f"Writer-{pid}", log_queue)
    logger.info(f"Writer {pid} started with {num_threads} threads")
    env = lmdb.open(lmdb_path, map_size=lmdb_map_size * 1024**3, max_dbs=16)
    db_result = env.open_db(b"result")
    db_skipped = env.open_db(b"skipped")
    items_written = 0
    try:
        while True:
            res = safe_get(output_queue)
            if res is None:
                logger.info(f"Writer {pid} received None, exiting")
                break
            video_path, result = res
            key = f"{os.path.basename(video_path)}".encode()
            value = pickle.dumps(result)
            db = db_skipped if result is None else db_result
            with env.begin(write=True) as txn:
                txn.put(key, value, db=db)
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
    video_paths = get_video_paths(args.input_dir, args.max_samples, args.resume)
    total_items = len(video_paths)
    logger.info(f"Total videos to process: {total_items}")

    # Initialize native queues
    ctx = mp.get_context("spawn")
    loader_queue = ctx.Queue(maxsize=args.queue_size)
    input_queue = ctx.Queue(maxsize=args.queue_size)
    output_queue = ctx.Queue(maxsize=args.queue_size)
    progress_queue = ctx.Queue()
    log_queue = ctx.Queue()
    logger.info("Queues created")

    # Initialize logging
    listener = QueueListener(log_queue, handler, respect_handler_level=True)
    listener.start()

    # Initialize loader_queue
    initial_batch_size = min(args.queue_size, total_items)
    for video_path in video_paths[:initial_batch_size]:
        safe_put(loader_queue, video_path)
    remaining_paths = video_paths[initial_batch_size:]
    logger.info(f"Initialized loader_queue with {initial_batch_size} paths")

    # Start processes
    loader_processes = []
    for i in range(args.num_loaders):
        p = ctx.Process(
            target=loader,
            args=(
                loader_queue,
                input_queue,
                progress_queue,
                caption_dict,
                log_queue,
                args.num_threads,
            ),
        )
        p.start()
        loader_processes.append(p)
        logger.info(f"Started loader {i}")

    processor_processes = []
    for gpu_id in range(args.num_gpus):
        c = ctx.Process(
            target=processor,
            args=(
                input_queue,
                output_queue,
                progress_queue,
                gpu_id,
                args.config_path,
                log_queue,
                args.num_threads,
            ),
        )
        c.start()
        processor_processes.append(c)
        logger.info(f"Started processor on GPU {gpu_id}")

    writer_process = ctx.Process(
        target=writer,
        args=(
            output_queue,
            progress_queue,
            args.output_dir,
            args.lmdb_map_size,
            log_queue,
            args.num_threads,
        ),
    )
    writer_process.start()
    logger.info("Started writer")

    # Monitor progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("Success: {task.fields[success]} | Skipped: {task.fields[skipped]}"),
        console=console,
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
                elif status == "processed":
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
                        loader_queue.put(next_path, timeout=0.1)
                        remaining_paths.pop(0)
                        logger.debug(f"Loader queue refilled with {next_path}")
                    except queue.Full:
                        logger.debug("Loader queue is full, skipping refill")

        except Exception as e:
            logger.error(f"Main process error: {e}")
            raise

    # Proper cleanup sequence
    logger.info("Processing complete, terminating workers in sequence")
    
    # Terminate loaders first
    logger.info("Terminating loader processes")
    for _ in range(args.num_loaders):
        safe_put(loader_queue, None)
    for p in loader_processes:
        p.join()
        logger.debug(f"Loader process {p.pid} joined")
    
    # Terminate processors next
    logger.info("Terminating processor processes")
    for _ in range(args.num_gpus):
        safe_put(input_queue, None)
    for p in processor_processes:
        p.join()
        logger.debug(f"Processor process {p.pid} joined")
    
    # Terminate writer last
    logger.info("Terminating writer process")
    safe_put(output_queue, None)
    writer_process.join()
    logger.debug(f"Writer process {writer_process.pid} joined")

    listener.stop()
    logger.info("All processes joined")

    # Summary
    console.print("\n[bold green]Pipeline Completed[/bold green]")
    console.print(f"[green]Succeeded:[/green] {success}")
    console.print(f"[yellow]Skipped:[/yellow] {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--caption_file", required=True, help="Pickle file with captions"
    )
    parser.add_argument("--output_dir", required=True, help="Output lmdb directory")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=float("inf"),
        help="Max number of videos to process",
    )
    parser.add_argument(
        "--num_loaders", type=int, default=2, help="Number of loader processes"
    )
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument(
        "--num_threads", type=int, default=8, help="Number of CPU threads per process"
    )
    parser.add_argument(
        "--queue_size", type=int, default=4, help="Queue size for loader and processor"
    )
    parser.add_argument(
        "--lmdb_map_size",
        type=int,
        default=256,
        help="LMDB map size in GB",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the config file for CollageConfig",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing, skipping already processed videos",
    )
    args = parser.parse_args()
    main(args)
