import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

# 初始化 rich 控制台
console = Console()

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger("video_pipeline")
logger.setLevel(logging.DEBUG)

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

from ..datasets.collage.pipeline import process_sample, load_all_models  # 假设路径正确


LMDB_MAP_SIZE = 256 * 1024 * 1024 * 1024  # 256 GB


def get_video_paths(input_dir, max_samples):
    """递归扫描输入目录中的 mp4 文件，最多返回 max_samples 个路径"""
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
    """加载 caption pickle 文件，返回 dict"""
    logger.debug(f"Loading caption file {caption_file}")
    with open(caption_file, "rb") as f:
        captions = pickle.load(f)
    logger.debug(f"Loaded {len(captions)} captions")
    return captions


def load_video(video_path):
    """加载视频并归一化"""
    logger.debug(f"Loading video {video_path}")
    video, _, _ = torchvision.io.read_video(
        video_path, output_format="TCHW", pts_unit="sec", end_pts=10
    )
    logger.debug(f"Loaded video {video_path}, shape={video.shape}")
    return video.float() / 255.0


def producer(loader_queue, input_queue, progress_queue, caption_dict, num_threads):
    """生产者：从 loader_queue 获取路径，加载视频和提示词，放入 input_queue"""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    logger.debug(f"Producer {pid} started with {num_threads} threads")
    try:
        while True:
            try:
                video_path = loader_queue.get(timeout=5)
                logger.debug(f"Producer {pid} got video path: {video_path}")
                if video_path is None:
                    loader_queue.put(None)
                    logger.debug(f"Producer {pid} received None, exiting")
                    break
                video = load_video(video_path)
                video_name = os.path.basename(video_path)
                prompt = caption_dict.get(video_name, "")
                logger.debug(
                    f"Producer {pid} putting video {video_path} to input_queue"
                )
                input_queue.put((video_path, video, prompt))
                progress_queue.put(("loaded", video_path))
            except queue.Empty:
                logger.debug(f"Producer {pid} timed out waiting for loader_queue")
                continue
            except Exception as e:
                logger.error(f"Producer {pid} error loading {video_path}: {e}")
                progress_queue.put(("load_failed", video_path, str(e)))
    except Exception as e:
        logger.error(f"Producer {pid} unexpected error: {e}")
    finally:
        logger.debug(f"Producer {pid} sending None to input_queue")
        input_queue.put(None)
        logger.debug(f"Producer {pid} exiting")


def consumer(input_queue, output_queue, progress_queue, gpu_id, num_threads):
    """消费者：在指定 GPU 上处理视频，放入 output_queue"""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    logger.debug(f"Consumer {pid} started on GPU {gpu_id} with {num_threads} threads")
    torch.cuda.set_device(gpu_id)
    logger.debug(f"Consumer {pid} loading models on GPU {gpu_id}")
    load_all_models(gpu_id)
    try:
        while True:
            logger.debug(f"Consumer {pid} waiting for input_queue")
            item = input_queue.get()
            if item is None:
                logger.debug(f"Consumer {pid} received None, exiting")
                input_queue.put(None)
                break
            video_path, video, prompt = item
            logger.debug(f"Consumer {pid} processing {video_path}")
            try:
                result = process_sample(
                    video_path, prompt, video_frames=video, device=gpu_id
                )
                del video
                if result is not None:
                    logger.debug(
                        f"Consumer {pid} putting result for {video_path} to output_queue"
                    )
                    output_queue.put((video_path, result))
                    progress_queue.put(("processed", video_path))
                else:
                    logger.debug(
                        f"Consumer {pid} skipped {video_path} (result is None)"
                    )
                    progress_queue.put(("skipped", video_path))
            except Exception as e:
                logger.error(f"Consumer {pid} error processing {video_path}: {e}")
                progress_queue.put(("process_failed", video_path, str(e)))
    except Exception as e:
        logger.error(f"Consumer {pid} unexpected error: {e}")
        progress_queue.put(("consumer_error", "", str(e)))
    finally:
        output_queue.put(None)
        logger.debug(f"Consumer {pid} exiting")


def lmdb_writer(output_queue, progress_queue, lmdb_path, total_items, num_threads):
    """写入线程：从 output_queue 获取数据，写入 lmdb"""
    torch.set_num_threads(num_threads)
    pid = os.getpid()
    logger.debug(f"Writer {pid} started with {num_threads} threads")
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)
    items_written = 0
    try:
        with env.begin(write=True) as txn:
            while items_written < total_items:
                try:
                    logger.debug(f"Writer {pid} waiting for output_queue")
                    res = output_queue.get(timeout=10)
                    if res is None:
                        logger.debug(f"Writer {pid} received None, exiting")
                        break
                    video_path, result = res
                    key = f"{items_written:08d}".encode()
                    value = pickle.dumps(result)
                    txn.put(key, value)
                    items_written += 1
                    logger.debug(
                        f"Writer {pid} wrote {video_path} ({items_written}/{total_items})"
                    )
                    progress_queue.put(("written", video_path))
                except queue.Empty:
                    logger.debug(f"Writer {pid} timed out waiting for output_queue")
    except Exception as e:
        logger.error(f"Writer {pid} error: {e}")
        progress_queue.put(("writer_error", "", str(e)))
    finally:
        env.close()
        logger.debug(f"Writer {pid} closed lmdb and exiting")


def main(args):
    # 设置多进程启动方式
    mp.set_start_method("spawn", force=True)
    logger.debug("Main process started")

    # 加载 caption 文件
    caption_dict = load_caption_file(args.caption_file)

    # 获取视频路径
    video_paths = get_video_paths(args.input_dir, args.max_samples)
    total_items = len(video_paths)
    logger.debug(f"Main process: Total videos to process: {total_items}")

    # 创建 Manager 和队列
    loader_queue = mp.Queue(maxsize=10)
    input_queue = mp.Queue(maxsize=10)
    output_queue = mp.Queue(maxsize=10)
    progress_queue = mp.Queue()
    logger.debug("Main process: Queues created")

    # 初始化 loader_queue（首批路径）
    initial_batch_size = min(10, total_items)
    for video_path in video_paths[:initial_batch_size]:
        loader_queue.put(video_path)
    remaining_paths = video_paths[initial_batch_size:]
    logger.debug(
        f"Main process: Initialized loader_queue with {initial_batch_size} paths"
    )

    # 启动生产者进程
    producers = []
    for i in range(args.num_loaders):
        p = mp.Process(
            target=producer,
            args=(
                loader_queue,
                input_queue,
                progress_queue,
                caption_dict,
                args.num_threads,
            ),
        )
        p.start()
        producers.append(p)
        logger.debug(f"Main process: Started producer {i}")

    # 启动消费者进程
    consumers = []
    for gpu_id in range(args.num_gpus):
        c = mp.Process(
            target=consumer,
            args=(input_queue, output_queue, progress_queue, gpu_id, args.num_threads),
        )
        c.start()
        consumers.append(c)
        logger.debug(f"Main process: Started consumer on GPU {gpu_id}")

    # 启动 lmdb 写入线程
    writer = mp.Process(
        target=lmdb_writer,
        args=(
            output_queue,
            progress_queue,
            args.output_dir,
            total_items,
            args.num_threads,
        ),
    )
    writer.start()
    logger.debug("Main process: Started writer")

    # 主进程：监控进度并补充 loader_queue
    failed_details = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn(
            "Success: {task.fields[success]} | Failed: {task.fields[failed]} | Skipped: {task.fields[skipped]}"
        ),
    ) as progress:
        task = progress.add_task(
            "[green]Processing videos...",
            total=total_items,
            success=0,
            failed=0,
            skipped=0,
        )
        processed = 0
        success = 0
        failed = 0
        skipped = 0

        while processed < total_items:
            try:
                logger.debug("Main process: Waiting for progress_queue")
                logger.debug(f"loader_queue size: {loader_queue.qsize()}")
                logger.debug(f"input_queue size: {input_queue.qsize()}")
                logger.debug(f"output_queue size: {output_queue.qsize()}")
                status, video_path, *extra = progress_queue.get(timeout=30)
                logger.debug(f"Main process: Received {status} for {video_path}")
                if status == "loaded":
                    pass
                elif status == "written":
                    success += 1
                    processed += 1
                    progress.update(
                        task, advance=1, success=success, failed=failed, skipped=skipped
                    )
                elif status == "skipped":
                    skipped += 1
                    processed += 1
                    progress.update(
                        task, advance=1, success=success, failed=failed, skipped=skipped
                    )
                elif status in ("load_failed", "process_failed"):
                    failed += 1
                    processed += 1
                    failed_details.append(
                        (status, video_path, extra[0] if extra else "Unknown error")
                    )
                    progress.update(
                        task, advance=1, success=success, failed=failed, skipped=skipped
                    )
                elif status in ("consumer_error", "writer_error"):
                    failed_details.append(
                        (status, "", extra[0] if extra else "Unknown error")
                    )
                    console.print(f"[red]Error: {extra[0]}[/red]")
                    break

                # 补充 loader_queue
                if remaining_paths:
                    next_path = remaining_paths.pop(0)
                    try:
                        loader_queue.put(next_path, timeout=5)
                        logger.debug(f"Main process: Added {next_path} to loader_queue")
                    except queue.Full:
                        remaining_paths.insert(0, next_path)
                        logger.debug(
                            "Main process: loader_queue is full, not adding more paths"
                        )
                            
            except queue.Empty:
                logger.debug("Main process: Timed out waiting for progress_queue")
                if all(not p.is_alive() for p in producers + consumers + [writer]):
                    logger.debug("Main process: All processes dead, exiting loop")
                    break

    # 清理
    for _ in range(args.num_loaders):
        loader_queue.put(None)
        logger.debug("Main process: Sent None to loader_queue")
    for p in producers:
        p.join()
        logger.debug(f"Main process: Producer {p.pid} joined")
    for c in consumers:
        c.join()
        logger.debug(f"Main process: Consumer {c.pid} joined")
    writer.join()
    logger.debug("Main process: Writer joined")

    # 美化总结输出
    console.print("\n[bold green]Pipeline Completed[/bold green]")
    console.print(f"[green]Succeeded:[/green] {success}")
    console.print(f"[yellow]Skipped:[/yellow] {skipped}")
    console.print(f"[red]Failed:[/red] {failed}")

    # 显示失败详情
    if failed_details:
        console.print("\n[bold red]Failed Details[/bold red]")
        table = Table(title="Failure Report")
        table.add_column("Type", style="cyan")
        table.add_column("Video Path", style="magenta")
        table.add_column("Error", style="red")

        for status, video_path, error in failed_details:
            table.add_row(status, video_path or "N/A", error)

        console.print(table)


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
    args = parser.parse_args()
    main(args)
