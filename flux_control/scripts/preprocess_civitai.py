import os
import argparse
import lmdb
import pickle
import numpy as np
import diffusers
import transformers
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
import torch
import torch.multiprocessing as mp
from PIL import Image
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import print

DTYPE = torch.bfloat16


def process_sample(image_path, prompt_path, pipe, device):
    """处理单个样本并返回处理后的数据"""
    try:
        with torch.no_grad():
            img = Image.open(image_path)
            img = pipe.image_processor.preprocess(img).to(device=device, dtype=DTYPE)
            latent = pipe.vae.encode(img).latent_dist.sample()
            latent = (
                latent - pipe.vae.config.shift_factor
            ) * pipe.vae.config.scaling_factor

            with open(prompt_path, "r") as f:
                prompt = f.read().strip()

            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
            )

        return {
            "clean_latents": latent.cpu().squeeze(0),
            "prompt_embeds": prompt_embeds.cpu().squeeze(0),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu().squeeze(0),
        }
    except Exception as e:
        raise RuntimeError(f"Error processing {image_path}") from e


def worker_process(file_chunk, gpu_id, result_queue):
    """工作进程，处理分配的文件块"""
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        diffusers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.disable_progress_bar()
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        
        # 初始化模型
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE, transformer=None
        ).to(device)

        # 处理每个文件对
        for image_path, prompt_path in file_chunk:
            try:
                sample = process_sample(image_path, prompt_path, pipe, device)
                result_queue.put(("success", sample, image_path))
            except Exception as e:
                result_queue.put(("error", str(e), image_path))

        # 发送完成信号
        result_queue.put(("done", gpu_id, None))

    except Exception as e:
        result_queue.put(("fatal", str(e), None))


def collect_file_pairs(directory):
    """收集所有有效的图像-文本对"""
    file_pairs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpeg", ".jpg", ".png")):
                image_path = os.path.join(root, file)
                base_name = os.path.splitext(image_path)[0]
                prompt_path = base_name + ".txt"
                if os.path.exists(prompt_path):
                    file_pairs.append((image_path, prompt_path))
    return file_pairs


def main():
    parser = argparse.ArgumentParser(
        description="多GPU加速处理图像数据到LMDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="包含图像和文本文件的输入目录")
    parser.add_argument("output_lmdb", help="输出LMDB文件路径")
    parser.add_argument("--gpus", default="0", help="使用的GPU ID列表，用逗号分隔")
    args = parser.parse_args()

    # 初始化设置
    gpu_ids = list(map(int, args.gpus.split(",")))
    file_pairs = collect_file_pairs(args.input_dir)
    chunks = np.array_split(file_pairs, len(gpu_ids))

    print(f"[bold green]🚀 启动处理:[/bold green]")
    print(f"• 找到 [bold]{len(file_pairs)}[/bold] 个有效文件对")
    print(f"• 使用GPU: [bold]{args.gpus}[/bold]")
    print(f"• 输出路径: [bold]{args.output_lmdb}[/bold]")

    # 创建进程间通信队列
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_queue = manager.Queue()

    # 启动工作进程
    processes = []
    for i, (chunk, gpu_id) in enumerate(zip(chunks, gpu_ids)):
        p = ctx.Process(target=worker_process, args=(chunk, gpu_id, result_queue))
        p.start()
        processes.append(p)

    # 初始化进度条
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn()
    )

    success_count = 0
    error_count = 0
    completed_gpus = set()

    with progress:
        task = progress.add_task("处理文件中", total=len(file_pairs))

        # 创建LMDB环境
        env = lmdb.open(args.output_lmdb, map_size=int(1e12), readonly=False)

        while True:
            result = result_queue.get()
            status = result[0]

            if status == "success":
                _, sample, path = result
                with env.begin(write=True) as txn:
                    key = f"{success_count:08}".encode()
                    txn.put(key, pickle.dumps(sample))
                success_count += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"处理文件: {success_count}/{len(file_pairs)}",
                )

            elif status == "error":
                _, error_msg, path = result
                print(f"[bold red]⚠️ 错误处理文件 {path}:[/bold red]\n{error_msg}")
                error_count += 1
                progress.update(task, advance=1)

            elif status == "done":
                _, gpu_id, _ = result
                completed_gpus.add(gpu_id)
                if len(completed_gpus) == len(gpu_ids):
                    break

            elif status == "fatal":
                _, error_msg, _ = result
                print(f"[bold red]💥 致命错误:[/bold red] {error_msg}")
                break

    # 清理进程
    for p in processes:
        p.join()
        p.close()

    print(f"\n[bold green]✅ 处理完成![/bold green]")
    print(f"• 成功处理: [bold green]{success_count}[/bold green]")
    print(f"• 失败文件: [bold red]{error_count}[/bold red]")
    print(f"• 输出路径: [bold]{args.output_lmdb}[/bold]")


if __name__ == "__main__":
    main()
