import torch
from safetensors.torch import load_file, save_file
import sys
import os

def fix_safetensors(input_path, output_path=None):
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fixed{ext}"

    # 加载 safetensors 文件
    tensors = load_file(input_path)

    new_tensors = {}
    for key, tensor in tensors.items():
        # 修正键名
        new_key = key.replace("module.", "").replace("default.", "")

        # 转换 float32 为 bfloat16（其他类型不变）
        if tensor.dtype == torch.float32:
            tensor = tensor.to(dtype=torch.bfloat16)

        new_tensors[new_key] = tensor

    # 保存修正后的 safetensors 文件
    save_file(new_tensors, output_path)
    print(f"Fixed safetensors saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_safetensors.py path/to/model.safetensors")
        sys.exit(1)

    input_file = sys.argv[1]
    fix_safetensors(input_file)
