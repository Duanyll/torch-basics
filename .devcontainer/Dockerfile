FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

COPY ./apt.conf /etc/apt/apt.conf.d/99custom

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git gdb ffmpeg libsm6 libxext6 wget tmux zip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-cache-dir \
    h5py einops tqdm matplotlib tensorboard torch-tb-profiler ninja scipy

# Append extra packages below if needed