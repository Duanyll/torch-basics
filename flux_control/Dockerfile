# PyTorch 2.6.0 is REQUIRED
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel AS kmcuda-builder

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.bfsu.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.bfsu.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

RUN cd /workspace \
    && git clone https://github.com/Duanyll/kmcuda

ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN cd /workspace/kmcuda \
    && cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCUDA_ARCH=89 -DCMAKE_CUDA_ARCHITECTURES=89 -Ssrc -Bbuild -G Ninja \
    && cmake --build build --config Release --target all --

FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.bfsu.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.bfsu.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git gdb ffmpeg libsm6 libxext6 wget tmux zip \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace/flux_control

COPY --from=kmcuda-builder /workspace/kmcuda/build/libKMCUDA.so /workspace/flux_control/datasets/collage/