docker build -t duanyll/flux-control:v3 flux_control

docker run --rm -it --gpus all \
  -v duanyll_huggingface:/root/.cache/huggingface/hub \
  -v duanyll_torch_hub:/root/.cache/torch/hub \
  -v ./data:/data \
  -v ./config:/config \
  -v ./runs:/runs \
  -v ./accelerate-config:/root/.cache/huggingface/accelerate \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  -e OMP_NUM_THREADS=8 \
  --shm-size=40.96gb \
  duanyll/flux-control:v3 python -m flux_control.scripts.make_collage_dataset \
  --input_dir /data/openvid \
  --output /data/openvid/lmdb \
  --caption_file /data/openvid/video_caption_dict.pkl \
  # --max_samples 500