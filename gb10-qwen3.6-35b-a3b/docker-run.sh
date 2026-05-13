#!/usr/bin/env bash
# Launch unsloth/Qwen3.6-35B-A3B-NVFP4 on NVIDIA DGX Spark (GB10)
# Tested config: see README.md for hardware/software stack details.
#
# Image digest tested:
#   vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit
#   - Model pre-downloaded: huggingface-cli download unsloth/Qwen3.6-35B-A3B-NVFP4
#   - Port 8000 free

set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:cu130-nightly}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen36-35b}"
PORT="${PORT:-8000}"
CPUSET="${CPUSET:-10-19}"

# Pin to the tested digest by default. Override with IMAGE=... to use a newer build.
if [[ "$IMAGE" == "vllm/vllm-openai:cu130-nightly" ]]; then
  echo "Tip: for byte-identical reproduction, run with:"
  echo "  IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 $0"
fi

docker run -d --name "$CONTAINER_NAME" \
  --cpuset-cpus="$CPUSET" --gpus all --network=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  "$IMAGE" \
    unsloth/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.25 \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-cache-dtype float16 \
    --mamba-cache-dtype float16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --generation-config vllm \
    --max-cudagraph-capture-size 256 \
    --trust-remote-code

echo ""
echo "Container started: $CONTAINER_NAME on port $PORT"
echo "Watch boot:        docker logs -f $CONTAINER_NAME"
echo "First boot takes ~4 minutes."