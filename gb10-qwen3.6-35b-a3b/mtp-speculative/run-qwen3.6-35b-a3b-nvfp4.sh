#!/bin/bash
# run-qwen3.6-35b-a3b-nvfp4.sh
docker rm -f vllm-8001; sleep 1

docker run -d --name vllm-8001 \
  --cpuset-cpus="10-19" --gpus all --network=host --ipc=host \
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
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  -v $HOME/.cache/torch:/root/.cache/torch \
  vllm/vllm-openai:cu130-nightly \
    unsloth/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name coder qwen3.6-35b-a3b \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 128K \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.265 \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-cache-dtype float16 \
    --mamba-cache-dtype float16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-cudagraph-capture-size 256 \
    --trust-remote-code \
    --generation-config vllm \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

docker logs -f vllm-8001
