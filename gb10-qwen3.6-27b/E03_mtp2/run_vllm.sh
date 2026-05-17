#!/bin/bash
# E03_mtp2/run_vllm.sh — launch vLLM serving Qwen3.6-27B-NVFP4 with MTP n=2.
#
# Identical to E01 except for the trailing --speculative-config flag.
# Self-contained:
#   - Tears down any existing container on the same name slot before launch.
#   - Idempotent: re-running this script does the right thing whether or not
#     a previous container exists or is wedged.
#   - The orchestrator (run_all_experiments.sh) does NOT need to pre-teardown;
#     each E0N script owns its slot.

set -euo pipefail

# ── pinned image (registry RepoDigest, captured 2026-05-16 from 35B run) ──────
# Tag at pull time: vllm/vllm-openai:cu130-nightly
# vllm version:    0.19.2rc1.dev134+gfe9c3d6c5
VLLM_IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"

CONTAINER_NAME="vllm-8001"
MODEL_REPO="ocicek/Qwen3.6-27B-NVFP4"

# ── teardown any existing slot occupant ───────────────────────────────────────
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
# Brief pause: `docker rm -f` returns once SIGKILL is sent, but the slot
# cleanup (name reservation, network namespace) can lag by ~100ms.
sleep 1

# ── launch ────────────────────────────────────────────────────────────────────
# No --rm: we want the container to persist after exit so logs are inspectable.
# The next run_vllm.sh invocation will clean it up at the top.
exec docker run -d --name "$CONTAINER_NAME" \
  --cpuset-cpus="10-19" \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  "$VLLM_IMAGE" \
    "$MODEL_REPO" \
    --served-model-name coder qwen3.6-27b \
    --host 0.0.0.0 --port 8001 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --generation-config vllm \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --max-model-len 128K \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.40 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'