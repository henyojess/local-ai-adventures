#!/bin/bash
# E01_no_mtp/run_vllm.sh — launch vLLM serving Qwen3.6-27B-NVFP4 with no spec decoding.
#
# Baseline for the MTP comparison matrix. All other E0N scripts are this file
# plus a single --speculative-config line.
#
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
# `docker rm -f` works on running OR stopped containers and is idempotent
# (returns nonzero only if there's no such container, which we swallow).
# This handles three cases:
#   1. Previous experiment exited cleanly but container persisted (no --rm).
#   2. Previous experiment crashed and left a stopped container.
#   3. Previous experiment is still running (e.g. operator re-ran by mistake).
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
    --gpu-memory-utilization 0.40