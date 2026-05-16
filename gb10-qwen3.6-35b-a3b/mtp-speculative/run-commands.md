# Run commands for MTP speculative decoding experiment

All four configurations use the same image digest, same model, same
host, same bench script. The only diff between configurations is the
presence (and value) of `--speculative-config`.

Image used for every run:

```
vllm/vllm-openai:cu130-nightly
sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc
```

The bench script auto-detects this digest from the container on the
target port and records it in every result file.

## Container lifecycle (per config)

```bash
# stop and remove any previous container before starting a new one
docker rm -f vllm-8001; 

# start the next config (commands below)
# ... wait for "Starting vLLM server on http://0.0.0.0:8001"

# save the boot log
docker logs vllm-8001 > boot-logs/<label>.log 2>&1
sed -i -E 's|tcp://[0-9.]+:[0-9]+|tcp://REDACTED:PORT|g' boot-logs/<label>.log

# run the bench
./bench.sh --label <label>

# repeat for the next config
```

## Config 1 — no-spec baseline

```bash
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
    --host 0.0.0.0 --port 8001 \
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
    --generation-config vllm
```

Bench: `./bench.sh --label no-spec`

## Config 2 — MTP, num_speculative_tokens=1

Add this flag to the end of the docker command above:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Bench: `./bench.sh --label mtpn1`

## Config 3 — MTP, num_speculative_tokens=2

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Bench: `./bench.sh --label mtpn2`

## Config 4 — MTP, num_speculative_tokens=3

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Bench: `./bench.sh --label mtpn3`

## Notes

- `--max-model-len 128K` is the operational config used for this
  experiment. The parent README uses `32768` for the headline numbers;
  context length does not materially affect decode throughput at the
  prompt sizes used by the bench.
- `--gpu-memory-utilization 0.265` leaves ~95 GB free for co-residency
  with other models on the same machine. Higher values would yield
  larger KV cache but were not the goal of this experiment.
- Boot logs need to be sanitized for the host IP that vLLM logs in
  its distributed-init line. The `sed` invocation above handles it.