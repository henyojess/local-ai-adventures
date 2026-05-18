# E06_mtp5

**Status**: PASS  
**Run**: 2026-05-17T14:20:21Z  
**Duration**: 595s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `ocicek/Qwen3.6-27B-NVFP4`
- **Max model len**: 128K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.40
- **Max num seqs**: 2
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 5}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 334.88 | 186.01 | 23.53 | 23.21 | 500 |
| code_gen | 331.68 | 186.20 | 26.80 | 26.38 | 500 |
| long_context | 338.01 | 184.92 | 19.84 | 19.62 | 500 |
| short_prose | 329.29 | 185.79 | 19.46 | 19.25 | 500 |
| reasoning | 336.63 | 185.81 | 24.41 | 24.07 | 500 |

**Aggregate decode throughput**: 22.81 tok/s (mean of medians, range 19.46–26.80)

## Speculative decoding metrics

- **Drafts produced**: 3,136
- **Draft tokens**: 15,680
- **Accepted tokens**: 10,030
- **Acceptance rate**: 63.97%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._