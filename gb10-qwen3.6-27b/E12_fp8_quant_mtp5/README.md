# E12_fp8_quant_mtp5

**Status**: PASS  
**Run**: 2026-05-23T14:10:47Z  
**Duration**: 699s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `Qwen/Qwen3.6-27B-FP8`
- **Max model len**: 32K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.42
- **Max num seqs**: 1
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 5}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 418.08 | 220.01 | 22.02 | 21.66 | 500 |
| code_gen | 414.97 | 218.79 | 21.72 | 21.38 | 500 |
| long_context | 421.68 | 219.18 | 14.23 | 14.09 | 500 |
| short_prose | 413.64 | 220.55 | 18.54 | 18.30 | 500 |
| reasoning | 426.74 | 218.93 | 20.91 | 20.58 | 500 |

**Aggregate decode throughput**: 19.48 tok/s (mean of medians, range 14.23–22.02)

## Speculative decoding metrics

- **Drafts produced**: 3,121
- **Draft tokens**: 15,605
- **Accepted tokens**: 10,037
- **Acceptance rate**: 64.32%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._