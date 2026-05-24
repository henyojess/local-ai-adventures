# E09_fp8_quant_mtp2

**Status**: PASS  
**Run**: 2026-05-23T13:08:44Z  
**Duration**: 833s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `Qwen/Qwen3.6-27B-FP8`
- **Max model len**: 32K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.42
- **Max num seqs**: 1
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 2}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 364.47 | 167.62 | 16.82 | 16.64 | 500 |
| code_gen | 358.82 | 167.72 | 16.90 | 16.73 | 500 |
| long_context | 365.36 | 166.75 | 14.81 | 14.68 | 500 |
| short_prose | 359.28 | 166.71 | 15.67 | 15.53 | 500 |
| reasoning | 372.87 | 167.92 | 16.06 | 15.90 | 500 |

**Aggregate decode throughput**: 16.05 tok/s (mean of medians, range 14.81–16.90)

## Speculative decoding metrics

- **Drafts produced**: 4,893
- **Draft tokens**: 9,786
- **Accepted tokens**: 8,250
- **Acceptance rate**: 84.30%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._