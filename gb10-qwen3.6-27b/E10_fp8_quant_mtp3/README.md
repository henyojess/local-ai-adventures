# E10_fp8_quant_mtp3

**Status**: PASS  
**Run**: 2026-05-23T13:30:49Z  
**Duration**: 740s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `Qwen/Qwen3.6-27B-FP8`
- **Max model len**: 32K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.42
- **Max num seqs**: 1
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 3}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 379.03 | 183.63 | 19.41 | 19.17 | 500 |
| code_gen | 380.34 | 185.36 | 19.79 | 19.54 | 500 |
| long_context | 384.60 | 185.44 | 15.74 | 15.58 | 500 |
| short_prose | 374.74 | 185.35 | 17.48 | 17.29 | 500 |
| reasoning | 389.56 | 185.24 | 19.24 | 19.00 | 500 |

**Aggregate decode throughput**: 18.33 tok/s (mean of medians, range 15.74–19.79)

## Speculative decoding metrics

- **Drafts produced**: 3,925
- **Draft tokens**: 11,775
- **Accepted tokens**: 9,321
- **Acceptance rate**: 79.16%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._