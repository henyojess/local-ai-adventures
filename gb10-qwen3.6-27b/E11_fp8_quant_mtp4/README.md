# E11_fp8_quant_mtp4

**Status**: PASS  
**Run**: 2026-05-23T13:51:36Z  
**Duration**: 699s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `Qwen/Qwen3.6-27B-FP8`
- **Max model len**: 32K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.42
- **Max num seqs**: 1
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 4}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 398.94 | 201.25 | 20.32 | 20.04 | 500 |
| code_gen | 397.52 | 201.25 | 21.94 | 21.61 | 500 |
| long_context | 399.83 | 201.58 | 15.38 | 15.22 | 500 |
| short_prose | 393.48 | 201.31 | 18.50 | 18.27 | 500 |
| reasoning | 408.23 | 201.30 | 21.01 | 20.70 | 500 |

**Aggregate decode throughput**: 19.43 tok/s (mean of medians, range 15.38–21.94)

## Speculative decoding metrics

- **Drafts produced**: 3,396
- **Draft tokens**: 13,584
- **Accepted tokens**: 9,774
- **Acceptance rate**: 71.95%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._