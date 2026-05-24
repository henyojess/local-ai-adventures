# E08_fp8_quant_mtp1

**Status**: PASS  
**Run**: 2026-05-23T12:31:39Z  
**Duration**: 1134s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `Qwen/Qwen3.6-27B-FP8`
- **Max model len**: 32K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.42
- **Max num seqs**: 1
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 1}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 327.69 | 146.36 | 13.37 | 13.28 | 500 |
| code_gen | 323.16 | 147.28 | 13.39 | 13.30 | 500 |
| long_context | 329.40 | 146.64 | 12.60 | 12.52 | 500 |
| short_prose | 322.00 | 146.84 | 12.97 | 12.89 | 500 |
| reasoning | 647.13 | 169.35 | 11.33 | 11.27 | 500 |

**Aggregate decode throughput**: 12.73 tok/s (mean of medians, range 11.33–13.39)

## Speculative decoding metrics

- **Drafts produced**: 6,812
- **Draft tokens**: 6,812
- **Accepted tokens**: 6,271
- **Acceptance rate**: 92.06%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._