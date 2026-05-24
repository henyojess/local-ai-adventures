# E07_fp8_quant_nomtp

**Status**: PASS  
**Run**: 2026-05-18T05:10:18Z  
**Duration**: 1664s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `Qwen/Qwen3.6-27B-FP8`
- **Max model len**: 8192
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.42
- **Max num seqs**: 2
- **Speculative decoding**: none

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 174.15 | 126.40 | 7.91 | 7.91 | 500 |
| code_gen | 171.73 | 126.39 | 7.91 | 7.91 | 500 |
| long_context | 175.88 | 126.40 | 7.91 | 7.91 | 500 |
| short_prose | 169.14 | 126.40 | 7.91 | 7.91 | 500 |
| reasoning | 181.12 | 126.40 | 7.91 | 7.90 | 500 |

**Aggregate decode throughput**: 7.91 tok/s (mean of medians, range 7.91–7.91)

## Speculative decoding metrics

Not applicable for this experiment (no speculative config).
- Drafts: 0
- Accepted: 0

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._