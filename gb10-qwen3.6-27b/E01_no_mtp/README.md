# E01_no_mtp

**Status**: PASS  
**Run**: 2026-05-17T09:20:21Z  
**Duration**: 1100s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `serve`
- **Max model len**: 128K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.40
- **Max num seqs**: 2
- **Speculative decoding**: none

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 122.23 | 82.69 | 12.09 | 12.08 | 500 |
| code_gen | 120.57 | 82.84 | 12.07 | 12.06 | 500 |
| long_context | 123.96 | 82.40 | 12.14 | 12.12 | 500 |
| short_prose | 118.33 | 82.72 | 12.09 | 12.08 | 500 |
| reasoning | 126.94 | 82.86 | 12.07 | 12.05 | 500 |

**Aggregate decode throughput**: 12.09 tok/s (mean of medians, range 12.07–12.14)

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