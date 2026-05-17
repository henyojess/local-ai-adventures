# E05_mtp4

**Status**: PASS  
**Run**: 2026-05-17T13:43:36Z  
**Duration**: 579s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `ocicek/Qwen3.6-27B-NVFP4`
- **Max model len**: 128K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.40
- **Max num seqs**: 2
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 4}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 312.29 | 167.83 | 24.57 | 24.25 | 500 |
| code_gen | 313.54 | 166.36 | 27.02 | 26.59 | 500 |
| long_context | 313.70 | 165.52 | 19.83 | 19.63 | 500 |
| short_prose | 310.88 | 166.36 | 20.69 | 20.46 | 500 |
| reasoning | 315.71 | 165.62 | 24.90 | 24.56 | 500 |

**Aggregate decode throughput**: 23.40 tok/s (mean of medians, range 19.83–27.02)

## Speculative decoding metrics

- **Drafts produced**: 3,410
- **Draft tokens**: 13,640
- **Accepted tokens**: 9,721
- **Acceptance rate**: 71.27%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._