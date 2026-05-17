# E02_mtp1

**Status**: PASS  
**Run**: 2026-05-17T11:53:51Z  
**Duration**: 737s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `ocicek/Qwen3.6-27B-NVFP4`
- **Max model len**: 128K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.40
- **Max num seqs**: 2
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 1}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 231.36 | 104.71 | 18.54 | 18.42 | 500 |
| code_gen | 226.48 | 104.72 | 17.91 | 17.80 | 500 |
| long_context | 231.35 | 105.13 | 17.32 | 17.22 | 500 |
| short_prose | 227.34 | 104.92 | 18.02 | 17.90 | 500 |
| reasoning | 236.27 | 104.56 | 18.57 | 18.45 | 500 |

**Aggregate decode throughput**: 18.07 tok/s (mean of medians, range 17.32–18.57)

## Speculative decoding metrics

- **Drafts produced**: 6,934
- **Draft tokens**: 6,934
- **Accepted tokens**: 6,212
- **Acceptance rate**: 89.59%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._