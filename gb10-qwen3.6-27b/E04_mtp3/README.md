# E04_mtp3

**Status**: PASS  
**Run**: 2026-05-17T12:47:15Z  
**Duration**: 594s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `ocicek/Qwen3.6-27B-NVFP4`
- **Max model len**: 128K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.40
- **Max num seqs**: 2
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 3}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 293.37 | 146.45 | 23.83 | 23.55 | 500 |
| code_gen | 291.27 | 145.83 | 25.54 | 25.19 | 500 |
| long_context | 295.30 | 146.67 | 20.62 | 20.41 | 500 |
| short_prose | 291.45 | 146.56 | 19.68 | 19.48 | 500 |
| reasoning | 297.03 | 147.56 | 23.65 | 23.37 | 500 |

**Aggregate decode throughput**: 22.66 tok/s (mean of medians, range 19.68–25.54)

## Speculative decoding metrics

- **Drafts produced**: 3,978
- **Draft tokens**: 11,934
- **Accepted tokens**: 9,148
- **Acceptance rate**: 76.65%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._