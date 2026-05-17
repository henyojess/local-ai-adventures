# E03_mtp2

**Status**: PASS  
**Run**: 2026-05-17T12:29:28Z  
**Duration**: 641s (bench only)

## Configuration

- **Image**: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- **vLLM**: `0.19.2rc1.dev134+gfe9c3d6c5`
- **Model**: `ocicek/Qwen3.6-27B-NVFP4`
- **Max model len**: 128K
- **KV cache dtype**: fp8
- **GPU memory utilization**: 0.40
- **Max num seqs**: 2
- **Speculative decoding**: {'method': 'mtp', 'num_speculative_tokens': 2}

## Results (median across runs)

| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |
|---|---|---|---|---|---|
| structured | 269.81 | 126.26 | 21.76 | 21.54 | 500 |
| code_gen | 274.36 | 125.70 | 22.95 | 22.70 | 500 |
| long_context | 275.02 | 126.41 | 19.64 | 19.47 | 500 |
| short_prose | 270.95 | 126.78 | 19.01 | 18.86 | 500 |
| reasoning | 276.51 | 125.97 | 21.77 | 21.55 | 500 |

**Aggregate decode throughput**: 21.03 tok/s (mean of medians, range 19.01–22.95)

## Speculative decoding metrics

- **Drafts produced**: 5,157
- **Draft tokens**: 10,314
- **Accepted tokens**: 8,438
- **Acceptance rate**: 81.81%

## Files

- `run_vllm.sh` — exact launch script (pinned image digest)
- `boot.log` — vLLM boot log captured after /health passed
- `boot_meta.json` — container provenance
- `nvidia_smi_pre.txt` — GPU state at experiment start
- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)
- `benchmark_result.txt` — raw bench output
- `benchmark_result.json` — parsed bench result (this file's source data)

_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._