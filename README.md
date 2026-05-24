# local-ai-adventures

A collection of reproducible, documented configurations for running open-weight LLMs locally with measured performance and exact reproducibility steps.

## Projects

### [GB10 Qwen3.6-35B-A3B](./gb10-qwen3.6-35b-a3b/)

First documented working configuration of `unsloth/Qwen3.6-35B-A3B-NVFP4` on ASUS Ascent GX10 / NVIDIA GB10.

**Highlights:**
- **44.5 tok/s** single-stream, **81.0 tok/s** aggregate (2-concurrent)
- 21.5 GB resident memory, leaves ~100 GB free for co-residency
- Marlin NVFP4 weight-only quantization on SM 12.1
- vLLM `cu130-nightly`, fully reproducible with immutable image digest
- **MTP finding:** unsloth's NVFP4 quant silently broke the MTP head (0.04% acceptance), regressing throughput 32–54% — the cautionary tale that motivated the 27B project

[Full setup guide →](./gb10-qwen3.6-35b-a3b/README.md)

### [GB10 Qwen3.6-27B](./gb10-qwen3.6-27b/)

First published NVFP4-vs-FP8 × MTP-depth sweep for dense `Qwen3.6-27B` on GB10, with vision and MTP preserved per the model card.

**Highlights:**
- **MTP head is healthy:** 89.59% acceptance at n=1 on the [`ocicek/Qwen3.6-27B-NVFP4`](https://huggingface.co/ocicek/Qwen3.6-27B-NVFP4) checkpoint — the opposite of the 35B's broken-MTP outcome, same test
- **23.40 tok/s** peak (NVFP4, MTP n=4), a 1.94× lift over the 12.09 tok/s no-MTP baseline
- **NVFP4 beats FP8 at every speculation depth** — 1.53× at baseline, ~1.20× at n=4
- Acceptance is governed by the MTP head, not the quant format: NVFP4 and FP8 acceptance curves nearly coincide
- 12 experiments, vLLM `cu130-nightly`, pinned image digest

[Full setup guide →](./gb10-qwen3.6-27b/README.md)

## Contributing

Each project documents:
- Exact image digests / versions (no mutable tags)
- Measured throughput with variance
- Memory footprint
- Reproducibility instructions

Use these as templates for your own benchmarks.
