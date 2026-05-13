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

[Full setup guide →](./gb10-qwen3.6-35b-a3b/README.md)

## Contributing

Each project documents:
- Exact image digests / versions (no mutable tags)
- Measured throughput with variance
- Memory footprint
- Reproducibility instructions

Use these as templates for your own benchmarks.