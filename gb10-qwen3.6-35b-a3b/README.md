# local-ai-adventures

Running SOTA open-weight LLMs on consumer hardware — measured and reproducible.

## Objective

Run state-of-the-art open-weight LLMs locally on consumer and prosumer hardware, with measured performance and exact reproducibility. Each project explores deployment across four dimensions: model choice, inference stack and container image, runtime flags, and model-side flags.

## Hardware

| Platform | GPU | Memory | Status |
|---|---|---|---|
| ASUS Ascent GX10 | NVIDIA GB10 | 128 GB unified | Active |
| Custom PC | RTX 5070 Ti | 16 GB VRAM + 32 GB DDR5 | Planned |

**ASUS Ascent GX10 / NVIDIA GB10.** ARM-based prosumer dev kit with 128 GB unified memory and SM 12.1 (`sm_121a`). The large unified memory pool enables co-residency of multiple quantized models with substantial headroom.

**Custom PC / RTX 5070 Ti.** Repurposed gaming build with 16 GB dedicated VRAM and 32 GB DDR5 system RAM. Representative consumer hardware class. Constrained VRAM means experiments here explore GPU/CPU offload tradeoffs and may use a different inference stack (e.g., llama.cpp) alongside or instead of vLLM.

## Projects

- **[Done]** [`gb10-qwen3.6-35b-a3b/`](./gb10-qwen3.6-35b-a3b/) — `unsloth/Qwen3.6-35B-A3B-NVFP4` on GB10. 44.5 tok/s single-stream, 81.0 tok/s aggregate (2-concurrent), 21.5 GB resident. Marlin NVFP4 on SM 12.1, vLLM `cu130-nightly`.
- **[WIP]** `gb10-qwen3.6-27b/` — `Qwen3.6-27B` on GB10. Four experiments pending.
- **[Planned]** Gemma 4 31B on GB10
- **[Planned]** Gemma 4 26B on GB10
- **[Planned]** Models on RTX 5070 Ti — exploring GPU/CPU offload with constrained VRAM; vLLM and/or alternatives (e.g., llama.cpp).

## Methodology

Shared principles across every project in this repo. Per-project READMEs link back here rather than restating them.

**Pinned image digests, never mutable tags.** Container images are referenced by `sha256:` digest, not by tags like `latest` or `cu130-nightly`. Tags move; digests don't. This is the difference between a benchmark someone can reproduce in six months and one they can't.

**Image choice is per-project.** Container image and inference stack are chosen per project based on the model and hardware, and documented in each project's README. There is no repo-wide standard image — what works for one model on one platform may not be optimal for another.

**Acceptance rate empirically measured.** For speculative decoding (MTP, EAGLE, draft models), acceptance rate is measured by scraping the inference server's `/metrics` endpoint during the actual benchmark run — not assumed from boot-log messages. vLLM will happily report MTP as "initialized" and "working" when the drafter is producing noise; the 35B project caught this with a 0.04% acceptance rate and 32-54% throughput regression on unsloth's NVFP4 quant. Boot logs are not evidence.

**Deterministic benchmarks.** Temperature 0, fixed prompt set across 5 categories × 5 runs plus 2 warmup runs, with a fixed token budget reported alongside results. This produces tight variance bounds and makes runs comparable across image versions, runtime flags, and model flags. Variance is reported alongside the mean.

**Self-contained per-experiment folders.** Each experiment's folder contains everything needed to reproduce it: exact launch command, image digest, environment, and measured results. Diffing two experiments is `diff -r exp-a/ exp-b/`.

**Caveats and open items, explicitly.** Every project documents what didn't work, what's uncertain, and what's untested. A configuration with caveats is more useful than one without — silence on a known limitation is a worse failure mode than acknowledging it.

## Use as a template

Each project folder can be forked as a starting point. The methodology section above is the contract.