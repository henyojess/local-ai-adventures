# gb10-qwen3.6-27b

Benchmarking dense Qwen3.6-27B-NVFP4 on GB10.

> 🚧 **Status: Work in progress.** Experiments E01-E04 pending. Headline numbers, comparison tables, and verdicts fill in as runs complete.

## TL;DR

*Filled in after experiments complete. Will include: median tok/s across categories, MTP acceptance rate (the key finding), and verdict on whether the ocicek checkpoint's MTP head survives quantization.*

## Methodology

This project follows the shared methodology defined in the [top-level README](../README.md#methodology): pinned image digests, acceptance rate empirically measured via `/metrics` scrape, deterministic benchmarks (temp=0, 5 categories × 5 runs + 2 warmup, `max_tokens=500`), and self-contained per-experiment folders. The MTP-acceptance discipline is inherited directly from the 35B project, which uncovered that vLLM reports MTP as "working" when the drafter is producing noise.

## Why this model: Qwen3.6-27B-NVFP4

The checkpoint is [`ocicek/Qwen3.6-27B-NVFP4`](https://huggingface.co/ocicek/Qwen3.6-27B-NVFP4).

- **Strongest open-weights model in its size class.** Qwen3.6-27B scores 46 on the Artificial Analysis Intelligence Index v4.0, versus a median of 15 for open-weights models of similar size. Artificial Analysis described it as the new open-weights leader under 150B parameters at release.
- **Fits GB10 with substantial headroom.** ~13.5 GB resident weight footprint at NVFP4. With KV cache and CUDA graph overhead, the working set is well under half of GB10's 128 GB unified memory, leaving room for co-resident models.
- **Dense, not MoE.** Simpler performance surface than the 35B-A3B work in this repo. No shared-expert handling, no MoE kernel fallback risk. 
- **Vision tower preserved (BF16).** Multimodal capability without a separate model.
- **MTP head preserved (BF16) per the model card.** This is the central test of this project: the 35B work found that unsloth's NVFP4 quantization silently broke the MTP head (0.04% acceptance, 32-54% throughput regression). The ocicek checkpoint explicitly preserves the MTP head in BF16. Does that translate to working speculative decoding in practice? E02-E04 measure it.
- **Quantized on DGX Spark GB10 with `sm_121a`.** The author targeted this exact hardware. It's about as well-fitted as a public NVFP4 checkpoint gets.

## Why this image: vLLM `cu130-nightly`

- **SM 12.1 (`sm_121a`) support is in the cu130 line.** Earlier cu128 images predate proper Blackwell-consumer support. cu130 is where the relevant kernels actually live.
- **Matches the 35B project in this repo.** Same image stack across both projects keeps cross-project comparisons apples-to-apples.
- **Matches stevescargall's published RedHatAI 35B-A3B-NVFP4 GB10 benchmark.** Same image as the closest peer-published result, for the same reason.
- **Pinned by sha256 digest.** The `cu130-nightly` tag identifier is informational; the digest is the contract. The exact digest used is captured in every per-experiment folder's `boot_meta.json`.

**Pre-integration floor.** NVIDIA's April 2026 NVFP4 perf update announced that FlashInfer 0.6.8 and Luke Alonso's SM 120/121-optimized MoE and GEMM kernels are landing in vLLM imminently. The numbers in this project represent performance *before* those kernels integrate. Treat these results as the conservative baseline; expect upstream improvements to raise the floor in subsequent releases.

## Experiment matrix

Four experiments, single dimension varied: speculative decoding configuration. Everything else (prefix caching, prompts, token budget, image digest, GPU memory utilization) is held constant.

| ID | Config | Hypothesis tested | Status | Result |
|---|---|---|---|---|
| E01 | No speculative decoding | Baseline floor. If MTP variants don't beat this, MTP head is broken or PIECEWISE graph overhead exceeds MTP gains. | Pending | TBD |
| E02 | MTP `n=1` | Single-layer MTP applied once. Cleanest test of whether ocicek's MTP head is healthy post-quantization. | Pending | TBD |
| E03 | MTP `n=2` | Tests whether vLLM's recursive application of the single MTP layer degrades acceptance for n>1. | Pending | TBD |
| E04 | MTP `n=3` | Continuation of the n-sweep. If E02 > E03 > E04 monotonically, recursive MTP application hurts and n=1 is optimal. | Pending | TBD |

**Story branches based on results.** If acceptance is healthy (~80%+) at E02 and degrades cleanly with n, the verdict is "MTP works on this checkpoint; recursive application of a single MTP layer is the limiting factor; n=1 is optimal." If acceptance is low at E02, this becomes a "two checkpoints, same broken-MTP pattern" story alongside the 35B finding — different cause (preserved-but-not-functional vs broken-by-quant), same outcome.

## Directory structure

```
gb10-qwen3.6-27b/
├── README.md                       # This file
├── benchmark.sh                    # Shared benchmark driver, copied from 35B repo
├── run_all_experiments.sh          # Overnight orchestrator for all 4 experiments
├── run_all_experiments.log         # Top-level orchestration trace
├── E01_no_mtp/                     # Experiment 1: baseline, no speculative decoding
│   ├── run_vllm.sh                 # Docker run script for this experiment
│   ├── boot.log                    # Full container startup log
│   ├── boot_meta.json              # Machine-readable run provenance
│   ├── benchmark_result.txt        # Raw benchmark output (human-readable)
│   ├── benchmark_result.json       # Same data, machine-parseable
│   ├── metrics_start.txt           # Full /metrics dump before bench
│   ├── metrics_end.txt             # Full /metrics dump after bench
│   └── README.md                   # Per-experiment digest and verdict
├── E02_mtp1/                       # Experiment 2: num_speculative_tokens=1
├── E03_mtp2/                       # Experiment 3: num_speculative_tokens=2
└── E04_mtp3/                       # Experiment 4: num_speculative_tokens=3
```

E02-E04 mirror E01's internal file structure exactly.

### Per-file details

**`benchmark.sh`** — Shared benchmark driver, copied from `gb10-qwen3.6-35b-a3b/mtp-speculative/bench.sh`. Streaming OpenAI client, 5 prompt categories × 5 runs + 2 warmup, temp=0, `max_tokens=500`. Scrapes `/metrics` at start and end for spec-decode counters (`vllm:spec_decode_num_drafts_total`, `num_draft_tokens_total`, `num_accepted_tokens_total`) and computes acceptance rate. Same script invoked by all 4 experiments to guarantee apples-to-apples comparison.

**`run_all_experiments.sh`** — Overnight orchestrator. For each E0N in order: tears down any existing container, sanity-checks GPU state, launches `E0N/run_vllm.sh`, polls `/health` with a 5-minute timeout, captures boot log and provenance metadata, snapshots `/metrics` before the bench, runs `benchmark.sh` writing into `E0N/`, snapshots `/metrics` after the bench, parses the txt into json, generates the per-experiment README stub, tears down the container, and sleeps for memory release. Crash-isolated so one experiment's failure doesn't stop the rest.

**`run_all_experiments.log`** — Top-level orchestrator log. Timestamped phase lines per experiment (start, boot ok, bench done, cleanup ok, or failure modes). Designed for a 30-second morning scan.

**`E0N/run_vllm.sh`** — Self-contained docker run script for that one experiment. Pinned image digest, all flags inline, no external dependencies. Only difference between E0N scripts is the `--speculative-config` flag (absent for E01, present with `n=1/2/3` for E02/E03/E04).

**`E0N/boot.log`** — Full `docker logs vllm-8001` output from container start through "Application startup complete." Captured after `/health` responds. Used to verify kernel selection (`FlashInferCutlassNvFp4LinearKernel` vs Marlin fallback), KV cache size, max concurrency, MTP head load status, CUDA graph mode, and any warnings.

**`E0N/boot_meta.json`** — Machine-readable provenance: `image_ref`, `image_digest_sha256`, `container_id`, start/ready timestamps, boot duration, hostname, GPU name, driver version, CUDA version, vLLM version, host free memory at launch.

**`E0N/benchmark_result.txt`** — Raw `benchmark.sh` output. Per-prompt per-run table of `ttft_ms`, `tpot_ms`, `decode_t/s`, `total_t/s`, `tokens`, with median and p95 rows, plus spec-decode counter deltas and acceptance rate footer.

**`E0N/benchmark_result.json`** — Same data as `benchmark_result.txt`, parsed into structured JSON for `jq` diffing and to drive the cross-experiment results table in this README.

**`E0N/metrics_start.txt`** and **`E0N/metrics_end.txt`** — Full Prometheus `/metrics` dumps before and after the benchmark. Diff captures every counter (KV hit rate, queue depth, scheduler iteration time), not just the spec-decode subset `benchmark.sh` extracts.

**`E0N/README.md`** — One-screen per-experiment digest auto-written by the orchestrator: hypothesis being tested, config diff vs baseline, kernel selected, KV cache size and max concurrency, MTP head status, median tok/s per category, spec-decode acceptance rate, and a verdict line.

## How to reproduce

*This section will be finalized once `run_all_experiments.sh` is committed. High-level: clone the repo, pull the pinned image digest, run `./run_all_experiments.sh` from `gb10-qwen3.6-27b/`. Expected total runtime ~3 hours for all 4 experiments including teardown and warmup.*

## Results

*Filled in as experiments complete. Will include a comparison table across E01-E04 with per-category medians, acceptance rates, and a short prose verdict on what the data says about MTP on this checkpoint.*

## Comparison with published GB10 benchmarks

*Filled in alongside results. Comparison targets:*
- *rikkarth — Qwen3.6-35B-A3B FP8 on GB10*
- *stevescargall — RedHatAI Qwen3.6-35B-A3B-NVFP4 on GB10*
- *AEON-7 — Qwen3.6-27B abliterated on GB10*
- *ai-muninn — Qwen3.5-35B on GB10*
- *adadrag — Qwen3.5-35B GB10 guide*

To our knowledge, this is the first published systematic benchmark of dense Qwen3.6-27B-NVFP4 on GB10 with both vision and MTP preserved per the model card.

## What we chose not to vary (v1 scope)

Three dimensions are deliberately held fixed in v1. Each is a defensible follow-up project rather than an in-scope variable.

- **Self-quantization with ModelOpt or similar.** Calibration dataset choice, per-tensor vs per-channel scales, vision tower handling, and MTP head handling are all non-trivial. The 35B MTP finding is the cautionary tale: unsloth's quantization silently broke the MTP head. Self-quant introduces a confound; we'd rather measure ocicek's published artifact first and establish what "works" looks like before generating our own.
- **Container image sweep.** Alternative images exist (eugr's source-built community image with fastsafetensors, NGC's `nvcr.io/nvidia/vllm`, AEON-7's patched build). A sanity check on eugr is a reasonable v2 addition. Full image comparison is its own project.
- **Alternative NVFP4 checkpoints.** Other quants of Qwen3.6-27B exist (sakamakismile, mmangkad). A "which 27B-NVFP4 is best on GB10" comparison post is an optional follow-up.

## Caveats and open items

- **PIECEWISE CUDA graph mode.** FlashInfer + speculative decoding forces `CUDAGraphMode=PIECEWISE`; FULL graphs are unsupported in this combination. This costs throughput; the size of the cost vs FULL is not measured in v1.
- **compressed-tensors NVFP4 split q/k/v scale warning.** Observed at load time. The warning notes "reduced accuracy"; no MoE here, so no shared-expert pathway is affected. Worth flagging.
- **Qwen3.6 has only 1 MTP layer.** vLLM applies this single layer recursively for `n>1`. Degradation across the n-sweep would indicate that recursion, not the head itself, is the limit. This is exactly the hypothesis E02-E04 test.
- **Single-image study in v1.** See "What we chose not to vary."
- **Imminent upstream improvements.** FlashInfer 0.6.8 and Luke Alonso's SM 120/121 kernels are landing in vLLM per NVIDIA's April 2026 perf update. These numbers represent the pre-integration floor.

## Related work in this repo

The sibling project [`gb10-qwen3.6-35b-a3b/`](../gb10-qwen3.6-35b-a3b/README.md) benchmarks `unsloth/Qwen3.6-35B-A3B-NVFP4` on the same hardware. The [`mtp-speculative/`](../gb10-qwen3.6-35b-a3b/mtp-speculative/README.md) subfolder closed out the MTP question for that checkpoint with a specific finding: unsloth's NVFP4 quantization broke the MTP head (0.04% acceptance), causing MTP `n=1/2/3` to monotonically degrade throughput by 32-54%. That finding directly motivates the 27B work — does ocicek's preserved-MTP-in-BF16 checkpoint avoid the same fate?

## Acknowledgements

- [ocicek](https://huggingface.co/ocicek) for the `Qwen3.6-27B-NVFP4` checkpoint with vision and MTP preserved
- The Qwen team for the open-weights model
- The vLLM project and NVIDIA for the NVFP4 kernel work landing in cu130
- Prior GB10 benchmarkers — rikkarth, stevescargall, AEON-7, ai-muninn, adadrag — whose published work establishes the comparison baseline