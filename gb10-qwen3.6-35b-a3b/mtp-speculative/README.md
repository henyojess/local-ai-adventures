# MTP speculative decoding on unsloth/Qwen3.6-35B-A3B-NVFP4

Closes the open MTP question from the [parent README](../README.md):
the unsloth NVFP4 checkpoint *does* ship an MTP head, vLLM *does* load
it, and enabling speculative decoding *makes throughput strictly worse*
at every speculation depth tested. Acceptance rate is ~0.04% at
`num_speculative_tokens=1` and degrades further at n=2 and n=3.

## TL;DR

- MTP head loads but produces unusable drafts (5–6 accepts out of
  13,000+ drafts per bench, across n=1, n=2, n=3)
- Decode throughput drops **−32%** at n=1, **−49%** at n=2, **−54%** at n=3
- Per-token latency (TPOT) gets monotonically worse: 22.9 ms → 33.7 ms →
  45.1 ms → 49.2 ms as n grows
- **Operational recommendation: do not enable spec decode on this
  checkpoint.** The 44 tok/s no-spec baseline is the ceiling.
- The ~11 tok/s gap to RedHatAI's published 55.9 (Scargall, Apr 2026)
  is a checkpoint defect, not a tuning gap — RedHatAI's MTP head works,
  this one does not.

## What was measured

Four configurations, same image digest, same prompts, same bench
script, three docker restarts in between to swap the spec config:

| config  | `num_speculative_tokens` | drafts | accepted | acceptance |
|---------|-------------------------:|-------:|---------:|-----------:|
| no-spec | n/a                      | 0      | 0        | n/a        |
| mtp-n1  | 1                        | 13,166 | 5        | 0.04%      |
| mtp-n2  | 2                        | 13,209 | 6        | 0.02%      |
| mtp-n3  | 3                        | 13,159 | 5        | 0.01%      |

Acceptance counters scraped from `/metrics` before and after each bench;
the table shows per-run deltas, not cumulative.

## Throughput and latency

Median across all five prompt categories (5 runs each, 500 max tokens,
temperature=0):

| config  | decode tok/s | TPOT (ms) | TTFT (ms) | vs. baseline |
|---------|-------------:|----------:|----------:|-------------:|
| no-spec |        43.71 |     22.88 |        83 |        ref   |
| mtp-n1  |        29.63 |     33.75 |       122 |     −32.2%   |
| mtp-n2  |        22.20 |     45.05 |       141 |     −49.2%   |
| mtp-n3  |        20.32 |     49.22 |       146 |     −53.5%   |

Throughput drops monotonically with speculation depth. The reason is
visible in the draft counts: with acceptance near zero, every decode
step still verifies exactly one token (the bonus token), so the model
takes the same number of forward passes — but each step also costs an
additional drafter forward pass per speculative token. More n, more
wasted work, fewer net tokens per second.

## Why the drafter doesn't work

Boot log emitted three accuracy warnings on every MTP run:

```
NVFP4 Marlin assumes the scales to be >=0, but has encountered negative
scales. Accuracy will likely be degraded.

In NVFP4 linear, the global scale for input or weight are different for
parallel layers (e.g. q_proj, k_proj, v_proj). This will likely result
in reduced accuracy.

w1_weight_global_scale must match w3_weight_global_scale. Accuracy may
be affected.
```

The verifier (35B parameters) absorbs these errors without producing
obviously broken output — coherence is preserved, smoke tests pass,
the 44 tok/s no-spec config remains operationally fine. The drafter
is a small MTP head and gets no second chance: its forward pass either
produces a usable token distribution or it doesn't. On this checkpoint,
it doesn't.

This is consistent with how llmcompressor recipes handle MTP heads:
they typically pass through the same NVFP4 quantization pass as the
rest of the model. The recipe was tuned for the main model's weight
distributions, not the head's. The result is a head that loads but
drafts noise.

We did not confirm the numerical mechanism with a per-layer activation
audit — that would be the next investigation if the goal were to *fix*
unsloth's checkpoint rather than characterize it. For the operational
question ("should I turn MTP on?"), the answer is settled.

## What vLLM logged on the MTP runs

Confirming the head loads — from `boot-logs/mtp-n1.log`:

```
Resolved architecture: Qwen3_5MoeMTP
Loading drafter model...
Loading weights took 13.44 seconds
Detected MTP model. Sharing target model embedding weights with the
  draft model.
Detected MTP model. Sharing target model lm_head weights with the
  draft model.
```

Spec decode is active and the head is loaded; the failure is downstream
in the numerical quality of the head's outputs, not in vLLM's plumbing.

One additional warning worth flagging:

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInferBackend; setting cudagraph_mode=PIECEWISE
```

vLLM downgrades the CUDA graph mode when spec decode is on. This costs
a small amount of throughput independently of drafter quality — even
a perfectly-working MTP head would lose some of its theoretical win to
this downgrade. On a working checkpoint that wouldn't matter (the
acceptance gain would dominate). On this one it doesn't help.

## Reproducing

The bench script and run commands are in this folder:

- [`bench.sh`](./bench.sh) — the harness used for all four runs
- [`run-commands.md`](./run-commands.md) — full docker invocations for
  each config
- [`boot-logs/`](./boot-logs/) — sanitized vLLM startup logs
- [`bench-results/`](./bench-results/) — raw bench output files
- [`00-inspection/`](./00-inspection/) — the weight-name inspection
  script and its output (see methodology footnote below)

Image digest used for every run:

```
vllm/vllm-openai:cu130-nightly
sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc
```

(Same digest as the parent README. The MTP failure is not an artifact
of an image change.)

## Operational implication

The parent README's no-spec configuration is the correct one for this
checkpoint. Do not add `--speculative-config` flags. If you want
working spec decode on Qwen3.6-35B-A3B at NVFP4 precision on GB10
hardware, the RedHatAI variant is the checkpoint to use; we have not
yet benchmarked it on this hardware, but Scargall's 55.9 tok/s number
suggests its MTP head is functional.

## What this closes and what's still open

**Closed by this work:**
- *Is MTP available on this checkpoint?* Yes — head present, vLLM loads it.
- *Can we close the gap to Scargall's 55.9 by enabling MTP?* No — the
  head is non-functional, n=1/2/3 all degrade throughput.
- *Is the ~11 tok/s published gap a tuning issue or a checkpoint
  issue?* Checkpoint issue.

**Still open (now reframed):**
- Direct A/B comparison against RedHatAI/Qwen3.6-35B-A3B-NVFP4 on this
  hardware to quantify exactly what the broken drafter costs vs. the
  working one. With both checkpoints in hand and the bench script
  stable, this is now a 2–3 hour experiment rather than the open-ended
  investigation it would have been before item 1.
- Quality evaluation (MMLU / GSM8K / HumanEval) on the unsloth
  checkpoint, given the three accuracy warnings logged at load time.
  The warnings affect the drafter visibly; whether they affect the
  verifier subtly is an empirical question.

## Methodology footnote

Initial weight-name inspection ([`00-inspection/inspect_mtp.py`](./00-inspection/inspect_mtp.py))
produced a false negative: searching the single `model.safetensors`
file for keys matching `mtp` / `nextn` / `next_n` returned zero hits.
This was wrong. Boot log evidence (vLLM resolving a second architecture
`Qwen3_5MoeMTP` and reporting a successful 13.44-second drafter weight
load) corrected the inspection. The MTP weights are present under names
that do not contain those substrings. The inspection script's output is
preserved in this folder as a record of the investigative path; the
correct ground truth is the boot log.

Bench script: [`bench.sh`](./bench.sh) v0.3.2. Sources of provenance
captured automatically in each result file:

- ISO timestamp (UTC)
- Hostname
- GPU model and driver version
- Endpoint URL and queried model name
- vLLM version (from `/version`)
- Docker image reference and SHA256 digest (resolved by port)
- Bench config (runs, max_tokens, prompts hash)
- Spec decode counters at start and end (deltas → acceptance rate)

Bench prompts are deterministic (`temperature=0`) and the same five
across all four configs, hashed in each output file as `prompts_hash:
7f3ff79401f3`.