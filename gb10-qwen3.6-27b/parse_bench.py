#!/usr/bin/env python3
"""
parse_bench.py — parse benchmark.sh output into benchmark_result.json and
                 auto-write a per-experiment README.md.

Usage:
  parse_bench.py --bench-txt PATH --experiment NAME --out-json PATH --out-readme PATH \
                 [--boot-meta PATH] [--status PASS|FAILED] [--fail-reason STRING]

Design:
  - Strict parsing: if --status FAILED is passed, write skeletal JSON and a
    failure-flavored README; do not attempt to parse a possibly-corrupt bench.txt.
  - Pass-through medians: the bench .txt already computed median and p95 per
    prompt; we parse those lines as authoritative rather than re-computing
    from individual runs (avoids algorithm drift between bench and parser).
  - Self-consistent JSON: still records individual runs in addition to
    median/p95, so a downstream re-analysis can recompute if it wants to.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ── parsers ──────────────────────────────────────────────────────────────────

HEADER_LINE = re.compile(r"^([a-z_@]+):\s+(.+?)\s*$")
BENCH_VERSION_LINE = re.compile(r"benchmark\.sh v(\S+)\s+label=(\S+)")
RUN_LINE = re.compile(
    r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+\(run \d+\)\s*$"
)
MEDIAN_LINE = re.compile(
    r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+← median\s*$"
)
P95_LINE = re.compile(
    r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+← p95\s*$"
)


def parse_metrics_row(m: re.Match) -> dict[str, float | int]:
    return {
        "ttft_ms":    float(m.group(2)),
        "tpot_ms":    float(m.group(3)),
        "decode_tps": float(m.group(4)),
        "total_tps":  float(m.group(5)),
        "tokens":     int(m.group(6)),
    }


def parse_bench_txt(path: Path) -> dict[str, Any]:
    """Parse a complete bench output file. Raises on malformed input."""
    lines = path.read_text().splitlines()

    header: dict[str, str] = {}
    bench_version = "unknown"
    label = "unlabeled"
    prompts: dict[str, dict[str, Any]] = {}
    footer: dict[str, str] = {}

    section = "preamble"  # preamble → body → footer

    for line in lines:
        bm = BENCH_VERSION_LINE.search(line)
        if bm:
            bench_version = bm.group(1)
            label = bm.group(2)
            continue

        if line.startswith("="):
            # transitions: first === ends preamble; second === starts body;
            # third === ends body / starts footer.
            if section == "preamble" and header:
                section = "body"
            elif section == "body" and prompts:
                section = "footer"
            continue

        if section == "preamble":
            hm = HEADER_LINE.match(line)
            if hm:
                header[hm.group(1)] = hm.group(2)
            continue

        if section == "body":
            rm = RUN_LINE.match(line)
            if rm:
                name = rm.group(1)
                p = prompts.setdefault(name, {"runs": [], "median": None, "p95": None})
                p["runs"].append(parse_metrics_row(rm))
                continue
            mm = MEDIAN_LINE.match(line)
            if mm:
                prompts[mm.group(1)]["median"] = parse_metrics_row(mm)
                continue
            pm = P95_LINE.match(line)
            if pm:
                prompts[pm.group(1)]["p95"] = parse_metrics_row(pm)
                continue
            continue

        if section == "footer":
            fm = HEADER_LINE.match(line)
            if fm:
                footer[fm.group(1)] = fm.group(2)
            continue

    # ── validate structural completeness ──
    if not header:
        raise ValueError("no header found in bench .txt")
    if not prompts:
        raise ValueError("no per-prompt run lines found in bench .txt")
    for name, p in prompts.items():
        if p["median"] is None or p["p95"] is None:
            raise ValueError(f"prompt {name!r} missing median or p95 line")

    # ── compute aggregate ──
    medians = [p["median"]["decode_tps"] for p in prompts.values()]
    aggregate = {
        "decode_tps_mean_of_medians": round(sum(medians) / len(medians), 4),
        "decode_tps_min_median": round(min(medians), 4),
        "decode_tps_max_median": round(max(medians), 4),
    }

    # ── spec decode block ──
    def _int(key: str) -> int:
        v = footer.get(key, header.get(key, "0"))
        try:
            return int(float(v))
        except ValueError:
            return 0

    drafts_start    = _int("spec_drafts@start")
    draft_toks_start = _int("spec_draft_tokens@start")
    accepted_start  = _int("spec_accepted@start")
    drafts_delta    = _int("spec_drafts_delta")
    draft_toks_delta = _int("spec_draft_tokens_delta")
    accepted_delta  = _int("spec_accepted_delta")

    accept_rate_str = footer.get("spec_accept_rate", "n/a")
    if accept_rate_str == "n/a":
        accept_rate: float | None = None
    else:
        # value is like "82.51%"
        try:
            accept_rate = round(float(accept_rate_str.rstrip("%")), 4)
        except ValueError:
            accept_rate = None

    spec_decode = {
        "drafts_start":         drafts_start,
        "drafts_end":           drafts_start + drafts_delta,
        "drafts_delta":         drafts_delta,
        "draft_tokens_start":   draft_toks_start,
        "draft_tokens_end":     draft_toks_start + draft_toks_delta,
        "draft_tokens_delta":   draft_toks_delta,
        "accepted_start":       accepted_start,
        "accepted_end":         accepted_start + accepted_delta,
        "accepted_delta":       accepted_delta,
        "accept_rate":          accept_rate,
    }

    return {
        "bench_version": bench_version,
        "label": label,
        "header": header,
        "footer": footer,
        "prompts": prompts,
        "aggregate": aggregate,
        "spec_decode": spec_decode,
    }


# ── JSON builder ─────────────────────────────────────────────────────────────

def build_json(experiment: str, parsed: dict[str, Any]) -> dict[str, Any]:
    h = parsed["header"]
    f = parsed["footer"]
    return {
        "experiment": experiment,
        "status": "PASS",
        "bench_version": parsed["bench_version"],

        "run": {
            "label": parsed["label"],
            "timestamp_utc":     h.get("timestamp_utc"),
            "end_timestamp_utc": f.get("end_timestamp_utc"),
            "elapsed_seconds":   int(f.get("elapsed_seconds", "0") or "0"),
            "num_runs":     int(h.get("num_runs", "0").split()[0]),  # "5 (+2 warmup)"
            "warmup_runs":  2,  # bench convention; not separately in header
            "max_tokens":   int(h.get("max_tokens", "0")),
            "prompts_hash": h.get("prompts_hash"),
        },

        "environment": {
            "hostname":      h.get("hostname"),
            "gpu":           h.get("gpu"),
            "endpoint":      h.get("endpoint"),
            "queried_as":    h.get("queried_as"),
            "vllm_version":  h.get("vllm_version"),
            "image_ref":     h.get("image_ref"),
            "image_digest":  h.get("image_digest"),
        },

        "prompts":     parsed["prompts"],
        "aggregate":   parsed["aggregate"],
        "spec_decode": parsed["spec_decode"],
    }


def build_failed_json(experiment: str, fail_reason: str, raw_path: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "status": "FAILED",
        "fail_reason": fail_reason,
        "raw_file": raw_path,
    }


# ── README writer ────────────────────────────────────────────────────────────

def write_readme_pass(out: Path, experiment: str, data: dict[str, Any],
                      boot_meta: dict[str, Any] | None) -> None:
    """Write the auto-generated README for a passing experiment."""
    run = data["run"]
    env = data["environment"]
    agg = data["aggregate"]
    spec = data["spec_decode"]

    spec_cfg = None
    if boot_meta:
        spec_cfg = boot_meta.get("vllm", {}).get("speculative_config")

    spec_cfg_str = "none" if spec_cfg in (None, "None", "null") else str(spec_cfg)

    lines: list[str] = []
    a = lines.append

    a(f"# {experiment}\n")
    a(f"**Status**: {data['status']}  ")
    a(f"**Run**: {run['timestamp_utc']}  ")
    a(f"**Duration**: {run['elapsed_seconds']}s (bench only)\n")

    a("## Configuration\n")
    a(f"- **Image**: `{env['image_digest']}`")
    a(f"- **vLLM**: `{env['vllm_version']}`")
    if boot_meta:
        a(f"- **Model**: `{boot_meta.get('vllm', {}).get('model_repo', 'unknown')}`")
        v = boot_meta.get("vllm", {})
        a(f"- **Max model len**: {v.get('max_model_len', 'unknown')}")
        a(f"- **KV cache dtype**: {v.get('kv_cache_dtype', 'unknown')}")
        a(f"- **GPU memory utilization**: {v.get('gpu_memory_utilization', 'unknown')}")
        a(f"- **Max num seqs**: {v.get('max_num_seqs', 'unknown')}")
    a(f"- **Speculative decoding**: {spec_cfg_str}\n")

    a("## Results (median across runs)\n")
    a("| Prompt | TTFT (ms) | TPOT (ms) | Decode tok/s | Total tok/s | Tokens |")
    a("|---|---|---|---|---|---|")
    for name, p in data["prompts"].items():
        m = p["median"]
        a(f"| {name} | {m['ttft_ms']:.2f} | {m['tpot_ms']:.2f} | "
          f"{m['decode_tps']:.2f} | {m['total_tps']:.2f} | {m['tokens']} |")
    a("")

    a(f"**Aggregate decode throughput**: {agg['decode_tps_mean_of_medians']:.2f} tok/s "
      f"(mean of medians, range {agg['decode_tps_min_median']:.2f}–"
      f"{agg['decode_tps_max_median']:.2f})\n")

    a("## Speculative decoding metrics\n")
    if spec["drafts_delta"] == 0 and spec["accept_rate"] is None:
        a("Not applicable for this experiment (no speculative config).")
        a(f"- Drafts: {spec['drafts_delta']}")
        a(f"- Accepted: {spec['accepted_delta']}")
    else:
        a(f"- **Drafts produced**: {spec['drafts_delta']:,}")
        a(f"- **Draft tokens**: {spec['draft_tokens_delta']:,}")
        a(f"- **Accepted tokens**: {spec['accepted_delta']:,}")
        rate = spec["accept_rate"]
        a(f"- **Acceptance rate**: {rate:.2f}%" if rate is not None else "- **Acceptance rate**: n/a")
    a("")

    a("## Files\n")
    a("- `run_vllm.sh` — exact launch script (pinned image digest)")
    a("- `boot.log` — vLLM boot log captured after /health passed")
    a("- `boot_meta.json` — container provenance")
    a("- `nvidia_smi_pre.txt` — GPU state at experiment start")
    a("- `metrics_start.txt` / `metrics_end.txt` — `/metrics` snapshots (full)")
    a("- `benchmark_result.txt` — raw bench output")
    a("- `benchmark_result.json` — parsed bench result (this file's source data)")
    a("")
    a(f"_Auto-written by parse_bench.py from benchmark_result.txt and boot_meta.json._")

    out.write_text("\n".join(lines))


def write_readme_failed(out: Path, experiment: str, data: dict[str, Any],
                        boot_meta: dict[str, Any] | None) -> None:
    """Write the auto-generated README for a failed experiment."""
    lines: list[str] = []
    a = lines.append

    a(f"# {experiment}\n")
    a("**Status**: FAILED  ")
    a(f"**Reason**: {data.get('fail_reason', 'unknown')}\n")

    a("## What happened\n")
    a("This experiment did not complete successfully. The orchestrator preserved")
    a("all artifacts that exist on disk for diagnosis. See:")
    a("")
    a("- `boot.log` — vLLM boot log (may be partial or absent)")
    a("- `benchmark_result.txt` — bench output (may be partial or absent)")
    a("- `boot_meta.json` — container provenance (may be partial or absent)")
    a("- `metrics_start.txt` / `metrics_end.txt` — may exist depending on phase reached")
    a("")
    a("Compare with the top-level `run_all_experiments.log` to determine which phase failed.")

    if boot_meta:
        a("\n## Boot metadata (captured before failure)\n")
        a("```json")
        a(json.dumps(boot_meta, indent=2))
        a("```")

    a("")
    a(f"_Auto-written by parse_bench.py._")

    out.write_text("\n".join(lines))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-txt", type=Path, required=True,
                    help="path to benchmark_result.txt")
    ap.add_argument("--experiment", required=True,
                    help="experiment identifier (e.g. E01_no_mtp)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-readme", type=Path, required=True)
    ap.add_argument("--boot-meta", type=Path, default=None,
                    help="optional boot_meta.json to inline config details from")
    ap.add_argument("--status", default="PASS", choices=("PASS", "FAILED"),
                    help="experiment status; FAILED skips bench parsing")
    ap.add_argument("--fail-reason", default="",
                    help="human-readable failure reason; only used if --status FAILED")
    args = ap.parse_args()

    boot_meta = None
    if args.boot_meta and args.boot_meta.exists():
        try:
            boot_meta = json.loads(args.boot_meta.read_text())
        except Exception as e:
            print(f"warning: could not parse boot_meta.json: {e}", file=sys.stderr)

    if args.status == "FAILED":
        data = build_failed_json(
            args.experiment,
            args.fail_reason or "unspecified",
            str(args.bench_txt),
        )
        args.out_json.write_text(json.dumps(data, indent=2))
        write_readme_failed(args.out_readme, args.experiment, data, boot_meta)
        return 0

    try:
        parsed = parse_bench_txt(args.bench_txt)
    except Exception as e:
        # Strict parsing: if we can't parse the .txt cleanly, treat it as failed.
        print(f"parse error: {e}", file=sys.stderr)
        data = build_failed_json(
            args.experiment,
            f"parse error: {e}",
            str(args.bench_txt),
        )
        args.out_json.write_text(json.dumps(data, indent=2))
        write_readme_failed(args.out_readme, args.experiment, data, boot_meta)
        return 1

    data = build_json(args.experiment, parsed)
    args.out_json.write_text(json.dumps(data, indent=2))
    write_readme_pass(args.out_readme, args.experiment, data, boot_meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())