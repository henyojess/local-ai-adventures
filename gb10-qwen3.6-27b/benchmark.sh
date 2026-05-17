#!/bin/bash
# benchmark.sh — vLLM throughput benchmark with provenance metadata
#
# Usage: ./benchmark.sh [options]
#
# Options:
#   --runs N            Runs per prompt (default: 5)
#   --host HOST         Server host (default: localhost)
#   --port PORT         Server port (default: 8001)
#   --model NAME        Model name to query (default: qwen3.6-35b-a3b)
#   --label LABEL       Run label, used in filename (default: unlabeled)
#   --image-digest STR  Override auto-detected RepoDigest. Pass a fully-qualified
#                       form (e.g. vllm/vllm-openai@sha256:...) for reproducibility.
#   --max-tokens N      Max output tokens per request (default: 500)
#   --results-dir DIR   Output directory (default: ./bench-results)
#   -h, --help          Show this help
#
# Changelog:
#   v0.3.3: Replace image_id (local content hash, not portable) with image_digest
#           (registry RepoDigest, e.g. vllm/vllm-openai@sha256:...). The latter is
#           the only identifier that lets a future reader re-pull the exact image.
#           Override flag --image-digest now populates the image_digest field
#           directly (was image_id in v0.3.2). Missing RepoDigest is now fatal:
#           refusing to write a bench result we cannot reproduce.
#   v0.3.2: Initial provenance header with image_ref + image_id.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
NUM_RUNS=5
HOST="localhost"
PORT=8001
MODEL="qwen3.6-27b"
LABEL="unlabeled"
IMAGE_DIGEST_OVERRIDE=""
MAX_TOKENS=500
RESULTS_DIR="./benchmak-results"
WARMUP_RUNS=2
SCRIPT_VERSION="0.3.3"

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

# ── argparse ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)         NUM_RUNS="$2"; shift 2 ;;
    --host)         HOST="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --label)        LABEL="$2"; shift 2 ;;
    --image-digest) IMAGE_DIGEST_OVERRIDE="$2"; shift 2 ;;
    --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
    --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

ENDPOINT="http://${HOST}:${PORT}"

# ── prompts ───────────────────────────────────────────────────────────────────
declare -A PROMPTS=(
  [short_prose]="Explain how neural networks work in 144-200 words."
  [code_gen]="Write a Python function that implements a binary search tree with insert, delete, and search methods. Include docstrings and type hints."
  [reasoning]="A train leaves Station A at 9:00 AM traveling east at 60 mph. Another train leaves Station B (300 miles east of A) at 9:30 AM traveling west at 80 mph. At what time and how far from Station A do they meet? Show your reasoning step by step."
  [structured]="Return a JSON object describing a fictional book with fields: title, author, year, genre, isbn, and a chapters array (5 entries) each with chapter_number, title, page_start, page_end."
  [long_context]="Summarize the key innovations of the transformer architecture, including: self-attention, multi-head attention, positional encoding, layer normalization, residual connections, and the encoder-decoder structure. For each, explain what problem it solves and why earlier approaches were insufficient. Be thorough."
)
PROMPTS_HASH=$(for k in "${!PROMPTS[@]}"; do echo "$k:${PROMPTS[$k]}"; done | sort | sha256sum | cut -c1-12)

# ── output redirect (bulletproof) ─────────────────────────────────────────────
RUN_TS=$(date -u +"%Y%m%d-%H%M%S")
mkdir -p "$RESULTS_DIR"
OUT_FILE="${RESULTS_DIR}/${LABEL}-${RUN_TS}.txt"
exec > >(tee "$OUT_FILE")
TEE_PID=$!
exec 2>&1
cleanup() {
  local exit_code=$?
  exec 1>&- 2>&-
  wait "$TEE_PID" 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

# ── helpers ───────────────────────────────────────────────────────────────────
percentile() {
  local pct=$1; shift
  local arr=("$@")
  local n=${#arr[@]}
  local idx
  idx=$(echo "($n - 1) * $pct / 100" | bc -l | awk '{printf "%d", $1+0.5}')
  echo "${arr[$idx]}"
}
median() { percentile 50 "$@"; }

scrape_metric() {
  curl -sf "$ENDPOINT/metrics" 2>/dev/null \
    | grep -E "^${1}\{" | head -1 | awk '{print $NF}' || echo "0"
}

resolve_image_digest() {
  # Returns two lines on stdout:
  #   1. image_ref     — string passed to `docker run` (tag or @sha256, "(override)" if flag used)
  #   2. image_digest  — registry RepoDigest (vllm/vllm-openai@sha256:...)
  #
  # Hard-fails if no RepoDigest can be resolved: a bench result we cannot
  # reproduce is worse than no bench result.
  #
  # Override path: caller explicitly passed --image-digest.
  if [[ -n "$IMAGE_DIGEST_OVERRIDE" ]]; then
    printf '%s\n%s\n' "(override)" "$IMAGE_DIGEST_OVERRIDE"
    return
  fi

  if ! command -v docker >/dev/null 2>&1; then
    die "docker not found on PATH; cannot resolve container for port ${PORT}. Use --image-digest <vllm/vllm-openai@sha256:...> to bypass auto-detection."
  fi

  local container_id=""

  # Strategy 1: published port (bridge networking)
  container_id=$(docker ps --quiet --filter "publish=${PORT}" 2>/dev/null | head -1)

  # Strategy 2: host networking — match on argv containing --port PORT
  if [[ -z "$container_id" ]]; then
    local host_net_ids
    host_net_ids=$(docker ps --quiet --filter "network=host" 2>/dev/null)
    for cid in $host_net_ids; do
      local args
      args=$(docker inspect --format '{{join .Args " "}}' "$cid" 2>/dev/null || echo "")
      if [[ "$args" == *"--port $PORT"* || "$args" == *"--port=$PORT"* ]]; then
        container_id="$cid"
        break
      fi
    done
  fi

  if [[ -z "$container_id" ]]; then
    die "No running container found serving port ${PORT}. Checked: (1) published-port mapping, (2) host-network containers with --port ${PORT} in args. Is the server running?"
  fi

  local image_local_id image_ref image_repo_digest
  # image_local_id is used internally only — to look up RepoDigests on the image
  # object. Not emitted in the bench header (v0.3.3: dropped as non-portable).
  image_local_id=$(docker inspect --format '{{.Image}}' "$container_id" 2>/dev/null || echo "")
  image_ref=$(docker inspect --format '{{.Config.Image}}' "$container_id" 2>/dev/null || echo "")
  if [[ -z "$image_local_id" ]]; then
    die "Found container ${container_id:0:12} for port ${PORT} but could not read its image. Docker permissions issue?"
  fi

  # RepoDigest: the registry-pullable identifier. Empty for locally-built images
  # or images pulled before digest tracking. First entry wins; multi-registry
  # images are rare for our use case.
  image_repo_digest=$(docker inspect --format \
    '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' \
    "$image_local_id" 2>/dev/null || echo "")

  if [[ -z "$image_repo_digest" ]]; then
    die "Container ${container_id:0:12} (image_ref=${image_ref:-unknown}, local_id=${image_local_id}) has no RepoDigest. The image was likely built locally or imported without registry metadata. Refusing to record a non-reproducible bench result. Pass --image-digest <vllm/vllm-openai@sha256:...> if you can supply a known-good digest manually."
  fi

  printf '%s\n%s\n' \
    "${image_ref:-unknown}" \
    "$image_repo_digest"
}

run_one() {
  local prompt="$1"
  ENDPOINT="$ENDPOINT" MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" python3 - "$prompt" <<'PYEOF'
import json, os, sys, time, urllib.request
ENDPOINT    = os.environ["ENDPOINT"]
MODEL       = os.environ["MODEL"]
MAX_TOKENS  = int(os.environ["MAX_TOKENS"])
prompt = sys.argv[1]
payload = json.dumps({
    "model": MODEL,
    "prompt": prompt,
    "max_tokens": MAX_TOKENS,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()
req = urllib.request.Request(
    f"{ENDPOINT}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
t_start = time.perf_counter()
t_first_token = None
last_t = None
intervals = []
completion_tokens = 0
with urllib.request.urlopen(req) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        text = choices[0].get("text", "")
        if not text:
            continue
        now = time.perf_counter()
        if t_first_token is None:
            t_first_token = now
            last_t = now
        else:
            intervals.append(now - last_t)
            last_t = now
t_end = time.perf_counter()
if t_first_token is None:
    print("0 0 0 0 0")
    sys.exit(0)
ttft_ms = (t_first_token - t_start) * 1000
total_ms = (t_end - t_start) * 1000
decode_ms = (t_end - t_first_token) * 1000
tpot_ms = (sum(intervals) / len(intervals) * 1000) if intervals else 0
decode_tps = ((completion_tokens - 1) * 1000 / decode_ms) if decode_ms > 0 and completion_tokens > 1 else 0
total_tps = completion_tokens * 1000 / total_ms if total_ms > 0 else 0
print(f"{ttft_ms:.2f} {tpot_ms:.2f} {decode_tps:.2f} {total_tps:.2f} {completion_tokens}")
PYEOF
}

# ── metadata header ───────────────────────────────────────────────────────────
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_EPOCH=$(date +%s)
HOSTNAME=$(hostname)
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
VLLM_VERSION=$(curl -sf "$ENDPOINT/version" 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('version','unknown'))" 2>/dev/null || echo "unknown")
{ read -r IMAGE_REF; read -r IMAGE_DIGEST; } < <(resolve_image_digest)
SPEC_DRAFTS_START=$(scrape_metric "vllm:spec_decode_num_drafts_total")
SPEC_DRAFT_TOKS_START=$(scrape_metric "vllm:spec_decode_num_draft_tokens_total")
SPEC_ACCEPTED_START=$(scrape_metric "vllm:spec_decode_num_accepted_tokens_total")

cat <<META
============================================================
benchmark.sh v${SCRIPT_VERSION} label=${LABEL} output=${OUT_FILE}
============================================================
timestamp_utc:           ${START_ISO}
hostname:                ${HOSTNAME}
gpu:                     ${GPU_INFO}
endpoint:                ${ENDPOINT}
queried_as:              ${MODEL}
vllm_version:            ${VLLM_VERSION}
image_ref:               ${IMAGE_REF}
image_digest:            ${IMAGE_DIGEST}
num_runs:                ${NUM_RUNS} (+${WARMUP_RUNS} warmup)
max_tokens:              ${MAX_TOKENS}
prompts_hash:            ${PROMPTS_HASH}
spec_drafts@start:       ${SPEC_DRAFTS_START}
spec_draft_tokens@start: ${SPEC_DRAFT_TOKS_START}
spec_accepted@start:     ${SPEC_ACCEPTED_START}
============================================================
META

# ── warmup + runs ─────────────────────────────────────────────────────────────
echo "Warming up..."
for i in $(seq 1 $WARMUP_RUNS); do
  run_one "Hello, how are you?" > /dev/null
done

printf "%-15s %10s %10s %10s %10s %8s\n" "prompt" "ttft_ms" "tpot_ms" "decode_t/s" "total_t/s" "tokens"
printf "%-15s %10s %10s %10s %10s %8s\n" "-------" "-------" "-------" "----------" "---------" "------"

for name in "${!PROMPTS[@]}"; do
  ttfts=(); tpots=(); decodes=(); totals=(); toks=()
  for i in $(seq 1 $NUM_RUNS); do
    read -r ttft tpot decode total nt <<< "$(run_one "${PROMPTS[$name]}")"
    ttfts+=("$ttft"); tpots+=("$tpot"); decodes+=("$decode"); totals+=("$total"); toks+=("$nt")
    printf "%-15s %10s %10s %10s %10s %8s (run %d)\n" "$name" "$ttft" "$tpot" "$decode" "$total" "$nt" "$i"
  done
  IFS=$'\n' s_ttfts=($(sort -n <<<"${ttfts[*]}"))
  IFS=$'\n' s_tpots=($(sort -n <<<"${tpots[*]}"))
  IFS=$'\n' s_decodes=($(sort -n <<<"${decodes[*]}"))
  IFS=$'\n' s_totals=($(sort -n <<<"${totals[*]}"))
  unset IFS
  printf "%-15s %10s %10s %10s %10s %8s ← median\n" "$name" \
    "$(median "${s_ttfts[@]}")" "$(median "${s_tpots[@]}")" \
    "$(median "${s_decodes[@]}")" "$(median "${s_totals[@]}")" "${toks[0]}"
  printf "%-15s %10s %10s %10s %10s %8s ← p95\n\n" "$name" \
    "$(percentile 95 "${s_ttfts[@]}")" "$(percentile 95 "${s_tpots[@]}")" \
    "$(percentile 95 "${s_decodes[@]}")" "$(percentile 95 "${s_totals[@]}")" "${toks[0]}"
done

# ── footer ────────────────────────────────────────────────────────────────────
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))

SPEC_DRAFTS_END=$(scrape_metric "vllm:spec_decode_num_drafts_total")
SPEC_DRAFT_TOKS_END=$(scrape_metric "vllm:spec_decode_num_draft_tokens_total")
SPEC_ACCEPTED_END=$(scrape_metric "vllm:spec_decode_num_accepted_tokens_total")

delta() { python3 -c "print(int(float('$1' or 0)) - int(float('$2' or 0)))"; }
D_DRAFTS=$(delta "$SPEC_DRAFTS_END" "$SPEC_DRAFTS_START")
D_DRAFT_TOKS=$(delta "$SPEC_DRAFT_TOKS_END" "$SPEC_DRAFT_TOKS_START")
D_ACCEPTED=$(delta "$SPEC_ACCEPTED_END" "$SPEC_ACCEPTED_START")

ACCEPT_RATE=$(python3 -c "d=$D_DRAFT_TOKS; a=$D_ACCEPTED; print(f'{100*a/d:.2f}%' if d else 'n/a')")

cat <<FOOTER
============================================================
end_timestamp_utc:       ${END_ISO}
elapsed_seconds:         ${ELAPSED}
spec_drafts_delta:       ${D_DRAFTS}
spec_draft_tokens_delta: ${D_DRAFT_TOKS}
spec_accepted_delta:     ${D_ACCEPTED}
spec_accept_rate:        ${ACCEPT_RATE}
============================================================
FOOTER