#!/bin/bash
# run_all_experiments.sh — orchestrate the MTP comparison matrix.
#
# Usage: ./run_all_experiments.sh [EXPERIMENT ...]
#
# With no args, runs the full hardcoded matrix (currently: E01 only).
# With args, runs just the named experiments. Useful for re-running one
# after a failure, e.g. ./run_all_experiments.sh E02_mtp1
#
# Design principles:
#   - The orchestrator orchestrates; it does not judge.
#     Failures captured here are operational only (timeout, nonzero exit,
#     malformed parse). Whether MTP "actually works" is for a human to
#     decide from the artifacts.
#   - Crash isolation: a failed experiment does not stop the matrix.
#   - Self-contained per-experiment folders: every artifact for E0N lives
#     under E0N/.
#   - Strict artifact capture: every file the brief lists gets produced,
#     even on failure (skeletal where unavoidable).

set -uo pipefail
# Note: NOT using `set -e`. The orchestrator handles per-phase errors
# explicitly so a single phase failure marks the experiment FAILED
# without killing the whole script.

# ── matrix & config ──────────────────────────────────────────────────────────
# Hardcoded matrix. To add E02-E04 later, just extend this array.
DEFAULT_MATRIX=(
  "E01_no_mtp"
)

# Pinned image digest the experiments are expected to run with.
# For now this only feeds the bench's --image-digest flag as a fallback
# (the bench prefers auto-detection from the running container).
PINNED_DIGEST="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"

MODEL_NAME="qwen3.6-27b"            # what the bench queries (matches --served-model-name)
PORT="8001"
ENDPOINT="http://localhost:${PORT}"
CONTAINER_NAME="vllm-8001"

BENCH_RUNS=5
BENCH_MAX_TOKENS=500

HEALTH_TIMEOUT_SECONDS=420          # 7 min: cold-boot E01 was 318s, leaves headroom
HEALTH_POLL_INTERVAL=5
INTER_EXPERIMENT_SLEEP=60

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${PROJECT_DIR}/run_all_experiments.log"
PARSE_SCRIPT="${PROJECT_DIR}/parse_bench.py"
BENCH_SCRIPT="${PROJECT_DIR}/benchmark.sh"

# ── logging ──────────────────────────────────────────────────────────────────
# Dual output: every log line goes to stderr (visible during run) and the
# log file (audit trail). Format: [iso_ts] [exp] [phase] message
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log_phase() {
  local exp="$1" phase="$2" msg="$3"
  local line
  line="[$(ts)] [${exp}] [${phase}] ${msg}"
  echo "$line" >&2
  echo "$line" >> "$LOG_FILE"
}

# ── interrupt handling ──────────────────────────────────────────────────────
INTERRUPTED=0
on_interrupt() {
  INTERRUPTED=1
  log_phase "ORCHESTRATOR" "interrupted" "caught signal, tearing down ${CONTAINER_NAME}"
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  exit 130
}
trap on_interrupt INT TERM

# ── helpers ──────────────────────────────────────────────────────────────────

# Poll /health until 200 or timeout.
# Returns 0 if healthy, 1 if timeout.
poll_health() {
  local exp="$1"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
  log_phase "$exp" "health_poll" "waiting for /health (timeout=${HEALTH_TIMEOUT_SECONDS}s, interval=${HEALTH_POLL_INTERVAL}s)"
  while (( $(date +%s) < deadline )); do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${ENDPOINT}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
      local elapsed=$(( HEALTH_TIMEOUT_SECONDS - (deadline - $(date +%s)) ))
      log_phase "$exp" "health_poll" "/health returned 200 after ~${elapsed}s"
      return 0
    fi
    sleep "$HEALTH_POLL_INTERVAL"
  done
  log_phase "$exp" "health_poll" "TIMEOUT after ${HEALTH_TIMEOUT_SECONDS}s waiting for /health"
  return 1
}

# Send a tiny test completion to verify the full request path works.
# Returns 0 on HTTP 200, 1 otherwise.
test_completion() {
  local exp="$1"
  log_phase "$exp" "test_completion" "sending 1-token probe"
  local body
  body=$(printf '{"model":"%s","prompt":"Hi","max_tokens":1,"temperature":0}' "$MODEL_NAME")
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
    -H "Content-Type: application/json" \
    -d "$body" \
    "${ENDPOINT}/v1/completions" 2>/dev/null || echo "000")
  if [[ "$code" == "200" ]]; then
    log_phase "$exp" "test_completion" "probe ok"
    return 0
  fi
  log_phase "$exp" "test_completion" "probe FAILED (http=${code})"
  return 1
}

# Synthesize boot_meta.json by querying the running container and vLLM endpoints.
synthesize_boot_meta() {
  local exp="$1" out="$2"
  log_phase "$exp" "boot_meta" "synthesizing from docker inspect + vLLM endpoints"

  # Capture inputs, fall back to empty strings on failure.
  local container_id container_started image_ref image_local_id image_digest image_created
  local vllm_version served_models cmd_args
  container_id=$(docker inspect --format='{{.Id}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
  container_started=$(docker inspect --format='{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
  image_ref=$(docker inspect --format='{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
  image_local_id=$(docker inspect --format='{{.Image}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
  image_digest=$(docker inspect --format='{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$image_local_id" 2>/dev/null || echo "")
  image_created=$(docker inspect --format='{{.Created}}' "$image_local_id" 2>/dev/null || echo "")
  vllm_version=$(curl -sf "${ENDPOINT}/version" 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('version','unknown'))" 2>/dev/null || echo "unknown")
  served_models=$(curl -sf "${ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys; print(','.join(m['id'] for m in json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "")
  cmd_args=$(docker inspect --format='{{json .Args}}' "$CONTAINER_NAME" 2>/dev/null || echo "[]")

  local hostname gpu_name driver_version
  hostname=$(hostname 2>/dev/null || echo "")
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
  driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")

  local captured_at
  captured_at=$(ts)

  # Hand off to python for clean JSON construction (escaping, args parsing).
  CONTAINER_ID="$container_id" \
  CONTAINER_STARTED="$container_started" \
  IMAGE_REF="$image_ref" \
  IMAGE_LOCAL_ID="$image_local_id" \
  IMAGE_DIGEST="$image_digest" \
  IMAGE_CREATED="$image_created" \
  VLLM_VERSION="$vllm_version" \
  SERVED_MODELS="$served_models" \
  CMD_ARGS_JSON="$cmd_args" \
  HOSTNAME_VAL="$hostname" \
  GPU_NAME="$gpu_name" \
  DRIVER_VERSION="$driver_version" \
  CAPTURED_AT="$captured_at" \
  EXPERIMENT="$exp" \
  python3 - <<'PYEOF' > "$out"
import json, os
from datetime import datetime, timezone

args = json.loads(os.environ.get("CMD_ARGS_JSON", "[]"))

# args is the list passed to the vLLM entrypoint after the image.
# First positional element (if present) is the model repo; the rest are flags.
model_repo = args[0] if args and not args[0].startswith("--") else "unknown"

def flag_value(name):
    """Return the value of --name from args, or None if absent."""
    if name in args:
        i = args.index(name)
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            return args[i + 1]
    return None

def flag_value_multi(name):
    """For flags like --served-model-name that take multiple positional values."""
    if name not in args:
        return None
    i = args.index(name)
    vals = []
    j = i + 1
    while j < len(args) and not args[j].startswith("--"):
        vals.append(args[j])
        j += 1
    return vals

def boot_duration_seconds(started_iso, captured_iso):
    try:
        a = datetime.fromisoformat(started_iso.replace("Z", "+00:00"))
        b = datetime.fromisoformat(captured_iso.replace("Z", "+00:00"))
        return int((b - a).total_seconds())
    except Exception:
        return None

# Parse --speculative-config if present (it's a JSON string we passed)
spec_cfg = flag_value("--speculative-config")
if spec_cfg:
    try:
        spec_cfg = json.loads(spec_cfg)
    except Exception:
        pass  # leave as string if not parseable

started = os.environ["CONTAINER_STARTED"]
captured = os.environ["CAPTURED_AT"]

doc = {
    "experiment": os.environ["EXPERIMENT"],
    "captured_at_utc": captured,
    "container": {
        "name": "vllm-8001",
        "id": os.environ["CONTAINER_ID"],
        "id_short": os.environ["CONTAINER_ID"][:12],
        "started_at_utc": started,
        "boot_duration_seconds": boot_duration_seconds(started, captured),
    },
    "image": {
        "ref": os.environ["IMAGE_REF"],
        "digest": os.environ["IMAGE_DIGEST"] or None,
        "local_id": os.environ["IMAGE_LOCAL_ID"],
        "created_at_utc": os.environ["IMAGE_CREATED"],
    },
    "vllm": {
        "version": os.environ["VLLM_VERSION"],
        "served_model_names": (os.environ["SERVED_MODELS"].split(",")
                                if os.environ["SERVED_MODELS"] else []),
        "model_repo": model_repo,
        "max_model_len": flag_value("--max-model-len"),
        "gpu_memory_utilization": flag_value("--gpu-memory-utilization"),
        "kv_cache_dtype": flag_value("--kv-cache-dtype"),
        "max_num_seqs": flag_value("--max-num-seqs"),
        "max_num_batched_tokens": flag_value("--max-num-batched-tokens"),
        "speculative_config": spec_cfg,
        "generation_config": flag_value("--generation-config"),
    },
    "host": {
        "hostname": os.environ["HOSTNAME_VAL"],
        "gpu_name": os.environ["GPU_NAME"],
        "driver_version": os.environ["DRIVER_VERSION"],
        "gpu_memory_free_mib": None,  # GB10 unified memory; nvidia-smi returns [N/A]
    },
}
print(json.dumps(doc, indent=2))
PYEOF
}

# Run a single experiment end-to-end. Always returns 0 (per crash-isolation
# decision; we don't want a failing experiment to abort the matrix).
# Writes status to global EXP_STATUSES[$exp].
declare -A EXP_STATUSES

run_one_experiment() {
  local exp="$1"
  local exp_dir="${PROJECT_DIR}/${exp}"
  local run_vllm_script="${exp_dir}/run_vllm.sh"

  log_phase "$exp" "start" "begin experiment"

  # Pre-flight: the experiment folder and its run_vllm.sh must exist.
  if [[ ! -x "$run_vllm_script" ]]; then
    log_phase "$exp" "preflight" "FAIL: ${run_vllm_script} not found or not executable"
    EXP_STATUSES[$exp]="FAILED_PREFLIGHT"
    return 0
  fi

  # ── phase 1: nvidia_smi_pre.txt ──
  log_phase "$exp" "nvidia_smi_pre" "capturing GPU state"
  if ! nvidia-smi > "${exp_dir}/nvidia_smi_pre.txt" 2>&1; then
    log_phase "$exp" "nvidia_smi_pre" "FAIL: nvidia-smi returned nonzero"
    EXP_STATUSES[$exp]="FAILED_GPU_CHECK"
    return 0
  fi

  # Cheap sanity: ensure at least one GPU is listed.
  if ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
    log_phase "$exp" "nvidia_smi_pre" "FAIL: no GPU detected by nvidia-smi"
    EXP_STATUSES[$exp]="FAILED_NO_GPU"
    return 0
  fi

  # ── phase 2: launch ──
  log_phase "$exp" "launch" "invoking ${run_vllm_script}"
  if ! "$run_vllm_script" >/dev/null 2>&1; then
    log_phase "$exp" "launch" "FAIL: run_vllm.sh returned nonzero"
    EXP_STATUSES[$exp]="FAILED_LAUNCH"
    return 0
  fi

  # ── phase 3: wait for /health ──
  if ! poll_health "$exp"; then
    EXP_STATUSES[$exp]="FAILED_HEALTH_TIMEOUT"
    # Capture boot log anyway for diagnosis.
    docker logs "$CONTAINER_NAME" > "${exp_dir}/boot.log" 2>&1 || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    return 0
  fi

  # ── phase 4: test completion ──
  if ! test_completion "$exp"; then
    EXP_STATUSES[$exp]="FAILED_TEST_COMPLETION"
    docker logs "$CONTAINER_NAME" > "${exp_dir}/boot.log" 2>&1 || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    return 0
  fi

  # ── phase 5: capture boot.log ──
  log_phase "$exp" "boot_log" "capturing docker logs ${CONTAINER_NAME}"
  docker logs "$CONTAINER_NAME" > "${exp_dir}/boot.log" 2>&1 || \
    log_phase "$exp" "boot_log" "warning: docker logs returned nonzero, file may be partial"

  # ── phase 6: synthesize boot_meta.json ──
  synthesize_boot_meta "$exp" "${exp_dir}/boot_meta.json" || \
    log_phase "$exp" "boot_meta" "warning: synthesis returned nonzero"

  # ── phase 7: metrics_start.txt ──
  log_phase "$exp" "metrics_start" "snapshotting /metrics"
  curl -sf "${ENDPOINT}/metrics" > "${exp_dir}/metrics_start.txt" 2>/dev/null || \
    log_phase "$exp" "metrics_start" "warning: scrape failed"

  # ── phase 8: bench ──
  log_phase "$exp" "bench" "running benchmark.sh (runs=${BENCH_RUNS}, max_tokens=${BENCH_MAX_TOKENS})"
  local bench_exit=0
  "$BENCH_SCRIPT" \
    --output-file "${exp_dir}/benchmark_result.txt" \
    --label "$exp" \
    --runs "$BENCH_RUNS" \
    --max-tokens "$BENCH_MAX_TOKENS" \
    --model "$MODEL_NAME" \
    --host localhost \
    --port "$PORT" \
    > /dev/null 2>&1 || bench_exit=$?

  # ── phase 9: metrics_end.txt (always, regardless of bench exit) ──
  log_phase "$exp" "metrics_end" "snapshotting /metrics"
  curl -sf "${ENDPOINT}/metrics" > "${exp_dir}/metrics_end.txt" 2>/dev/null || \
    log_phase "$exp" "metrics_end" "warning: scrape failed"

  # ── phase 10/11: parse + write README ──
  if (( bench_exit == 0 )); then
    log_phase "$exp" "parse" "parsing bench output → JSON + README"
    if python3 "$PARSE_SCRIPT" \
        --bench-txt "${exp_dir}/benchmark_result.txt" \
        --experiment "$exp" \
        --boot-meta "${exp_dir}/boot_meta.json" \
        --out-json "${exp_dir}/benchmark_result.json" \
        --out-readme "${exp_dir}/README.md"; then
      EXP_STATUSES[$exp]="PASS"
    else
      log_phase "$exp" "parse" "FAIL: parser returned nonzero (bench .txt may be malformed)"
      EXP_STATUSES[$exp]="FAILED_PARSE"
    fi
  else
    log_phase "$exp" "bench" "FAIL: benchmark.sh exited ${bench_exit}"
    # Still write a FAILED skeleton.
    python3 "$PARSE_SCRIPT" \
      --bench-txt "${exp_dir}/benchmark_result.txt" \
      --experiment "$exp" \
      --boot-meta "${exp_dir}/boot_meta.json" \
      --out-json "${exp_dir}/benchmark_result.json" \
      --out-readme "${exp_dir}/README.md" \
      --status FAILED \
      --fail-reason "benchmark.sh exited ${bench_exit}" || true
    EXP_STATUSES[$exp]="FAILED_BENCH"
  fi

  # ── phase 12: teardown ──
  log_phase "$exp" "teardown" "removing container ${CONTAINER_NAME}"
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || \
    log_phase "$exp" "teardown" "warning: docker rm returned nonzero"

  log_phase "$exp" "done" "status=${EXP_STATUSES[$exp]}"
  return 0
}

# ── main ─────────────────────────────────────────────────────────────────────

# Truncate the log file on each invocation.
: > "$LOG_FILE"

# Select experiments to run.
if (( $# > 0 )); then
  MATRIX=("$@")
else
  MATRIX=("${DEFAULT_MATRIX[@]}")
fi

log_phase "ORCHESTRATOR" "start" "matrix=$(IFS=, ; echo "${MATRIX[*]}") project=${PROJECT_DIR}"

# Pre-flight: required tools.
for cmd in docker curl python3 nvidia-smi jq sed; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    if [[ "$cmd" == "jq" || "$cmd" == "sed" ]]; then
      continue  # not strictly required
    fi
    log_phase "ORCHESTRATOR" "preflight" "FAIL: required command '$cmd' not found"
    exit 2
  fi
done

# Pre-flight: bench script and parser exist.
for f in "$BENCH_SCRIPT" "$PARSE_SCRIPT"; do
  if [[ ! -e "$f" ]]; then
    log_phase "ORCHESTRATOR" "preflight" "FAIL: required file ${f} missing"
    exit 2
  fi
done

# Run each experiment, with cooldown between (but not after the last).
for i in "${!MATRIX[@]}"; do
  exp="${MATRIX[$i]}"
  run_one_experiment "$exp"

  if (( INTERRUPTED == 1 )); then
    log_phase "ORCHESTRATOR" "interrupted" "stopping matrix at ${exp}"
    break
  fi

  if (( i < ${#MATRIX[@]} - 1 )); then
    log_phase "ORCHESTRATOR" "cooldown" "sleeping ${INTER_EXPERIMENT_SLEEP}s before next experiment"
    sleep "$INTER_EXPERIMENT_SLEEP"
  fi
done

# ── summary ─────────────────────────────────────────────────────────────────
log_phase "ORCHESTRATOR" "summary" "matrix complete"
overall_status=0
for exp in "${MATRIX[@]}"; do
  status="${EXP_STATUSES[$exp]:-NOT_RUN}"
  log_phase "ORCHESTRATOR" "summary" "${exp}: ${status}"
  [[ "$status" != "PASS" ]] && overall_status=1
done

exit "$overall_status"