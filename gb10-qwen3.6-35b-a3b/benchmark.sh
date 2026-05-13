#!/usr/bin/env bash
# Benchmark script for unsloth/Qwen3.6-35B-A3B-NVFP4 on GB10.
# Reproduces the numbers in README.md.
#
# Usage:
#   ./benchmark.sh                  # full suite
#   ./benchmark.sh single           # just single-stream (n=3)
#   ./benchmark.sh parallel         # just concurrency-2

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-qwen3.6-35b-a3b}"
PROMPT="Write a 200 word technical explanation of why memory bandwidth matters more than raw FLOPs for LLM inference on edge accelerators."

# Check server is up
if ! curl -sf "${ENDPOINT%/chat/completions}/models" > /dev/null; then
  echo "ERROR: vLLM server not reachable at $ENDPOINT"
  echo "Is the container running? Try: docker logs vllm-qwen36-35b"
  exit 1
fi

# Tool check
command -v jq >/dev/null || { echo "ERROR: jq is required (apt install jq)"; exit 1; }

request_payload() {
  local content="$1"
  jq -nc \
    --arg model "$MODEL" \
    --arg content "$content" \
    '{
      model: $model,
      messages: [{role:"user", content:$content}],
      max_tokens: 300,
      temperature: 0,
      chat_template_kwargs: {enable_thinking: false}
    }'
}

run_single() {
  local payload tokens
  payload=$(request_payload "$PROMPT")

  local start end elapsed
  start=$(date +%s.%N)
  tokens=$(curl -sf "$ENDPOINT" -H 'Content-Type: application/json' -d "$payload" \
    | jq -r '.usage.completion_tokens')
  end=$(date +%s.%N)
  elapsed=$(echo "$end - $start" | bc -l)

  local tps
  tps=$(echo "scale=2; $tokens / $elapsed" | bc -l)
  printf "  tokens=%s  wall_clock=%.3fs  tok/s=%s\n" "$tokens" "$elapsed" "$tps"
  echo "$tokens $elapsed $tps" >> /tmp/bench-single.tsv
}

run_parallel() {
  local payload_a payload_b
  payload_a=$(request_payload "$PROMPT (focus on case A)")
  payload_b=$(request_payload "$PROMPT (focus on case B)")

  local start
  start=$(date +%s.%N)

  curl -sf "$ENDPOINT" -H 'Content-Type: application/json' -d "$payload_a" \
    | jq -r '.usage.completion_tokens' > /tmp/bench-r1 &
  local pid1=$!

  curl -sf "$ENDPOINT" -H 'Content-Type: application/json' -d "$payload_b" \
    | jq -r '.usage.completion_tokens' > /tmp/bench-r2 &
  local pid2=$!

  wait $pid1 $pid2
  local end=$(date +%s.%N)
  local elapsed=$(echo "$end - $start" | bc -l)

  local r1 r2 total tps
  r1=$(cat /tmp/bench-r1)
  r2=$(cat /tmp/bench-r2)
  total=$((r1 + r2))
  tps=$(echo "scale=2; $total / $elapsed" | bc -l)

  printf "  R1=%s tokens  R2=%s tokens  wall_clock=%.3fs  aggregate_tok/s=%s\n" \
    "$r1" "$r2" "$elapsed" "$tps"
}

MODE="${1:-all}"

if [[ "$MODE" == "single" || "$MODE" == "all" ]]; then
  echo "=== Single-stream (n=3) ==="
  rm -f /tmp/bench-single.tsv
  for i in 1 2 3; do
    echo "Run $i:"
    run_single
  done
  echo
  echo "Summary:"
  awk '{sum+=$3; n++} END {printf "  mean tok/s = %.2f (n=%d)\n", sum/n, n}' /tmp/bench-single.tsv
  echo
fi

if [[ "$MODE" == "parallel" || "$MODE" == "all" ]]; then
  echo "=== Concurrency 2 ==="
  run_parallel
  echo
fi