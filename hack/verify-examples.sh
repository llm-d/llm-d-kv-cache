#!/usr/bin/env bash
# verify-examples.sh: Verify example builds and basic runtime checks for examples.
set -euo pipefail

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

# Wait up to 30s for a pattern to appear in a log file; returns 1 on timeout.
wait_for_log() {
  local log=$1 pattern=$2
  for i in {1..30}; do
    grep -q "$pattern" "$log" 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

# 1. Test build
make build-examples || fail "make build-examples failed."
echo "[OK] build-examples succeeded."

# Start the tokenizer once for all examples; ensure it is cleaned up on exit.
make start-tokenizer
trap 'make stop-tokenizer' EXIT

# 2. Test offline example
echo "[INFO] Running offline example..."
timeout 30s make run-example-only offline >offline.log 2>&1 &
pid=$!
wait_for_log offline.log 'Events demo completed.' || { cat offline.log; fail "offline example did not complete successfully."; }
kill -INT "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
echo "[OK] offline example completed."

# 3. Test online example
echo "[INFO] Running online example..."
timeout 30s make run-example-only online >online.log 2>&1 &
pid=$!
wait_for_log online.log '8080' || { cat online.log; fail "online example did not listen on 8080."; }
kill -INT "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
echo "[OK] online example is listening on 8080."

# 4. Test kv_cache_index example
echo "[INFO] Running kv_cache_index example..."
make run-example-only kv_cache_index >kv_cache_index.log 2>&1 || { cat kv_cache_index.log; fail "kv_cache_index example did not complete successfully."; }
grep -q 'Got pod.*"pod1"' kv_cache_index.log || { cat kv_cache_index.log; fail "kv_cache_index.log does not contain expected pod1 score output."; }
echo "[OK] kv_cache_index example completed."

# TODO: Add more example verifications as needed.

echo "[SUCCESS] All example verifications passed."
