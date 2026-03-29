#!/usr/bin/env bash
# verify-examples.sh: Verify example builds and basic runtime checks for examples.
set -euo pipefail

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

stop_tokenizer() {
  container_tool="${CONTAINER_TOOL:-docker}"
  "${container_tool}" rm -f uds-tokenizer-example 2>/dev/null || true
}

# Ensure tokenizer container is always cleaned up, even on failures/timeouts.
trap 'stop_tokenizer' EXIT INT TERM

# 1. Test build
if ! make build-examples; then
  fail "make build-examples failed."
fi

echo "[OK] build-examples succeeded."

# 2. Test offline example
echo "[INFO] Running offline example..."
timeout 30s make run-example offline >offline.log 2>&1 &
pid=$!
found=0
for i in {1..30}; do
  if grep -q 'Events demo completed.' offline.log 2>/dev/null; then
    found=1
    break
  fi
  sleep 1
done
kill -INT $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
stop_tokenizer
if [ $found -eq 0 ]; then
  cat offline.log
  fail "offline example did not complete successfully."
fi
echo "[OK] offline example completed."

# 3. Test online example
echo "[INFO] Running online example..."
timeout 30s make run-example online >online.log 2>&1 &
pid=$!
found=0
for i in {1..30}; do
  if grep -q '8080' online.log 2>/dev/null; then
    found=1
    break
  fi
  sleep 1
done
kill -INT $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
stop_tokenizer
if [ $found -eq 0 ]; then
  cat online.log
  fail "online example did not listen on 8080."
fi
echo "[OK] online example is listening on 8080."

# 4. Test kv_cache_index example
echo "[INFO] Running kv_cache_index example..."
if ! make run-example kv_cache_index >kv_cache_index.log 2>&1; then
  cat kv_cache_index.log
  fail "kv_cache_index example did not complete successfully."
fi
echo "[OK] kv_cache_index example completed."
if ! grep -q 'Got pod.*"pod1"' kv_cache_index.log; then
  cat kv_cache_index.log
  fail "kv_cache_index.log does not contain expected pod1 score output."
fi

# TODO: Add more example verifications as needed.

echo "[SUCCESS] All example verifications passed."
