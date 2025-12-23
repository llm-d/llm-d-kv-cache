#!/usr/bin/env bash
# verify-examples.sh: Verify example builds and basic runtime checks for examples.
set -euo pipefail

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

# 1. Test build
if ! make build-examples; then
  fail "make build-examples failed."
fi

echo "[OK] build-examples succeeded."

# 2. Test offline example
(
  set -m
  echo "[INFO] Running offline example..."
  timeout 30s make run-example offline >offline.log 2>&1 &
  pid=$!
  found=0
  for i in {1..30}; do
    if grep -q 'Events demo completed.' offline.log; then
      found=1
      break
    fi
    sleep 1
  done
  if [ $found -eq 1 ]; then
    echo "[OK] offline example completed."
    kill -INT $pid || true
    wait $pid || true
  else
    kill -INT $pid || true
    wait $pid || true
    cat offline.log
    fail "offline example did not complete successfully."
  fi
)

# 3. Test online example
(
  set -m
  echo "[INFO] Running online example..."
  timeout 30s make run-example online >online.log 2>&1 &
  pid=$!
  found=0
  for i in {1..30}; do
    if grep -q '8080' online.log; then
      found=1
      break
    fi
    sleep 1
  done
  if [ $found -eq 1 ]; then
    echo "[OK] online example is listening on 8080."
    kill -INT $pid || true
    wait $pid || true
  else
    # Try checking if port 8080 is open (Linux/macOS)
    if command -v lsof >/dev/null && lsof -i:8080 | grep LISTEN; then
      echo "[OK] online example is listening on 8080 (lsof check)."
      kill -INT $pid || true
      wait $pid || true
    else
      kill -INT $pid || true
      wait $pid || true
      cat online.log
      fail "online example did not listen on 8080."
    fi
  fi
)

# 4. Test kv_cache_index example
(
  set -m
  echo "[INFO] Running kv_cache_index example..."
  if make run-example kv_cache_index >kv_cache_index.log 2>&1; then
    echo "[OK] kv_cache_index example completed."
    # Validate kv_cache_index.log contains a single line that has both
    # the text 'Got pod' and the pods JSON containing pod1.
    found_line=0
    while IFS= read -r line; do
      if [[ "$line" == *"Got pod"* && "$line" == *'{"pods": {"pod1'* ]]; then
        found_line=1
        break
      fi
    done < <(grep 'Got pod' kv_cache_index.log || true)
    if [ $found_line -ne 1 ]; then
      cat kv_cache_index.log
      fail "kv_cache_index.log does not contain a line with 'Got pod' and '{\"pods\": {\"pod1\"'"
    fi
  else
    cat kv_cache_index.log
    fail "kv_cache_index example did not complete successfully."
  fi
)

# TODO: Add more example verifications as needed.

echo "[SUCCESS] All example verifications passed."
