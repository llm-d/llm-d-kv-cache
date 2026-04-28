#!/usr/bin/env bash
# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generates a PEP 503 simple package index whose links point directly at
# GitHub Release asset download URLs, splitting CUDA variants across
# separate index paths based on the wheel's PEP 440 local-version segment.
#
# Layout:
#   <output-dir>/simple/<pkg>/index.html              wheels with NO local segment (default, e.g. cu12)
#   <output-dir>/simple/<local>/<pkg>/index.html      wheels stamped X.Y+<local>  (e.g. simple/cu130/llmd-fs-connector/)
#   <output-dir>/simple/index.html                    root listing of buckets
#
# Wheels are NOT copied into the index; pip downloads bytes straight from Releases.
#
# Usage: generate-pip-index.sh <repo> <output-dir>

set -euo pipefail

REPO="${1:?Usage: $0 <repo> <output-dir>}"
OUTPUT_DIR="${2:?Usage: $0 <repo> <output-dir>}"
SIMPLE_DIR="${OUTPUT_DIR}/simple"
mkdir -p "$SIMPLE_DIR"

mapfile -t entries < <(
  gh api --paginate "/repos/${REPO}/releases?per_page=100" \
    --jq '.[] | .assets[] | select(.name|endswith(".whl")) | "\(.name) \(.browser_download_url)"'
)

if [ ${#entries[@]} -eq 0 ]; then
    echo "ERROR: No .whl assets found across releases of ${REPO}" >&2
    exit 1
fi

# Split filename into (normalized_pkg, local_segment_or_empty).
# Wheel: {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# Version may include +<local> (PEP 440). We extract local from the version field.
parse_wheel() {
    local filename="$1"
    local stem="${filename%.whl}"
    local raw_name="${stem%%-[0-9]*}"
    local rest="${stem#${raw_name}-}"
    local version="${rest%%-*}"
    local local_seg=""
    if [[ "$version" == *"+"* ]]; then
        local_seg="${version#*+}"
    fi
    local pkg
    pkg="$(echo "$raw_name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g')"
    echo "${pkg}|${local_seg}"
}

declare -A BUCKET_PACKAGES
declare -a ROUTED

for line in "${entries[@]}"; do
    filename="${line%% *}"
    url="${line#* }"
    parsed="$(parse_wheel "$filename")"
    pkg="${parsed%|*}"
    local_seg="${parsed#*|}"
    bucket="${local_seg:-_default}"
    BUCKET_PACKAGES["${bucket}/${pkg}"]=1
    ROUTED+=("${bucket}|${pkg}|${filename}|${url}")
done

# Per-bucket per-package index files
for key in "${!BUCKET_PACKAGES[@]}"; do
    bucket="${key%%/*}"
    pkg="${key#*/}"
    if [ "$bucket" = "_default" ]; then
        pkg_dir="${SIMPLE_DIR}/${pkg}"
    else
        pkg_dir="${SIMPLE_DIR}/${bucket}/${pkg}"
    fi
    mkdir -p "$pkg_dir"
    {
        echo '<!DOCTYPE html>'
        echo "<html><head><meta charset=\"utf-8\"><title>Links for ${pkg}</title></head>"
        echo '<body>'
        echo "<h1>Links for ${pkg}</h1>"
        declare -A SEEN=()
        for entry in "${ROUTED[@]}"; do
            IFS='|' read -r e_bucket e_pkg e_filename e_url <<< "$entry"
            [ "$e_bucket" = "$bucket" ] && [ "$e_pkg" = "$pkg" ] || continue
            if [ -n "${SEEN[$e_filename]:-}" ]; then
                echo "WARN: duplicate filename ${e_filename} in bucket ${bucket} - keeping first" >&2
                continue
            fi
            SEEN["$e_filename"]=1
            echo "  <a href=\"${e_url}\">${e_filename}</a><br/>"
        done
        unset SEEN
        echo '</body></html>'
    } > "${pkg_dir}/index.html"
done

# Root index listing default packages and any CUDA buckets
{
    echo '<!DOCTYPE html>'
    echo '<html><head><meta charset="utf-8"><title>Simple Package Index</title></head>'
    echo '<body>'
    echo '<h1>Simple Package Index</h1>'
    declare -A SEEN_DIRS=()
    for key in $(printf '%s\n' "${!BUCKET_PACKAGES[@]}" | sort); do
        bucket="${key%%/*}"
        pkg="${key#*/}"
        if [ "$bucket" = "_default" ]; then
            href="${pkg}/"
        else
            href="${bucket}/${pkg}/"
        fi
        [ -z "${SEEN_DIRS[$href]:-}" ] || continue
        SEEN_DIRS["$href"]=1
        echo "  <a href=\"${href}\">${href}</a><br/>"
    done
    echo '</body></html>'
} > "${SIMPLE_DIR}/index.html"

echo "Generated PEP 503 index at ${SIMPLE_DIR}/"
echo "Buckets/packages: ${!BUCKET_PACKAGES[*]}"
echo "Wheel links: ${#ROUTED[@]}"
