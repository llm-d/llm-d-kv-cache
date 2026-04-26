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

# Generates a PEP 503 simple package index from .whl files.
#
# Usage: generate-pip-index.sh <wheels-dir> <output-dir>
#
# The output directory will contain:
#   simple/index.html                  - root index listing all packages
#   simple/<package-name>/index.html   - per-package index with links to wheels
#   simple/<package-name>/<file>.whl   - the wheel files themselves

set -euo pipefail

WHEELS_DIR="${1:?Usage: $0 <wheels-dir> <output-dir>}"
OUTPUT_DIR="${2:?Usage: $0 <wheels-dir> <output-dir>}"

SIMPLE_DIR="${OUTPUT_DIR}/simple"

# Collect all unique normalized package names from wheel filenames.
# Wheel filename format: {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# PEP 503 normalizes names to lowercase with hyphens replacing underscores/dots.
declare -A PACKAGES

for whl in "${WHEELS_DIR}"/*.whl; do
    [ -f "$whl" ] || continue
    filename="$(basename "$whl")"
    # Extract the distribution name (everything before the first '-' that precedes a version)
    raw_name="${filename%%-[0-9]*}"
    # PEP 503 normalization: lowercase, replace [_.-]+ with single hyphen
    normalized="$(echo "$raw_name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g')"
    PACKAGES["$normalized"]=1
done

if [ ${#PACKAGES[@]} -eq 0 ]; then
    echo "ERROR: No .whl files found in ${WHEELS_DIR}" >&2
    exit 1
fi

# --- Root index ---
mkdir -p "$SIMPLE_DIR"
{
    echo '<!DOCTYPE html>'
    echo '<html><head><meta charset="utf-8"><title>Simple Package Index</title></head>'
    echo '<body>'
    echo '<h1>Simple Package Index</h1>'
    for pkg in $(echo "${!PACKAGES[@]}" | tr ' ' '\n' | sort); do
        echo "  <a href=\"${pkg}/\">${pkg}</a><br/>"
    done
    echo '</body></html>'
} > "${SIMPLE_DIR}/index.html"

# --- Per-package indexes ---
for pkg in "${!PACKAGES[@]}"; do
    pkg_dir="${SIMPLE_DIR}/${pkg}"
    mkdir -p "$pkg_dir"

    {
        echo '<!DOCTYPE html>'
        echo "<html><head><meta charset=\"utf-8\"><title>Links for ${pkg}</title></head>"
        echo '<body>'
        echo "<h1>Links for ${pkg}</h1>"

        for whl in "${WHEELS_DIR}"/*.whl; do
            [ -f "$whl" ] || continue
            filename="$(basename "$whl")"
            raw_name="${filename%%-[0-9]*}"
            normalized="$(echo "$raw_name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g')"

            if [ "$normalized" = "$pkg" ]; then
                # Compute SHA256 for PEP 503 hash verification
                sha256="$(shasum -a 256 "$whl" | awk '{print $1}')"
                # Copy wheel into the package directory
                cp "$whl" "$pkg_dir/"
                echo "  <a href=\"${filename}#sha256=${sha256}\">${filename}</a><br/>"
            fi
        done

        echo '</body></html>'
    } > "${pkg_dir}/index.html"
done

echo "Generated PEP 503 index at ${SIMPLE_DIR}/"
echo "Packages: ${!PACKAGES[*]}"
