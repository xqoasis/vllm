#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

url="${1:?CUDA runfile URL is required}"
output="${2:?CUDA runfile output path is required}"
md5="${3:?CUDA runfile MD5 is required}"
if [[ ! "${md5}" =~ ^[0-9a-f]{32}$ ]]; then
    echo "CUDA runfile MD5 must contain 32 lowercase hex digits." >&2
    exit 1
fi

# Respect the build environment's proxy settings and retain partial downloads
# for retries. Avoid per-chunk progress output in the SCM log.
aria2c --no-conf --continue=true --auto-file-renaming=false \
    --max-connection-per-server=16 --split=16 --min-split-size=4M \
    --file-allocation=none --summary-interval=60 --show-console-readout=false \
    --connect-timeout=30 --timeout="${CUSTOM_WGET_TIMEOUT:-300}" \
    --max-tries="${CUSTOM_WGET_TRIES:-5}" --retry-wait=5 \
    --dir="$(dirname "${output}")" --out="$(basename "${output}")" \
    "${url}"

printf '%s  %s\n' "${md5}" "${output}" | md5sum --check -
