#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

mkdir -p "${SCRIPT_DIR}/tmp" "${SCRIPT_DIR}/.flatpak-builder" \
    "${SCRIPT_DIR}/build-dir" "${SCRIPT_DIR}/repo"

export TMPDIR="${SCRIPT_DIR}/tmp"

cd "${PROJECT_DIR}"
flatpak-builder \
    --user \
    --force-clean \
    --install-deps-from=flathub \
    --default-branch=stable \
    --state-dir="${SCRIPT_DIR}/.flatpak-builder" \
    --repo="${SCRIPT_DIR}/repo" \
    "${SCRIPT_DIR}/build-dir" \
    "${SCRIPT_DIR}/io.github.lostjared.ACMX2.yml"

flatpak build-bundle \
    "${SCRIPT_DIR}/repo" \
    "${SCRIPT_DIR}/ACMX2.flatpak" \
    io.github.lostjared.ACMX2 \
    stable
