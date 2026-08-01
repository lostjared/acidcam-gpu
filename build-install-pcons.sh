#!/usr/bin/env bash

set -Eeuo pipefail

readonly REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly INTERFACE_DIR="$REPO_DIR/ACMX2/interface"
readonly ROOT_STAGE_DIR="$REPO_DIR/dist"
readonly INTERFACE_STAGE_DIR="$INTERFACE_DIR/dist"

show_usage() {
    cat <<'EOF'
Usage: ./build-install-pcons.sh [PCONS_VAR=value ...]

Build and install ACMX2, the CUDA tools (when enabled), helper programs, and
the Qt interface. The default installation prefix is /usr/local.

Examples:
  ./build-install-pcons.sh
  ./build-install-pcons.sh AUDIO=1 MIDI=1 DNN=1
  ./build-install-pcons.sh WITH_CUDA=0
  VARIANT=debug ./build-install-pcons.sh WITH_CUDA=0

Environment:
  CC, CXX                  C and C++ compilers (defaults: gcc and g++)
  VARIANT                  release or debug (default: release)
  ACMX2_INSTALL_PREFIX     installation prefix (default: /usr/local)
  ACMX2_DESTDIR            optional packaging root prepended to the prefix
  ACMX2_JOBS               optional parallel build job count
  ACMX2_PCONS              pcons executable (default: pcons)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

compiler_c="${CC:-gcc}"
compiler_cxx="${CXX:-g++}"
variant="${VARIANT:-release}"
pcons_command="${ACMX2_PCONS:-pcons}"
install_prefix="${ACMX2_INSTALL_PREFIX:-/usr/local}"
destination_root="${ACMX2_DESTDIR:-}"
jobs="${ACMX2_JOBS:-}"

for argument in "$@"; do
    if [[ "$argument" == VARIANT=* ]]; then
        variant="${argument#VARIANT=}"
    fi
done

if [[ "$install_prefix" != /* ]]; then
    echo "ACMX2_INSTALL_PREFIX must be an absolute path: $install_prefix" >&2
    exit 2
fi
if [[ -n "$destination_root" && "$destination_root" != /* ]]; then
    echo "ACMX2_DESTDIR must be an absolute path: $destination_root" >&2
    exit 2
fi
if [[ -n "$jobs" && ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACMX2_JOBS must be a positive integer: $jobs" >&2
    exit 2
fi

for command_name in "$compiler_c" "$compiler_cxx" "$pcons_command" install cp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command not found: $command_name" >&2
        exit 3
    fi
done

install_prefix="${install_prefix%/}"
readonly INSTALL_ROOT="${destination_root}${install_prefix}"

pcons_options=(--reconfigure)
if [[ -n "$jobs" ]]; then
    pcons_options+=(-j "$jobs")
fi

echo "Building and staging ACMX2 targets..."
(
    cd "$REPO_DIR"
    CC="$compiler_c" CXX="$compiler_cxx" \
        "$pcons_command" "${pcons_options[@]}" "VARIANT=$variant" "$@" all
)

echo "Building and staging the Qt interface..."
(
    cd "$INTERFACE_DIR"
    CC="$compiler_c" CXX="$compiler_cxx" VARIANT="$variant" \
        "$pcons_command" "${pcons_options[@]}" all
)

required_files=(
    "$ROOT_STAGE_DIR/bin/acmx2"
    "$ROOT_STAGE_DIR/bin/audio_transfer"
    "$ROOT_STAGE_DIR/bin/shader_generator"
    "$INTERFACE_STAGE_DIR/bin/acmx2_interface"
    "$INTERFACE_STAGE_DIR/bin/midi-map"
    "$INTERFACE_STAGE_DIR/share/applications/acmx2-interface.desktop"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Expected staged file was not produced: $required_file" >&2
        exit 4
    fi
done
if [[ ! -d "$ROOT_STAGE_DIR/share/acmx2/data" ]]; then
    echo "Expected staged runtime data was not produced." >&2
    exit 4
fi

install_parent="$INSTALL_ROOT"
while [[ ! -e "$install_parent" && "$install_parent" != "/" ]]; do
    install_parent="$(dirname -- "$install_parent")"
done

privilege_command=()
if ((EUID != 0)) && [[ ! -w "$install_parent" ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Installing to $INSTALL_ROOT requires root privileges, but sudo was not found." >&2
        exit 5
    fi
    privilege_command=(sudo)
fi

run_install() {
    "${privilege_command[@]}" "$@"
}

echo "Installing into $INSTALL_ROOT..."

for program_name in acmx2 audio_transfer shader_generator acidcam; do
    program_source="$ROOT_STAGE_DIR/bin/$program_name"
    if [[ -f "$program_source" ]]; then
        run_install install -Dm755 \
            "$program_source" "$INSTALL_ROOT/bin/$program_name"
    fi
done

run_install install -Dm755 \
    "$INTERFACE_STAGE_DIR/bin/acmx2_interface" \
    "$INSTALL_ROOT/bin/acmx2_interface"
run_install install -Dm755 \
    "$INTERFACE_STAGE_DIR/bin/midi-map" \
    "$INSTALL_ROOT/bin/midi-map"

library_source="$ROOT_STAGE_DIR/lib/libacidcam-gpu.so"
if [[ -f "$library_source" ]]; then
    run_install install -Dm755 \
        "$library_source" "$INSTALL_ROOT/lib/libacidcam-gpu.so"
fi

run_install install -d "$INSTALL_ROOT/share/acmx2"
run_install cp -a \
    "$ROOT_STAGE_DIR/share/acmx2/." "$INSTALL_ROOT/share/acmx2/"
run_install install -Dm644 \
    "$REPO_DIR/ACMX2/data/win-icon.png" \
    "$INSTALL_ROOT/share/acmx2/acmx2.png"
run_install install -Dm644 \
    "$INTERFACE_STAGE_DIR/share/applications/acmx2-interface.desktop" \
    "$INSTALL_ROOT/share/applications/acmx2-interface.desktop"

if [[ -z "$destination_root" && ("$install_prefix" == "/usr" || "$install_prefix" == "/usr/local") ]]; then
    if command -v ldconfig >/dev/null 2>&1; then
        run_install ldconfig
    fi
fi

echo
echo "Installation complete."
echo "  Programs: $INSTALL_ROOT/bin"
if [[ -f "$library_source" ]]; then
    echo "  Library:  $INSTALL_ROOT/lib/libacidcam-gpu.so"
fi
echo "  Data:     $INSTALL_ROOT/share/acmx2"
echo "  Desktop:  $INSTALL_ROOT/share/applications/acmx2-interface.desktop"
