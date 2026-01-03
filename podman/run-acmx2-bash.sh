#!/bin/bash
set -euo pipefail

IMAGE="ghcr.io/lostjared/acmx2:latest"

if command -v xhost >/dev/null 2>&1; then
  xhost +si:localuser:root >/dev/null 2>&1 || true
fi

exec podman run --rm -it \
  --security-opt=label=disable \
  --net=host \
  --cap-add=SYS_NICE \
  --cap-add=SYS_RESOURCE \
  --device nvidia.com/gpu=all \
  --device /dev/video0 \
  --device /dev/snd \
  -e DISPLAY="${DISPLAY:-}" \
  -e QT_QPA_PLATFORM=xcb \
  -e XDG_RUNTIME_DIR=/tmp/xdg \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "$IMAGE" bash 

