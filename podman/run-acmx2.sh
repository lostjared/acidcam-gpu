#!/bin/bash
xhost +local:docker
set -euo pipefail

IMAGE="ghcr.io/lostjared/acmx2:latest"

# 1. Get Host Audio Paths
PULSE_SOCKET="/run/user/$(id -u)/pulse/native"
PULSE_COOKIE="$HOME/.config/pulse/cookie"
HOST_SHARE="$HOME/container_share"

if command -v xhost >/dev/null 2>&1; then
  xhost +si:localuser:root >/dev/null 2>&1 || true
fi

# 2. Check if cookie exists
if [ ! -f "$PULSE_COOKIE" ]; then
    echo "Warning: Pulse Cookie not found at $PULSE_COOKIE"
fi

# 3. Ensure the share directory exists on host
mkdir -p "$HOST_SHARE"

exec podman run -it \
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
  -e PULSE_SERVER=unix:/tmp/pulse-socket \
  -e PULSE_COOKIE=/tmp/pulse-cookie \
  -v "$PULSE_SOCKET":/tmp/pulse-socket \
  -v "$PULSE_COOKIE":/tmp/pulse-cookie \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOST_SHARE":/root/share \
  "$IMAGE" bash -lc '
    mkdir -p /tmp/xdg
    chmod 700 /tmp/xdg
    # Double check audio inside before launching
    echo "Checking audio connection..."
    pactl info || echo "pactl failed, continuing anyway..."
    exec /opt/src/acidcam-gpu/ACMX2/interface/build/interface
  '
