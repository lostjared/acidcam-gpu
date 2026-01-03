#!/bin/bash
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --device /dev/video0 \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp/xdg \
  -e QT_QPA_PLATFORM=xcb \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/xdg/$WAYLAND_DISPLAY \
  -v ~/ACMX2:/root/share \
  -w /opt/src/acidcam-gpu/ACMX2/interface/build \
  localhost/acmx2-cuda-opencv:dev
