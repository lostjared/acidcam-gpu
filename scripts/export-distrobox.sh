#!/bin/sh
# Export ACMX2 applications from Distrobox to the host desktop.
# Run this script from inside the Distrobox container.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ICON_SRC="$REPO_DIR/acmx2.png"
ICON_DIR="$HOME/.local/share/icons"
ICON_PATH="$ICON_DIR/acmx2.png"

# Install the icon to the user icon directory
mkdir -p "$ICON_DIR"
cp "$ICON_SRC" "$ICON_PATH"
echo "Installed icon to $ICON_PATH"

# Export the Qt interface
distrobox-export --app acmx2_interface \
    --export-path "$HOME/.local/bin" \
    --extra-flags "--icon $ICON_PATH"

# Create a .desktop file for the interface
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/acmx2_interface.desktop" << EOF
[Desktop Entry]
Name=ACMX2
Comment=ACMX2 GPU Shader Effects Interface
Exec=$HOME/.local/bin/acmx2_interface
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=AudioVideo;Graphics;
Keywords=shader;gpu;video;effects;cuda;
EOF
echo "Created $DESKTOP_DIR/acmx2_interface.desktop"

# Export the MIDI map tool
distrobox-export --app midi-map \
    --export-path "$HOME/.local/bin" \
    --extra-flags "--icon $ICON_PATH"

cat > "$DESKTOP_DIR/acmx2_midi-map.desktop" << EOF
[Desktop Entry]
Name=ACMX2 MIDI Map
Comment=MIDI Controller Mapping Tool for ACMX2
Exec=$HOME/.local/bin/midi-map
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Midi;
Keywords=midi;controller;mapping;acmx2;
EOF
echo "Created $DESKTOP_DIR/acmx2_midi-map.desktop"

# Update the desktop database if available
if command -v update-desktop-database > /dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
fi

echo "Export complete. Applications should appear in your host application menu."
