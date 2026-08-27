# ACMXVK Controls

This document is the runtime control reference for ACMXVK. Controls that depend
on an optional feature are active only when that feature was compiled and
enabled. Most toggles respond once per key press; the continuous 3D movement
controls respond while a key is held.

## Shader, playlist, and playback controls

| Input | Action | Notes |
|---|---|---|
| `Up` / `Down` | Select the previous or next shader | When playlist mode is active, these select playlist nodes instead. |
| `Shift+Up` / `Shift+Down` | Select the previous or next final shader | Forces shader selection while a playlist is active. |
| `Left` / `Right` | Select the previous or next CUDA filter | Requires a CUDA build with the acidcam-gpu filter engine active. |
| `Space` | Bypass or enable shader effects | The change uses the selected crossfade. |
| `P` | Toggle playlist mode | If no playlist was loaded, pauses or resumes video/graphic input instead. Camera input cannot be paused. |
| `L` | Freeze or resume the input and shader animation | Available for video and graphic input; camera input cannot be frozen. |
| `M` | Toggle the configured multipass chain | Has an effect only when shader passes were configured. |
| `J` | Toggle random autopilot | Playlist mode must be active. |
| `Y` | Toggle sequential autopilot | Playlist mode must be active. |
| `N` | Toggle random XFade selection for autopilot | Changes whether autopilot chooses a new XFade style automatically. |
| `K` | Lock or unlock shader and playlist selection | Useful while performing with a selected effect. |
| `[` / `]` | Select the previous or next XFade style | These keys are reserved for XFade selection and do not resize a 3D model. |

## Shader time and audio controls

| Input | Action | Notes |
|---|---|---|
| `T` | Toggle shader-time advancement | Freezes or resumes the normal shader clock. |
| `U` / `I` | Step shader time forward or backward | Each press changes shader time by `0.05`. |
| `Page Up` / `Page Down` | Increase or decrease shader-time speed | Changes the multiplier by `0.1`; these keys repeat while held. |
| `Q` | Toggle audio-reactive shader time | Requires an audio-enabled build and an open audio source. |
| `Home` | Toggle audio delta-time scaling | Scales reactive time and amplitude using frame delta; requires audio. |
| `End` | Toggle spectrum sensitivity scaling | Applies sensitivity to FFT and FFT-history data; requires audio. |
| `Insert` / `Delete` | Increase or decrease live-audio sensitivity | Changes sensitivity by `0.1`; requires live audio input. |

## Window, HUD, and capture controls

| Input | Action | Notes |
|---|---|---|
| `F` | Toggle fullscreen | Switches the ACMXVK window between windowed and fullscreen modes. |
| `F9` | Show or hide the ACMXVK runtime HUD | The HUD is preview-only and is **not encoded** into the output video. |
| `F12` | Show or hide MXVK's FPS counter | This is the renderer-level counter supplied by MXVK. |
| `E` | Show or hide the watermark | Available when watermark text was configured. The watermark is part of the encoded result. |
| `Z` | Save a processed PNG snapshot | Saves beneath the directory selected by `--prefix`. Snapshot writing runs in the background. |
| `F10` | Save an MXVK screenshot | Requires `--enable-screenshot`. |
| `Escape` | Quit ACMXVK | Closing the window also quits. |

## 3D model controls

These controls require `--enable-3d` and a model loaded with `--model`.

| Input | Action | Notes |
|---|---|---|
| `3` | Toggle between 2D sprite and 3D model rendering | Preserves the current model view. |
| `W` / `S` | Look up or down | Active while automatic view rotation is disabled. Hold to move continuously. |
| `A` / `D` | Look left or right | Active while automatic view rotation is disabled. Hold to move continuously. |
| `=` or keypad `+` | Move backward along the view direction | Hold for continuous movement. The `=` key is the unshifted key commonly shared with `+`. |
| `-` or keypad `-` | Move forward along the view direction | Hold for continuous movement. |
| `Shift+=` / `Shift+-` | Increase or decrease model scale | Changes scale by `0.05`, between `0.05` and `20.0`. On most keyboards `Shift+=` is `+`. |
| `1` / `2` | Increase or decrease movement sensitivity | Hold to adjust the 3D camera movement speed. |
| `V` | Toggle automatic view rotation | Manual `W/A/S/D` look is suspended while this is active. |
| `,` / `.` | Decrease or increase automatic rotation speed | Changes the speed by 5 degrees per second. |
| `O` | Toggle camera-distance oscillation | While active, manual distance, scale, sensitivity, and mouse-wheel changes are suspended. |
| `C` | Toggle three-axis model wave deformation | Applies ACMX2-compatible animated model deformation. |
| `X` | Reset the model view | Restores the centered skybox view, default orientation, distance, and model scale. |
| Left mouse drag | Look around | Also updates the mouse state exposed to shaders. |
| Mouse wheel up/down | Move forward/backward along the view direction | Distance is clamped to the supported model-camera range. |

The left mouse position and button state are supplied to compatible fragment and
compute shaders even when 3D mode is not active.

## MIDI controls

MIDI support requires an ACMXVK build configured with `-DMIDI=ON`. List input
ports with `--list-midi`, select one with `--midi-device`, and load an ACMX2
mapping file with `--midi-map`:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --shaders ./build/acmxvk/shaders \
    --midi-device 0 \
    --midi-map controller.midi_cfg \
    --midi-monitor
```

ACMXVK recognizes the following ACMX2 MIDI-map action codes:

| Action code | Keyboard equivalent | Action |
|---:|---|---|
| `32` | `Space` | Toggle shader bypass |
| `260` / `261` | `Insert` / `Delete` | Increase or decrease audio sensitivity |
| `262` / `263` | `Right` / `Left` | Next or previous CUDA filter |
| `264` / `265` | `Down` / `Up` | Next or previous shader/playlist node |
| `266`, `504` / `267`, `505` | `Page Up` / `Page Down` | Increase or decrease shader-time speed |
| `268` | `Home` | Toggle audio delta-time scaling |
| `269` | `End` | Toggle spectrum sensitivity scaling |
| `500` / `501` | `U` / `I` | Step shader time forward or backward |
| `44` / `46` | `,` / `.` | Decrease or increase 3D rotation speed |
| `51` | `3` | Toggle 2D/3D rendering |
| `67` | `C` | Toggle the 3D wave effect |
| `79` | `O` | Toggle 3D camera-distance oscillation |
| `86` | `V` | Toggle automatic 3D view rotation |
| `88` | `X` | Reset the model view |
| `91` / `93` | `Shift+-` / `Shift+=` | Decrease or increase model scale |
| `69` | `E` | Toggle the watermark |
| `74`, `78` | `J` | Toggle random autopilot |
| `75` | `K` | Toggle shader lock |
| `76` | `L` | Toggle rendering freeze |
| `77` | `M` | Toggle multipass rendering |
| `80` | `P` | Toggle playlist mode or input pause |
| `81` | `Q` | Toggle audio-reactive shader time |
| `84` | `T` | Toggle normal shader time |
| `89` | `Y` | Toggle sequential autopilot |
| `90` | `Z` | Save a processed PNG snapshot |
| `600:601` through `606:607` | Custom uniform sliders 1-4 | Map absolute MIDI values to `slider1` through `slider4`. |

Paired action mappings use a centered controller value: `64` is neutral, and
distance from the center controls how often the action repeats. Slider pairs
and direct `--midi-cc [channel:]CC=uniform` mappings instead normalize the
controller's `0`-`127` range into the custom uniform range declared by
`library.json`.

## Feature requirements at a glance

| Control group | Build/runtime requirement |
|---|---|
| CUDA filter selection | Configure with `-DWITH_CUDA=ON` and run with the acidcam-gpu engine available. |
| Audio controls | Configure with `-DAUDIO=ON` and open a live, file, or source-audio input as appropriate. |
| MIDI controls | Configure with `-DMIDI=ON`, then use `--midi-device`, `--midi-map`, or `--midi-cc`. |
| 3D controls | Start with `--enable-3d --model <file>`. |
| Playlist and autopilot | Load a playlist and enable playlist mode with `P`. |
| Multipass toggle | Configure one or more shader passes. |
| `F10` screenshot | Start ACMXVK with `--enable-screenshot`. |

Run `acmxvk --help` for the full command-line option reference.
