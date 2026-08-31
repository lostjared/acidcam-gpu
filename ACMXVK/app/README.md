# ACMXVK application implementation

`acmx.cpp` owns the process entry point. `main_window.hpp` and
`main_window.cpp` own the render-window boundary. Supporting application code
is organized in this directory by responsibility:

- `options.hpp` / `options.cpp`: command-line parsing, validation, resources,
  and help.
- `resource_paths.hpp` / `resource_paths.cpp`: built, installed, and
  user-selected shader, model, font, and crossfade asset resolution.
- `shader_library.hpp` / `shader_library.cpp`: manifests and source-library
  compilation.
- `playlist.hpp` / `playlist.cpp`: shader-name resolution and bounded,
  validated playlist parsing.
- `output_paths.hpp` / `output_paths.cpp`: PNG sequence directories, numbered
  frame paths, and collision-safe timestamped snapshot names.
- `media_utils.hpp` / `media_utils.cpp`: image loading, video metadata, frame
  rotation, and CUDA device selection.
- `media_helpers.hpp` / `media_helpers.cpp`: asynchronous camera capture state.
- `camera_probe.hpp` with `camera_probe.cpp` on Linux and
  `camera_probe_macos.mm` on macOS: native camera device, resolution, format,
  and frame-rate discovery for the CLI and Qt interface.
- `interface_client.hpp` / `interface_client.cpp`: validated shared-memory
  connection and synchronized snapshots of live interface controls.
- `snapshot_writer.hpp` / `snapshot_writer.cpp`: background snapshot queue and
  PNG, raw RGBA, WebP, and TIFF encoding.
- `../main_window.hpp` / `../main_window.cpp`: the `MainWindow` declaration,
  private runtime state, and definitions for lifecycle, rendering, shaders,
  interface control, audio/MIDI, overlays, media I/O, 3D, crossfades, history,
  capture, and frame uploads.

The low-coupling options, resource-path, shader-library, playlist, output-path,
media-utility, camera-helper, interface-client, and snapshot-writer modules are
independently compiled translation units. The render-window class now has a
dedicated header and translation unit. All former `window_*.ipp` sections have
been retired; the declaration and state live in `main_window.hpp`, and the
method definitions live in `main_window.cpp`.
