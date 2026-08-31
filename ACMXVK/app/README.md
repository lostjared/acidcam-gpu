# ACMXVK application implementation

`acmx.cpp` owns platform/library includes, compile-time resource defaults, and
the process entry point. The implementation sections in this directory keep
the application organized by responsibility:

- `options.hpp` / `options.cpp`: command-line parsing, validation, resources,
  and help.
- `shader_library.hpp` / `shader_library.cpp`: manifests and source-library
  compilation.
- `media_utils.hpp` / `media_utils.cpp`: image loading, video metadata, frame
  rotation, and CUDA device selection.
- `media_helpers.hpp` / `media_helpers.cpp`: playlist data and asynchronous
  camera capture state.
- `interface_client.hpp` / `interface_client.cpp`: validated shared-memory
  connection and synchronized snapshots of live interface controls.
- `snapshot_writer.hpp` / `snapshot_writer.cpp`: background snapshot queue and
  PNG, raw RGBA, WebP, and TIFF encoding.
- `window_lifecycle.ipp`: construction, events, and frame processing.
- `window_state.ipp`: `MainWindow` state shared by the other sections.
- `window_audio_midi.ipp`: audio, MIDI, controls, and media clocks.
- `window_shaders.ipp`: shader loading, interface IPC, and playlists.
- `window_overlay.ipp`: resources, HUD, watermark, and DNN overlays.
- `window_io.ipp`: inputs, encoding, snapshots, and frame readback.
- `window_rendering.ipp`: 3D, crossfades, pipelines, history, and uploads.

The low-coupling options, shader-library, media-utility, camera-helper,
interface-client, and snapshot-writer modules are independently compiled
translation units. Only the stateful window implementation remains in ordered
`.ipp` sections because those sections share one class definition and many
compile-time feature switches.
