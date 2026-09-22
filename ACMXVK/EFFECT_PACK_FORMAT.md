# ACMXVK Effect Pack Format

This document defines version 1 of the portable ACMXVK effect-pack manifest.
Parsing, discovery, pack-local compilation, cache validation, and live runtime
activation are implemented. The interface also supports pack authoring and
folder transfer. The remaining project integration and hardening work is
tracked in [`../Effects.md`](../Effects.md).

## Directory layout

An effect pack is a directory containing an `effect.json` manifest and the
resources named by that manifest. Paths use `/` separators and are relative to
the directory containing `effect.json`.

```text
dreaming-crystal/
    effect.json
    icon.png
    shaders/
        warp.comp
        kaleidoscope.frag
```

Absolute paths, Windows drive paths, backslashes, and `..` parent traversal are
rejected. Model names are logical identifiers, not filesystem paths. This keeps
packs portable among Linux, macOS, and Windows.

## Root object

The root object accepts the following fields. Unknown fields are errors.

| Field | Required | Value |
| --- | --- | --- |
| `format` | yes | Must be `acmxvk-effect-pack`. |
| `version` | yes | Integer `1`. |
| `id` | yes | Stable ASCII token, up to 128 bytes. |
| `name` | yes | Display name, up to 1024 bytes. |
| `description` | no | Display description, up to 1024 bytes. |
| `icon` | no | Safe relative image path. |
| `passes` | yes | One to 64 relative shader paths in execution order; repeats are allowed. |
| `requires` | no | Resource requirements object. |
| `controls` | no | Up to 64 friendly custom-uniform controls. |
| `audio_mappings` | no | Up to 64 audio-to-uniform declarations. |
| `midi_mappings` | no | Up to 64 logical MIDI-input-to-uniform declarations. |
| `deep_dream` | no | Optional Deep Dream configuration. |

Stable Diffusion is intentionally not part of the effect-pack format. A
`stable_diffusion` field is rejected rather than silently ignored.

## Resource requirements

`requires` may contain the Boolean fields `history`, `spectrum`,
`spectrum_history`, and `original_frame`. Missing fields default to `false`.
Later increments validate these declarations against the compiled shader
pipeline.

## Controls

Each control requires:

- `id`: stable, pack-local ASCII token;
- `label`: user-facing name;
- `uniform`: valid GLSL-style identifier;
- `minimum` and `maximum`: finite values with `minimum < maximum`;
- `step`: finite positive value no larger than the range;
- `default`: finite value within the inclusive range.

Control IDs and uniform names must each be unique within the pack.
In the ACMXVK interface, the Effect Pack Controls panel uses `label` for the
slider, shows `uniform` for diagnostics, and remembers edited values by pack ID.
Reset returns one control or the entire pack to its declared defaults. Saved
values are sent with pack activation; they are not written back to `effect.json`.

## Audio and MIDI mappings

An audio mapping contains `source`, `uniform`, `minimum`, and `maximum`. The
source is one of `low`, `mid`, `high`, `peak`, `smooth`, or `rms`. ACMXVK takes
the corresponding live audio metric after sensitivity and warmup, clamps it to
0–1, and maps it linearly into the declared numeric range each frame. Without
an active audio source the control retains its saved/default value. Shader
spectrum and spectrum-history resources remain available through `requires`;
these mappings do not replace the shader's spectrum textures.

A MIDI mapping contains `input`, `uniform`, `minimum`, and `maximum`. Version 1
supports the device-independent inputs `slider_1` through `slider_4`. The
user's selected ACMX2 MIDI controller profile maps physical channels/CCs to
those four actions (600/601 through 606/607); the pack contains no device
numbers. A received 0–127 controller value maps linearly into the declared
range. A profile and MIDI input must be enabled before launch.

Both mapping types must target a declared control uniform, and their numeric
ranges must lie within that control's range. A control may have only one
audio-or-MIDI mapping, avoiding conflicting live writers. Switching packs
switches these mappings with the pipeline. Leaving pack mode resumes the
user's normal MIDI mappings; builds without audio or MIDI retain the pack's
static controls and warn that the respective live mappings are inactive.

## Deep Dream

`deep_dream` accepts these fields:

| Field | Range/default |
| --- | --- |
| `enabled` | Boolean, default `false` |
| `model` | Logical model token |
| `layer` | Model layer token |
| `channel` | `-1` for all or 0–65535 |
| `iterations` | 1–100 |
| `strength` | Greater than 0 through 10 |
| `feedback` | 0–0.99 |
| `zoom` | 0.9–1.1 |
| `rotation` | -5–5 |
| `working_size` | 0 for automatic or 64–4096 |
| `fp16` | Boolean |
| `octaves` | 1–8 |
| `octave_scale` | 1.1–3.0 |
| `jitter` | 0–64 |
| `smoothing` | 0–16 |
| `gpu_filter_before_dream` | Boolean |

When Deep Dream is enabled, both `model` and `layer` are required. Models are
resolved locally and are not bundled by default. The Effect Pack browser treats
`model` as a logical ID (for example, `vgg16`), not a portable filesystem path.
It searches the folders configured through **Model Folder...**, the directory
of the interface's current Deep Dream model, the application's data `models`
folder, the installed ACMXVK `models` folder, and `models` in the current
working directory. It tries the ID as written, `<id>.pt`,
`deep-dream-<id>.pt`, and `<id>.torchscript`. **Choose Dream Model...** sets a
local override for one pack; that path is stored in the interface's local
settings, never in the shareable pack JSON.

The browser marks an enabled Dream pack unavailable if its model cannot be
resolved or the current build lacks Deep Dream support. An unavailable model
in a disabled Dream configuration produces a warning but does not prevent
activation. Switching packs applies the resolved model and the pack's Dream
options with its shader pipeline and controls. A model/layer validation failure
rejects the switch and preserves the previously active effect. A pack without
enabled Deep Dream disables Dream; returning to the normal shader library
restores the Dream configuration that was active before entering pack mode.

## Validation behavior

The parser rejects malformed JSON, duplicate JSON keys, comments, trailing
commas, unknown fields, invalid types, non-finite or out-of-range numbers,
excessive list sizes, duplicate control IDs or uniform declarations, and unsafe
resource paths. Repeated shader pass paths are permitted and retain their order.
Errors identify the failing field or array entry. Parsing creates an in-memory
value only and never changes renderer state.

See `tests/effect_packs/complete/effect.json` for a complete non-rendering
version 1 example.

To build one pack from a terminal, run
`acmxvk --build-effect-pack /path/to/effect.json --glslc /path/to/glslc --parallel 2`.
The compiler writes the pack-local `.acmxvk-build` cache and reports progress
for each unique shader source. A repeated pass still appears multiple times in
the rendered pipeline.

## Authoring and transfer in the interface

Open **Playback → Effect Packs** in ACMXVK mode. **Create from Current...**
captures the selected shader or enabled multipass order from a GLSL source
library, the current custom-uniform definitions and values, inferred resource
bindings, and enabled Deep Dream settings. It prompts for an optional icon.
When a pack is active, this action saves a copy of that pack instead, retaining
its audio/MIDI mappings. Hardware-specific MIDI controller settings and Deep
Dream model binaries are not embedded.

**Save Pack As...** creates a new user pack from the selected pack and uses its
locally saved control values as the new defaults. **Export...** copies a pack
into a chosen folder, preserving its ID. **Import...** copies a chosen pack
folder into the user's effect-pack root; an ID collision produces a new local
ID, and a folder-name collision receives a numbered suffix. Copying is done
off the UI thread with a progress dialog. These operations copy only the
manifest, referenced shader passes and recursive includes, and the optional
icon. They reject paths that escape the source root and never copy rendered
outputs, logs, or temporary files. Transfer requires GLSL `.frag` or `.comp`
passes; SPIR-V-only packs need source shaders before they can be exported.

Portable exports currently omit `.acmxvk-build`: the existing cache manifest
does not yet contain all required shader-ABI, Vulkan-target, source, and
recursive-include hashes. Rebuild the imported pack before activation; the
browser's **Build & Activate** action does this automatically. Compatible
compiled-cache transfer will be enabled only after the required hard keys
are present and checked, without making compiler or ACMXVK patch versions
unconditional invalidation keys.

## Discovery and compiled cache

Catalog discovery accepts one or more user-selected or installed-data roots,
searches up to eight directory levels, and ignores symbolic links,
`.acmxvk-build`, and editor-preview directories. Packs are ordered by stable ID.
If two manifests declare the same ID, the first canonical manifest is retained
and the duplicate is reported as a warning. Missing optional icons are warnings;
an invalid manifest does not prevent other packs from being discovered.

Compiled shaders are stored inside the pack:

```text
dreaming-crystal/
    .acmxvk-build/
        effect-cache.json
        shaders/
            warp.comp.spv
            kaleidoscope.frag.spv
```

The source directory structure and pass order are preserved. A shader is
rebuilt when its source, any recursively included file, or `effect.json` is
newer than its valid SPIR-V output. Otherwise it remains current. Builds use
the same atomic compiler/install path as the full shader-library builder and
may run up to 64 jobs. Interrupted `.acmxvk-tmp-*`, live-preview temporary
files, and `.editor-preview` contents are removed before a build.

After compilation, ACMXVK verifies that every pass is fragment or compute
SPIR-V and that source extensions match their compiled stages. Bindings for
history, spectrum, spectrum history, and `originalFrame` must be declared in
the manifest's `requires` object before a pack can be considered ready.

## Live activation

The Qt interface and ACMXVK use shared-memory protocol version 12 for sequenced
effect-pack activation requests. A request names the pack's canonical
`effect.json`; ACMXVK reparses the manifest and accepts only current, validated
SPIR-V beneath that pack's `.acmxvk-build` directory. Ordinary shader,
multipass, and playlist requests remain restricted to the configured shader
library.

Activation stages the complete pass order and default control values before it
changes the existing post-processing pipeline. ACMXVK provisions newly required
history, spectrum, and spectrum-history resources and uses the existing
crossfade path where possible. If validation, resource creation, or pipeline
attachment fails, the previous pipeline and uniform state are restored. Sending
an empty pack path leaves pack mode and restores the normal shader workflow that
was active before the first pack was selected.
