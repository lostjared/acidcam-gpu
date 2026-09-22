# ACMXVK Effect Pack Format

This document defines version 1 of the portable ACMXVK effect-pack manifest.
Increment 1 provides parsing and validation only; discovery, compilation, and
live activation are introduced by later increments described in
[`../Effects.md`](../Effects.md).

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
| `passes` | yes | One to 64 unique relative shader paths in execution order. |
| `requires` | no | Resource requirements object. |
| `controls` | no | Up to 64 friendly custom-uniform controls. |
| `audio_mappings` | no | Up to 64 audio-to-uniform declarations. |
| `midi_mappings` | no | Up to 64 MIDI CC-to-uniform declarations. |
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

## Audio and MIDI mappings

An audio mapping contains `source`, `uniform`, `minimum`, and `maximum`. The
source is a portable token such as `low`, `mid`, `high`, `peak`, or `smooth`.
The numeric range maps the normalized source value to the named uniform.
Increment 7 will finalize runtime source availability and application behavior.

A MIDI mapping contains `uniform`, a `channel` from 1 through 16, a MIDI CC
`controller` from 0 through 127, and finite `minimum` and `maximum` values with
`minimum < maximum`.

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
resolved locally and are not bundled by default.

## Validation behavior

The parser rejects malformed JSON, duplicate JSON keys, comments, trailing
commas, unknown fields, invalid types, non-finite or out-of-range numbers,
excessive list sizes, duplicate passes or controls, and unsafe resource paths.
Errors identify the failing field or array entry. Parsing creates an in-memory
value only and never changes renderer state.

See `tests/effect_packs/complete/effect.json` for a complete non-rendering
version 1 example.

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
