# ACMXVK Effect Packs

## Goal

Add portable, self-contained effect packs that organize finished ACMXVK visual
setups without replacing the existing shader-library workflow. A pack selects an
ordered fragment/compute pipeline, presents friendly controls, restores its last
values, and can optionally configure audio, MIDI, and Deep Dream. Stable
Diffusion remains outside the effect-pack format as an offline-render feature.

Effect packs must use the existing ACMXVK/MXVK renderer, shader pipeline,
custom-uniform storage, resource bindings, Deep Dream implementation, and live
interface control. The feature is an organization and recall layer, not a new
rendering architecture.

## Estimated implementation

The complete feature is planned as **10 increments**. Each increment should
compile and have its own focused tests before the next begins.

## Proposed version 1 layout

```text
dreaming-crystal/
    effect.json
    icon.png
    shaders/
        warp.comp
        kaleidoscope.frag
        original_blend.comp
    mappings/
        midi.json
```

The pack owns its source shaders. Compiled SPIR-V is stored in a pack-local
`.acmxvk-build/` cache and is regenerated only when its source, includes, or
manifest data is newer. Portable export may include compatible compiled output
for fast startup, while temporary compiler and editor-preview files are
excluded. Exported cache metadata must identify the shader ABI, target Vulkan
environment, source and recursive-include hashes, compiler identity/version,
and ACMXVK build version so an incompatible cache can be rejected cheaply and
rebuilt from source.

## Proposed JSON contract

```json
{
    "format": "acmxvk-effect-pack",
    "version": 1,
    "id": "org.example.dreaming-crystal",
    "name": "Dreaming Crystal",
    "description": "A crystalline feedback and symmetry effect.",
    "icon": "icon.png",
    "passes": [
        "shaders/warp.comp",
        "shaders/kaleidoscope.frag",
        "shaders/original_blend.comp"
    ],
    "requires": {
        "history": true,
        "spectrum": false,
        "spectrum_history": false,
        "original_frame": true
    },
    "controls": [
        {
            "label": "Symmetry",
            "uniform": "symmetry",
            "minimum": 1.0,
            "maximum": 16.0,
            "step": 1.0,
            "default": 6.0
        }
    ],
    "deep_dream": {
        "enabled": false,
        "model": "vgg16",
        "layer": "relu4_2"
    }
}
```

The final schema will also allow optional audio and MIDI sections. It will reject
unknown unsafe paths, absolute bundled-resource paths, parent traversal,
duplicate control IDs or uniform declarations, invalid ranges, excessive
counts, and Stable Diffusion fields. Duplicate pass paths are valid and retain
their declared order because running the same shader more than once is a useful
multipass operation. Pack paths are resolved relative to the directory
containing `effect.json`.

## Design decisions

- The full shader library remains available and unchanged.
- Pack identity uses a stable `id`; display names may change or be translated.
- Passes are an ordered execution list, not a set. Repeated paths such as
  blur → blur, feedback → feedback, or repeated sharpening are preserved.
- Control IDs and control uniform declarations must be unique even though pass
  paths may repeat.
- Pack source files are authoritative. `.acmxvk-build` is only an incremental
  cache. Cache reuse requires matching ABI, Vulkan target, content hashes,
  compiler/build identity, and any other compatibility fields defined by the
  cache format; otherwise ACMXVK recompiles from source.
- A live activation is transactional: validate and prepare the new pack first,
  then replace the active pipeline and settings together. A failed activation
  leaves the current effect running.
- Shader resource declarations are validated against SPIR-V reflection when
  possible. ACMXVK remains authoritative and provisions the same history,
  spectrum, spectrum-history, and `originalFrame` resources used today.
- Friendly controls map by uniform name, not by raw slot number. ACMXVK resolves
  each name against the active custom-uniform table.
- MIDI remains device-independent at the pack layer. Packs map logical inputs
  such as `slider_1` to named controls/uniforms, or reference a selected
  pack-specific MIDI profile. The user's controller profile remains responsible
  for translating physical device messages and CC numbers into those inputs.
- Per-pack control state is user state keyed by pack ID. It is not written back
  into a shared or read-only pack unless the user explicitly saves the pack.
- Deep Dream model references are logical identifiers or requirements. A local
  resolver maps them to installed model paths. Models are not copied by default.
- Stable Diffusion is explicitly excluded from version 1 effect packs.
- Linux, macOS, and Windows use the same format and path rules.

## Increment plan

### Increment 1 — Format, parser, validation, and tests

- Add effect-pack value types for metadata, passes, requirements, controls,
  audio/MIDI declarations, and Deep Dream settings.
- Implement strict `effect.json` parsing and relative-path validation.
- Define limits consistent with the renderer: 64 passes, 64 custom uniforms,
  bounded strings, finite numeric values, and unique pack/control IDs. Preserve
  repeated pass paths in their declared order while rejecting duplicate control
  IDs and control uniform declarations.
- Add parser fixtures covering a minimal valid pack, a complete valid pack, and
  malformed/unsafe packs.
- Add a versioned format document and one non-rendering example pack fixture.

Acceptance: parser tests pass on all portable builds and invalid packs produce a
specific field/path error without touching runtime state.

### Increment 2 — Discovery, pack-local compilation, and cache validation

- Discover packs from user-selected roots and installed data roots.
- Build a catalog keyed by pack ID with duplicate-ID and missing-icon warnings.
- Reuse the existing ACMXVK shader compiler/build logic to compile only a pack's
  source files into its local `.acmxvk-build` directory.
- Preserve timestamps and skip valid SPIR-V; remove or ignore interrupted
  `.acmxvk-tmp-*`, live-preview, and editor-preview files.
- Record the shader ABI version, Vulkan target environment, source/include
  hashes, compiler identity/version, and ACMXVK build version in cache metadata.
  Reuse compiled output only when all required compatibility fields match.
- Validate shader stages, pass count, and declared resource requirements.

Acceptance: a three-pass sample pack builds incrementally and a second build does
no work until a source or include changes.

### Increment 3 — Runtime activation through the existing pipeline

- Extend the interface protocol with a versioned effect-pack activation request.
- Allow validated compiled shaders beneath the requested pack root while keeping
  the current library path restrictions for ordinary shader selection.
- Feed pack passes into the existing `configured_passes`, SPIR-V inspection,
  resource provisioning, crossfade, and `applyShaderPipeline()` paths.
- Make activation transactional and return/log clear rejection reasons.
- Keep ordinary shaders, multipass, and playlists working as they do now.

Acceptance: a running ACMXVK process switches between the normal library and two
external packs without restart or partial pipeline state.

### Increment 4 — Icon-based Effect Pack browser

- Add an ACMXVK-only Effect Packs menu entry and modeless browser.
- Show a responsive icon grid with name, optional description, validation/build
  status, refresh, and pack-root management.
- Load icons asynchronously and cache scaled thumbnails so large catalogs do not
  freeze the interface.
- Activate a ready pack with one click; build stale packs with visible progress.
- Remember the last selected pack without forcing it onto a new session.

Acceptance: the browser remains responsive while discovering/building packs and
can switch the running engine from the icon grid.

### Increment 5 — Friendly controls and per-pack state

- Add a modeless control panel generated from the pack's `controls` array.
- Reuse the existing slider/spin-box behavior and live custom-uniform publication.
- Show friendly labels while retaining exact GLSL uniform names in diagnostics.
- Save current values by pack ID and restore them when returning to the pack.
- Add reset-current-control and reset-pack-to-defaults actions.

Acceptance: switching A → B → A restores A's edited values and publishes all
restored values atomically with the pipeline switch.

### Increment 6 — Deep Dream integration and model resolution

- Add the complete supported Deep Dream configuration to the schema.
- Resolve logical model IDs through configured model search paths, with an
  optional explicit local override.
- Reuse the current shared-memory Deep Dream update and validation path.
- Apply Deep Dream in the same activation transaction and preserve the previous
  configuration if model/layer validation fails.
- Clearly mark packs whose optional or required model is unavailable.

Acceptance: switching packs can enable, reconfigure, or disable Deep Dream live;
missing models produce a useful warning and do not break the active effect.

### Increment 7 — Audio and MIDI mappings

- Finalize version 1 audio mapping semantics around existing spectrum bands,
  spectrum history, and named uniforms.
- Define logical, device-independent MIDI inputs such as `slider_1`, `knob_2`,
  or named pack actions and map those inputs to pack controls/uniforms.
- Reuse the user's selected controller profile to translate physical MIDI
  devices, channels, and CC numbers into logical inputs; optionally let a pack
  reference a compatible pack-specific MIDI profile without embedding hardware
  assumptions in `effect.json`.
- Extend live control only where needed so mappings can change with the pack.
- Restore the user's previous non-pack mappings when leaving pack mode.
- Keep builds without audio or MIDI functional and show capability warnings.

Acceptance: an audio-reactive pack and a MIDI-controlled pack switch live without
restarting ACMXVK, including graceful behavior when those features are disabled.

### Increment 8 — Authoring, save, import, and export

- Add “Create Effect Pack from Current Setup” and “Save Effect Pack” workflows.
- Capture the current pass order, uniform definitions/values, resource needs,
  mappings, and Deep Dream configuration.
- Copy required shaders/includes and an optional icon using the existing
  background-copy/progress patterns.
- Export a portable folder or archive with safe relative paths and no rendered
  videos, logs, snapshots, or temporary files.
- Include compiled SPIR-V only with complete compatibility metadata; consumers
  discard and rebuild cache entries whose ABI, Vulkan target, hashes, compiler,
  or ACMXVK build requirements do not match.
- Import into a chosen user pack root with collision handling.

Acceptance: a setup created from the current session can be exported, moved to a
different directory/computer, imported, built, and activated.

### Increment 9 — Project/session integration

- Save active pack identity and per-project overrides in `.acmxproj` without
  duplicating global pack state.
- Bundle project-local packs and their valid compiled cache during Save Project.
- Include source/resources but omit generated output during Export Project.
- Restore a project pack only after its files and dependencies validate.
- Define precedence among project overrides, per-user pack state, and pack
  defaults.

Acceptance: save/load/export of an ACMX project restores the same pack, controls,
pipeline, and Deep Dream setup without rebuilding unchanged shaders.

### Increment 10 — Hardening, examples, documentation, and release readiness

- Add integration tests for protocol compatibility, transactional failure,
  traversal rejection, duplicate control declarations, intentional repeated
  passes, stale/incompatible caches, and missing capabilities.
- Exercise Linux, macOS, and Windows path and shared-memory behavior.
- Add several curated example packs demonstrating fragment, compute, multipass,
  history, original-frame, audio, MIDI, and optional Deep Dream use.
- Document the schema, authoring workflow, model resolution, distribution, and
  troubleshooting.
- Add migration rules for future schema versions and finalize release notes.

Acceptance: all targeted configurations compile, automated tests pass, examples
load without absolute paths, and the existing full-library workflow is unchanged.

## Risks and controls

- **External shader paths:** Pack activation must canonicalize every path and
  prove it remains below the selected pack root.
- **Partial live updates:** Pipeline, uniforms, mappings, and Deep Dream must be
  staged before a single activation sequence is published.
- **IPC compatibility:** The shared-memory protocol must receive a version bump
  in both interface and ACMXVK, with size assertions and Windows tests updated.
- **Slow model changes:** Deep Dream model loads can be asynchronous, but an old
  pack remains active until the replacement is ready.
- **UI stalls:** Discovery, compilation, icon decoding, copying, and import/export
  run outside the GUI thread with bounded progress reporting.
- **Untrusted packs:** Enforce file-size/count limits, JSON type/range checks,
  bounded images, safe relative paths, and SPIR-V validation before activation.
- **Portable compiled caches:** Treat source as authoritative. Accept exported
  SPIR-V only when ABI, Vulkan environment, source/include hashes, compiler, and
  build compatibility metadata all match; otherwise rebuild locally.

## Progress log

| Date | Increment | Status | Work completed |
| --- | --- | --- | --- |
| 2026-09-22 | Planning | Complete | Reviewed existing multipass, custom-uniform, resource-reflection, Deep Dream, MIDI/audio, project-copy, and cross-platform shared-memory paths. Defined the version 1 direction and ten-increment implementation plan. |
| 2026-09-22 | 1 | Complete (baseline) | Added the version 1 value model, strict portable parser, cross-platform path/range/count validation, format documentation, valid/invalid fixtures, and focused automated tests. CMake and Pcons include the parser; the main executable and all five core portable test groups build and pass. A plan correction now requires intentional duplicate pass paths to be preserved before Increment 4. |
| 2026-09-22 | 2 | Complete (baseline) | Added bounded multi-root discovery, duplicate-ID and missing-icon diagnostics, shared compiler extraction, pack-local incremental SPIR-V caches, include/manifest staleness checks, temporary cleanup, stage/resource validation, and focused build fixtures. Verified a three-pass incremental build and all ten configured tests. Full ABI/Vulkan/hash/compiler compatibility metadata remains required before compiled caches are portable export artifacts. |
| 2026-09-22 | 3 | Complete | Added shared-memory protocol version 12 with sequenced effect-pack requests, validated pack-local cache loading, transactional live activation and rollback, existing-pipeline integration, resource provisioning, crossfades, friendly-control defaults, ordinary-workflow isolation, protocol-layout coverage, and stale-cache activation tests. ACMXVK, the Qt interface, and all eleven configured tests build and pass. |
| 2026-09-22 | Plan revision | Complete | Clarified that pass order may intentionally contain repeated shader paths, kept control IDs/uniform declarations unique, moved Effect Pack MIDI semantics above device-specific CC mappings, and defined the compatibility metadata required for safely exporting compiled SPIR-V caches. |
| 2026-09-22 | 4 | Not started | — |
| 2026-09-22 | 5 | Not started | — |
| 2026-09-22 | 6 | Not started | — |
| 2026-09-22 | 7 | Not started | — |
| 2026-09-22 | 8 | Not started | — |
| 2026-09-22 | 9 | Not started | — |
| 2026-09-22 | 10 | Not started | — |

## Current status

Increment 3 is complete. Protocol version 12, validated external-pack cache
loading, transactional live activation/rollback, renderer resource provisioning,
crossfades, and restoration of the ordinary shader workflow are implemented and
verified. The ACMXVK and Qt interface builds pass, as do all eleven configured
tests.

Before or as part of Increment 4, the parser/build validation must be corrected
to preserve intentional duplicate pass paths; the current Increment 1 parser
still rejects them. Cache compatibility metadata also remains a planned
follow-up before compiled caches are treated as portable export artifacts.
Increment 4 is next: the modeless icon-based Effect Pack browser and its
asynchronous discovery/build/activation workflow.
