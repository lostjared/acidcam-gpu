#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pcons>=0.24"]
# ///
"""Native pcons build for ACMXVK.

ACMXVK consumes an installed MXVK pcons package. Build/install MXVK first,
then pass its prefix here when it is not discoverable through pkg-config:

    pcons -B build/pcons PREFIX=/opt/mxvk \
        PCONS_INSTALL_PREFIX=/opt/acmxvk \
        PCONS_FINAL_PREFIX=/opt/acmxvk all install

Options mirror the CMake project where applicable:

    AUDIO=0|1, MIDI=0|1, WEBP=0|1, TIFF=0|1, DNN=0|1,
    VALIDATION=0|1, WITH_CUDA=0|1, VARIANT=release|debug,
    PREFIX=<dependency-prefix>, PCONS_INSTALL_PREFIX=<stage-prefix>,
    PCONS_FINAL_PREFIX=<installed-prefix>.

The CUDA filter configuration currently remains CMake-only because it requires
a matching CUDA-enabled MXVK and installed acidcam-gpu library.
"""

import os
import shlex
from pathlib import Path

from pcons import Project, Target, find_c_toolchain, get_platform, get_var

project_dir = Path(__file__).parent.resolve()
platform = get_platform()


def configure_homebrew_paths() -> None:
    """Make direct macOS Pcons invocations see Homebrew's keg-only packages."""
    if not platform.is_macos:
        return
    package_dirs: list[str] = []
    binary_dirs: list[str] = []
    for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
        package_dirs.extend(
            str(path)
            for path in (prefix / "lib" / "pkgconfig", prefix / "share" / "pkgconfig")
            if path.is_dir()
        )
        opt_dir = prefix / "opt"
        if opt_dir.is_dir():
            package_dirs.extend(str(path) for path in opt_dir.glob("*/lib/pkgconfig") if path.is_dir())
            package_dirs.extend(str(path) for path in opt_dir.glob("*/share/pkgconfig") if path.is_dir())
            binary_dirs.extend(str(path) for path in opt_dir.glob("*/bin") if path.is_dir())
    if package_dirs:
        os.environ["PKG_CONFIG_PATH"] = os.pathsep.join(
            package_dirs + [os.environ.get("PKG_CONFIG_PATH", "")]
        )
    if binary_dirs:
        os.environ["PATH"] = os.pathsep.join(binary_dirs + [os.environ.get("PATH", "")])


configure_homebrew_paths()


def option(name: str, default: bool = False) -> bool:
    """Read an ON/OFF pcons option."""
    return get_var(name, "1" if default else "0").lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def require_package(name: str) -> Target:
    """Find a required pkg-config package with a concise error."""
    package = project.find_package(name)
    if package is None:
        raise SystemExit(
            f"Missing dependency: {name}. Build/install MXVK first and pass "
            "its prefix with PREFIX=/path/to/prefix when necessary."
        )
    return package


def quoted(path: Path | str) -> str:
    return shlex.quote(str(path))


extra_prefixes = [
    Path(prefix)
    for prefix in (get_var("PREFIX") or "").split(os.pathsep)
    if prefix
]
if extra_prefixes:
    os.environ["PKG_CONFIG_PATH"] = os.pathsep.join(
        [str(prefix / "lib" / "pkgconfig") for prefix in extra_prefixes]
        + [os.environ.get("PKG_CONFIG_PATH", "")]
    )

with_audio = option("AUDIO")
with_midi = option("MIDI")
with_webp = option("WEBP")
with_tiff = option("TIFF")
with_dnn = option("DNN")
with_validation = option("VALIDATION")
with_cuda = option("WITH_CUDA")

if with_cuda and platform.is_macos:
    raise SystemExit("WITH_CUDA=1 is unavailable with MoltenVK on macOS.")

project = Project("acmxvk", root_dir=project_dir)
env = project.Environment(toolchain=find_c_toolchain())
env.cxx.set_standard(20)
env.set_variant(get_var("VARIANT", "release"))
env.cxx.flags.extend(["-Wall", "-Wextra", "-Wpedantic"])
if platform.is_linux:
    env.cxx.flags.append("-fPIC")
if with_validation:
    env.cxx.defines.append("ENABLE_VALIDATION")

mxvk = require_package("mxvk")
mxvk_defines = " ".join(str(define) for define in mxvk.public.defines)
if "MXVK_CUDA" in mxvk_defines and not with_cuda:
    raise SystemExit(
        "The selected MXVK prefix was built with CUDA, but WITH_CUDA=0 was "
        "requested. Rebuild/install MXVK with WITH_CUDA=OFF, or use the "
        "matching CUDA ACMXVK CMake build."
    )
ffmpeg = require_package("libavcodec")
for package_name in ("libavformat", "libavutil", "libswscale", "libswresample"):
    ffmpeg.link(require_package(package_name))
opencv = require_package(get_var("OPENCV_PACKAGE", "opencv5"))

# ACMXVK deliberately uses the repository MXWrite API. Building it locally
# makes pcons match CMake's ACMXVK_USE_BUNDLED_MXWRITE=ON default.
mxwrite_dir = project_dir.parent / "MXWrite"
if not (mxwrite_dir / "mxwrite.cpp").is_file():
    raise SystemExit(f"Bundled MXWrite source was not found: {mxwrite_dir}")
mxwrite = project.StaticLibrary("mxwrite", env, sources=[mxwrite_dir / "mxwrite.cpp"])
mxwrite.public.include_dirs.append(mxwrite_dir)
mxwrite.link(ffmpeg)

libraries: list[Target] = [mxvk, mxwrite, ffmpeg, opencv]
sources: list[Path] = [
    project_dir / "acmx.cpp",
    project_dir / "main_window.cpp",
    project_dir / "app" / "interface_client.cpp",
    project_dir / "app" / "media_helpers.cpp",
    project_dir / "app" / "media_utils.cpp",
    project_dir / "app" / "options.cpp",
    project_dir / "app" / "output_paths.cpp",
    project_dir / "app" / "playlist.cpp",
    project_dir / "app" / "resource_paths.cpp",
    project_dir / "app" / "shader_library.cpp",
    project_dir / "app" / "snapshot_writer.cpp",
    project_dir / "input_validation.cpp",
]

if platform.is_macos:
    sources.append(project_dir / "app" / "camera_probe_macos.mm")
    env.Framework("AVFoundation")
    env.Framework("CoreMedia")
    env.Framework("Foundation")
else:
    sources.append(project_dir / "app" / "camera_probe.cpp")

if with_audio:
    sources.extend([project_dir / "audio.cpp", project_dir / "file_audio.cpp"])
    libraries.append(require_package("rtaudio"))
    env.cxx.defines.append("AUDIO_ENABLED")
if with_midi:
    sources.append(project_dir / "midi.cpp")
    libraries.append(require_package("rtmidi"))
    env.cxx.defines.append("MIDI_ENABLED")
if with_webp:
    libraries.append(require_package("libwebp"))
    env.cxx.defines.append("ACMXVK_WITH_WEBP")
if with_tiff:
    libraries.append(require_package("libtiff-4"))
    env.cxx.defines.append("ACMXVK_WITH_TIFF")
if with_dnn:
    sources.append(project_dir / "edge_dnn.cpp")
    env.cxx.defines.append("ACMXVK_WITH_DNN")
if with_cuda:
    raise SystemExit(
        "WITH_CUDA=1 is not yet supported by the ACMXVK pcons target. "
        "Use CMake for CUDA ACMXVK builds."
    )

runtime_dir = (project_dir / project.build_dir / "runtime").resolve()
shader_output_dir = runtime_dir / "shaders"
final_prefix = Path(get_var("PCONS_FINAL_PREFIX", str(project_dir / "dist")))
install_resource_dir = final_prefix / "share" / "acmxvk"

shader_targets: list[Target] = []
shader_outputs: list[Path] = []


def compile_shader(name: str, source: Path, output: Path, flags: str = "") -> None:
    shader_targets.append(
        project.Command(
            name,
            env,
            target=output,
            source=source,
            command=(
                f"mkdir -p {quoted(output.parent)} && glslc {flags} "
                f"{quoted(source)} -o {quoted(output)}"
            ),
        )
    )
    shader_outputs.append(output)


for shader in sorted((project_dir / "shaders").glob("*")):
    # These sources intentionally require one of the specialized HDR defines
    # emitted below, so compiling their undecorated forms would fail.
    if shader.name in ("hdr_preview.frag", "hdr_transfer.frag"):
        continue
    if shader.suffix in (".vert", ".frag", ".comp"):
        compile_shader(
            f"shader-{shader.name}", shader, shader_output_dir / f"{shader.name}.spv"
        )

# CMake emits HDR variants from the same sources with different defines.
for name, define in (
    ("pq_decode", "ACMXVK_HDR_PQ_DECODE"),
    ("pq_encode", "ACMXVK_HDR_PQ_ENCODE"),
    ("hlg_decode", "ACMXVK_HDR_HLG_DECODE"),
    ("hlg_encode", "ACMXVK_HDR_HLG_ENCODE"),
):
    compile_shader(
        f"shader-hdr-{name}",
        project_dir / "shaders" / "hdr_transfer.frag",
        shader_output_dir / f"hdr_{name}.frag.spv",
        f"-D{define}=1",
    )
for name in ("pq", "hlg"):
    compile_shader(
        f"shader-hdr-preview-{name}",
        project_dir / "shaders" / "hdr_preview.frag",
        shader_output_dir / f"hdr_preview_{name}.frag.spv",
        f"-DACMXVK_HDR_PREVIEW_{name.upper()}=1",
    )
compile_shader(
    "shader-compute-hdr",
    project_dir / "shaders" / "compute_test.comp",
    shader_output_dir / "compute_test_hdr.comp.spv",
    "-DACMXVK_HDR_COMPUTE=1",
)
for shader in sorted((project_dir / "shaders" / "xfade").glob("xfade_*.glsl")):
    compile_shader(
        f"shader-xfade-{shader.stem}",
        shader,
        shader_output_dir / "xfade" / f"{shader.stem}.frag.spv",
        "-fshader-stage=fragment",
    )

default_model_output = runtime_dir / "models" / "cube.obj"
overlay_font_output = runtime_dir / "data" / "font.ttf"
resource_targets = [
    project.Command(
        "acmxvk-default-model",
        env,
        target=default_model_output,
        source=project_dir / "models" / "cube.obj",
        command=(
            f"mkdir -p {quoted(default_model_output.parent)} && "
            f"cp {quoted(project_dir / 'models' / 'cube.obj')} {quoted(default_model_output)}"
        ),
    ),
    project.Command(
        "acmxvk-overlay-font",
        env,
        target=overlay_font_output,
        source=project_dir.parent / "ACMX2" / "data" / "font.ttf",
        command=(
            f"mkdir -p {quoted(overlay_font_output.parent)} && "
            f"cp {quoted(project_dir.parent / 'ACMX2' / 'data' / 'font.ttf')} {quoted(overlay_font_output)}"
        ),
    ),
]

acmxvk = project.Program("acmxvk", env, sources=sources)
acmxvk.private.include_dirs.extend([project_dir, project_dir / "app"])
acmxvk.link(*libraries)
acmxvk.add_dependency(*resource_targets, *shader_targets)

build_defines = {
    "ACMXVK_BUILD_RESOURCE_DIRECTORY": runtime_dir,
    "ACMXVK_INSTALL_RESOURCE_DIRECTORY": install_resource_dir,
    "ACMXVK_BUILD_SPRITE_VERTEX_SHADER": shader_output_dir / "sprite.vert.spv",
    "ACMXVK_INSTALL_SPRITE_VERTEX_SHADER": install_resource_dir / "shaders" / "sprite.vert.spv",
    "ACMXVK_BUILD_ECHO_CACHE_SHADER": shader_output_dir / "echo_cache.frag.spv",
    "ACMXVK_INSTALL_ECHO_CACHE_SHADER": install_resource_dir / "shaders" / "echo_cache.frag.spv",
    "ACMXVK_BUILD_FLIP_SHADER": shader_output_dir / "flip.frag.spv",
    "ACMXVK_INSTALL_FLIP_SHADER": install_resource_dir / "shaders" / "flip.frag.spv",
    "ACMXVK_BUILD_PASSTHROUGH_SHADER": shader_output_dir / "passthrough.frag.spv",
    "ACMXVK_INSTALL_PASSTHROUGH_SHADER": install_resource_dir / "shaders" / "passthrough.frag.spv",
    "ACMXVK_BUILD_HDR_TRANSFER_DIRECTORY": shader_output_dir,
    "ACMXVK_INSTALL_HDR_TRANSFER_DIRECTORY": install_resource_dir / "shaders",
    "ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER": shader_output_dir / "human_composite.frag.spv",
    "ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER": install_resource_dir / "shaders" / "human_composite.frag.spv",
    "ACMXVK_BUILD_MODEL_VERTEX_SHADER": shader_output_dir / "model.vert.spv",
    "ACMXVK_INSTALL_MODEL_VERTEX_SHADER": install_resource_dir / "shaders" / "model.vert.spv",
    "ACMXVK_BUILD_MODEL_FRAGMENT_SHADER": shader_output_dir / "model.frag.spv",
    "ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER": install_resource_dir / "shaders" / "model.frag.spv",
    "ACMXVK_BUILD_DEFAULT_MODEL": default_model_output,
    "ACMXVK_INSTALL_DEFAULT_MODEL": install_resource_dir / "models" / "cube.obj",
    "ACMXVK_BUILD_OVERLAY_FONT": overlay_font_output,
    "ACMXVK_INSTALL_OVERLAY_FONT": install_resource_dir / "data" / "font.ttf",
    "ACMXVK_BUILD_CROSSFADE_DIRECTORY": shader_output_dir / "xfade",
    "ACMXVK_INSTALL_CROSSFADE_DIRECTORY": install_resource_dir / "shaders" / "xfade",
}
for name, path in build_defines.items():
    acmxvk.private.defines.append(f'{name}="{path}"')


def install_tree(destination: str, source_dir: Path) -> list[Target]:
    """Install a directory's contents without adding its basename twice."""
    files_by_destination: dict[Path, list[Path]] = {}
    for source in sorted(source_dir.rglob("*")):
        if source.is_file():
            relative_parent = source.relative_to(source_dir).parent
            target_dir = Path(destination) / relative_parent
            files_by_destination.setdefault(target_dir, []).append(source)
    return [
        project.Install(str(target_dir), sources)
        for target_dir, sources in files_by_destination.items()
    ]


installed: list[Target] = [
    project.Install("bin", [acmxvk]),
    project.Install(
        "share/acmxvk/shaders",
        [output for output in shader_outputs if output.parent == shader_output_dir]
        + [project_dir / "shaders" / "library.json"],
    ),
    project.Install(
        "share/acmxvk/shaders/xfade",
        [output for output in shader_outputs if output.parent == shader_output_dir / "xfade"],
    ),
    project.Install("share/acmxvk/data", [project_dir.parent / "ACMX2" / "data" / "font.ttf"]),
]
installed.extend(install_tree("share/acmxvk/playlists", project_dir / "playlists"))
installed.extend(install_tree("share/acmxvk/midi-examples", project_dir / "midi-examples"))
installed.extend(install_tree("share/acmxvk/models", project_dir / "models"))
project.Alias("install", *installed)
