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
    STABLE_DIFFUSION=0|1, DEEP_DREAM=0|1,
    VALIDATION=0|1, WITH_CUDA=0|1, VARIANT=release|debug,
    PREFIX=<dependency-prefix>, TORCH_PREFIX=/opt/libtorch,
    CUDA_PREFIX=/opt/cuda, PCONS_INSTALL_PREFIX=<stage-prefix>,
    PCONS_FINAL_PREFIX=<installed-prefix>.

Stable Diffusion requires libcurl and jsoncpp and supports Windows when a
compatible sd-server.exe is available. Deep Dream is supported on Linux with
CUDA-enabled LibTorch and CUDA-enabled OpenCV; it is independent of the
CMake-only acidcam-gpu CUDA-filter configuration.
"""

import os
import shutil
from pathlib import Path

from pcons import ImportedTarget, PackageDescription, Project, Target, find_c_toolchain, get_platform, get_var
from pcons.core.subst import PathToken

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


def windows_icon_source(name: str, env):
    if not platform.is_windows:
        return []
    rc_file = project_dir.parent / "ACMX2" / "interface" / "win-icon.rc"
    icon_file = project_dir.parent / "ACMX2" / "interface" / "win-icon.ico"
    if env.has_tool("rc"):
        return [rc_file]
    windres = shutil.which("windres")
    if windres is None:
        raise SystemExit("A Windows resource compiler is required to embed win-icon.ico")
    resource = project.Command(
        f"{name}-windows-icon",
        env,
        target=project_dir / project.build_dir / f"{name}-icon{platform.object_suffix}",
        source=rc_file,
        command=[windres, "--input", "$SOURCE", "--output", "$TARGET", "--output-format", "coff"],
    )
    resource.depends(icon_file)
    return [resource]


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


def imported_package(name: str, include_dirs: list[Path], library_dir: Path, libraries: list[str]) -> ImportedTarget:
    """Create a target for a dependency distributed without a pkg-config file."""
    return ImportedTarget.from_package(
        PackageDescription(
            name=name,
            include_dirs=[str(directory) for directory in include_dirs],
            library_dirs=[str(library_dir)],
            libraries=libraries,
            compile_flags=[flag for directory in include_dirs for flag in ("-isystem", str(directory))],
        )
    )


def find_cuda_runtime() -> ImportedTarget:
    """Locate CUDA's runtime library for a CUDA-enabled LibTorch install."""
    cuda_root = Path(get_var("CUDA_PREFIX", "/opt/cuda")).expanduser()
    layouts = (
        (cuda_root / "include", cuda_root / "lib64"),
        (cuda_root / "include", cuda_root / "lib"),
        (cuda_root / "targets" / "x86_64-linux" / "include", cuda_root / "targets" / "x86_64-linux" / "lib"),
    )
    for include_dir, library_dir in layouts:
        if (include_dir / "cuda_runtime.h").is_file() and (library_dir / "libcudart.so").exists():
            return imported_package("cuda-runtime", [include_dir], library_dir, ["cudart", "nppicc", "nppidei", "nppc"])
    raise SystemExit(f"DEEP_DREAM=1 requires CUDA under {cuda_root}. Set CUDA_PREFIX=/path/to/cuda.")


def find_libtorch() -> ImportedTarget:
    """Locate CUDA-enabled LibTorch from an archive or a system package."""
    configured_prefix = get_var("TORCH_PREFIX", "")
    roots = [Path(configured_prefix).expanduser()] if configured_prefix else [Path("/opt/libtorch"), Path("/usr"), Path("/usr/local")]
    libraries = ["torch", "torch_cpu", "c10", "torch_cuda", "c10_cuda"]
    checked = []
    for torch_root in roots:
        include_dir = torch_root / "include"
        api_include_dir = include_dir / "torch" / "csrc" / "api" / "include"
        header_dirs = [api_include_dir]
        if include_dir not in (Path("/usr/include"), Path("/usr/local/include")):
            header_dirs.insert(0, include_dir)
        has_torch_header = (include_dir / "torch" / "torch.h").is_file() or (api_include_dir / "torch" / "torch.h").is_file()
        library_dirs = [torch_root / "lib", torch_root / "lib64", torch_root / "lib" / "x86_64-linux-gnu"]
        for library_dir in library_dirs:
            checked.append(str(library_dir))
            if not has_torch_header or not library_dir.is_dir():
                continue
            missing = [library for library in libraries if not any((library_dir / f"lib{library}{suffix}").exists() for suffix in (".so", ".dylib", ".a"))]
            if not missing:
                return imported_package("LibTorch", header_dirs, library_dir, libraries)
    if configured_prefix:
        raise SystemExit(f"DEEP_DREAM=1 requires CUDA-enabled LibTorch under {roots[0]}. Set TORCH_PREFIX=/path/to/libtorch.")
    raise SystemExit("DEEP_DREAM=1 requires CUDA-enabled LibTorch. Searched " + ", ".join(checked) + ". Set TORCH_PREFIX=/path/to/libtorch.")


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
with_stable_diffusion = option("STABLE_DIFFUSION")
with_deep_dream = option("DEEP_DREAM")
with_validation = option("VALIDATION")
with_cuda = option("WITH_CUDA")

if with_cuda and platform.is_macos:
    raise SystemExit("WITH_CUDA=1 is unavailable with MoltenVK on macOS.")
if with_deep_dream and not platform.is_linux:
    raise SystemExit("DEEP_DREAM=1 currently requires Linux, CUDA, and CUDA-enabled LibTorch.")

project = Project("acmxvk", root_dir=project_dir)
env = project.Environment(toolchain=find_c_toolchain())
env.cxx.set_standard(20)
env.set_variant(get_var("VARIANT", "release"))
env.cxx.flags.extend(["-Wall", "-Wextra", "-Wpedantic"])
if platform.is_linux:
    env.cxx.flags.append("-fPIC")
if with_validation:
    env.cxx.defines.append("ENABLE_VALIDATION")
if platform.is_macos:
    # Match ACMXVK_USE_MOLTENVK=ON in CMake and keep Vulkan beta-extension
    # declarations consistent with the installed MXVK package.
    env.cxx.defines.extend(["MXVK_USE_MOLTENVK", "VK_ENABLE_BETA_EXTENSIONS"])

mxvk = require_package("mxvk")
mxvk_defines = " ".join(str(define) for define in mxvk.public.defines)
mxvk_with_cuda = "MXVK_CUDA" in mxvk_defines
if mxvk_with_cuda and not with_cuda:
    raise SystemExit(
        "The selected MXVK prefix was built with CUDA, but WITH_CUDA=0 was "
        "requested. Rebuild/install MXVK with WITH_CUDA=OFF, or configure "
        "ACMXVK with WITH_CUDA=1."
    )
if with_cuda and not mxvk_with_cuda:
    raise SystemExit("WITH_CUDA=1 requires an MXVK prefix built with WITH_CUDA=ON.")
ffmpeg = require_package("libavcodec")
for package_name in ("libavformat", "libavutil", "libswscale", "libswresample"):
    ffmpeg.link(require_package(package_name))
opencv = require_package(get_var("OPENCV_PACKAGE", "opencv5"))
if not platform.is_windows:
    opencv.public.system_include_dirs.extend(opencv.public.include_dirs)
    opencv.public.include_dirs.clear()

# ACMXVK deliberately uses the repository MXWrite API. Building it locally
# makes pcons match CMake's ACMXVK_USE_BUNDLED_MXWRITE=ON default.
mxwrite_dir = project_dir.parent / "MXWrite"
if not (mxwrite_dir / "mxwrite.cpp").is_file():
    raise SystemExit(f"Bundled MXWrite source was not found: {mxwrite_dir}")
mxwrite = project.StaticLibrary("mxwrite", env, sources=[mxwrite_dir / "mxwrite.cpp"])
mxwrite.public.include_dirs.append(mxwrite_dir)
mxwrite.link(ffmpeg)

libraries: list[Target] = [mxvk, mxwrite, ffmpeg, opencv, require_package("jsoncpp")]
sources: list[Path] = [
    project_dir / "acmx.cpp",
    project_dir / "main_window.cpp",
    project_dir / "app" / "effect_pack.cpp",
    project_dir / "app" / "effect_pack_build.cpp",
    project_dir / "app" / "interface_client.cpp",
    project_dir / "app" / "media_helpers.cpp",
    project_dir / "app" / "media_utils.cpp",
    project_dir / "app" / "options.cpp",
    project_dir / "app" / "output_paths.cpp",
    project_dir / "app" / "playlist.cpp",
    project_dir / "app" / "resource_paths.cpp",
    project_dir / "app" / "shader_library.cpp",
    project_dir / "app" / "shader_compiler.cpp",
    project_dir / "app" / "snapshot_writer.cpp",
    project_dir / "input_validation.cpp",
]

if platform.is_macos:
    sources.append(project_dir / "app" / "camera_probe_macos.mm")
    env.Framework("AVFoundation")
    env.Framework("CoreMedia")
    env.Framework("Foundation")
elif platform.is_windows:
    sources.append(project_dir / "app" / "camera_probe_windows.cpp")
    env.link.libs.extend(["ole32", "oleaut32", "strmiids"])
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
if with_stable_diffusion:
    sources.append(project_dir / "stable_diffusion.cpp")
    libraries.append(require_package("libcurl"))
    env.cxx.defines.append("ACMXVK_WITH_STABLE_DIFFUSION")
cuda_runtime: ImportedTarget | None = find_cuda_runtime() if with_deep_dream or with_cuda else None
if with_deep_dream:
    sources.extend([project_dir / "deep_dream.cpp", project_dir / "deep_dream_model.cpp"])
    libraries.extend([cuda_runtime, find_libtorch(), require_package("gflags"), require_package("libglog")])
    env.cxx.defines.extend(["ACMXVK_WITH_DEEP_DREAM", "GLOG_USE_GFLAGS", "GLOG_USE_GLOG_EXPORT"])
if mxvk_with_cuda:
    env.cxx.defines.append("ACMXVK_WITH_MXVK_CUDA")
if with_cuda:
    sources.extend([project_dir / "gpu_filters.cpp"])
    libraries.extend([require_package("acidcam-gpu"), cuda_runtime])
    env.cxx.defines.append("ACMXVK_WITH_CUDA")

runtime_dir = (project_dir / project.build_dir / "runtime").resolve()
shader_output_dir = runtime_dir / "shaders"
final_prefix = Path(get_var("PCONS_FINAL_PREFIX", str(project_dir / "dist")))
install_resource_dir = final_prefix / "share" / "acmxvk"
shader_output_dir.mkdir(parents=True, exist_ok=True)

shader_targets: list[Target] = []
shader_outputs: list[Path] = []


def compile_shader(name: str, source: Path, output: Path, flags: str = "") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arguments = ["glslc"]
    if flags:
        arguments.append(flags)
    arguments.extend(["$SOURCE", "-o", "$TARGET"])
    shader_targets.append(
        project.Command(
            name,
            env,
            target=output,
            source=source,
            command=arguments,
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
default_model_output.parent.mkdir(parents=True, exist_ok=True)
overlay_font_output.parent.mkdir(parents=True, exist_ok=True)
resource_targets = [
    project.Command(
        "acmxvk-default-model",
        env,
        target=default_model_output,
        source=project_dir / "models" / "cube.obj",
        command=["cmake", "-E", "copy_if_different", "$SOURCE", "$TARGET"],
    ),
    project.Command(
        "acmxvk-overlay-font",
        env,
        target=overlay_font_output,
        source=project_dir.parent / "ACMX2" / "data" / "font.ttf",
        command=["cmake", "-E", "copy_if_different", "$SOURCE", "$TARGET"],
    ),
]

acmxvk = project.Program("acmxvk", env, sources=[*sources, *windows_icon_source("acmxvk", env)])
acmxvk.private.include_dirs.extend([project_dir, project_dir / "app"])
acmxvk.link(*libraries)
acmxvk.depends(*resource_targets, *shader_targets)

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
    "ACMXVK_BUILD_STABLE_DIFFUSION_UPSCALE_SHADER": shader_output_dir / "sd_upscale.comp.spv",
    "ACMXVK_INSTALL_STABLE_DIFFUSION_UPSCALE_SHADER": install_resource_dir / "shaders" / "sd_upscale.comp.spv",
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
    resolved_path = Path(path).resolve()
    try:
        relative_path = resolved_path.relative_to(project_dir)
    except ValueError:
        path_token = PathToken(
            f'-D{name}="', str(resolved_path), "absolute", '"'
        )
    else:
        path_token = PathToken(
            f'-D{name}="', relative_path.as_posix(), "project", '"'
        )
    acmxvk.private.compile_flags.append(path_token)


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
