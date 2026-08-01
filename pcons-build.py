#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pcons"]
# ///
"""pcons build for acidcam-gpu / ACMX2 (https://github.com/lostjared/acidcam-gpu).

One build description covering the repository's four CMake projects:

    MXWrite/                 -> libmxwrite.a                (FFmpeg encoder)
    ACMX2/                   -> acmx2, audio_transfer       (SDL2/OpenGL engine)
    ACMX2/shader_generator/  -> shader_generator            (needs libcurl)
    acidcam-gpu/             -> libacidcam-gpu, acidcam     (CUDA only)

The Qt6 interface has its own script (ACMX2/interface/pcons-build.py),
mirroring the upstream layout where it is a separate CMake project.

Options (pcons VAR=value, or as environment variables). CMake spellings in
parentheses:

    WITH_CUDA=0|1     (-DWITH_CUDA)          default: on when nvcc is present
    AUDIO=1           (-DAUDIO)              RtAudio live input
    MIDI=1            (-DMIDI)               RtMidi control input
    DNN=1             (-DWITH_OPENCV_DNN)    OpenCV DNN filters (needs yaml-cpp)
    WEBP=1            (-DWEBP)               WebP HDR snapshots
    TIFF=1            (-DTIFF)               16-bit TIFF HDR snapshots
    VARIANT=debug     (-DCMAKE_BUILD_TYPE)   default: release
    PREFIX=<dir>      (-DCMAKE_PREFIX_PATH)  extra dependency search prefix

Everything is found through pkg-config except libmx2
(https://github.com/lostjared/libmx2) and glm, which install CMake config
files but no .pc; those two are located by prefix. Point PREFIX at them when
they are installed somewhere other than /usr/local or the Homebrew prefix.

Usage:
    uvx pcons                  # configure, generate and build
    uvx pcons WITH_CUDA=0      # no CUDA even if nvcc is installed
"""

import os
import shutil
from pathlib import Path

from pcons import (
    ImportedTarget,
    PackageDescription,
    Project,
    find_c_toolchain,
    find_cuda_toolchain,
    get_platform,
    get_var,
)
from pcons.core.node import Node
from pcons.core.target import Target
from pcons.packages.finders import PkgConfigFinder

project_dir = Path(__file__).parent
platform = get_platform()


def option(name: str, default: bool = False) -> bool:
    """Read a boolean build option, mirroring CMake's option()."""
    return get_var(name, "1" if default else "0").lower() in ("1", "on", "true", "yes")


# A prefix given here feeds both pkg-config and the libmx2 search below.
extra_prefixes = [Path(p) for p in (get_var("PREFIX") or "").split(os.pathsep) if p]
if extra_prefixes:
    os.environ["PKG_CONFIG_PATH"] = os.pathsep.join(
        [str(p / "lib" / "pkgconfig") for p in extra_prefixes]
        + [os.environ.get("PKG_CONFIG_PATH", "")]
    )

# =============================================================================
# Project and toolchains
# =============================================================================

project = Project("acidcam-gpu", root_dir=project_dir)
env = project.Environment(toolchain=find_c_toolchain())
env.cxx.set_standard(20)
env.set_variant(get_var("VARIANT", "release"))

cuda = find_cuda_toolchain()
with_cuda = option("WITH_CUDA", default=cuda is not None)
if with_cuda:
    if cuda is None:
        raise SystemExit(
            "WITH_CUDA is on but nvcc was not found. Install the CUDA Toolkit, "
            "or configure with WITH_CUDA=0."
        )
    env.add_toolchain(cuda)
    env.cuda.flags.extend(
        [
            "--std=c++20",
            "--use_fast_math",
            "--forward-unknown-to-host-compiler",
        ]
    )

# CMake applies these to every target in the repo.
env.cxx.flags.extend(["-O3", "-Wall", "-pedantic"])

# =============================================================================
# External packages
# =============================================================================


def package(name: str) -> Target:
    """Required find_package(), narrowed to Target for type checkers."""
    pkg = project.find_package(name)
    assert pkg is not None  # required=True never returns None
    return pkg


ffmpeg = package("libavcodec")
for name in ("libavformat", "libavutil", "libswscale", "libswresample"):
    ffmpeg.link(package(name))

sdl2 = package("sdl2")
sdl2_ttf = package("SDL2_ttf")


def find_opencv(*components: str) -> ImportedTarget:
    """OpenCV, restricted to the modules actually used.

    CMake asks for components and resolves their dependencies; opencv4.pc
    has no notion of components and lists every module, so filter its
    library list down to the transitive set named here.
    """
    pkg = PkgConfigFinder().find("opencv5")
    if pkg is None:
        raise SystemExit("OpenCV not found: install OpenCV development libraries.")
    if not platform.is_windows:
        for include_dir in pkg.include_dirs:
            pkg.compile_flags.extend(["-isystem", include_dir])
        pkg.include_dirs.clear()
    wanted = {f"opencv_{component}" for component in components}
    pkg.libraries = [lib for lib in pkg.libraries if lib in wanted]
    missing = wanted - set(pkg.libraries)
    if missing:
        raise SystemExit(f"OpenCV is missing components: {', '.join(sorted(missing))}")
    return ImportedTarget.from_package(pkg)


opencv_components = ["core", "imgproc", "imgcodecs", "highgui", "videoio"]
if with_cuda:
    opencv_components += ["cudaimgproc", "cudawarping", "cudaarithm"]
if option("DNN"):
    opencv_components.append("dnn")
opencv = find_opencv(*opencv_components)


SEARCH_PREFIXES = extra_prefixes + [
    Path("/usr/local"),
    Path("/opt/homebrew"),
    Path("/usr"),
]


def find_by_prefix(
    name: str,
    probe: str,
    *,
    include_subdir: str = "",
    libraries: tuple[str, ...] = (),
    hint: str = "",
) -> ImportedTarget:
    """Locate a dependency that ships CMake config files but no .pc file.

    Args:
        name: Package name, for diagnostics.
        probe: Header to look for, relative to <prefix>/include.
        include_subdir: Subdirectory of <prefix>/include to put on the
            include path (libmx2 headers are included unqualified).
        libraries: Libraries to link, without the -l.
        hint: Extra text for the not-found message.
    """
    for prefix in SEARCH_PREFIXES:
        if (prefix / "include" / probe).exists():
            libdir = prefix / "lib"
            return ImportedTarget.from_package(
                PackageDescription(
                    name=name,
                    prefix=str(prefix),
                    include_dirs=[str(prefix / "include" / include_subdir)],
                    library_dirs=[str(libdir)] if libraries else [],
                    libraries=list(libraries),
                    # CMake's INSTALL_RPATH, so the binaries run in place.
                    link_flags=(
                        [f"-Wl,-rpath,{libdir}"]
                        if libraries and not platform.is_windows
                        else []
                    ),
                )
            )
    raise SystemExit(
        f"{name} not found (looked for include/{probe} under "
        + ", ".join(str(p) for p in SEARCH_PREFIXES)
        + f").\n{hint}Pass its install prefix as PREFIX=<dir>."
    )


def find_cuda_runtime() -> ImportedTarget:
    """The CUDA runtime, taken from the toolkit that owns nvcc (CUDA::cudart)."""
    nvcc = shutil.which("nvcc")
    assert nvcc is not None  # only called once nvcc has been found
    root = Path(nvcc).resolve().parent.parent
    libdir = root / "lib64" if (root / "lib64").is_dir() else root / "lib"
    return ImportedTarget.from_package(
        PackageDescription(
            name="cudart",
            prefix=str(root),
            include_dirs=[str(root / "include")],
            library_dirs=[str(libdir)],
            libraries=["cudart"],
        )
    )


# libmx2 headers include glm unqualified, so glm is a public dependency of it.
glm = find_by_prefix("glm", "glm/glm.hpp", hint="Install glm (a header-only library). ")
libmx2 = find_by_prefix(
    "libmx2",
    "mx2/mx.hpp",
    include_subdir="mx2",
    libraries=("mx", "mxgl"),
    hint="Build it from https://github.com/lostjared/libmx2. ",
)
libmx2.link(glm)

# OpenGL: a framework on macOS, a plain library elsewhere.
if platform.is_macos:
    env.Framework("OpenGL")
elif platform.is_windows:
    env.link.libs.append("opengl32")
else:
    env.link.libs.append("GL")

if not platform.is_windows:
    env.cxx.flags.append("-pthread")
    env.link.flags.append("-pthread")

# =============================================================================
# MXWrite: FFmpeg encoder library, built in-tree as a private dependency
# =============================================================================

mxwrite = project.StaticLibrary("mxwrite", env, sources=["MXWrite/mxwrite.cpp"])
mxwrite.public.include_dirs.append("MXWrite")
mxwrite.link(ffmpeg)
if with_cuda:
    # PUBLIC in CMake, and it has to stay public: the define changes the
    # layout of class Writer, so every TU including mxwrite.hpp must agree.
    mxwrite.public.defines.append("MXWRITE_HAS_CUDA_COPY=1")
    mxwrite.link(find_cuda_runtime())

# =============================================================================
# acidcam-gpu: CUDA filter library and its standalone player
# =============================================================================

acidcam_gpu: Target | None = None
acidcam: Target | None = None

if with_cuda:
    cuda_env = env.clone()
    cuda_env.cxx.flags.extend(["-march=native", "-ffast-math", "-fomit-frame-pointer"])

    acidcam_gpu = project.SharedLibrary(
        "acidcam-gpu", cuda_env, sources=["acidcam-gpu/src/filters.cu"]
    )
    acidcam_gpu.public.include_dirs.append("acidcam-gpu/include")
    acidcam_gpu.private.include_dirs.append("acidcam-gpu/src")
    acidcam_gpu.link(opencv, mxwrite)

    acidcam = project.Program(
        "acidcam", cuda_env, sources=["acidcam-gpu/app/main_cv.cu"]
    )
    acidcam.link(acidcam_gpu, opencv)

# =============================================================================
# ACMX2: the engine
# =============================================================================

sources: list[str | Path | Node] = ["ACMX2/acmx.cpp", "ACMX2/program.cpp"]
defines = ["WITH_GL"]
libs: list[Target] = [libmx2, opencv, sdl2, sdl2_ttf, mxwrite, ffmpeg]

if acidcam_gpu is not None:
    defines.append("ACMX2_WITH_CUDA")
    libs.append(acidcam_gpu)

if option("AUDIO"):
    sources += ["ACMX2/audio.cpp", "ACMX2/file_audio.cpp"]
    defines.append("AUDIO_ENABLED")
    libs.append(package("rtaudio"))

if option("MIDI"):
    defines.append("MIDI_ENABLED")
    libs.append(package("rtmidi"))

if option("DNN"):
    sources.append("ACMX2/dnn.cpp")
    defines.append("ACMX2_WITH_DNN")
    libs.append(package("yaml-cpp"))

if option("WEBP"):
    defines.append("ACMX2_WITH_WEBP")
    libs.append(package("libwebp"))

if option("TIFF"):
    defines.append("ACMX2_WITH_TIFF")
    libs.append(package("libtiff-4"))

acmx2 = project.Program("acmx2", env, sources=sources)
acmx2.private.defines.extend(defines)
acmx2.link(*libs)

audio_transfer = project.Program(
    "audio_transfer", env, sources=["ACMX2/audio_transfer.cpp"]
)
audio_transfer.link(mxwrite, ffmpeg)

# =============================================================================
# shader_generator: standalone Ollama-backed shader authoring tool
# =============================================================================

shader_generator = project.Program(
    "shader_generator",
    env,
    sources=[
        "ACMX2/shader_generator/gencode.cpp",
        "ACMX2/shader_generator/mx2-ollama.cpp",
    ],
)
shader_generator.private.include_dirs.append("ACMX2/shader_generator")
shader_generator.link(package("libcurl"))

# =============================================================================
# Install
# =============================================================================

programs = [acmx2, audio_transfer, shader_generator]
if acidcam is not None:
    programs.append(acidcam)
project.Install("bin", programs)
project.InstallDir("share/acmx2", "ACMX2/data")
if acidcam_gpu is not None:
    project.Install("lib", [acidcam_gpu])
