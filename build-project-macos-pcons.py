#!/usr/bin/env python3
"""Build the macOS/MoltenVK stack with native Pcons build descriptions.

The script is deliberately self-contained: it installs the Homebrew
dependencies, checks out libmx2 and MXVK if necessary, then stages every
project under one local prefix.  It never uses sudo.

Environment overrides mirror build-project-macos-cmake.sh:
  ACMX_MACOS_BUILD_DIRECTORY, ACMX_MACOS_INSTALL_PREFIX,
  LIBMX2_SOURCE_DIR, MXVK_SOURCE_DIR.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MXVK_REPOSITORY = "https://github.com/lostjared/MXVK.git"
LIBMX2_REPOSITORY = "https://github.com/lostjared/libmx2.git"

BREW_PACKAGES = (
    "git",
    "ninja",
    "pkgconf",
    "uv",
    "curl",
    "unzip",
    "ffmpeg",
    "opencv",
    "sdl2",
    "sdl2_ttf",
    "sdl2_mixer",
    "sdl2_image",
    "glew",
    "sdl3",
    "sdl3_ttf",
    "sdl3_mixer",
    "qt",
    "glm",
    "eigen",
    "fmt",
    "rtaudio",
    "rtmidi",
    "jpeg-turbo",
    "libpng",
    "libtiff",
    "webp",
    "yaml-cpp",
    "zlib",
    "freetype",
    "fontconfig",
    "vulkan-headers",
    "vulkan-loader",
    "molten-vk",
    "shaderc",
    "glslang",
)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=environment, check=True)


def require_command(name: str, environment: dict[str, str] | None = None) -> None:
    path = None if environment is None else environment.get("PATH")
    if shutil.which(name, path=path) is None:
        raise RuntimeError(f"required command was not found: {name}")


def find_opencv_package(environment: dict[str, str], dry_run: bool) -> str:
    """Homebrew has used both opencv4.pc and opencv5.pc across releases."""
    if dry_run:
        return "opencv5"
    for package in ("opencv5", "opencv4"):
        if subprocess.run(
            ["pkg-config", "--exists", package], env=environment, check=False
        ).returncode == 0:
            return package
    raise RuntimeError("Homebrew OpenCV did not provide an opencv5.pc or opencv4.pc package")


def checkout(directory: Path, repository: str, required_file: Path, dry_run: bool) -> None:
    if (directory / ".git").is_dir():
        print(f"Using existing checkout: {directory}")
    elif directory.exists():
        raise RuntimeError(f"source directory exists but is not a Git checkout: {directory}")
    else:
        print(f"Cloning {repository} into {directory}...")
        run(["git", "clone", repository, str(directory)], dry_run=dry_run)
    if not dry_run and not required_file.is_file():
        raise RuntimeError(f"checkout is missing required file: {required_file}")


def pcons_build(
    source_dir: Path,
    build_dir: Path,
    install_prefix: Path,
    options: list[str],
    *,
    environment: dict[str, str],
    dry_run: bool,
) -> None:
    command = [
        "uvx",
        "--from",
        "pcons>=0.24",
        "pcons",
        "-B",
        str(build_dir),
        "--reconfigure",
        "VARIANT=release",
        f"PCONS_INSTALL_PREFIX={install_prefix}",
        f"PCONS_FINAL_PREFIX={install_prefix}",
        *options,
        "all",
        "install",
    ]
    run(command, cwd=source_dir, environment=environment, dry_run=dry_run)


def brew_environment() -> dict[str, str]:
    """Expose non-default Homebrew pkg-config directories to Pcons."""
    brew_prefix = Path(
        subprocess.check_output(["brew", "--prefix"], text=True).strip()
    )
    formulae = (
        "ffmpeg", "fontconfig", "freetype", "jpeg-turbo", "libpng", "libtiff",
        "opencv", "qt", "rtaudio", "rtmidi", "sdl2", "sdl2_mixer", "sdl2_ttf",
        "sdl3", "sdl3_mixer", "sdl3_ttf", "shaderc", "vulkan-loader", "webp",
        "yaml-cpp", "zlib",
    )
    package_dirs = [brew_prefix / "lib" / "pkgconfig", brew_prefix / "share" / "pkgconfig"]
    for formula in formulae:
        result = subprocess.run(
            ["brew", "--prefix", formula], text=True, capture_output=True, check=False
        )
        if result.returncode == 0:
            prefix = Path(result.stdout.strip())
            package_dirs.extend([prefix / "lib" / "pkgconfig", prefix / "share" / "pkgconfig"])
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [str(brew_prefix / "bin"), str(brew_prefix / "sbin"), environment.get("PATH", "")]
    )
    valid_dirs = [str(directory) for directory in package_dirs if directory.is_dir()]
    environment["PKG_CONFIG_PATH"] = os.pathsep.join(
        valid_dirs + [environment.get("PKG_CONFIG_PATH", "")]
    )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, help="Pcons build root")
    parser.add_argument("--prefix", type=Path, help="local installation prefix")
    parser.add_argument("--libmx2-source", type=Path, help="existing libmx2 checkout")
    parser.add_argument("--mxvk-source", type=Path, help="existing MXVK checkout")
    parser.add_argument("--skip-brew", action="store_true", help="do not install Homebrew packages")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args()


def require_outside_checkout(path: Path, label: str) -> None:
    """Avoid Pcons non-relocatable-build warnings from embedded local paths."""
    try:
        path.relative_to(ROOT_DIR)
    except ValueError:
        return
    raise RuntimeError(
        f"{label} must be outside the checkout to keep build.ninja relocatable: {path}"
    )


def main() -> int:
    args = parse_args()
    if platform.system() != "Darwin":
        print("error: build-project-macos-pcons.py must be run on macOS", file=sys.stderr)
        return 1
    if platform.machine() not in {"arm64", "x86_64"}:
        print(f"error: unsupported macOS architecture: {platform.machine()}", file=sys.stderr)
        return 1
    if not args.skip_brew:
        require_command("brew")
        print("Installing required Homebrew packages...")
        run(["brew", "install", *BREW_PACKAGES], dry_run=args.dry_run)
    environment = brew_environment()
    for command in ("git", "uvx", "glslc", "pkg-config"):
        require_command(command, environment)
    opencv_package = find_opencv_package(environment, args.dry_run)

    build_dir = args.build_dir or Path(
        os.environ.get(
            "ACMX_MACOS_BUILD_DIRECTORY", ROOT_DIR.parent / f"{ROOT_DIR.name}-macos-pcons-build"
        )
    )
    install_prefix = args.prefix or Path(
        os.environ.get(
            "ACMX_MACOS_INSTALL_PREFIX", ROOT_DIR.parent / f"{ROOT_DIR.name}-macos-pcons-prefix"
        )
    )
    libmx2_dir = args.libmx2_source or Path(
        os.environ.get("LIBMX2_SOURCE_DIR", ROOT_DIR.parent / "libmx2")
    )
    mxvk_dir = args.mxvk_source or Path(
        os.environ.get("MXVK_SOURCE_DIR", ROOT_DIR.parent / "MXVK")
    )
    build_dir = build_dir.expanduser().resolve()
    install_prefix = install_prefix.expanduser().resolve()
    libmx2_dir = libmx2_dir.expanduser().resolve()
    mxvk_dir = mxvk_dir.expanduser().resolve()
    require_outside_checkout(build_dir, "--build-dir")
    require_outside_checkout(install_prefix, "--prefix")

    checkout(libmx2_dir, LIBMX2_REPOSITORY, libmx2_dir / "libmx" / "CMakeLists.txt", args.dry_run)
    checkout(mxvk_dir, MXVK_REPOSITORY, mxvk_dir / "CMakeLists.txt", args.dry_run)
    if not args.dry_run:
        build_dir.mkdir(parents=True, exist_ok=True)
        install_prefix.mkdir(parents=True, exist_ok=True)

    print("Building libmx2 with Pcons...")
    pcons_build(
        libmx2_dir,
        build_dir / "libmx2",
        install_prefix,
        ["OPENGL=1", "VULKAN=0", "MOLTEN=0", "MIXER=1", "JPEG=1", "EXAMPLES=0"],
        environment=environment,
        dry_run=args.dry_run,
    )
    print("Building MXVK with Pcons...")
    pcons_build(
        mxvk_dir,
        build_dir / "mxvk",
        install_prefix,
        [
            "PREFIX=" + str(install_prefix),
            "EXAMPLES=0",
            "WITH_CUDA=OFF",
            "CV=ON",
            "VALIDATION=OFF",
            "OPENCV_PACKAGE=" + opencv_package,
        ],
        environment=environment,
        dry_run=args.dry_run,
    )
    print("Building ACMX2 with Pcons...")
    pcons_build(
        ROOT_DIR,
        build_dir / "acmx2",
        install_prefix,
        [
            "PREFIX=" + str(install_prefix),
            "OPENCV_PACKAGE=" + opencv_package,
            "AUDIO=1",
            "MIDI=1",
            "WEBP=1",
            "TIFF=1",
            "DNN=1",
            "WITH_CUDA=0",
        ],
        environment=environment,
        dry_run=args.dry_run,
    )
    print("Building ACMXVK with Pcons...")
    pcons_build(
        ROOT_DIR / "ACMXVK",
        build_dir / "acmxvk",
        install_prefix,
        [
            "PREFIX=" + str(install_prefix),
            "OPENCV_PACKAGE=" + opencv_package,
            "AUDIO=1",
            "MIDI=1",
            "WEBP=1",
            "TIFF=1",
            "DNN=1",
            "VALIDATION=0",
            "WITH_CUDA=0",
        ],
        environment=environment,
        dry_run=args.dry_run,
    )
    print("Building the Qt interface with Pcons...")
    pcons_build(
        ROOT_DIR / "ACMX2" / "interface",
        build_dir / "interface",
        install_prefix,
        [],
        environment=environment,
        dry_run=args.dry_run,
    )

    print("\nmacOS Pcons build complete.")
    print(f"  Build directory: {build_dir}")
    print(f"  Install prefix:  {install_prefix}")
    print(f"  libmx2 source:   {libmx2_dir}")
    print(f"  MXVK source:     {mxvk_dir}")
    print("\nAdd the local prefix to PATH before launching installed executables:")
    print(f'  export PATH="{install_prefix}/bin:$PATH"')
    print("\nPersist that PATH setting for future zsh sessions:")
    print(f"  echo 'export PATH=\"{install_prefix}/bin:$PATH\"' >> ~/.zshenv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
