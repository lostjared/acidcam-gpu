#!/usr/bin/env python3
"""Build the Acid Cam projects with Pcons without installing system packages.

This is a Linux-first, package-manager-independent counterpart to
build-project-macos-pcons.py. It works on Linux and can also be used on macOS
when the required compilers, Vulkan loader/MoltenVK, Qt, FFmpeg, OpenCV, SDL,
and optional feature libraries are already installed and discoverable through
PATH and pkg-config.

The script installs only the projects it builds into a local prefix; it never
uses a package manager or sudo.
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


def run(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required command was not found: {name}")


def find_opencv_package(dry_run: bool) -> str:
    """Accept the package name exposed by either OpenCV 4 or OpenCV 5."""
    if dry_run:
        return "opencv5"
    for package in ("opencv5", "opencv4"):
        if subprocess.run(["pkg-config", "--exists", package], check=False).returncode == 0:
            return package
    raise RuntimeError("pkg-config could not find OpenCV (opencv5 or opencv4)")


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
    jobs: int | None,
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
    ]
    if jobs is not None:
        command.extend(["-j", str(jobs)])
    command.extend(
        [
            "VARIANT=release",
            f"PCONS_INSTALL_PREFIX={install_prefix}",
            f"PCONS_FINAL_PREFIX={install_prefix}",
            *options,
            "all",
            "install",
        ]
    )
    run(command, cwd=source_dir, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, help="Pcons build root")
    parser.add_argument("--prefix", type=Path, help="local installation prefix")
    parser.add_argument("--libmx2-source", type=Path, help="existing libmx2 checkout")
    parser.add_argument("--mxvk-source", type=Path, help="existing MXVK checkout")
    parser.add_argument("--jobs", type=int, help="parallel Pcons job count")
    parser.add_argument("--skip-clone", action="store_true", help="require existing dependency checkouts")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    parser.add_argument("--cuda", action="store_true", help="enable CUDA for ACMX2/acidcam-gpu only")
    for name, help_text in (
        ("audio", "enable RtAudio support"),
        ("midi", "enable RtMidi support"),
        ("webp", "enable WebP snapshots"),
        ("tiff", "enable TIFF snapshots"),
        ("dnn", "enable OpenCV DNN effects"),
    ):
        parser.add_argument(
            f"--{name}",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=help_text + " (default: enabled)",
        )
    return parser.parse_args()


def enabled(value: bool) -> str:
    return "1" if value else "0"


def main() -> int:
    args = parse_args()
    if platform.system() not in {"Linux", "Darwin"}:
        print(f"error: unsupported platform: {platform.system()}", file=sys.stderr)
        return 1
    if args.jobs is not None and args.jobs < 1:
        print("error: --jobs must be positive", file=sys.stderr)
        return 2
    for command in ("git", "pkg-config", "uvx", "glslc"):
        require_command(command)
    opencv_package = find_opencv_package(args.dry_run)

    build_dir = args.build_dir or Path(
        os.environ.get("ACMX_PCONS_BUILD_DIRECTORY", ROOT_DIR / "build" / "pcons")
    )
    install_prefix = args.prefix or Path(
        os.environ.get("ACMX_PCONS_INSTALL_PREFIX", ROOT_DIR / "build" / "pcons-prefix")
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

    if args.skip_clone and not (libmx2_dir / ".git").is_dir():
        raise RuntimeError(f"libmx2 checkout is required: {libmx2_dir}")
    if args.skip_clone and not (mxvk_dir / ".git").is_dir():
        raise RuntimeError(f"MXVK checkout is required: {mxvk_dir}")
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
        jobs=args.jobs,
        dry_run=args.dry_run,
    )
    print("Building MXVK with Pcons...")
    pcons_build(
        mxvk_dir,
        build_dir / "mxvk",
        install_prefix,
        [
            "PREFIX=" + str(install_prefix),
            "OPENCV_PACKAGE=" + opencv_package,
            "EXAMPLES=0",
            "WITH_CUDA=OFF",
            "CV=ON",
            "VALIDATION=OFF",
        ],
        jobs=args.jobs,
        dry_run=args.dry_run,
    )

    common_options = [
        "PREFIX=" + str(install_prefix),
        "OPENCV_PACKAGE=" + opencv_package,
        "AUDIO=" + enabled(args.audio),
        "MIDI=" + enabled(args.midi),
        "WEBP=" + enabled(args.webp),
        "TIFF=" + enabled(args.tiff),
        "DNN=" + enabled(args.dnn),
    ]
    print("Building ACMX2 with Pcons...")
    pcons_build(
        ROOT_DIR,
        build_dir / "acmx2",
        install_prefix,
        [*common_options, "WITH_CUDA=" + enabled(args.cuda)],
        jobs=args.jobs,
        dry_run=args.dry_run,
    )
    print("Building ACMXVK with Pcons...")
    pcons_build(
        ROOT_DIR / "ACMXVK",
        build_dir / "acmxvk",
        install_prefix,
        [*common_options, "VALIDATION=0", "WITH_CUDA=0"],
        jobs=args.jobs,
        dry_run=args.dry_run,
    )
    print("Building the Qt interface with Pcons...")
    pcons_build(
        ROOT_DIR / "ACMX2" / "interface",
        build_dir / "interface",
        install_prefix,
        [],
        jobs=args.jobs,
        dry_run=args.dry_run,
    )

    print("\nPcons build complete.")
    print(f"  Platform:        {platform.system()} {platform.machine()}")
    print(f"  Build directory: {build_dir}")
    print(f"  Install prefix:  {install_prefix}")
    print("\nLaunch installed programs with:")
    print(f'  export PATH="{install_prefix}/bin:$PATH"')
    if platform.system() == "Darwin":
        print("\nIf the dynamic loader cannot find local libraries, also use:")
        print(f'  export DYLD_FALLBACK_LIBRARY_PATH="{install_prefix}/lib:${{DYLD_FALLBACK_LIBRARY_PATH:-}}"')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
