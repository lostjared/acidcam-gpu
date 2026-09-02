#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pcons>=0.24"]
# ///
"""pcons build for the ACMX2 Qt 6 interface and MIDI map utility.

The CMakeLists.txt equivalent used AUTOMOC/AUTOUIC/AUTORCC; here
QtProgram() does the same job: the .qrc goes straight into sources and
every Q_OBJECT header (editor.hpp, main_window.hpp, syntax.hpp, ...) is
found by the generate-time scan — no header list needed at all.

Note: the checked-in qrc_qresource.cpp is a stale generated file and is
deliberately not a source; pcons generates its own from qresource.qrc.

Usage:
    pcons -B build/interface-pcons --reconfigure
    pcons -B build/interface-pcons VARIANT=debug

The install alias matches CMake's executable and desktop-entry installation:

    pcons -B build/interface-pcons \
        PCONS_INSTALL_PREFIX=/opt/acmx \
        PCONS_FINAL_PREFIX=/opt/acmx all install
"""

import os
from pathlib import Path

from pcons import Project, find_c_toolchain, get_platform
from pcons.toolchains.qt import find_qt

project = Project("acmx2_interface")
project_dir = Path(__file__).parent.resolve()
env = project.Environment(toolchain=find_c_toolchain())
env.cxx.set_standard(17)
env.set_variant(os.environ.get("VARIANT", "release"))
platform = get_platform()

qt = find_qt(
    project, env, modules=["Core", "Gui", "Widgets", "Concurrent", "Network"]
)

if not platform.is_windows:
    env.cxx.flags.append("-pthread")
    env.link.flags.append("-pthread")
if platform.is_linux:
    # Qt on Linux is built with reduced relocations and requires PIC objects.
    env.cxx.flags.append("-fPIC")

app = project.QtProgram(
    "acmx2_interface",
    env,
    sources=[
        "audio-playlist.cpp",
        "audio-window.cpp",
        "acmxvk-source-manifest.cpp",
        "custom-uniforms.cpp",
        "editor.cpp",
        "find-shader.cpp",
        "gpufilter.cpp",
        "library-builder.cpp",
        "main.cpp",
        "main_window.cpp",
        "metadata-viewer.cpp",
        "midi-settings.cpp",
        "playlist.cpp",
        "prop.cpp",
        "settings.cpp",
        "shader.cpp",
        "shader-manifest.cpp",
        "shaderlibrary.cpp",
        "shaderpass.cpp",
        "syntax.cpp",
        "uniform-reference.cpp",
        "qresource.qrc",
    ],
    link=[qt.Widgets, qt.Gui, qt.Concurrent, qt.Network, qt.Core],
)

# The ACMXVK source-library builder is installed beside the interface just as
# it is in CMake.  Keep its manifest implementation shared with the GUI.
acmxvk_source_manifest = project.Program(
    "create_acmxvk_source_manifest",
    env,
    sources=[
        "acmxvk-source-manifest-cli.cpp",
        "acmxvk-source-manifest.cpp",
    ],
)
acmxvk_source_manifest.private.include_dirs.append(".")
acmxvk_source_manifest.link(qt.Core)

rtmidi = project.find_package("rtmidi")
assert rtmidi is not None

midi_env = env.clone()
midi_env.cxx.set_standard(20)
midi_env.cxx.flags.extend(["-O3", "-Wall", "-pedantic"])

midi_sources = ["midi-map/main.cpp", "midi-map/midi_window.cpp"]
if platform.is_windows:
    midi_sources.append("midi-map/win-icon.rc")

midi_map = project.QtProgram(
    "midi_map",
    midi_env,
    sources=midi_sources,
    link=[qt.Widgets, rtmidi],
)
midi_map.output_name = "midi-map"

installed = [
    project.Install("bin", [app, acmxvk_source_manifest]),
    project.InstallAs("bin/midi-map", midi_map, name="install_midi_map"),
    project.Install("share/applications", ["acmx2-interface.desktop"]),
    project.Install("share/acmx2", [project_dir.parent / "data" / "win-icon.png"]),
]
project.Alias("install", *installed)
