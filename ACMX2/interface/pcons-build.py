#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pcons>=0.24"]
# ///
"""pcons build for the ACMX2 Qt 6 interface (acmx2_interface).

The CMakeLists.txt equivalent used AUTOMOC/AUTOUIC/AUTORCC; here
QtProgram() does the same job: the .qrc goes straight into sources and
every Q_OBJECT header (editor.hpp, main_window.hpp, syntax.hpp, ...) is
found by the generate-time scan — no header list needed at all.

Note: the checked-in qrc_qresource.cpp is a stale generated file and is
deliberately not a source; pcons generates its own from qresource.qrc.

Usage:
    uvx pcons          # configure + generate, then: ninja -C build
    VARIANT=debug uvx pcons
"""

import os

from pcons import Project, find_c_toolchain, get_platform
from pcons.toolchains.qt import find_qt

project = Project("acmx2_interface")
env = project.Environment(toolchain=find_c_toolchain())
env.cxx.set_standard(17)
env.set_variant(os.environ.get("VARIANT", "release"))

qt = find_qt(
    project, env, modules=["Core", "Gui", "Widgets", "Concurrent", "Network"]
)

if not get_platform().is_windows:
    env.cxx.flags.append("-pthread")
    env.link.flags.append("-pthread")

app = project.QtProgram(
    "acmx2_interface",
    env,
    sources=[
        "audio-window.cpp",
        "editor.cpp",
        "gpufilter.cpp",
        "main.cpp",
        "main_window.cpp",
        "metadata-viewer.cpp",
        "midi-settings.cpp",
        "playlist.cpp",
        "prop.cpp",
        "settings.cpp",
        "shader.cpp",
        "shaderlibrary.cpp",
        "shaderpass.cpp",
        "syntax.cpp",
        "qresource.qrc",
    ],
    link=[qt.Widgets, qt.Gui, qt.Concurrent, qt.Network, qt.Core],
)

project.Install("bin", [app])
project.Install("share/applications", ["acmx2-interface.desktop"])
