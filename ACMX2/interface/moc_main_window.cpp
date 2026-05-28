/****************************************************************************
** Compatibility shim for legacy checked-in moc file.
**
** The actual interface build uses CMake AUTOMOC-generated sources under
** interface/build/acmx2_interface_autogen/. This file exists only to avoid
** stale Qt5-generated diagnostics when editing the repository directly.
*****************************************************************************/

#if defined(__has_include)
#if __has_include("build/acmx2_interface_autogen/EWIEGA46WW/moc_main_window.cpp")
#include "build/acmx2_interface_autogen/EWIEGA46WW/moc_main_window.cpp"
#endif
#endif
