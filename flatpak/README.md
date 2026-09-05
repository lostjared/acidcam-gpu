# ACMX Flatpak

This manifest builds the current ACMX2 OpenGL and ACMXVK Vulkan backends without
CUDA. It includes the shared Qt interface, RtAudio support, MIDI support, the
`midi-map` utility, and the `create_acmxvk_source_manifest` utility. Select the
backend from the interface's Backend menu.

ACMX2 runtime assets are installed below `/app/share/acmx2`, including the
complete interface data set and tracked playlists. The separately maintained
ACMX2 shader collection is pinned and installed below
`/app/share/acmx2/shaders` as the default OpenGL library. ACMXVK resources,
test shaders, playlists, MIDI examples, models, and fonts are installed below
`/app/share/acmxvk`. Editable ACMXVK source libraries can be stored in the home
directory and built to SPIR-V from the interface with the packaged `glslc`.

The x86_64 package builds OpenCV 4.12.0 with Intel IPP 2022.1.0 and the OpenCV
IPP integration wrappers. OpenCV selects supported CPU paths at runtime,
including SSE4, AVX, AVX2/FMA, and AVX-512. This improves CPU image-processing
performance without changing the package into a CUDA build.

The OpenCV module explicitly enables `WITH_IPP` and grants network access only
to that module's build sandbox because OpenCV downloads its pinned IPP archive
during configuration. A clean build therefore needs access to GitHub and
`raw.githubusercontent.com` in addition to Flathub.

All builder state, temporary files, the OSTree repository, and the final bundle
stay below this `flatpak/` directory.

```bash
chmod +x flatpak/build-flatpak.sh
./flatpak/build-flatpak.sh
```

Install and run the resulting bundle with:

```bash
flatpak install --user flatpak/ACMX2.flatpak
flatpak run io.github.lostjared.ACMX2
```

The command-line programs can also be invoked directly:

```bash
flatpak run --command=acmx2 io.github.lostjared.ACMX2 --help
flatpak run --command=acmxvk io.github.lostjared.ACMX2 --help
flatpak run --command=create_acmxvk_source_manifest \
    io.github.lostjared.ACMX2 --help
flatpak run --command=midi-map io.github.lostjared.ACMX2
```

The Flatpak exposes the host GPU through the standard Flatpak graphics-driver
extensions. CUDA remains disabled for portability; ACMXVK uses the Vulkan
driver made available by the runtime.
