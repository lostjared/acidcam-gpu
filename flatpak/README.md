# ACMX Flatpak

This manifest builds the current ACMX2 OpenGL and ACMXVK Vulkan backends without
CUDA. It includes the shared Qt interface, RtAudio support, MIDI support, the
`midi-map` utility, the `create_acmxvk_source_manifest` utility, and the
Vulkan-enabled `sd-server` executable from stable-diffusion.cpp. Select the
backend from the interface's Backend menu.

ACMX2 runtime assets are installed below `/app/share/acmx2`, including the
complete interface data set and tracked playlists. The separately maintained
ACMX2 shader collection is pinned and installed below
`/app/share/acmx2/shaders` as the default OpenGL library. ACMXVK resources,
test shaders, playlists, MIDI examples, models, and fonts are installed below
`/app/share/acmxvk`. Editable ACMXVK source libraries can be stored in the home
directory and built to SPIR-V from the interface with the packaged `glslc`.
Stable Diffusion models, LoRAs, and ESRGAN models are not included. Select
user-provided model files from the Stable Diffusion settings dialog; the
interface automatically uses the bundled `/app/bin/sd-server` executable.

The x86_64 package builds OpenCV 4.12.0 with Intel IPP 2022.1.0 and the OpenCV
IPP integration wrappers. OpenCV selects supported CPU paths at runtime,
including SSE4, AVX, AVX2/FMA, and AVX-512. This improves CPU image-processing
performance without changing the package into a CUDA build.

The current release bundle is ACMX 2.140.0. Its SHA-256 is
`707ba8c89e0846d0f0b7418a41ebc76306b72fe2c3a3fb6d1c4151502eb03b3a` and its
size is 71,666,656 bytes (69 MiB). This release prevents the Qt interface log
from freezing during rapid ACMXVK shader changes, condenses visible Vulkan log
traffic while retaining complete per-run file logs, and includes the current
MXVK and bundled stable-diffusion.cpp server revisions.

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
flatpak run --command=sd-server io.github.lostjared.ACMX2 --help
```

The Flatpak exposes the host GPU through the standard Flatpak graphics-driver
extensions. CUDA remains disabled for portability; ACMXVK uses the Vulkan
driver made available by the runtime. The bundled sd-server uses that same
Vulkan path. It is distributed under the MIT License; its license text is
installed at `/app/share/licenses/acmx2/stable-diffusion.cpp-MIT.txt`.
