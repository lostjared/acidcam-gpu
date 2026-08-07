# ACMX2 Flatpak

This manifest builds ACMX2 without CUDA and includes the Qt interface, RtAudio
support, MIDI support, and the `midi-map` utility. Runtime assets are installed below
`/app/share/acmx2`, including the complete interface data set and tracked
playlists. The separately maintained ACMX2 shader collection is pinned and
installed below `/app/share/acmx2/shaders` as the default library.

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
flatpak run --command=midi-map io.github.lostjared.ACMX2
```
