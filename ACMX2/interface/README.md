# ACMX2 Interface

Qt6-based GUI launcher for the ACMX2 engine.

## Building

```bash
cd interface
mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

## Install

`make install` places:

- `acmx2_interface` → `<prefix>/bin/`
- `acmx2-interface.desktop` → `<prefix>/share/applications/`
- `acmx2.png` → `<prefix>/share/acmx2/`

The interface automatically resolves the ACMX2 data directory at `<prefix>/share/acmx2/` when the local `./data` directory is not present beside the binary.
