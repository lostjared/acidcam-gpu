#!/bin/sh
#
# install.sh -- One-shot orchestrator: install Homebrew deps then build
#               and install ACMX2 + Qt6 interface.
#
# Run from an empty working directory; it will clone libmx2 and ACMX2
# alongside this script. Requires sudo for `make install`.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"${SCRIPT_DIR}/install-dep.sh"
"${SCRIPT_DIR}/build-macos.sh"
