#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOXYFILE="$ROOT_DIR/Doxyfile"
DOCS_DIR="$ROOT_DIR/docs"

if ! command -v doxygen >/dev/null 2>&1; then
    echo "Error: doxygen is not installed or not in PATH" >&2
    exit 1
fi

version="$(awk -F= '/^PROJECT_NUMBER[[:space:]]*=/{v=$2; gsub(/"/,"",v); gsub(/[[:space:]]/,"",v); print v; exit}' "$DOXYFILE")"
if [[ -z "$version" ]]; then
    echo "Error: failed to parse PROJECT_NUMBER from Doxyfile" >&2
    exit 1
fi

version_output="versions/$version"
version_dir="$DOCS_DIR/$version_output"
latest_dir="$DOCS_DIR/latest"

mkdir -p "$(dirname "$version_dir")"

# Build docs into a versioned folder regardless of current checked-in HTML_OUTPUT value.
tmp_doxyfile="$(mktemp)"
trap 'rm -f "$tmp_doxyfile"' EXIT
cp "$DOXYFILE" "$tmp_doxyfile"
printf "\nOUTPUT_DIRECTORY = docs\nHTML_OUTPUT = %s\n" "$version_output" >> "$tmp_doxyfile"

(
    cd "$ROOT_DIR"
    doxygen "$tmp_doxyfile"
)

if [[ ! -d "$version_dir" ]]; then
    echo "Error: expected generated version directory not found: $version_dir" >&2
    exit 1
fi

while IFS= read -r -d '' html_file; do
    ACMX_DOC_ASSET_VERSION="$version" perl -0pi -e '
        s{((?:src|href)="(?!https?://|//)[^"]+\.(?:js|css))(?:\?[^"#]*)?"}
         {$1 . "?v=" . $ENV{ACMX_DOC_ASSET_VERSION} . "\""}ge
    ' "$html_file"
done < <(find "$version_dir" -type f -name '*.html' -print0)

perl -pi -e "s{scriptName\+'\\.js'}{scriptName+'.js?v=$version'}g" \
    "$version_dir/navtree.js"
perl -pi -e "s{scriptTag\\.src = url;}{scriptTag.src = url + '?v=$version';}g" \
    "$version_dir/search/search.js"

rm -rf "$latest_dir"
mkdir -p "$latest_dir"
cp -a "$version_dir/." "$latest_dir/"

cat > "$DOCS_DIR/index.html" <<'HTML'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=./latest/index.html">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACMX Docs</title>
</head>
<body>
  <p>Redirecting to latest docs...</p>
  <p><a href="./latest/index.html">If you are not redirected, open the latest docs.</a></p>
</body>
</html>
HTML

echo "Generated docs version: $version_dir"
echo "Updated latest alias: $latest_dir"
echo "Entry page: $DOCS_DIR/index.html"
