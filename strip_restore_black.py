#!/usr/bin/env python3
"""Strip the restore_black_value discard pattern from fragment shaders."""

import os
import re

SHADER_DIR = "shaders"

def process_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    removed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Remove: in float restore_black_value;
        if stripped == "in float restore_black_value;":
            removed += 1
            i += 1
            continue

        # Remove: if(restore_black_value == 1.0 ...) + discard;
        if re.match(r'\s*if\s*\(\s*restore_black_value\s*==', stripped):
            removed += 1
            i += 1
            # Also remove the following discard; line
            if i < len(lines) and lines[i].strip() == "discard;":
                removed += 1
                i += 1
            continue

        new_lines.append(line)
        i += 1

    if removed == 0:
        return 0

    # Clean up double blank lines
    cleaned = []
    for line in new_lines:
        if line.strip() == "" and cleaned and cleaned[-1].strip() == "":
            continue
        cleaned.append(line)

    with open(path, "w") as f:
        f.writelines(cleaned)

    return removed


total_removed = 0
total_files = 0

for name in sorted(os.listdir(SHADER_DIR)):
    if not name.endswith(".glsl") or name == "vertex.glsl":
        continue
    path = os.path.join(SHADER_DIR, name)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        continue
    n = process_file(path)
    if n > 0:
        print(f"  {name}: removed {n} line(s)")
        total_removed += n
        total_files += 1

print(f"Done: removed {total_removed} lines from {total_files} files.")
