#!/usr/bin/env python3
"""Strip unused 'in' variable declarations from GLSL fragment shaders."""

import re
import os
import glob

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")

# Regex to match 'in' declarations: in <type> <name>;
IN_DECL_RE = re.compile(r'^(\s*in\s+\w+\s+)(\w+)\s*;(.*)$')

def strip_unused_in_vars(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First pass: collect all 'in' declarations and their variable names
    in_decls = {}  # line_index -> variable_name
    for i, line in enumerate(lines):
        m = IN_DECL_RE.match(line)
        if m:
            varname = m.group(2)
            in_decls[i] = varname

    if not in_decls:
        return 0

    # Build the full text minus the declaration lines to check usage
    removed = []
    for line_idx, varname in in_decls.items():
        # Build text from all lines EXCEPT this declaration
        other_lines = [l for j, l in enumerate(lines) if j != line_idx]
        other_text = ''.join(other_lines)
        # Check if the variable name appears as a word elsewhere
        if not re.search(r'\b' + re.escape(varname) + r'\b', other_text):
            removed.append(line_idx)

    if not removed:
        return 0

    # Remove unused lines
    new_lines = [l for i, l in enumerate(lines) if i not in removed]

    # Clean up resulting double blank lines
    cleaned = []
    for line in new_lines:
        if line.strip() == '' and cleaned and cleaned[-1].strip() == '':
            continue
        cleaned.append(line)

    with open(filepath, 'w') as f:
        f.writelines(cleaned)

    return len(removed)


def main():
    pattern = os.path.join(SHADER_DIR, "*.glsl")
    files = sorted(glob.glob(pattern))
    total_removed = 0
    files_changed = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        if filename == "vertex.glsl":
            continue
        count = strip_unused_in_vars(filepath)
        if count > 0:
            print(f"  {filename}: removed {count} unused 'in' declaration(s)")
            total_removed += count
            files_changed += 1

    print(f"\nDone: removed {total_removed} unused 'in' declarations from {files_changed} files.")


if __name__ == "__main__":
    main()
