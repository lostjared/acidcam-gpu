#!/usr/bin/env python3
"""Parse FILTERS.md and filters.cu, generate a Markdown filter reference."""

import re
import os


def parse_filters(md_path):
    with open(md_path, "r") as f:
        text = f.read()

    # Split on ## headers
    entries = re.split(r'\n## ', text)
    filters = []
    for entry in entries[1:]:  # skip preamble
        lines = entry.strip().split('\n')
        # First line: "123 — FilterName"
        header = lines[0].strip()
        m = re.match(r'(\d+)\s*[—–-]+\s*(.*)', header)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip()

        # Collect description and technique
        desc_lines = []
        technique = ""
        for line in lines[1:]:
            line_s = line.strip()
            if line_s.startswith('**Technique:**'):
                technique = line_s.replace('**Technique:**', '').strip()
            elif line_s == '---' or line_s == '':
                continue
            else:
                desc_lines.append(line_s)

        description = ' '.join(desc_lines)
        filters.append({
            "id": idx,
            "name": name,
            "desc": description,
            "technique": technique,
        })

    return filters


def extract_device_functions(cu_source):
    """Extract all __device__ function bodies from the .cu source."""
    funcs = {}
    pattern = re.compile(r'^\s+__device__\s+\w+\s+(\w+)\s*\(', re.MULTILINE)
    for m in pattern.finditer(cu_source):
        fname = m.group(1)
        start = m.start()
        brace_pos = cu_source.index('{', m.end())
        depth = 1
        pos = brace_pos + 1
        while depth > 0 and pos < len(cu_source):
            if cu_source[pos] == '{':
                depth += 1
            elif cu_source[pos] == '}':
                depth -= 1
            pos += 1
        line_start = cu_source.rfind('\n', 0, start) + 1
        func_text = cu_source[line_start:pos].rstrip()
        lines = func_text.split('\n')
        if lines:
            min_indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
            lines = [l[min_indent:] for l in lines]
        funcs[fname] = '\n'.join(lines)
    return funcs


def extract_switch_cases(cu_source):
    """Extract each case block from the unifiedFilterKernel switch statement."""
    switch_match = re.search(r'switch\s*\(filters\[i\]\.index\)\s*\{', cu_source)
    if not switch_match:
        return {}

    switch_start = switch_match.end()
    case_pattern = re.compile(r'^\s+case\s+(\d+)\s*:', re.MULTILINE)
    cases_pos = []
    cases_pos_dict = {}
    for m in case_pattern.finditer(cu_source, switch_start):
        case_id = int(m.group(1))
        cases_pos.append((case_id, m.start(), m.end()))
        cases_pos_dict[case_id] = True

    case_code = {}
    for i, (case_id, start, end) in enumerate(cases_pos):
        if i + 1 < len(cases_pos):
            next_start = cases_pos[i + 1][1]
        else:
            next_start = cu_source.index('\n            }', end)

        block = cu_source[end:next_start].strip()
        block = re.sub(r'\bbreak;\s*$', '', block).strip()
        if not block:
            continue
        if block.startswith('{') and block.endswith('}'):
            inner = block[1:-1]
            inner_lines = inner.split('\n')
            while inner_lines and not inner_lines[0].strip():
                inner_lines.pop(0)
            while inner_lines and not inner_lines[-1].strip():
                inner_lines.pop()
            block = '\n'.join(inner_lines)
        lines = block.split('\n')
        if lines:
            min_indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
            lines = [l[min_indent:] for l in lines]
        case_code[case_id] = '\n'.join(lines)

    all_ids = sorted(cases_pos_dict.keys())
    for case_id in all_ids:
        if case_id not in case_code:
            for next_id in all_ids:
                if next_id > case_id and next_id in case_code:
                    case_code[case_id] = case_code[next_id]
                    break

    return case_code


def build_filter_code(case_code, device_funcs):
    """For each case, build a complete code snippet with referenced __device__ functions."""
    result = {}
    for case_id, code in case_code.items():
        called = re.findall(r'\b(process\w+|acgl_\w+)\b', code)
        called_funcs = []
        seen = set()
        for fname in called:
            if fname in device_funcs and fname not in seen:
                seen.add(fname)
                called_funcs.append(device_funcs[fname])

        parts = [f"// Case {case_id} in unifiedFilterKernel:", code]
        if called_funcs:
            parts.append("")
            parts.append("// Referenced __device__ function(s):")
            for f in called_funcs:
                parts.append(f)

        result[case_id] = '\n'.join(parts)
    return result


def extract_global_function(cu_source):
    """Extract the full __global__ unifiedFilterKernel function."""
    match = re.search(r'^\s+__global__\s+void\s+unifiedFilterKernel\s*\(', cu_source, re.MULTILINE)
    if not match:
        return ""
    start = match.start()
    brace_pos = cu_source.index('{', match.end())
    depth = 1
    pos = brace_pos + 1
    while depth > 0 and pos < len(cu_source):
        if cu_source[pos] == '{':
            depth += 1
        elif cu_source[pos] == '}':
            depth -= 1
        pos += 1
    line_start = cu_source.rfind('\n', 0, start) + 1
    func_text = cu_source[line_start:pos].rstrip()
    lines = func_text.split('\n')
    if lines:
        min_indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        lines = [l[min_indent:] for l in lines]
    return '\n'.join(lines)


def parse_kernel_code(cu_path):
    """Parse the .cu file and return (per-filter code dict, full kernel string)."""
    with open(cu_path, 'r') as f:
        cu_source = f.read()

    device_funcs = extract_device_functions(cu_source)
    case_code = extract_switch_cases(cu_source)
    full_kernel = extract_global_function(cu_source)
    return build_filter_code(case_code, device_funcs), full_kernel


CATEGORIES = [
    ("Core Processing (0–99)", 0, 99),
    ("Color & Temporal (100–199)", 100, 199),
    ("Spatial & Pattern (200–299)", 200, 299),
    ("Advanced Blend (300–399)", 300, 399),
    ("Extended Effects (400–499)", 400, 499),
    ("Visual FX A (500–599)", 500, 599),
    ("Visual FX B (600–699)", 600, 699),
    ("Visual FX C (700–735)", 700, 735),
    ("Glitch Series (736–857)", 736, 857),
    ("AC Glitch Library (858–904)", 858, 904),
]


def generate_filter_markdown(filters, full_kernel=""):
    """Generate the Markdown string for the CUDA filters section."""
    lines = []
    lines.append("# Part I — CUDA Filters\n")
    lines.append(f"This section documents all **{len(filters)} CUDA filter kernels** in `filters.cu`. "
                 "Each filter runs as a `__device__` function called per-pixel from the unified GPU kernel.\n")

    if full_kernel:
        lines.append("## Unified Filter Kernel\n")
        lines.append("The unified GPU filter kernel dispatches all filter effects from a single "
                     "CUDA `__global__` function. Each thread processes one pixel, iterating over "
                     "the active filter chain and dispatching via a switch on the filter index.\n")
        lines.append("```cpp")
        lines.append(full_kernel)
        lines.append("```\n")

    lines.append("---\n")

    for cat_label, lo, hi in CATEGORIES:
        cat_filters = [f for f in filters if lo <= f['id'] <= hi]
        if not cat_filters:
            continue

        lines.append(f"## {cat_label}\n")

        for f in cat_filters:
            lines.append(f"### Filter #{f['id']} — {f['name']}\n")
            if f['desc']:
                lines.append(f"{f['desc']}\n")
            if f['technique']:
                lines.append(f"**Technique:** {f['technique']}\n")
            if f.get('code'):
                lines.append("```cpp")
                lines.append(f['code'])
                lines.append("```\n")
            lines.append("---\n")

    return '\n'.join(lines)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(script_dir, 'acidcam-gpu', 'src', 'filters.cu')

    filters = parse_filters(os.path.join(script_dir, 'FILTERS.md'))
    print(f"Parsed {len(filters)} filters from FILTERS.md")

    full_kernel = ""
    if os.path.exists(cu_path):
        kernel_code, full_kernel = parse_kernel_code(cu_path)
        print(f"Extracted kernel code for {len(kernel_code)} cases from filters.cu")
        for f in filters:
            if f['id'] in kernel_code:
                f['code'] = kernel_code[f['id']]
        with_code = sum(1 for f in filters if 'code' in f)
        print(f"Attached code to {with_code}/{len(filters)} filters")
    else:
        print(f"Warning: {cu_path} not found, skipping kernel code extraction")

    md = generate_filter_markdown(filters, full_kernel)
    out_path = os.path.join(script_dir, 'effects_reference_filters.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Written: {out_path}")
