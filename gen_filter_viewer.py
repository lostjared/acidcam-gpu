#!/usr/bin/env python3
"""Parse FILTERS.md and filters.cu, generate an HTML5 filter browser with kernel code."""

import re
import json
import html
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
        # Strip wrapping braces from inline case blocks: { ... } break;
        if block.startswith('{') and block.endswith('}'):
            # Remove outer braces, preserving internal whitespace structure
            inner = block[1:-1]
            # Remove leading/trailing blank lines only
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

    # Handle fall-through cases: if a case has no code, inherit from the next case
    all_ids = sorted(cases_pos_dict.keys())
    for case_id in all_ids:
        if case_id not in case_code:
            # Find the next case that has code
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

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AcidCam GPU Filter Browser</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#f0f4f8; color:#1a1a2e; }

/* Top bar */
.topbar {
    display:flex; align-items:center; gap:16px;
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%);
    color:#fff; padding:12px 24px; height:60px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.topbar h1 { font-size:20px; white-space:nowrap; }
.search-box {
    flex:1; max-width:420px; position:relative;
}
.search-box input {
    width:100%; padding:8px 12px 8px 36px;
    border:none; border-radius:6px; font-size:14px;
    background:rgba(255,255,255,0.15); color:#fff;
    outline:none; transition: background 0.2s;
}
.search-box input::placeholder { color:rgba(255,255,255,0.6); }
.search-box input:focus { background:rgba(255,255,255,0.25); }
.search-box svg {
    position:absolute; left:10px; top:50%; transform:translateY(-50%);
    width:16px; height:16px; fill:rgba(255,255,255,0.6);
}
.search-results-count {
    font-size:12px; color:rgba(255,255,255,0.7); white-space:nowrap;
}

/* Layout */
.container { display:flex; height:calc(100% - 60px); }

/* Tree panel */
.tree-panel {
    width:340px; min-width:260px; background:#fff;
    border-right:2px solid #bbdefb;
    overflow-y:auto; padding:8px 0;
    scrollbar-width:thin;
}
.tree-panel::-webkit-scrollbar { width:6px; }
.tree-panel::-webkit-scrollbar-thumb { background:#90caf9; border-radius:3px; }

/* Category group */
.cat-header {
    display:flex; align-items:center; gap:6px;
    padding:8px 12px; cursor:pointer;
    font-weight:700; font-size:13px; color:#1a237e;
    background:#e3f2fd; border-bottom:1px solid #bbdefb;
    user-select:none; position:sticky; top:0; z-index:2;
    transition: background 0.15s;
}
.cat-header:hover { background:#bbdefb; }
.cat-header .arrow {
    display:inline-block; width:16px; text-align:center;
    font-size:10px; transition:transform 0.2s;
}
.cat-header.open .arrow { transform:rotate(90deg); }
.cat-header .count {
    margin-left:auto; font-weight:400; font-size:11px;
    color:#5c6bc0; background:#c5cae9; border-radius:10px;
    padding:1px 8px;
}
.cat-items { display:none; }
.cat-items.open { display:block; }

/* Tree item */
.tree-item {
    display:flex; align-items:baseline; gap:6px;
    padding:5px 12px 5px 28px; cursor:pointer;
    font-size:13px; color:#333; border-left:3px solid transparent;
    transition: background 0.1s, border-color 0.1s;
}
.tree-item:hover { background:#e8eaf6; }
.tree-item.active { background:#e3f2fd; border-left-color:#1565c0; font-weight:600; }
.tree-item .idx { color:#1565c0; font-weight:700; font-size:12px; min-width:32px; }
.tree-item .fname { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.tree-item.search-hit { background:#fff9c4; }
.tree-item.search-hit.active { background:#fff176; border-left-color:#f9a825; }

/* Main panel */
.main-panel {
    flex:1; overflow-y:auto; padding:32px 48px;
    background:#f8fafc;
}
.main-panel::-webkit-scrollbar { width:6px; }
.main-panel::-webkit-scrollbar-thumb { background:#90caf9; border-radius:3px; }

.welcome {
    text-align:center; margin-top:15vh; color:#90a4ae;
}
.welcome h2 { font-size:28px; margin-bottom:12px; color:#78909c; }
.welcome p { font-size:15px; }

.filter-view { animation: fadeIn 0.2s ease; }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

.filter-view .fv-index {
    font-size:14px; font-weight:700; color:#1565c0;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;
}
.filter-view .fv-name {
    font-size:32px; font-weight:800; color:#1a1a2e;
    margin-bottom:20px; line-height:1.2;
}
.filter-view .fv-desc {
    font-size:16px; line-height:1.7; color:#222;
    margin-bottom:24px; max-width:720px;
}
.filter-view .fv-technique {
    display:inline-block;
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left:4px solid #1565c0;
    padding:12px 20px; border-radius:0 8px 8px 0;
    font-size:14px; color:#0d47a1; max-width:720px;
}
.filter-view .fv-technique strong { color:#b71c1c; }

/* Search results view */
.search-results-view { animation: fadeIn 0.2s ease; }
.search-results-view h2 {
    font-size:22px; color:#1a237e; margin-bottom:16px;
    border-bottom:2px solid #bbdefb; padding-bottom:8px;
}
.sr-card {
    background:#fff; border:1px solid #e0e0e0; border-radius:8px;
    padding:16px 20px; margin-bottom:12px; cursor:pointer;
    transition: box-shadow 0.15s, border-color 0.15s;
}
.sr-card:hover { box-shadow:0 2px 12px rgba(21,101,192,0.12); border-color:#90caf9; }
.sr-card .sr-idx { font-size:12px; font-weight:700; color:#1565c0; }
.sr-card .sr-name { font-size:18px; font-weight:700; color:#1a1a2e; margin:2px 0 6px; }
.sr-card .sr-desc { font-size:13px; color:#555; line-height:1.5; }
.sr-card .sr-tech { font-size:12px; color:#b71c1c; margin-top:6px; font-style:italic; }
mark { background:#fff176; color:#1a1a2e; border-radius:2px; padding:0 1px; }

/* Kernel code block */
.fv-code-header {
    margin-top:28px; margin-bottom:8px;
    font-size:14px; font-weight:700; color:#1a237e;
    text-transform:uppercase; letter-spacing:1px;
}
.fv-code-wrap {
    max-width:820px; border-radius:8px; overflow:hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.fv-code-wrap pre {
    margin:0; padding:16px 20px; font-size:13px; line-height:1.6;
    overflow-x:auto;
}
.fv-code-wrap code { font-family: 'Cascadia Code', 'Fira Code', 'Source Code Pro', Consolas, monospace; }

/* Mobile layout */
@media (max-width: 767px) {
    .topbar { padding:8px 12px; height:52px; gap:8px; }
    .topbar h1 { font-size:15px; }
    .search-box { max-width:none; flex:1; }
    .container { flex-direction:row; height:calc(100% - 52px); }
    .tree-panel {
        width:42vw; min-width:120px; max-width:180px;
        overflow-y:auto; overflow-x:hidden;
        scrollbar-width:thin; flex-shrink:0;
    }
    .cat-header { padding:6px 8px; font-size:11px; }
    .cat-header .count { padding:1px 5px; font-size:10px; }
    .tree-item { padding:5px 6px 5px 16px; font-size:11px; }
    .tree-item .idx { min-width:24px; font-size:10px; }
    .main-panel { flex:1; min-width:0; padding:16px 14px; overflow-y:auto; }
    .filter-view .fv-name { font-size:22px; margin-bottom:14px; }
    .filter-view .fv-desc { font-size:14px; }
    .filter-view .fv-technique { font-size:13px; padding:10px 14px; }
    .fv-code-wrap pre { font-size:11px; padding:12px 14px; }
    .welcome { margin-top:8vh; }
    .welcome h2 { font-size:20px; }
}
</style>
</head>
<body>

<div class="topbar">
    <h1>AcidCam GPU Filters</h1>
    <div class="search-box">
        <svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
        <input type="text" id="searchInput" placeholder="Search filters by name, description, or technique..." autocomplete="off">
    </div>
    <span class="search-results-count" id="searchCount"></span>
</div>

<div class="container">
    <div class="tree-panel" id="treePanel"></div>
    <div class="main-panel" id="mainPanel"></div>
</div>

<script>
const FILTERS = %%FILTERS_JSON%%;

const CATEGORIES = %%CATEGORIES_JSON%%;

const FULL_KERNEL = %%FULL_KERNEL_JSON%%;

// Build tree
const treePanel = document.getElementById('treePanel');
const mainPanel = document.getElementById('mainPanel');
const searchInput = document.getElementById('searchInput');
const searchCount = document.getElementById('searchCount');

let activeItem = null;
let searchHits = new Set();

function showBuildInstructions() {
    if (activeItem) { activeItem.classList.remove('active'); activeItem = null; }
    const el = treePanel.querySelector('.tree-item[data-id="build-instructions"]');
    if (el) { el.classList.add('active'); activeItem = el; }

    mainPanel.innerHTML = `
        <div class="filter-view">
            <div class="fv-index">CONTAINER / PODMAN</div>
            <div class="fv-name">Build Instructions</div>
            <div class="fv-desc">Build ACMX2 from source using the included Arch Linux Podman container. The container installs <code style="background:#e8eaf6;padding:2px 6px;border-radius:4px;font-size:14px;color:#1a237e">opencv-cuda</code> directly from pacman &mdash; no manual OpenCV compilation required. Packages are installed in separate cached layers so the build is resumable if interrupted.</div>

            <div class="fv-technique" style="display:block;margin-bottom:24px">
                <strong>Prerequisites:</strong> Podman with NVIDIA Container Toolkit (CDI) &bull; ~20 GB free disk space &bull; 8+ GB RAM &bull; Linux x86_64 with NVIDIA GPU (CUDA 12.x)
            </div>

            <div class="fv-code-header">Step 1 &mdash; Enter the podman directory</div>
            <div class="fv-code-wrap" style="max-width:640px">
                <pre><code class="language-bash">cd /path/to/acidcam-gpu/podman</code></pre>
            </div>

            <div class="fv-code-header" style="margin-top:20px">Step 2 &mdash; Build the container image</div>
            <div class="fv-desc" style="margin-top:8px">Packages are split into separately cached layers so a failed build resumes from the last completed step.</div>
            <div class="fv-code-wrap" style="max-width:640px">
                <pre><code class="language-bash">podman build -t acmx2-arch -f Containerfile.arch .</code></pre>
            </div>
            <div class="fv-technique" style="display:block;margin-top:12px;border-left-color:#e65100;background:linear-gradient(135deg,#fff3e0,#ffe0b2);color:#bf360c">
                <strong>&#9888; Build time:</strong> 30&ndash;90 minutes depending on hardware and internet speed. Compilation is capped at 4 parallel jobs to avoid OOM lockups.<br><br>
                To free disk space from failed attempts:<br>
                <code style="display:inline-block;margin-top:6px;background:rgba(0,0,0,0.1);padding:2px 8px;border-radius:4px;font-family:monospace">podman system df</code><br>
                <code style="display:inline-block;margin-top:4px;background:rgba(0,0,0,0.1);padding:2px 8px;border-radius:4px;font-family:monospace">podman image prune --force</code>
            </div>

            <div class="fv-code-header" style="margin-top:24px">Step 3 &mdash; CUDA Architecture (optional)</div>
            <div class="fv-desc" style="margin-top:8px">The default is <code style="background:#e8eaf6;padding:2px 6px;border-radius:4px;font-size:14px;color:#1a237e">CMAKE_CUDA_ARCHITECTURES=&quot;75&quot;</code>. Edit <code style="background:#e8eaf6;padding:2px 6px;border-radius:4px;font-size:14px;color:#1a237e">Containerfile.arch</code> before building if your GPU differs:</div>
            <table style="max-width:520px;margin-top:10px;border-collapse:collapse;background:#fff;border:1px solid #bbdefb;border-radius:8px;overflow:hidden">
                <thead><tr style="background:linear-gradient(135deg,#1a237e,#1565c0)">
                    <th style="padding:10px 16px;text-align:left;color:#fff;font-size:13px">GPU Generation</th>
                    <th style="padding:10px 16px;text-align:left;color:#fff;font-size:13px">Architecture</th>
                </tr></thead>
                <tbody>
                    <tr style="border-bottom:1px solid #bbdefb"><td style="padding:10px 16px;color:#333;font-size:13px">GTX 16xx / RTX 20xx (Turing)</td><td style="padding:10px 16px"><code style="background:#e8eaf6;padding:2px 8px;border-radius:3px;color:#1a237e">75</code></td></tr>
                    <tr style="border-bottom:1px solid #bbdefb"><td style="padding:10px 16px;color:#333;font-size:13px">RTX 30xx (Ampere)</td><td style="padding:10px 16px"><code style="background:#e8eaf6;padding:2px 8px;border-radius:3px;color:#1a237e">86</code></td></tr>
                    <tr><td style="padding:10px 16px;color:#333;font-size:13px">RTX 40xx (Ada Lovelace)</td><td style="padding:10px 16px"><code style="background:#e8eaf6;padding:2px 8px;border-radius:3px;color:#1a237e">89</code></td></tr>
                </tbody>
            </table>

            <div class="fv-code-header" style="margin-top:24px">Step 4 &mdash; Verify the image</div>
            <div class="fv-code-wrap" style="max-width:640px">
                <pre><code class="language-bash">podman images | grep acmx2-arch</code></pre>
            </div>

            <div class="fv-code-header" style="margin-top:20px">Step 5 &mdash; Launch ACMX2</div>
            <div class="fv-desc" style="margin-top:8px">Use <code style="background:#e8eaf6;padding:2px 6px;border-radius:4px;font-size:14px;color:#1a237e">run-acmx2-arch.sh</code> from the <code style="background:#e8eaf6;padding:2px 6px;border-radius:4px;font-size:14px;color:#1a237e">podman/</code> directory:</div>
            <div class="fv-code-wrap" style="max-width:640px">
                <pre><code class="language-bash">chmod +x run-acmx2-arch.sh\n./run-acmx2-arch.sh</code></pre>
            </div>
            <div class="fv-technique" style="display:block;margin-top:12px;border-left-color:#2e7d32;background:linear-gradient(135deg,#e8f5e9,#c8e6c9);color:#1b5e20">
                <strong>&#10003; The script automatically:</strong> detects all /dev/video* webcam devices, mounts PulseAudio for audio, passes <code style="background:rgba(0,0,0,0.08);padding:1px 6px;border-radius:3px;font-family:monospace">--device nvidia.com/gpu=all</code> for GPU access, and mounts <code style="background:rgba(0,0,0,0.08);padding:1px 6px;border-radius:3px;font-family:monospace">~/container_share</code> at <code style="background:rgba(0,0,0,0.08);padding:1px 6px;border-radius:3px;font-family:monospace">/root/share</code> for file exchange.
            </div>

            <div class="fv-code-header" style="margin-top:28px">First-Time Setup</div>
            <ol style="margin:12px 0 0 20px;color:#222;font-size:15px;line-height:2">
                <li>Go to <strong>File &rarr; Properties</strong></li>
                <li>Set the path to your <strong>ACMX2 executable</strong></li>
                <li>Set the path to your <strong>Shader Library</strong> folder (must contain an <code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">index.txt</code> file)</li>
                <li>Click <strong>OK</strong> to save settings</li>
            </ol>
            <div class="fv-technique" style="display:block;margin-top:16px">
                <strong>Shader Packs Available:</strong> Download additional shader and model packs from <a href="https://lostsidedead.biz/packs/" style="color:#1565c0">https://lostsidedead.biz/packs/</a>
            </div>
        </div>`;

    mainPanel.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });
}

function showUnifiedKernel() {
    if (activeItem) { activeItem.classList.remove('active'); activeItem = null; }
    const el = treePanel.querySelector('.tree-item[data-id="kernel"]');
    if (el) { el.classList.add('active'); activeItem = el; }

    mainPanel.innerHTML = `
        <div class="filter-view">
            <div class="fv-index">CUDA / GPU</div>
            <div class="fv-name">unifiedFilterKernel</div>
            <div class="fv-desc">The unified GPU filter kernel dispatches all 905 filter effects from a single CUDA __global__ function. Each thread processes one pixel, iterating over the active filter chain and dispatching via a switch on the filter index. Select a filter from the tree or use the search bar to view individual kernel code.</div>
            <div class="fv-technique"><strong>Technique:</strong> Single-kernel multi-filter dispatch with per-pixel thread mapping and chained filter evaluation.</div>
            <div class="fv-code-header">CUDA Kernel Code</div>
            <div class="fv-code-wrap">
                <pre><code class="language-cpp">${esc(FULL_KERNEL)}</code></pre>
            </div>
        </div>`;
    mainPanel.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });
}

function buildTree() {
    treePanel.innerHTML = '';

    // Build Instructions entry
    const buildItem = document.createElement('div');
    buildItem.className = 'tree-item';
    buildItem.setAttribute('data-id', 'build-instructions');
    buildItem.innerHTML = '<span class="idx">&#9881;</span><span class="fname">Build Instructions</span>';
    buildItem.addEventListener('click', () => showBuildInstructions());
    treePanel.appendChild(buildItem);

    // Unified Kernel entry
    const kernelItem = document.createElement('div');
    kernelItem.className = 'tree-item active';
    kernelItem.setAttribute('data-id', 'kernel');
    kernelItem.innerHTML = '<span class="idx">&#9889;</span><span class="fname">Unified Kernel</span>';
    kernelItem.addEventListener('click', () => showUnifiedKernel());
    treePanel.appendChild(kernelItem);
    activeItem = kernelItem;

    CATEGORIES.forEach((cat, ci) => {
        const [label, lo, hi] = cat;
        const items = FILTERS.filter(f => f.id >= lo && f.id <= hi);

        const header = document.createElement('div');
        header.className = 'cat-header';
        header.innerHTML = `<span class="arrow">&#9654;</span><span>${esc(label)}</span><span class="count">${items.length}</span>`;

        const list = document.createElement('div');
        list.className = 'cat-items';
        list.setAttribute('data-cat', ci);

        items.forEach(f => {
            const item = document.createElement('div');
            item.className = 'tree-item';
            item.setAttribute('data-id', f.id);
            item.innerHTML = `<span class="idx">${f.id}</span><span class="fname">${esc(f.name)}</span>`;
            item.addEventListener('click', () => selectFilter(f.id));
            list.appendChild(item);
        });

        header.addEventListener('click', () => {
            header.classList.toggle('open');
            list.classList.toggle('open');
        });

        treePanel.appendChild(header);
        treePanel.appendChild(list);
    });
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function selectFilter(id) {
    const f = FILTERS.find(x => x.id === id);
    if (!f) return;

    // Update active
    if (activeItem) activeItem.classList.remove('active');
    const el = treePanel.querySelector(`.tree-item[data-id="${id}"]`);
    if (el) {
        el.classList.add('active');
        activeItem = el;
        // Ensure parent category is open
        const catList = el.parentElement;
        if (!catList.classList.contains('open')) {
            catList.classList.add('open');
            catList.previousElementSibling.classList.add('open');
        }
        el.scrollIntoView({block:'nearest'});
    }

    mainPanel.innerHTML = `
        <div class="filter-view">
            <div class="fv-index">Filter #${f.id}</div>
            <div class="fv-name">${esc(f.name)}</div>
            <div class="fv-desc">${esc(f.desc)}</div>
            <div class="fv-technique"><strong>Technique:</strong> ${esc(f.technique)}</div>
            ${f.code ? `
            <div class="fv-code-header">CUDA Kernel Code</div>
            <div class="fv-code-wrap">
                <pre><code class="language-cpp">${esc(f.code)}</code></pre>
            </div>` : ''}
        </div>`;

    // Apply syntax highlighting
    mainPanel.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });
}

// Search
let searchTimeout = null;
searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(doSearch, 200);
});

function doSearch() {
    const q = searchInput.value.trim().toLowerCase();

    // Clear old highlights
    treePanel.querySelectorAll('.search-hit').forEach(el => el.classList.remove('search-hit'));
    searchHits.clear();
    searchCount.textContent = '';

    if (!q) {
        // Show welcome or keep current
        return;
    }

    const terms = q.split(/\s+/);
    const results = FILTERS.filter(f => {
        const hay = (f.id + ' ' + f.name + ' ' + f.desc + ' ' + f.technique).toLowerCase();
        return terms.every(t => hay.includes(t));
    });

    searchCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
    results.forEach(f => searchHits.add(f.id));

    // Highlight tree items
    results.forEach(f => {
        const el = treePanel.querySelector(`.tree-item[data-id="${f.id}"]`);
        if (el) {
            el.classList.add('search-hit');
            // Open parent category
            const catList = el.parentElement;
            if (!catList.classList.contains('open')) {
                catList.classList.add('open');
                catList.previousElementSibling.classList.add('open');
            }
        }
    });

    // Show results in main panel
    if (results.length === 0) {
        mainPanel.innerHTML = `<div class="welcome"><h2>No Results</h2><p>No filters match "<b>${esc(q)}</b>".</p></div>`;
        return;
    }

    let html = `<div class="search-results-view"><h2>${results.length} filter${results.length !== 1 ? 's' : ''} matching "${esc(q)}"</h2>`;
    results.forEach(f => {
        html += `<div class="sr-card" data-id="${f.id}">
            <div class="sr-idx">Filter #${f.id}</div>
            <div class="sr-name">${highlight(f.name, terms)}</div>
            <div class="sr-desc">${highlight(f.desc, terms)}</div>
            <div class="sr-tech">Technique: ${highlight(f.technique, terms)}</div>
        </div>`;
    });
    html += '</div>';
    mainPanel.innerHTML = html;

    // Click cards to select
    mainPanel.querySelectorAll('.sr-card').forEach(card => {
        card.addEventListener('click', () => {
            selectFilter(parseInt(card.getAttribute('data-id')));
        });
    });
}

function highlight(text, terms) {
    let s = esc(text);
    terms.forEach(t => {
        if (!t) return;
        const re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
        s = s.replace(re, '<mark>$1</mark>');
    });
    return s;
}

// Keyboard shortcut: focus search on Ctrl+K or /
document.addEventListener('keydown', e => {
    if ((e.ctrlKey && e.key === 'k') || (e.key === '/' && document.activeElement !== searchInput)) {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
    }
    if (e.key === 'Escape') {
        searchInput.blur();
    }
});

buildTree();
showUnifiedKernel();
</script>
</body>
</html>"""

def generate_html(filters, output_path, full_kernel=""):
    cats_json = json.dumps(CATEGORIES)
    filters_json = json.dumps(filters, ensure_ascii=False)
    kernel_json = json.dumps(full_kernel, ensure_ascii=False)

    content = HTML_TEMPLATE.replace('%%FILTERS_JSON%%', filters_json)
    content = content.replace('%%CATEGORIES_JSON%%', cats_json)
    content = content.replace('%%FULL_KERNEL_JSON%%', kernel_json)

    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated {output_path} with {len(filters)} filters.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(script_dir, 'acidcam-gpu', 'src', 'filters.cu')

    filters = parse_filters(os.path.join(script_dir, 'FILTERS.md'))
    print(f"Parsed {len(filters)} filters from FILTERS.md")

    # Extract kernel code from .cu file
    full_kernel = ""
    if os.path.exists(cu_path):
        kernel_code, full_kernel = parse_kernel_code(cu_path)
        print(f"Extracted kernel code for {len(kernel_code)} cases from filters.cu")
        print(f"Full kernel function: {len(full_kernel)} chars, {full_kernel.count(chr(10))+1} lines")
        for f in filters:
            if f['id'] in kernel_code:
                f['code'] = kernel_code[f['id']]
        with_code = sum(1 for f in filters if 'code' in f)
        print(f"Attached code to {with_code}/{len(filters)} filters")
    else:
        print(f"Warning: {cu_path} not found, skipping kernel code extraction")

    generate_html(filters, os.path.join(script_dir, 'filter_browser.html'), full_kernel)
