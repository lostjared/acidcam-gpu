#!/usr/bin/env python3
"""
Generate effects_reference.md from CUDA filters and GLSL shaders.
Mirrors gen_shader_browser.py but outputs Markdown instead of HTML.
Imports filter data from gen_filter_viewer_md.py to produce a combined document.
"""

import os
import re
import glob


SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effects_reference.md")


def strip_comments(code: str) -> str:
    """Remove // and /* */ comments from GLSL code."""
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//[^\n]*', '', code)
    lines = [l for l in code.split('\n') if l.strip()]
    return '\n'.join(lines)


def extract_comments(code: str) -> list[str]:
    """Extract meaningful comment text from the shader, filtering boilerplate."""
    comments = []
    skip_patterns = [
        r'^-+$', r'^=+$', r'^~+$', r'^\*+$',
        r'^---\s*\w+\s*---$',
        r'^={5,}',
        r'^\d+\.',
        r'^(input|output|final|uniform|strict|version|helper|layout|sampler)',
        r'^#',
        r'^(in|out|void|float|int|vec|mat|sampler|uniform)\b',
    ]
    skip_re = [re.compile(p, re.IGNORECASE) for p in skip_patterns]
    section_header_re = re.compile(r'^---\s*.+\s*---$')

    def is_useful(text: str) -> bool:
        if len(text) < 8:
            return False
        for p in skip_re:
            if p.search(text):
                return False
        if text.isupper() and len(text) < 30:
            return False
        if section_header_re.match(text):
            return False
        return True

    for m in re.finditer(r'/\*(.*?)\*/', code, re.DOTALL):
        raw = m.group(1).strip()
        lines = []
        for line in raw.split('\n'):
            line = re.sub(r'^\s*\*\s?', '', line).strip()
            if line:
                lines.append(line)
        text = ' '.join(lines)
        if is_useful(text):
            comments.append(text)

    for m in re.finditer(r'//\s*(.*)', code):
        text = m.group(1).strip()
        if is_useful(text):
            comments.append(text)

    return comments


def humanize_name(filename: str) -> str:
    """Convert filename to human-readable name."""
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    return name


def categorize_shader(filename: str, code: str) -> str:
    """Auto-categorize a shader based primarily on filename keywords, with code as fallback."""
    fn = filename.lower().replace('.glsl', '')

    fn_categories = [
        ("Fractal & Mandelbrot", ["fractal", "mandelbrot", "julia", "mandel"]),
        ("Cursor & Mouse", ["cursor", "mouse", "bubble"]),
        ("Liquid & Fluid", ["liquid", "fluid", "water", "ripple", "ocean", "aqua"]),
        ("Glitch & Digital", ["glitch", "digital", "8bit", "vhs", "scanline", "corrupt", "pixel"]),
        ("Mirror & Symmetry", ["mirror", "symmetr", "reflect", "flip", "fold"]),
        ("Light & Electric", ["light", "electric", "flash", "spark", "lightning", "neon", "laser"]),
        ("Blur & Smooth", ["blur", "smooth", "gaussian", "bokeh", "bloom", "soft"]),
        ("Edge & Outline", ["edge", "outline", "contour", "sobel", "laplace", "detect"]),
        ("Tile & Mosaic", ["tile", "mosaic", "grid", "cell", "block", "checker", "kaleidoscope"]),
        ("XOR & Math", ["xor", "xorkern"]),
        ("Acid & Psychedelic", ["acid", "psyche", "trip", "hallucin", "dream"]),
        ("Noise & Random", ["noise", "rand", "random", "perlin", "simplex", "voronoi", "static"]),
        ("Color Manipulation", ["color", "rgb", "hue", "saturat", "hsv", "hsl", "palette", "gradient", "rainbow", "invert", "sepia", "gray", "grey", "tint", "chroma"]),
        ("Distortion & Warp", ["warp", "distort", "bend", "twist", "swirl", "curl", "stretch", "squeeze", "barrel", "fisheye", "lens", "dispers"]),
        ("Blend & Composite", ["blend", "composite", "overlay", "multiply"]),
        ("Wave & Oscillation", ["wave", "oscillat", "pulse", "vibrat"]),
        ("Trail & Motion", ["trail", "motion", "temporal", "feedback", "echo", "ghost"]),
        ("Scale & Transform", ["scale", "zoom", "rotate", "resize", "pan", "scroll"]),
        ("Audio Reactive", ["amp_", "_amp", "uamp", "audio", "beat", "sound", "spectrum"]),
    ]

    for cat_name, keywords in fn_categories:
        for kw in keywords:
            if kw in fn:
                return cat_name

    cl = code.lower()
    code_categories = [
        ("Fractal & Mandelbrot", lambda c: "mandelbrot" in c or ("z = " in c and "c = " in c and "max_iter" in c)),
        ("Audio Reactive", lambda c: "uniform float amp" in c and "uniform float uamp" in c),
        ("Cursor & Mouse", lambda c: "imouse.z" in c or "imouse.w" in c),
        ("Liquid & Fluid", lambda c: "fluid" in c or ("ripple" in c and "wave" in c)),
        ("Distortion & Warp", lambda c: "warp" in c or "distort" in c or ("barrel" in c and "distortion" in c)),
        ("Blur & Smooth", lambda c: c.count("texture(samp") > 4 and ("blur" in c or "kernel" in c or "offset" in c)),
        ("Edge & Outline", lambda c: "sobel" in c or "laplacian" in c or ("edge" in c and "detect" in c)),
        ("Glitch & Digital", lambda c: "glitch" in c or ("scanline" in c and "noise" in c)),
        ("Mirror & Symmetry", lambda c: fn.count("mirror") > 0 or ("1.0 - " in c and "abs(" in c and "mirror" in c)),
        ("Color Manipulation", lambda c: "hsv" in c or "hsl" in c or "rgb2hsv" in c or "hue" in c),
        ("XOR & Math", lambda c: "^^" in c or "xor" in c),
    ]

    for cat_name, check_fn in code_categories:
        try:
            if check_fn(cl):
                return cat_name
        except Exception:
            pass

    return "Other Effects"


def analyze_shader(code: str) -> dict:
    """Deep analysis of what a shader actually does, returning structured traits."""
    cl = code.lower()
    traits = {}

    traits["uses_time"] = "time_f" in cl
    traits["uses_mouse"] = "imouse" in cl
    traits["uses_audio"] = bool(re.search(r'uniform\s+float\s+(amp|uamp)\b', cl))
    traits["uses_resolution"] = "iresolution" in cl
    traits["uses_seed"] = "uniform float seed" in cl
    traits["uses_alpha"] = bool(re.search(r'uniform\s+float\s+alpha', cl))

    tex_count = cl.count("texture(samp") + cl.count("texture2d(samp")
    traits["tex_samples"] = tex_count
    traits["multi_sample"] = tex_count > 3
    traits["samples_at_offsets"] = bool(re.search(r'texture\(samp\s*,\s*tc\s*[+\-]', cl))
    traits["has_cache_textures"] = "samp1" in cl or "samp2" in cl or "cache" in cl.replace("_cache", "")

    traits["hsv_convert"] = "hsv" in cl or "rgb2hsv" in cl or "hsv2rgb" in cl
    traits["hsl_convert"] = "hsl" in cl
    traits["luminance"] = "luminance" in cl or "luma" in cl or bool(re.search(r'0\.299|0\.2126', cl))
    traits["channel_swap"] = bool(re.search(r'color\s*=\s*vec[34]\(.+\.(gbr|brg|grb|rbg|bgr)', cl))

    traits["xor_op"] = "^" in code and bool(re.search(r'\^\s*\d|\^\s*source|\^\s*int_color', cl))
    traits["mod_wrap"] = "mod(" in cl
    traits["sin_wave"] = "sin(" in cl
    traits["cos_wave"] = "cos(" in cl
    traits["atan_polar"] = "atan(" in cl
    traits["smoothstep"] = "smoothstep(" in cl
    traits["pow_func"] = "pow(" in cl
    traits["abs_func"] = "abs(" in cl
    traits["fract_func"] = "fract(" in cl
    traits["floor_func"] = "floor(" in cl
    traits["mix_blend"] = "mix(" in cl

    for_loops = re.findall(r'for\s*\(', cl)
    traits["loop_count"] = len(for_loops)
    max_iter_match = re.search(r'max_iter\w*\s*=\s*(\d+)', cl)
    traits["max_iterations"] = int(max_iter_match.group(1)) if max_iter_match else 0

    traits["matrix_transform"] = "mat2" in cl or "mat3" in cl or "mat4" in cl
    traits["rotation"] = bool(re.search(r'rotate|mat2\s*\(.*cos', cl))
    traits["mirror_symmetry"] = "1.0 - tc" in cl or "1.0-tc" in cl or ("abs(" in cl and "mirror" in cl.replace("_", ""))
    traits["uv_scale"] = bool(re.search(r'tc\s*\*\s*\d|uv\s*\*\s*\d', cl))
    traits["polar_coords"] = "atan(" in cl and "length(" in cl

    traits["edge_detect"] = bool(re.search(r'sobel|laplacian|edgedetect|edge.*detection|gx.*gy', cl))
    traits["blur_kernel"] = traits["multi_sample"] and bool(re.search(r'offset|kernel|gauss|blur|weight', cl))
    traits["color_quantize"] = "floor(" in cl and bool(re.search(r'floor\(.+\*\s*\d+', cl))
    traits["palette_map"] = bool(re.search(r'palette|nespalette|float\[\d+\]', cl))
    traits["feedback_echo"] = traits["has_cache_textures"]
    traits["noise_gen"] = bool(re.search(r'fract\(sin\(dot\(', cl)) or "noise(" in cl or "perlin" in cl
    traits["fractal_iter"] = traits["max_iterations"] > 10 or ("z =" in cl and bool(re.search(r'z\s*=\s*vec2', cl)))
    traits["distance_field"] = "distance(" in cl or "sdf" in cl
    traits["vignette"] = "vignette" in cl or bool(re.search(r'distance.*0\.5.*0\.5', cl))
    traits["chromatic_aberration"] = bool(re.search(r'texture.*\+.*offset.*\.r.*texture.*\-.*offset.*\.b|chrom', cl))
    traits["scanlines"] = "scanline" in cl
    traits["discard_black"] = "discard" in cl

    return traits


def generate_description(filename: str, comments: list[str], code: str, traits: dict) -> tuple[str, str]:
    """Generate a human-readable explanation and technique description from code analysis."""
    name = humanize_name(filename)

    good_comments = [c for c in comments if len(c) > 15 and not c.startswith(('Ensure', 'We '))]
    if good_comments:
        desc_from_comments = ' '.join(good_comments[:3])
        if len(desc_from_comments) > 350:
            desc_from_comments = desc_from_comments[:347] + '...'
    else:
        desc_from_comments = None

    desc_parts = []

    if traits["fractal_iter"]:
        iters = traits["max_iterations"] or "many"
        desc_parts.append(f"Renders a fractal pattern using iterative computation ({iters} iterations)")
    elif traits["edge_detect"]:
        desc_parts.append("Performs edge detection on the input image, highlighting boundaries and contours")
    elif traits["blur_kernel"]:
        desc_parts.append("Applies a blur/smoothing convolution to the input image using multi-sample averaging")
    elif traits["palette_map"] and traits["color_quantize"]:
        desc_parts.append("Quantizes colors to a restricted palette, producing a retro/pixelated aesthetic")
    elif traits["color_quantize"]:
        desc_parts.append("Reduces the color depth by quantizing pixel values, creating a posterized look")
    elif traits["noise_gen"] and traits["sin_wave"]:
        desc_parts.append("Generates procedural noise patterns combined with sinusoidal distortion")
    elif traits["noise_gen"]:
        desc_parts.append("Uses procedural noise generation to create organic, randomized visual patterns")
    elif traits["feedback_echo"]:
        desc_parts.append("Blends the current frame with cached previous frames to create motion trails and temporal echoes")
    elif traits["xor_op"]:
        desc_parts.append("Applies bitwise XOR operations on pixel color channels, creating digital-art style interference patterns")
    elif traits["chromatic_aberration"]:
        desc_parts.append("Simulates chromatic aberration by offsetting color channels, mimicking lens imperfections")
    elif traits["polar_coords"] and traits["sin_wave"]:
        desc_parts.append("Transforms the image into polar coordinates and applies wave-based distortion for a swirling effect")
    elif traits["mirror_symmetry"]:
        desc_parts.append("Creates mirror/symmetry effects by reflecting or folding the texture coordinates")
    elif traits["scanlines"]:
        desc_parts.append("Overlays CRT-style scanlines onto the image for a retro display effect")
    elif traits["vignette"]:
        desc_parts.append("Applies a vignette effect, darkening the image edges while keeping the center bright")
    elif traits["hsv_convert"]:
        desc_parts.append("Converts to HSV color space for hue/saturation manipulation, then converts back to RGB")
    elif traits["distance_field"] and traits["sin_wave"]:
        desc_parts.append("Uses distance fields with sinusoidal modulation to generate animated visual patterns")
    elif traits["rotation"] and traits["uv_scale"]:
        desc_parts.append("Applies geometric transformations including rotation and scaling to distort the image")
    elif traits["mix_blend"] and traits["multi_sample"]:
        desc_parts.append("Blends multiple texture samples together using interpolation for a composite effect")
    elif traits["sin_wave"] and traits["cos_wave"] and traits["uses_time"]:
        desc_parts.append("Animates the image using time-driven sinusoidal and cosinusoidal wave distortions")
    elif traits["sin_wave"] and traits["uses_time"]:
        desc_parts.append("Applies time-animated sinusoidal transformations to the image for a flowing, organic look")
    elif traits["multi_sample"]:
        desc_parts.append("Samples the texture at multiple coordinates and composites the results")
    elif traits["mod_wrap"] and traits["uses_time"]:
        desc_parts.append("Uses modular arithmetic with time-based animation to create repeating, evolving patterns")
    elif traits["uses_time"] and traits["loop_count"] > 0:
        desc_parts.append("Applies an iterative, time-animated effect that evolves the image over time")
    elif traits["uses_time"]:
        desc_parts.append("Applies a time-animated visual transformation to the input texture")
    else:
        desc_parts.append(f"Processes the input texture to produce the \"{name}\" visual effect")

    interaction = []
    if traits["uses_mouse"]:
        interaction.append("mouse position")
    if traits["uses_audio"]:
        interaction.append("audio input (amplitude/volume)")
    if traits["uses_seed"]:
        interaction.append("random seed value")
    if interaction:
        desc_parts.append(f"Responds to {' and '.join(interaction)} for interactive control")

    details = []
    if traits["discard_black"]:
        details.append("discards pure black pixels for transparency")
    if traits["luminance"]:
        details.append("computes luminance for brightness-based effects")
    if traits["channel_swap"]:
        details.append("swaps color channels for chromatic shifts")
    if traits["matrix_transform"] and not traits["rotation"]:
        details.append("applies matrix transformations")
    if details:
        desc_parts.append(". ".join(d.capitalize() for d in details))

    synthetic_desc = ". ".join(desc_parts) + "."
    if desc_from_comments and len(desc_from_comments) > 40:
        desc = synthetic_desc + " " + desc_from_comments
    else:
        desc = synthetic_desc

    if len(desc) > 500:
        desc = desc[:497] + "..."

    tech_parts = []

    if traits["fractal_iter"]:
        tech_parts.append("Iterative fractal computation with complex-plane mapping")
    if traits["edge_detect"]:
        tech_parts.append("Sobel/Laplacian convolution kernel for edge detection")
    if traits["blur_kernel"]:
        n = traits["tex_samples"]
        tech_parts.append(f"Multi-tap convolution blur ({n} texture samples)")
    if traits["noise_gen"]:
        tech_parts.append("Pseudo-random noise via fract(sin(dot())) hash function")
    if traits["xor_op"]:
        tech_parts.append("Bitwise XOR on integer color channels")
    if traits["polar_coords"]:
        tech_parts.append("Polar coordinate transformation (atan2 + length)")
    if traits["hsv_convert"]:
        tech_parts.append("RGB ↔ HSV color space conversion")
    if traits["hsl_convert"]:
        tech_parts.append("RGB ↔ HSL color space conversion")
    if traits["palette_map"]:
        tech_parts.append("Indexed color palette look-up table")
    if traits["color_quantize"] and not traits["palette_map"]:
        tech_parts.append("Color quantization via floor rounding")
    if traits["feedback_echo"]:
        tech_parts.append("Temporal feedback using cached frame buffers")
    if traits["chromatic_aberration"]:
        tech_parts.append("Per-channel UV offset for chromatic aberration")
    if traits["scanlines"]:
        tech_parts.append("Scanline overlay using modular y-coordinate")

    if traits["rotation"]:
        tech_parts.append("2D rotation matrix")
    if traits["mirror_symmetry"] and "Mirror" not in str(tech_parts):
        tech_parts.append("UV mirroring/folding for symmetry")
    if traits["smoothstep"]:
        tech_parts.append("Hermite smoothstep interpolation")
    if traits["mix_blend"]:
        tech_parts.append("Linear mix/lerp blending")
    if traits["distance_field"] and "distance" not in str(tech_parts).lower():
        tech_parts.append("Distance field computation")
    if traits["vignette"] and "vignette" not in str(tech_parts).lower():
        tech_parts.append("Radial vignette falloff")

    input_parts = []
    if traits["uses_time"]:
        input_parts.append("time")
    if traits["uses_mouse"]:
        input_parts.append("mouse")
    if traits["uses_audio"]:
        input_parts.append("audio")
    if traits["uses_resolution"]:
        input_parts.append("resolution")
    if input_parts:
        tech_parts.append(f"Driven by: {', '.join(input_parts)}")

    if tech_parts:
        technique_str = " · ".join(tech_parts)
    else:
        technique_str = "Direct texture color manipulation with arithmetic operations"

    return desc, technique_str


def process_shaders():
    """Read shaders listed in index.txt and return structured data."""
    index_path = os.path.join(SHADER_DIR, "index.txt")
    with open(index_path, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
    files = [os.path.join(SHADER_DIR, fn) for fn in filenames]
    print(f"Loaded {len(files)} shaders from index.txt")

    shaders = []
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', errors='replace') as f:
                raw_code = f.read()
        except Exception as e:
            print(f"  Warning: Could not read {filename}: {e}")
            continue

        comments = extract_comments(raw_code)
        clean_code = strip_comments(raw_code)
        category = categorize_shader(filename, raw_code)
        traits = analyze_shader(raw_code)
        desc, technique = generate_description(filename, comments, raw_code, traits)
        name = humanize_name(filename)

        shaders.append({
            "id": i,
            "filename": filename,
            "name": name,
            "category": category,
            "desc": desc,
            "technique": technique,
            "code": clean_code,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(files)} shaders...")

    print(f"  Processed {len(files)}/{len(files)} shaders. Done.")
    return shaders


def generate_shader_markdown(shaders: list[dict]) -> str:
    """Generate the Markdown string for the GLSL shaders section."""
    lines = []
    lines.append("# Part II — GLSL Shaders\n")
    lines.append(f"This section documents all **{len(shaders)} GLSL fragment shaders** in the `shaders/` directory. "
                 "These shaders are designed for ACMX2 (AcidCam MX2), a real-time GPU-accelerated video effects "
                 "processor. Each shader processes the input texture in real time — many respond to time, mouse "
                 "position, and audio amplitude.\n")

    # Read vertex shader
    vertex_path = os.path.join(SHADER_DIR, "vertex.glsl")
    if os.path.exists(vertex_path):
        with open(vertex_path, 'r') as f:
            vertex_code = f.read()
        lines.append("## Vertex Shader\n")
        lines.append("The vertex shader is shared by all fragment shaders in ACMX2. It transforms each vertex "
                     "position by the model-view and projection matrices and passes through the texture coordinate "
                     "(`tc`) to the fragment stage.\n")
        lines.append("```glsl")
        lines.append(vertex_code.rstrip())
        lines.append("```\n")

    lines.append("---\n")

    # Group by category
    cat_map: dict[str, list[dict]] = {}
    for s in shaders:
        cat_map.setdefault(s["category"], []).append(s)

    sorted_cats = sorted(cat_map.keys(), key=lambda c: (-len(cat_map[c]), c))

    for cat in sorted_cats:
        items = cat_map[cat]
        lines.append(f"## {cat} ({len(items)} shaders)\n")

        for s in items:
            lines.append(f"### Shader #{s['id']} — {s['name']}\n")
            lines.append(f"**File:** `{s['filename']}`\n")
            if s['desc']:
                lines.append(f"{s['desc']}\n")
            if s['technique']:
                lines.append(f"**Technique:** {s['technique']}\n")
            if s.get('code'):
                lines.append("```glsl")
                lines.append(s['code'])
                lines.append("```\n")
            lines.append("---\n")

    return '\n'.join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== AcidCam GPU Effects Reference (Markdown) ===\n")

    # ---- Part I: CUDA Filters ----
    print("--- Part I: CUDA Filters ---")
    from gen_filter_viewer_md import parse_filters, parse_kernel_code, generate_filter_markdown

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

    filter_md = generate_filter_markdown(filters, full_kernel)

    # ---- Part II: GLSL Shaders ----
    print("\n--- Part II: GLSL Shaders ---")
    print(f"Shader directory: {SHADER_DIR}")

    shaders = process_shaders()
    print(f"Total shaders processed: {len(shaders)}")

    cat_counts: dict[str, int] = {}
    for s in shaders:
        cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    shader_md = generate_shader_markdown(shaders)

    # ---- Combine into effects_reference.md ----
    header = (
        "# AcidCam GPU — Effects Reference\n\n"
        "Complete reference for all CUDA filter kernels and GLSL fragment shaders in ACMX2.\n\n"
        f"- **CUDA Filters:** {len(filters)}\n"
        f"- **GLSL Shaders:** {len(shaders)}\n\n"
        "---\n\n"
    )

    combined = header + filter_md + "\n\n" + shader_md

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(combined)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nWritten: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
