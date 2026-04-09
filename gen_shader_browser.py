#!/usr/bin/env python3
"""
Generate shader_browser.html from all .glsl files in the shaders/ directory.
Mirrors the style of filter_browser.html but for GLSL shaders.
Processes shaders in chunks to handle large counts.
"""

import os
import re
import json
import glob
import html

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shader_browser.html")


def strip_comments(code: str) -> str:
    """Remove // and /* */ comments from GLSL code."""
    # Remove block comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove line comments
    code = re.sub(r'//[^\n]*', '', code)
    # Remove blank lines that result
    lines = [l for l in code.split('\n') if l.strip()]
    return '\n'.join(lines)


def extract_comments(code: str) -> list[str]:
    """Extract meaningful comment text from the shader, filtering boilerplate."""
    comments = []
    skip_patterns = [
        r'^-+$', r'^=+$', r'^~+$', r'^\*+$',
        r'^---\s*\w+\s*---$',            # --- SECTION ---
        r'^={5,}',                         # ======= dividers
        r'^\d+\.',                         # numbered steps like "1. foo"
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
        # Skip ALL-CAPS short headers like "UTILITIES", "NOISE", "CONTROLS"
        if text.isupper() and len(text) < 30:
            return False
        # Skip section-header comments
        if section_header_re.match(text):
            return False
        return True

    # Block comments
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

    # Line comments — gather consecutive comment blocks
    for m in re.finditer(r'//\s*(.*)', code):
        text = m.group(1).strip()
        if is_useful(text):
            comments.append(text)

    return comments


def humanize_name(filename: str) -> str:
    """Convert filename to human-readable name."""
    name = os.path.splitext(filename)[0]
    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    return name


def categorize_shader(filename: str, code: str) -> str:
    """Auto-categorize a shader based primarily on filename keywords, with code as fallback."""
    fn = filename.lower().replace('.glsl', '')

    # Primary: match on filename only (most reliable signal)
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

    # Secondary: check code content for specific patterns (not generic words like 'color')
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

    # --- Inputs & uniforms ---
    traits["uses_time"] = "time_f" in cl
    traits["uses_mouse"] = "imouse" in cl
    traits["uses_audio"] = bool(re.search(r'uniform\s+float\s+(amp|uamp)\b', cl))
    traits["uses_resolution"] = "iresolution" in cl
    traits["uses_seed"] = "uniform float seed" in cl
    traits["uses_alpha"] = bool(re.search(r'uniform\s+float\s+alpha', cl))

    # --- Texture sampling patterns ---
    tex_count = cl.count("texture(samp") + cl.count("texture2d(samp")
    traits["tex_samples"] = tex_count
    traits["multi_sample"] = tex_count > 3
    traits["samples_at_offsets"] = bool(re.search(r'texture\(samp\s*,\s*tc\s*[+\-]', cl))
    traits["has_cache_textures"] = "samp1" in cl or "samp2" in cl or "cache" in cl.replace("_cache", "")

    # --- Color space ---
    traits["hsv_convert"] = "hsv" in cl or "rgb2hsv" in cl or "hsv2rgb" in cl
    traits["hsl_convert"] = "hsl" in cl
    traits["luminance"] = "luminance" in cl or "luma" in cl or bool(re.search(r'0\.299|0\.2126', cl))
    traits["channel_swap"] = bool(re.search(r'color\s*=\s*vec[34]\(.+\.(gbr|brg|grb|rbg|bgr)', cl))

    # --- Math patterns ---
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

    # --- Iteration ---
    for_loops = re.findall(r'for\s*\(', cl)
    traits["loop_count"] = len(for_loops)
    max_iter_match = re.search(r'max_iter\w*\s*=\s*(\d+)', cl)
    traits["max_iterations"] = int(max_iter_match.group(1)) if max_iter_match else 0

    # --- Geometric transforms ---
    traits["matrix_transform"] = "mat2" in cl or "mat3" in cl or "mat4" in cl
    traits["rotation"] = bool(re.search(r'rotate|mat2\s*\(.*cos', cl))
    traits["mirror_symmetry"] = "1.0 - tc" in cl or "1.0-tc" in cl or ("abs(" in cl and "mirror" in cl.replace("_", ""))
    traits["uv_scale"] = bool(re.search(r'tc\s*\*\s*\d|uv\s*\*\s*\d', cl))
    traits["polar_coords"] = "atan(" in cl and "length(" in cl

    # --- Visual effect type ---
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

    # ---- BUILD DESCRIPTION ----
    # First try to use meaningful comments
    good_comments = [c for c in comments if len(c) > 15 and not c.startswith(('Ensure', 'We '))]
    if good_comments:
        desc_from_comments = ' '.join(good_comments[:3])
        if len(desc_from_comments) > 350:
            desc_from_comments = desc_from_comments[:347] + '...'
    else:
        desc_from_comments = None

    # Build a synthetic description from code analysis
    desc_parts = []

    # Effect type
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

    # Add interaction info
    interaction = []
    if traits["uses_mouse"]:
        interaction.append("mouse position")
    if traits["uses_audio"]:
        interaction.append("audio input (amplitude/volume)")
    if traits["uses_seed"]:
        interaction.append("random seed value")
    if interaction:
        desc_parts.append(f"Responds to {' and '.join(interaction)} for interactive control")

    # Add detail about techniques used
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

    # Combine: prefer synthetic + comments supplement
    synthetic_desc = ". ".join(desc_parts) + "."
    if desc_from_comments and len(desc_from_comments) > 40:
        # Use comments if they add real info beyond what we detected
        desc = synthetic_desc + " " + desc_from_comments
    else:
        desc = synthetic_desc

    if len(desc) > 500:
        desc = desc[:497] + "..."

    # ---- BUILD TECHNIQUE ----
    tech_parts = []

    # Core technique
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

    # Secondary techniques
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

    # Inputs
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
    """Read all shaders and return structured data."""
    pattern = os.path.join(SHADER_DIR, "*.glsl")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} shader files")

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


def build_html(shaders: list[dict]) -> str:
    """Build the full HTML page."""

    # Group by category and sort categories
    cat_map: dict[str, list[dict]] = {}
    for s in shaders:
        cat_map.setdefault(s["category"], []).append(s)

    # Sort categories by count descending, then alphabetically
    sorted_cats = sorted(cat_map.keys(), key=lambda c: (-len(cat_map[c]), c))

    # Build CATEGORIES and SHADERS JSON
    categories_js = []
    for cat in sorted_cats:
        items = cat_map[cat]
        lo = min(s["id"] for s in items)
        hi = max(s["id"] for s in items)
        categories_js.append([cat, lo, hi])

    shaders_js = []
    for s in shaders:
        shaders_js.append({
            "id": s["id"],
            "name": s["name"],
            "filename": s["filename"],
            "category": s["category"],
            "desc": s["desc"],
            "technique": s["technique"],
            "code": s["code"],
        })

    # Read vertex shader
    vertex_path = os.path.join(SHADER_DIR, "vertex.glsl")
    vertex_code = ""
    if os.path.exists(vertex_path):
        with open(vertex_path, 'r') as f:
            vertex_code = f.read()
    vertex_code_json = json.dumps(vertex_code, ensure_ascii=False)

    shaders_json = json.dumps(shaders_js, ensure_ascii=False)
    categories_json = json.dumps(sorted_cats, ensure_ascii=False)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AcidCam GPU Shader Browser</title>
<meta name="description" content="Browse {len(shaders)} GLSL fragment shaders for ACMX2, a real-time GPU-accelerated video effects processor. View source code with syntax highlighting, search by technique, and explore shader categories.">
<meta name="author" content="Jared Bruni (lostjared)">
<meta name="keywords" content="GLSL, shaders, ACMX2, AcidCam, GPU, OpenGL, CUDA, real-time video effects, fragment shader, visual effects">

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:title" content="AcidCam GPU &mdash; GLSL Shader Browser">
<meta property="og:description" content="Browse {len(shaders)} GLSL fragment shaders for ACMX2. View source code, descriptions, and techniques for real-time GPU video effects.">
<meta property="og:image" content="https://github.com/lostjared/acidcam-gpu/raw/main/acmx2.png">
<meta property="og:url" content="https://lostsidedead.biz/acmx2/shader_browser.html">
<meta property="og:site_name" content="LostSideDead">

<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="AcidCam GPU &mdash; GLSL Shader Browser">
<meta name="twitter:description" content="Browse {len(shaders)} GLSL fragment shaders for ACMX2. Real-time GPU video effects with source code and syntax highlighting.">
<meta name="twitter:image" content="https://github.com/lostjared/acidcam-gpu/raw/main/acmx2.png">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/glsl.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ height:100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#f0f4f8; color:#1a1a2e; }}

.topbar {{
    display:flex; align-items:center; gap:16px;
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%);
    color:#fff; padding:12px 24px; height:60px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}}
.topbar h1 {{ font-size:20px; white-space:nowrap; }}
.search-box {{
    flex:1; max-width:420px; position:relative;
}}
.search-box input {{
    width:100%; padding:8px 12px 8px 36px;
    border:none; border-radius:6px; font-size:14px;
    background:rgba(255,255,255,0.15); color:#fff;
    outline:none; transition: background 0.2s;
}}
.search-box input::placeholder {{ color:rgba(255,255,255,0.6); }}
.search-box input:focus {{ background:rgba(255,255,255,0.25); }}
.search-box svg {{
    position:absolute; left:10px; top:50%; transform:translateY(-50%);
    width:16px; height:16px; fill:rgba(255,255,255,0.6);
}}
.search-results-count {{
    font-size:12px; color:rgba(255,255,255,0.7); white-space:nowrap;
}}

.container {{ display:flex; height:calc(100% - 60px); }}

.tree-panel {{
    width:340px; min-width:260px; background:#fff;
    border-right:2px solid #bbdefb;
    overflow-y:auto; padding:8px 0;
    scrollbar-width:thin;
}}
.tree-panel::-webkit-scrollbar {{ width:6px; }}
.tree-panel::-webkit-scrollbar-thumb {{ background:#90caf9; border-radius:3px; }}

.cat-header {{
    display:flex; align-items:center; gap:6px;
    padding:8px 12px; cursor:pointer;
    font-weight:700; font-size:13px; color:#1a237e;
    background:#e3f2fd; border-bottom:1px solid #bbdefb;
    user-select:none; position:sticky; top:0; z-index:2;
    transition: background 0.15s;
}}
.cat-header:hover {{ background:#bbdefb; }}
.cat-header .arrow {{
    display:inline-block; width:16px; text-align:center;
    font-size:10px; transition:transform 0.2s;
}}
.cat-header.open .arrow {{ transform:rotate(90deg); }}
.cat-header .count {{
    margin-left:auto; font-weight:400; font-size:11px;
    color:#5c6bc0; background:#c5cae9; border-radius:10px;
    padding:1px 8px;
}}
.cat-items {{ display:none; }}
.cat-items.open {{ display:block; }}

.tree-item {{
    display:flex; align-items:baseline; gap:6px;
    padding:5px 12px 5px 28px; cursor:pointer;
    font-size:13px; color:#333; border-left:3px solid transparent;
    transition: background 0.1s, border-color 0.1s;
}}
.tree-item:hover {{ background:#e8eaf6; }}
.tree-item.active {{ background:#e3f2fd; border-left-color:#1565c0; font-weight:600; }}
.tree-item .idx {{ color:#1565c0; font-weight:700; font-size:12px; min-width:32px; }}
.tree-item .fname {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.tree-item.search-hit {{ background:#fff9c4; }}
.tree-item.search-hit.active {{ background:#fff176; border-left-color:#f9a825; }}

.main-panel {{
    flex:1; overflow-y:auto; padding:32px 48px;
    background:#f8fafc;
}}
.main-panel::-webkit-scrollbar {{ width:6px; }}
.main-panel::-webkit-scrollbar-thumb {{ background:#90caf9; border-radius:3px; }}

.welcome {{
    text-align:center; margin-top:15vh; color:#90a4ae;
}}
.welcome h2 {{ font-size:28px; margin-bottom:12px; color:#78909c; }}
.welcome p {{ font-size:15px; }}

.filter-view {{ animation: fadeIn 0.2s ease; }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}

.filter-view .fv-index {{
    font-size:14px; font-weight:700; color:#1565c0;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;
}}
.filter-view .fv-name {{
    font-size:32px; font-weight:800; color:#1a1a2e;
    margin-bottom:8px; line-height:1.2;
}}
.filter-view .fv-filename {{
    font-size:13px; color:#78909c; margin-bottom:20px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
}}
.filter-view .fv-desc {{
    font-size:16px; line-height:1.7; color:#222;
    margin-bottom:24px; max-width:720px;
}}
.filter-view .fv-technique {{
    display:inline-block;
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left:4px solid #1565c0;
    padding:12px 20px; border-radius:0 8px 8px 0;
    font-size:14px; color:#0d47a1; max-width:720px;
    margin-bottom:8px;
}}
.filter-view .fv-technique strong {{ color:#b71c1c; }}

.fv-code-header {{
    margin-top:28px; margin-bottom:8px;
    font-size:14px; font-weight:700; color:#1a237e;
    text-transform:uppercase; letter-spacing:1px;
}}
.fv-code-wrap {{
    max-width:820px; border-radius:8px; overflow:hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.fv-code-wrap pre {{
    margin:0; padding:16px 20px; font-size:13px; line-height:1.6;
    overflow-x:auto;
}}
.fv-code-wrap code {{ font-family: 'Cascadia Code', 'Fira Code', 'Source Code Pro', Consolas, monospace; }}

.search-results-view {{ animation: fadeIn 0.2s ease; }}
.search-results-view h2 {{
    font-size:22px; color:#1a237e; margin-bottom:16px;
    border-bottom:2px solid #bbdefb; padding-bottom:8px;
}}
.sr-card {{
    background:#fff; border:1px solid #e0e0e0; border-radius:8px;
    padding:16px 20px; margin-bottom:12px; cursor:pointer;
    transition: box-shadow 0.15s, border-color 0.15s;
}}
.sr-card:hover {{ box-shadow:0 2px 12px rgba(21,101,192,0.12); border-color:#90caf9; }}
.sr-card .sr-idx {{ font-size:12px; font-weight:700; color:#1565c0; }}
.sr-card .sr-name {{ font-size:18px; font-weight:700; color:#1a1a2e; margin:2px 0 6px; }}
.sr-card .sr-desc {{ font-size:13px; color:#555; line-height:1.5; }}
.sr-card .sr-tech {{ font-size:12px; color:#b71c1c; margin-top:6px; font-style:italic; }}
mark {{ background:#fff176; color:#1a1a2e; border-radius:2px; padding:0 1px; }}

@media (max-width: 767px) {{
    .topbar {{ padding:8px 12px; height:52px; gap:8px; }}
    .topbar h1 {{ font-size:15px; }}
    .search-box {{ max-width:none; flex:1; }}
    .container {{ flex-direction:row; height:calc(100% - 52px); }}
    .tree-panel {{
        width:42vw; min-width:120px; max-width:180px;
        overflow-y:auto; overflow-x:hidden;
        scrollbar-width:thin; flex-shrink:0;
    }}
    .cat-header {{ padding:6px 8px; font-size:11px; }}
    .cat-header .count {{ padding:1px 5px; font-size:10px; }}
    .tree-item {{ padding:5px 6px 5px 16px; font-size:11px; }}
    .tree-item .idx {{ min-width:24px; font-size:10px; }}
    .main-panel {{ flex:1; min-width:0; padding:16px 14px; overflow-y:auto; }}
    .filter-view .fv-name {{ font-size:22px; margin-bottom:14px; }}
    .filter-view .fv-desc {{ font-size:14px; }}
    .filter-view .fv-technique {{ font-size:13px; padding:10px 14px; }}
    .fv-code-wrap pre {{ font-size:11px; padding:12px 14px; }}
    .welcome {{ margin-top:8vh; }}
    .welcome h2 {{ font-size:20px; }}
}}
</style>
</head>
<body>

<div class="topbar">
    <h1>AcidCam GPU &mdash; GLSL Shader Browser</h1>
    <div class="search-box">
        <svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
        <input type="text" id="searchInput" placeholder="Search shaders by name, description, or technique..." autocomplete="off">
    </div>
    <span class="search-results-count" id="searchCount"></span>
</div>

<div class="container">
    <div class="tree-panel" id="treePanel"></div>
    <div class="main-panel" id="mainPanel">
        <div class="welcome">
            <h2>GLSL Shader Browser</h2>
            <p>{len(shaders)} shaders &bull; Select a shader from the tree or use search (Ctrl+K)</p>
        </div>
    </div>
</div>

<script>
const SHADERS = {shaders_json};
const CATEGORIES = {categories_json};
const VERTEX_CODE = {vertex_code_json};

const treePanel = document.getElementById('treePanel');
const mainPanel = document.getElementById('mainPanel');
const searchInput = document.getElementById('searchInput');
const searchCount = document.getElementById('searchCount');
let activeItem = null;
const searchHits = new Set();

function esc(s) {{
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}}

function showAbout() {{
    if (activeItem) {{ activeItem.classList.remove('active'); activeItem = null; }}
    const el = treePanel.querySelector('.tree-item[data-id="about"]');
    if (el) {{ el.classList.add('active'); activeItem = el; }}

    mainPanel.innerHTML = `
        <div class="filter-view">
            <div class="fv-index">About</div>
            <div class="fv-name">ACMX2 &mdash; GLSL Shader Library</div>
            <div class="fv-desc" style="max-width:780px">
                <p>These GLSL fragment shaders are designed for <strong>ACMX2</strong> (AcidCam MX2), a real-time GPU-accelerated video effects processor for Linux. ACMX2 uses NVIDIA CUDA for its core filter pipeline and OpenGL GLSL shaders for additional visual effects that run on the GPU.</p>
                <br>
                <p>ACMX2 is built locally using the included <code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">Containerfile.arch</code> in the <code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">podman/</code> directory, or natively on Arch Linux using the build scripts in <code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">build-script/</code>. It requires an <strong>NVIDIA GPU</strong> with proprietary drivers. It captures live webcam input and processes each frame through configurable filter chains and shader effects in real time.</p>
                <br>
                <p><strong>System Requirements:</strong></p>
                <ul style="margin:8px 0 8px 24px;line-height:2">
                    <li>Linux (x86_64) with NVIDIA GPU</li>
                    <li>NVIDIA proprietary drivers + Container Toolkit</li>
                    <li>Podman (for container build) or Arch Linux (for native build)</li>
                    <li>X11 or XWayland</li>
                    <li>Webcam device (<code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">/dev/video0</code>)</li>
                    <li>Audio input device (microphone) for audio-reactive shaders</li>
                </ul>
                <br>
                <p><strong>Quick Start (Container Build):</strong></p>
                <div class="fv-code-wrap" style="max-width:640px;margin:8px 0 16px">
                    <pre><code class="language-bash">cd podman
podman build -t acmx2-arch:latest -f Containerfile.arch .
chmod +x run-acmx2-arch.sh
./run-acmx2-arch.sh</code></pre>
                </div>
                <p><strong>Native Build (Arch Linux):</strong></p>
                <div class="fv-code-wrap" style="max-width:640px;margin:8px 0 16px">
                    <pre><code class="language-bash">sudo bash build-script/install-deps-arch.sh
sudo bash build-script/acidcam-gpu-arch.sh</code></pre>
                </div>
                <p>The container run script automatically detects webcam devices, enables NVIDIA GPU passthrough, mounts PulseAudio for audio input, and opens the ACMX2 interface on your desktop.</p>
                <br>
                <p><strong>Downloads &amp; Links:</strong></p>
                <ul style="margin:8px 0 8px 24px;line-height:2.2">
                    <li><a href="https://lostsidedead.biz/acmx2/shaders.zip" style="color:#1565c0;font-weight:600">&#11015; Download Shader Pack (shaders.zip)</a></li>
                    <li><a href="https://lostsidedead.biz/packs/" style="color:#1565c0">Additional Shader &amp; Model Packs</a></li>
                    <li><a href="https://lostsidedead.biz/acmx2-explained.html" style="color:#1565c0">Project Documentation</a></li>
                    <li><a href="https://lostsidedead.biz/acmx2/filter_browser.html" style="color:#1565c0">GPU Filters Explained (CUDA)</a></li>
                    <li><a href="https://github.com/lostjared/acidcam-gpu" style="color:#1565c0">GitHub Repository</a></li>
                </ul>
                <br>
                <div class="fv-technique" style="display:block;border-left-color:#2e7d32;background:linear-gradient(135deg,#e8f5e9,#c8e6c9);color:#1b5e20;max-width:780px">
                    <strong>This browser contains {len(shaders)} GLSL fragment shaders</strong> organized into {len(sorted_cats)} categories. Each shader processes the input texture in real time &mdash; many respond to time, mouse position, and audio amplitude. Select a shader from the tree to view its source code and description, or use search (Ctrl+K) to find effects by name or technique.
                </div>
            </div>
        </div>`;
    mainPanel.querySelectorAll('pre code').forEach(block => {{
        hljs.highlightElement(block);
    }});
}}

function showVertexShader() {{
    if (activeItem) {{ activeItem.classList.remove('active'); activeItem = null; }}
    const el = treePanel.querySelector('.tree-item[data-id="vertex"]');
    if (el) {{ el.classList.add('active'); activeItem = el; }}

    mainPanel.innerHTML = `
        <div class="filter-view">
            <div class="fv-index">Core Shader</div>
            <div class="fv-name">Vertex Shader</div>
            <div class="fv-filename">vertex.glsl</div>
            <div class="fv-desc">The vertex shader is shared by all fragment shaders in ACMX2. It transforms each vertex position by the model-view and projection matrices and passes through the texture coordinate (<code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">tc</code>) to the fragment stage. Every fragment shader listed in this browser receives the interpolated <code style="background:#e8eaf6;padding:1px 6px;border-radius:3px;color:#1a237e">tc</code> varying as its primary UV coordinate for texture sampling and effect computation.</div>
            <div class="fv-technique"><strong>Technique:</strong> Standard MVP matrix transformation &middot; Texture coordinate passthrough &middot; Vertex attributes: position (vec3) + texCoord (vec2)</div>
            <div class="fv-code-header">GLSL Vertex Shader Code</div>
            <div class="fv-code-wrap">
                <pre><code class="language-glsl">${{esc(VERTEX_CODE)}}</code></pre>
            </div>
        </div>`;
    mainPanel.querySelectorAll('pre code').forEach(block => {{
        hljs.highlightElement(block);
    }});
}}

function buildTree() {{
    treePanel.innerHTML = '';

    // About ACMX2 entry
    const aboutItem = document.createElement('div');
    aboutItem.className = 'tree-item active';
    aboutItem.setAttribute('data-id', 'about');
    aboutItem.innerHTML = '<span class="idx">&#9432;</span><span class="fname">About ACMX2</span>';
    aboutItem.addEventListener('click', () => showAbout());
    treePanel.appendChild(aboutItem);
    activeItem = aboutItem;

    // Vertex Shader entry
    const vertexItem = document.createElement('div');
    vertexItem.className = 'tree-item';
    vertexItem.setAttribute('data-id', 'vertex');
    vertexItem.innerHTML = '<span class="idx">&#9889;</span><span class="fname">Vertex Shader</span>';
    vertexItem.addEventListener('click', () => showVertexShader());
    treePanel.appendChild(vertexItem);

    CATEGORIES.forEach((catName, ci) => {{
        const items = SHADERS.filter(s => s.category === catName);
        if (items.length === 0) return;

        const header = document.createElement('div');
        header.className = 'cat-header';
        header.innerHTML = `<span class="arrow">&#9654;</span><span>${{esc(catName)}}</span><span class="count">${{items.length}}</span>`;

        const list = document.createElement('div');
        list.className = 'cat-items';
        list.setAttribute('data-cat', ci);

        items.forEach(s => {{
            const item = document.createElement('div');
            item.className = 'tree-item';
            item.setAttribute('data-id', s.id);
            item.innerHTML = `<span class="idx">${{s.id}}</span><span class="fname">${{esc(s.name)}}</span>`;
            item.addEventListener('click', () => selectShader(s.id));
            list.appendChild(item);
        }});

        header.addEventListener('click', () => {{
            header.classList.toggle('open');
            list.classList.toggle('open');
        }});

        treePanel.appendChild(header);
        treePanel.appendChild(list);
    }});
}}

function selectShader(id) {{
    const s = SHADERS.find(x => x.id === id);
    if (!s) return;

    if (activeItem) activeItem.classList.remove('active');
    const el = treePanel.querySelector(`.tree-item[data-id="${{id}}"]`);
    if (el) {{
        el.classList.add('active');
        activeItem = el;
        const catList = el.parentElement;
        if (!catList.classList.contains('open')) {{
            catList.classList.add('open');
            catList.previousElementSibling.classList.add('open');
        }}
        el.scrollIntoView({{block:'nearest'}});
    }}

    mainPanel.innerHTML = `
        <div class="filter-view">
            <div class="fv-index">Shader #${{s.id}}</div>
            <div class="fv-name">${{esc(s.name)}}</div>
            <div class="fv-filename">${{esc(s.filename)}}</div>
            <div class="fv-desc">${{esc(s.desc)}}</div>
            <div class="fv-technique"><strong>Technique:</strong> ${{esc(s.technique)}}</div>
            ${{s.code ? `
            <div class="fv-code-header">GLSL Shader Code</div>
            <div class="fv-code-wrap">
                <pre><code class="language-glsl">${{esc(s.code)}}</code></pre>
            </div>` : ''}}
        </div>`;

    mainPanel.querySelectorAll('pre code').forEach(block => {{
        hljs.highlightElement(block);
    }});
}}

let searchTimeout = null;
searchInput.addEventListener('input', () => {{
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(doSearch, 200);
}});

function doSearch() {{
    const q = searchInput.value.trim().toLowerCase();

    treePanel.querySelectorAll('.search-hit').forEach(el => el.classList.remove('search-hit'));
    searchHits.clear();
    searchCount.textContent = '';

    if (!q) return;

    const terms = q.split(/\\s+/);
    const results = SHADERS.filter(s => {{
        const hay = (s.id + ' ' + s.name + ' ' + s.filename + ' ' + s.desc + ' ' + s.technique + ' ' + s.category + ' ' + s.code).toLowerCase();
        return terms.every(t => hay.includes(t));
    }});

    searchCount.textContent = `${{results.length}} result${{results.length !== 1 ? 's' : ''}}`;
    results.forEach(s => searchHits.add(s.id));

    results.forEach(s => {{
        const el = treePanel.querySelector(`.tree-item[data-id="${{s.id}}"]`);
        if (el) {{
            el.classList.add('search-hit');
            const catList = el.parentElement;
            if (!catList.classList.contains('open')) {{
                catList.classList.add('open');
                catList.previousElementSibling.classList.add('open');
            }}
        }}
    }});

    if (results.length === 0) {{
        mainPanel.innerHTML = `<div class="welcome"><h2>No Results</h2><p>No shaders match "<b>${{esc(q)}}</b>".</p></div>`;
        return;
    }}

    let html = `<div class="search-results-view"><h2>${{results.length}} shader${{results.length !== 1 ? 's' : ''}} matching "${{esc(q)}}"</h2>`;
    results.forEach(s => {{
        html += `<div class="sr-card" data-id="${{s.id}}">
            <div class="sr-idx">Shader #${{s.id}} &mdash; ${{esc(s.category)}}</div>
            <div class="sr-name">${{highlight(s.name, terms)}}</div>
            <div class="sr-desc">${{highlight(s.desc, terms)}}</div>
            <div class="sr-tech">Technique: ${{highlight(s.technique, terms)}}</div>
        </div>`;
    }});
    html += '</div>';
    mainPanel.innerHTML = html;

    mainPanel.querySelectorAll('.sr-card').forEach(card => {{
        card.addEventListener('click', () => {{
            selectShader(parseInt(card.getAttribute('data-id')));
        }});
    }});
}}

function highlight(text, terms) {{
    let s = esc(text);
    terms.forEach(t => {{
        if (!t) return;
        const re = new RegExp('(' + t.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
        s = s.replace(re, '<mark>$1</mark>');
    }});
    return s;
}}

document.addEventListener('keydown', e => {{
    if ((e.ctrlKey && e.key === 'k') || (e.key === '/' && document.activeElement !== searchInput)) {{
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
    }}
    if (e.key === 'Escape') {{
        searchInput.blur();
        searchInput.value = '';
        doSearch();
    }}
}});

buildTree();
showAbout();
</script>
</body>
</html>'''

    return html_content


def main():
    print("=== AcidCam GPU Shader Browser Generator ===")
    print(f"Shader directory: {SHADER_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    shaders = process_shaders()
    print(f"\nTotal shaders processed: {len(shaders)}")

    # Show category breakdown
    cat_counts: dict[str, int] = {}
    for s in shaders:
        cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nGenerating HTML...")
    html_out = build_html(shaders)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_out)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Written: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
