# AcidCam GPU Filters — Kernel Reference

This document describes every CUDA filter kernel in `filters.cu`. Each filter runs as a `__device__` function called per-pixel from the unified GPU kernel. There are **905 filters** total (indices 0–904).

---

## 0 — SelfAlphaBlend

Multiplies each pixel's color channels by the current alpha parameter, brightening or dimming the image. When negative mode is on, the colors are inverted first. Creates a pulsing glow or fade effect.

**Technique:** Per-channel alpha self-multiplication with optional color inversion.

---

## 1 — MedianBlend

Averages the pixel across all stored history frames, then XORs the result with the current pixel and blends the two together with a contrast boost. Produces a shimmering, ghostly look with enhanced contrast and subtle digital artifacts.

**Technique:** Multi-frame temporal averaging + XOR blend + contrast boost.

---

## 2 — MedianBlendXor

Variant of MedianBlend with additional XOR strobing layered on top, creating a more aggressively glitchy and pulsating version of the temporal average effect.

**Technique:** Multi-frame averaging with XOR strobe overlay.

---

## 3 — SquareBlockResize

Divides the frame into horizontal block rows. Each row pulls from a different history frame based on a ping-pong pattern, then blends 50/50 with the current frame. Creates a layered time-slicing effect where different horizontal bands show different moments in time.

**Technique:** Row-based frame history selection with ping-pong indexing.

---

## 4 — SelfScaleRefined

Scales each pixel's color channels by the alpha parameter, clamping to 255. Produces a clean brightness/exposure adjustment that can push colors toward white at higher alpha values.

**Technique:** Direct per-channel scaling with clamping.

---

## 5 — StrangeGlitch

Compares the current pixel against history frames. If the colors differ by more than a threshold (30 per channel), it replaces the pixel with the first significantly different history frame's pixel. Creates a glitchy freeze effect where only moving areas update.

**Technique:** Color-distance threshold replacement from frame history.

---

## 6 — MatrixOutline

Compares the current pixel against a specific history frame. If the colors are similar (within threshold), the pixel is set to black. Only pixels that have changed significantly remain visible, creating a motion-outline "Matrix"-style wireframe look.

**Technique:** Color-similarity detection rendering similar pixels black.

---

## 7 — AuraTrails

Blends the current pixel with three specific history frames (indices 1, 4, 7) using cascaded 50/50 averaging. Creates smooth, glowing trails behind moving objects with an ethereal aura.

**Technique:** Cascaded multi-frame 50/50 blending from selected history frames.

---

## 8 — MirrorReverseColor

Mirrors the frame while reversing color channels, creating a symmetrical image with swapped color tones.

**Technique:** Coordinate mirroring with channel reordering.

---

## 9 — SquareShrink

Blends the current pixel with frame history only within a shrinking rectangular border. The border offset animates over time, creating a picture-within-a-picture that shrinks inward.

**Technique:** Border-masked 50/50 frame blending with animated inset.

---

## 10 — MotionGhostTrails

Layered ghost trails from multiple history frames create persistent semi-transparent echoes of past motion.

**Technique:** Multi-frame weighted overlay for motion trailing.

---

## 11 — StretchColMatrix8

Stretches columns of pixels at an 8-pixel interval, pulling color data from frame history to create a horizontal stretch-line matrix pattern.

**Technique:** Column-interval frame history sampling at 8px blocks.

---

## 12 — StretchColMatrix16

Same as StretchColMatrix8 but with 16-pixel column intervals, creating wider horizontal stretch bands.

**Technique:** Column-interval frame history sampling at 16px blocks.

---

## 13 — StretchColMatrix32

Same concept at 32-pixel column intervals, producing very wide horizontal stretch bands with a blocky, datamosh-like appearance.

**Technique:** Column-interval frame history sampling at 32px blocks.

---

## 14 — GradientFlashColor

Adds animated gradient colors that flash and shift over time, mixing with the existing frame to create a colorful pulsing gradient overlay.

**Technique:** Animated color gradient addition.

---

## 15 — HorizontalGlitch

Shifts horizontal rows of pixels by variable amounts, creating a broken/torn horizontal glitch distortion.

**Technique:** Per-row horizontal pixel displacement.

---

## 16 — VerticalGlitch

Shifts vertical columns of pixels by variable amounts, creating a vertical tearing/glitch distortion.

**Technique:** Per-column vertical pixel displacement.

---

## 17 — WaveTrails

Blends the current pixel with two history frames at equal weight (33% each). The history frame indices animate over time, producing smooth trails with a wave-like temporal selection pattern.

**Technique:** Three-way equal blend with animated frame selection.

---

## 18 — PixelInterlace

Interlaces pixels from even/odd rows or columns with frame history, creating a woven/interleaved temporal pattern.

**Technique:** Even/odd row interlacing with frame history.

---

## 19 — ColorWaveTrails

Motion trails with per-channel wave-modulated color, producing trails that shimmer with different hues as they fade.

**Technique:** Wave-modulated per-channel frame trailing.

---

## 20 — ParticleSlide

Randomly offsets each pixel's source position using GPU-generated pseudo-random values, then blends with a history frame. Makes the image look like it's dissolving into scattered particles sliding around.

**Technique:** Random coordinate jitter + 50/50 frame blend.

---

## 21 — DiagPixelated

Applies diagonal pixelation, grouping pixels along diagonal lines into blocks that share the same color.

**Technique:** Diagonal block averaging/sampling.

---

## 22 — DiagPixelatedResize

Diagonal pixelation with an animated block size that grows and shrinks over time.

**Technique:** Animated-size diagonal block sampling.

---

## 23 — RGBShiftTrails

Pulls each R, G, B channel from a different history frame (indices 0, 7, 15), cycling the channel assignment every frame. Creates vivid chromatic separation trails where red, green, and blue ghost independently.

**Technique:** Per-channel frame history separation with rotating assignment.

---

## 24 — PictureShiftDown

Shifts the entire image downward over time, with the top filling from frame history. Creates a scrolling/falling picture effect.

**Technique:** Vertical coordinate offset with frame history fill.

---

## 25 — PictureShiftRight

Shifts the entire image rightward over time, with the left edge filling from frame history. Creates a horizontal scrolling effect.

**Technique:** Horizontal coordinate offset with frame history fill.

---

## 26 — PictureShiftVariable

Shifts the image by a variable amount in both directions, creating an unpredictable sliding picture effect.

**Technique:** Variable-offset coordinate displacement.

---

## 27 — StretchR_Right

Stretches only the red channel rightward while leaving green and blue in place, creating a directional red smear.

**Technique:** Single-channel horizontal coordinate offset (red).

---

## 28 — StretchG_Right

Stretches only the green channel rightward, creating a directional green smear trail.

**Technique:** Single-channel horizontal coordinate offset (green).

---

## 29 — StretchB_Right

Stretches only the blue channel rightward, creating a directional blue smear trail.

**Technique:** Single-channel horizontal coordinate offset (blue).

---

## 30 — StretchR_Down

Stretches the red channel downward, producing vertical red trailing/smearing.

**Technique:** Single-channel vertical coordinate offset (red).

---

## 31 — StretchG_Down

Stretches the green channel downward, producing vertical green trailing/smearing.

**Technique:** Single-channel vertical coordinate offset (green).

---

## 32 — StretchB_Down

Stretches the blue channel downward, producing vertical blue trailing/smearing.

**Technique:** Single-channel vertical coordinate offset (blue).

---

## 33 — Distorted_LinesY

Distorts the image along horizontal lines using variable vertical offsets, creating wavy horizontal bands of distortion.

**Technique:** Per-row vertical pixel displacement.

---

## 34 — Distorted_LinesX

Distorts the image along vertical lines using variable horizontal offsets, creating wavy vertical bands of distortion.

**Technique:** Per-column horizontal pixel displacement.

---

## 35 — TripHSV

Converts each pixel to HSV color space, rotates the hue by an animated shift amount, then converts back to RGB. Creates a psychedelic color-cycling effect where all colors rotate around the color wheel.

**Technique:** HSV hue rotation with animated shift.

---

## 36 — XorSumStrobe

XORs pixel values with animated sums of color components, creating a pulsating digital distortion with rapidly shifting false colors.

**Technique:** Animated XOR with accumulated color sums.

---

## 37 — DetectEdges

Applies edge detection (Sobel operator) to highlight boundaries and outlines in the image while suppressing flat areas.

**Technique:** Sobel gradient edge detection.

---

## 38 — SobelNorm

Normalized Sobel edge detection with thresholding, producing clean black-and-white line art from the video.

**Technique:** Sobel edge detection with normalization and threshold.

---

## 39 — LineInLineOut

Alternating horizontal lines pull from different frame history indices, creating a scan-line interlace effect where even and odd rows show different moments in time.

**Technique:** Row-alternating frame history selection.

---

## 40 — LineInLineOut4_Increase

Four-way line interlace with increasing frame offsets. Different rows pull from increasingly distant history frames, creating a fanning time-spread across horizontal bands.

**Technique:** Multi-level row-based frame history with increasing offset.

---

## 41 — LineInLineOut_ReverseIncrease

Reverse direction variant of the line-interlace increase—rows count backward through frame history, producing a reverse temporal cascade.

**Technique:** Reverse-indexed row-based frame history interlace.

---

## 42 — LineInLineOut_ReverseIncrease2

Second variant of the reverse-increase line interlace with different offset calculations for a subtly different temporal pattern.

**Technique:** Alternate reverse-indexed row interlace.

---

## 43 — LineInLineOut_InvertedY

Line interlace where the Y-axis mapping is inverted, so the bottom of the frame shows the most recent history and the top shows the oldest, flipping the temporal spread upside-down.

**Technique:** Y-inverted row-based frame history interlace.

---

## 44 — LineInLineOut_ReverseInvertedY

Combines reverse indexing with Y-inversion for a doubly-flipped temporal interlace pattern.

**Technique:** Reverse + Y-inverted row interlace.

---

## 45 — LineInLineOut_Vertical

Vertical variant: columns (instead of rows) pull from different history frames, creating vertical band interlacing.

**Technique:** Column-based frame history interlace.

---

## 46 — LineInLineOut_VerticalIncrease

Vertical column interlace with increasing frame offsets across columns.

**Technique:** Column-based frame history with increasing offset.

---

## 47 — LineInLineOut_IncreaseImage

Line interlace with increasing offsets applied to the full image blend, producing a graduated time-spread across the frame.

**Technique:** Full-image graduated frame history interlace.

---

## 48 — SquareByRow

Divides the image into rows of square blocks. Each row selects a different history frame using a ping-pong offset, and blends with XOR. Creates blocky temporal bands with digital artifacts.

**Technique:** Row-block frame selection with XOR blending.

---

## 49 — SquareByRowRev

Reverse variant of SquareByRow—the frame selection order is reversed, producing a mirrored temporal band pattern.

**Technique:** Reverse row-block frame selection with XOR blending.

---

## 50 — SquareByRow2

Second variant of SquareByRow with a different block-size and frame offset calculation, producing a denser temporal block pattern.

**Technique:** Alternate row-block frame selection.

---

## 51 — DivideByValue

Averages pixel values across up to 8 history frames, then XORs that average with the current pixel. Creates a shimmering digital interference pattern that reacts to motion.

**Technique:** Multi-frame average XOR blend.

---

## 52 — ColorCollectionSubtleStrobe

Pulls each R, G, B channel from three different history frames (indices 1, 3, 6). Creates a subtle color-separated temporal effect where colors come from different moments in time.

**Technique:** Per-channel selection from fixed history frame indices.

---

## 53 — CollectionRandom

Replaces the current pixel with one from a pseudo-randomly selected history frame (based on seed, x, y). Each pixel may come from a different point in time, creating a shattered temporal mosaic.

**Technique:** Random frame selection per pixel.

---

## 54 — CollectionAlphaXor

XORs each color channel of the current pixel with a different history frame (indices 1, 3, and the newest). Produces colorful digital distortion that shifts based on motion between frames.

**Technique:** Per-channel XOR with different history frames.

---

## 55 — ColorCollection64X

Assembles each color channel from three history frames: frame 1, the middle frame, and an animated frame index. Creates a smooth color-shifting temporal blend.

**Technique:** Per-channel history frame assembly with animated index.

---

## 56 — ColorCollectionSwitch

Rotates which history frame provides which color channel on a 3-frame cycle. Every frame, the R/G/B channel source swaps, creating a rotating color-temporal strobe.

**Technique:** Cycling per-channel frame assignment.

---

## 57 — ColorCollectionRGB_Index

Replaces only one color channel per frame (cycling R→G→B) with data from a corresponding history frame. Creates a subtle per-channel flickering update.

**Technique:** Single-channel-per-frame history replacement.

---

## 58 — ColorCollectionGhostTrails

Blends three history frames (near, middle, far) at equal 33% weight, creating smooth ghost trails that average the recent past into a single ethereal image.

**Technique:** Three-frame equal-weight temporal averaging.

---

## 59 — ColorCollectionScale

Assembles channels from three history frames like ColorCollectionSubtleStrobe, then scales one channel by the alpha parameter. Creates a color-biased temporal composite.

**Technique:** Per-channel history assembly with alpha scaling.

---

## 60 — ColorCollectionReverseStrobe

Alternates the order of history frame channel assignment every other frame—forward order on even frames, reversed on odd frames. Creates a strobing color-swap effect.

**Technique:** Alternating forward/reverse per-channel frame assignment.

---

## 61 — ColorCollectionXorPixel

XORs each pixel channel with a corresponding history frame's channel, with alpha scaling applied to both sides. Alternates between forward and reverse frame ordering every frame.

**Technique:** Alpha-scaled per-channel XOR with alternating frame order.

---

## 62 — BlendWithSource25

Blends the current (filtered) frame with the original unfiltered source frame at 25% source / 75% current. Gently grounds the image back toward reality.

**Technique:** 25/75 source frame blend-back.

---

## 63 — BlendWithSource50

50/50 blend between the current filtered frame and the original source. Cuts filter intensity in half, keeping one foot in reality.

**Technique:** 50/50 source frame blend-back.

---

## 64 — BlendWithSource75

75% original source / 25% current filtered frame. Only a hint of the filter effect shows through.

**Technique:** 75/25 source frame blend-back.

---

## 65 — BlendWithSource100

Full alpha-controlled blend back to the original source. At alpha=1, the image is entirely the original source; at lower alpha, the filter shows through.

**Technique:** Alpha-parameterized source frame blend-back.

---

## 66 — ColorCollectionXorOffsetFlash

Assembles pixels from three history frames, but on a cycling 30-frame interval one channel gets XORed with an offset channel from its source frame instead of directly copied. Creates a periodic flash of XOR distortion.

**Technique:** Periodic single-channel XOR flash from history frames.

---

## 67 — ColorCollectionMatrixGhost

Averages three history frames (first, middle, last) at equal weight. Similar to GhostTrails but using the full range of the frame buffer. Creates a smooth, ghostly temporal composite.

**Technique:** Full-range three-frame temporal averaging.

---

## 68 — MildStrobe

Rotates which color channel each of three history frames provides on a 3-frame cycle. The channel *index* also rotates, creating a gentle strobing color-swap between time-separated frames.

**Technique:** Cycling channel-index rotation across history frames.

---

## 69 — ReduceBy50

Halves every pixel's brightness (multiplies by 0.5). Darkens the entire image uniformly, useful as a utility filter combined with others.

**Technique:** Uniform 50% brightness reduction.

---

## 70 — ColorPositionAverageXor

XORs each pixel with a global color value derived from the sum of all pixels in the frame (modulo 256). The XOR value is the same for all pixels in a frame but changes every frame, creating a uniform color-shifting distortion.

**Technique:** Global average color XOR.

---

## 71 — ColorPositionXor

XORs the current pixel with the corresponding pixel from the newest history frame, then XORs again with the global average color. Combines temporal and global XOR distortion.

**Technique:** History frame XOR + global average color XOR.

---

## 72 — GrayStrobe

Converts the image to grayscale on every other frame using standard luminance weights (0.299R + 0.587G + 0.114B). Alternates between full color and black-and-white.

**Technique:** Alternating-frame grayscale conversion.

---

## 73 — ColorStrobeXor

Averages up to 8 history frames per pixel, then XORs with both the current pixel and a global random color derived from frame sums. Creates a busy digital strobe with color artifacts.

**Technique:** Multi-frame average XOR with global color XOR.

---

## 74 — ColorGhost

Assembles R, G, B channels from three different history frames—one animated, one from the middle, one from near the end. Creates a smooth color-ghost composite with animated channel drift.

**Technique:** Animated per-channel history frame assembly.

---

## 75 — BlurredOutXor

Averages three history frames (near, middle, far), then XORs the averaged pixel with the current pixel. Produces a soft, blurred XOR distortion.

**Technique:** Three-frame temporal average then XOR blend.

---

## 76 — DizzyFilter

Averages three history frames (near, middle, far) at 33% each, replacing the current pixel entirely. Creates a dizzy, blurred composite of the recent past.

**Technique:** Three-frame equal-weight temporal replacement.

---

## 77 — Buzzed

Averages up to 8 history frames with equal weighting, replacing the current pixel. Creates a buzzy, smoothed temporal blur like seeing through vibrating glass.

**Technique:** Multi-frame (up to 8) equal-weight averaging.

---

## 78 — BuzzedDark

Same as Buzzed but using up to 16 frames and then darkening the result to 70% brightness. Creates a darker, more smeared temporal blur.

**Technique:** Multi-frame averaging with 70% brightness reduction.

---

## 79 — AllRed

Zeros out the blue and green channels, leaving only the red channel. Turns the entire image into a red monochrome view.

**Technique:** Blue and green channel zeroing.

---

## 80 — AllGreen

Zeros out the blue and red channels, leaving only green. Turns the image into a green monochrome view.

**Technique:** Blue and red channel zeroing.

---

## 81 — AllBlue

Zeros out the green and red channels, leaving only blue. Turns the image into a blue monochrome view.

**Technique:** Green and red channel zeroing.

---

## 82 — NegativeStrobe

Inverts all colors (bitwise NOT) on every other frame, alternating between the normal image and its photographic negative.

**Technique:** Alternating-frame bitwise color inversion.

---

## 83 — XorAddMul

Applies three different operations to the three channels: XOR+add on blue, plain add on green, multiply on red, all with an animated integer value. Creates chaotic per-channel math distortion.

**Technique:** Mixed XOR/add/multiply per-channel with animated operand.

---

## 84 — HorizontalLines

Adds position-and-time-based values to each color channel with offset cycling, creating scrolling diagonal color line patterns across the image.

**Technique:** Position-modulated color addition with time offset.

---

## 85 — StrobeRedGreenBlue

Cycles through showing only red, only green, or only blue on each frame (zeroing the other two channels). Produces a rapid R→G→B color strobe.

**Technique:** Single-channel isolation cycling per frame.

---

## 86 — Pulse

Multiplies all channels by a sine-wave-modulated brightness factor, causing the entire image to rhythmically pulse brighter and darker.

**Technique:** Sine-wave global brightness modulation.

---

## 87 — DiamondPattern

Creates a diamond/checkerboard pattern by applying different math operations to pixels at even/odd x,y coordinates. Produces a geometric diamond lattice overlay with position-derived colors.

**Technique:** Even/odd coordinate conditional per-channel math.

---

## 88 — Bitwise_XOR

XORs each pixel with the corresponding pixel from a recent history frame. Creates ghostly digital artifacts where the image has changed between frames.

**Technique:** Frame-to-frame bitwise XOR.

---

## 89 — Bitwise_AND

ANDs each pixel with a recent history frame pixel. Darkens the image, keeping only the bits that are set in both frames. Moving areas get partially blacked out.

**Technique:** Frame-to-frame bitwise AND.

---

## 90 — Bitwise_OR

ORs each pixel with a recent history frame pixel. Brightens the image, setting any bit that was set in either frame. Creates bright ghost overlays.

**Technique:** Frame-to-frame bitwise OR.

---

## 91 — BlendSwitch

Modulates one color channel (cycling which one based on x position) by a pixel-position-animated blend value. Creates subtle shifting color tones across the frame.

**Technique:** Position-modulated single-channel blend.

---

## 92 — LineRGB

Assigns each row to show only one color channel in a repeating R/G/B pattern (row 0 = red only, row 1 = green only, row 2 = blue only). Creates horizontal RGB-striped scan lines.

**Technique:** Row-based single-channel isolation.

---

## 93 — PixelRGB

Same as LineRGB but per-pixel instead of per-row: each individual pixel shows only one channel in a cycling R/G/B pattern. Creates a fine-grained RGB mosaic like a Bayer filter.

**Technique:** Per-pixel single-channel isolation.

---

## 94 — InvertedScanlines

Inverts alternate scan lines (rows), creating horizontal stripes of normal and inverted color.

**Technique:** Row-alternating color inversion.

---

## 95 — ScanSwitch

Switches between normal and history-frame pixels on alternating scan lines, creating an interlaced blend of present and past.

**Technique:** Row-alternating current/history frame switching.

---

## 96 — ScanAlphaSwitch

Scan-line switching with alpha blending—alternating rows alpha-blend with a history frame rather than hard-switching. Softer interlace effect.

**Technique:** Row-alternating alpha-blended frame switching.

---

## 97 — RGBFlash

Rapidly flashes the image with different single-channel emphasis each frame, creating a fast R/G/B flash strobe similar to StrobeRedGreenBlue but with blending.

**Technique:** Per-frame channel emphasis with flash.

---

## 98 — DiagonalLines

Adds diagonal-stripe color patterns to the image, with the stripe offset animating over time.

**Technique:** Diagonal coordinate-based color addition.

---

## 99 — Darken

Reduces pixel brightness by subtracting or scaling down all channels, creating a uniform darkening effect.

**Technique:** Per-channel brightness reduction.

---

## 100 — SelfXorBlend

XORs each pixel with a scaled version of itself, creating a self-referencing digital distortion that varies with alpha intensity.

**Technique:** Self-pixel XOR with alpha scaling.

---

## 101 — SelfXorDoubleFlash

Double-intensity self-XOR that alternates with a bright flash, producing an aggressive strobe-distortion combo.

**Technique:** Intensity-doubled self-XOR with alternating flash.

---

## 102 — SelfOrDoubleFlash

Bitwise OR of the pixel with itself scaled, plus alternating flash. Brightens and washes out the image with periodic bright bursts.

**Technique:** Self-pixel OR with alternating flash.

---

## 103 — BlendRowCurvedSqrt

Blends rows using a square-root curve for the alpha interpolation, creating a non-linear gradient blend that emphasizes midtones.

**Technique:** Square-root curved row-based alpha blending.

---

## 104 — XorAlpha

XORs pixel channels with alpha-modulated values, creating a shifting digital mask that responds to the alpha parameter.

**Technique:** Alpha-parameterized XOR.

---

## 105 — RandomXorBlend

XORs with random-offset values and blends with frame history, creating unpredictable glitch-like color artifacts.

**Technique:** Random-value XOR combined with frame blending.

---

## 106 — AndStrobe

Bitwise AND with animated strobe values, periodically darkening the image in a rhythmic pattern.

**Technique:** Animated AND strobe.

---

## 107 — AndStrobeScale

AND strobe with an additional scaling factor, creating a more intense darkening strobe with brightness variation.

**Technique:** Scaled AND strobe.

---

## 108 — AndPixelStrobe

AND operation applied per-pixel with a strobing mask, selectively darkening individual pixels in a rhythmic pattern.

**Technique:** Per-pixel AND with strobing mask.

---

## 109 — AndOrXorStrobe

Cycles through AND, OR, and XOR operations on alternating frames, creating a complex tri-phase bitwise strobe.

**Technique:** Cycling AND/OR/XOR bitwise strobe.

---

## 110 — FadeInAndOut

Smoothly fades the image brightness up and down using a sine wave, creating a breathing/pulsing fade effect.

**Technique:** Sine-wave brightness fade modulation.

---

## 111 — BrightStrobe

Strobes between normal and boosted brightness, creating a pulsing flash-bright effect.

**Technique:** Alternating brightness boost strobe.

---

## 112 — DarkStrobe

Strobes between normal and darkened image, creating a pulsing dim/shadow effect.

**Technique:** Alternating brightness reduction strobe.

---

## 113 — RandomXorOpposite

XORs each pixel with a random-offset value and the inverted version of the pixel, creating chaotic complementary-color glitch artifacts.

**Technique:** Random XOR with inverted self-pixel.

---

## 114 — GradientRainbow

Adds a rainbow gradient overlay that shifts vertically across the frame, with the colors animated over time. The gradient uses different frequencies for R, G, B creating an animated spectral sweep.

**Technique:** Position-based multi-frequency color gradient addition.

---

## 115 — cossinMultiply

Multiplies the pixel's blue channel additive blend with sin(alpha * x), green with cos(alpha * y), and red with the average. Creates swirling trigonometric color bands that evolve over time.

**Technique:** Sine/cosine position-modulated per-channel addition.

---

## 116 — colorAccumulate1

Cross-accumulates color channels: blue gets red × alpha, green gets blue × alpha, red gets green × alpha. Creates a psychedelic color-rotation feedback effect.

**Technique:** Cross-channel accumulation with alpha scaling.

---

## 117 — colorAccumulate2

Cross-channel accumulation like colorAccumulate1, but with x and y position offsets added to the channels. Creates position-dependent color-rotation gradients.

**Technique:** Cross-channel accumulation with position offsets.

---

## 118 — WeakBlend

Mildly scales one channel (selected by pixel position) with a small animated value. Creates a subtle, gentle color shimmer.

**Technique:** Single-channel mild animated scaling.

---

## 119 — StrobeEffect

Cycles through four different per-channel scaling modes on each frame, with sine-wave modulated alpha. Each mode emphasizes different channel combinations, creating a 4-phase color strobe.

**Technique:** 4-phase per-channel scaling strobe with sine modulation.

---

## 120 — Blend3

Modulates each color channel with a different-frequency sine wave (0.1, 0.13, 0.17), creating subtle out-of-phase color pulsing.

**Technique:** Multi-frequency sine-wave per-channel modulation.

---

## 121 — NegParadox

Doubles plus alpha-blends each channel, with green mixing in a global sum value. Creates an over-bright paradoxical look with color bleeding.

**Technique:** Self-doubling with alpha and global color bleed.

---

## 122 — ThoughtMode

Conditionally adds or subtracts channel values based on position, frame count, and global sums. Creates an unpredictable, "thinking" color mutation that varies across the frame.

**Technique:** Conditional position-dependent channel mutation.

---

## 123 — Tri

Self-adds each channel with alpha modulation, plus adds global sum values to green and red. Creates a bright, washed-out tricolor effect.

**Technique:** Self-add with global color sum injection.

---

## 124 — Distort

Adds scaled x-position to red, y-position to blue, and alpha-scales green. Creates a diagonal position-based color gradient distortion that shifts over time.

**Technique:** Position-to-color mapping with alpha animation.

---

## 125 — colorAccumulate3

Cross-channel accumulation with x added to green and y to red, creating position-aware color rotation with a spatial gradient.

**Technique:** Cross-channel accumulation with position-biased offsets.

---

## 126 — filter8

Adds the sum of (x + y) × alpha to all three channels equally. Creates a diagonal brightness gradient that animates with sine-wave alpha.

**Technique:** Uniform diagonal position-based brightness addition.

---

## 127 — filter3

Adds each channel to the blue channel value scaled by alpha. Tends to shift the entire image toward the blue channel's brightness pattern.

**Technique:** Blue-channel-biased brightness redistribution.

---

## 128 — rainbowBlend

Adds animated rainbow colors derived from global sums and frame count to each pixel. The R, G, B offsets animate at different speeds, creating an evolving rainbow wash overlay.

**Technique:** Animated per-channel global-sum color addition.

---

## 129 — pixelScale

Scales each channel by a sine-animated factor, then adds inter-channel differences/sums. Creates a contrasty, color-shifting effect that oscillates.

**Technique:** Sine-animated per-channel scaling with cross-channel math.

---

## 130 — GradientSelf

Modulates one channel (selected by row position) with a sine-animated scale, then adds a position-derived counter. Creates a row-cycling color gradient.

**Technique:** Row-selected channel with position gradient.

---

## 131 — GradientSelfVertical

Same as GradientSelf but the channel is selected by column position instead of row, creating vertical color band gradients.

**Technique:** Column-selected channel with position gradient.

---

## 132 — GradientDown

Modulates a position-selected channel and adds a vertical (y-based) gradient counter. Creates a top-to-bottom color gradient.

**Technique:** Vertical gradient with position-selected channel.

---

## 133 — GraidentHorizontal

Modulates a position-selected channel and adds a horizontal (x-based) gradient counter. Creates a left-to-right color gradient.

**Technique:** Horizontal gradient with position-selected channel.

---

## 134 — Inter

Blacks out every other row, with the visible/blacked rows alternating each frame. Creates a flickering interlace/scan-line effect.

**Technique:** Alternating row blanking with frame toggle.

---

## 135 — BlendedScanLines

Adds a row-and-position-based color value to one channel (cycling which), creating colored scan-line bands that scroll diagonally.

**Technique:** Position-modulated single-channel scan-line addition.

---

## 136 — GradientStripes

Adds a vertical gradient to one channel while subtracting from the opposite channel, cycling which channels are affected. Creates animated moving color stripes.

**Technique:** Channel-opposed vertical gradient with cycling offset.

---

## 137 — XorSine

XORs each channel with its own sine value scaled by an animated factor and global color sums. Creates a wavy digital distortion with sine-modulated XOR patterns.

**Technique:** Sine-of-value XOR with animated scaling.

---

## 138 — Circular

Adds cosine- and sine-derived color values based on pixel position with an animated radius. Creates circular/radial color patterns that evolve over time.

**Technique:** Polar-coordinate-derived color pattern addition.

---

## 139 — RandomPixels

Adds random brightness values (0–255) to each channel using GPU pseudo-random function. Creates a heavy static/noise overlay.

**Technique:** Per-pixel random noise addition.

---

## 140 — DarkRandomPixels

Adds random brightness up to a variable max, then divides by 4 to darken the result. Creates a darker, grittier static noise.

**Technique:** Scaled random noise with brightness reduction.

---

## 141 — Bars

Sets one channel to full 255 based on cycling x-position bands, creating vertical color bars that shift through R, G, B.

**Technique:** Position-based single-channel maxing for color bars.

---

## 142 — NegativeByRow

Inverts colors on every other row, creating horizontal alternating-stripe negative bands.

**Technique:** Row-alternating bitwise NOT.

---

## 143 — XorScale

XORs each channel with a different animated scale value derived from global sums + frame count. Creates a globally-shifting color distortion.

**Technique:** Per-channel XOR with animated global scale values.

---

## 144 — SelfAlphaRGB

Cycles through 4 modes that variously scale and cross-combine channels with alpha: additive cascading, reversed cascading, shuffled cascading, or XOR cascading. Creates an animated multi-mode color transformation.

**Technique:** 4-phase cross-channel alpha scaling with mode rotation.

---

## 145 — BitwiseXorStrobe

Applies different sine-frequency alpha scaling to each channel, rotating which channel gets which frequency every frame. Creates a smooth multi-tempo color strobe.

**Technique:** Multi-frequency sine-scaled channel rotation.

---

## 146 — OrStrobe

ORs the current pixel with a random global color and the previous frame pixel, all scaled by sine-animated alpha. Creates a bright, OR-heavy strobe with color drift.

**Technique:** Animated OR with random color and history frame.

---

## 147 — DivideAndIncH

Adds horizontal and vertical position-mapped gradient increments to the blue and green channels, creating a grid-aligned color gradient based on frame dimensions.

**Technique:** Dimension-normalized position-to-color gradient (horizontal).

---

## 148 — DivideAndIncW

Same as DivideAndIncH but with x and y mapping swapped, creating the orthogonal gradient direction.

**Technique:** Dimension-normalized position-to-color gradient (vertical).

---

## 149 — RandomIncrease

Adds pseudo-random color offsets derived from global sums and frame count (different multipliers per channel). Creates animated, frame-wide random color shifting.

**Technique:** Global random per-channel color addition.

---

## 150 — SelfAlphaScaleBlend

Scales each channel by (alpha + 1), accumulates a running sum across channels, then XORs each channel with that sum. Creates a cascading self-referencing digital distortion.

**Technique:** Cumulative cross-channel alpha scaling with XOR feedback.

---

## 151 — FadeBars

XORs each channel with a position-derived color, then scales by animated alpha. Creates fading vertical color bars with XOR artifacts.

**Technique:** Position-color XOR with alpha fade.

---

## 152 — StrobeXor

XORs with an animated strobe value that changes per frame, creating a rapid digital strobe effect.

**Technique:** Frame-animated XOR strobe.

---

## 153 — Blank

Sets all pixels to black (or a uniform color), effectively clearing the frame. Useful as a reset or transition filter.

**Technique:** Full-frame pixel zeroing.

---

## 154 — ColorVariableBlend

Blends channels with animated variable amounts, creating color-shifting effects that smoothly vary over time.

**Technique:** Per-channel variable-rate blending.

---

## 155 — ColorXorBlend

XOR blending with animated color channel mixing, creating shifting digital color distortion.

**Technique:** Color-animated XOR blend.

---

## 156 — ColorAddBlend

Additive blending of color channels with animated mixing, creating a bright, washed-out color-shifting overlay.

**Technique:** Animated per-channel additive blend.

---

## 157 — SurroundingPixels

Averages each pixel with its immediate neighbors (3×3 box filter kernel), creating a simple spatial blur effect.

**Technique:** 3×3 box filter spatial averaging.

---

## 158 — SurroundingPixelsAlpha

3×3 box blur with alpha-modulated intensity, controlling how much the blur mixes with the original.

**Technique:** Alpha-weighted 3×3 spatial blur.

---

## 159 — DarkModBlend

Blends with frame history using modular arithmetic that tends to darken the result. Creates dark, moody motion trails.

**Technique:** Modular dark-biased frame blending.

---

## 160 — IncreaseDecreaseGamma

Cycles gamma brightness up and then back down over time, creating an animated exposure pulsation.

**Technique:** Cycling gamma adjustment.

---

## 161 — BlendChannelXor

XORs individual channels with values from frame history and blends, creating channel-specific digital artifacts.

**Technique:** Per-channel history XOR blend.

---

## 162 — IncDifference

Computes the difference between the current frame and a history frame, with an increasing amplification factor. Highlights motion with growing intensity.

**Technique:** Amplified frame differencing.

---

## 163 — IncDifferenceAlpha

Frame difference with alpha-modulated amplification, controlling how strongly motion is highlighted.

**Technique:** Alpha-controlled amplified frame differencing.

---

## 164 — MirrorXorAlpha

XORs the pixel with its mirrored counterpart from the opposite side of the frame, with alpha blending. Creates symmetric digital distortion.

**Technique:** Mirror-coordinate XOR with alpha blend.

---

## 165 — IntertwinedMirror

Intertwines normal pixels with horizontally-mirrored pixels in alternating bands. Creates a woven symmetrical pattern.

**Technique:** Alternating-band mirror interleaving.

---

## 166 — ColorFadeFilter

Smoothly fades colors toward a target hue over time, creating a gradual color wash transition.

**Technique:** Time-interpolated color fade.

---

## 167 — ColorChannelMoveUpAndDown

Shifts color channels vertically in opposite directions (one up, one down), creating vertical chromatic separation.

**Technique:** Per-channel vertical displacement.

---

## 168 — MedianStrobe

Alternates between median-blended and normal frames. On median frames, pixels are averaged from history; on normal frames, they pass through. Creates a strobed smoothing effect.

**Technique:** Alternating median temporal blur.

---

## 169 — RGBBlend

Blends R, G, B channels from different history frames with the current frame, creating a per-channel temporal blend effect.

**Technique:** Per-channel history blending (RGB order).

---

## 170 — BGRBlend

Same as RGBBlend but with reversed channel order (B, G, R), creating a different color-temporal combination.

**Technique:** Per-channel history blending (BGR order).

---

## 171 — FlipAlphaBlend

Blends the current frame with a vertically-flipped version using alpha, creating a semi-transparent mirror reflection effect.

**Technique:** Vertical flip with alpha blend.

---

## 172 — RandomFlipFilter

Randomly decides per-pixel whether to show the normal or flipped pixel, creating a chaotic partial-mirror effect.

**Technique:** Random per-pixel flip selection.

---

## 173 — SelfScaleByFrame

Scales pixel brightness by a factor derived from the frame counter, creating brightness that grows or shifts over time.

**Technique:** Frame-count-derived brightness scaling.

---

## 174 — AlphaBlendMirror

Alpha-blends the current pixel with its horizontally-mirrored counterpart, creating a translucent symmetrical overlay.

**Technique:** Horizontal mirror alpha blend.

---

## 175 — TwistedVision

Applies a sine/cosine-based coordinate twist to pixel sampling, creating a swirling, twisted distortion of the image.

**Technique:** Trigonometric coordinate twist distortion.

---

## 176 — TruncateColor

Truncates color values to lower bit depths (e.g., multiples of 32 or 64), creating a posterized/quantized color look with fewer distinct colors.

**Technique:** Bit-depth color truncation/quantization.

---

## 177 — TruncateVariable

Variable-strength color truncation that changes over time, creating an animated posterization that cycles between smooth and blocky color.

**Technique:** Animated variable-rate color quantization.

---

## 178 — TruncateVariableScale

Variable truncation with additional scaling, creating animated posterization with brightness modulation.

**Technique:** Scaled animated color quantization.

---

## 179 — XorFade

XOR operation that fades in and out over time—the XOR operand animates from 0 to max and back, creating a smooth digital distortion fade.

**Technique:** Sine-animated XOR fade.

---

## 180 — SineValue

Applies the sine function to each pixel's color values, remapping brightness through a sinusoidal curve. Creates a wavy tone-mapping effect.

**Technique:** Sine-of-value tone remapping.

---

## 181 — FadeRtoGtoB

Gradually transitions emphasis from red to green to blue channels over time, creating a slow color wash that cycles through the primary colors.

**Technique:** Cycling primary channel emphasis.

---

## 182 — FadeRandomChannel

Fades a randomly selected channel in and out each cycle, creating unpredictable per-channel brightness pulsing.

**Technique:** Random channel selection with fade.

---

## 183 — VariableLines

Draws horizontal lines with variable spacing and color, creating an animated line-pattern overlay.

**Technique:** Variable-pitch horizontal line rendering.

---

## 184 — VariableLinesVertical

Same as VariableLines but vertical, creating animated vertical line-pattern overlay.

**Technique:** Variable-pitch vertical line rendering.

---

## 185 — RowMedianBlend

Computes median/average values along each row from frame history, then blends. Creates horizontally-coherent temporal smoothing.

**Technique:** Row-wise temporal median blending.

---

## 186 — MirrorReverseColorBlend

Mirrors the image and reverses color channels while blending with the original, creating a blended symmetric false-color effect.

**Technique:** Mirror + channel reversal + alpha blend.

---

## 187 — PsychoticVision

Extreme animated color distortion combining multiple techniques—channel swaps, XOR, and sine modulation—for a chaotic psychedelic look.

**Technique:** Multi-technique combined per-channel distortion.

---

## 188 — PixelGlitch

Randomly replaces some pixels with offset-sourced or corrupted values, simulating digital glitch artifacts that flicker and scatter.

**Technique:** Random pixel displacement/corruption.

---

## 189 — StaticGlitch

Adds random static noise blocks that move around the frame, simulating television static interference.

**Technique:** Random-block noise overlay.

---

## 190 — WavePattern

Applies a sine-wave pattern to pixel brightness, creating undulating wave-shaped brightness bands across the image.

**Technique:** Sine-wave brightness modulation pattern.

---

## 191 — WavePatternXor

Wave brightness pattern combined with XOR operations, creating undulating bands of digital distortion.

**Technique:** Sine-wave XOR pattern.

---

## 192 — DiagonalXor

XOR applied along diagonal axes, creating diagonal stripe-shaped digital artifacts.

**Technique:** Diagonal position-based XOR.

---

## 193 — RGBShiftBlend

Shifts each R, G, B channel to different spatial positions and blends, creating a chromatic aberration/prism-split effect with blending.

**Technique:** Per-channel spatial offset with blend.

---

## 194 — ChannelShuffle

Randomly permutes which color channel is assigned to which output channel, creating false-color images.

**Technique:** Random channel permutation.

---

## 195 — ChannelShuffleRand

Randomized channel shuffle that changes per pixel/frame, creating a more chaotic false-color mosaic.

**Technique:** Per-frame random channel reassignment.

---

## 196 — PixelCounter

Uses a running pixel counter to modify color values, creating position-dependent progressive color shifts.

**Technique:** Sequential pixel counter color modification.

---

## 197 — PixelCounterXor

Pixel counter combined with XOR, creating position-dependent digital distortion patterns.

**Technique:** Sequential pixel counter XOR.

---

## 198 — RowColorBlend

Blends each row with a different animated color, creating horizontal color bands.

**Technique:** Per-row animated color blending.

---

## 199 — ColumnColorBlend

Blends each column with a different animated color, creating vertical color bands.

**Technique:** Per-column animated color blending.

---

## 200 — CheckerboardXor

XOR applied in a checkerboard pattern—even squares XOR with one value, odd squares with another. Creates a tiled digital distortion grid.

**Technique:** Checkerboard-masked XOR.

---

## 201 — CheckerboardBlend

Blends current and history frames in a checkerboard pattern—alternating tiles show different time points.

**Technique:** Checkerboard-masked temporal blending.

---

## 202 — SineWaveDistort

Displaces pixel positions using a sine wave, creating horizontal wavy distortion like looking through rippled glass.

**Technique:** Sine-modulated horizontal coordinate displacement.

---

## 203 — CosineWaveDistort

Displaces pixels using a cosine wave, creating vertical wavy distortion complementary to SineWaveDistort.

**Technique:** Cosine-modulated vertical coordinate displacement.

---

## 204 — SinCosBlend

Combines sine and cosine wave blending of color channels, creating smooth oscillating color transitions.

**Technique:** Combined sine/cosine per-channel blending.

---

## 205 — PixelReverseXor

XORs pixels with their reverse-ordered counterpart (reading the frame backward), creating mirrored digital artifacts.

**Technique:** Reverse-position pixel XOR.

---

## 206 — LinesAcrossX

Draws horizontal lines by replacing certain rows with data from frame history across the x-axis.

**Technique:** Row-based history frame line replacement.

---

## 207 — XorLineX

XOR applied on specific horizontal lines/rows, creating periodic horizontal digital distortion bands.

**Technique:** Row-selective XOR.

---

## 208 — AlphaComponentIncrease

Progressively increases alpha-blending intensity over time, creating a gradually intensifying blend effect.

**Technique:** Time-increasing alpha blend.

---

## 209 — ExpandContract

Expands and contracts the image from the center using animated scaling, like a breathing zoom effect.

**Technique:** Animated center-point scaling.

---

## 210 — LongLines

Creates long continuous horizontal lines using frame history, with lines extending across the full frame width.

**Technique:** Full-width horizontal history line replacement.

---

## 211 — TearRight

Tears the image rightward—horizontal slices shift right by increasing amounts, creating a rightward shearing/tearing distortion.

**Technique:** Row-based rightward pixel displacement.

---

## 212 — TearDown

Tears the image downward—vertical columns shift down by increasing amounts, creating a downward shearing distortion.

**Technique:** Column-based downward pixel displacement.

---

## 213 — DistortionByRow

Each row shifts horizontally by a different animated amount, creating wavy row-by-row horizontal distortion.

**Technique:** Per-row variable horizontal displacement.

---

## 214 — DistortionByCol

Each column shifts vertically by a different animated amount, creating wavy column-by-column vertical distortion.

**Technique:** Per-column variable vertical displacement.

---

## 215 — AlternateAlpha

Alternates between two different alpha-blend settings on different frames or pixel groups, creating a dual-mode blending strobe.

**Technique:** Alternating alpha blend modes.

---

## 216 — DiagSquareRGB

Diagonal square blocks with per-channel color selection from frame history. Creates a diagonal tiled mosaic with RGB separation.

**Technique:** Diagonal block layout with per-channel history sampling.

---

## 217 — ShiftPixelsRGB

Shifts each R, G, B channel by different pixel offsets, creating a per-channel spatial displacement/chromatic aberration.

**Technique:** Per-channel pixel position shifting.

---

## 218 — ColorWaveTrailsRGB

Trails with per-channel wave-modulated offsets—each R, G, B channel follows a different wave pattern through frame history. Creates vivid chromatic trail separations.

**Technique:** Per-channel sine-wave frame history offset.

---

## 219 — ProperTrails

Clean motion trails built by averaging the current frame with multiple history frames at graduated weights. Creates smooth, natural-looking motion blur trails.

**Technique:** Weighted multi-frame temporal trailing.

---

## 220 — XorLag

XORs the current pixel with a frame from a few steps back in history, creating a lagged digital distortion that highlights what changed recently.

**Technique:** Time-lagged frame XOR.

---

## 221 — PixelateBlend

Pixelates the image into blocks and blends with frame history. Combines a mosaic/pixelation look with temporal ghosting.

**Technique:** Block pixelation + history frame blending.

---

## 222 — DiagPixel

Samples pixels along diagonal lines, replacing each pixel with one found along its diagonal. Creates diagonal smear/streak artifacts.

**Technique:** Diagonal pixel position remapping.

---

## 223 — DiagPixelY

Diagonal pixel remapping along the Y axis variant, creating differently-angled diagonal streaks.

**Technique:** Y-biased diagonal position remapping.

---

## 224 — ExpandLeftRight

Stretches the image outward from the center, expanding left and right halves away from the midline. Creates a mirror-expand spread effect.

**Technique:** Center-outward horizontal expansion.

---

## 225 — DiagSquare

Diagonal square block pattern applied to the image, grouping pixels into diagonal tiles.

**Technique:** Diagonal square block grouping.

---

## 226 — HorizontalColorOffset

Offsets each color channel horizontally by different amounts, creating a horizontal chromatic spread/aberration.

**Technique:** Per-channel horizontal offset.

---

## 227 — PrevFrameNotEqual

Replaces pixels only where the current frame differs from the previous frame, creating a motion-only display where static areas freeze.

**Technique:** Frame-difference conditional replacement.

---

## 228 — BlackLines

Draws black horizontal lines at regular intervals, creating a scan-line/venetian-blind overlay.

**Technique:** Row-interval black line rendering.

---

## 229 — DizzyMode

Averages multiple history frames together, creating a heavily blurred temporal composite that looks dizzy and smeared.

**Technique:** Multi-frame temporal averaging.

---

## 230 — GhostShift

Shifts ghost images from frame history horizontally or vertically over time, creating moving transparent afterimages.

**Technique:** Animated-offset history frame ghosting.

---

## 231 — RGBSplitFilter

Splits the image into separate R, G, B components displayed in different spatial positions, creating a prism-like color separation.

**Technique:** Spatially-separated RGB channel display.

---

## 232 — PixelateRect

Combines rectangular blocks of pixels into single colors (block averaging), creating a classic mosaic/pixelation effect.

**Technique:** Rectangular block color averaging.

---

## 233 — CollectionXor4

XORs the current pixel with four different history frames in sequence, layering multiple XOR operations for complex digital artifacts.

**Technique:** Quad-frame sequential XOR.

---

## 234 — RectangleSpin

Rotates/spins rectangular regions of the image at different rates, creating a spinning tile mosaic.

**Technique:** Per-rectangle rotation animation.

---

## 235 — RectanglePlotXY

Renders rectangles at positions derived from x/y coordinates, creating a geometric grid plot overlay.

**Technique:** Coordinate-derived rectangle rendering.

---

## 236 — ShiftLinesDown

Shifts alternating horizontal lines downward by different amounts, creating a cascading downward slip distortion.

**Technique:** Alternating row downward shift.

---

## 237 — PictureStretch

Stretches the image from the center or edges, creating a rubber-like distortion effect.

**Technique:** Position-scaled coordinate stretching.

---

## 238 — PictureStretchPieces

Stretches the image in separate pieces/segments, each with different stretch amounts, creating a fragmented stretch distortion.

**Technique:** Segmented variable stretching.

---

## 239 — VisualSnow

Adds fine-grained random noise (visual snow) to the image, simulating the "visual snow" neurological phenomenon.

**Technique:** Fine per-pixel random noise addition.

---

## 240 — VisualSnowX2

Double-intensity visual snow with more aggressive noise, creating heavier static interference.

**Technique:** Doubled per-pixel random noise.

---

## 241 — LineGlitch

Randomly glitches entire horizontal lines, replacing them with displaced or corrupted data.

**Technique:** Random row glitch/displacement.

---

## 242 — SlitReverse64

Reverses pixel order within 64-pixel slits/segments, creating a segmented mirror effect.

**Technique:** 64-pixel segment order reversal.

---

## 243 — SlitReverse64_Increase

Slit reversal with increasing slit sizes over time, creating an evolving segmented mirror pattern.

**Technique:** Animated-size segment reversal.

---

## 244 — SlitStretch

Stretches pixels within slit segments, creating elongated repeated pixel bands.

**Technique:** Segment-internal pixel stretching.

---

## 245 — LineLeftRight

Shifts alternating rows left and right by animated amounts, creating a horizontal shaking/jitter distortion.

**Technique:** Alternating-row horizontal oscillation.

---

## 246 — LineLeftRightResize

LineLeftRight with animated line height changes, creating a shaking distortion with growing/shrinking row bands.

**Technique:** Animated-height alternating row oscillation.

---

## 247 — RGBLineTrails

Horizontal line trails where each R, G, B channel trails independently on different rows, creating a chromatic line trail pattern.

**Technique:** Per-channel row-based temporal trailing.

---

## 248 — RGBCollectionBlend

Collects each R, G, B channel from different history frames and blends them together, creating a temporal chromatic composite.

**Technique:** Per-channel multi-frame collection blend.

---

## 249 — RGBCollectionIncrease

RGB collection with increasing intensity over successive frames, building up channel separation over time.

**Technique:** Accumulating per-channel history collection.

---

## 250 — RGBLongTrails

Extended R, G, B channel trails spanning many history frames, creating long vivid chromatic motion trails.

**Technique:** Long-range per-channel frame trailing.

---

## 251 — FadeRGB_Speed

Fades each R, G, B channel at different speeds, creating a desynchronized color fade effect.

**Technique:** Per-channel variable-rate fade.

---

## 252 — RGBStrobeTrails

Combines RGB-separated trails with strobing visibility, creating pulsing chromatic trails.

**Technique:** Strobed per-channel trailing.

---

## 253 — BoxGlitch

Randomly displaces rectangular blocks of pixels, creating a classic digital "box glitch" artifact.

**Technique:** Random rectangular block displacement.

---

## 254 — VerticalPictureDistort

Distorts the image with vertical displacement that varies across the frame, creating a wavy vertical stretch.

**Technique:** Variable vertical position distortion.

---

## 255 — ShortTrail

Minimal motion trail using just one or two history frames blended lightly, for a subtle ghosting effect.

**Technique:** Minimal two-frame temporal blend.

---

## 256 — DiagInward

Diagonal lines sweep inward from corners toward the center, each line pulling from different history frames.

**Technique:** Inward-diagonal frame history sweep.

---

## 257 — DiagSquareInward

Diagonal square blocks sweep inward, with each block layer showing a different temporal frame.

**Technique:** Inward diagonal square temporal layering.

---

## 258 — DiagSquareInwardResize

Diagonal square inward sweep with animated block sizes that grow and shrink.

**Technique:** Animated-size inward diagonal block sweep.

---

## 259 — PictureShiftDownRight

Shifts the image simultaneously down and to the right over time.

**Technique:** Combined downward + rightward coordinate shift.

---

## 260 — FlipPictureShift

Flips the image (vertically or horizontally) with an animated shift offset, creating a rolling mirror transition.

**Technique:** Flip + animated coordinate offset.

---

## 261 — RGBWideTrails

Wide spatial RGB-separated trails where each channel smears over a large area.

**Technique:** Wide per-channel spatial trailing.

---

## 262 — LineInLineOut_Increase

Line interlace effect with increasing frame offsets over time, gradually spreading the temporal range.

**Technique:** Time-increasing row interlace offset.

---

## 263 — LineInLineOut2_Increase

Second variant of increasing line interlace with different offset progression.

**Technique:** Alternate time-increasing row interlace.

---

## 264 — LineInLineOut3_Increase

Third variant with yet another offset progression pattern for line interlacing.

**Technique:** Third-variant time-increasing row interlace.

---

## 265 — SquareByRow2Plus

Enhanced SquareByRow2 with additional blending or offset, creating richer temporal block banding.

**Technique:** Enhanced row-block temporal blending.

---

## 266 — FrameSep

Separates the frame into sections that each display a different history frame, creating a split-screen temporal montage.

**Technique:** Frame-section temporal separation.

---

## 267 — FrameSep2

Second variant of frame separation with different sectioning geometry.

**Technique:** Alternate frame-section temporal layout.

---

## 268 — FrameStopStart

Alternately freezes (stops) and resumes (starts) frame updates in sections, creating a stuttered/freeze-frame mosaic.

**Technique:** Section-based frame freeze/resume alternation.

---

## 269 — OutOfOrder

Displays frame history in a shuffled/out-of-order arrangement across the image, creating a temporal puzzle effect.

**Technique:** Shuffled frame history spatial mapping.

---

## 270 — TrackingDown

Shifts rows downward progressively like a VHS tracking error, with the displacement increasing from top to bottom.

**Technique:** Progressive downward row displacement.

---

## 271 — TrackingDownBlend

Tracking down effect blended with the original for a softer VHS-tracking look.

**Technique:** Blended progressive downward row displacement.

---

## 272 — TrackingRev

Reverse tracking—rows shift upward instead of downward.

**Technique:** Progressive upward row displacement.

---

## 273 — TrackingMirror

Tracking displacement mirrored from center, creating symmetrical tracking errors spreading from the middle.

**Technique:** Center-mirrored row displacement.

---

## 274 — BlockPixels

Groups pixels into large blocks (larger than PixelateRect), each showing a single averaged color. Creates a heavily pixelated, blocky look.

**Technique:** Large-block pixel averaging.

---

## 275 — FrameChop

Chops the frame into segments and rearranges them, creating a cut-up/shuffled image.

**Technique:** Frame segment rearrangement.

---

## 276 — YLineDown

Shifts pixels downward along Y-axis lines with frame history blending.

**Technique:** Y-line-based downward shift with blending.

---

## 277 — YLineDownBlend

YLineDown with additional alpha blending for smoother transitions.

**Technique:** Blended Y-line downward shift.

---

## 278 — SquareDiff1

Computes difference between square-block-sampled history frames, highlighting changes at block level.

**Technique:** Block-level temporal difference.

---

## 279 — LineAcrossX

Draws a line across the frame at an animated position, with history frame data along the line.

**Technique:** Animated horizontal line from history.

---

## 280 — ColorGlitch

Randomly glitches color values—swapping, corrupting, or shifting channels unpredictably. Creates a colorful digital corruption effect.

**Technique:** Random per-channel color corruption.

---

## 281 — PixelShiftUp

Shifts all pixels upward by an animated amount, with the bottom filling from frame history.

**Technique:** Animated upward pixel shift.

---

## 282 — PixelShiftDown

Shifts all pixels downward by an animated amount, with the top filling from frame history.

**Technique:** Animated downward pixel shift.

---

## 283 — PixelShiftLeft

Shifts all pixels leftward by an animated amount.

**Technique:** Animated leftward pixel shift.

---

## 284 — PixelShiftRight

Shifts all pixels rightward by an animated amount.

**Technique:** Animated rightward pixel shift.

---

## 285 — PixelShiftDiagonal

Shifts pixels diagonally (both X and Y simultaneously), creating a diagonal scrolling effect.

**Technique:** Animated diagonal pixel shift.

---

## 286 — WaveBlend

Blends the current frame with a wave-distorted version of itself, creating a soft waving blend effect.

**Technique:** Wave-distorted self-blend.

---

## 287 — WaveBlendX2

Double-intensity wave blending for a more pronounced wavy distortion.

**Technique:** Intensity-doubled wave self-blend.

---

## 288 — SineWaveBlend

Blends frames using sine-wave modulated alpha, so the blend strength ripples across the image.

**Technique:** Sine-wave alpha modulation blending.

---

## 289 — CosineWaveBlend

Blends using cosine-wave modulated alpha—complementary wave pattern to SineWaveBlend.

**Technique:** Cosine-wave alpha modulation blending.

---

## 290 — SpiralWave

Applies a spiral-shaped wave distortion from the center, twisting the image outward in a spiral pattern.

**Technique:** Spiral polar-coordinate wave distortion.

---

## 291 — RadialBlur

Motion blur radiating outward from the center point, creating a speed-zoom blur.

**Technique:** Radial vector motion blur.

---

## 292 — ZoomBlur

Zooming motion blur creating a rush/tunnel-speed effect toward or away from center.

**Technique:** Zoom-direction motion blur.

---

## 293 — RotateBlend

Blends the current frame with a rotated version, creating a spinning overlay effect.

**Technique:** Rotation blending.

---

## 294 — MirrorWave

Mirrors the image at a wave-shaped boundary that undulates over time.

**Technique:** Wave-boundary mirror reflection.

---

## 295 — MirrorWaveX

Horizontal variant of wave mirror—the mirror boundary undulates horizontally.

**Technique:** Horizontal wave-boundary mirror.

---

## 296 — MirrorWaveY

Vertical variant of wave mirror—the mirror boundary undulates vertically.

**Technique:** Vertical wave-boundary mirror.

---

## 297 — PixelDrift

Gradually shifts pixel positions over time using sine/cosine modulation, creating a floating drift effect.

**Technique:** Sine/cosine animated position drift.

---

## 298 — PixelDriftX

Horizontal-only pixel drift variant.

**Technique:** Horizontal sine-animated position drift.

---

## 299 — PixelDriftY

Vertical-only pixel drift variant.

**Technique:** Vertical sine-animated position drift.

---

## 300 — ColorPulse

Pulses color intensity using sine/cosine modulation per channel, creating a rhythmic color brightness oscillation across the image.

**Technique:** Per-channel sine/cosine amplitude modulation.

---

## 301 — ColorPulseRGB

Each R, G, B channel pulses at a different frequency, creating desynchronized color pulsing.

**Technique:** Multi-frequency per-channel pulse.

---

## 302 — ColorPulseXor

Color pulsing combined with XOR operations for a digital pulsating distortion.

**Technique:** XOR-modulated color pulse.

---

## 303 — GlitchBlock

Randomly displaces rectangular blocks, creating the classic datamosh/macroblock glitch look.

**Technique:** Random rectangular block displacement.

---

## 304 — GlitchBlockXor

Block displacement combined with XOR on the displaced blocks for added digital artifact texture.

**Technique:** XOR'd random block displacement.

---

## 305 — GlitchLine

Randomly displaces entire horizontal lines, creating a horizontal shatter/glitch.

**Technique:** Random horizontal line displacement.

---

## 306 — GlitchLineX

Extended variant of line glitch with larger/more frequent displacements.

**Technique:** Intensified random line displacement.

---

## 307 — NoiseBlend

Blends random noise with the current frame, creating a film-grain-like static overlay.

**Technique:** Random noise alpha blending.

---

## 308 — NoiseBlendX2

Double-intensity noise blend for heavier static interference.

**Technique:** Intensified noise blending.

---

## 309 — NoiseXor

XORs the frame with random noise values, creating digital static distortion.

**Technique:** Random noise XOR.

---

## 310 — ChannelShift

Rotates color channels by one position (R→G→B→R) with animated shift amount. Creates a cycling false-color effect.

**Technique:** Animated circular channel rotation.

---

## 311 — ChannelShiftX

Extended channel shift with larger or variable rotation amounts.

**Technique:** Variable-rate channel rotation.

---

## 312 — ChannelRotate

Continuously and smoothly rotates color channels, creating an evolving false-color animation.

**Technique:** Continuous channel rotation.

---

## 313 — DiagonalStretch

Stretches the image diagonally, creating a sheared distortion along a 45-degree axis.

**Technique:** Diagonal coordinate scaling.

---

## 314 — DiagonalStretchX

Extended diagonal stretch with greater intensity.

**Technique:** Intensified diagonal coordinate scaling.

---

## 315 — DiagonalMirror

Mirrors the image along a diagonal axis, creating diagonal symmetry.

**Technique:** Diagonal-axis coordinate reflection.

---

## 316 — SquareWave

Applies a square wave function to pixel brightness, creating hard-edged alternating bright/dark bands.

**Technique:** Square wave brightness modulation.

---

## 317 — SquareWaveX

Extended square wave with variable frequency/amplitude.

**Technique:** Variable square wave modulation.

---

## 318 — SquareWaveBlend

Square wave brightness banding blended with frame history.

**Technique:** Square wave with temporal blending.

---

## 319 — TriangleWave

Applies a triangle wave function to brightness, creating softer saw-like alternating bright/dark bands.

**Technique:** Triangle wave brightness modulation.

---

## 320 — TriangleWaveBlend

Triangle wave banding blended with frame history.

**Technique:** Triangle wave with temporal blending.

---

## 321 — SawtoothWave

Applies a sawtooth wave—brightness ramps up then drops sharply in repeating bands.

**Technique:** Sawtooth wave brightness modulation.

---

## 322 — SawtoothWaveBlend

Sawtooth wave with frame history blending.

**Technique:** Sawtooth wave with temporal blending.

---

## 323 — PulseWave

Short-duty-cycle pulse wave creating narrow bright stripes on a darker background.

**Technique:** Pulse wave brightness pattern.

---

## 324 — PulseWaveBlend

Pulse wave with frame blending.

**Technique:** Pulse wave with temporal blending.

---

## 325 — StepWave

Staircase/stepped brightness pattern creating quantized brightness bands.

**Technique:** Stepped quantized brightness.

---

## 326 — StepWaveBlend

Stepped brightness blended with frame history.

**Technique:** Step wave with temporal blending.

---

## 327 — RippleEffect

Concentric ripple distortion from center, displacing pixels outward in circular wavefronts like a water drop impact.

**Technique:** Radial sine-wave coordinate displacement.

---

## 328 — RippleEffectX2

Double-intensity ripple with more wave cycles visible.

**Technique:** Intensified radial ripple distortion.

---

## 329 — ShockWave

Expanding ring of distortion from center, like a single explosion shockwave propagating outward.

**Technique:** Animated expanding-radius ring distortion.

---

## 330 — ShockWaveBlend

Shockwave distortion blended with frame history.

**Technique:** Shockwave with temporal blending.

---

## 331 — TwistEffect

Rotational twisting distortion around the center—pixels near center rotate more than those at edges, creating a spiral twist.

**Technique:** Radius-dependent rotational coordinate warp.

---

## 332 — TwistEffectBlend

Twist effect blended with frame history.

**Technique:** Spiral twist with temporal blending.

---

## 333 — FishEye

Barrel distortion simulating a fish-eye lens—center area magnified, edges compressed.

**Technique:** Radial barrel coordinate distortion.

---

## 334 — FishEyeBlend

Fish-eye distortion blended with frame history.

**Technique:** Fish-eye with temporal blending.

---

## 335 — Kaleidoscope

Folds the image into mirrored symmetrical sectors like a kaleidoscope, creating mandala-like symmetry.

**Technique:** Angular sector folding and mirroring.

---

## 336 — KaleidoscopeBlend

Kaleidoscope effect with frame history blending for temporal depth.

**Technique:** Kaleidoscope with temporal blending.

---

## 337 — TunnelEffect

Warps the image into a receding tunnel using logarithmic polar coordinates, creating the illusion of depth.

**Technique:** Logarithmic polar coordinate transformation.

---

## 338 — TunnelEffectBlend

Tunnel effect blended with frame history.

**Technique:** Tunnel warp with temporal blending.

---

## 339 — VortexEffect

Spinning vortex distortion that swirls pixels around the center with intensity increasing toward the core.

**Technique:** Spiral vortex coordinate transformation.

---

## 340 — VortexEffectBlend

Vortex effect with frame blending.

**Technique:** Vortex with temporal blending.

---

## 341 — ColorDrift

Gradually drifts each channel's brightness in a random direction over time, creating slow color wandering.

**Technique:** Animated per-channel brightness drift.

---

## 342 — ColorDriftX

Extended color drift with larger drift range.

**Technique:** Intensified per-channel brightness drift.

---

## 343 — RGBShift

Shifts each R, G, B channel to a slightly different spatial position, creating chromatic aberration/color fringing at edges.

**Technique:** Per-channel spatial coordinate offset.

---

## 344 — RGBShiftX

Extended RGB shift with larger offsets for more pronounced chromatic aberration.

**Technique:** Intensified per-channel spatial offset.

---

## 345 — ChromaticAberration

Simulates lens chromatic aberration—channels spread radially outward from center, with more separation at edges.

**Technique:** Radial distance-proportional per-channel offset.

---

## 346 — ChromaticAberrationX

Extended chromatic aberration with greater radial spread.

**Technique:** Intensified radial per-channel offset.

---

## 347 — Posterize

Reduces the number of distinct color levels per channel (quantization), creating a poster-art look with flat color regions.

**Technique:** Color level quantization.

---

## 348 — PosterizeBlend

Posterization blended with frame history for temporal depth.

**Technique:** Posterization with temporal blending.

---

## 349 — Solarize

Applies a solarization curve—pixels above a threshold are inverted, creating the classic darkroom solarization effect with partial negative tones.

**Technique:** Threshold-based partial color inversion.

---

## 350 — SolarizeBlend

Solarization blended with frame history.

**Technique:** Solarization with temporal blending.

---

## 351 — GammaBright

Applies a gamma curve greater than 1, brightening midtones while preserving blacks and whites. Creates a lifted, bright look.

**Technique:** Gamma > 1 brightness adjustment.

---

## 352 — GammaDark

Applies a gamma curve less than 1, darkening midtones. Creates a moody, shadowed look.

**Technique:** Gamma < 1 darkness adjustment.

---

## 353 — ContrastBoost

Increases contrast by pushing values away from midpoint (128)—darks get darker, brights get brighter.

**Technique:** Midpoint-based contrast expansion.

---

## 354 — ContrastReduce

Decreases contrast by pulling values toward midpoint, creating a flat, washed-out look.

**Technique:** Midpoint-based contrast compression.

---

## 355 — EdgeGlowBlend

Applies edge detection then adds a glow around detected edges, blended with frame history.

**Technique:** Edge detection + glow + temporal blending.

---

## 356 — FrameBlendMulti

Blends multiple history frames (more than 3) together for heavy temporal smoothing.

**Technique:** Multi-frame temporal blend averaging.

---

## 357 — FrameBlendMultiX

Extended multi-frame blend with more history frames included.

**Technique:** Extended multi-frame temporal averaging.

---

## 358 — AcidTrailsBlend

Acid-style trails with strong color saturation and wave distortion blended together—a signature psychedelic trail effect.

**Technique:** Saturated wave-distorted temporal trails.

---

## 359 — AcidGlitchX

Acid-colored glitch artifacts with strong color shifts and random displacement, creating a vivid digital acid trip.

**Technique:** Color-boosted random glitch displacement.

---

## 360 — AlphaXorBlend

XOR blending controlled by alpha parameter—at low alpha, subtle XOR artifacts; at high alpha, full digital distortion.

**Technique:** Alpha-parameterized XOR blending.

---

## 361 — AlphaXorBlendDouble

Double-intensity alpha XOR blend for more aggressive distortion.

**Technique:** Intensity-doubled alpha XOR blending.

---

## 362 — AndOrXorStrobeScale

Cycles through AND, OR, XOR with additional per-channel scaling, creating an evolving multi-phase bitwise strobe.

**Technique:** Cycling AND/OR/XOR with scaling.

---

## 363 — AveragePixelsXorBlend

Averages neighboring pixels then XORs with frame history, creating a blurred-XOR hybrid.

**Technique:** Spatial average + history XOR blend.

---

## 364 — BitwiseRotateBlend

Bit-rotates pixel channel values and blends with history, creating subtle binary-level distortion.

**Technique:** Bit rotation + temporal blending.

---

## 365 — BitwiseRotateDiffBlend

Bit rotation of the frame difference between current and history, highlighting motion with binary artifacts.

**Technique:** Bit-rotated frame difference blend.

---

## 366 — BitwiseXorScaleBlend

XOR with animated scale factor and frame history blending.

**Technique:** Scaled XOR + temporal blend.

---

## 367 — BlackAndWhiteStrobe

Alternates rapidly between a grayscale and a full-color version of the image.

**Technique:** Alternating desaturation strobe.

---

## 368 — BlendAlphaXorScale

Alpha-blended XOR with additional scaling for controlled-intensity digital distortion.

**Technique:** Scaled alpha XOR blend.

---

## 369 — BlendBurredXor

Blurs the frame, then XORs with history, creating a soft-focus digital distortion.

**Technique:** Blur + XOR blend.

---

## 370 — BlendCombinedXor

Multi-layer combined XOR using several history frames, creating dense layered digital artifacts.

**Technique:** Multi-frame combined XOR.

---

## 371 — BlendIncreaseRGB

Increases each R, G, B channel incrementally while blending with history, creating a color-accumulating glow.

**Technique:** Incremental per-channel boost + blending.

---

## 372 — BlendThreeXor

XOR of three different frame history layers with the current frame, creating triple-source digital distortion.

**Technique:** Three-frame XOR combination.

---

## 373 — BlurDistortionBlend

Combines spatial blur with coordinate distortion and frame blending—a triple-effect combo.

**Technique:** Blur + distortion + temporal blend.

---

## 374 — ColorAccumulate

Accumulates color channel values over time, letting brightness build up in a feedback-like way.

**Technique:** Temporal color accumulation.

---

## 375 — ColorAccumulateBlend

Color accumulation with frame history blending for controlled build-up.

**Technique:** Blended color accumulation.

---

## 376 — ColorAccumulateXor

Color accumulation combined with XOR operations, creating an evolving digital buildup.

**Technique:** XOR-modulated color accumulation.

---

## 377 — ColorChannelBlend

Blends individual color channels with corresponding channels from history frames.

**Technique:** Per-channel temporal blending.

---

## 378 — ColorChannelXor

XORs individual channels with history frame channels.

**Technique:** Per-channel temporal XOR.

---

## 379 — ColorCollectionEnergy

Energized/boosted color values collected from multiple history frames, creating a vibrant multi-frame composite.

**Technique:** Boosted multi-frame color collection.

---

## 380 — ColorCollectionWave

Collects colors from history frames using wave-modulated selection indices, creating a pulsating temporal color composite.

**Technique:** Wave-indexed multi-frame color collection.

---

## 381 — ColorFadeXor

Gradual color fade combined with XOR operations for a morphing digital transition.

**Technique:** Fading XOR blend.

---

## 382 — ColorIntensityBlend

Blends with frame history weighted by pixel color intensity—brighter pixels blend more.

**Technique:** Luminance-weighted temporal blending.

---

## 383 — ColorIntensityXor

XOR operations weighted by luminance for brightness-dependent digital distortion.

**Technique:** Luminance-weighted XOR.

---

## 384 — ColorMoveBlend

Shifts colors in a direction while blending with history, creating moving color trails.

**Technique:** Directional color shift + blend.

---

## 385 — ColorPixelBlend

Per-pixel conditional color blending based on color values.

**Technique:** Conditional per-pixel color blend.

---

## 386 — ColorPixelXor

Per-pixel conditional XOR based on color values.

**Technique:** Conditional per-pixel XOR.

---

## 387 — ColorScaleBlend

Scales colors dynamically while blending with history.

**Technique:** Dynamic color scaling + blend.

---

## 388 — ColorWaveXor

Wave-modulated XOR per channel, creating sine-pattern digital artifacts.

**Technique:** Sine-wave XOR per channel.

---

## 389 — CosineMultiplyBlend

Multiplies pixel values by cosine-derived factors and blends with history, creating smooth oscillating brightness variations.

**Technique:** Cosine-weighted multiply + blend.

---

## 390 — DarkModBlendXor

Dark modular blending with additional XOR, creating dark moody artifacts.

**Technique:** Dark modular blend + XOR.

---

## 391 — DifferenceBlend

Computes absolute difference between current and history frames and blends, highlighting motion as bright areas.

**Technique:** Frame difference blend.

---

## 392 — DifferenceXorBlend

Frame difference combined with XOR for dual-layer motion-highlighting digital distortion.

**Technique:** Frame difference + XOR blend.

---

## 393 — DistortBlend

Coordinate distortion (sine warp) combined with multi-frame blending.

**Technique:** Sine-distortion + temporal blend.

---

## 394 — DiamondPatternBlend

Diamond pattern overlay blended with frame history for a tiled temporal effect.

**Technique:** Diamond pattern + temporal blend.

---

## 395 — FadeBlendXor

Gradual fade combined with XOR and blending for a smooth evolving digital distortion.

**Technique:** Fade + XOR + blend.

---

## 396 — FlashBlendXor

Bright flash with XOR and blending, creating periodic digital flash bursts.

**Technique:** Flash + XOR + blend.

---

## 397 — GhostTrailsBlend

Layered ghost trails from multiple history frames blended together for smooth spectral motion echoes.

**Technique:** Multi-frame ghost trail blending.

---

## 398 — AddInvert

Adds the inverted (negative) version of frame history to the current frame, creating surreal doubled imagery with reversed tones.

**Technique:** Additive inverted frame blending.

---

## 399 — AlphaBlendSimple

Simple 50/50 alpha blend between current frame and first history frame.

**Technique:** Simple half-and-half temporal blend.

---

## 400 — AlphaBlendDoubleX

Double-intensity alpha blending with accelerated cross-fade animation between frames.

**Technique:** Intensity-doubled alpha blending.

---

## 401 — AlphaStrobeBlendX

Strobing alpha blend that rapidly switches blend intensity, combined with frame history.

**Technique:** Strobed alpha blending.

---

## 402 — BitwiseAndBlend

Bitwise AND between current and history frames with additional blending.

**Technique:** Bitwise AND + blend.

---

## 403 — BitwiseXorAverage

XORs the pixel with the average of multiple history frames.

**Technique:** Multi-frame average XOR.

---

## 404 — BitwiseXorBlendX

Extended XOR blend using more history frames for deeper artifact layering.

**Technique:** Extended multi-frame XOR blend.

---

## 405 — BlackStrobe

Alternates between the normal image and a fully black frame, creating a harsh on/off strobe.

**Technique:** Alternating black frame strobe.

---

## 406 — BlendAlphaXorX

Extended alpha XOR blend with additional complexity.

**Technique:** Extended alpha XOR blend.

---

## 407 — BlendCombinedValuesX

Combines multiple blending operations—add, XOR, average—into a single compound filter.

**Technique:** Multi-operation combined blend.

---

## 408 — BlendFor360

Blends across a large range of frame history (up to 360-degree coverage), creating a full-cycle temporal composite.

**Technique:** Wide-range temporal blending.

---

## 409 — BlendForward16

Blends 16 consecutive history frames forward, creating a 16-frame-deep motion blur.

**Technique:** 16-frame forward temporal blur.

---

## 410 — BlendForward32

Blends 32 consecutive history frames for a very deep, heavy motion blur.

**Technique:** 32-frame forward temporal blur.

---

## 411 — BlendFromXtoY

Blends between two specific history frame indices, with the range animated over time.

**Technique:** Animated frame-range blending.

---

## 412 — BlendIncreaseX

Blending with progressively increasing intensity over time.

**Technique:** Time-increasing blend intensity.

---

## 413 — BlendRedGreenBlue

Selectively blends R, G, B channels from different history frames—red from one, green from another, blue from a third.

**Technique:** Per-channel history frame selection blend.

---

## 414 — BlendWithColorX

Blends with an animated solid color that cycles through hues over time.

**Technique:** Animated solid color blend.

---

## 415 — BlendAngle

Blends frames at angle-derived positions, creating a rotational gradient between time points.

**Technique:** Angle-based frame blending.

---

## 416 — BlockScale

Scales pixel values within blocks, with different blocks getting different scale factors.

**Technique:** Per-block brightness scaling.

---

## 417 — BlockStrobe

Alternates block visibility in a tiled pattern, creating a checkerboard strobe.

**Technique:** Block-tiled strobing.

---

## 418 — BlockXor

XORs pixels within tiled blocks with history frame blocks.

**Technique:** Block-tiled XOR.

---

## 419 — BlockyTrails16

Creates blocky 16×16 pixel motion trails from frame history, like trailing pixelated afterimages.

**Technique:** 16×16 block temporal trailing.

---

## 420 — BlockyTrails32

32×32 pixel block trails from frame history—larger, more prominent blocky afterimages.

**Technique:** 32×32 block temporal trailing.

---

## 421 — BlurDistortionX

Blur combined with sine-based positional distortion for a ghostly, warped soft-focus look.

**Technique:** Spatial blur + coordinate distortion.

---

## 422 — CannyStrobe

Canny-style edge detection that strobes on and off, alternating between the outline and the full image.

**Technique:** Alternating edge detection display.

---

## 423 — ColorFadeSlow

Very slow color fade toward a target hue, creating a gradual tinting effect over many frames.

**Technique:** Slow time-interpolated color tint.

---

## 424 — ColorFibonacci

Selects colors from a Fibonacci-indexed pattern, creating mathematically-beautiful color progression.

**Technique:** Fibonacci-sequence color selection.

---

## 425 — CurtainEffect

Opens/closes like a curtain—a vertical split wipe that reveals/hides frame history behind the current frame.

**Technique:** Animated vertical split transition.

---

## 426 — DarkColorFibonacci

Dark-variant Fibonacci color selection for a moody, mathematically-ordered color palette.

**Technique:** Dark Fibonacci-sequence color selection.

---

## 427 — DarkColorsBlend

Blends selectively with darker color values from history, biasing toward shadows.

**Technique:** Dark-biased selective blending.

---

## 428 — EnergizeBlend

Amplifies the difference between current and history frames, creating high-energy emphasis on motion.

**Technique:** Amplified frame difference blending.

---

## 429 — AverageLines

Averages each row's pixels for a horizontally-smoothed, banded look.

**Technique:** Row-wise pixel averaging.

---

## 430 — AverageLinesBlendX

Row averaging blended with frame history for temporal depth.

**Technique:** Row averaging + temporal blend.

---

## 431 — BlendRowAlpha

Alpha-blends specific rows with history frames, creating selective horizontal band blending.

**Technique:** Row-selective alpha blending.

---

## 432 — BlendInOut

Progressive fade in/out effect cycling between full opacity and transparency.

**Technique:** Cycling opacity modulation.

---

## 433 — ColorFlashIncreaseX

Color flash with increasing intensity over time—flashes get progressively brighter.

**Technique:** Time-increasing color flash.

---

## 434 — ColorIncreaseInOut

Color intensity increases then decreases in a cycle, creating a breathing color effect.

**Technique:** Cycling color intensity modulation.

---

## 435 — ColorLinesX

Horizontal color lines with variable color and spacing, animated over time.

**Technique:** Animated horizontal colored lines.

---

## 436 — ColorMoveDownX

Shifts color channels downward at different rates, creating vertical chromatic dragging.

**Technique:** Per-channel downward shift.

---

## 437 — ColorOrderSwapX

Swaps the order of color channels with animation—cycling through all permutations.

**Technique:** Animated channel permutation cycling.

---

## 438 — ColorPulseAlphaX

Color pulse modulated by alpha parameter for controllable pulse intensity.

**Technique:** Alpha-controlled color pulsing.

---

## 439 — ColorRowShiftX

Shifts colors differently on each row, creating horizontal rainbow banding.

**Technique:** Per-row color channel shifting.

---

## 440 — ColorShiftXorX

Color shifting combined with XOR operations for animated digital color distortion.

**Technique:** Color shift + XOR.

---

## 441 — CopyXorAlphaX

Copies frame with XOR and alpha blending for a layered digital copy effect.

**Technique:** Copy + XOR + alpha blend.

---

## 442 — CycleShiftRGBX

Cycles through all 6 RGB permutations over time, creating a continuously rotating false-color animation.

**Technique:** Cycling RGB channel permutation.

---

## 443 — DarkNegateX

Darkens the image then applies partial negation, creating a dark, inverted-tone look.

**Technique:** Darken + partial inversion.

---

## 444 — DarkSelfAlphaX

Dark self-alpha blend creating a moody, dimmed version of the self-blend effect.

**Technique:** Dark alpha self-blend.

---

## 445 — DiagonalGlitch

Glitches along diagonal lines—displacing pixel data diagonally for a sheared glitch look.

**Technique:** Diagonal line displacement.

---

## 446 — DigitalHaze

Soft digital haze/fog effect using blurred history frame overlay.

**Technique:** Blurred history overlay for haze.

---

## 447 — DoubleXorBlend

XOR between current and history frames applied twice for deeper, more complex digital artifacts.

**Technique:** Double-pass XOR blend.

---

## 448 — EchoBlend

Creates an echo/delay effect by blending progressively older frames with decreasing opacity—like an audio echo but for video.

**Technique:** Multi-frame diminishing-weight echo blend.

---

## 449 — ElectricEdge

Applies Sobel edge detection then renders edges as bright, jagged electric-looking lines.

**Technique:** Sobel edges rendered as electric arcs.

---

## 450 — FlashColorStrobe

Bright colored flash strobe that cycles through different colors on each flash.

**Technique:** Cycling color flash strobe.

---

## 451 — FrameDiffXor

XORs the frame difference (current minus previous) with the current frame, highlighting motion with digital artifacts.

**Technique:** Frame difference XOR.

---

## 452 — GhostMirror

Ghost trails combined with horizontal mirroring—transparent echoes appear mirrored across the center.

**Technique:** Mirrored ghost trail overlay.

---

## 453 — GlitchSort

Sorts pixels/blocks by brightness or color value, creating a pixel-sorting glitch art effect.

**Technique:** Pixel/block brightness sorting.

---

## 454 — HeatWave

Simulates heat-shimmer distortion—vertical wavy displacement that increases from bottom to top, like hot rising air.

**Technique:** Bottom-up increasing vertical wave distortion.

---

## 455 — InterlaceBlend

Interlaces even and odd rows from different time points, creating a temporal interlace with blending.

**Technique:** Even/odd row temporal interlace + blend.

---

## 456 — InvertStrobe

Strobes with color inversion—alternates between normal and inverted colors rapidly.

**Technique:** Alternating color inversion strobe.

---

## 457 — KaleidoBlend

Simplified kaleidoscope with frame blending, creating symmetric patterns with temporal depth.

**Technique:** Simplified kaleidoscope + blend.

---

## 458 — LightStrobe

Bright white strobe flash effect—periodically flashes the entire frame to white.

**Technique:** Periodic white flash overlay.

---

## 459 — LineGlitchX

Extended line glitch with more aggressive and frequent horizontal line displacement.

**Technique:** Intensified horizontal line glitch.

---

## 460 — MosaicBlend

Mosaic tile pattern blended with frame history, creating a tiled temporal composite.

**Technique:** Block mosaic + temporal blend.

---

## 461 — NegatePulse

Pulsing negation—the image smoothly oscillates between normal and inverted using sine modulation.

**Technique:** Sine-modulated color inversion.

---

## 462 — OffsetGhost

Ghost overlay from history frames at an animated spatial offset, creating moving transparent echoes.

**Technique:** Spatially-offset ghost overlay.

---

## 463 — PixelateWave

Pixelation with a wave-shaped boundary—the block size undulates across the frame.

**Technique:** Wave-modulated pixelation.

---

## 464 — QuantizeBlend

Color quantization (reduced palette) blended with frame history.

**Technique:** Color quantization + temporal blend.

---

## 465 — RandomLines

Draws lines at random positions and orientations across the frame.

**Technique:** Random line position rendering.

---

## 466 — RippleDisplace

Water-ripple displacement—pixels are pushed outward from a center point in concentric rings.

**Technique:** Radial ripple coordinate displacement.

---

## 467 — RotateShift

Rotates the image with an animated shift, creating a spinning/rotating effect.

**Technique:** Animated rotation + shift.

---

## 468 — SaturationGlow

Boosts color saturation and adds a glowing bloom around saturated areas.

**Technique:** Saturation boost + bloom.

---

## 469 — ScaleToCenter

Scales pixel positions toward the center, creating a zoom-in/magnification effect.

**Technique:** Center-point coordinate scaling.

---

## 470 — ShadowMirror

Creates a shadow/darkened mirrored copy of the image, like a reflection on a dark surface.

**Technique:** Darkened mirror overlay.

---

## 471 — ShiftChannels

Shifts each color channel by a different frame-animated offset, creating desynchronized channel movement.

**Technique:** Per-channel animated offset shift.

---

## 472 — SliceGlitch

Slices the image into horizontal strips and randomly offsets some of them, creating a sliced glitch effect.

**Technique:** Random horizontal slice displacement.

---

## 473 — SobelGlow

Sobel edge detection with a bright glow added to the detected edges, creating luminous outlines.

**Technique:** Sobel edges + glow bloom.

---

## 474 — SpectralShift

Shifts colors through the visible spectrum over time, creating a continuously cycling rainbow-wash effect.

**Technique:** Continuous spectral hue cycling.

---

## 475 — SpiralTrail

Motion trails arranged in a spiral pattern from center, creating a swirling trail vortex.

**Technique:** Spiral-coordinate temporal trailing.

---

## 476 — SquareTrails

Block-shaped motion trails where each block retains prev frame data, creating a tiled trailing effect.

**Technique:** Block-aligned temporal trailing.

---

## 477 — StrobeNegate

Strobing with alternating color negation for a harsh flashing negative effect.

**Technique:** Alternating strobe + negation.

---

## 478 — ThermalBlend

Maps pixel brightness to a thermal (heat-map) color palette—blue for cold/dark, red/white for hot/bright—and blends with history.

**Technique:** False-color thermal mapping + blend.

---

## 479 — TintShift

Shifts the overall color tint of the image over time, cycling through different color washes.

**Technique:** Animated color tint cycling.

---

## 480 — TrailEcho

Echoing motion trails with each echo layer at decreasing opacity, like multiple afterimages fading away.

**Technique:** Multi-layer diminishing-opacity trailing.

---

## 481 — TransitionBlend

Generic smooth blend transition between current and history frames with animated alpha.

**Technique:** Animated alpha transition blend.

---

## 482 — TwistWarp

Rotational twist warping around center—combines rotation and radial distortion for a vortex-warp effect.

**Technique:** Combined rotational twist + radial warp.

---

## 483 — VerticalShift

Shifts the image vertically with animated offset.

**Technique:** Animated vertical position shift.

---

## 484 — VortexBlend

Spinning vortex distortion blended with frame history for temporal depth.

**Technique:** Vortex warp + temporal blend.

---

## 485 — WeavePattern

Creates an interlocking woven/weave pattern overlay from alternating horizontal and vertical strips.

**Technique:** Orthogonal strip interleaving pattern.

---

## 486 — WhiteBurst

Expanding white burst from center, like a flash-bang transitioning from a central bright point outward.

**Technique:** Expanding radial white transition.

---

## 487 — WiggleDisplace

Jittery per-pixel displacement creating a nervous, wiggling distortion.

**Technique:** Random coordinate jitter per pixel.

---

## 488 — XorPulseX

Pulsing XOR with animated intensity—the XOR strength oscillates over time.

**Technique:** Sine-animated XOR intensity.

---

## 489 — YellowShift

Shifts the image's color balance toward yellow by boosting red and green while reducing blue.

**Technique:** Yellow color bias shift.

---

## 490 — ZigzagGlitch

Zigzag-shaped row displacement—every other group of rows shifts in alternate directions, creating a zigzag tear pattern.

**Technique:** Alternating-direction row group displacement.

---

## 491 — AlphaModulate

Modulates alpha blending intensity per pixel based on position, creating spatially-varying blend strength.

**Technique:** Position-dependent alpha modulation.

---

## 492 — BlockSwap

Swaps rectangular blocks between the current frame and history frames, creating a tiled time-shuffle.

**Technique:** Block-level frame history swapping.

---

## 493 — ColorResonance

Amplifies specific color frequencies/harmonics, creating ringing/oscillating color emphasis.

**Technique:** Harmonic color amplification.

---

## 494 — DepthGlitch

Glitch artifacts that simulate depth—far areas glitch more than near areas, creating a 3D glitch illusion.

**Technique:** Brightness-as-depth glitch modulation.

---

## 495 — EchoShift

Echo trails with a spatial shift—each echo layer is offset both in time and space.

**Technique:** Spatially-shifting temporal echo.

---

## 496 — FractalNoise

Generates fractal noise (layered Perlin/simplex-style noise) and blends it with the image.

**Technique:** Multi-octave fractal noise blending.

---

## 497 — GradientRotate

Rotates a color gradient overlay over time, creating a spinning color wheel effect.

**Technique:** Animated rotating gradient.

---

## 498 — HarmonicShift

Shifts colors by harmonic intervals, creating musically-inspired color progressions.

**Technique:** Harmonic-interval color shifts.

---

## 499 — AcidWarp

Chaotic, multi-layered sine/cosine warping that creates organic, acid-trip-like flowing distortion.

**Technique:** Multi-frequency sine/cosine coordinate warping.

---

## 500 — BlendDiagonal

Animates brightness along diagonal lines using a triangular wave pattern. Diagonal bands move across the frame creating a sweeping light/dark stripe effect.

**Technique:** Animated diagonal triangular-wave brightness modulation.

---

## 501 — ChromaFlash

Flashes one color channel at a time in rotation (R→G→B), boosting it by 50%—creates a cycling single-channel color flash.

**Technique:** Rotating single-channel brightness boost.

---

## 502 — CircleWave

Concentric circular waves radiate from center, modulating brightness with a sine function of distance from center, animated outward over time.

**Technique:** Radial sine-wave brightness modulation from center.

---

## 503 — ColorCrush

Quantizes/crushes color values to a reduced palette that changes over time—animated posterization with a shifting crush factor.

**Technique:** Animated color quantization (posterize).

---

## 504 — CrosshatchBlend

Overlays an animated crosshatch pattern—thin horizontal and vertical lines darken the image in a woven grid that moves over time.

**Technique:** Animated crosshatch darkening pattern.

---

## 505 — CyberGlitch

Sparse random horizontal pixel displacement—2% of pixels get randomly shifted sideways by up to 250 pixels, creating sudden horizontal tears.

**Technique:** Sparse random horizontal pixel displacement.

---

## 506 — DarkPulse

Pulsing darkness on shadow areas—pixels below mid-brightness darken and brighten with a sine oscillation, while bright areas stay untouched.

**Technique:** Sine-pulsed shadow darkening.

---

## 507 — DiamondPatternX

Overlays an animated diamond (rhombus) pattern—Manhattan-distance-based diamonds tile the frame with brightness modulation.

**Technique:** Animated diamond/rhombus tiling brightness pattern.

---

## 508 — DigitalRain

Matrix-style digital rain—vertical streaks of bright green fall down the frame, tinting nearby pixels green while dimming red and blue.

**Technique:** Vertical green streak overlay (Matrix-style).

---

## 509 — DisplaceX

Sine-wave horizontal displacement—each row is shifted left/right by a sine function of its Y position, creating a gentle animated wobble.

**Technique:** Sine-modulated horizontal row displacement.

---

## 510 — DriftBlend

Blends current frame with a drifting (horizontally offset) version of the previous frame, creating a slow sideways ghosting trail.

**Technique:** Horizontally-drifting temporal blend.

---

## 511 — EdgePulse

Edge-detected pixels pulse brighter with a sine animation—detected edges glow and fade rhythmically while flat areas remain unchanged.

**Technique:** Pulsing edge brightness enhancement.

---

## 512 — FlameEffect

Simulates flames rising from the bottom—red/orange warming increases toward the bottom of the frame with animated sine flickering, while blue cools at the top.

**Technique:** Bottom-up warm color gradient with sine flicker.

---

## 513 — FlickerShift

Sparse random pixel inversion—every 5th pixel (animated pattern) gets color-inverted, creating an uneven rapid flicker.

**Technique:** Patterned sparse pixel inversion.

---

## 514 — GhostLayer

Layers up to 4 history frames as transparent ghost overlays with decreasing opacity (15%, 7.5%, 5%, 3.75%).

**Technique:** Multi-frame diminishing ghost layering.

---

## 515 — GlitchBlockX

Block-based glitch—16×16 pixel blocks randomly get their pixels shifted horizontally, creating periodic block-level displacement.

**Technique:** Random 16×16 block horizontal displacement.

---

## 516 — GlowPulse

Bright areas (>150 brightness) pulse with a sine glow—already-bright pixels get brighter and dimmer rhythmically, creating a pulsing bloom effect.

**Technique:** Sine-pulsed brightness boost on highlights.

---

## 517 — GridDistort

Checkerboard grid distortion—alternating 8×8 grid cells shift pixels left or right based on an animated offset.

**Technique:** Checkerboard-alternating grid pixel shifts.

---

## 518 — HexPattern

Overlays a hexagonal tile pattern that dims alternating hex regions, creating an animated honeycomb brightness pattern.

**Technique:** Animated hexagonal tile brightness modulation.

---

## 519 — HueRotate

True hue rotation—rotates the entire color wheel by an angle that continuously increases (full 360° cycle), using a proper color matrix transformation.

**Technique:** Color-matrix hue rotation.

---

## 520 — InterweaveX

Interweaves current and previous frames in alternating 2-pixel-wide vertical stripes that shift over time.

**Technique:** Alternating vertical stripe temporal interweave.

---

## 521 — JitterBlend

Random per-pixel horizontal jitter (±3 pixels) blended with the original—creates a subtle nervous/shaky displacement.

**Technique:** Random horizontal jitter with blend-back.

---

## 522 — KaleidoScope4

4-way kaleidoscope mirror—folds the image into quadrants by mirroring X and Y around center, blended with the original for a semi-transparent kaleidoscope.

**Technique:** Quadrant-fold mirror blend.

---

## 523 — LaserScan

A bright red laser scan line sweeps vertically down the frame, tinting a 3-pixel-wide band to bright red as it passes.

**Technique:** Vertical red scan line sweep.

---

## 524 — LightLeak

Simulates photographic light leaks—warm orange/yellow glow that drifts across the frame as a sine-modulated lens-flare-like overlay.

**Technique:** Animated warm light leak overlay.

---

## 525 — MeltDown

Vertical melt distortion—pixels are pulled from slightly above or below based on a sine wave, creating a dripping/melting effect.

**Technique:** Sine-wave vertical pixel displacement (melt).

---

## 526 — MirrorDiag

Diagonal mirror blend—alternating pixels (checkerboard) are blended with their diagonally-opposite counterpart, creating an animated diagonal symmetry.

**Technique:** Diagonal-opposite pixel blending.

---

## 527 — NeonGlow

Strong edges glow in neon colors—edge detection highlights are rendered as single bright channel values (R, G, or B rotating), creating neon-sign-style outlines.

**Technique:** Edge detection + rotating single-channel neon glow.

---

## 528 — NoiseBlendX

Subtle per-pixel noise modulation—each pixel's brightness is randomly varied by up to ±30%, creating a gentle animated grain.

**Technique:** Per-pixel random brightness noise.

---

## 529 — PixelDrift

Block-aligned horizontal drift—pixels shift sideways in 4-pixel groups with an animated ping-pong offset.

**Technique:** 4-pixel block horizontal drift.

---

## 530 — PlasmaWave

Classic plasma effect—overlapping sine waves on X and Y axes create a flowing plasma pattern that modulates RGB channels differently.

**Technique:** Dual-axis sine plasma color modulation.

---

## 531 — PrismSplit

Chromatic prism split—the red channel is offset rightward and blue leftward by an animated amount, simulating light passing through a prism.

**Technique:** Animated opposing R/B channel offset (chromatic aberration).

---

## 532 — PulseRadial

Radial brightness pulse—concentric rings of brightness modulation pulse outward from center, like a radar ping expanding.

**Technique:** Outward-pulsing radial sine brightness rings.

---

## 533 — RainbowStrobe

Strobes through rainbow colors—cycles through boosting R, G, then B channels in a 6-phase sequence, creating a fast rainbow flash.

**Technique:** 6-phase RGB channel boost cycling.

---

## 534 — RefractionX

Refraction displacement—pixels are shifted horizontally by a sine function of (x+y), simulating light bending through an uneven medium.

**Technique:** Diagonal sine-refraction horizontal displacement.

---

## 535 — ScanlineX

Animated CRT scanlines—every 4th row (animating downward) is darkened by 30%, simulating moving CRT scan lines.

**Technique:** Animated horizontal scanline darkening.

---

## 536 — ShatterEffect

Block-shatter displacement—32×32 blocks are displaced by random offsets that increase over time, simulating the image shattering into drifting pieces.

**Technique:** Time-expanding random block displacement.

---

## 537 — StaticNoise

TV static—5% of pixels are randomly replaced with monochrome noise values, creating scattered white noise dots.

**Technique:** Sparse random pixel static noise.

---

## 538 — TunnelVision

Animated vignette/tunnel effect—edges of the frame darken while center stays bright, with the vignette intensity pulsing over time.

**Technique:** Pulsing radial vignette.

---

## 539 — AberrationPulse

Pulsing chromatic aberration—red and blue channels are offset in opposite horizontal directions by an amount that oscillates with a sine wave.

**Technique:** Sine-pulsed opposing R/B channel offset.

---

## 540 — AquaWave

Aquatic wave distortion with color tinting—vertical sine waves displace pixels while boosting green/cyan and reducing blue, creating an underwater look.

**Technique:** Vertical wave displacement + aqua color tint.

---

## 541 — BinaryFlash

Animated binary threshold—converts to black and white with a threshold that oscillates via sine, creating a flashing high-contrast silhouette effect.

**Technique:** Oscillating-threshold binary conversion.

---

## 542 — BloomGlow

Bloom/glow effect—all pixels pulse brighter with a sine wave, and highlights (>200) get an extra boost, creating a pulsing cinematic bloom.

**Technique:** Sine-pulsed brightness + highlight bloom.

---

## 543 — CellularNoise

Cellular/Voronoi-style pattern—divides space into 16×16 cells and dims pixels based on distance from cell center, creating a cellular shadow pattern.

**Technique:** Distance-from-cell-center brightness modulation.

---

## 544 — ChromaShift2

Chromatic aberration variant—fixed red-right and blue-left channel offset with animated shift distance (3-7 pixels).

**Technique:** Animated chromatic channel separation.

---

## 545 — ColorBands

Horizontal animated color bands—the frame is divided into horizontal thirds (scrolling) that boost red, green, or blue respectively.

**Technique:** Scrolling horizontal RGB band boost.

---

## 546 — ColorVortex

Color vortex—hue shifts based on angle from center and distance, creating a spinning color wheel overlay that rotates over time.

**Technique:** Angle+distance based color vortex rotation.

---

## 547 — CrystalMosaic

Crystal/mosaic pixelation—samples from the center of each block (variable size 8-15), with block size animating over time.

**Technique:** Animated variable-size center-sampled mosaic.

---

## 548 — CubicDistort

Cubic barrel/pincushion distortion—applies a cubic polynomial warp from center with oscillating strength, creating breathing lens distortion.

**Technique:** Oscillating cubic lens distortion.

---

## 549 — DepthFade

Vertical depth fade—the image fades darker toward the top and brighter toward the bottom, with the gradient position pulsing via sine.

**Technique:** Pulsing vertical depth-gradient brightness.

---

## 550 — DiscoFlash

Disco flash—adds strong color (+100) to one channel at a time cycling R→G→B in 4-frame phases, creating a bold color strobe.

**Technique:** 3-phase channel additive color flash.

---

## 551 — DitherBlend

Animated checker dither—alternating pixels in a checkerboard are either brightened (×1.2) or darkened (×0.8), with the pattern flipping each frame.

**Technique:** Animated checkerboard brightness dithering.

---

## 552 — DoubleVision

Double vision displacement—the image is blended 50/50 with a horizontally-offset copy of itself, with the offset distance oscillating.

**Technique:** Oscillating horizontal double-image blend.

---

## 553 — DreamHaze

Dreamy haze—pixels are blended toward a bright haze with a spatially-varying sine pattern, creating soft, foggy dream-like brightness.

**Technique:** Spatially-varying sine haze blend.

---

## 554 — ElectricStorm

Sparse blue-white electric sparks—2% of pixels randomly flash to bright blue-white, simulating electric discharges.

**Technique:** Sparse random blue-white pixel flash.

---

## 555 — EmbossShift

Animated emboss effect—calculates difference from a diagonally-offset neighbor (offset distance animates 1-3), creating a shifting emboss relief.

**Technique:** Animated-offset diagonal emboss.

---

## 556 — FiberOptic

Radial fiber-optic coloring—pixels are tinted toward a single channel (R, G, or B) based on their angle from center, creating radial fiber-optic color spokes.

**Technique:** Angle-based radial channel boost.

---

## 557 — FilmGrain

Realistic film grain—adds random brightness noise (±15 levels) to each pixel, simulating photographic film grain.

**Technique:** Per-pixel additive random brightness noise.

---

## 558 — FireWorks

Firework burst—renders an expanding ring of bright red/orange from a fixed explosion center that grows outward and repeats.

**Technique:** Expanding radial ring burst animation.

---

## 559 — FluidMotion

Fluid flow distortion—pixels are displaced by sinusoidal flow in both X and Y, creating a smooth, liquid/flowing warp.

**Technique:** Dual-axis sinusoidal flow displacement.

---

## 560 — FogRoll

Rolling fog—a horizontal fog line oscillates vertically, with pixels below it gradually blending toward grey/white, simulating rolling ground fog.

**Technique:** Animated horizontal fog-line gradient blend.

---

## 561 — GlassRefract

Glass refraction distortion—pixels are displaced in both X and Y by a sine×cosine pattern, simulating light bending through wavy glass.

**Technique:** 2D sine-cosine glass refraction displacement.

---

## 562 — GlowTrails

Highlight glow trails—only bright pixels (>150) get a sine-pulsed brightness boost, creating glowing trails from bright objects.

**Technique:** Conditional highlight sine-pulsed glow.

---

## 563 — GridPulse

Pulsing grid overlay—grid lines (2px wide every 32px) pulse between their original color and a mid-grey, creating a breathing grid overlay.

**Technique:** Sine-pulsed grid line brightness.

---

## 564 — HalftoneBlend

Halftone printing effect—converts to dot pattern where dot radius is proportional to local brightness, blended with original colors. Dot size animates.

**Technique:** Animated halftone dot pattern blend.

---

## 565 — HeatDistort

Heat shimmer distortion—horizontal displacement that increases toward the top of the frame, simulating rising heat waves.

**Technique:** Height-proportional horizontal sine heat shimmer.

---

## 566 — HoloGlitch

Holographic glitch scanlines—fast-moving thin horizontal bands (5px) get color-shifted by adding different fixed amounts to R, G, B channels.

**Technique:** Moving scanline RGB offset bands.

---

## 567 — InfraredView

False-color infrared visualization—remaps brightness to a thermal-IR palette: hot whites→yellow→red→purple→dark blue for different intensity ranges, with pulsing intensity.

**Technique:** Brightness-to-infrared false color mapping.

---

## 568 — LavaLamp

Lava lamp blobs—two overlapping sinusoidal blob patterns create organic flowing shapes that shift between warm red/orange and cool green.

**Technique:** Dual-frequency sinusoidal blob color blending.

---

## 569 — LensFlare

Simulated lens flare—a bright circular flare moves horizontally across the upper third of the frame, adding intense white where it overlaps.

**Technique:** Animated moving circular lens flare spot.

---

## 570 — LightningBolt

Random lightning bolt—a jagged vertical bright line flashes briefly every 15 frames, with random horizontal jitter per row.

**Technique:** Periodic random vertical lightning flash.

---

## 571 — LiquidMetal

Liquid metal sheen—converts to metallic-looking surface using brightness-dependent silver/chrome coloring with animated reflections.

**Technique:** Brightness-based metallic color mapping.

---

## 572 — MatrixCode

Matrix code rain—columns of bright green characters rain downward with brighter leading edges and darker tails.

**Technique:** Vertical green code column animation.

---

## 573 — MirrorKaleid

Multi-mirror kaleidoscope—more complex mirror folding than KaleidoScope4, with additional angular reflections for richer symmetry.

**Technique:** Multi-angle mirror kaleidoscope.

---

## 574 — NightVision

Night vision simulation—converts to bright green monochrome with intensity amplification and slight noise, mimicking night-vision goggles.

**Technique:** Green monochrome conversion + amplification + noise.

---

## 575 — OilSlick

Oil slick rainbow—maps thin-film interference-style rainbow colors based on angle and distance, creating iridescent oil-on-water coloring.

**Technique:** Angle/distance thin-film interference coloring.

---

## 576 — ParticleField

Sparse bright particle dots scattered across the frame that animate over time, creating a starfield/particle effect overlay.

**Technique:** Random sparse bright pixel particle overlay.

---

## 577 — PinwheelSpin

Spinning pinwheel pattern—alternating angular sectors are darkened/brightened with the pattern rotating over time, like a spinning pinwheel.

**Technique:** Rotating angular sector brightness pattern.

---

## 578 — PixelStorm

Aggressive random pixel displacement—pixels are randomly relocated in a storm-like fashion, creating a raging pixel noise storm.

**Technique:** Heavy random pixel coordinate displacement.

---

## 579 — PlaidPattern

Overlapping horizontal and vertical color bands create a plaid/tartan pattern over the image, animated with scrolling.

**Technique:** Scrolling overlapping orthogonal color bands.

---

## 580 — PolarInvert

Polar coordinate inversion—maps pixels using inverted radial distance from center, creating a polar inside-out transformation.

**Technique:** Inverted polar coordinate mapping.

---

## 581 — PolychromeTint

Multiple overlapping color tints applied across the frame—different regions get different color washes, creating a polychrome mosaic of tints.

**Technique:** Multi-region color tint overlay.

---

## 582 — PopArtDots

Pop-art style large colored dots—renders oversized halftone dots in bold primary colors, simulating Roy Lichtenstein-style pop art.

**Technique:** Large bold-colored halftone dots.

---

## 583 — PrismaticEdge

Edge detection rendered with prismatic/rainbow colors—edges cycle through spectrum colors rather than white, creating rainbow outlines.

**Technique:** Rainbow-colored edge detection.

---

## 584 — PulseWarp

Pulsing spatial warp—the image warps inward/outward from center with a sine-pulsed intensity, creating a breathing zoom effect.

**Technique:** Sine-pulsed radial zoom warp.

---

## 585 — QuantumNoise

High-frequency random noise with quantum-inspired patterns—rapid, dense per-pixel noise that changes every frame.

**Technique:** Dense per-pixel per-frame random noise.

---

## 586 — QuiltBlend

Quilt-pattern blending—square patches alternate between current and history frames in a quilt-like grid layout.

**Technique:** Block-grid alternating temporal quilt blend.

---

## 587 — RadarSweep

Radar sweep—a rotating bright line sweeps clockwise from center like a radar display, brightening pixels it passes over.

**Technique:** Rotating angular bright sweep line.

---

## 588 — RaindropRipple

Simulates raindrop ripples—circular wave distortions appear at random positions, creating concentric ring displacement patterns.

**Technique:** Random radial ripple displacement.

---

## 589 — RasterBars

Retro raster bars—horizontal bars of bright color scroll vertically, creating Amiga/C64-style raster bar overlays.

**Technique:** Scrolling horizontal color bar overlay.

---

## 590 — RetroTube

CRT TV simulation—adds scanlines, slight curvature, color bleeding, and phosphor glow to mimic an old CRT television.

**Technique:** CRT scanlines + curvature + color bleed.

---

## 591 — RingWave

Expanding ring waves from center—concentric bright rings pulse outward, modulating brightness as they pass.

**Technique:** Outward-expanding concentric ring brightness pulse.

---

## 592 — RippleTank

Ripple tank physics simulation—overlapping circular wave patterns create interference fringe brightness modulation.

**Technique:** Circular wave interference pattern.

---

## 593 — RotatingPrism

A rotating prism overlay that shifts color channels at different angular offsets, creating a spinning prismatic color separation.

**Technique:** Angle-rotation-based channel separation.

---

## 594 — SandStorm

Sand storm noise—dense brownish/tan noise particles scattered across the frame, simulating a sandstorm overlay.

**Technique:** Dense warm-tinted noise particle overlay.

---

## 595 — SaturationPulse

Color saturation pulses—saturation is boosted and reduced with a sine wave, making colors alternately vivid and muted.

**Technique:** Sine-modulated saturation amplification.

---

## 596 — ScatterPixel

Pixel scattering—each pixel is displaced to a nearby random neighbor position, creating a dissolved/scattered look.

**Technique:** Random neighbor pixel coordinate scatter.

---

## 597 — ShadowPlay

Enhances shadows with animated movement—dark areas are emphasized and their boundaries shift with an animated offset.

**Technique:** Animated shadow enhancement and displacement.

---

## 598 — ShimmerGlass

Shimmering glass overlay—pixels are displaced by a rapidly oscillating fine-grain sine pattern, like looking through textured glass.

**Technique:** Fine-grain oscillating displacement shimmer.

---

## 599 — SilhouetteBlend

Silhouette extraction blended with original—creates a dark silhouette outline and blends it back at variable opacity.

**Technique:** Silhouette extraction + variable blend.

---

## 600 — SketchOutline

Pencil sketch effect—extracts edges using neighboring pixel differences and renders them as dark pencil-like outlines on a lighter background.

**Technique:** Edge-difference pencil outline rendering.

---

## 601 — SliceShift

Horizontal slice displacement—random horizontal strips shift left or right, creating a sliced/torn glitch effect.

**Technique:** Random horizontal slice offset.

---

## 602 — SmearMotion

Horizontal smear—pixels are pulled from a row of neighboring pixels to one side, creating a directional smearing/motion blur effect.

**Technique:** Horizontal neighbor-sampling directional smear.

---

## 603 — SmokeWisp

Smoke wisp overlay—soft, flowing cloud-like patterns drift across the frame, blending with the image at low opacity.

**Technique:** Flowing noise-based smoke overlay.

---

## 604 — SnowDrift

Snow drift effect—white particles drift diagonally downward, accumulating a scattered snow-like overlay.

**Technique:** Diagonal-drifting white particle overlay.

---

## 605 — SolarFlare

Solar flare from center—a bright warm radial burst emanates from a point, creating a lens-flare-like solar bloom effect.

**Technique:** Radial warm glow burst from center.

---

## 606 — SparkShower

Shower of bright spark pixels that rain downward, creating scattered bright falling dots.

**Technique:** Downward-falling bright pixel sparks.

---

## 607 — SpectrumWave

Spectrum-colored wave—a horizontal wave pattern colors the image through the visible spectrum, creating a scrolling rainbow wave.

**Technique:** Horizontal sine-wave spectrum coloring.

---

## 608 — SpiralZoom

Spiral zoom distortion—pixels are displaced both radially and rotationally from center based on their distance, creating a spiraling zoom tunnel.

**Technique:** Radial + rotational spiral displacement.

---

## 609 — SplitMirror

Split mirror—the frame is divided into halves (animated split point) with one half mirrored, revealing/hiding the mirror effect over time.

**Technique:** Animated split-point half-frame mirror.

---

## 610 — StarBurst

Star burst pattern—radiating lines from center create a star/starburst brightness pattern that rotates over time.

**Technique:** Rotating radial line starburst pattern.

---

## 611 — StaticPulse

TV static that pulses—static noise density oscillates with a sine wave, creating periods of heavy and light static.

**Technique:** Sine-density-modulated static noise.

---

## 612 — StencilCut

Stencil cut-out effect—removes image data in shaped patterns, leaving stencil-like cut-outs that reveal black or an alternate pattern.

**Technique:** Pattern-based pixel masking/removal.

---

## 613 — StippleShade

Stipple shading—converts brightness to a density of small dots, creating an artistic stippled/pointillist shading effect.

**Technique:** Brightness-to-dot-density stippling.

---

## 614 — StormCloud

Storm cloud overlay—dark, roiling cloud patterns pass over the frame, dimming and tinting areas beneath them.

**Technique:** Animated noise-based dark cloud shadow.

---

## 615 — StreakBlur

Directional streak blur—applies a linear motion blur along a single direction, creating speed-line-like streaks.

**Technique:** Single-direction linear motion blur.

---

## 616 — StrobeEdge

Edge detection that strobes—alternates between showing detected edges and the normal image on a rapid cycle.

**Technique:** Alternating edge-detect/normal frame strobe.

---

## 617 — SubpixelShift

Sub-pixel chromatic shift—each RGB channel is offset by a sub-pixel amount, creating a subtle LCD-like color fringing.

**Technique:** Per-channel sub-pixel offset.

---

## 618 — SwimDistort

Swimming distortion—pixels wobble in both X and Y with slow sine oscillation, creating an underwater swimming effect.

**Technique:** Dual-axis slow sine coordinate wobble.

---

## 619 — TangentWarp

Tangent-based warp—uses tangent function for displacement, creating discontinuous warping at tangent poles for aggressive distortion.

**Technique:** Tangent-function coordinate warping.

---

## 620 — TapeGlitch

VHS tape glitch—horizontal strips shift sideways with noise, and occasional tracking bars roll through, simulating degraded VHS playback.

**Technique:** Horizontal strip offset + rolling tracking bar.

---

## 621 — TechnoGrid

Techno grid overlay—bright neon-colored grid lines in a techy pattern overlay the image, pulsing to an animated rhythm.

**Technique:** Animated neon grid overlay.

---

## 622 — TeleportPixel

Pixel teleportation—random pixels are swapped to distant positions, creating a teleportation/scatter effect.

**Technique:** Random long-distance pixel position swap.

---

## 623 — TemporalBlur

Multi-frame temporal blur—averages multiple history frames to create a smooth motion blur over time.

**Technique:** Multi-frame temporal averaging.

---

## 624 — TerraFracture

Terra/ground fracture effect—irregular fracture lines cut across the frame with offset displacement on each side, like cracking earth.

**Technique:** Irregular fracture-line displacement.

---

## 625 — TextureWave

Texture wave—applies a sine-wave distortion that follows the image's natural texture/brightness contours.

**Technique:** Brightness-contour-following sine wave.

---

## 626 — ThresholdPulse

Pulsing threshold—converts to binary black/white with a threshold that sweeps from dark to bright and back, revealing different detail levels.

**Technique:** Sweeping binary threshold animation.

---

## 627 — TidalWave

Tidal wave—a large rolling wave distortion sweeps horizontally across the frame, pushing pixels sideways in a crest pattern.

**Technique:** Horizontal rolling wave displacement.

---

## 628 — TintCycle

Color tint cycling—the overall color tint rotates through warm→cool→warm in a continuous smooth cycle.

**Technique:** Smooth continuous color temperature cycling.

---

## 629 — TraceEdge

Edge tracing—detects and draws thin, continuous edge lines like a pen tracing the image contours.

**Technique:** Thin continuous edge line rendering.

---

## 630 — TriangleMosaic

Triangle mosaic—divides the image into triangular tiles and fills each with its average color, creating a geometric low-poly look.

**Technique:** Triangular tile average-color mosaic.

---

## 631 — TripleSplit

Triple split—divides the frame into three horizontal or vertical strips, each from a different time point.

**Technique:** Three-strip temporal split display.

---

## 632 — TurbulentFlow

Turbulent flow field—complex multi-frequency noise drives pixel displacement, creating chaotic fluid-like distortion.

**Technique:** Multi-frequency noise flow displacement.

---

## 633 — UnderwaterCaustic

Underwater caustic lighting—bright, shifting caustic light patterns dance across the frame, simulating light shining through water.

**Technique:** Sine-interference caustic light pattern.

---

## 634 — UnsharpPulse

Pulsing unsharp mask—sharpening intensity oscillates, making the image alternately sharp and soft with a rhythmic pulse.

**Technique:** Oscillating unsharp-mask intensity.

---

## 635 — VaporTrail

Vapor trail afterimages—moving bright areas leave fading white vapor-like trails behind them.

**Technique:** Bright-pixel temporal fading trail.

---

## 636 — VectorField

Vector field visualization—pixel displacement follows a simulated vector field pattern, creating flowing directional distortion.

**Technique:** Vector-field-directed pixel displacement.

---

## 637 — VelocityBlur

Velocity-proportional blur—pixels that have moved more between frames get more blur, creating a realistic motion-dependent blur.

**Technique:** Motion-proportional directional blur.

---

## 638 — VerticalMelt

Vertical melt—pixels drip/melt downward with animated speed, creating a vertical smearing melt effect.

**Technique:** Downward vertical pixel displacement melt.

---

## 639 — VHSTracking

VHS tracking distortion—horizontal offset bands roll through the frame with color bleeding, simulating bad VHS tracking.

**Technique:** Rolling horizontal offset + color bleed bands.

---

## 640 — VibrantPop

Vibrant color pop—saturates colors to maximum and increases contrast for a bold, eye-popping vivid look.

**Technique:** Saturation + contrast amplification.

---

## 641 — VignetteFlash

Vignette with flash—a dark vignette frames the image but periodically flashes bright, creating a camera-flash-within-vignette effect.

**Technique:** Pulsing vignette with periodic bright flash.

---

## 642 — VoronoiShatter

Voronoi diagram shatter—divides the image into Voronoi cells that displace/shatter apart over time.

**Technique:** Voronoi-cell-based displacement shatter.

---

## 643 — WarpSpeed

Warp speed radial zoom—pixels streak radially from center as if traveling at warp speed, creating a hyperspace zoom effect.

**Technique:** Radial streak zoom from center.

---

## 644 — WaterColor

Watercolor painting effect—simplifies colors and softens edges to create a painted watercolor look.

**Technique:** Color simplification + edge softening.

---

## 645 — WaveCollapse

Wave collapse animation—expanding wave patterns collapse inward then burst outward, creating a rhythmic wave interference pattern.

**Technique:** Alternating expand/collapse radial wave.

---

## 646 — WebPattern

Spider web pattern overlay—radiating lines and concentric rings create a web-like geometric pattern.

**Technique:** Radial + concentric web grid pattern.

---

## 647 — WhirlpoolSpin

Whirlpool spinning vortex—pixels spiral inward toward center with increasing angular velocity, creating a draining whirlpool effect.

**Technique:** Distance-proportional angular rotation vortex.

---

## 648 — WindBlast

Wind blast distortion—pixels are pushed strongly to one side with a turbulent offset, simulating a powerful wind.

**Technique:** Directional turbulent pixel push.

---

## 649 — WireframePulse

Wireframe pulsing overlay—a wireframe grid is drawn over the image with line brightness that pulses.

**Technique:** Pulsing wireframe grid overlay.

---

## 650 — XRayFlash

X-ray flash—periodically inverts to a blue-tinted negative reminiscent of X-ray film, then snaps back to normal.

**Technique:** Periodic blue-negative X-ray inversion.

---

## 651 — ZebraStripe

Zebra stripe pattern—alternating black and white diagonal stripes scroll across the frame, blended with the image.

**Technique:** Scrolling diagonal zebra stripe blend.

---

## 652 — ZenRipple

Gentle zen-garden ripples—very subtle, slow concentric ripples from center, creating a calm, meditative distortion.

**Technique:** Slow gentle radial ripple displacement.

---

## 653 — ZigzagWave

Zigzag wave distortion—rows are displaced in a zigzag pattern that animates over time, creating a zigzag wave effect.

**Technique:** Animated zigzag row displacement.

---

## 654 — ZoomPulse

Zoom pulse—the image alternately zooms in and out from center with a pulsing sine animation, like a heartbeat zoom.

**Technique:** Sine-pulsed center zoom in/out.

---

## 655 — ZoneTint

Zone tinting—divides the frame into zones that each get a different color tint, with zone boundaries animating over time.

**Technique:** Variable-zone animated color tinting.

---

## 656 — AcidDrip

Vertical dripping distortion—pixels drip downward in long streaks like acid melting, with animated drip speed.

**Technique:** Long-streak vertical drip displacement.

---

## 657 — AuroraWave

Aurora borealis effect—shimmering curtains of green/pink/blue light wave across the top of the frame.

**Technique:** Horizontal curtain sine-wave color overlay.

---

## 658 — BandPass

Band-pass filter—only allows a narrow range of brightness values through, dimming everything outside the animated pass band.

**Technique:** Animated brightness band-pass gating.

---

## 659 — BilinearStretch

Bilinear stretch—stretches the image non-uniformly using bilinear interpolation with animated anchor points.

**Technique:** Non-uniform bilinear stretch with moving anchors.

---

## 660 — BleedThrough

Color bleed-through—colors from adjacent pixels bleed outward as if printed on wet paper, creating a watercolor-bleed effect.

**Technique:** Directional neighbor color bleeding.

---

## 661 — BlockShatter

Block-level shatter—rectangular blocks fly apart in random directions over time, like an exploding mosaic.

**Technique:** Random-direction block explosion displacement.

---

## 662 — BlurMask

Selective blur mask—blurs only certain regions (e.g., by brightness or position) while keeping others sharp.

**Technique:** Region-selective conditional blur.

---

## 663 — BokehBlur

Bokeh blur simulation—out-of-focus areas render as circular bokeh blur discs, simulating shallow depth-of-field.

**Technique:** Circular disc blur (bokeh simulation).

---

## 664 — BounceWave

Bouncing wave—a sine wave with hard bounces (absolute value) displaces pixels, creating a rubber-bounce wave pattern.

**Technique:** Absolute-sine bouncing wave displacement.

---

## 665 — BrokenGlass

Broken glass fragment overlay—the image appears shattered into irregular glass-like shards with edge highlights.

**Technique:** Irregular polygon shard fragmentation.

---

## 666 — BubbleWarp

Bubble/sphere warping—circular regions inflate outward like bubbles, warping the image in spherical bumps.

**Technique:** Circular spherical inflation warp.

---

## 667 — CRTCurvature

CRT screen curvature—applies barrel distortion to simulate the curved surface of a CRT monitor, with scanlines and vignetting.

**Technique:** Barrel distortion + scanlines + vignette.

---

## 668 — CascadeBlend

Cascading multi-frame blend—blends a cascading series of history frames with progressively decreasing weights.

**Technique:** Multi-frame cascading weighted blend.

---

## 669 — CelShade

Cel shade/toon shader—quantizes colors to a few flat levels with hard edge outlines, creating a cartoon/anime aesthetic.

**Technique:** Color quantization + edge outline (toon).

---

## 670 — ChainReaction

Chain reaction spread—bright pixels trigger their neighbors to brighten in the next frame, spreading outward like a chain reaction.

**Technique:** Brightness-threshold spreading activation.

---

## 671 — ChannelDelay

Per-channel temporal delay—each RGB channel is sourced from a different time point, creating a temporal chromatic separation.

**Technique:** Per-channel different-frame sourcing.

---

## 672 — ChromaBleed

Chroma bleed—color information bleeds horizontally from saturated areas, simulating analog video chroma bleed.

**Technique:** Horizontal color saturation bleed.

---

## 673 — CircuitTrace

Circuit board trace pattern—draws right-angle line patterns reminiscent of PCB traces overlaid on the image.

**Technique:** Right-angle grid circuit pattern overlay.

---

## 674 — ClockWipe

Clock wipe transition—a radial wipe rotates around center like a clock hand, revealing the history frame behind the current frame.

**Technique:** Animated radial clock-hand wipe.

---

## 675 — CloudShadow

Cloud shadow overlay—dark shadow patches drift across the frame as if clouds are passing overhead.

**Technique:** Drifting noise-based shadow patch overlay.

---

## 676 — ColorBurn

Color burn blend mode—applies the Photoshop-style "color burn" blend between current and history frames, deepening dark colors.

**Technique:** Color burn composite blend.

---

## 677 — ColorHalves

Color halves—splits the frame into two halves with different color processing (e.g., warm left, cool right) with animated split.

**Technique:** Animated split-frame dual color grading.

---

## 678 — ComicDots

Comic book Benday dots—converts to large visible halftone dots in CMYK-like notation for a comic book print look.

**Technique:** Large color halftone Benday dot pattern.

---

## 679 — ConcentricPulse

Concentric brightness pulse—expanding rings of alternating brightness pulse outward from center.

**Technique:** Alternating expanding concentric brightness rings.

---

## 680 — CopperTone

Copper/bronze tone—maps colors to a warm copper-bronze monochrome palette, like sepia but warmer/more metallic.

**Technique:** Copper/bronze monochrome color mapping.

---

## 681 — CornerStretch

Corner stretch—warps the image by stretching corners outward, creating a pillow/pin-cushion type distortion from corners.

**Technique:** Corner-anchored outward stretch distortion.

---

## 682 — CosmicDust

Cosmic dust particles—overlays twinkling colored particles that drift slowly, creating a starfield/cosmic-dust effect.

**Technique:** Drifting colored particle twinkle overlay.

---

## 683 — CrossBlur

Cross-shaped blur—blurs in a cross/plus pattern (horizontal + vertical only), creating a star-filter-like soft focus.

**Technique:** Orthogonal cross-pattern blur.

---

## 684 — CrossProcess

Cross-process film look—shifted color curves simulating cross-processed film (E-6 in C-41), with boosted greens and contrasted reds.

**Technique:** Cross-process color curve simulation.

---

## 685 — CrystalEdge

Crystal edge highlight—edges are highlighted with bright crystalline-colored outlines, creating a jewel-like edge effect.

**Technique:** Bright crystal-colored edge highlighting.

---

## 686 — CubeRotate

Cube rotation—the frame is mapped onto a 3D cube face that rotates over time, showing the image from different angles with perspective.

**Technique:** 3D perspective cube-face rotation mapping.

---

## 687 — CurtainReveal

Curtain reveal transition—the image opens from center like parting curtains, revealing a history frame behind.

**Technique:** Center-opening curtain wipe transition.

---

## 688 — CyberPunk

Cyberpunk aesthetic—high contrast with neon magenta/cyan color grading, scanlines, and glitch artifacts for a cyberpunk look.

**Technique:** Neon color grade + scanlines + glitch.

---

## 689 — DataCorrupt

Data corruption simulation—randomly corrupts pixel data in block-sized chunks, mimicking file corruption artifacts.

**Technique:** Random block data corruption.

---

## 690 — DebrisField

Debris field particles—chunks of pixel data scatter outward from random points like exploding debris.

**Technique:** Outward-scattering pixel chunk particles.

---

## 691 — DeepFry

Deep fried meme aesthetic—extreme saturation boost, heavy sharpening, and JPEG-like compression artifacts for the "deep fried" meme look.

**Technique:** Extreme saturation + sharpening + artifact.

---

## 692 — DesyncRGB

RGB desynchronization—each color channel is offset in a different direction and by different amounts, creating severe chromatic desync.

**Technique:** Multi-directional per-channel offset desync.

---

## 693 — DiagonalWipe

Diagonal wipe transition—a diagonal line sweeps across the frame, transitioning between current and history frames.

**Technique:** Diagonal line wipe transition.

---

## 694 — DigitalArtifact

Digital artifact blocks—randomly places colored artifact rectangles (macroblocks) across the frame, simulating digital compression artifacts.

**Technique:** Random colored macroblock artifact placement.

---

## 695 — DimensionRift

Dimension rift—a tear/rift appears in the image showing a warped/inverted version behind it, like a tear in space-time.

**Technique:** Spatial tear with warped/inverted inner reveal.

---

## 696 — DotCrawl

Dot crawl artifact—simulates the NTSC dot-crawl artifact where fine patterns create crawling dot interference along edges.

**Technique:** Edge-following crawling dot pattern overlay.

---

## 697 — DualTone

Dual-tone color grading—maps shadows to one color and highlights to another (duotone), creating a two-color stylized look.

**Technique:** Shadow/highlight dual-color mapping.

---

## 698 — EchoFade

Echo fade—multiple temporal echo layers fade progressively to black, creating a dark-fading trail effect.

**Technique:** Multi-layer fade-to-black echo.

---

## 699 — EdgeMelt

Edge melt—detected edges blur and spread outward over time, as if the edges are melting and bleeding.

**Technique:** Edge-detected blur spreading.

---

## 700 — ElasticWarp

Elastic rubber-sheet warp—pixels are displaced as if the image is printed on a rubber sheet being stretched and pulled with animated anchor points.

**Technique:** Animated multi-point elastic displacement.

---

## 701 — EmberGlow

Glowing embers effect—bright orange/red points pulse and fade like glowing embers scattered across the frame.

**Technique:** Pulsing warm-colored point glow overlay.

---

## 702 — EntropyShift

Entropy-based color shift—regions with high local variance (detail/texture) get color-shifted more than smooth regions.

**Technique:** Local-variance-proportional color shifting.

---

## 703 — ErosionBlend

Morphological erosion blended with original—shrinks bright features to their minimum neighbor value, blended back for a rough-textured look.

**Technique:** Minimum-neighbor erosion blend.

---

## 704 — ExplosionBurst

Explosion burst—pixels fly radially outward from a center point in an animated burst, creating an explosion-scatter effect.

**Technique:** Radial outward pixel scatter burst.

---

## 705 — FacetMirror

Faceted mirror—divides the image into irregularly-shaped facets that each reflect/mirror a slightly different region, like a gem-cut mirror.

**Technique:** Irregular multi-facet region mirror.

---

## 706 — FadeStreak

Fade streak—directional fading streaks trail behind bright objects, creating comet-like tail trails that fade to black.

**Technique:** Directional brightness trailing fade streaks.

---

## 707 — FeatherEdge

Feathered edge vignette—soft, feathered edges gradually blend to black around the frame border with animated feather width.

**Technique:** Animated soft-edge border feathering.

---

## 708 — FlashFreeze

Flash freeze—periodically freezes the entire frame for several frames then resumes, simulating a freeze-frame strobe.

**Technique:** Periodic multi-frame freeze.

---

## 709 — FlipMirror

Animated flip mirror—alternates between horizontal and vertical flipping with blend transitions between orientations.

**Technique:** Alternating H/V flip with blend transition.

---

## 710 — FloatDrift

Floating drift—the entire image gently drifts/floats in X and Y with slow sine movement, creating a floating/bobbing effect.

**Technique:** Slow sine whole-frame position drift.

---

## 711 — FlowField

Flow field visualization—renders pixel displacement guided by a procedural flow field, creating flowing organic distortion paths.

**Technique:** Procedural flow-field pixel displacement.

---

## 712 — FoldWarp

Paper fold warp—simulates folding the image along animated fold lines, with perspective foreshortening on each fold.

**Technique:** Fold-line-based perspective warp.

---

## 713 — FragmentScatter

Fragment scatter—the image is broken into small fragments that scatter outward from their original positions.

**Technique:** Small fragment random scatter displacement.

---

## 714 — FrequencyPulse

Frequency pulse—different spatial frequencies (detail levels) pulse at different rates, creating a shimmering multi-frequency effect.

**Technique:** Multi-frequency brightness pulsing.

---

## 715 — FrostBite

Frost/ice effect—adds a white crystalline frost overlay that spreads from edges inward, with animated frost-creep.

**Technique:** Edge-inward spreading white frost overlay.

---

## 716 — FuseBlend

Fuse blend—blends frames together with a "fuse/ignite" pattern that sweeps across the frame like a burning fuse.

**Technique:** Sweeping fuse-line blend transition.

---

## 717 — GalaxySpiral

Galaxy spiral distortion—pixels are displaced in a logarithmic spiral pattern from center, creating a spiral galaxy arm structure.

**Technique:** Logarithmic spiral displacement from center.

---

## 718 — GelWobble

Gel wobble—the image wobbles as if made of soft gel/jelly, with squishy multi-frequency wobble animation.

**Technique:** Multi-frequency soft wobble displacement.

---

## 719 — GhostEcho

Ghost echo—multiple ghost copies of the image at different positions and opacities, like looking through haunted glass.

**Technique:** Multi-position offset ghost overlay.

---

## 720 — GlassShatter

Glass shatter animation—the image cracks into glass-like shards that separate and fall over time.

**Technique:** Irregular shard crack + separation animation.

---

## 721 — GlimmerPulse

Glimmer pulse—scattered individual pixels glimmer/sparkle with bright flashes, creating a twinkling sparkle overlay.

**Technique:** Random sparse bright pixel sparkle flash.

---

## 722 — GlitchMosaic

Glitched mosaic—mosaic tiles with random glitchy offsets and color corruption within each tile.

**Technique:** Mosaic tiling + per-tile glitch corruption.

---

## 723 — GlowEdge

Glowing edge detection—edges are detected and rendered with a bright, soft glow around them.

**Technique:** Edge detection + soft glow halo.

---

## 724 — GradientMelt

Gradient melt—pixels melt/flow downward at rates proportional to their brightness gradient, creating a waterfall-like flow.

**Technique:** Brightness-gradient-proportional downward flow.

---

## 725 — GrainStorm

Heavy grain storm—intense film grain/noise that storms across the frame much more aggressively than normal film grain.

**Technique:** Heavy per-pixel random noise storm.

---

## 726 — GravityPull

Gravity pull—pixels are displaced downward by an amount proportional to their distance from top, simulating gravitational pull/sag.

**Technique:** Top-distance-proportional downward displacement.

---

## 727 — GridWarp

Grid warp—divides the image into a grid and warps each grid cell independently, creating a patchwork distortion.

**Technique:** Per-grid-cell independent warp.

---

## 728 — HazeLayer

Haze layer overlay—adds a semi-transparent uniform haze/fog layer over the image with animated density.

**Technique:** Animated semi-transparent haze overlay.

---

## 729 — HeatRipple

Heat ripple distortion—stronger sine-wave horizontal displacement concentrated toward the bottom, creating aggressive heat-haze ripples.

**Technique:** Bottom-concentrated horizontal heat ripple.

---

## 730 — HexagonBlur

Hexagonal blur—applies blur in a hexagonal kernel shape rather than circular, creating a distinctive hex-shaped bokeh.

**Technique:** Hexagonal kernel blur.

---

## 731 — HighContrast

Animated high contrast—increases contrast with an animated curve that oscillates between moderate and extreme contrast.

**Technique:** Oscillating contrast curve amplification.

---

## 732 — HologramScan

Hologram scan effect—scrolling horizontal bright bands with blue/cyan tinting, simulating a sci-fi holographic display scanning.

**Technique:** Scrolling cyan-tinted scan band overlay.

---

## 733 — HorizonBend

Horizon bend—bends the horizon line of the image into a curve, creating a fisheye-like horizon warp.

**Technique:** Horizontal horizon-line curve distortion.

---

## 734 — HotSpot

Spotlight hot spot—a bright circular hot spot moves around the frame, brightening the area it covers.

**Technique:** Moving circular brightness spotlight.

---

## 735 — HueWobble

Hue wobble—the hue of the image oscillates/wobbles back and forth around its original value with sine modulation.

**Technique:** Sine-oscillated hue shift wobble.

---

## 736 — glitch_alpha_diamond1

Alpha diamond pattern glitch variant 1—displaces pixels in a diamond-shaped pattern with alpha blending from history frames.

**Technique:** Diamond-pattern displacement + alpha blend.

---

## 737 — glitch_alpha_diamond2

Alpha diamond variant 2—larger diamond blocks with different blending intensity than variant 1.

**Technique:** Larger diamond-pattern alpha blend.

---

## 738 — glitch_alpha_diamond3

Alpha diamond variant 3—animated diamond size with increased displacement.

**Technique:** Animated-size diamond displacement blend.

---

## 739 — glitch_alpha_diamond4

Alpha diamond variant 4—combines diamond pattern with XOR blending for digital artifacts.

**Technique:** Diamond displacement + XOR blend.

---

## 740 — glitch_alpha_diamond5

Alpha diamond variant 5—maximum-intensity diamond pattern with aggressive displacement and heavy blending.

**Technique:** Aggressive diamond displacement + heavy blend.

---

## 741 — glitch_alphatrails

Alpha blended trails from history frames—smooth, translucent motion trails using alpha compositing for clean trailing.

**Technique:** Alpha-composited temporal trailing.

---

## 742 — glitch_color_fade_ex1

Extended color fade glitch 1—colors fade through specific transitions with glitch interruptions and jumps.

**Technique:** Glitch-interrupted color fade transition v1.

---

## 743 — glitch_color_fade_ex2

Extended color fade glitch 2—alternative color fade path with different transition timing and more aggressive interruption.

**Technique:** Glitch-interrupted color fade transition v2.

---

## 744 — glitch_color_shift

Global color channel shifting glitch—shifts all color channels by random offsets simultaneously, creating sudden color jumps.

**Technique:** Random global channel value shift.

---

## 745 — glitch_colorxor01

Color XOR glitch variant 01—XORs pixel colors with an animated value pattern. Each variant in the series (01-20) uses different XOR patterns and timing.

**Technique:** Animated color XOR pattern v01.

---

## 746 — glitch_colorxor02

Color XOR glitch variant 02—different XOR bitmask pattern producing alternate digital color artifacts.

**Technique:** Animated color XOR pattern v02.

---

## 747 — glitch_colorxor03

Color XOR glitch variant 03—XOR with position-dependent masks for spatially-varying artifacts.

**Technique:** Position-dependent color XOR v03.

---

## 748 — glitch_colorxor04

Color XOR glitch variant 04—frame-count-dependent XOR patterns for time-varying color distortion.

**Technique:** Time-varying color XOR v04.

---

## 749 — glitch_colorxor05

Color XOR glitch variant 05—combined position and time XOR for complex evolving patterns.

**Technique:** Position+time combined XOR v05.

---

## 750 — glitch_colorxor06

Color XOR glitch variant 06—per-channel independent XOR masks.

**Technique:** Per-channel independent XOR v06.

---

## 751 — glitch_colorxor07

Color XOR glitch variant 07—XOR with history frame values for temporal XOR artifacts.

**Technique:** History-frame XOR blend v07.

---

## 752 — glitch_colorxor08

Color XOR glitch variant 08—high-frequency XOR pattern creating fine digital noise.

**Technique:** High-frequency XOR noise v08.

---

## 753 — glitch_colorxor09

Color XOR glitch variant 09—low-frequency XOR pattern creating broad color bands.

**Technique:** Low-frequency XOR banding v09.

---

## 754 — glitch_colorxor10

Color XOR glitch variant 10—diagonal XOR pattern creating diagonal digital stripes.

**Technique:** Diagonal XOR stripe pattern v10.

---

## 755 — glitch_colorxor11

Color XOR glitch variant 11—radial XOR pattern from center.

**Technique:** Radial XOR pattern v11.

---

## 756 — glitch_colorxor12

Color XOR glitch variant 12—block-based XOR with tile boundaries.

**Technique:** Block-tiled XOR pattern v12.

---

## 757 — glitch_colorxor13

Color XOR glitch variant 13—row-alternating XOR patterns.

**Technique:** Row-alternating XOR v13.

---

## 758 — glitch_colorxor14

Color XOR glitch variant 14—column-alternating XOR patterns.

**Technique:** Column-alternating XOR v14.

---

## 759 — glitch_colorxor15

Color XOR glitch variant 15—spiral XOR pattern from center.

**Technique:** Spiral XOR pattern v15.

---

## 760 — glitch_colorxor16

Color XOR glitch variant 16—checkerboard XOR pattern.

**Technique:** Checkerboard XOR pattern v16.

---

## 761 — glitch_colorxor17

Color XOR glitch variant 17—frame-difference-based XOR.

**Technique:** Frame-difference XOR v17.

---

## 762 — glitch_colorxor18

Color XOR glitch variant 18—brightness-threshold XOR.

**Technique:** Brightness-threshold XOR v18.

---

## 763 — glitch_colorxor19

Color XOR glitch variant 19—edge-weighted XOR pattern.

**Technique:** Edge-weighted XOR v19.

---

## 764 — glitch_colorxor20

Color XOR glitch variant 20—maximum complexity combined XOR from all previous patterns.

**Technique:** Multi-pattern combined XOR v20.

---

## 765 — glitch_distort_picture

Picture distortion glitch—randomly warps and stretches rectangular regions of the frame, creating data-corruption-style picture distortion.

**Technique:** Random region stretch/warp distortion.

---

## 766 — glitch_echo

Glitch echo—delayed frame echo with glitchy interruptions, creating stuttered temporal echoes with corruption.

**Technique:** Stuttered glitchy frame echo.

---

## 767 — glitch_fast_monocolortrails

Fast monochrome color trails—rapid single-color-channel motion trails that produce fast-moving colored ghost streaks.

**Technique:** Single-channel rapid temporal trailing.

---

## 768 — glitch_frame_pixels

Frame pixel glitch—individual pixels are randomly grabbed from different history frames, creating a per-pixel temporal scatter.

**Technique:** Per-pixel random history-frame sourcing.

---

## 769 — glitch_frame_resize_stretch

Frame resize stretch glitch—the image is stretched/squished to random sizes, creating rubber-band resize distortion.

**Technique:** Random resize/stretch distortion.

---

## 770 — glitch_frame_skip

Frame skip glitch—randomly skips to different history frames, creating a stuttering/jumping playback effect.

**Technique:** Random history-frame skip jump.

---

## 771 — glitch_frame_skip_resize

Frame skip with resize—combines frame skipping with random resizing for a compounded skip+stretch glitch.

**Technique:** Combined frame skip + random resize.

---

## 772 — glitch_frame_skip_shadow

Frame skip with shadow—frame skipping plus a dark ghost of the skipped frames trailing behind.

**Technique:** Frame skip + dark ghost trail.

---

## 773 — glitch_frame_skip_shadow_x2

Frame skip with double shadow—frame skipping with two shadow layers at different opacities for deeper trailing.

**Technique:** Frame skip + double ghost trail.

---

## 774 — glitch_frame_slide01

Frame slide glitch 01—the current frame slides horizontally, revealing history frames beneath at the edge.

**Technique:** Horizontal frame slide reveal v1.

---

## 775 — glitch_frame_slide02

Frame slide glitch 02—vertical sliding variant.

**Technique:** Vertical frame slide reveal v2.

---

## 776 — glitch_frame_slide03

Frame slide glitch 03—diagonal sliding variant.

**Technique:** Diagonal frame slide reveal v3.

---

## 777 — glitch_frame_slide04

Frame slide glitch 04—multi-direction slide with animated direction changes.

**Technique:** Multi-direction animated frame slide v4.

---

## 778 — glitch_frame_square_col

Frame square column glitch—columns of square blocks sourced from different history frames, creating a columnar temporal mosaic.

**Technique:** Column-aligned block history mosaic.

---

## 779 — glitch_frame_stutter_filter

Frame stutter filter—the video stutters/repeats frames with a filtered blend between repeats.

**Technique:** Filtered frame stutter/repeat.

---

## 780 — glitch_glitch_x1

Meta glitch X1—combines multiple glitch techniques (offset, color shift, noise) into one compound glitch effect.

**Technique:** Multi-technique compound glitch.

---

## 781 — glitch_gpt1

GPT-generated glitch 1—AI-designed glitch pattern with unique mathematical distortion formula.

**Technique:** Custom formula glitch v1.

---

## 782 — glitch_gpt2

GPT-generated glitch 2—different AI-designed glitch with alternative mathematical transformation.

**Technique:** Custom formula glitch v2.

---

## 783 — glitch_gpt3

GPT-generated glitch 3—third AI-designed glitch pattern.

**Technique:** Custom formula glitch v3.

---

## 784 — glitch_invertflash

Invert flash glitch—harsh periodic color inversion flash with glitch timing (not regular strobe).

**Technique:** Irregular periodic inversion flash.

---

## 785 — glitch_line_across_side01

Line across side glitch 01—horizontal glitch lines shoot across from one side of the frame with displacement. Variants 01-06 differ in speed, direction, and frequency.

**Technique:** Horizontal shooting glitch line v01.

---

## 786 — glitch_line_across_side02

Line across side variant 02—from opposite direction.

**Technique:** Reverse horizontal shooting glitch line v02.

---

## 787 — glitch_line_across_side03

Line across side variant 03—faster speed.

**Technique:** Fast horizontal glitch line v03.

---

## 788 — glitch_line_across_side04

Line across side variant 04—with color shift.

**Technique:** Color-shifted horizontal glitch line v04.

---

## 789 — glitch_line_across_side05

Line across side variant 05—multiple simultaneous lines.

**Technique:** Multi-line horizontal glitch v05.

---

## 790 — glitch_line_across_side06

Line across side variant 06—with displacement wake.

**Technique:** Displacement-wake horizontal glitch line v06.

---

## 791 — glitch_line_collection01

Line collection glitch 01—draws collections of displaced horizontal lines at various positions. Variants 01-07 use different arrangements and densities.

**Technique:** Displaced horizontal line collection v01.

---

## 792 — glitch_line_collection02

Line collection variant 02—denser line packing.

**Technique:** Dense horizontal line collection v02.

---

## 793 — glitch_line_collection03

Line collection variant 03—variable-width lines.

**Technique:** Variable-width line collection v03.

---

## 794 — glitch_line_collection04

Line collection variant 04—with color tinting per line.

**Technique:** Color-tinted line collection v04.

---

## 795 — glitch_line_collection05

Line collection variant 05—animated line positions.

**Technique:** Animated position line collection v05.

---

## 796 — glitch_line_collection06

Line collection variant 06—crossing/intersecting lines.

**Technique:** Crossing line collection v06.

---

## 797 — glitch_line_collection07

Line collection variant 07—maximum density combined.

**Technique:** Maximum density line collection v07.

---

## 798 — glitch_line_offset_inout

Line offset in/out—horizontal lines grow outward from center then retract inward, oscillating in/out.

**Technique:** Center-expanding/contracting line offset v1.

---

## 799 — glitch_line_offset_inout2

Line offset in/out variant 2—faster oscillation with more lines.

**Technique:** Fast expanding/contracting line offset v2.

---

## 800 — glitch_line_offset_inout3

Line offset in/out variant 3—asymmetric expansion with trailing displacement.

**Technique:** Asymmetric expanding line offset v3.

---

## 801 — glitch_newrandblend

New random blend—randomly blends rectangular patches from different history frames, creating a patchwork of time points.

**Technique:** Random rectangular patch temporal blend.

---

## 802 — glitch_newstretchlines

New stretch lines—horizontal lines are randomly stretched or compressed, creating rubber-band-line distortion.

**Technique:** Random horizontal line stretch/compress.

---

## 803 — glitch_newvarlines

New variable lines—draws lines of varying thickness, color, and offset across the frame in glitchy patterns.

**Technique:** Variable-parameter random line drawing.

---

## 804 — glitch_outoforder

Out of order—rows or blocks of the frame are shuffled into a random order, scrambling the vertical arrangement of the image.

**Technique:** Random row/block order shuffling.

---

## 805 — glitch_pic_adjust1

Picture adjustment glitch 1—randomly adjusts brightness, contrast, and color balance per frame, creating fluctuating exposure.

**Technique:** Random per-frame exposure adjustment v1.

---

## 806 — glitch_pic_adjust2

Picture adjustment glitch 2—more aggressive version with wider adjustment ranges.

**Technique:** Aggressive random exposure adjustment v2.

---

## 807 — glitch_picture_jump01

Picture jump glitch 01—the entire image jumps/teleports to a random position offset for a few frames then returns. Variants 01-31 use different jump patterns, distances, and timing.

**Technique:** Random position jump teleport v01.

---

## 808 — glitch_picture_jump02

Picture jump variant 02—horizontal-only jumps.

**Technique:** Horizontal position jump v02.

---

## 809 — glitch_picture_jump03

Picture jump variant 03—vertical-only jumps.

**Technique:** Vertical position jump v03.

---

## 810 — glitch_picture_jump04

Picture jump variant 04—diagonal jumps.

**Technique:** Diagonal position jump v04.

---

## 811 — glitch_picture_jump05

Picture jump variant 05—small rapid micro-jumps.

**Technique:** Rapid micro position jump v05.

---

## 812 — glitch_picture_jump06

Picture jump variant 06—large infrequent jumps.

**Technique:** Infrequent large position jump v06.

---

## 813 — glitch_picture_jump07

Picture jump variant 07—smooth slide transition between jumps.

**Technique:** Sliding position jump transition v07.

---

## 814 — glitch_picture_jump08

Picture jump variant 08—with zoom during jump.

**Technique:** Position jump + zoom v08.

---

## 815 — glitch_picture_jump09

Picture jump variant 09—with rotation during jump.

**Technique:** Position jump + rotation v09.

---

## 816 — glitch_picture_jump10

Picture jump variant 10—with color shift during jump.

**Technique:** Position jump + color shift v10.

---

## 817 — glitch_picture_jump11

Picture jump variant 11—bouncing jump with deceleration.

**Technique:** Bouncing deceleration jump v11.

---

## 818 — glitch_picture_jump12

Picture jump variant 12—elastic snap-back after jump.

**Technique:** Elastic snap-back jump v12.

---

## 819 — glitch_picture_jump13

Picture jump variant 13—spiral jump path.

**Technique:** Spiral path position jump v13.

---

## 820 — glitch_picture_jump14

Picture jump variant 14—multi-step sequential jumps.

**Technique:** Sequential multi-step jump v14.

---

## 821 — glitch_picture_jump15

Picture jump variant 15—X channel jumps separately from Y.

**Technique:** Independent X/Y channel jump v15.

---

## 822 — glitch_picture_jump16

Picture jump variant 16—with ghost trail during jump.

**Technique:** Ghost-trailed position jump v16.

---

## 823 — glitch_picture_jump17

Picture jump variant 17—frame-skip combined with jump.

**Technique:** Frame-skip + position jump v17.

---

## 824 — glitch_picture_jump18

Picture jump variant 18—jump with XOR artifact.

**Technique:** XOR artifact position jump v18.

---

## 825 — glitch_picture_jump19

Picture jump variant 19—sinusoidal jump path.

**Technique:** Sine-wave position jump path v19.

---

## 826 — glitch_picture_jump20

Picture jump variant 20—random walk jump sequence.

**Technique:** Random walk position jump v20.

---

## 827 — glitch_picture_jump21

Picture jump variant 21—accelerating jumps.

**Technique:** Accelerating position jump v21.

---

## 828 — glitch_picture_jump22

Picture jump variant 22—decelerating jumps.

**Technique:** Decelerating position jump v22.

---

## 829 — glitch_picture_jump23

Picture jump variant 23—pulsating amplitude jumps.

**Technique:** Pulse-amplitude position jump v23.

---

## 830 — glitch_picture_jump24

Picture jump variant 24—corner-biased jumps.

**Technique:** Corner-targeted position jump v24.

---

## 831 — glitch_picture_jump25

Picture jump variant 25—edge-biased jumps.

**Technique:** Edge-targeted position jump v25.

---

## 832 — glitch_picture_jump26

Picture jump variant 26—center-avoidant jumps.

**Technique:** Center-avoidant position jump v26.

---

## 833 — glitch_picture_jump27

Picture jump variant 27—mirrored double-jump.

**Technique:** Mirrored double position jump v27.

---

## 834 — glitch_picture_jump28

Picture jump variant 28—with scale distortion.

**Technique:** Scale-distorted position jump v28.

---

## 835 — glitch_picture_jump29

Picture jump variant 29—with wrap-around.

**Technique:** Wrap-around position jump v29.

---

## 836 — glitch_picture_jump30

Picture jump variant 30—with partial visibility.

**Technique:** Partial-visibility position jump v30.

---

## 837 — glitch_picture_jump31

Picture jump variant 31—maximum combined jump effects.

**Technique:** Maximum-complexity combined jump v31.

---

## 838 — glitch_rect_size

Rectangle size glitch—randomly-sized rectangles from the image are resized (scaled up/down) and placed back, creating scaled-block artifacts.

**Technique:** Random rectangle resize-in-place.

---

## 839 — glitch_rsquare2

Random square glitch 2—randomly-positioned square blocks are filled or displaced, creating square artifact blocks.

**Technique:** Random square block displacement v2.

---

## 840 — glitch_rsquare3

Random square glitch 3—larger blocks with color tinting.

**Technique:** Large tinted square block displacement v3.

---

## 841 — glitch_rsquare4

Random square glitch 4—with XOR color treatment.

**Technique:** XOR-colored square blocks v4.

---

## 842 — glitch_rsquare5

Random square glitch 5—overlapping squares with blend.

**Technique:** Overlapping blended square blocks v5.

---

## 843 — glitch_slice_frame

Frame slice glitch—slices the frame into horizontal strips and offsets them randomly, creating a horizontal-slice glitch.

**Technique:** Random horizontal frame slicing.

---

## 844 — glitch_slice_frame_w

Frame slice width variant—vertical slicing variant that cuts the frame into vertical strips with random offsets.

**Technique:** Random vertical frame slicing.

---

## 845 — glitch_square_block_v2a

Square block glitch v2a—square-tiled block manipulation variant A. The v2 series (a-h) each apply different per-block transformations.

**Technique:** Per-block transformation variant A.

---

## 846 — glitch_square_block_v2b

Square block variant B—per-block color inversion.

**Technique:** Per-block color inversion variant B.

---

## 847 — glitch_square_block_v2c

Square block variant C—per-block brightness shift.

**Technique:** Per-block brightness shift variant C.

---

## 848 — glitch_square_block_v2d

Square block variant D—per-block horizontal flip.

**Technique:** Per-block horizontal mirror variant D.

---

## 849 — glitch_square_block_v2e

Square block variant E—per-block temporal swap (from history).

**Technique:** Per-block history swap variant E.

---

## 850 — glitch_square_block_v2f

Square block variant F—per-block XOR with neighbor block.

**Technique:** Per-block neighbor XOR variant F.

---

## 851 — glitch_square_block_v2g

Square block variant G—per-block channel rotation.

**Technique:** Per-block channel rotation variant G.

---

## 852 — glitch_square_block_v2h

Square block variant H—combined all block transformations randomly per block.

**Technique:** Random mixed per-block transforms variant H.

---

## 853 — glitch_square_xor

Square XOR glitch—XORs square-shaped regions with their background, creating sharp-edged XOR artifact squares.

**Technique:** Square-region XOR.

---

## 854 — glitch_store_frame10

Store frame 10 glitch—stores every 10th frame and replays it unexpectedly, creating periodic stuck-frame flashes.

**Technique:** Periodic stored-frame playback.

---

## 855 — glitch_stuckframe

Stuck frame glitch—the frame gets "stuck" for a random number of frames before updating, simulating frozen/stuck playback.

**Technique:** Random-duration frame freeze.

---

## 856 — glitch_stutter_long

Long stutter glitch—extends the stutter effect over many frames, creating a long, droning repetition of a single frame.

**Technique:** Extended duration frame stutter.

---

## 857 — glitch_stutter_sbrv

Stutter with sub-reverse—stutters frames with occasional short reverse playback segments, creating a glitchy forward-reverse stutter.

**Technique:** Stutter + short reverse playback.

---

## 858 — acgl_glitch_AlphaBlendFive

AC Glitch Library: blends 5 history frames with decreasing alpha weights for a deep, smooth multi-frame ghosting.

**Technique:** 5-frame decreasing-alpha blend.

---

## 859 — acgl_glitch_AlphaBlendTri

AC Glitch Library: blends 3 history frames (triangle pattern) with alternating alpha weights.

**Technique:** 3-frame triangular alpha blend.

---

## 860 — acgl_glitch_AlphaBlendExpand

AC Glitch Library: alpha blend with an expanding region—the blended area grows outward from center over time.

**Technique:** Center-expanding alpha blend region.

---

## 861 — acgl_glitch_BarsCol

AC Glitch Library: vertical color bars—columns of the image are replaced with solid color bars at regular intervals.

**Technique:** Regular vertical solid color bars.

---

## 862 — acgl_glitch_BarsColAlpha

AC Glitch Library: semi-transparent vertical color bars blended with the underlying image.

**Technique:** Alpha-blended vertical color bars.

---

## 863 — acgl_glitch_BarsHoriz

AC Glitch Library: horizontal color bars—rows are replaced with solid color bars.

**Technique:** Regular horizontal solid color bars.

---

## 864 — acgl_glitch_BarsHorizAlpha

AC Glitch Library: semi-transparent horizontal color bars blended with the image.

**Technique:** Alpha-blended horizontal color bars.

---

## 865 — acgl_glitch_BlackSquare

AC Glitch Library: random black squares appear across the frame, blocking out portions of the image.

**Technique:** Random black square block overlay.

---

## 866 — acgl_glitch_BlendStuck

AC Glitch Library: blends with a stuck/frozen frame—a single captured frame is continuously blended with the live feed.

**Technique:** Stuck-frame continuous blend.

---

## 867 — acgl_glitch_ColorDistort

AC Glitch Library: per-channel color distortion—each RGB channel is independently warped/shifted for aggressive color corruption.

**Technique:** Independent per-channel color distortion.

---

## 868 — acgl_glitch_ColorRect

AC Glitch Library: colored rectangles drawn at random positions and sizes, overlaid on the image.

**Technique:** Random colored rectangle overlay.

---

## 869 — acgl_glitch_ColorOnOff

AC Glitch Library: rapid color/colorless switching—alternates between full color and desaturated/grayscale on each frame or few frames.

**Technique:** Rapid color/grayscale toggle.

---

## 870 — acgl_glitch_DEM

AC Glitch Library: DEM (digital elevation map) style visualization—maps brightness to a false-color elevation palette.

**Technique:** Brightness-to-elevation false color mapping.

---

## 871 — acgl_glitch_FrameMirror

AC Glitch Library: mirrors entire frames from history—alternates between the current frame and various mirrored history frames.

**Technique:** History-frame mirroring alternation.

---

## 872 — acgl_glitch_FramePix

AC Glitch Library: pixelates using frame history—mosaics the image using large blocks sourced from different time points.

**Technique:** History-frame-sourced block mosaic.

---

## 873 — acgl_glitch_FrameReverse

AC Glitch Library: reverse frame playback—periodically plays history frames in reverse order, creating backward motion sequences.

**Technique:** Periodic reverse frame playback.

---

## 874 — acgl_glitch_FrameReverse2

AC Glitch Library: reverse frame playback variant 2—longer reverse sequences with blending between forward and reverse.

**Technique:** Extended reverse playback + blend.

---

## 875 — acgl_glitch_FrameReverseNoBlend

AC Glitch Library: hard-cut reverse playback without blending—abrupt switches between forward and reverse for a jarring stutter.

**Technique:** Hard-cut reverse playback without blend.

---

## 876 — acgl_glitch_FrameSepBand

AC Glitch Library: frame separation bands—horizontal bands alternate between current and different history frames, creating temporal banding.

**Technique:** Horizontal temporal separation bands.

---

## 877 — acgl_glitch_FrameSwap

AC Glitch Library: frame swap—randomly swaps the current frame with a history frame for a few frames, creating time-jump flashes.

**Technique:** Random history-frame swap.

---

## 878 — acgl_glitch_FrameXBlend

AC Glitch Library: horizontal frame blend—blends current frame with history using a horizontal gradient mask (more history on one side).

**Technique:** Horizontal gradient temporal blend.

---

## 879 — acgl_glitch_FrameXBlendXor

AC Glitch Library: horizontal frame XOR blend—combines horizontal gradient blending with XOR for digital artifacts along the gradient.

**Technique:** Horizontal gradient XOR blend.

---

## 880 — acgl_glitch_FrameYBlend

AC Glitch Library: vertical frame blend—blends current with history using a vertical gradient mask.

**Technique:** Vertical gradient temporal blend.

---

## 881 — acgl_glitch_FrameYBlendXor

AC Glitch Library: vertical frame XOR blend—vertical gradient with XOR artifacts.

**Technique:** Vertical gradient XOR blend.

---

## 882 — acgl_glitch_AddMulXor

AC Glitch Library: combined add, multiply, and XOR operations between current and history frames—triple-operation digital manipulation.

**Technique:** Add + multiply + XOR triple blend.

---

## 883 — acgl_glitch_AddXor

AC Glitch Library: addition followed by XOR between frames—adds then XORs for layered digital artifacts.

**Technique:** Add then XOR dual operation.

---

## 884 — acgl_glitch_ColorBarsRand

AC Glitch Library: random color bars—color bar positions, widths, and colors randomize each cycle, creating chaotic bar patterns.

**Technique:** Randomized color bar pattern.

---

## 885 — acgl_glitch_ColorBarsX

AC Glitch Library: vertical color bars with animated scrolling—bars scroll horizontally across the frame.

**Technique:** Scrolling vertical color bars.

---

## 886 — acgl_glitch_ColorBarsY

AC Glitch Library: horizontal color bars with animated scrolling—bars scroll vertically.

**Technique:** Scrolling horizontal color bars.

---

## 887 — acgl_glitch_ColorShiftY

AC Glitch Library: vertical color shifting—color channels shift vertically at different rates, creating vertical chromatic separation.

**Technique:** Vertical per-channel offset shift.

---

## 888 — acgl_glitch_LineCollectionRGB

AC Glitch Library: RGB line collection—draws horizontal lines where each line is a single color channel (red, green, or blue) from the image.

**Technique:** Single-channel horizontal line extraction.

---

## 889 — acgl_glitch_FrameX2

AC Glitch Library: double-frame blend—blends two history frames at once for a deeper temporal composite.

**Technique:** Dual history frame blend.

---

## 890 — acgl_glitch_NewBars

AC Glitch Library: new-style color bars—redesigned bar pattern with modern width and color choices.

**Technique:** Modern styled color bar pattern.

---

## 891 — acgl_glitch_NewBars2

AC Glitch Library: new bars variant 2—alternative bar layout with different spacing and color assignment.

**Technique:** Alternative modern bar layout.

---

## 892 — acgl_glitch_NewBlendLines

AC Glitch Library: new blend lines—horizontal lines sourced from blended history frames, creating temporal line composites.

**Technique:** History-blended horizontal line rendering.

---

## 893 — acgl_glitch_NewLines

AC Glitch Library: new lines—horizontal displacement lines with updated generation algorithm for cleaner glitch lines.

**Technique:** Updated algorithm horizontal glitch lines.

---

## 894 — acgl_glitch_NewOne

AC Glitch Library: new experimental glitch—a novel combined glitch technique unique to this library.

**Technique:** Experimental combined glitch.

---

## 895 — acgl_glitch_StrobeCycle

AC Glitch Library: strobe cycle—cycles through multiple strobe patterns (white, black, color, invert) in a repeating sequence.

**Technique:** Multi-pattern strobe cycling.

---

## 896 — acgl_glitch_StuckFrame2

AC Glitch Library: stuck frame variant 2—holds a frame longer with gradual blend back to live, smoother than the original stuck frame.

**Technique:** Extended stuck frame with gradual resume.

---

## 897 — acgl_glitch_StuckLine

AC Glitch Library: stuck line—individual horizontal lines get stuck (frozen) at random intervals while the rest of the image updates.

**Technique:** Random per-line freeze.

---

## 898 — acgl_glitch_StuckRow

AC Glitch Library: stuck row—entire rows of pixels freeze for varying durations, creating a row-by-row temporal desync.

**Technique:** Per-row variable-duration freeze.

---

## 899 — acgl_glitch_StuckRowLine

AC Glitch Library: stuck row + line combination—both individual lines and row groups freeze independently.

**Technique:** Combined line + row freeze.

---

## 900 — acgl_glitch_SepBlocks

AC Glitch Library: separated blocks—block tiles separate with visible gaps between them, creating a tiled grid with spacing.

**Technique:** Block tile separation with gap.

---

## 901 — acgl_glitch_Plug1

AC Glitch Library: plugin slot 1—a general-purpose glitch filter template used as an extensible plugin point.

**Technique:** General-purpose glitch plugin.

---

## 902 — acgl_glitch_OppositeDir

AC Glitch Library: opposite direction—alternating rows/blocks move in opposite directions, creating a push-pull shearing effect.

**Technique:** Alternating opposite-direction row displacement.

---

## 903 — acgl_glitch_OffStuck

AC Glitch Library: offset stuck—combines spatial offset with stuck-frame, showing a frozen frame at a spatial offset position.

**Technique:** Spatially-offset stuck frame.

---

## 904 — acgl_glitch_NewVarBlendLines

AC Glitch Library: new variable blend lines—horizontal lines blended from various history frames with variable blend ratios and widths.

**Technique:** Variable-ratio history-blended horizontal lines.











