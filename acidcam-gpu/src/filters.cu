#include "ac-gpu/ac-gpu.hpp"
#include <cuda_runtime.h>
#include <string>
namespace ac_gpu {
    Filter filters[] = {
        { 0, "SelfAlphaBlend" },
        { 1, "MedianBlend" },
        { 2, "MedianBlendXor" },
        { 3, "SquareBlockResize" },
        { 4, "SelfScaleRefined" },
        { 5, "StrangeGlitch" },
        { 6, "MatrixOutline" },
        { 7, "AuraTrails" },
        { 8, "MirrorReverseColor" },
        { 9, "SquareShrink" },
        { 10, "MotionGhostTrails" },
        { 11, "StretchColMatrix8" },
        { 12, "StretchColMatrix16" },
        { 13, "StretchColMatrix32" },
        { 14, "GradientFlashColor" },
        { 15, "HorizontalGlitch" },
        { 16, "VerticalGlitch" },
        { 17, "WaveTrails" },
        { 18, "PixelInterlace" },
        { 19, "ColorWaveTrails" },
        { 20, "ParticleSlide" },
        { 21, "DiagPixelated" },
        { 22, "DiagPixelatedResize" },
        { 23, "RGBShiftTrails" },
        { 24, "PictureShiftDown" },
        { 25, "PictureShiftRight" },
        { 26, "PictureShiftVariable" },
        { 27, "StretchR_Right" },
        { 28, "StretchG_Right" },
        { 29, "StretchB_Right" },
        { 30, "StretchR_Down" },
        { 31, "StretchG_Down" },
        { 32, "StretchB_Down" },
        { 33, "Distorted_LinesY" },
        { 34, "Distorted_LinesX" },
        { 35, "TripHSV" },
        { 36, "XorSumStrobe" },
        { 37, "DetectEdges" },
        { 38, "SobelNorm" },
        { 39, "SobelThreshold" },
        { 40, "LineInLineOut" },
        { 41, "LineInLineOut4_Increase" },
        { 42, "LineInLineOut_ReverseIncrease" },
        { 43, "LineInLineOut_InvertedY" },
        { 44, "LineInLineOut_ReverseInvertedY" },
        { 45, "LineInLineOut_Vertical" },
        { 46, "LineInLineOut_VerticalIncrease" },
        { 47, "LineInLineOut_IncreaseImage" },
        { 48, "SquareByRow" },
        { 49, "SquareByRowRev" },
        { 50, "SquareByRow2" },
        { 51, "DivideByValue" },
        { 52, "ColorCollectionSubtleStrobe" },
        { 53, "CollectionRandom" },
        { 54, "CollectionAlphaXor" },
        { 55, "ColorCollection64X" },
        { 56, "ColorCollectionSwitch" },
        { 57, "ColorCollectionRGB_Index" },
        { 58, "ColorCollectionGhostTrails" },
        { 59, "ColorCollectionScale" },
        { 60, "ColorCollectionReverseStrobe" },
        { 61, "ColorCollectionXorPixel" },
        { 62, "BlendWithSource25" },
        { 63, "BlendWithSource50" },
        { 64, "BlendWithSource75" },
        { 65, "BlendWithSource100" },
        { 66, "ColorCollectionXorOffsetFlash" },
        { 67, "ColorCollectionMatrixGhost" },
        { 68, "MildStrobe" },
        { 69, "ReduceBy50" },
        { 70, "ColorPositionAverageXor" },
        { 71, "ColorPositionXor" },
        { 72, "GrayStrobe" },
        { 73, "ColorStrobeXor" },
        { 74, "ColorGhost" },
        { 75, "BlurredOutXor" },
        { 76, "DizzyFilter" },
        { 77, "Buzzed" },
        { 78, "BuzzedDark" },
        { 79, "AllRed" },
        { 80, "AllGreen" },
        { 81, "AllBlue" },
        { 82, "NegativeStrobe" },
        { 83, "XorAddMul" },
        { 84, "HorizontalLines" },
        { 85, "StrobeRedGreenBlue" },
        { 86, "Pulse" },
        { 87, "DiamondPattern" },
        { 88, "Bitwise_XOR" },
        { 89, "Bitwise_AND" },
        { 90, "Bitwise_OR" },
        { 91, "BlendSwitch" },
        { 92, "LineRGB" },
        { 93, "PixelRGB" },
        { 94, "InvertedScanlines" },
        { 95, "ScanSwitch" },
        { 96, "ScanAlphaSwitch" },
        { 97, "RGBFlash" },
        { 98, "DiagonalLines" },
        { 99, "Darken" },
        { 100, "SelfXorBlend" },
        { 101, "SelfXorDoubleFlash" },
        { 102, "SelfOrDoubleFlash" },
        { 103, "BlendRowCurvedSqrt" },
        { 104, "XorAlpha" },
        { 105, "RandomXorBlend" },
        { 106, "AndStrobe" },
        { 107, "AndStrobeScale" },
        { 108, "AndPixelStrobe" },
        { 109, "AndOrXorStrobe" },
        { 110, "FadeInAndOut" },
        { 111, "BrightStrobe" },
        { 112, "DarkStrobe" },
        { 113, "RandomXorOpposite" },
        { 114, "GradientRainbow" },
        { 115, "cossinMultiply" },
        { 116, "colorAccumulate1" },
        { 117, "colorAccumulate2" },
        { 118, "WeakBlend" },
        { 119, "StrobeEffect" },
        { 120, "Blend3" },
        { 121, "NegParadox" },
        { 122, "ThoughtMode" },
        { 123, "Tri" },
        { 124, "Distort" },
        { 125, "colorAccumulate3" },
        { 126, "filter8" },
        { 127, "filter3" },
        { 128, "rainbowBlend" },
        { 129, "pixelScale" },
        { 130, "GradientSelf" },
        { 131, "GradientSelfVertical" },
        { 132, "GradientDown" },
        { 133, "GraidentHorizontal" },
        { 134, "Inter" },
        { 135, "BlendedScanLines" },
        { 136, "GradientStripes" },
        { 137, "XorSine" },
        { 138, "Circular" },
        { 139, "RandomPixels" },
        { 140, "DarkRandomPixels" },
        { 141, "Bars" },
        { 142, "NegativeByRow" },
        { 143, "XorScale" },
        { 144, "SelfAlphaRGB" },
        { 145, "BitwiseXorStrobe" },
        { 146, "OrStrobe" },
        { 147, "DivideAndIncH" },
        { 148, "DivideAndIncW" },
        { 149, "RandomIncrease" },
        { 150, "SelfAlphaScaleBlend" },
        { 151, "FadeBars" },
        { 152, "StrobeXor" },
        { 153, "Blank" },
        { 154, "ColorVariableBlend" },
        { 155, "ColorXorBlend" },
        { 156, "ColorAddBlend" },
        { 157, "SurroundingPixels" },
        { 158, "SurroundingPixelsAlpha" },
        { 159, "DarkModBlend" },
        { 160, "IncreaseDecreaseGamma" },
        { 161, "BlendChannelXor" },
        { 162, "IncDifference" },
        { 163, "IncDifferenceAlpha" },
        { 164, "MirrorXorAlpha" },
        { 165, "IntertwinedMirror" },
        { 166, "ColorFadeFilter" },
        { 167, "ColorChannelMoveUpAndDown" },
        { 168, "MedianStrobe" },
        { 169, "RGBBlend" },
        { 170, "BGRBlend" },
        { 171, "FlipAlphaBlend" },
        { 172, "RandomFlipFilter" },
        { 173, "SelfScaleByFrame" },
        { 174, "AlphaBlendMirror" },
        { 175, "TwistedVision" },
        { 176, "TruncateColor" },
        { 177, "TruncateVariable" },
        { 178, "TruncateVariableScale" },
        { 179, "XorFade" },
        { 180, "SineValue" },
        { 181, "FadeRtoGtoB" },
        { 182, "FadeRandomChannel" },
        { 183, "VariableLines" },
        { 184, "VariableLinesVertical" },
        { 185, "RowMedianBlend" },
        { 186, "MirrorReverseColorBlend" },
        { 187, "PsychoticVision" },
        { 188, "PixelGlitch" },
        { 189, "StaticGlitch" },
        { 190, "WavePattern" },
        { 191, "WavePatternXor" },
        { 192, "DiagonalXor" },
        { 193, "RGBShiftBlend" },
        { 194, "ChannelShuffle" },
        { 195, "ChannelShuffleRand" },
        { 196, "PixelCounter" },
        { 197, "PixelCounterXor" },
        { 198, "RowColorBlend" },
        { 199, "ColumnColorBlend" },
        { 200, "CheckerboardXor" },
        { 201, "CheckerboardBlend" },
        { 202, "SineWaveDistort" },
        { 203, "CosineWaveDistort" },
        { 204, "SinCosBlend" },
        { 205, "PixelReverseXor" },
        { 206, "LinesAcrossX" },
        { 207, "XorLineX" },
        { 208, "AlphaComponentIncrease" },
        { 209, "ExpandContract" },
        { 210, "LongLines" },
        { 211, "TearRight" },
        { 212, "TearDown" },
        { 213, "DistortionByRow" },
        { 214, "DistortionByCol" },
        { 215, "AlternateAlpha" },
        { 216, "DiagSquareRGB" },
        { 217, "ShiftPixelsRGB" },
        { 218, "ColorWaveTrailsRGB" },
        { 219, "ProperTrails" },
        { 220, "XorLag" },
        { 221, "PixelateBlend" },
        { 222, "DiagPixel" },
        { 223, "DiagPixelY" },
        { 224, "ExpandLeftRight" },
        { 225, "DiagSquare" },
        { 226, "HorizontalColorOffset" },
        { 227, "PrevFrameNotEqual" },
        { 228, "BlackLines" },
        { 229, "DizzyMode" },
        { 230, "GhostShift" },
        { 231, "RGBSplitFilter" },
        { 232, "PixelateRect" },
        { 233, "CollectionXor4" },
        { 234, "RectangleSpin" },
        { 235, "RectanglePlotXY" },
        { 236, "ShiftLinesDown" },
        { 237, "PictureStretch" },
        { 238, "PictureStretchPieces" },
        { 239, "VisualSnow" },
        { 240, "VisualSnowX2" },
        { 241, "LineGlitch" },
        { 242, "SlitReverse64" },
        { 243, "SlitReverse64_Increase" },
        { 244, "SlitStretch" },
        { 245, "LineLeftRight" },
        { 246, "LineLeftRightResize" },
        { 247, "RGBLineTrails" },
        { 248, "RGBCollectionBlend" },
        { 249, "RGBCollectionIncrease" },
        { 250, "RGBLongTrails" },
        { 251, "FadeRGB_Speed" },
        { 252, "RGBStrobeTrails" },
        { 253, "BoxGlitch" },
        { 254, "VerticalPictureDistort" },
        { 255, "ShortTrail" },
        { 256, "DiagInward" },
        { 257, "DiagSquareInward" },
        { 258, "DiagSquareInwardResize" },
        { 259, "PictureShiftDownRight" },
        { 260, "FlipPictureShift" },
        { 261, "RGBWideTrails" },
        { 262, "LineInLineOut_Increase" },
        { 263, "LineInLineOut2_Increase" },
        { 264, "LineInLineOut3_Increase" },
        { 265, "SquareByRow2Plus" },
        { 266, "FrameSep" },
        { 267, "FrameSep2" },
        { 268, "FrameStopStart" },
        { 269, "OutOfOrder" },
        { 270, "TrackingDown" },
        { 271, "TrackingDownBlend" },
        { 272, "TrackingRev" },
        { 273, "TrackingMirror" },
        { 274, "BlockPixels" },
        { 275, "FrameChop" },
        { 276, "YLineDown" },
        { 277, "YLineDownBlend" },
        { 278, "SquareDiff1" },
        { 279, "LineAcrossX" },
        { 280, "ColorGlitch" },
        { 281, "PixelShiftUp" },
        { 282, "PixelShiftDown" },
        { 283, "PixelShiftLeft" },
        { 284, "PixelShiftRight" },
        { 285, "PixelShiftDiagonal" },
        { 286, "WaveBlend" },
        { 287, "WaveBlendX2" },
        { 288, "SineWaveBlend" },
        { 289, "CosineWaveBlend" },
        { 290, "SpiralWave" },
        { 291, "RadialBlur" },
        { 292, "ZoomBlur" },
        { 293, "RotateBlend" },
        { 294, "MirrorWave" },
        { 295, "MirrorWaveX" },
        { 296, "MirrorWaveY" },
        { 297, "PixelDrift" },
        { 298, "PixelDriftX" },
        { 299, "PixelDriftY" },
        { 300, "ColorPulse" },
        { 301, "ColorPulseRGB" },
        { 302, "ColorPulseXor" },
        { 303, "GlitchBlock" },
        { 304, "GlitchBlockXor" },
        { 305, "GlitchLine" },
        { 306, "GlitchLineX" },
        { 307, "NoiseBlend" },
        { 308, "NoiseBlendX2" },
        { 309, "NoiseXor" },
        { 310, "ChannelShift" },
        { 311, "ChannelShiftX" },
        { 312, "ChannelRotate" },
        { 313, "DiagonalStretch" },
        { 314, "DiagonalStretchX" },
        { 315, "DiagonalMirror" },
        { 316, "SquareWave" },
        { 317, "SquareWaveX" },
        { 318, "SquareWaveBlend" },
        { 319, "TriangleWave" },
        { 320, "TriangleWaveBlend" },
        { 321, "SawtoothWave" },
        { 322, "SawtoothWaveBlend" },
        { 323, "PulseWave" },
        { 324, "PulseWaveBlend" },
        { 325, "StepWave" },
        { 326, "StepWaveBlend" },
        { 327, "RippleEffect" },
        { 328, "RippleEffectX2" },
        { 329, "ShockWave" },
        { 330, "ShockWaveBlend" },
        { 331, "TwistEffect" },
        { 332, "TwistEffectBlend" },
        { 333, "FishEye" },
        { 334, "FishEyeBlend" },
        { 335, "Kaleidoscope" },
        { 336, "KaleidoscopeBlend" },
        { 337, "TunnelEffect" },
        { 338, "TunnelEffectBlend" },
        { 339, "VortexEffect" },
        { 340, "VortexEffectBlend" },
        { 341, "ColorDrift" },
        { 342, "ColorDriftX" },
        { 343, "RGBShift" },
        { 344, "RGBShiftX" },
        { 345, "ChromaticAberration" },
        { 346, "ChromaticAberrationX" },
        { 347, "Posterize" },
        { 348, "PosterizeBlend" },
        { 349, "Solarize" },
        { 350, "SolarizeBlend" },
        { 351, "GammaBright" },
        { 352, "GammaDark" },
        { 353, "ContrastBoost" },
        { 354, "ContrastReduce" },
        { 355, "EdgeGlowBlend" },
        { 356, "FrameBlendMulti" },
        { 357, "FrameBlendMultiX" },
        { 358, "AcidTrailsBlend" },
        { 359, "AcidGlitchX" },
        { 360, "AlphaXorBlend" },
        { 361, "AlphaXorBlendDouble" },
        { 362, "AndOrXorStrobeScale" },
        { 363, "AveragePixelsXorBlend" },
        { 364, "BitwiseRotateBlend" },
        { 365, "BitwiseRotateDiffBlend" },
        { 366, "BitwiseXorScaleBlend" },
        { 367, "BlackAndWhiteStrobe" },
        { 368, "BlendAlphaXorScale" },
        { 369, "BlendBurredXor" },
        { 370, "BlendCombinedXor" },
        { 371, "BlendIncreaseRGB" },
        { 372, "BlendThreeXor" },
        { 373, "BlurDistortionBlend" },
        { 374, "ColorAccumulate" },
        { 375, "ColorAccumulateBlend" },
        { 376, "ColorAccumulateXor" },
        { 377, "ColorChannelBlend" },
        { 378, "ColorChannelXor" },
        { 379, "ColorCollectionEnergy" },
        { 380, "ColorCollectionWave" },
        { 381, "ColorFadeXor" },
        { 382, "ColorIntensityBlend" },
        { 383, "ColorIntensityXor" },
        { 384, "ColorMoveBlend" },
        { 385, "ColorPixelBlend" },
        { 386, "ColorPixelXor" },
        { 387, "ColorScaleBlend" },
        { 388, "ColorWaveXor" },
        { 389, "CosineMultiplyBlend" },
        { 390, "DarkModBlendXor" },
        { 391, "DifferenceBlend" },
        { 392, "DifferenceXorBlend" },
        { 393, "DistortBlend" },
        { 394, "DiamondPatternBlend" },
        { 395, "FadeBlendXor" },
        { 396, "FlashBlendXor" },
        { 397, "GhostTrailsBlend" },
        { 398, "AddInvert" },
        { 399, "AlphaBlendSimple" },
        { 400, "AlphaBlendDoubleX" },
        { 401, "AlphaStrobeBlendX" },
        { 402, "BitwiseAndBlend" },
        { 403, "BitwiseXorAverage" },
        { 404, "BitwiseXorBlendX" },
        { 405, "BlackStrobe" },
        { 406, "BlendAlphaXorX" },
        { 407, "BlendCombinedValuesX" },
        { 408, "BlendFor360" },
        { 409, "BlendForward16" },
        { 410, "BlendForward32" },
        { 411, "BlendFromXtoY" },
        { 412, "BlendIncreaseX" },
        { 413, "BlendRedGreenBlue" },
        { 414, "BlendWithColorX" },
        { 415, "BlendAngle" },
        { 416, "BlockScale" },
        { 417, "BlockStrobe" },
        { 418, "BlockXor" },
        { 419, "BlockyTrails16" },
        { 420, "BlockyTrails32" },
        { 421, "BlurDistortionX" },
        { 422, "CannyStrobe" },
        { 423, "ColorFadeSlow" },
        { 424, "ColorFibonacci" },
        { 425, "CurtainEffect" },
        { 426, "DarkColorFibonacci" },
        { 427, "DarkColorsBlend" },
        { 428, "EnergizeBlend" },
        { 429, "AverageLines" },
        { 430, "AverageLinesBlendX" },
        { 431, "BlendRowAlpha" },
        { 432, "BlendInOut" },
        { 433, "ColorFlashIncreaseX" },
        { 434, "ColorIncreaseInOut" },
        { 435, "ColorLinesX" },
        { 436, "ColorMoveDownX" },
        { 437, "ColorOrderSwapX" },
        { 438, "ColorPulseAlphaX" },
        { 439, "ColorRowShiftX" },
        { 440, "ColorShiftXorX" },
        { 441, "CopyXorAlphaX" },
        { 442, "CycleShiftRGBX" },
        { 443, "DarkNegateX" },
        { 444, "DarkSelfAlphaX" },
        { 445, "DiagonalGlitch" },
        { 446, "DigitalHaze" },
        { 447, "DoubleXorBlend" },
        { 448, "EchoBlend" },
        { 449, "ElectricEdge" },
        { 450, "FlashColorStrobe" },
        { 451, "FrameDiffXor" },
        { 452, "GhostMirror" },
        { 453, "GlitchSort" },
        { 454, "HeatWave" },
        { 455, "InterlaceBlend" },
        { 456, "InvertStrobe" },
        { 457, "KaleidoBlend" },
        { 458, "LightStrobe" },
        { 459, "LineGlitchX" },
        { 460, "MosaicBlend" },
        { 461, "NegatePulse" },
        { 462, "OffsetGhost" },
        { 463, "PixelateWave" },
        { 464, "QuantizeBlend" },
        { 465, "RandomLines" },
        { 466, "RippleDisplace" },
        { 467, "RotateShift" },
        { 468, "SaturationGlow" },
        { 469, "ScaleToCenter" },
        { 470, "ShadowMirror" },
        { 471, "ShiftChannels" },
        { 472, "SliceGlitch" },
        { 473, "SobelGlow" },
        { 474, "SpectralShift" },
        { 475, "SpiralTrail" },
        { 476, "SquareTrails" },
        { 477, "StrobeNegate" },
        { 478, "ThermalBlend" },
        { 479, "TintShift" },
        { 480, "TrailEcho" },
        { 481, "TransitionBlend" },
        { 482, "TwistWarp" },
        { 483, "VerticalShift" },
        { 484, "VortexBlend" },
        { 485, "WeavePattern" },
        { 486, "WhiteBurst" },
        { 487, "WiggleDisplace" },
        { 488, "XorPulseX" },
        { 489, "YellowShift" },
        { 490, "ZigzagGlitch" },
        { 491, "AlphaModulate" },
        { 492, "BlockSwap" },
        { 493, "ColorResonance" },
        { 494, "DepthGlitch" },
        { 495, "EchoShift" },
        { 496, "FractalNoise" },
        { 497, "GradientRotate" },
        { 498, "HarmonicShift" },
        { 499, "AcidWarp" },
        { 500, "BlendDiagonal" },
        { 501, "ChromaFlash" },
        { 502, "CircleWave" },
        { 503, "ColorCrush" },
        { 504, "CrosshatchBlend" },
        { 505, "CyberGlitch" },
        { 506, "DarkPulse" },
        { 507, "DiamondPatternX" },
        { 508, "DigitalRain" },
        { 509, "DisplaceX" },
        { 510, "DriftBlend" },
        { 511, "EdgePulse" },
        { 512, "FlameEffect" },
        { 513, "FlickerShift" },
        { 514, "GhostLayer" },
        { 515, "GlitchBlockX" },
        { 516, "GlowPulse" },
        { 517, "GridDistort" },
        { 518, "HexPattern" },
        { 519, "HueRotate" },
        { 520, "InterweaveX" },
        { 521, "JitterBlend" },
        { 522, "KaleidoScope4" },
        { 523, "LaserScan" },
        { 524, "LightLeak" },
        { 525, "MeltDown" },
        { 526, "MirrorDiag" },
        { 527, "NeonGlow" },
        { 528, "NoiseBlendX" },
        { 529, "PixelDrift" },
        { 530, "PlasmaWave" },
        { 531, "PrismSplit" },
        { 532, "PulseRadial" },
        { 533, "RainbowStrobe" },
        { 534, "RefractionX" },
        { 535, "ScanlineX" },
        { 536, "ShatterEffect" },
        { 537, "StaticNoise" },
        { 538, "TunnelVision" },
        { 539, "AberrationPulse" },
        { 540, "AquaWave" },
        { 541, "BinaryFlash" },
        { 542, "BloomGlow" },
        { 543, "CellularNoise" },
        { 544, "ChromaShift2" },
        { 545, "ColorBands" },
        { 546, "ColorVortex" },
        { 547, "CrystalMosaic" },
        { 548, "CubicDistort" },
        { 549, "DepthFade" },
        { 550, "DiscoFlash" },
        { 551, "DitherBlend" },
        { 552, "DoubleVision" },
        { 553, "DreamHaze" },
        { 554, "ElectricStorm" },
        { 555, "EmbossShift" },
        { 556, "FiberOptic" },
        { 557, "FilmGrain" },
        { 558, "FireWorks" },
        { 559, "FluidMotion" },
        { 560, "FogRoll" },
        { 561, "GlassRefract" },
        { 562, "GlowTrails" },
        { 563, "GridPulse" },
        { 564, "HalftoneBlend" },
        { 565, "HeatDistort" },
        { 566, "HoloGlitch" },
        { 567, "InfraredView" },
        { 568, "LavaLamp" },
        { 569, "LensFlare" },
        { 570, "LightningBolt" },
        { 571, "LiquidMetal" },
        { 572, "MatrixCode" },
        { 573, "MirrorKaleid" },
        { 574, "NightVision" },
        { 575, "OilSlick" },
        { 576, "ParticleField" },
        { 577, "PinwheelSpin" },
        { 578, "PixelStorm" },
        { 579, "PlaidPattern" },
        { 580, "PolarInvert" },
        { 581, "PolychromeTint" },
        { 582, "PopArtDots" },
        { 583, "PrismaticEdge" },
        { 584, "PulseWarp" },
        { 585, "QuantumNoise" },
        { 586, "QuiltBlend" },
        { 587, "RadarSweep" },
        { 588, "RaindropRipple" },
        { 589, "RasterBars" },
        { 590, "RetroTube" },
        { 591, "RingWave" },
        { 592, "RippleTank" },
        { 593, "RotatingPrism" },
        { 594, "SandStorm" },
        { 595, "SaturationPulse" },
        { 596, "ScatterPixel" },
        { 597, "ShadowPlay" },
        { 598, "ShimmerGlass" },
        { 599, "SilhouetteBlend" },
        { 600, "SketchOutline" },
        { 601, "SliceShift" },
        { 602, "SmearMotion" },
        { 603, "SmokeWisp" },
        { 604, "SnowDrift" },
        { 605, "SolarFlare" },
        { 606, "SparkShower" },
        { 607, "SpectrumWave" },
        { 608, "SpiralZoom" },
        { 609, "SplitMirror" },
        { 610, "StarBurst" },
        { 611, "StaticPulse" },
        { 612, "StencilCut" },
        { 613, "StippleShade" },
        { 614, "StormCloud" },
        { 615, "StreakBlur" },
        { 616, "StrobeEdge" },
        { 617, "SubpixelShift" },
        { 618, "SwimDistort" },
        { 619, "TangentWarp" },
        { 620, "TapeGlitch" },
        { 621, "TechnoGrid" },
        { 622, "TeleportPixel" },
        { 623, "TemporalBlur" },
        { 624, "TerraFracture" },
        { 625, "TextureWave" },
        { 626, "ThresholdPulse" },
        { 627, "TidalWave" },
        { 628, "TintCycle" },
        { 629, "TraceEdge" },
        { 630, "TriangleMosaic" },
        { 631, "TripleSplit" },
        { 632, "TurbulentFlow" },
        { 633, "UnderwaterCaustic" },
        { 634, "UnsharpPulse" },
        { 635, "VaporTrail" },
        { 636, "VectorField" },
        { 637, "VelocityBlur" },
        { 638, "VerticalMelt" },
        { 639, "VHSTracking" },
        { 640, "VibrantPop" },
        { 641, "VignetteFlash" },
        { 642, "VoronoiShatter" },
        { 643, "WarpSpeed" },
        { 644, "WaterColor" },
        { 645, "WaveCollapse" },
        { 646, "WebPattern" },
        { 647, "WhirlpoolSpin" },
        { 648, "WindBlast" },
        { 649, "WireframePulse" },
        { 650, "XRayFlash" },
        { 651, "ZebraStripe" },
        { 652, "ZenRipple" },
        { 653, "ZigzagWave" },
        { 654, "ZoomPulse" },
        { 655, "ZoneTint" },
        { 656, "AcidDrip" },
        { 657, "AuroraWave" },
        { 658, "BandPass" },
        { 659, "BilinearStretch" },
        { 660, "BleedThrough" },
        { 661, "BlockShatter" },
        { 662, "BlurMask" },
        { 663, "BokehBlur" },
        { 664, "BounceWave" },
        { 665, "BrokenGlass" },
        { 666, "BubbleWarp" },
        { 667, "CRTCurvature" },
        { 668, "CascadeBlend" },
        { 669, "CelShade" },
        { 670, "ChainReaction" },
        { 671, "ChannelDelay" },
        { 672, "ChromaBleed" },
        { 673, "CircuitTrace" },
        { 674, "ClockWipe" },
        { 675, "CloudShadow" },
        { 676, "ColorBurn" },
        { 677, "ColorHalves" },
        { 678, "ComicDots" },
        { 679, "ConcentricPulse" },
        { 680, "CopperTone" },
        { 681, "CornerStretch" },
        { 682, "CosmicDust" },
        { 683, "CrossBlur" },
        { 684, "CrossProcess" },
        { 685, "CrystalEdge" },
        { 686, "CubeRotate" },
        { 687, "CurtainReveal" },
        { 688, "CyberPunk" },
        { 689, "DataCorrupt" },
        { 690, "DebrisField" },
        { 691, "DeepFry" },
        { 692, "DesyncRGB" },
        { 693, "DiagonalWipe" },
        { 694, "DigitalArtifact" },
        { 695, "DimensionRift" },
        { 696, "DotCrawl" },
        { 697, "DualTone" },
        { 698, "EchoFade" },
        { 699, "EdgeMelt" },
        { 700, "ElasticWarp" },
        { 701, "EmberGlow" },
        { 702, "EntropyShift" },
        { 703, "ErosionBlend" },
        { 704, "ExplosionBurst" },
        { 705, "FacetMirror" },
        { 706, "FadeStreak" },
        { 707, "FeatherEdge" },
        { 708, "FlashFreeze" },
        { 709, "FlipMirror" },
        { 710, "FloatDrift" },
        { 711, "FlowField" },
        { 712, "FoldWarp" },
        { 713, "FragmentScatter" },
        { 714, "FrequencyPulse" },
        { 715, "FrostBite" },
        { 716, "FuseBlend" },
        { 717, "GalaxySpiral" },
        { 718, "GelWobble" },
        { 719, "GhostEcho" },
        { 720, "GlassShatter" },
        { 721, "GlimmerPulse" },
        { 722, "GlitchMosaic" },
        { 723, "GlowEdge" },
        { 724, "GradientMelt" },
        { 725, "GrainStorm" },
        { 726, "GravityPull" },
        { 727, "GridWarp" },
        { 728, "HaloRing" },
        { 729, "HarshLight" },
        { 730, "HazeLayer" },
        { 731, "HeatRipple" },
        { 732, "HexagonBlur" },
        { 733, "HighContrast" },
        { 734, "HologramScan" },
        { 735, "HorizonBend" },
        { 736, "HotSpot" },
        { 737, "HueWobble" }
    };
    struct FilterParams {
        float alpha;
        bool isNegative;
        int numFrames;
        int square_size;
        int start_index;
        int start_dir;
        int int_param1;
        int int_param2;
        float float_param1;
        int seed;
        int threshold;
        int frame_count;
        int sumR, sumG, sumB;
        int sw, sh;
    };
    __device__ float gpu_rand(int x, int y, int seed) {
        size_t res = (x * 1597334677U) ^ (y * 3812015801U) ^ (seed * 354856327U);
        res = res * (res ^ (res >> 16));
        res = res * (res ^ (res >> 13));
        return (float)(res & 0xFFFFFF) / 16777216.0f;
    }
    __device__ void setAlpha(unsigned char* data, int idx, bool isNegative) {
        if (isNegative) {
            data[idx] = 255 - data[idx];
            data[idx + 1] = 255 - data[idx + 1];
            data[idx + 2] = 255 - data[idx + 2];
        }
        data[idx + 3] = 255;
    }
    __device__ bool colorBounds(unsigned char r1, unsigned char g1, unsigned char b1, 
                                unsigned char r2, unsigned char g2, unsigned char b2, 
                                int ir, int ig, int ib) {
        return (abs(r1 - r2) < ir && abs(g1 - g2) < ig && abs(b1 - b2) < ib);
    }
    __device__ void processSelfAlphaBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        data[idx] = (unsigned char)(b + (b * params.alpha));
        data[idx + 1] = (unsigned char)(g + (g * params.alpha));
        data[idx + 2] = (unsigned char)(r + (r * params.alpha));
        setAlpha(data, idx, params.isNegative);
    }
    __device__ void processSelfScaleRefined(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;    
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        float alpha = params.alpha;
        data[idx] = (unsigned char)fminf(255.0f, b * alpha);
        data[idx + 1] = (unsigned char)fminf(255.0f, g * alpha);
        data[idx + 2] = (unsigned char)fminf(255.0f, r * alpha);
        setAlpha(data, idx, params.isNegative);
    }
    __device__ void processMedianBlend(int x, int y, unsigned char* currentFrame, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sumB = 0;
        int sumG = 0;
        int sumR = 0;
        for (int j = 0; j < params.numFrames; ++j) {
            unsigned char* framePtr = allFrames[j];
            if (framePtr != nullptr) {
                sumB += framePtr[idx];
                sumG += framePtr[idx + 1];
                sumR += framePtr[idx + 2];
            }
        }
        unsigned char newB = currentFrame[idx] ^ (unsigned char)(1 + sumB);
        unsigned char newG = currentFrame[idx + 1] ^ (unsigned char)(1 + sumG);
        unsigned char newR = currentFrame[idx + 2] ^ (unsigned char)(1 + sumR);
        currentFrame[idx] = newG;
        currentFrame[idx + 1] = newR;
        currentFrame[idx + 2] = newB;
    }
    __device__ void processSquareBlockResize(int x, int y, unsigned char* currentFrame, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int block_row = y / params.square_size;
        int period = 2 * (params.numFrames - 1);
        if (period <= 0) {
            period = 1;
        }
        int start_pos = (params.start_dir == 1) ? params.start_index : (2 * (params.numFrames - 1)) - params.start_index;
        int pos = (start_pos + block_row) % period;
        int frame_index = (pos < params.numFrames) ? pos : period - pos;
        if (frame_index < 0) {
            frame_index = 0;
        }
        if (frame_index >= params.numFrames) {
            frame_index = params.numFrames - 1;
        }
        int idx = y * step + x * 4;
        unsigned char* historyFrame = allFrames[frame_index];
        if (historyFrame != nullptr) {
            currentFrame[idx] = (unsigned char)(0.5f * currentFrame[idx] + 0.5f * historyFrame[idx]);
            currentFrame[idx + 1] = (unsigned char)(0.5f * currentFrame[idx + 1] + 0.5f * historyFrame[idx + 1]);
            currentFrame[idx + 2] = (unsigned char)(0.5f * currentFrame[idx + 2] + 0.5f * historyFrame[idx + 2]);
        }
    }
    __device__ void processStrangeGlitch(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        for (int q = 0; q < params.numFrames; ++q) {
            unsigned char* otherFrame = allFrames[q];
            if (otherFrame != nullptr) {
                unsigned char ob = otherFrame[idx];
                unsigned char og = otherFrame[idx + 1];
                unsigned char or_ = otherFrame[idx + 2];
                if (!colorBounds(r, g, b, or_, og, ob, 30, 30, 30)) {
                    data[idx] = ob;
                    data[idx + 1] = og;
                    data[idx + 2] = or_;
                    break; 
                }
            }
        }
    }
    __device__ void processMatrixOutline(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* compareFrame = allFrames[4 % params.numFrames]; 
        if (compareFrame != nullptr) {
            if (colorBounds(data[idx + 2], data[idx + 1], data[idx], compareFrame[idx + 2], compareFrame[idx + 1], compareFrame[idx], 30, 30, 30)) {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            }
        }
    }
    __device__ void processAuraTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int indices[] = {1, 4, 7};
        float sumB = (float)data[idx];
        float sumG = (float)data[idx + 1];
        float sumR = (float)data[idx + 2];
        for (int i = 0; i < 3; ++i) {
            unsigned char* f = allFrames[indices[i] % params.numFrames];
            if (f != nullptr) {
                sumB = (sumB * 0.5f) + (f[idx] * 0.5f);
                sumG = (sumG * 0.5f) + (f[idx + 1] * 0.5f);
                sumR = (sumR * 0.5f) + (f[idx + 2] * 0.5f);
            }
        }
        data[idx] = (unsigned char)sumB;
        data[idx + 1] = (unsigned char)sumG;
        data[idx + 2] = (unsigned char)sumR;
    }
    __device__ void processSquareShrink(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int off = params.start_index; 
        if (y >= off && y < (height - off) && x >= off && x < (width - off)) {
            int idx = y * step + x * 4;
            unsigned char* reimage = allFrames[0]; 
            if (reimage != nullptr) {
                data[idx] = (unsigned char)((data[idx] * 0.5f) + (reimage[idx] * 0.5f));
                data[idx + 1] = (unsigned char)((data[idx + 1] * 0.5f) + (reimage[idx + 1] * 0.5f));
                data[idx + 2] = (unsigned char)((data[idx + 2] * 0.5f) + (reimage[idx + 2] * 0.5f));
            }
        }
    }
    __device__ void processWaveTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* frame1 = allFrames[params.int_param1 % params.numFrames];
        unsigned char* frame2 = allFrames[params.int_param2 % params.numFrames];
        if (frame1 && frame2) {
            for (int j = 0; j < 3; ++j) {
                float val = (0.33f * data[idx + j]) + (0.33f * frame1[idx + j]) + (0.33f * frame2[idx + j]);
                data[idx + j] = (unsigned char)fminf(255.0f, val);
            }
        }
    }
    __device__ void processParticleSlide(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float rx = gpu_rand(x, y, params.seed) * 200.0f - 100.0f;
        float ry = gpu_rand(y, x, params.seed) * 200.0f - 100.0f;
        int px = (x + (int)rx) % width;
        int py = (y + (int)ry) % height;
        if (px < 0) {
            px += width;
        }
        if (py < 0) {
            py += height;
        }
        unsigned char* ref = allFrames[params.frame_count % params.numFrames];
        if (ref) {
            int p_idx = py * step + px * 4;
            data[idx] = (unsigned char)(0.5f * data[idx] + 0.5f * ref[p_idx]);
            data[idx + 1] = (unsigned char)(0.5f * data[idx + 1] + 0.5f * ref[p_idx + 1]);
            data[idx + 2] = (unsigned char)(0.5f * data[idx + 2] + 0.5f * ref[p_idx + 2]);
        }
    }
    __device__ void processRGBShiftTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* f0 = allFrames[0];
        unsigned char* f7 = allFrames[7 % params.numFrames];
        unsigned char* f15 = allFrames[15 % params.numFrames];
        if (f0 && f7 && f15) {
            int off = params.frame_count % 3;
            if (off == 0) {
                data[idx] = f0[idx];
                data[idx + 1] = f7[idx + 1];
                data[idx + 2] = f15[idx + 2];
            } else if (off == 1) {
                data[idx] = f7[idx];
                data[idx + 1] = f15[idx + 1];
                data[idx + 2] = f0[idx + 2];
            } else {
                data[idx] = f15[idx];
                data[idx + 1] = f0[idx + 1];
                data[idx + 2] = f7[idx + 2];
            }
        }
    }
    __device__ void processTripHSV(int x, int y, unsigned char* data, size_t step, int shift) {
        int idx = y * step + x * 4;
        float b = data[idx] / 255.0f;
        float g = data[idx + 1] / 255.0f;
        float r = data[idx + 2] / 255.0f;
        float max_v = fmaxf(r, fmaxf(g, b));
        float min_v = fminf(r, fminf(g, b));
        float delta = max_v - min_v;
        float h = 0.0f;
        if (delta > 0) {
            if (max_v == r) {
                h = 60.0f * fmodf(((g - b) / delta), 6.0f);
            } else if (max_v == g) {
                h = 60.0f * (((b - r) / delta) + 2.0f);
            } else {
                h = 60.0f * (((r - g) / delta) + 4.0f);
            }
        }
        h = fmodf(h + shift, 360.0f);
        if (h < 0) {
            h += 360.0f;
        }
        float s = (max_v == 0) ? 0 : delta / max_v;
        float c = max_v * s;
        float x_h = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
        float m = max_v - c;
        float r1, g1, b1;
        if (h < 60) { r1 = c; g1 = x_h; b1 = 0; }
        else if (h < 120) { r1 = x_h; g1 = c; b1 = 0; }
        else if (h < 180) { r1 = 0; g1 = c; b1 = x_h; }
        else if (h < 240) { r1 = 0; g1 = x_h; b1 = c; }
        else if (h < 300) { r1 = x_h; g1 = 0; b1 = c; }
        else { r1 = c; g1 = 0; b1 = x_h; }
        data[idx] = (unsigned char)((b1 + m) * 255.0f);
        data[idx + 1] = (unsigned char)((g1 + m) * 255.0f);
        data[idx + 2] = (unsigned char)((r1 + m) * 255.0f);
    }
    __device__ void processSobel(int x, int y, unsigned char* data, int width, int height, size_t step, bool norm, int threshold) {
        if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
            return;
        }
        int idx = y * step + x * 4;
        float gx = 0;
        float gy = 0;
        int weights[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int c_idx = (y + i) * step + (x + j) * 4;
                float val = (data[c_idx] + data[c_idx + 1] + data[c_idx + 2]) / 3.0f;
                gx += val * weights[i + 1][j + 1];
                gy += val * weights[j + 1][i + 1];
            }
        }
        float mag = fabsf(gx) + fabsf(gy);
        unsigned char v = norm ? (unsigned char)fminf(255.0f, mag) : ((mag > threshold) ? 255 : 0);
        data[idx] = v;
        data[idx + 1] = v;
        data[idx + 2] = v;
    }
    __device__ int get_osc_offset(int frame_count, int max_val) {
        if (max_val == 0) return 0;
        int cycle = frame_count % (2 * max_val);
        return (cycle < max_val) ? cycle : (2 * max_val - cycle);
    }
    __device__ void processLineInLineOut4_Increase(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = get_osc_offset(params.frame_count, 250); 
        unsigned char* hist = allFrames[params.frame_count % params.numFrames];
        if (hist) {
            int h_idx = y * step + x * 4; 
            for(int j=0; j<3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx+j]) + (0.5f * hist[h_idx+j]));
            }
        }
    }
    __device__ void processLineInLineOut_ReverseIncrease(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = get_osc_offset(params.frame_count, 250);
        int src_x = (x + offset) % width;
        int tpos = width - src_x - 1;
        if (tpos >= 0 && tpos < width) {
            unsigned char* hist = allFrames[params.frame_count % params.numFrames];
            if (hist) {
                for(int j=0; j<3; ++j) {
                    data[idx+j] = (unsigned char)((0.5f * data[idx+j]) + (0.5f * hist[idx+j]));
                }
            }
        }
    }
    __device__ void processLineInLineOut_InvertedY(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = get_osc_offset(params.frame_count, 400);
        int zpos = height - y - 1; 
        if(zpos >= 0 && zpos < height) {
            unsigned char* hist = allFrames[params.frame_count % params.numFrames];
            if(hist) {
                 for(int j=0; j<3; ++j) {
                    data[idx+j] = (unsigned char)((0.5f * data[idx+j]) + (0.5f * hist[idx+j]));
                 }
            }
        }
    }
    __device__ void processLineInLineOut_Vertical(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = get_osc_offset(params.frame_count, 100);
        int src_y = (y + offset) % height;
        unsigned char* hist = allFrames[params.frame_count % params.numFrames];
        if(hist) {
            int src_idx = src_y * step + x * 4;
            for(int j=0; j<3; ++j) {
                data[src_idx+j] = (unsigned char)((0.5f * data[src_idx+j]) + (0.5f * hist[idx+j]));
            }
        }
    }
    __device__ void processLineInLineOut_IncreaseImage(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* reimage = allFrames[0]; 
        unsigned char* hist = allFrames[params.frame_count % params.numFrames];
        if(reimage && hist) {
            for(int j=0; j<3; ++j) {
                float val = (0.3f * data[idx+j]) + (0.3f * hist[idx+j]) + (0.3f * reimage[idx+j]);
                data[idx+j] = (unsigned char)fminf(255.0f, val);
            }
        }
    }
    __device__ void processSquareByRow(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params, bool reverse, bool strobe) {
        int SIZE_VALUE = 96;
        int idx = y * step + x * 4;
        int block_x = x / SIZE_VALUE;
        int block_y = y / SIZE_VALUE;
        float r_val = gpu_rand(block_x, block_y + params.frame_count, params.seed); 
        int index = (int)(r_val * 64.0f) % params.numFrames;
        unsigned char* hist = allFrames[index];
        if(hist) {
            int target_x = x;
            if(reverse) {
                if ((block_y % 2) != 0) {
                    target_x = width - x - 1;
                }
            }
            if (target_x >= 0 && target_x < width) {
                int h_idx = y * step + target_x * 4;
                if(strobe) {
                    for(int j=0; j<3; ++j) {
                        data[idx+j] = (unsigned char)((0.5f * data[idx+j]) + (0.5f * hist[h_idx+j]));
                    }
                } else {
                    data[idx] = hist[h_idx];
                    data[idx+1] = hist[h_idx+1];
                    data[idx+2] = hist[h_idx+2];
                }
            }
        }
    }
    __device__ void processDivideByValue(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned int total[3] = {0, 0, 0};
        int count = (params.numFrames > 8) ? 8 : params.numFrames;
        if(count == 0) return;
        for(int j = 0; j < count; ++j) {
            unsigned char* f = allFrames[j];
            if(f) {
                total[0] += f[idx];
                total[1] += f[idx+1];
                total[2] += f[idx+2];
            }
        }
        for(int j=0; j<3; ++j) {
            unsigned char avg = (unsigned char)(total[j] / 5);
            data[idx+j] = data[idx+j] ^ avg;
        }
    }
    __device__ void processColorCollectionSubtleStrobe(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = 3 % params.numFrames;
        int idx3 = 6 % params.numFrames;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        if (f1 && f2 && f3) {
            data[idx] = f1[idx];       
            data[idx + 1] = f2[idx + 1]; 
            data[idx + 2] = f3[idx + 2]; 
        }
    }
    __device__ void processCollectionRandom(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int frameIdx = (params.seed + x + y) % (params.numFrames > 1 ? params.numFrames - 1 : 1);
        unsigned char* f = allFrames[frameIdx];
        if (f) {
            data[idx] = f[idx];
            data[idx + 1] = f[idx + 1];
            data[idx + 2] = f[idx + 2];
        }
    }
    __device__ void processCollectionAlphaXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = 3 % params.numFrames;
        int idx3 = (params.numFrames - 1) % params.numFrames;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        if (f1 && f2 && f3) {
            data[idx] = data[idx] ^ f1[idx];
            data[idx + 1] = data[idx + 1] ^ f2[idx + 1];
            data[idx + 2] = data[idx + 2] ^ f3[idx + 2];
        }
    }
    __device__ void processColorCollection64X(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int midIdx = params.numFrames / 2;
        int animIdx = params.start_index % params.numFrames;
        unsigned char* f1 = allFrames[1 % params.numFrames];
        unsigned char* f2 = allFrames[midIdx];
        unsigned char* f3 = allFrames[animIdx];
        if (f1 && f2 && f3) {
            data[idx] = f1[idx];
            data[idx + 1] = f2[idx + 1];
            data[idx + 2] = f3[idx + 2];
        }
    }
    __device__ void processColorCollectionSwitch(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int switchIdx = params.frame_count % 3;
        int fidx1 = 1 % params.numFrames;
        int fidx2 = params.numFrames / 2;
        int fidx3 = (params.numFrames - 1);
        unsigned char* frames[3];
        frames[0] = allFrames[fidx1];
        frames[1] = allFrames[fidx2];
        frames[2] = allFrames[fidx3];
        if (frames[0] && frames[1] && frames[2]) {
            switch (switchIdx) {
                case 0:
                    data[idx] = frames[0][idx];
                    data[idx + 1] = frames[1][idx + 1];
                    data[idx + 2] = frames[2][idx + 2];
                    break;
                case 1:
                    data[idx] = frames[2][idx];
                    data[idx + 1] = frames[0][idx + 1];
                    data[idx + 2] = frames[1][idx + 2];
                    break;
                case 2:
                    data[idx] = frames[1][idx];
                    data[idx + 1] = frames[2][idx + 1];
                    data[idx + 2] = frames[0][idx + 2];
                    break;
            }
        }
    }
    __device__ void processColorCollectionRGB_Index(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channelIdx = params.frame_count % 3;
        int frameIdx = (channelIdx + 1) % params.numFrames;
        unsigned char* f = allFrames[frameIdx];
        if (f) {
            data[idx + channelIdx] = f[idx + channelIdx];
        }
    }
    __device__ void processColorCollectionGhostTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = (params.numFrames / 2) % params.numFrames;
        int idx3 = (params.numFrames - 1) % params.numFrames;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        if (f1 && f2 && f3) {
            for (int j = 0; j < 3; ++j) {
                float val = (f1[idx + j] * 0.33f) + (f2[idx + j] * 0.33f) + (f3[idx + j] * 0.33f);
                data[idx + j] = (unsigned char)fminf(255.0f, val);
            }
        }
    }
    __device__ void processColorCollectionScale(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = (params.numFrames / 2) % params.numFrames;
        int idx3 = (params.numFrames - 1) % params.numFrames;
        int channelIdx = params.frame_count % 3;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        if (f1 && f2 && f3) {
            data[idx] = f1[idx];
            data[idx + 1] = f2[idx + 1];
            data[idx + 2] = f3[idx + 2];
            data[idx + channelIdx] = (unsigned char)fminf(255.0f, data[idx + channelIdx] * params.alpha);
        }
    }
    __device__ void processColorCollectionReverseStrobe(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = 4 % params.numFrames;
        int idx3 = (params.numFrames - 1) % params.numFrames;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        bool reversed = (params.frame_count % 2) == 0;
        if (f1 && f2 && f3) {
            if (reversed) {
                data[idx] = f3[idx];
                data[idx + 1] = f2[idx + 1];
                data[idx + 2] = f1[idx + 2];
            } else {
                data[idx] = f1[idx];
                data[idx + 1] = f2[idx + 1];
                data[idx + 2] = f3[idx + 2];
            }
        }
    }
    __device__ void processColorCollectionXorPixel(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = 4 % params.numFrames;
        int idx3 = (params.numFrames - 1) % params.numFrames;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        unsigned char* frames[3] = {f1, f2, f3};
        bool reversed = (params.frame_count % 2) == 0;
        if (f1 && f2 && f3) {
            for (int j = 0; j < 3; ++j) {
                unsigned char* f = reversed ? frames[2 - j] : frames[j];
                unsigned char scaled_pix = (unsigned char)fminf(255.0f, data[idx + j] * params.alpha);
                unsigned char scaled_val = (unsigned char)fminf(255.0f, f[idx + j] * params.alpha);
                data[idx + j] = scaled_pix ^ scaled_val;
            }
        }
    }
    __device__ void processBlendWithSource25(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* orig = allFrames[0];
        if (orig) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.75f + orig[idx + j] * 0.25f);
            }
        }
    }
    __device__ void processBlendWithSource50(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* orig = allFrames[0];
        if (orig) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + orig[idx + j] * 0.5f);
            }
        }
    }
    __device__ void processBlendWithSource75(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* orig = allFrames[0];
        if (orig) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.25f + orig[idx + j] * 0.75f);
            }
        }
    }
    __device__ void processBlendWithSource100(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* orig = allFrames[0];
        if (orig) {
            float a = fminf(1.0f, params.alpha);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - a) + orig[idx + j] * a);
            }
        }
    }
    __device__ void processColorCollectionXorOffsetFlash(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offsetVal = (params.frame_count / 30) % 3;
        int idx1 = 1 % params.numFrames;
        int idx2 = (params.numFrames / 2) % params.numFrames;
        int idx3 = (params.numFrames - 1) % params.numFrames;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        unsigned char* frames[3] = {f1, f2, f3};
        if (f1 && f2 && f3) {
            for (int j = 0; j < 3; ++j) {
                if (offsetVal == j) {
                    data[idx + j] = data[idx + j] ^ frames[j][idx + offsetVal];
                } else {
                    data[idx + j] = frames[j][idx + j];
                }
            }
        }
    }
    __device__ void processColorCollectionMatrixGhost(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 0;
        int idx2 = params.numFrames / 2;
        int idx3 = params.numFrames - 1;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        if (f1 && f2 && f3) {
            for (int j = 0; j < 3; ++j) {
                float val = (f1[idx + j] * 0.33f) + (f2[idx + j] * 0.33f) + (f3[idx + j] * 0.33f);
                data[idx + j] = (unsigned char)fminf(255.0f, val);
            }
        }
    }
    __device__ void processMildStrobe(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int modeIdx = params.frame_count % 3;
        int fidx0 = 0;
        int fidx1 = 3 % params.numFrames;
        int fidx2 = 6 % params.numFrames;
        unsigned char* f0 = allFrames[fidx0];
        unsigned char* f1 = allFrames[fidx1];
        unsigned char* f2 = allFrames[fidx2];
        if (f0 && f1 && f2) {
            int values[3];
            switch (modeIdx) {
                case 0: values[0] = 0; values[1] = 1; values[2] = 2; break;
                case 1: values[0] = 1; values[1] = 2; values[2] = 0; break;
                case 2: values[0] = 0; values[1] = 2; values[2] = 1; break;
            }
            unsigned char* frames[3] = {f0, f1, f2};
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = frames[j][idx + values[j]];
            }
        }
    }
    __device__ void processReduceBy50(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f);
        }
    }
    __device__ void processColorPositionAverageXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int values[3];
        values[0] = (params.sumB) % 256;
        values[1] = (params.sumG) % 256;
        values[2] = (params.sumR) % 256;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[idx + j] ^ values[j];
        }
    }
    __device__ void processColorPositionXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* hist = allFrames[(params.numFrames - 1) % params.numFrames];
        int values[3];
        values[0] = (params.sumB) % 256;
        values[1] = (params.sumG) % 256;
        values[2] = (params.sumR) % 256;
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (data[idx + j] ^ hist[idx + j]) ^ values[j];
            }
        }
    }
    __device__ void processGrayStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if ((params.frame_count % 2) == 0) {
            float gray = 0.299f * data[idx + 2] + 0.587f * data[idx + 1] + 0.114f * data[idx];
            unsigned char g = (unsigned char)fminf(255.0f, gray);
            data[idx] = g;
            data[idx + 1] = g;
            data[idx + 2] = g;
        }
    }
    __device__ void processColorStrobeXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char randVal[3] = {
            (unsigned char)(params.sumB % 256),
            (unsigned char)(params.sumG % 256),
            (unsigned char)(params.sumR % 256)
        };
        float sval[3] = {0, 0, 0};
        int count = (params.numFrames > 8) ? 8 : params.numFrames;
        for (int q = 0; q < count; ++q) {
            unsigned char* f = allFrames[q];
            if (f) {
                for (int j = 0; j < 3; ++j) {
                    sval[j] += f[idx + j];
                }
            }
        }
        for (int j = 0; j < 3; ++j) {
            int avg = (int)(sval[j] / count);
            data[idx + j] = data[idx + j] ^ avg ^ randVal[j];
        }
    }
    __device__ void processColorGhost(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int animIdx = params.start_index % params.numFrames;
        int midIdx = params.numFrames / 2;
        int endIdx = params.numFrames - 2;
        if (endIdx < 0) endIdx = 0;
        unsigned char* f1 = allFrames[animIdx];
        unsigned char* f2 = allFrames[midIdx];
        unsigned char* f3 = allFrames[endIdx];
        if (f1 && f2 && f3) {
            data[idx] = f1[idx];
            data[idx + 1] = f2[idx + 1];
            data[idx + 2] = f3[idx + 2];
        }
    }
    __device__ void processBlurredOutXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int idx1 = 1 % params.numFrames;
        int idx2 = params.numFrames / 2;
        int idx3 = params.numFrames - 1;
        unsigned char* f1 = allFrames[idx1];
        unsigned char* f2 = allFrames[idx2];
        unsigned char* f3 = allFrames[idx3];
        if (f1 && f2 && f3) {
            for (int j = 0; j < 3; ++j) {
                unsigned char avg = (unsigned char)((f1[idx + j] + f2[idx + j] + f3[idx + j]) / 3);
                data[idx + j] = data[idx + j] ^ avg;
            }
        }
    }
    __device__ void processBoxFilter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        if (x <= 1 || x >= width - 2 || y <= 1 || y >= height - 2) return;
        int idx = y * step + x * 4;
        float sum[3] = {0, 0, 0};
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nidx = (y + dy) * step + (x + dx) * 4;
                for (int j = 0; j < 3; ++j) {
                    sum[j] += data[nidx + j];
                }
                count++;
            }
        }
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(sum[j] / count);
        }
    }
    __device__ void processDizzyFilter(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* f1 = allFrames[1 % params.numFrames];
        unsigned char* f2 = allFrames[params.numFrames / 2];
        unsigned char* f3 = allFrames[params.numFrames - 1];
        if (f1 && f2 && f3) {
            for (int j = 0; j < 3; ++j) {
                float val = (f1[idx + j] * 0.33f) + (f2[idx + j] * 0.33f) + (f3[idx + j] * 0.33f);
                data[idx + j] = (unsigned char)val;
            }
        }
    }
    __device__ void processBuzzed(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = (params.numFrames > 8) ? 8 : params.numFrames;
        float sep_value = 1.0f / count;
        float value[3] = {0, 0, 0};
        for (int q = 0; q < count; ++q) {
            unsigned char* f = allFrames[q];
            if (f) {
                for (int j = 0; j < 3; ++j) {
                    value[j] += f[idx + j] * sep_value;
                }
            }
        }
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)value[j];
        }
    }
    __device__ void processBuzzedDark(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = (params.numFrames > 16) ? 16 : params.numFrames;
        float sep_value = 1.0f / (count > 1 ? count - 1 : 1);
        float value[3] = {0, 0, 0};
        for (int q = 0; q < count; ++q) {
            unsigned char* f = allFrames[q];
            if (f) {
                for (int j = 0; j < 3; ++j) {
                    value[j] += f[idx + j] * sep_value;
                }
            }
        }
        for (int j = 0; j < 3; ++j) {
            float v = value[j] * 0.7f; 
            data[idx + j] = (unsigned char)fminf(255.0f, v);
        }
    }
    __device__ void processAllRed(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        data[idx] = 0;      
        data[idx + 1] = 0;  
    }
    __device__ void processAllGreen(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        data[idx] = 0;      
        data[idx + 2] = 0;  
    }
    __device__ void processAllBlue(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        data[idx + 1] = 0;  
        data[idx + 2] = 0;  
    }
    __device__ void processNegativeStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if ((params.frame_count % 2) == 0) {
            data[idx] = ~data[idx];
            data[idx + 1] = ~data[idx + 1];
            data[idx + 2] = ~data[idx + 2];
        }
    }
    __device__ void processXorAddMul(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float blend = 1.0f + (params.frame_count % 13);
        int b = (int)blend;
        data[idx] = (unsigned char)(data[idx] + (data[idx] ^ b));
        data[idx + 1] = (unsigned char)(data[idx + 1] + b);
        data[idx + 2] = (unsigned char)(data[idx + 2] * b);
    }
    __device__ void processHorizontalLines(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos0 = (float)((x + y + params.frame_count) % 100);
        float pos1 = (float)((x + y + params.frame_count + 16) % 100);
        float pos2 = (float)((x + y + params.frame_count + 32) % 100);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + pos0);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + pos1);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + pos2);
    }
    __device__ void processStrobeRedGreenBlue(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int color = params.frame_count % 3;
        switch (color) {
            case 0: 
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                break;
            case 1: 
                data[idx] = 0;
                data[idx + 2] = 0;
                break;
            case 2: 
                data[idx] = 0;
                data[idx + 1] = 0;
                break;
        }
    }
    __device__ void processPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * pos);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] * pos);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * pos);
    }
    __device__ void processDiamondPattern(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 0.3f * sinf(params.frame_count * 0.05f);
        if ((x % 2) == 0) {
            if ((y % 2) == 0) {
                data[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, (1.0f - pos * data[idx])));
                data[idx + 2] = (unsigned char)((x + y) * pos);
            } else {
                data[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, pos * data[idx] - y));
                data[idx + 2] = (unsigned char)fabs((x - y) * pos);
            }
        } else {
            if ((y % 2) == 0) {
                data[idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, (1.0f - pos * data[idx + 1])));
                data[idx + 2] = (unsigned char)((x + y) * pos);
            } else {
                data[idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, pos * data[idx + 1] - y));
            }
        }
    }
    __device__ void processBitwiseXOR(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[1 % params.numFrames];
        if (prev) {
            data[idx] ^= prev[idx];
            data[idx + 1] ^= prev[idx + 1];
            data[idx + 2] ^= prev[idx + 2];
        }
    }
    __device__ void processBitwiseAND(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[1 % params.numFrames];
        if (prev) {
            data[idx] &= prev[idx];
            data[idx + 1] &= prev[idx + 1];
            data[idx + 2] &= prev[idx + 2];
        }
    }
    __device__ void processBitwiseOR(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[1 % params.numFrames];
        if (prev) {
            data[idx] |= prev[idx];
            data[idx + 1] |= prev[idx + 1];
            data[idx + 2] |= prev[idx + 2];
        }
    }
    __device__ void processBlendSwitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int pos = x % 3;
        unsigned char blend_pixel = (unsigned char)((x + y + params.frame_count) % 256);
        data[idx + pos] = (unsigned char)(data[idx + pos] * blend_pixel / 255);
    }
    __device__ void processLineRGB(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        int counter = y % 3;
        switch (counter) {
            case 0:
                data[idx] = 0;
                data[idx + 1] = 0;
                break;
            case 1:
                data[idx] = 0;
                data[idx + 2] = 0;
                break;
            case 2:
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                break;
        }
    }
    __device__ void processPixelRGB(int x, int y, unsigned char* data, int width, size_t step) {
        int idx = y * step + x * 4;
        int counter = (y * width + x) % 3;
        switch (counter) {
            case 0:
                data[idx] = 0;
                data[idx + 1] = 0;
                break;
            case 1:
                data[idx] = 0;
                data[idx + 2] = 0;
                break;
            case 2:
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                break;
        }
    }
    __device__ void processInvertedScanlines(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.3f * sinf(params.frame_count * 0.1f);
        int index = (x + y) % 3;
        switch (index) {
            case 0:
                for (int j = 0; j < 3; ++j)
                    data[idx + j] = (unsigned char)fminf(255.0f, (~data[idx + j]) * alpha);
                break;
            case 1: {
                unsigned char temp = data[idx];
                data[idx] = (unsigned char)fminf(255.0f, data[idx + 2] * alpha);
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * alpha);
                data[idx + 2] = (unsigned char)fminf(255.0f, temp * alpha);
                break;
            }
            case 2:
                break;
        }
    }
    __device__ void processScanSwitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int start_index = params.frame_count % 2;
        int index = (start_index + x + y) % 2;
        if (index == 1) {
            data[idx] = ~data[idx];
            data[idx + 1] = ~data[idx + 1];
            data[idx + 2] = ~data[idx + 2];
        }
    }
    __device__ void processScanAlphaSwitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        int start_index = params.frame_count % 2;
        int index = (start_index + x + y) % 2;
        if (index == 0) {
            data[idx] = ~data[idx];
            data[idx + 1] = ~data[idx + 1];
            data[idx + 2] = ~data[idx + 2];
        } else {
            data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] * alpha);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * alpha);
        }
    }
    __device__ void processRGBFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 0.3f * sinf(params.frame_count * 0.2f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + pos * params.sumB);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + pos * params.sumG);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + pos * params.sumR);
    }
    __device__ void processDiagonalLines(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int tz = height - y - 1;
        if (tz >= 0 && tz < height) {
            int tidx = tz * step + x * 4;
            float pos = (float)((x + y + params.frame_count) % 100);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)fminf(255.0f, (data[idx + j] + data[tidx + j]) + pos);
            }
        }
    }
    __device__ void processDarken(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        data[idx] /= 6;
        data[idx + 1] /= 6;
        data[idx + 2] /= 6;
    }
    __device__ void processSelfXorBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char index0 = (unsigned char)((params.frame_count) % 256);
        unsigned char index1 = (unsigned char)((params.frame_count + 85) % 256);
        unsigned char index2 = (unsigned char)((params.frame_count + 170) % 256);
        data[idx] ^= index0;
        data[idx + 1] ^= index1;
        data[idx + 2] ^= index2;
    }
    __device__ void processSelfXorDoubleFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 128.0f + 127.0f * sinf(params.frame_count * 0.1f);
        unsigned char a = (unsigned char)alpha;
        data[idx] ^= (a + params.sumB);
        data[idx + 1] ^= (a + params.sumG);
        data[idx + 2] ^= (a + params.sumR);
    }
    __device__ void processSelfOrDoubleFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 128.0f + 127.0f * sinf(params.frame_count * 0.1f);
        unsigned char a = (unsigned char)alpha;
        data[idx] |= (a ^ params.sumB);
        data[idx + 1] |= (a ^ params.sumG);
        data[idx + 2] |= (a ^ params.sumR);
    }
    __device__ void processBlendRowCurvedSqrt(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        float amount = sqrtf((float)(x * y));
        unsigned char amt = (unsigned char)fmodf(amount, 256.0f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, (data[idx + j] ^ amt) * alpha);
        }
    }
    __device__ void processXorAlpha(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 2.0f * sinf(params.frame_count * 0.05f);
        data[idx] ^= (unsigned char)(params.sumB * alpha);
        data[idx + 1] ^= (unsigned char)(params.sumG * alpha);
        data[idx + 2] ^= (unsigned char)(params.sumR * alpha);
    }
    __device__ void processRandomXorBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        data[idx] ^= params.sumB;
        data[idx + 1] ^= params.sumG;
        data[idx + 2] ^= params.sumR;
    }
    __device__ void processAndStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        data[idx] &= params.sumB;
        data[idx + 1] &= params.sumG;
        data[idx + 2] &= params.sumR;
    }
    __device__ void processAndStrobeScale(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        data[idx] = (unsigned char)fminf(255.0f, (data[idx] & params.sumB) * alpha);
        data[idx + 1] = (unsigned char)fminf(255.0f, (data[idx + 1] & params.sumG) * alpha);
        data[idx + 2] = (unsigned char)fminf(255.0f, (data[idx + 2] & params.sumR) * alpha);
    }
    __device__ void processAndPixelStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char col = (unsigned char)((params.frame_count * 50) % 256);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[idx + j] & col;
            data[idx + j] = data[idx + j] ^ col;
        }
    }
    __device__ void processAndOrXorStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int index = params.frame_count % 3;
        for (int j = 0; j < 3; ++j) {
            unsigned char col = (j == 0) ? params.sumB : (j == 1 ? params.sumG : params.sumR);
            switch (index) {
                case 0: data[idx + j] &= col; break;
                case 1: data[idx + j] |= col; break;
                case 2: data[idx + j] ^= col; break;
            }
        }
    }
    __device__ void processFadeInAndOut(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float s0 = 128.0f + 127.0f * sinf(params.frame_count * 0.05f);
        float s1 = 128.0f + 127.0f * sinf(params.frame_count * 0.07f);
        float s2 = 128.0f + 127.0f * sinf(params.frame_count * 0.09f);
        data[idx] ^= (unsigned char)s0;
        data[idx + 1] ^= (unsigned char)s1;
        data[idx + 2] ^= (unsigned char)s2;
    }
    __device__ void processBrightStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int speed = 1 + (params.frame_count % 25);
        data[idx] ^= (unsigned char)((params.sumB * speed) % 256);
        data[idx + 1] ^= (unsigned char)((params.sumG * speed) % 256);
        data[idx + 2] ^= (unsigned char)((params.sumR * speed) % 256);
    }
    __device__ void processDarkStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        unsigned char c0 = (params.sumB > 0) ? params.sumB : 1;
        unsigned char c1 = (params.sumG > 0) ? params.sumG : 1;
        unsigned char c2 = (params.sumR > 0) ? params.sumR : 1;
        data[idx] = (unsigned char)fminf(255.0f, (data[idx] % c0) * alpha);
        data[idx + 1] = (unsigned char)fminf(255.0f, (data[idx + 1] % c1) * alpha);
        data[idx + 2] = (unsigned char)fminf(255.0f, (data[idx + 2] % c2) * alpha);
    }
    __device__ void processRandomXorOpposite(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            unsigned char col1 = (j == 0) ? params.sumB : (j == 1 ? params.sumG : params.sumR);
            unsigned char col2 = (unsigned char)((col1 + 128) % 256);
            data[idx + j] = ~data[idx + j] ^ col1 ^ ~col2;
        }
    }
    __device__ void processGradientRainbow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float start_color = ((params.sumR % 255) + 1) * 0.5f + y * 0.1f;
        int color_R = (int)(start_color * 4) % 256;
        int color_G = (int)(start_color * 6) % 256;
        int color_B = (int)(start_color * 8) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + color_B);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + color_G);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + color_R);
    }
    __device__ void processCossinMultiply(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.05f;
        float sinVal = sinf(alpha) * x;
        float cosVal = cosf(alpha) * y;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + fabs(sinVal));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + fabs(cosVal));
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + (data[idx] + data[idx + 1] + data[idx + 2]) / 3.0f);
    }
    __device__ void processColorAccumulate1(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.05f;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        data[idx] = (unsigned char)fminf(255.0f, b + r * alpha);
        data[idx + 1] = (unsigned char)fminf(255.0f, g + b * alpha);
        data[idx + 2] = (unsigned char)fminf(255.0f, r + g * alpha);
    }
    __device__ void processColorAccumulate2(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.05f;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        data[idx] = (unsigned char)fminf(255.0f, b + r * alpha + y);
        data[idx + 1] = (unsigned char)fminf(255.0f, g + b * alpha + x);
        data[idx + 2] = (unsigned char)fminf(255.0f, r + g * alpha + y - x);
    }
    __device__ void processWeakBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int index = (x + y) % 3;
        float value = 1.0f + (params.sumR % 10) * 0.5f;
        int val = (int)(data[idx + index] + (data[idx + index] * value)) / 2;
        data[idx + index] = (unsigned char)fminf(255.0f, (float)val);
    }
    __device__ void processStrobeEffect(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        int passIndex = params.frame_count % 4;
        switch (passIndex) {
            case 0:
                data[idx] = (unsigned char)fminf(255.0f, data[idx] * (-alpha + 2.0f));
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * alpha);
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * alpha);
                break;
            case 1:
                data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha);
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (-alpha + 2.0f));
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * alpha);
                break;
            case 2:
                data[idx] = (unsigned char)fminf(255.0f, data[idx] * alpha);
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * alpha);
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (-alpha + 2.0f));
                break;
            case 3:
                data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha);
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + params.sumG * alpha);
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + params.sumR * alpha);
                break;
        }
    }
    __device__ void processBlend3(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float rValue0 = 0.1f * sinf(params.frame_count * 0.1f);
        float rValue1 = 0.1f * sinf(params.frame_count * 0.13f);
        float rValue2 = 0.1f * sinf(params.frame_count * 0.17f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * rValue0);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] * rValue1);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * rValue2);
    }
    __device__ void processNegParadox(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.3f * sinf(params.frame_count * 0.05f);
        data[idx] = (unsigned char)fminf(255.0f, (data[idx] * alpha) + (data[idx] * alpha));
        data[idx + 1] = (unsigned char)fminf(255.0f, (data[idx + 1] * alpha) + (params.sumG * alpha));
        data[idx + 2] = (unsigned char)fminf(255.0f, (data[idx + 1] * alpha) + (data[idx + 2] * alpha));
    }
    __device__ void processThoughtMode(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        int mode = (x + y) % 3;
        int sw = params.frame_count % 2;
        int tr = (params.frame_count + 1) % 2;
        if (sw == 1) data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[mode] * alpha);
        if (tr == 0) data[idx + mode] = (unsigned char)fmaxf(0.0f, data[idx + mode] - data[params.sumR % 3] * alpha);
        data[idx + mode] = (unsigned char)fminf(255.0f, data[idx + mode] + data[idx + mode] * alpha);
    }
    __device__ void processTri(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.3f * sinf(params.frame_count * 0.1f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] + params.sumG * alpha);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] + params.sumR * alpha);
    }
    __device__ void processDistort(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.3f * sinf(params.frame_count * 0.1f);
        data[idx] = (unsigned char)fminf(255.0f, (y * alpha) + data[idx]);
        data[idx + 2] = (unsigned char)fminf(255.0f, (x * alpha) + data[idx + 2]);
        data[idx + 1] = (unsigned char)fminf(255.0f, alpha * data[idx + 1]);
    }
    __device__ void processColorAccumulate3(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.05f;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        data[idx] = (unsigned char)fminf(255.0f, b + (r * alpha));
        data[idx + 1] = (unsigned char)fminf(255.0f, g + (b * alpha) + x);
        data[idx + 2] = (unsigned char)fminf(255.0f, r + (g * alpha) + y);
    }
    __device__ void processFilter8(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + ((x + y) * alpha));
        }
    }
    __device__ void processFilter3(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        unsigned char b = data[idx];
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, b + (data[idx + j]) * alpha);
        }
    }
    __device__ void processRainbowBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.3f * sinf(params.frame_count * 0.05f);
        int rb = (params.sumR + params.frame_count) % 256;
        int gb = (params.sumG + params.frame_count * 2) % 256;
        int bb = (params.sumB + params.frame_count * 3) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + alpha * bb);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + alpha * gb);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + alpha * rb);
    }
    __device__ void processPixelScale(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        data[idx] = (unsigned char)fminf(255.0f, (b * pos) + (b - r));
        data[idx + 1] = (unsigned char)fminf(255.0f, (g * pos) + (g + g));
        data[idx + 2] = (unsigned char)fminf(255.0f, (r * pos) + (r - b));
    }
    __device__ void processGradientSelf(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        int index = y % 3;
        int count = (x + y) % 256;
        data[idx + index] = (unsigned char)fminf(255.0f, (data[idx + index] * pos) + count);
    }
    __device__ void processGradientSelfVertical(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        int index = x % 3;
        int count = (x + y) % 256;
        data[idx + index] = (unsigned char)fminf(255.0f, (data[idx + index] * pos) + count);
    }
    __device__ void processGradientDown(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        int index = (x + y) % 3;
        int count = y % 256;
        data[idx + index] = (unsigned char)fminf(255.0f, (data[idx + index] * pos) + count);
    }
    __device__ void processGraidentHorizontal(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        int index = (x + y) % 3;
        int count = x % 256;
        data[idx + index] = (unsigned char)fminf(255.0f, (data[idx + index] * pos) + count);
    }
    __device__ void processInter(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int start_x = params.frame_count % 2;
        if ((y + start_x) % 2 == 0) {
            data[idx] = 0;
            data[idx + 1] = 0;
            data[idx + 2] = 0;
        }
    }
    __device__ void processBlendedScanLines(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cnt = (x + y) % 3;
        int r = (params.sumR + y) % 256;
        data[idx + cnt] = (unsigned char)fminf(255.0f, data[idx + cnt] + r);
    }
    __device__ void processGradientStripes(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.frame_count % 3;
        int count = y % 256;
        int count_i = (params.sumR + params.sumG) % 256;
        data[idx + offset] = (unsigned char)fminf(255.0f, data[idx + offset] + count);
        data[idx + (2 - offset)] = (unsigned char)fmaxf(0.0f, data[idx + (2 - offset)] - count_i);
    }
    __device__ void processXorSine(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        int val0 = (params.sumB % 10) + 1;
        int val1 = (params.sumG % 10) + 1;
        int val2 = (params.sumR % 10) + 1;
        data[idx] ^= (unsigned char)(sinf(data[idx]) * val0 * pos);
        data[idx + 1] ^= (unsigned char)(sinf(data[idx + 1]) * val1 * pos);
        data[idx + 2] ^= (unsigned char)(sinf(data[idx + 2]) * val2 * pos);
    }
    __device__ void processCircular(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pos = 1.0f + 2.0f * sinf(params.frame_count * 0.1f);
        float rad = 50.0f + (params.frame_count % 50);
        float deg = (x + y) * 0.1f;
        int X_color = (int)(rad * cosf(deg));
        int Y_color = (int)(rad * sinf(deg));
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + pos * X_color);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * pos);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + pos * Y_color);
    }
    __device__ void processRandomPixels(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + r * 255);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + gpu_rand(x + 1, y, params.seed) * 255);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + gpu_rand(x, y + 1, params.seed) * 255);
    }
    __device__ void processDarkRandomPixels(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_v = 1 + (params.sumR % 255);
        float r0 = gpu_rand(x, y, params.seed) * max_v;
        float r1 = gpu_rand(x + 1, y, params.seed) * max_v;
        float r2 = gpu_rand(x, y + 1, params.seed) * max_v;
        data[idx] = (unsigned char)fminf(255.0f, (data[idx] + r0) / 4);
        data[idx + 1] = (unsigned char)fminf(255.0f, (data[idx + 1] + r1) / 4);
        data[idx + 2] = (unsigned char)fminf(255.0f, (data[idx + 2] + r2) / 4);
    }
    __device__ void processBars(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int start = params.frame_count % 3;
        int index = (start + x / 3) % 3;
        data[idx + index] = 255;
    }
    __device__ void processNegativeByRow(int x, int y, unsigned char* data, size_t step) {
        int idx = y * step + x * 4;
        if (y % 2 == 0) {
            data[idx] = ~data[idx];
            data[idx + 1] = ~data[idx + 1];
            data[idx + 2] = ~data[idx + 2];
        }
    }
    __device__ void processXorScale(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int scale0 = (params.sumB + params.frame_count) % 256;
        int scale1 = (params.sumG + params.frame_count) % 256;
        int scale2 = (params.sumR + params.frame_count) % 256;
        data[idx] ^= scale0;
        data[idx + 1] ^= scale1;
        data[idx + 2] ^= scale2;
    }
    __device__ void processSelfAlphaRGB(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        int index = params.frame_count % 4;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        switch (index) {
            case 0:
                data[idx] = (unsigned char)fminf(255.0f, b * alpha);
                data[idx + 1] = (unsigned char)fminf(255.0f, (b + g) * alpha);
                data[idx + 2] = (unsigned char)fminf(255.0f, (b + g + r) * alpha);
                break;
            case 1:
                data[idx + 2] = (unsigned char)fminf(255.0f, b * alpha);
                data[idx + 1] = (unsigned char)fminf(255.0f, (b + g) * alpha);
                data[idx] = (unsigned char)fminf(255.0f, (b + g + r) * alpha);
                break;
            case 2:
                data[idx + 1] = (unsigned char)fminf(255.0f, b * alpha);
                data[idx] = (unsigned char)fminf(255.0f, (b + g) * alpha);
                data[idx + 2] = (unsigned char)fminf(255.0f, (b + g + r) * alpha);
                break;
            case 3:
                data[idx] ^= (unsigned char)(b * alpha);
                data[idx + 1] ^= (unsigned char)((b + g) * alpha);
                data[idx + 2] ^= (unsigned char)((b + g + r) * alpha);
                break;
        }
    }
    __device__ void processBitwiseXorStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha1 = 1.0f + 0.3f * sinf(params.frame_count * 0.1f);
        float alpha2 = 1.0f + 0.5f * sinf(params.frame_count * 0.15f);
        float alpha3 = 1.0f + 0.7f * sinf(params.frame_count * 0.2f);
        int index = params.frame_count % 3;
        switch (index) {
            case 0:
                data[idx] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha1);
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] * alpha2);
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * alpha3);
                break;
            case 1:
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha1);
                data[idx] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] * alpha2);
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * alpha3);
                break;
            case 2:
                data[idx + 1] = (unsigned char)fminf(255.0f, data[idx] + data[idx] * alpha1);
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 1] + data[idx + 1] * alpha2);
                data[idx] = (unsigned char)fminf(255.0f, data[idx + 2] + data[idx + 2] * alpha3);
                break;
        }
    }
    __device__ void processOrStrobe(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.05f);
        unsigned char* prev = allFrames[0];
        unsigned char randB = (params.sumB + params.frame_count) % 256;
        unsigned char randG = (params.sumG + params.frame_count * 2) % 256;
        unsigned char randR = (params.sumR + params.frame_count * 3) % 256;
        if (prev) {
            data[idx] = (unsigned char)((data[idx] | randB | prev[idx]) * alpha);
            data[idx + 1] = (unsigned char)((data[idx + 1] | randG | prev[idx + 1]) * alpha);
            data[idx + 2] = (unsigned char)((data[idx + 2] | randR | prev[idx + 2]) * alpha);
        } else {
            data[idx] = (unsigned char)(data[idx] | randB);
            data[idx + 1] = (unsigned char)(data[idx + 1] | randG);
            data[idx + 2] = (unsigned char)(data[idx + 2] | randR);
        }
    }
    __device__ void processDivideAndIncH(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int row_x = width / 255;
        int row_y = height / 255;
        if (row_x == 0) row_x = 1;
        if (row_y == 0) row_y = 1;
        int inc_x = (x / row_x) % 256;
        int inc_y = (y / row_y) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + inc_x);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + inc_y);
    }
    __device__ void processDivideAndIncW(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int row_x = width / 255;
        int row_y = height / 255;
        if (row_x == 0) row_x = 1;
        if (row_y == 0) row_y = 1;
        int inc_x = (y / row_x) % 256;
        int inc_y = (x / row_y) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + inc_x);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + inc_y);
    }
    __device__ void processRandomIncrease(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char r0 = (params.sumB + params.frame_count * 17) % 256;
        unsigned char r1 = (params.sumG + params.frame_count * 23) % 256;
        unsigned char r2 = (params.sumR + params.frame_count * 31) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + r0);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + r1);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + r2);
    }
    __device__ void processSelfAlphaScaleBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.05f;
        unsigned int val = 0;
        for (int j = 0; j < 3; ++j) {
            val += (unsigned char)(data[idx + j] * (alpha + 1));
            data[idx + j] = data[idx + j] ^ (unsigned char)val;
        }
    }
    __device__ void processFadeBars(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.1f;
        unsigned char ch0 = (params.sumB + x) % 256;
        unsigned char ch1 = (params.sumG + x * 2) % 256;
        unsigned char ch2 = (params.sumR + x * 3) % 256;
        data[idx] = (unsigned char)((data[idx] ^ ch0) * alpha);
        data[idx + 1] = (unsigned char)((data[idx + 1] ^ ch1) * alpha);
        data[idx + 2] = (unsigned char)((data[idx + 2] ^ ch2) * alpha);
    }
    __device__ void processStrobeXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        data[idx] ^= (unsigned char)params.sumB;
        data[idx + 1] ^= (unsigned char)params.sumG;
        data[idx + 2] ^= (unsigned char)params.sumR;
    }
    __device__ void processBlank(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.1f;
        bool color_switch = (params.frame_count % 2) == 0;
        unsigned char val[3];
        for (int j = 0; j < 3; ++j) {
            val[j] = (unsigned char)((alpha * data[idx + j]) / (2 - j + 1));
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + (val[2 - j] / (j + 1)));
            if (color_switch) data[idx + j] = ~data[idx + j];
        }
    }
    __device__ void processColorVariableBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb0 = (params.sumB + params.frame_count * 10) % 256;
        int rgb1 = (params.sumG + params.frame_count * 10) % 256;
        int rgb2 = (params.sumR + params.frame_count * 10) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + rgb0);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + rgb1);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + rgb2);
    }
    __device__ void processColorXorBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb0 = (params.sumB + params.frame_count * 5) % 256;
        int rgb1 = (params.sumG + params.frame_count * 7) % 256;
        int rgb2 = (params.sumR + params.frame_count * 11) % 256;
        data[idx] ^= (unsigned char)rgb0;
        data[idx + 1] ^= (unsigned char)rgb1;
        data[idx + 2] ^= (unsigned char)rgb2;
    }
    __device__ void processColorAddBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        int rgb0 = (int)(params.sumB + r * 50) % 256;
        int rgb1 = (int)(params.sumG + r * 50) % 256;
        int rgb2 = (int)(params.sumR + r * 50) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] + rgb0);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + rgb1);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + rgb2);
    }
    __device__ void processSurroundingPixels(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        if (x >= width - 1 || y >= height - 1) return;
        int idx = y * step + x * 4;
        int idx1 = y * step + (x + 1) * 4;
        int idx2 = (y + 1) * step + x * 4;
        int idx3 = (y + 1) * step + (x + 1) * 4;
        for (int j = 0; j < 3; ++j) {
            int sum = data[idx + j] + data[idx1 + j] + data[idx2 + j] + data[idx3 + j];
            int avg = sum / 4;
            data[idx + j] ^= (unsigned char)avg;
        }
    }
    __device__ void processSurroundingPixelsAlpha(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        if (x >= width - 1 || y >= height - 1) return;
        int idx = y * step + x * 4;
        int idx1 = y * step + (x + 1) * 4;
        int idx2 = (y + 1) * step + x * 4;
        int idx3 = (y + 1) * step + (x + 1) * 4;
        float alpha = 1.0f + (params.frame_count % 24) * 0.05f;
        for (int j = 0; j < 3; ++j) {
            int sum = data[idx + j] + data[idx1 + j] + data[idx2 + j] + data[idx3 + j];
            int avg = sum / 3;
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (alpha + 1) + avg * alpha);
        }
    }
    __device__ void processDarkModBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb0 = (params.sumB + params.frame_count * 5) % 256;
        int rgb1 = (params.sumG + params.frame_count * 5) % 256;
        int rgb2 = (params.sumR + params.frame_count * 5) % 256;
        for (int j = 0; j < 3; ++j) {
            unsigned char val = (j == 0) ? rgb0 : (j == 1 ? rgb1 : rgb2);
            if (data[idx + j] == 0) data[idx + j] = 1;
            data[idx + j] = (val ^ data[idx + j]);
            if (data[idx + j] == 0) data[idx + j] = 1;
            data[idx + j] = val % data[idx + j];
        }
    }
    __device__ void processIncreaseDecreaseGamma(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int light = 1 + (params.frame_count % 10);
        float gamma = 1.0f + light * 0.1f;
        for (int j = 0; j < 3; ++j) {
            float normalized = data[idx + j] / 255.0f;
            data[idx + j] = (unsigned char)(powf(normalized, gamma) * 255.0f);
        }
    }
    __device__ void processBlendChannelXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        int cur_color = (params.sumR + params.frame_count) % 100;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * alpha) ^ cur_color;
        }
    }
    __device__ void processIncDifference(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + prev[idx + j]);
            }
        }
    }
    __device__ void processIncDifferenceAlpha(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)fminf(255.0f, (data[idx + j] + prev[idx + j]) * alpha);
            }
        }
    }
    __device__ void processMirrorXorAlpha(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha0 = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        float alpha1 = 1.0f + 0.5f * sinf(params.frame_count * 0.15f + 1.0f);
        float alpha2 = 1.0f + 0.5f * sinf(params.frame_count * 0.2f + 2.0f);
        int mx = width - x - 1;
        int my = height - y - 1;
        if (mx < 0) mx = 0;
        if (my < 0) my = 0;
        int idx0 = my * step + mx * 4;
        int idx1 = my * step + x * 4;
        int idx2 = y * step + mx * 4;
        unsigned char colorB = (params.sumB + params.frame_count) % 256;
        unsigned char colorG = (params.sumG + params.frame_count) % 256;
        unsigned char colorR = (params.sumR + params.frame_count) % 256;
        for (int j = 0; j < 3; ++j) {
            unsigned char p = (unsigned char)(data[idx + j] * alpha0);
            unsigned char v0 = (unsigned char)(data[idx0 + j] * alpha0);
            unsigned char v1 = (unsigned char)(data[idx1 + j] * alpha1);
            unsigned char v2 = (unsigned char)(data[idx2 + j] * alpha2);
            unsigned char col = (j == 0) ? colorB : (j == 1 ? colorG : colorR);
            data[idx + j] = (p ^ v0 ^ v1 ^ v2) ^ col;
        }
    }
    __device__ void processIntertwinedMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mx = width - x - 1;
        int my = height - y - 1;
        if (mx < 0) mx = 0;
        if (my < 0) my = 0;
        int idx0 = my * step + mx * 4;
        int idx1 = my * step + x * 4;
        int idx2 = y * step + mx * 4;
        int lines = y % 4;
        for (int j = 0; j < 3; ++j) {
            unsigned char val;
            switch (lines) {
                case 0: val = data[idx0 + j]; break;
                case 1: val = data[idx1 + j]; break;
                case 2: val = data[idx2 + j]; break;
                default: val = data[idx + j]; break;
            }
            data[idx + j] = val;
        }
    }
    __device__ void processColorFadeFilter(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float fade = 0.5f + 0.5f * sinf(params.frame_count * 0.05f);
        int rgb0 = (params.sumB + params.frame_count * 3) % 256;
        int rgb1 = (params.sumG + params.frame_count * 5) % 256;
        int rgb2 = (params.sumR + params.frame_count * 7) % 256;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * fade + rgb0 * (1 - fade));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * fade + rgb1 * (1 - fade));
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * fade + rgb2 * (1 - fade));
    }
    __device__ void processColorChannelMoveUpAndDown(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channel = params.frame_count % 3;
        int dir = (params.frame_count / 3) % 2;
        int shift = 10 + (params.frame_count % 20);
        if (dir == 0) {
            data[idx + channel] = (unsigned char)fminf(255.0f, data[idx + channel] + shift);
        } else {
            data[idx + channel] = (unsigned char)fmaxf(0.0f, data[idx + channel] - shift);
        }
    }
    __device__ void processMedianStrobe(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sw = params.frame_count % 2;
        unsigned char* hist = allFrames[params.frame_count % params.numFrames];
        if (sw == 0 && hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = hist[idx + j];
            }
        }
    }
    __device__ void processRGBBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float scale = 1.5f;
        int sum0 = data[idx];
        int sum1 = data[idx + 1];
        int sum2 = data[idx + 2];
        for (int q = 0; q < 3; ++q) {
            sum0 = (int)(sum0 / scale);
            sum1 = (int)(sum1 / scale);
            sum2 = (int)(sum2 / scale);
        }
        data[idx] ^= (unsigned char)sum0;
        data[idx + 1] ^= (unsigned char)sum1;
        data[idx + 2] ^= (unsigned char)sum2;
    }
    __device__ void processBGRBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        int sum0 = b + b + b;
        int sum1 = g + g + g;
        int sum2 = r + r + r;
        data[idx] = r ^ (unsigned char)sum0;
        data[idx + 1] = g ^ (unsigned char)sum1;
        data[idx + 2] = b ^ (unsigned char)sum2;
    }
    __device__ void processFlipAlphaBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fx = width - x - 1;
        int fy = height - y - 1;
        if (fx < 0) fx = 0;
        if (fy < 0) fy = 0;
        int fidx = fy * step + fx * 4;
        float alpha = 0.5f + 0.3f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (1 - alpha) + data[fidx + j] * alpha);
        }
    }
    __device__ void processRandomFlipFilter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mode = (params.seed + x + y) % 3;
        int fx = (mode == 0 || mode == 2) ? (width - x - 1) : x;
        int fy = (mode == 1 || mode == 2) ? (height - y - 1) : y;
        if (fx < 0) fx = 0;
        if (fy < 0) fy = 0;
        int fidx = fy * step + fx * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((data[idx + j] + data[fidx + j]) / 2);
        }
    }
    __device__ void processSelfScaleByFrame(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float scale = 1.0f + (params.frame_count % 50) * 0.02f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * scale);
        }
    }
    __device__ void processAlphaBlendMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mx = width - x - 1;
        int my = height - y - 1;
        if (mx < 0) mx = 0;
        if (my < 0) my = 0;
        int midx = my * step + mx * 4;
        float alpha = 0.5f + 0.3f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (1 - alpha) + data[midx + j] * alpha);
        }
    }
    __device__ void processTwistedVision(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = 2 + (params.frame_count % 20);
        int cx = x + offset;
        if (cx >= 0 && cx < width) {
            int cidx = y * step + cx * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= data[cidx + j];
            }
        }
    }
    __device__ void processTruncateColor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_val = 128;  
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] > max_val) {
                data[idx + j] = max_val;
            }
        }
    }
    __device__ void processTruncateVariable(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_val = 150 + (int)(50.0f * sinf(params.frame_count * 0.05f));
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] > max_val) {
                data[idx + j] = max_val;
            }
        }
    }
    __device__ void processTruncateVariableScale(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_val = 75 + (int)(100.0f * sinf(params.frame_count * 0.05f));
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] > max_val) {
                data[idx + j] = data[idx + j] ^ max_val;
            }
        }
    }
    __device__ void processXorFade(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 0.1f + 0.9f * (0.5f + 0.5f * sinf(params.frame_count * 0.01f));
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= (unsigned char)(alpha * data[idx + j]);
        }
    }
    __device__ void processSineValue(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 1.0f + 0.5f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) {
            int val = 1 + (int)(sinf(alpha) * (x + y));
            data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * (val % 256)));
        }
    }
    __device__ void processFadeRtoGtoB(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int colors[3];
        colors[0] = (params.sumB + params.frame_count) % 256;
        colors[1] = (params.sumG + params.frame_count * 2) % 256;
        colors[2] = (params.sumR + params.frame_count * 3) % 256;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((data[idx + j] * 0.5f) + (0.5f * colors[j]));
        }
    }
    __device__ void processFadeRandomChannel(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb = (params.seed + params.frame_count) % 3;
        int color = (params.sumB + params.sumG + params.sumR + params.frame_count) % 256;
        data[idx + rgb] = (unsigned char)((data[idx + rgb] * 0.5f) + (0.5f * color));
    }
    __device__ void processVariableLines(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = (params.seed + y * 17) % width;
        int src_x = (x + offset) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((0.7f * data[idx + j]) + (0.3f * data[src_idx + j]));
        }
    }
    __device__ void processVariableLinesVertical(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = (params.seed + x * 13) % height;
        int src_y = (y + offset) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((0.7f * data[idx + j]) + (0.3f * data[src_idx + j]));
        }
    }
    __device__ void processRowMedianBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fidx = (y + params.frame_count) % params.numFrames;
        unsigned char* hist = allFrames[fidx];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * hist[idx + j]));
            }
        }
    }
    __device__ void processMirrorReverseColor(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int px = width - x - 1;
        int py = height - y - 1;
        if (px < 0) px = 0;
        if (py < 0) py = 0;
        int idx0 = py * step + x * 4;
        int idx1 = py * step + px * 4;
        int idx2 = y * step + px * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + (2 - j)] = (unsigned char)((data[idx + j] * 0.25f) + (data[idx0 + j] * 0.25f) + (data[idx1 + j] * 0.25f) + (data[idx2 + j] * 0.25f));
        }
    }
    __device__ void processPsychoticVision(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 0.1f + 0.9f * (0.5f + 0.5f * sinf(params.frame_count * 0.1f));
        int speed = 1 + (params.frame_count % 5);
        for (int j = 0; j < 3; ++j) {
            int col = (params.sumB + j * 50 + params.frame_count * speed) % 200;
            data[idx + j] = (unsigned char)((0.7f * data[idx + j]) - (0.3f * alpha * col));
        }
    }
    __device__ void processPixelGlitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.1f) {
            int glitch = (int)(gpu_rand(x + 1, y + 1, params.seed) * 255);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= glitch;
            }
        }
    }
    __device__ void processStaticGlitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.05f) {
            int noise = (int)(r * 255 * 10) % 256;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = noise;
            }
        }
    }
    __device__ void processWavePattern(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((x + params.frame_count) * 0.05f) * 30.0f;
        int src_y = (int)(y + wave) % width;
        if (src_y < 0) src_y = 0;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + fabs(wave));
        }
    }
    __device__ void processWavePatternXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((x + y + params.frame_count) * 0.05f) * 50.0f;
        int wave_int = (int)fabs(wave) % 256;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= wave_int;
        }
    }
    __device__ void processDiagonalXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int diag = (x + y + params.frame_count) % 256;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= diag;
        }
    }
    __device__ void processRGBShiftBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = params.frame_count % 3;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        switch (shift) {
            case 0: data[idx] = g; data[idx + 1] = r; data[idx + 2] = b; break;
            case 1: data[idx] = r; data[idx + 1] = b; data[idx + 2] = g; break;
            case 2: data[idx] = (b + g) / 2; data[idx + 1] = (g + r) / 2; data[idx + 2] = (r + b) / 2; break;
        }
    }
    __device__ void processChannelShuffle(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int pos = (x + y) % 3;
        unsigned char temp = data[idx + pos];
        data[idx + pos] = data[idx + (pos + 1) % 3];
        data[idx + (pos + 1) % 3] = temp;
    }
    __device__ void processChannelShuffleRand(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        int pos = (int)(r * 3) % 3;
        unsigned char temp = data[idx + pos];
        data[idx + pos] = data[idx + (pos + 1) % 3];
        data[idx + (pos + 1) % 3] = temp;
    }
    __device__ void processPixelCounter(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = (x + y + params.frame_count) % 256;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + count * 0.3f);
        }
    }
    __device__ void processPixelCounterXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = (x + y + params.frame_count) % 256;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= count;
        }
    }
    __device__ void processRowColorBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int color = (y + params.frame_count) % 256;
        int channel = y % 3;
        data[idx + channel] = (unsigned char)((data[idx + channel] * 0.5f) + (color * 0.5f));
    }
    __device__ void processColumnColorBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int color = (x + params.frame_count) % 256;
        int channel = x % 3;
        data[idx + channel] = (unsigned char)((data[idx + channel] * 0.5f) + (color * 0.5f));
    }
    __device__ void processCheckerboardXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int size = 8 + (params.frame_count % 24);
        int check = ((x / size) + (y / size)) % 2;
        if (check) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= 128;
            }
        }
    }
    __device__ void processCheckerboardBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int size = 16 + (params.frame_count % 32);
        int check = ((x / size) + (y / size)) % 2;
        unsigned char* hist = allFrames[params.frame_count % params.numFrames];
        if (check && hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * hist[idx + j]));
            }
        }
    }
    __device__ void processSineWaveDistort(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf(y * 0.1f + params.frame_count * 0.1f) * 10.0f;
        int src_x = ((int)(x + wave) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((0.6f * data[idx + j]) + (0.4f * data[src_idx + j]));
        }
    }
    __device__ void processCosineWaveDistort(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = cosf(x * 0.1f + params.frame_count * 0.1f) * 10.0f;
        int src_y = ((int)(y + wave) % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((0.6f * data[idx + j]) + (0.4f * data[src_idx + j]));
        }
    }
    __device__ void processSinCosBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float sinVal = sinf((x + params.frame_count) * 0.05f) * 50.0f;
        float cosVal = cosf((y + params.frame_count) * 0.05f) * 50.0f;
        data[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx] + sinVal));
        data[idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + 1] + cosVal));
        data[idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + 2] + (sinVal + cosVal) / 2));
    }
    __device__ void processPixelReverseXor(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rx = width - x - 1;
        int ry = height - y - 1;
        if (rx < 0) rx = 0;
        if (ry < 0) ry = 0;
        int ridx = ry * step + rx * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= data[ridx + j];
        }
    }
    __device__ void processLinesAcrossX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0) {
            int prev_idx = y * step + (x - 1) * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= data[prev_idx + j];
            }
        }
    }
    __device__ void processXorLineX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (params.sumR + params.sumG + params.sumB) / 3;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((0.5f * (data[idx + j] ^ avg)) + (0.5f * data[idx + j]));
        }
    }
    __device__ void processAlphaComponentIncrease(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channel = params.frame_count % 3;
        float alpha = 0.1f + 0.9f * fabsf(sinf(params.frame_count * 0.01f));
        data[idx + channel] ^= (unsigned char)(alpha * data[idx + channel]);
    }
    __device__ void processExpandContract(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int nw = width + (int)(sinf(params.frame_count * 0.02f) * width);
        if (nw < width) nw = width;
        int src_x = (x * nw) / width;
        if (src_x >= 0 && src_x < width) {
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processLongLines(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, 0, params.seed);
        if (r < 0.007f) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 0;
            }
        }
    }
    __device__ void processTearRight(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_off = 50 + (params.frame_count % 350);
        float r = gpu_rand(y, params.seed, params.frame_count);
        int offset = 25 + (int)(r * max_off);
        int total_size = width + offset;
        int src_x = (x * total_size) / width;
        if (src_x >= width) src_x = width - 1;
        if (src_x < 0) src_x = 0;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processTearDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_off = 50 + (params.frame_count % 350);
        float r = gpu_rand(x, params.seed, params.frame_count);
        int offset = 25 + (int)(r * max_off);
        int total_size = height + offset;
        int src_y = (y * total_size) / height;
        if (src_y >= height) src_y = height - 1;
        if (src_y < 0) src_y = 0;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processDistortionByRow(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int nw = height / 16 + (y % (height - height / 16));
        int src_y = (y * nw) / height;
        if (src_y >= height) src_y = height - 1;
        if (src_y < 0) src_y = 0;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processDistortionByCol(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int nh = width / 16 + (x % (width - width / 16));
        int src_x = (x * nh) / width;
        if (src_x >= width) src_x = width - 1;
        if (src_x < 0) src_x = 0;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processAlternateAlpha(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb = params.frame_count % 3;
        float alpha = 0.1f + 0.4f * fabsf(sinf(params.frame_count * 0.01f));
        data[idx + rgb] = (unsigned char)((data[idx + rgb] * 0.5f) * alpha);
    }
    __device__ void processDiagSquareRGB(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb = params.frame_count % 3;
        int sq_x = x / 32;
        int sq_y = y / 32;
        int offset = (sq_x + sq_y) % params.numFrames;
        unsigned char* hist = allFrames[offset];
        if (hist) {
            data[idx + rgb] = hist[idx + rgb];
        }
    }
    __device__ void processShiftPixelsRGB(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb = params.frame_count % 3;
        int offset = 1 + (params.frame_count % (width / 16));
        int src_x = (x + offset) % width;
        int src_idx = y * step + src_x * 4;
        data[idx + rgb] += data[src_idx + rgb];
    }
    __device__ void processColorWaveTrailsRGB(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rgb = params.frame_count % 3;
        int fi = y % params.numFrames;
        unsigned char* hist = allFrames[fi];
        if (hist) {
            data[idx + rgb] = (unsigned char)((0.3f * data[idx + rgb]) + (0.7f * hist[idx + rgb]));
        }
    }
    __device__ void processProperTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* hist = allFrames[params.numFrames > 7 ? 7 : params.numFrames - 1];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * hist[idx + j]));
            }
        }
    }
    __device__ void processXorLag(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fi = params.numFrames > 31 ? 31 : params.numFrames - 1;
        unsigned char* hist = allFrames[fi];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= hist[idx + j];
            }
        }
    }
    __device__ void processPixelateBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int wait = 1 + ((x + y + params.seed) % 50);
        int fi = ((x * params.sh + y) / wait) % params.numFrames;
        unsigned char* hist = allFrames[fi];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * hist[idx + j]));
            }
        }
    }
    __device__ void processDiagPixel(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int off = y % params.numFrames;
        unsigned char* hist = allFrames[off];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * hist[idx + j]));
            }
        }
    }
    __device__ void processDiagPixelY(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int off = (x + y) % params.numFrames;
        unsigned char* hist = allFrames[off];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = hist[idx + j];
            }
        }
    }
    __device__ void processExpandLeftRight(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int off = (params.frame_count + y) % width;
        int fi = y % params.numFrames;
        unsigned char* hist = allFrames[fi];
        if (hist) {
            int src_x = (x + off) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * hist[src_idx + j]));
            }
        }
    }
    __device__ void processDiagSquare(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sq_x = x / 32;
        int sq_y = y / 32;
        int offset = (sq_x + sq_y) % params.numFrames;
        unsigned char* hist = allFrames[offset];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = hist[idx + j];
            }
        }
    }
    __device__ void processHorizontalColorOffset(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int coffset = (y / 50) % 3;
        float color = 0.3f + 0.7f * fabsf(sinf((x + params.frame_count) * 0.01f));
        data[idx + coffset] = (unsigned char)(data[idx + coffset] * color);
    }
    __device__ void processPrevFrameNotEqual(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            int threshold = 25;
            for (int j = 0; j < 3; ++j) {
                if (data[idx + j] >= prev[idx + j] && data[idx + j] <= prev[idx + j] + threshold) {
                    data[idx + j] = prev[idx + j];
                }
            }
        }
    }
    __device__ void processBlackLines(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int depth = params.seed % 50;
        bool on = ((y / 50) % 2) == 0;
        if (on && x < depth) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 0;
            }
        } else if (on && x + depth < width) {
            int src_idx = y * step + (x + depth) * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * data[src_idx + j]));
            }
        }
    }
    __device__ void processDizzyMode(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((0.5f * data[idx + j]) + (0.5f * prev[idx + j]));
            }
        }
    }
    __device__ void processGhostShift(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int i = 0; i < params.numFrames && i < 8; ++i) {
            unsigned char* hist = allFrames[i];
            if (hist) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)((0.6f * data[idx + j]) + (0.4f * hist[idx + j]));
                }
            }
        }
    }
    __device__ void processRGBSplitFilter(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int f0 = params.numFrames > 15 ? 15 : params.numFrames - 1;
        int f1 = params.numFrames > 7 ? 7 : params.numFrames / 2;
        int f2 = 0;
        unsigned char* rgb0 = allFrames[f0];
        unsigned char* rgb1 = allFrames[f1];
        unsigned char* rgb2 = allFrames[f2];
        if (rgb0) data[idx] = rgb0[idx];
        if (rgb1) data[idx + 1] = rgb1[idx + 1];
        if (rgb2) data[idx + 2] = rgb2[idx + 2];
    }
    __device__ void processPixelateRect(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.1f) {
            int fi = (int)(r * params.numFrames * 10) % params.numFrames;
            unsigned char* hist = allFrames[fi];
            if (hist) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = hist[idx + j];
                }
            }
        }
    }
    __device__ void processCollectionXor4(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 0.1f + 0.9f * fabsf(sinf(params.frame_count * 0.01f));
        int colors[3] = {0, 0, 0};
        int count = params.numFrames < 4 ? params.numFrames : 4;
        for (int q = 0; q < count; ++q) {
            unsigned char* hist = allFrames[q];
            if (hist) {
                for (int j = 0; j < 3; ++j) {
                    colors[j] += hist[idx + j];
                }
            }
        }
        for (int j = 0; j < 3; ++j) {
            int avg_color = colors[j] / count;
            int pix_val = data[idx + j] ^ avg_color;
            data[idx + j] = (unsigned char)((1.0f - alpha) * data[idx + j] + alpha * pix_val);
        }
    }
    __device__ void processRectangleSpin(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.05f) {
            int src_y = (int)(r * height * 20) % height;
            int src_idx = src_y * step + x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processRectanglePlotXY(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.1f) {
            int src_x = (int)(r * width * 10) % width;
            int src_y = (int)(r * height * 10) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= data[src_idx + j];
            }
        }
    }
    __device__ void processShiftLinesDown(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shiftAmount = (params.frame_count / 4) % height;
        int src_y = (y + shiftAmount) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processPictureStretch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float stretch = 1.0f + 0.5f * sinf(params.frame_count * 0.02f);
        int src_x = (int)(x / stretch) % width;
        if (src_x < 0) src_x += width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processPictureStretchPieces(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int piece = y / 32;
        float stretch = 1.0f + 0.3f * sinf((params.frame_count + piece * 10) * 0.03f);
        int src_x = (int)(x / stretch) % width;
        if (src_x < 0) src_x += width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processVisualSnow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.1f) {
            unsigned char snow = (unsigned char)(r * 255 * 10);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.7f * data[idx + j] + 0.3f * snow);
            }
        }
    }
    __device__ void processVisualSnowX2(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.2f) {
            unsigned char snow = (unsigned char)(r * 255 * 5);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * snow);
            }
        }
    }
    __device__ void processLineGlitch(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 32, y, params.seed);
        if (r < 0.15f) {
            int offset = (int)(r * 100) % 50;
            int src_x = (x + offset) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processSlitReverse64(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int slit = y / 64;
        if (slit % 2 == 0) {
            int src_x = width - 1 - x;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
            }
        }
    }
    __device__ void processSlitReverse64_Increase(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int slitSize = 32 + (params.frame_count % 64);
        int slit = y / slitSize;
        if (slit % 2 == 0) {
            int src_x = width - 1 - x;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
            }
        }
    }
    __device__ void processSlitStretch(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int slit = y / 32;
        float stretch = 1.0f + 0.5f * (slit % 4) * 0.25f;
        int src_x = (int)(x * stretch) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processLineLeftRight(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int dir = (y / 16) % 2;
        int offset = params.frame_count % width;
        int src_x = dir == 0 ? (x + offset) % width : (x - offset + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processLineLeftRightResize(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineSize = 8 + (params.frame_count / 8) % 32;
        int dir = (y / lineSize) % 2;
        int offset = params.frame_count % width;
        int src_x = dir == 0 ? (x + offset) % width : (x - offset + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processRGBLineTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channel = y % 3;
        unsigned char* prev = allFrames[0];
        if (prev) {
            data[idx + channel] = (unsigned char)(0.6f * data[idx + channel] + 0.4f * prev[idx + channel]);
        }
    }
    __device__ void processRGBCollectionBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            int fi = j % params.numFrames;
            unsigned char* hist = allFrames[fi];
            if (hist) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * hist[idx + j]);
            }
        }
    }
    __device__ void processRGBCollectionIncrease(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 0.1f + 0.9f * fabsf(sinf(params.frame_count * 0.02f));
        for (int j = 0; j < 3; ++j) {
            int fi = (j + params.frame_count) % params.numFrames;
            unsigned char* hist = allFrames[fi];
            if (hist) {
                data[idx + j] = (unsigned char)((1.0f - alpha) * data[idx + j] + alpha * hist[idx + j]);
            }
        }
    }
    __device__ void processRGBLongTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            int fi = (j * 5) % params.numFrames;
            unsigned char* hist = allFrames[fi];
            if (hist) {
                data[idx + j] = (unsigned char)(0.3f * data[idx + j] + 0.7f * hist[idx + j]);
            }
        }
    }
    __device__ void processFadeRGB_Speed(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float speedR = fabsf(sinf(params.frame_count * 0.05f));
        float speedG = fabsf(sinf(params.frame_count * 0.07f));
        float speedB = fabsf(sinf(params.frame_count * 0.09f));
        data[idx] = (unsigned char)(data[idx] * speedB);
        data[idx + 1] = (unsigned char)(data[idx + 1] * speedG);
        data[idx + 2] = (unsigned char)(data[idx + 2] * speedR);
    }
    __device__ void processRGBStrobeTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 3;
        unsigned char* prev = allFrames[0];
        if (prev) {
            data[idx + strobe] = (unsigned char)(0.5f * data[idx + strobe] + 0.5f * prev[idx + strobe]);
        }
    }
    __device__ void processBoxGlitch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 64, y / 64, params.seed);
        if (r < 0.1f) {
            int offset_x = (int)(r * 100) % 32;
            int offset_y = (int)(r * 50) % 16;
            int src_x = (x + offset_x) % width;
            int src_y = (y + offset_y) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processVerticalPictureDistort(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float distort = sinf(x * 0.05f + params.frame_count * 0.02f) * 10;
        int src_y = ((int)(y + distort) % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processShortTrail(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.8f * data[idx + j] + 0.2f * prev[idx + j]);
            }
        }
    }
    __device__ void processDiagInward(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        int dx = x - cx;
        int dy = y - cy;
        float dist = sqrtf((float)(dx * dx + dy * dy));
        float factor = 0.5f + 0.5f * sinf(dist * 0.05f - params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processDiagSquareInward(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sq = params.square_size > 0 ? params.square_size : 32;
        int sx = (x / sq) * sq + sq / 2;
        int sy = (y / sq) * sq + sq / 2;
        int cx = width / 2;
        int cy = height / 2;
        float dist = sqrtf((float)((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)));
        float factor = 0.5f + 0.5f * sinf(dist * 0.02f - params.frame_count * 0.03f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processDiagSquareInwardResize(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sq = 16 + (params.frame_count / 4) % 48;
        int sx = (x / sq) * sq + sq / 2;
        int sy = (y / sq) * sq + sq / 2;
        int cx = width / 2;
        int cy = height / 2;
        float dist = sqrtf((float)((sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)));
        float factor = 0.5f + 0.5f * sinf(dist * 0.02f - params.frame_count * 0.03f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processPictureShiftDownRight(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param1 % 100;
        int src_x = (x - offset + width) % width;
        int src_y = (y - offset + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processFlipPictureShift(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flip = (params.frame_count / 30) % 4;
        int src_x = x, src_y = y;
        if (flip == 1) { src_x = width - 1 - x; }
        else if (flip == 2) { src_y = height - 1 - y; }
        else if (flip == 3) { src_x = width - 1 - x; src_y = height - 1 - y; }
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processRGBWideTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            int fi = (j * 8) % params.numFrames;
            unsigned char* hist = allFrames[fi];
            if (hist) {
                data[idx + j] = (unsigned char)(0.4f * data[idx + j] + 0.6f * hist[idx + j]);
            }
        }
    }
    __device__ void processLineInLineOut_Increase(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineSize = 4 + (params.frame_count / 4) % 32;
        int segment = x / lineSize;
        float factor = (segment % 2 == 0) ? 1.0f + params.alpha * 0.5f : 1.0f - params.alpha * 0.3f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processLineInLineOut2_Increase(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineSize = 4 + (params.frame_count / 4) % 32;
        int segment = y / lineSize;
        float factor = (segment % 2 == 0) ? 1.0f + params.alpha * 0.5f : 1.0f - params.alpha * 0.3f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processLineInLineOut3_Increase(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineSize = 8 + (params.frame_count / 8) % 24;
        int segX = x / lineSize;
        int segY = y / lineSize;
        float factor = ((segX + segY) % 2 == 0) ? 1.2f : 0.8f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processSquareByRow2Plus(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sq = params.square_size > 0 ? params.square_size : 16;
        int row = y / sq;
        int offset = (row * 5) % width;
        int src_x = (x + offset) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processFrameSep(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int half = width / 2;
        int fi = (x < half) ? 0 : (params.numFrames > 1 ? 1 : 0);
        unsigned char* hist = allFrames[fi];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = hist[idx + j];
            }
        }
    }
    __device__ void processFrameSep2(int x, int y, unsigned char* data, unsigned char** allFrames, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int half = height / 2;
        int fi = (y < half) ? 0 : (params.numFrames > 1 ? 1 : 0);
        unsigned char* hist = allFrames[fi];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = hist[idx + j];
            }
        }
    }
    __device__ void processFrameStopStart(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cycle = (params.frame_count / 30) % 2;
        if (cycle == 0) {
            unsigned char* hist = allFrames[0];
            if (hist) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = hist[idx + j];
                }
            }
        }
    }
    __device__ void processOutOfOrder(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fi = (x ^ y ^ params.seed) % params.numFrames;
        unsigned char* hist = allFrames[fi];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * hist[idx + j]);
            }
        }
    }
    __device__ void processTrackingDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int track = (params.frame_count * 2) % height;
        if (y == track) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255;
            }
        }
    }
    __device__ void processTrackingDownBlend(int x, int y, unsigned char* data, unsigned char** allFrames, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int track = (params.frame_count * 2) % height;
        if (abs(y - track) < 5) {
            unsigned char* prev = allFrames[0];
            if (prev) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)(0.3f * data[idx + j] + 0.7f * prev[idx + j]);
                }
            }
        }
    }
    __device__ void processTrackingRev(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int track = height - 1 - ((params.frame_count * 2) % height);
        if (y == track) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255 - data[idx + j];
            }
        }
    }
    __device__ void processTrackingMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int track = (params.frame_count * 2) % height;
        if (abs(y - track) < 10) {
            int src_x = width - 1 - x;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processBlockPixels(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bx = x / 8;
        int by = y / 8;
        float r = gpu_rand(bx, by, params.seed);
        if (r < 0.3f) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + r));
            }
        }
    }
    __device__ void processFrameChop(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int chop = width / 4;
        int section = x / chop;
        int fi = section % params.numFrames;
        unsigned char* hist = allFrames[fi];
        if (hist) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = hist[idx + j];
            }
        }
    }
    __device__ void processYLineDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int linePos = (params.frame_count + x) % height;
        if (y == linePos) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255;
            }
        }
    }
    __device__ void processYLineDownBlend(int x, int y, unsigned char* data, unsigned char** allFrames, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int linePos = (params.frame_count + x) % height;
        if (abs(y - linePos) < 3) {
            unsigned char* prev = allFrames[0];
            if (prev) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * prev[idx + j]);
                }
            }
        }
    }
    __device__ void processSquareDiff1(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sq = 16;
        int sx = (x / sq) * sq;
        int sy = (y / sq) * sq;
        int src_idx = sy * step + sx * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - prev[src_idx + j]);
                data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + diff * 0.5f);
            }
        }
    }
    __device__ void processLineAcrossXNew(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineX = (params.frame_count * 3) % width;
        if (abs(x - lineX) < 2) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255 - data[idx + j];
            }
        }
    }
    __device__ void processColorGlitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.05f) {
            int channel = (int)(r * 60) % 3;
            data[idx + channel] = 255 - data[idx + channel];
        }
    }
    __device__ void processPixelShiftUp(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param1 % height;
        int src_y = (y + offset) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processPixelShiftDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param1 % height;
        int src_y = (y - offset + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processPixelShiftLeft(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param2 % width;
        int src_x = (x + offset) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processPixelShiftRight(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param2 % width;
        int src_x = (x - offset + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processPixelShiftDiagonal(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset_x = params.int_param2 % width;
        int offset_y = params.int_param1 % height;
        int src_x = (x + offset_x) % width;
        int src_y = (y + offset_y) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processWaveBlend(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((x + params.frame_count) * 0.05f) * 20.0f;
        int src_x = ((int)(x + wave) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processWaveBlendX2(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float waveX = sinf((x + params.frame_count) * 0.05f) * 20.0f;
        float waveY = cosf((y + params.frame_count) * 0.05f) * 20.0f;
        int src_x = ((int)(x + waveX) % width + width) % width;
        int src_y = ((int)(y + waveY) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processSineWaveBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float factor = 0.5f + 0.5f * sinf((x + y + params.frame_count) * 0.03f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processCosineWaveBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float factor = 0.5f + 0.5f * cosf((x - y + params.frame_count) * 0.03f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processSpiralWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float angle = atan2f(dy, dx) + params.frame_count * 0.02f;
        float dist = sqrtf(dx * dx + dy * dy);
        float factor = 0.5f + 0.5f * sinf(angle * 3.0f + dist * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processRadialBlur(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float blur = fminf(dist * 0.01f, 0.5f);
        int src_x = (int)(cx + dx * (1.0f - blur));
        int src_y = (int)(cy + dy * (1.0f - blur));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processZoomBlur(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float zoom = 1.0f + 0.1f * sinf(params.frame_count * 0.05f);
        int src_x = (int)(cx + (x - cx) / zoom);
        int src_y = (int)(cy + (y - cy) / zoom);
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processRotateBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float angle = params.frame_count * 0.01f;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        int src_x = (int)(cx + dx * cosf(angle) - dy * sinf(angle));
        int src_y = (int)(cy + dx * sinf(angle) + dy * cosf(angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processMirrorWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((y + params.frame_count) * 0.1f) * 30.0f;
        int src_x = (int)(width - 1 - x + wave);
        src_x = (src_x % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processMirrorWaveX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((x + params.frame_count) * 0.08f) * 20.0f;
        int src_x = (int)(width - 1 - x + wave);
        src_x = (src_x % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.6f * data[idx + j] + 0.4f * data[src_idx + j]);
        }
    }
    __device__ void processMirrorWaveY(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((y + params.frame_count) * 0.08f) * 20.0f;
        int src_y = (int)(height - 1 - y + wave);
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.6f * data[idx + j] + 0.4f * data[src_idx + j]);
        }
    }
    __device__ void processPixelDrift(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drift_x = sinf(y * 0.02f + params.frame_count * 0.03f) * 10.0f;
        float drift_y = cosf(x * 0.02f + params.frame_count * 0.03f) * 10.0f;
        int src_x = ((int)(x + drift_x) % width + width) % width;
        int src_y = ((int)(y + drift_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[src_idx + j] + 0.5f * prev[idx + j]);
            }
        }
    }
    __device__ void processPixelDriftX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drift = sinf(y * 0.03f + params.frame_count * 0.05f) * 15.0f;
        int src_x = ((int)(x + drift) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processPixelDriftY(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drift = cosf(x * 0.03f + params.frame_count * 0.05f) * 15.0f;
        int src_y = ((int)(y + drift) % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processColorPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * pulse);
        }
    }
    __device__ void processColorPulseRGB(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pulseR = 0.5f + 0.5f * sinf(params.frame_count * 0.08f);
        float pulseG = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
        float pulseB = 0.5f + 0.5f * sinf(params.frame_count * 0.12f);
        data[idx] = (unsigned char)(data[idx] * pulseB);
        data[idx + 1] = (unsigned char)(data[idx + 1] * pulseG);
        data[idx + 2] = (unsigned char)(data[idx + 2] * pulseR);
    }
    __device__ void processColorPulseXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int pulse = (int)(128 * sinf(params.frame_count * 0.05f));
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= (unsigned char)abs(pulse);
        }
    }
    __device__ void processGlitchBlock(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bx = x / 32;
        int by = y / 32;
        float r = gpu_rand(bx, by, params.seed);
        if (r < 0.15f) {
            int offset = (int)(r * 200) - 100;
            int src_x = (x + offset) % width;
            if (src_x < 0) src_x += width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processGlitchBlockXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bx = x / 32;
        int by = y / 32;
        float r = gpu_rand(bx, by, params.seed);
        if (r < 0.2f) {
            unsigned char xv = (unsigned char)(r * 255);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= xv;
            }
        }
    }
    __device__ void processGlitchLine(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(0, y, params.seed);
        if (r < 0.1f) {
            int offset = (int)(r * 100) % 50;
            int src_x = (x + offset) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processGlitchLineX(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, 0, params.seed);
        if (r < 0.1f) {
            int offset = (int)(r * 100) % 50;
            int src_y = (y + offset) % height;
            int src_idx = src_y * step + x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processNoiseBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        unsigned char noise = (unsigned char)(r * 255);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.8f * data[idx + j] + 0.2f * noise);
        }
    }
    __device__ void processNoiseBlendX2(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        unsigned char noise = (unsigned char)(r * 255);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.6f * data[idx + j] + 0.4f * noise);
        }
    }
    __device__ void processNoiseXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.3f) {
            unsigned char noise = (unsigned char)(r * 255 * 3.33f);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= noise;
            }
        }
    }
    __device__ void processChannelShift(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = params.frame_count % 30;
        int src_x_r = (x + shift) % width;
        int src_x_b = (x - shift + width) % width;
        unsigned char r_val = data[y * step + src_x_r * 4 + 2];
        unsigned char b_val = data[y * step + src_x_b * 4];
        data[idx + 2] = r_val;
        data[idx] = b_val;
    }
    __device__ void processChannelShiftX(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = params.frame_count % 20;
        int src_y_r = (y + shift) % height;
        int src_y_b = (y - shift + height) % height;
        unsigned char r_val = data[src_y_r * step + x * 4 + 2];
        unsigned char b_val = data[src_y_b * step + x * 4];
        data[idx + 2] = r_val;
        data[idx] = b_val;
    }
    __device__ void processChannelRotate(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rot = params.frame_count % 3;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        if (rot == 0) { data[idx] = r; data[idx + 1] = b; data[idx + 2] = g; }
        else if (rot == 1) { data[idx] = g; data[idx + 1] = r; data[idx + 2] = b; }
    }
    __device__ void processDiagonalStretch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float stretch = 1.0f + 0.3f * sinf(params.frame_count * 0.03f);
        int diag = x + y;
        int src_x = (int)(x + (diag % 50) * (stretch - 1.0f)) % width;
        if (src_x < 0) src_x += width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processDiagonalStretchX(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float stretch = 1.0f + 0.3f * cosf(params.frame_count * 0.03f);
        int diag = x - y;
        int src_y = (int)(y + (abs(diag) % 50) * (stretch - 1.0f)) % height;
        if (src_y < 0) src_y += height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processDiagonalMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x + y < (width + height) / 2) {
            int src_x = y % width;
            int src_y = x % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
            }
        }
    }
    __device__ void processSquareWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 32;
        int phase = (x + params.frame_count) / period;
        float factor = (phase % 2 == 0) ? 1.2f : 0.8f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processSquareWaveX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 32;
        int phase = (y + params.frame_count) / period;
        float factor = (phase % 2 == 0) ? 1.2f : 0.8f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processSquareWaveBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 48;
        int phase = (x + y + params.frame_count) / period;
        unsigned char* src = (phase % 2 == 0) ? allFrames[0] : NULL;
        if (src) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * src[idx + j]);
            }
        }
    }
    __device__ void processTriangleWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 64;
        int pos = (x + params.frame_count) % period;
        float factor = (pos < period / 2) ? (float)pos / (period / 2) : 2.0f - (float)pos / (period / 2);
        factor = 0.5f + 0.5f * factor;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processTriangleWaveBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 64;
        int pos = (y + params.frame_count) % period;
        float alpha = (pos < period / 2) ? (float)pos / (period / 2) : 2.0f - (float)pos / (period / 2);
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - alpha) + prev[idx + j] * alpha);
            }
        }
    }
    __device__ void processSawtoothWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 64;
        int pos = (x + params.frame_count) % period;
        float factor = 0.5f + 0.5f * ((float)pos / period);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processSawtoothWaveBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 64;
        int pos = (y + params.frame_count) % period;
        float alpha = (float)pos / period;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - alpha) + prev[idx + j] * alpha);
            }
        }
    }
    __device__ void processPulseWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 48;
        int pos = (x + y + params.frame_count) % period;
        float factor = (pos < period / 4) ? 1.3f : 0.9f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processPulseWaveBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 48;
        int pos = (x + y + params.frame_count) % period;
        float blend = (pos < period / 4) ? 0.8f : 0.3f;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * blend + prev[idx + j] * (1.0f - blend));
            }
        }
    }
    __device__ void processStepWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int steps = 4;
        int period = 64;
        int pos = (x + params.frame_count) % period;
        int stepVal = (pos * steps) / period;
        float factor = 0.5f + 0.5f * ((float)stepVal / steps);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processStepWaveBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int steps = 4;
        int period = 64;
        int pos = (y + params.frame_count) % period;
        int stepVal = (pos * steps) / period;
        float blend = (float)stepVal / steps;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + prev[idx + j] * blend);
            }
        }
    }
    __device__ void processRippleEffect(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float ripple = sinf(dist * 0.1f - params.frame_count * 0.1f) * 10.0f;
        int src_x = (int)(x + ripple * dx / (dist + 1.0f));
        int src_y = (int)(y + ripple * dy / (dist + 1.0f));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processRippleEffectX2(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float ripple = sinf(dist * 0.15f - params.frame_count * 0.15f) * 15.0f;
        int src_x = (int)(x + ripple * dx / (dist + 1.0f));
        int src_y = (int)(y + ripple * dy / (dist + 1.0f));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processShockWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float wave_pos = (params.frame_count * 3) % (int)(sqrtf((float)(width * width + height * height)));
        float wave_width = 30.0f;
        if (fabsf(dist - wave_pos) < wave_width) {
            float offset = sinf((dist - wave_pos) * 0.2f) * 10.0f;
            int src_x = (int)(x + offset * dx / (dist + 1.0f));
            int src_y = (int)(y + offset * dy / (dist + 1.0f));
            src_x = (src_x % width + width) % width;
            src_y = (src_y % height + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processShockWaveBlend(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float wave_pos = (params.frame_count * 3) % (int)(sqrtf((float)(width * width + height * height)));
        float wave_width = 40.0f;
        unsigned char* prev = allFrames[0];
        if (fabsf(dist - wave_pos) < wave_width && prev) {
            float blend = 1.0f - fabsf(dist - wave_pos) / wave_width;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + prev[idx + j] * blend);
            }
        }
    }
    __device__ void processTwistEffect(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float max_dist = sqrtf((float)(cx * cx + cy * cy));
        float twist = (1.0f - dist / max_dist) * sinf(params.frame_count * 0.02f) * 1.5f;
        float angle = atan2f(dy, dx) + twist;
        int src_x = (int)(cx + dist * cosf(angle));
        int src_y = (int)(cy + dist * sinf(angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processTwistEffectBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float max_dist = sqrtf((float)(cx * cx + cy * cy));
        float twist = (1.0f - dist / max_dist) * sinf(params.frame_count * 0.03f) * 2.0f;
        float angle = atan2f(dy, dx) + twist;
        int src_x = (int)(cx + dist * cosf(angle));
        int src_y = (int)(cy + dist * sinf(angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processFishEye(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx) / cx;
        float dy = (float)(y - cy) / cy;
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < 1.0f) {
            float factor = powf(dist, 1.5f) / dist;
            int src_x = (int)(cx + dx * factor * cx);
            int src_y = (int)(cy + dy * factor * cy);
            src_x = (src_x % width + width) % width;
            src_y = (src_y % height + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processFishEyeBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx) / cx;
        float dy = (float)(y - cy) / cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float strength = 0.5f + 0.5f * sinf(params.frame_count * 0.05f);
        if (dist < 1.0f) {
            float factor = powf(dist, 1.0f + strength) / dist;
            int src_x = (int)(cx + dx * factor * cx);
            int src_y = (int)(cy + dy * factor * cy);
            src_x = (src_x % width + width) % width;
            src_y = (src_y % height + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
            }
        }
    }
    __device__ void processKaleidoscope(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float angle = atan2f(dy, dx);
        float dist = sqrtf(dx * dx + dy * dy);
        int segments = 6;
        float segment_angle = (2.0f * 3.14159265f) / segments;
        float new_angle = fmodf(fabsf(angle), segment_angle);
        if ((int)(angle / segment_angle) % 2 == 1) new_angle = segment_angle - new_angle;
        new_angle += params.frame_count * 0.01f;
        int src_x = (int)(cx + dist * cosf(new_angle));
        int src_y = (int)(cy + dist * sinf(new_angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processKaleidoscopeBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float angle = atan2f(dy, dx);
        float dist = sqrtf(dx * dx + dy * dy);
        int segments = 8;
        float segment_angle = (2.0f * 3.14159265f) / segments;
        float new_angle = fmodf(fabsf(angle), segment_angle);
        if ((int)(angle / segment_angle) % 2 == 1) new_angle = segment_angle - new_angle;
        new_angle += params.frame_count * 0.02f;
        int src_x = (int)(cx + dist * cosf(new_angle));
        int src_y = (int)(cy + dist * sinf(new_angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processTunnelEffect(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float angle = atan2f(dy, dx);
        float dist = sqrtf(dx * dx + dy * dy) + 1.0f;
        float tunnel_x = angle / 3.14159265f * width;
        float tunnel_y = 100.0f / dist * height + params.frame_count * 2.0f;
        int src_x = ((int)tunnel_x % width + width) % width;
        int src_y = ((int)tunnel_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processTunnelEffectBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float angle = atan2f(dy, dx);
        float dist = sqrtf(dx * dx + dy * dy) + 1.0f;
        float tunnel_x = angle / 3.14159265f * width;
        float tunnel_y = 100.0f / dist * height + params.frame_count * 2.0f;
        int src_x = ((int)tunnel_x % width + width) % width;
        int src_y = ((int)tunnel_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.6f * data[idx + j] + 0.4f * data[src_idx + j]);
        }
    }
    __device__ void processVortexEffect(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float angle = atan2f(dy, dx) + dist * 0.01f * sinf(params.frame_count * 0.03f);
        int src_x = (int)(cx + dist * cosf(angle));
        int src_y = (int)(cy + dist * sinf(angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processVortexEffectBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        float dx = (float)(x - cx);
        float dy = (float)(y - cy);
        float dist = sqrtf(dx * dx + dy * dy);
        float angle = atan2f(dy, dx) + dist * 0.015f * sinf(params.frame_count * 0.04f);
        int src_x = (int)(cx + dist * cosf(angle));
        int src_y = (int)(cy + dist * sinf(angle));
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    __device__ void processColorDrift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drift = sinf(params.frame_count * 0.02f);
        int shift = (int)(drift * 50);
        data[idx] = (unsigned char)((data[idx] + shift + 256) % 256);
        data[idx + 1] = (unsigned char)((data[idx + 1] - shift + 256) % 256);
        data[idx + 2] = (unsigned char)((data[idx + 2] + shift / 2 + 256) % 256);
    }
    __device__ void processColorDriftX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drift = cosf(params.frame_count * 0.025f);
        int shift = (int)(drift * 40);
        data[idx] = (unsigned char)((data[idx] - shift + 256) % 256);
        data[idx + 1] = (unsigned char)((data[idx + 1] + shift + 256) % 256);
        data[idx + 2] = (unsigned char)((data[idx + 2] - shift / 2 + 256) % 256);
    }
    __device__ void processRGBShift(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = 3 + (params.frame_count % 10);
        int r_x = (x + shift) % width;
        int b_x = (x - shift + width) % width;
        unsigned char r = data[y * step + r_x * 4 + 2];
        unsigned char b = data[y * step + b_x * 4];
        data[idx + 2] = r;
        data[idx] = b;
    }
    __device__ void processRGBShiftX(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = 3 + (params.frame_count % 10);
        int r_y = (y + shift) % height;
        int b_y = (y - shift + height) % height;
        unsigned char r = data[r_y * step + x * 4 + 2];
        unsigned char b = data[b_y * step + x * 4];
        data[idx + 2] = r;
        data[idx] = b;
    }
    __device__ void processChromaticAberration(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = 5;
        int r_x = (x + shift) % width;
        int g_x = x;
        int b_x = (x - shift + width) % width;
        data[idx + 2] = data[y * step + r_x * 4 + 2];
        data[idx + 1] = data[y * step + g_x * 4 + 1];
        data[idx] = data[y * step + b_x * 4];
    }
    __device__ void processChromaticAberrationX(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = 5;
        int r_x = (x + shift) % width;
        int r_y = (y + shift) % height;
        int b_x = (x - shift + width) % width;
        int b_y = (y - shift + height) % height;
        data[idx + 2] = data[r_y * step + r_x * 4 + 2];
        data[idx] = data[b_y * step + b_x * 4];
    }
    __device__ void processPosterize(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int levels = 4;
        float factor = 255.0f / levels;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(roundf(data[idx + j] / factor) * factor);
        }
    }
    __device__ void processPosterizeBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int levels = 4 + (params.frame_count % 4);
        float factor = 255.0f / levels;
        for (int j = 0; j < 3; ++j) {
            unsigned char poster = (unsigned char)(roundf(data[idx + j] / factor) * factor);
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * poster);
        }
    }
    __device__ void processSolarize(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int threshold = 128;
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] > threshold) {
                data[idx + j] = 255 - data[idx + j];
            }
        }
    }
    __device__ void processSolarizeBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int threshold = 100 + (params.frame_count % 100);
        for (int j = 0; j < 3; ++j) {
            unsigned char solar = (data[idx + j] > threshold) ? (255 - data[idx + j]) : data[idx + j];
            data[idx + j] = (unsigned char)(0.6f * data[idx + j] + 0.4f * solar);
        }
    }
    __device__ void processGammaBright(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float gamma = 0.7f;
        for (int j = 0; j < 3; ++j) {
            float normalized = data[idx + j] / 255.0f;
            data[idx + j] = (unsigned char)(powf(normalized, gamma) * 255.0f);
        }
    }
    __device__ void processGammaDark(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float gamma = 1.5f;
        for (int j = 0; j < 3; ++j) {
            float normalized = data[idx + j] / 255.0f;
            data[idx + j] = (unsigned char)(powf(normalized, gamma) * 255.0f);
        }
    }
    __device__ void processContrastBoost(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float contrast = 1.5f;
        for (int j = 0; j < 3; ++j) {
            float val = (data[idx + j] - 128) * contrast + 128;
            data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    }
    __device__ void processContrastReduce(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float contrast = 0.6f;
        for (int j = 0; j < 3; ++j) {
            float val = (data[idx + j] - 128) * contrast + 128;
            data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    }
    __device__ void processEdgeGlowBlend(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && prev) {
            int sum = 0;
            for (int j = 0; j < 3; ++j) {
                int gx = data[(y - 1) * step + (x + 1) * 4 + j] - data[(y - 1) * step + (x - 1) * 4 + j]
                       + 2 * data[y * step + (x + 1) * 4 + j] - 2 * data[y * step + (x - 1) * 4 + j]
                       + data[(y + 1) * step + (x + 1) * 4 + j] - data[(y + 1) * step + (x - 1) * 4 + j];
                sum += abs(gx);
            }
            int edge = sum / 3;
            float blend = fminf(1.0f, edge / 100.0f);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + prev[idx + j] * blend);
            }
        }
    }
    __device__ void processFrameBlendMulti(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = min(params.numFrames, 4);
        for (int j = 0; j < 3; ++j) {
            int sum = data[idx + j];
            for (int f = 0; f < count; ++f) {
                if (allFrames[f]) sum += allFrames[f][idx + j];
            }
            data[idx + j] = (unsigned char)(sum / (count + 1));
        }
    }
    __device__ void processFrameBlendMultiX(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = min(params.numFrames, 8);
        float weights[9] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f};
        float total_weight = weights[0];
        for (int j = 0; j < 3; ++j) {
            float sum = data[idx + j] * weights[0];
            for (int f = 0; f < count; ++f) {
                if (allFrames[f]) {
                    sum += allFrames[f][idx + j] * weights[f + 1];
                    if (j == 0) total_weight += weights[f + 1];
                }
            }
            data[idx + j] = (unsigned char)(sum / total_weight);
        }
    }
    __device__ void processAcidTrailsBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            float factor = 0.5f + 0.3f * sinf(params.frame_count * 0.05f);
            for (int j = 0; j < 3; ++j) {
                int val = (int)(data[idx + j] * factor + prev[idx + j] * (1.0f - factor));
                data[idx + j] = (unsigned char)fminf(255.0f, (float)val);
            }
        }
    }
    __device__ void processAcidGlitchX(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 16, y / 16, params.seed);
        if (r < 0.15f) {
            int shift_x = (int)(r * 50) - 25;
            int shift_y = (int)(r * 30) - 15;
            int src_x = (x + shift_x + width) % width;
            int src_y = (y + shift_y + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processAlphaXorBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                unsigned char xored = data[idx + j] ^ prev[idx + j];
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + xored * 0.5f);
            }
        }
    }
    __device__ void processAlphaXorBlendDouble(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev1 = allFrames[0];
        unsigned char* prev2 = (params.numFrames > 1) ? allFrames[1] : NULL;
        if (prev1 && prev2) {
            for (int j = 0; j < 3; ++j) {
                unsigned char xored = data[idx + j] ^ prev1[idx + j] ^ prev2[idx + j];
                data[idx + j] = (unsigned char)(data[idx + j] * 0.3f + xored * 0.7f);
            }
        }
    }
    __device__ void processAndOrXorStrobeScale(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mode = params.frame_count % 3;
        float scale = 0.8f + 0.4f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) {
            unsigned char val = data[idx + j];
            if (mode == 0) val = val & (unsigned char)(params.sumR);
            else if (mode == 1) val = val | (unsigned char)(params.sumG);
            else val = val ^ (unsigned char)(params.sumB);
            data[idx + j] = (unsigned char)fminf(255.0f, val * scale);
        }
    }
    __device__ void processAveragePixelsXorBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)((data[idx + j] ^ avg) * 0.5f + data[idx + j] * 0.5f);
        }
    }
    __device__ void processBitwiseRotateBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = params.frame_count % 8;
        for (int j = 0; j < 3; ++j) {
            unsigned char val = data[idx + j];
            unsigned char rotated = (val << shift) | (val >> (8 - shift));
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + rotated * 0.5f);
        }
    }
    __device__ void processBitwiseRotateDiffBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        int shift = params.frame_count % 8;
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                unsigned char val = data[idx + j];
                unsigned char rotated = (val << shift) | (val >> (8 - shift));
                int diff = abs(rotated - prev[idx + j]);
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + diff * 0.5f);
            }
        }
    }
    __device__ void processBitwiseXorScaleBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float scale = 0.5f + 0.5f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            unsigned char val = data[idx + j];
            unsigned char xored = val ^ (unsigned char)(params.sumR + j * 30);
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f - scale) + xored * scale);
        }
    }
    __device__ void processBlackAndWhiteStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 10;
        if (strobe < 3) {
            int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)avg;
            }
        }
    }
    __device__ void processBlendAlphaXorScale(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 0.5f + 0.5f * cosf(params.frame_count * 0.03f);
        for (int j = 0; j < 3; ++j) {
            unsigned char xored = data[idx + j] ^ (unsigned char)(params.sumB);
            data[idx + j] = (unsigned char)(data[idx + j] * alpha + xored * (1.0f - alpha));
        }
    }
    __device__ void processBlendBurredXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                unsigned char blurred = (unsigned char)((data[idx + j] + prev[idx + j]) / 2);
                data[idx + j] = blurred ^ data[idx + j];
            }
        }
    }
    __device__ void processBlendCombinedXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char combined = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + (data[idx + j] ^ combined) * 0.4f);
        }
    }
    __device__ void processBlendIncreaseRGB(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float inc = (params.frame_count % 128) / 128.0f;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * (1.0f + inc * 0.3f));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (1.0f + inc * 0.2f));
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (1.0f + inc * 0.1f));
    }
    __device__ void processBlendThreeXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* p1 = allFrames[0];
        unsigned char* p2 = (params.numFrames > 1) ? allFrames[1] : NULL;
        if (p1 && p2) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (data[idx + j] ^ p1[idx + j] ^ p2[idx + j]);
            }
        }
    }
    __device__ void processBlurDistortionBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float distort = sinf((x + params.frame_count) * 0.05f) * 5.0f;
        int src_y = (int)(y + distort);
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processColorAccumulate(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float factor = (params.frame_count % 64) / 64.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + factor * 0.5f));
        }
    }
    __device__ void processColorAccumulateBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        float factor = (params.frame_count % 64) / 64.0f;
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int accum = (int)(data[idx + j] * (1.0f + factor * 0.3f));
                data[idx + j] = (unsigned char)fminf(255.0f, accum * 0.5f + prev[idx + j] * 0.5f);
            }
        }
    }
    __device__ void processColorAccumulateXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float factor = (params.frame_count % 64) / 64.0f;
        for (int j = 0; j < 3; ++j) {
            unsigned char accum = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + factor * 0.5f));
            data[idx + j] = data[idx + j] ^ accum;
        }
    }
    __device__ void processColorChannelBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channel = params.frame_count % 3;
        float blend = 0.5f + 0.5f * sinf(params.frame_count * 0.08f);
        unsigned char avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        data[idx + channel] = (unsigned char)(data[idx + channel] * blend + avg * (1.0f - blend));
    }
    __device__ void processColorChannelXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channel = params.frame_count % 3;
        data[idx + channel] ^= (unsigned char)(params.sumR);
    }
    __device__ void processColorCollectionEnergy(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - prev[idx + j]);
                int energy = data[idx + j] + diff / 2;
                data[idx + j] = (unsigned char)fminf(255.0f, (float)energy);
            }
        }
    }
    __device__ void processColorCollectionWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((x + y + params.frame_count) * 0.03f);
        for (int j = 0; j < 3; ++j) {
            float factor = 0.8f + 0.2f * wave * (j + 1) / 3.0f;
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processColorFadeXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float fade = (params.frame_count % 128) / 128.0f;
        for (int j = 0; j < 3; ++j) {
            unsigned char faded = (unsigned char)(data[idx + j] * fade);
            data[idx + j] = data[idx + j] ^ faded;
        }
    }
    __device__ void processColorIntensityBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int intensity = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float factor = intensity / 255.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * factor));
        }
    }
    __device__ void processColorIntensityXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int intensity = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= (unsigned char)intensity;
        }
    }
    __device__ void processColorMoveBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.frame_count % 32;
        int src_x = (x + offset) % width;
        int src_y = (y + offset) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + data[src_idx + j] * 0.4f);
        }
    }
    __device__ void processColorPixelBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.3f) {
            unsigned char avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.7f + avg * 0.3f);
            }
        }
    }
    __device__ void processColorPixelXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.25f) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= (unsigned char)(r * 255);
            }
        }
    }
    __device__ void processColorScaleBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float scale = 0.7f + 0.6f * sinf(params.frame_count * 0.04f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * scale);
        }
    }
    __device__ void processColorWaveXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((x + y + params.frame_count) * 0.05f);
        unsigned char xorVal = (unsigned char)(128 + 127 * wave);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= xorVal;
        }
    }
    __device__ void processCosineMultiplyBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cos_val = cosf((x + y) * 0.02f + params.frame_count * 0.03f);
        float factor = 0.7f + 0.3f * cos_val;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    __device__ void processDarkModBlendXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            unsigned char dark = (unsigned char)(data[idx + j] * 0.5f);
            data[idx + j] = (data[idx + j] ^ dark);
        }
    }
    __device__ void processDifferenceBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - prev[idx + j]);
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + diff * 0.5f);
            }
        }
    }
    __device__ void processDifferenceXorBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - prev[idx + j]);
                data[idx + j] = (unsigned char)((data[idx + j] ^ diff) * 0.7f + data[idx + j] * 0.3f);
            }
        }
    }
    __device__ void processDistortBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float dx = sinf(y * 0.03f + params.frame_count * 0.05f) * 10.0f;
        float dy = cosf(x * 0.03f + params.frame_count * 0.05f) * 10.0f;
        int src_x = ((int)(x + dx) % width + width) % width;
        int src_y = ((int)(y + dy) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processDiamondPatternBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        int diamond = abs(x - cx) + abs(y - cy);
        float factor = 0.5f + 0.5f * sinf(diamond * 0.02f + params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processFadeBlendXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float fade = (params.frame_count % 256) / 256.0f;
        for (int j = 0; j < 3; ++j) {
            unsigned char faded = (unsigned char)(data[idx + j] * fade);
            data[idx + j] = (unsigned char)((data[idx + j] ^ faded) * 0.5f + data[idx + j] * 0.5f);
        }
    }
    __device__ void processFlashBlendXor(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flash = params.frame_count % 20;
        if (flash < 5) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] ^= 0xFF;
            }
        } else {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.9f);
            }
        }
    }
    __device__ void processGhostTrailsBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int count = min(params.numFrames, 4);
        for (int j = 0; j < 3; ++j) {
            float sum = data[idx + j];
            float weight = 1.0f;
            float total_weight = weight;
            for (int f = 0; f < count; ++f) {
                if (allFrames[f]) {
                    weight *= 0.7f;
                    sum += allFrames[f][idx + j] * weight;
                    total_weight += weight;
                }
            }
            data[idx + j] = (unsigned char)(sum / total_weight);
        }
    }
    __device__ void processAddInvert(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            int val = data[idx + j] + (255 - data[idx + j]) / 2;
            data[idx + j] = (unsigned char)fminf(255.0f, (float)val);
        }
    }
    __device__ void processAlphaBlendSimple(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            float alpha = params.alpha;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * alpha + prev[idx + j] * (1.0f - alpha));
            }
        }
    }
    __device__ void processAlphaBlendDoubleX(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* p1 = allFrames[0];
        unsigned char* p2 = (params.numFrames > 1) ? allFrames[1] : NULL;
        if (p1 && p2) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.34f + p1[idx + j] * 0.33f + p2[idx + j] * 0.33f);
            }
        }
    }
    __device__ void processAlphaStrobeBlendX(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        int strobe = params.frame_count % 6;
        if (prev) {
            float blend = (strobe < 3) ? 0.3f : 0.7f;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * blend + prev[idx + j] * (1.0f - blend));
            }
        }
    }
    __device__ void processBitwiseAndBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char mask = (unsigned char)(128 + 64 * sinf(params.frame_count * 0.05f));
        for (int j = 0; j < 3; ++j) {
            unsigned char anded = data[idx + j] & mask;
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + anded * 0.5f);
        }
    }
    __device__ void processBitwiseXorAverage(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= avg;
        }
    }
    __device__ void processBitwiseXorBlendX(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                unsigned char xored = data[idx + j] ^ prev[idx + j];
                data[idx + j] = (unsigned char)(xored * 0.6f + data[idx + j] * 0.4f);
            }
        }
    }
    __device__ void processBlackStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 8;
        if (strobe < 2) {
            data[idx] = data[idx + 1] = data[idx + 2] = 0;
        }
    }
    __device__ void processBlendAlphaXorX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float alpha = 0.5f + 0.5f * sinf(params.frame_count * 0.06f);
        for (int j = 0; j < 3; ++j) {
            unsigned char xored = data[idx + j] ^ (unsigned char)(params.sumB + j * 40);
            data[idx + j] = (unsigned char)(data[idx + j] * alpha + xored * (1.0f - alpha));
        }
    }
    __device__ void processBlendCombinedValuesX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int combined = (data[idx] + data[idx + 1] + data[idx + 2]);
        float factor = (combined % 256) / 255.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.6f + 0.4f * factor));
        }
    }
    __device__ void processBlendFor360(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float angle = atan2f((float)(y - height/2), (float)(x - width/2));
        float factor = 0.5f + 0.5f * sinf(angle * 4.0f + params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processBlendForward16(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int frame_idx = params.frame_count % min(params.numFrames, 16);
        unsigned char* frame = allFrames[frame_idx];
        if (frame) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + frame[idx + j] * 0.5f);
            }
        }
    }
    __device__ void processBlendForward32(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int frame_idx = params.frame_count % min(params.numFrames, 32);
        unsigned char* frame = allFrames[frame_idx];
        if (frame) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + frame[idx + j] * 0.4f);
            }
        }
    }
    __device__ void processBlendFromXtoY(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float blend = (params.frame_count % 256) / 255.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + (255 - data[idx + j]) * blend);
        }
    }
    __device__ void processBlendIncreaseX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float inc = 1.0f + (params.frame_count % 100) / 200.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * inc);
        }
    }
    __device__ void processBlendRedGreenBlue(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cycle = params.frame_count % 3;
        float factors[3] = {0.7f, 0.8f, 0.9f};
        factors[cycle] = 1.2f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factors[j]);
        }
    }
    __device__ void processBlendWithColorX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char colors[3] = {(unsigned char)params.sumR, (unsigned char)params.sumG, (unsigned char)params.sumB};
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.7f + colors[j] * 0.3f);
        }
    }
    __device__ void processBlendAngle(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float angle = atan2f((float)(y - height/2), (float)(x - width/2));
        float norm = (angle + 3.14159f) / (2.0f * 3.14159f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * norm));
        }
    }
    __device__ void processBlockScale(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block_x = x / 16;
        int block_y = y / 16;
        float scale = 0.5f + 0.5f * sinf((block_x + block_y + params.frame_count) * 0.2f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * scale);
        }
    }
    __device__ void processBlockStrobe(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block_x = x / 32;
        int block_y = y / 32;
        int strobe = (block_x + block_y + params.frame_count) % 4;
        if (strobe == 0) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(255 - data[idx + j]);
            }
        }
    }
    __device__ void processBlockXor(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block_x = x / 16;
        int block_y = y / 16;
        unsigned char xor_val = (unsigned char)((block_x ^ block_y ^ params.frame_count) % 256);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] ^= xor_val;
        }
    }
    __device__ void processBlockyTrails16(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block_y = y / 16;
        int frame_idx = block_y % min(params.numFrames, 8);
        unsigned char* frame = allFrames[frame_idx];
        if (frame) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + frame[idx + j] * 0.5f);
            }
        }
    }
    __device__ void processBlockyTrails32(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block_y = y / 32;
        int frame_idx = block_y % min(params.numFrames, 8);
        unsigned char* frame = allFrames[frame_idx];
        if (frame) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + frame[idx + j] * 0.4f);
            }
        }
    }
    __device__ void processBlurDistortionX(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float dx = sinf((y + params.frame_count) * 0.03f) * 8.0f;
        float dy = cosf((x + params.frame_count) * 0.03f) * 8.0f;
        int src_x = ((int)(x + dx) % width + width) % width;
        int src_y = ((int)(y + dy) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processCannyStrobe(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 4;
        if (strobe < 2 && x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            int gx = 0, gy = 0;
            for (int j = 0; j < 3; ++j) {
                int val_left = data[y * step + (x-1) * 4 + j];
                int val_right = data[y * step + (x+1) * 4 + j];
                int val_up = data[(y-1) * step + x * 4 + j];
                int val_down = data[(y+1) * step + x * 4 + j];
                gx += abs(val_right - val_left);
                gy += abs(val_down - val_up);
            }
            int edge = min(255, (gx + gy) / 3);
            data[idx] = data[idx + 1] = data[idx + 2] = (unsigned char)edge;
        }
    }
    __device__ void processColorFadeSlow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float fade = (params.frame_count % 512) / 512.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * sinf(fade * 6.28f + j * 2.0f)));
        }
    }
    __device__ void processColorFibonacci(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fib[8] = {1, 1, 2, 3, 5, 8, 13, 21};
        int fib_idx = (x + y + params.frame_count) % 8;
        float factor = 0.5f + 0.5f * (fib[fib_idx] / 21.0f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processCurtainEffect(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int curtain_pos = (params.frame_count * 4) % width;
        if (x < curtain_pos) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.3f);
            }
        }
    }
    __device__ void processDarkColorFibonacci(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fib[8] = {1, 1, 2, 3, 5, 8, 13, 21};
        int fib_idx = (x + y + params.frame_count) % 8;
        float factor = 0.3f + 0.4f * (fib[fib_idx] / 21.0f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processDarkColorsBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] > 128) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.6f);
            } else {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.9f);
            }
        }
    }
    __device__ void processEnergizeBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - prev[idx + j]);
                int energized = data[idx + j] + diff;
                data[idx + j] = (unsigned char)fminf(255.0f, (float)energized);
            }
        }
    }
    __device__ void processAverageLines(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int line_width = 8 + (params.frame_count % 16);
        int block = x / line_width;
        if (block % 2 == 0) {
            int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + avg * 0.5f);
            }
        }
    }
    __device__ void processAverageLinesBlendX(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        int line_width = 16;
        int block = x / line_width;
        if (prev && block % 2 == 0) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)((data[idx + j] + prev[idx + j]) / 2);
            }
        }
    }
    __device__ void processBlendRowAlpha(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float row_alpha = (float)(y % 64) / 64.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.6f + 0.4f * row_alpha));
        }
    }
    __device__ void processBlendInOut(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        float cycle = 0.5f + 0.5f * sinf(params.frame_count * 0.03f);
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * cycle + prev[idx + j] * (1.0f - cycle));
            }
        }
    }
    __device__ void processColorFlashIncreaseX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flash = params.frame_count % 30;
        float inc = 1.0f + (flash < 10 ? flash * 0.05f : 0.0f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * inc);
        }
    }
    __device__ void processColorIncreaseInOut(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cycle = 0.5f + 0.5f * sinf(params.frame_count * 0.04f);
        float inc = 0.8f + 0.4f * cycle;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * inc);
        }
    }
    __device__ void processColorLinesX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int line = y % 8;
        if (line < 4) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.7f);
            }
        }
    }
    __device__ void processColorMoveDownX(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.frame_count % height;
        int src_y = (y + offset) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processColorOrderSwapX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mode = params.frame_count % 6;
        unsigned char b = data[idx], g = data[idx + 1], r = data[idx + 2];
        switch (mode) {
            case 0: data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; break;
            case 1: data[idx] = g; data[idx + 1] = b; data[idx + 2] = r; break;
            case 2: data[idx] = b; data[idx + 1] = r; data[idx + 2] = g; break;
            case 3: data[idx] = r; data[idx + 1] = b; data[idx + 2] = g; break;
            case 4: data[idx] = g; data[idx + 1] = r; data[idx + 2] = b; break;
            default: break;
        }
    }
    __device__ void processColorPulseAlphaX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.08f);
        float alpha = params.alpha * pulse;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * alpha + data[idx + j] * (1.0f - alpha) * 0.5f);
        }
    }
    __device__ void processColorRowShiftX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = (y + params.frame_count) % 32;
        int src_x = (x + shift) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processColorShiftXorX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = params.frame_count % 8;
        for (int j = 0; j < 3; ++j) {
            unsigned char shifted = (data[idx + j] << shift) | (data[idx + j] >> (8 - shift));
            data[idx + j] ^= shifted;
        }
    }
    __device__ void processCopyXorAlphaX(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                unsigned char xored = data[idx + j] ^ prev[idx + j];
                data[idx + j] = (unsigned char)(xored * params.alpha + data[idx + j] * (1.0f - params.alpha));
            }
        }
    }
    __device__ void processCycleShiftRGBX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cycle = params.frame_count % 3;
        unsigned char temp[3] = {data[idx], data[idx + 1], data[idx + 2]};
        data[idx] = temp[(0 + cycle) % 3];
        data[idx + 1] = temp[(1 + cycle) % 3];
        data[idx + 2] = temp[(2 + cycle) % 3];
    }
    __device__ void processDarkNegateX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (avg < 100) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(255 - data[idx + j]);
            }
        }
    }
    __device__ void processDarkSelfAlphaX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float alpha = (avg < 128) ? 0.5f : 1.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * alpha);
        }
    }
    __device__ void processDiagonalGlitch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int diag = (x + y + params.frame_count) % 64;
        if (diag < 8) {
            int shift = diag * 2;
            int src_x = (x + shift) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processDigitalHaze(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float haze = 0.1f + 0.1f * sinf((x + y) * 0.02f + params.frame_count * 0.03f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - haze) + 200 * haze);
        }
    }
    __device__ void processDoubleXorBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* p1 = allFrames[0];
        unsigned char* p2 = (params.numFrames > 1) ? allFrames[1] : NULL;
        if (p1 && p2) {
            for (int j = 0; j < 3; ++j) {
                unsigned char xor1 = data[idx + j] ^ p1[idx + j];
                unsigned char xor2 = xor1 ^ p2[idx + j];
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + xor2 * 0.5f);
            }
        }
    }
    __device__ void processEchoBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int numEcho = min(params.numFrames, 4);
        for (int j = 0; j < 3; ++j) {
            float sum = data[idx + j];
            float weight = 1.0f;
            for (int f = 0; f < numEcho; ++f) {
                if (allFrames[f]) {
                    weight *= 0.6f;
                    sum += allFrames[f][idx + j] * weight;
                }
            }
            data[idx + j] = (unsigned char)(sum / (1.0f + weight * numEcho * 0.3f));
        }
    }
    __device__ void processElectricEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) {
                int dx = abs(data[y * step + (x+1) * 4 + j] - data[y * step + (x-1) * 4 + j]);
                int dy = abs(data[(y+1) * step + x * 4 + j] - data[(y-1) * step + x * 4 + j]);
                edge += dx + dy;
            }
            edge /= 6;
            if (edge > 30) {
                int pulse = (params.frame_count % 10 < 5) ? 255 : 180;
                data[idx] = (unsigned char)fminf(255.0f, data[idx] * 0.3f + pulse * 0.7f);
                data[idx + 1] = (unsigned char)(data[idx + 1] * 0.5f);
                data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 0.3f + pulse * 0.7f);
            }
        }
    }
    __device__ void processFlashColorStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 15;
        if (strobe < 3) {
            int channel = strobe % 3;
            data[idx + channel] = 255;
        }
    }
    __device__ void processFrameDiffXor(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - prev[idx + j]);
                data[idx + j] ^= (unsigned char)diff;
            }
        }
    }
    __device__ void processGhostMirror(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mirror_x = width - x - 1;
        int mirror_idx = y * step + mirror_x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + prev[mirror_idx + j] * 0.5f);
            }
        }
    }
    __device__ void processGlitchSort(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 32, y / 32, params.seed);
        if (r < 0.1f) {
            int offset = (int)(r * 20) - 10;
            int src_x = (x + offset + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processHeatWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((y + params.frame_count * 2) * 0.05f) * 5.0f;
        int src_x = ((int)(x + wave) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.3f + data[src_idx + j] * 0.7f);
        }
    }
    __device__ void processInterlaceBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev && y % 2 == 0) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + prev[idx + j] * 0.5f);
            }
        }
    }
    __device__ void processInvertStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 12;
        if (strobe < 4) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(255 - data[idx + j]);
            }
        }
    }
    __device__ void processKaleidoBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = width / 2;
        int cy = height / 2;
        int dx = x - cx;
        int dy = y - cy;
        float angle = atan2f((float)dy, (float)dx);
        float sector = fmodf(angle + 3.14159f, 3.14159f / 3.0f);
        float factor = 0.5f + 0.5f * cosf(sector * 6.0f + params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * factor);
        }
    }
    __device__ void processLightStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 8;
        if (strobe < 2) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * 1.5f);
            }
        }
    }
    __device__ void processLineGlitchX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(0, y, params.seed + params.frame_count);
        if (r < 0.05f) {
            int shift = (int)(r * 100) - 50;
            int src_x = (x + shift + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processMosaicBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block = 4 + (params.frame_count / 10) % 4;
        int bx = (x / block) * block;
        int by = (y / block) * block;
        int bidx = by * step + bx * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + data[bidx + j] * 0.4f);
        }
    }
    __device__ void processNegatePulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int pulse = (params.frame_count / 5) % 2;
        if (pulse == 1) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(255 - data[idx + j]);
            }
        }
    }
    __device__ void processOffsetGhost(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            int offset = params.frame_count % 8;
            int src_x = (x + offset) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.7f + prev[src_idx + j] * 0.3f);
            }
        }
    }
    __device__ void processPixelateWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf((y + params.frame_count * 2) * 0.05f) * 4.0f;
        int px = ((int)(x + wave) % width + width) % width;
        int pidx = y * step + px * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[pidx + j] * 0.5f);
        }
    }
    __device__ void processQuantizeBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int levels = 4 + (params.frame_count % 5);
        for (int j = 0; j < 3; ++j) {
            int val = (data[idx + j] * levels) / 256;
            data[idx + j] = (unsigned char)((val * 256) / levels);
        }
    }
    __device__ void processRandomLines(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.1f) {
            int line = y % (4 + (params.frame_count % 3));
            if (line == 0) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)(data[idx + j] * 0.5f);
                }
            }
        }
    }
    __device__ void processRippleDisplace(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float ripple = sinf((y + params.frame_count * 3) * 0.04f) * 6.0f;
        int src_x = ((int)(x + ripple) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + data[src_idx + j] * 0.4f);
        }
    }
    __device__ void processRotateShift(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = (params.frame_count / 3) % width;
        int src_x = (x + shift) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processSaturationGlow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int max_c = max(data[idx], max(data[idx + 1], data[idx + 2]));
        float glow = (params.frame_count % 100) / 100.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + max_c * 0.3f * glow);
        }
    }
    __device__ void processScaleToCenter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float scale = 0.5f + 0.5f * sinf(dist * 0.02f + params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * scale);
        }
    }
    __device__ void processShadowMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mirror_x = width - 1 - x;
        int mirror_idx = y * step + mirror_x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (avg < 100) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[mirror_idx + j] * 0.5f);
            }
        }
    }
    __device__ void processShiftChannels(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mode = (params.frame_count / 10) % 3;
        unsigned char b = data[idx], g = data[idx + 1], r = data[idx + 2];
        if (mode == 0) {
            data[idx] = g; data[idx + 1] = b; data[idx + 2] = r;
        } else if (mode == 1) {
            data[idx] = b; data[idx + 1] = r; data[idx + 2] = g;
        }
    }
    __device__ void processSliceGlitch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int slice = (params.frame_count / 2) % 8;
        if (y % 8 == slice) {
            int offset = ((params.frame_count % 30) - 15);
            int src_x = (x + offset + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processSobelGlow(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            int gx = 0, gy = 0;
            for (int j = 0; j < 3; ++j) {
                gx += abs(data[y * step + (x+1) * 4 + j] - data[y * step + (x-1) * 4 + j]);
                gy += abs(data[(y+1) * step + x * 4 + j] - data[(y-1) * step + x * 4 + j]);
            }
            int edge = (gx + gy) / 6;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + edge * 0.4f);
            }
        }
    }
    __device__ void processSpectralShift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float shift = (params.frame_count % 256) / 256.0f;
        float r_shift = shift;
        float g_shift = fmodf(shift + 0.33f, 1.0f);
        float b_shift = fmodf(shift + 0.66f, 1.0f);
        data[idx] = (unsigned char)(data[idx] * (0.5f + 0.5f * sinf(b_shift * 6.28f)));
        data[idx + 1] = (unsigned char)(data[idx + 1] * (0.5f + 0.5f * sinf(g_shift * 6.28f)));
        data[idx + 2] = (unsigned char)(data[idx + 2] * (0.5f + 0.5f * sinf(r_shift * 6.28f)));
    }
    __device__ void processSpiralTrail(int x, int y, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            float angle = atan2f(y - height / 2.0f, x - width / 2.0f);
            float dist = sqrtf((x - width / 2.0f) * (x - width / 2.0f) + (y - height / 2.0f) * (y - height / 2.0f));
            float spiral = sinf(angle * 4.0f + dist * 0.05f + params.frame_count * 0.1f);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * spiral) + prev[idx + j] * (0.5f - 0.5f * spiral));
            }
        }
    }
    __device__ void processSquareTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            int block = 8 + (params.frame_count / 20) % 4;
            int bx = (x / block) % 2;
            int by = (y / block) % 2;
            if ((bx + by) % 2 == 0) {
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + prev[idx + j] * 0.5f);
                }
            }
        }
    }
    __device__ void processStrobeNegate(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = (params.frame_count / 8) % 2;
        if (strobe == 1) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(255 - data[idx + j]);
            }
        }
    }
    __device__ void processThermalBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float heat = (avg / 255.0f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * (1.0f + heat * 0.3f));
        data[idx + 1] = (unsigned char)(data[idx + 1] * (1.0f - heat * 0.2f));
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (1.0f + heat * 0.1f));
    }
    __device__ void processTintShift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int tint = (params.frame_count / 15) % 4;
        if (tint == 0) data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 1.2f);
        else if (tint == 1) data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * 1.2f);
        else if (tint == 2) data[idx] = (unsigned char)fminf(255.0f, data[idx] * 1.2f);
    }
    __device__ void processTrailEcho(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int i = 1; i < params.numFrames && i < 4; ++i) {
            unsigned char* frame = allFrames[i];
            if (frame) {
                float blend = 1.0f / (i + 1.0f);
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend * 0.5f) + frame[idx + j] * blend * 0.5f);
                }
            }
        }
    }
    __device__ void processTransitionBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* next = allFrames[0];
        if (next) {
            float transition = ((params.frame_count % 100) / 100.0f);
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - transition) + next[idx + j] * transition);
            }
        }
    }
    __device__ void processTwistWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float twist = dist * 0.01f * sinf(params.frame_count * 0.05f);
        int src_x = (int)(cx + (x - cx) * cosf(twist) - (y - cy) * sinf(twist));
        int src_y = (int)(cy + (x - cx) * sinf(twist) + (y - cy) * cosf(twist));
        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processVerticalShift(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = (params.frame_count / 2) % height;
        int src_y = (y + offset) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + data[src_idx + j] * 0.4f);
        }
    }
    __device__ void processVortexBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float max_dist = sqrtf(cx * cx + cy * cy);
        float intensity = 1.0f - (dist / max_dist);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * intensity));
        }
    }
    __device__ void processWeavePattern(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int weave_x = (x + params.frame_count / 2) % 32;
        int weave_y = (y + params.frame_count / 3) % 32;
        float pattern = sinf(weave_x * 0.2f) * cosf(weave_y * 0.2f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.6f + 0.4f * (pattern + 1.0f) / 2.0f));
        }
    }
    __device__ void processWhiteBurst(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float max_dist = sqrtf(cx * cx + cy * cy);
        float burst = sinf((dist / max_dist - params.frame_count * 0.05f) * 6.28f);
        float white = fmaxf(0.0f, burst);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + white * 100);
        }
    }
    __device__ void processWiggleDisplace(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wiggle_x = sinf(y * 0.05f + params.frame_count * 0.03f) * 5.0f;
        float wiggle_y = cosf(x * 0.05f + params.frame_count * 0.02f) * 5.0f;
        int src_x = ((int)(x + wiggle_x) % width + width) % width;
        int src_y = ((int)(y + wiggle_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processXorPulseX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.05f);
        unsigned char xor_val = (unsigned char)(params.frame_count * pulse);
        for (int j = 0; j < 3; ++j) {
            unsigned char xored = data[idx + j] ^ xor_val;
            data[idx + j] = (unsigned char)(xored * 0.7f + data[idx + j] * 0.3f);
        }
    }
    __device__ void processYellowShift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = (params.frame_count / 20) % 2;
        if (shift == 0) {
            data[idx] = (unsigned char)(data[idx] * 0.5f);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * 1.2f);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 1.2f);
        }
    }
    __device__ void processZigzagGlitch(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int zigzag = (params.frame_count / 3) % 10;
        if (x % 10 == zigzag) {
            int offset = ((params.frame_count % 20) - 10);
            int src_x = (x + offset + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processAlphaModulate(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float modulation = 0.5f + 0.5f * sinf((x + y) * 0.01f + params.frame_count * 0.04f);
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * modulation);
        }
    }
    __device__ void processBlockSwap(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block = 16;
        int bx = x / block;
        int swap_px = ((bx + (params.frame_count / 5)) % (width / block)) * block + (x % block);
        int swap_py = y;
        if (swap_px >= 0 && swap_px < width) {
            int swap_idx = swap_py * step + swap_px * 4;
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[swap_idx + j] * 0.5f);
            }
        }
    }
    __device__ void processColorResonance(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float freq = 0.03f + (params.frame_count % 100) * 0.001f;
        float res = sinf(x * freq) + cosf(y * freq);
        data[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx] + res * 30));
        data[idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + 1] + res * 20));
        data[idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + 2] + res * 40));
    }
    __device__ void processDepthGlitch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float glitch_factor = (avg > 128) ? 0.8f : 1.2f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * glitch_factor);
        }
    }
    __device__ void processEchoShift(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int i = 0; i < min(3, params.numFrames); ++i) {
            unsigned char* frame = allFrames[i];
            if (frame) {
                float echo = 0.3f / (i + 1.0f);
                for (int j = 0; j < 3; ++j) {
                    data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - echo) + frame[idx + j] * echo);
                }
            }
        }
    }
    __device__ void processFractalNoise(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float noise = 0.0f;
        float amplitude = 1.0f;
        float frequency = 0.01f;
        for (int i = 0; i < 3; ++i) {
            noise += amplitude * sinf(x * frequency + params.frame_count * 0.02f);
            noise += amplitude * cosf(y * frequency + params.frame_count * 0.015f);
            amplitude *= 0.5f;
            frequency *= 2.0f;
        }
        noise = (noise + 3.0f) / 6.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * noise));
        }
    }
    __device__ void processGradientRotate(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx) + params.frame_count * 0.03f;
        float gradient = (sinf(angle) + 1.0f) / 2.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * gradient));
        }
    }
    __device__ void processHarmonicShift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float harmonic = sinf(params.frame_count * 0.05f);
        float harmonic2 = cosf(params.frame_count * 0.03f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * (1.0f + harmonic * 0.2f));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (1.0f + harmonic2 * 0.2f));
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (1.0f - harmonic * 0.1f));
    }
    __device__ void processAcidWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float warp = sinf(dist * 0.03f - params.frame_count * 0.1f) * 10.0f;
        int src_x = (int)(x + warp * dx / (dist + 1.0f)) % width;
        int src_y = (int)(y + warp * dy / (dist + 1.0f)) % height;
        if (src_x < 0) src_x += width; if (src_y < 0) src_y += height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processBlendDiagonal(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int diag = (x + y + params.frame_count) % 32;
        float blend = (diag < 16) ? diag / 16.0f : (32 - diag) / 16.0f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * blend));
    }
    __device__ void processChromaFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flash = (params.frame_count / 4) % 3;
        data[idx + flash] = (unsigned char)fminf(255.0f, data[idx + flash] * 1.5f);
    }
    __device__ void processCircleWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float wave = sinf(dist * 0.1f - params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.6f + 0.4f * wave));
    }
    __device__ void processColorCrush(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int crush = 3 + (params.frame_count / 20) % 5;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)((data[idx + j] / crush) * crush);
    }
    __device__ void processCrosshatchBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int hatch = ((x + params.frame_count) % 8 < 2) || ((y + params.frame_count) % 8 < 2);
        if (hatch) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.6f);
    }
    __device__ void processCyberGlitch(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.02f) {
            int offset = (int)(r * 500) - 250;
            int src_x = (x + offset + width) % width;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[y * step + src_x * 4 + j];
        }
    }
    __device__ void processDarkPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pulse = 0.5f + 0.3f * sinf(params.frame_count * 0.08f);
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (avg < 128) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * pulse);
    }
    __device__ void processDiamondPatternX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int dx = abs((x + params.frame_count) % 32 - 16);
        int dy = abs((y + params.frame_count) % 32 - 16);
        float diamond = 1.0f - (dx + dy) / 32.0f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + 0.5f * diamond));
    }
    __device__ void processDigitalRain(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int rain = (y + params.frame_count * 3) % height;
        if (rain < 20) {
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * 1.5f);
            data[idx] = (unsigned char)(data[idx] * 0.7f);
            data[idx + 2] = (unsigned char)(data[idx + 2] * 0.7f);
        }
    }
    __device__ void processDisplaceX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int disp = (int)(sinf(y * 0.05f + params.frame_count * 0.05f) * 8);
        int src_x = (x + disp + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
    }
    __device__ void processDriftBlend(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            int drift = (params.frame_count / 2) % 10;
            int src_x = (x + drift) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + prev[src_idx + j] * 0.4f);
        }
    }
    __device__ void processEdgePulse(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) edge += abs(data[y * step + (x+1) * 4 + j] - data[y * step + (x-1) * 4 + j]);
            float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
            if (edge > 50) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + pulse * 0.5f));
        }
    }
    __device__ void processFlameEffect(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float flame = sinf(x * 0.1f + params.frame_count * 0.15f) * 0.5f + 0.5f;
        float yf = 1.0f - (float)y / height;
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (1.0f + flame * yf * 0.5f));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (1.0f + flame * yf * 0.3f));
        data[idx] = (unsigned char)(data[idx] * (1.0f - flame * yf * 0.2f));
    }
    __device__ void processFlickerShift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flicker = (params.frame_count + x + y) % 5;
        if (flicker == 0) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(255 - data[idx + j]);
    }
    __device__ void processGhostLayer(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        for (int i = 0; i < min(4, params.numFrames); ++i) {
            unsigned char* frame = allFrames[i];
            if (frame) {
                float ghost = 0.15f / (i + 1.0f);
                for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - ghost) + frame[idx + j] * ghost);
            }
        }
    }
    __device__ void processGlitchBlockX(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bx = x / 16, by = y / 16;
        float r = gpu_rand(bx, by, params.seed + params.frame_count / 5);
        if (r < 0.1f) {
            int offset = (int)(r * 160) - 80;
            int src_x = (x + offset + width) % width;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[y * step + src_x * 4 + j];
        }
    }
    __device__ void processGlowPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.07f);
        if (avg > 150) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + pulse * 0.3f));
    }
    __device__ void processGridDistort(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gx = (x / 8) % 2, gy = (y / 8) % 2;
        int offset = ((gx + gy) % 2) * ((params.frame_count % 10) - 5);
        int src_x = (x + offset + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processHexPattern(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int hx = (x + (y / 2) * 3 + params.frame_count) % 12;
        int hy = y % 6;
        float hex = (hx < 6 && hy < 3) ? 1.0f : 0.7f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * hex);
    }
    __device__ void processHueRotate(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float angle = (params.frame_count % 360) * 0.0174533f;
        float cs = cosf(angle), sn = sinf(angle);
        float r = data[idx + 2], g = data[idx + 1], b = data[idx];
        data[idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r * (0.213f + cs * 0.787f - sn * 0.213f) + g * (0.715f - cs * 0.715f - sn * 0.715f) + b * (0.072f - cs * 0.072f + sn * 0.928f)));
        data[idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r * (0.213f - cs * 0.213f + sn * 0.143f) + g * (0.715f + cs * 0.285f + sn * 0.140f) + b * (0.072f - cs * 0.072f - sn * 0.283f)));
        data[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r * (0.213f - cs * 0.213f - sn * 0.787f) + g * (0.715f - cs * 0.715f + sn * 0.715f) + b * (0.072f + cs * 0.928f + sn * 0.072f)));
    }
    __device__ void processInterweaveX(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            int weave = (x + params.frame_count) % 4;
            if (weave < 2) for (int j = 0; j < 3; ++j) data[idx + j] = prev[idx + j];
        }
    }
    __device__ void processJitterBlend(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        int jitter = (int)(r * 6) - 3;
        int src_x = (x + jitter + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.7f + data[src_idx + j] * 0.3f);
    }
    __device__ void processKaleidoScope4(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mx = (x < width / 2) ? x : width - 1 - x;
        int my = (y < height / 2) ? y : height - 1 - y;
        int src_idx = my * step + mx * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
    }
    __device__ void processLaserScan(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int scan = (params.frame_count * 4) % height;
        if (abs(y - scan) < 3) {
            data[idx + 2] = 255; data[idx + 1] = (unsigned char)(data[idx + 1] * 0.5f); data[idx] = (unsigned char)(data[idx] * 0.5f);
        }
    }
    __device__ void processLightLeak(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float lx = (float)x / width, ly = (float)y / height;
        float leak = sinf(lx * 3.14159f + params.frame_count * 0.05f) * sinf(ly * 3.14159f);
        leak = fmaxf(0.0f, leak);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + leak * 80);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + leak * 60);
    }
    __device__ void processMeltDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int melt = (int)(sinf(x * 0.1f + params.frame_count * 0.05f) * 5);
        int src_y = min(max(y + melt, 0), height - 1);
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.4f + data[src_idx + j] * 0.6f);
    }
    __device__ void processMirrorDiag(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if ((x + y) % 2 == params.frame_count % 2) {
            int mx = width - 1 - x, my = height - 1 - y;
            mx = min(max(mx, 0), width - 1); my = min(max(my, 0), height - 1);
            int src_idx = my * step + mx * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + data[src_idx + j] * 0.5f);
        }
    }
    __device__ void processNeonGlow(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) {
                edge += abs(data[(y+1) * step + x * 4 + j] - data[(y-1) * step + x * 4 + j]);
                edge += abs(data[y * step + (x+1) * 4 + j] - data[y * step + (x-1) * 4 + j]);
            }
            if (edge > 100) {
                int ch = params.frame_count % 3;
                data[idx + ch] = 255;
            }
        }
    }
    __device__ void processNoiseBlendX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float noise = gpu_rand(x, y, params.seed + params.frame_count) * 0.3f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.85f + noise));
    }
    __device__ void processPixelDrift(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int drift = ((x / 4 + params.frame_count) % 8) - 4;
        int src_x = (x + drift + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processPlasmaWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float plasma = sinf(x * 0.03f + params.frame_count * 0.05f) + sinf(y * 0.03f + params.frame_count * 0.04f);
        plasma = (plasma + 2.0f) / 4.0f;
        data[idx] = (unsigned char)(data[idx] * (0.5f + plasma * 0.5f));
        data[idx + 1] = (unsigned char)(data[idx + 1] * (1.0f - plasma * 0.3f));
        data[idx + 2] = (unsigned char)(data[idx + 2] * (0.5f + (1.0f - plasma) * 0.5f));
    }
    __device__ void processPrismSplit(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = 2 + (params.frame_count % 4);
        int rx = (x + offset) % width, bx = (x - offset + width) % width;
        data[idx + 2] = data[y * step + rx * 4 + 2];
        data[idx] = data[y * step + bx * 4];
    }
    __device__ void processPulseRadial(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float pulse = sinf(dist * 0.05f - params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.7f + 0.3f * pulse));
    }
    __device__ void processRainbowStrobe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int phase = (params.frame_count / 3) % 6;
        float boost = 1.3f;
        if (phase == 0 || phase == 5) data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * boost);
        if (phase == 1 || phase == 2) data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * boost);
        if (phase == 3 || phase == 4) data[idx] = (unsigned char)fminf(255.0f, data[idx] * boost);
    }
    __device__ void processRefractionX(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float refract = sinf((x + y) * 0.05f + params.frame_count * 0.03f) * 4.0f;
        int src_x = ((int)(x + refract) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.6f + data[src_idx + j] * 0.4f);
    }
    __device__ void processScanlineX(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int scanline = (y + params.frame_count) % 4;
        if (scanline == 0) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.7f);
    }
    __device__ void processShatterEffect(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bx = x / 32, by = y / 32;
        float r = gpu_rand(bx, by, params.seed);
        int ox = (int)((r - 0.5f) * (params.frame_count % 20));
        int oy = (int)((r - 0.5f) * (params.frame_count % 20));
        int src_x = (x + ox + width) % width, src_y = (y + oy + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processStaticNoise(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.05f) {
            unsigned char noise = (unsigned char)(r * 5000) % 256;
            for (int j = 0; j < 3; ++j) data[idx + j] = noise;
        }
    }
    __device__ void processTunnelVision(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float max_dist = sqrtf(cx * cx + cy * cy);
        float vignette = 1.0f - (dist / max_dist) * 0.7f;
        vignette = fmaxf(0.3f, vignette + 0.2f * sinf(params.frame_count * 0.05f));
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * vignette);
    }

    __device__ void processAberrationPulse(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = (int)(5.0f * sinf(params.frame_count * 0.1f));
        int rx = (x + offset + width) % width;
        int bx = (x - offset + width) % width;
        data[idx] = data[y * step + bx * 4];
        data[idx + 2] = data[y * step + rx * 4 + 2];
    }
    __device__ void processAquaWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf(x * 0.05f + params.frame_count * 0.08f) * 10.0f;
        int src_y = ((int)(y + wave) + height) % height;
        int src_idx = src_y * step + x * 4;
        data[idx] = (unsigned char)fminf(255.0f, data[src_idx] * 0.8f);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[src_idx + 1] * 1.2f);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[src_idx + 2] * 1.1f);
    }
    __device__ void processBinaryFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int threshold = 128 + (int)(64.0f * sinf(params.frame_count * 0.15f));
        unsigned char val = (gray > threshold) ? 255 : 0;
        for (int j = 0; j < 3; ++j) data[idx + j] = val;
    }
    __device__ void processBloomGlow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float bloom = 1.0f + 0.3f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            float val = data[idx + j] * bloom;
            if (data[idx + j] > 200) val += 30;
            data[idx + j] = (unsigned char)fminf(255.0f, val);
        }
    }
    __device__ void processCellularNoise(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = (x / 16) * 16 + 8, cy = (y / 16) * 16 + 8;
        float dist = sqrtf((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
        float factor = fmaxf(0.5f, 1.0f - dist / 16.0f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * factor);
    }
    __device__ void processChromaShift2(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shift = 3 + (params.frame_count % 5);
        int r_idx = y * step + ((x + shift) % width) * 4;
        int b_idx = y * step + ((x - shift + width) % width) * 4;
        data[idx + 2] = data[r_idx + 2];
        data[idx] = data[b_idx];
    }
    __device__ void processColorBands(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int band = (y + params.frame_count) % 60;
        if (band < 20) data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 1.5f);
        else if (band < 40) data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * 1.5f);
        else data[idx] = (unsigned char)fminf(255.0f, data[idx] * 1.5f);
    }
    __device__ void processColorVortex(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float angle = atan2f(dy, dx) + params.frame_count * 0.02f;
        float dist = sqrtf(dx * dx + dy * dy);
        int hue_shift = (int)(angle * 40.0f + dist * 0.1f) % 256;
        data[idx] = (data[idx] + hue_shift) % 256;
        data[idx + 1] = (data[idx + 1] + hue_shift / 2) % 256;
    }
    __device__ void processCrystalMosaic(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int size = 8 + (params.frame_count % 8);
        int bx = (x / size) * size + size / 2;
        int by = (y / size) * size + size / 2;
        bx = bx < width ? bx : width - 1;
        by = by < height ? by : height - 1;
        int src_idx = by * step + bx * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processCubicDistort(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float nx = (x - width / 2.0f) / (width / 2.0f);
        float ny = (y - height / 2.0f) / (height / 2.0f);
        float distort = 0.1f * sinf(params.frame_count * 0.05f);
        nx = nx + distort * nx * nx * nx;
        ny = ny + distort * ny * ny * ny;
        int src_x = (int)((nx + 1.0f) * width / 2.0f) % width;
        int src_y = (int)((ny + 1.0f) * height / 2.0f) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processDepthFade(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float depth = (float)y / height;
        depth = depth + 0.2f * sinf(params.frame_count * 0.03f);
        depth = fmaxf(0.3f, fminf(1.0f, depth));
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * depth);
    }
    __device__ void processDiscoFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int phase = params.frame_count % 12;
        if (phase < 4) data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + 100);
        else if (phase < 8) data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 100);
        else data[idx] = (unsigned char)fminf(255.0f, data[idx] + 100);
    }
    __device__ void processDitherBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int pattern = ((x + params.frame_count) % 2) ^ ((y + params.frame_count) % 2);
        float factor = pattern ? 1.2f : 0.8f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
    }
    __device__ void processDoubleVision(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = 10 + (int)(5.0f * sinf(params.frame_count * 0.1f));
        int x2 = (x + offset) % width;
        int idx2 = y * step + x2 * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)((data[idx + j] + data[idx2 + j]) / 2);
    }
    __device__ void processDreamHaze(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float haze = 0.7f + 0.3f * sinf(params.frame_count * 0.02f + x * 0.01f + y * 0.01f);
        for (int j = 0; j < 3; ++j) {
            float val = data[idx + j] * haze + 50 * (1.0f - haze);
            data[idx + j] = (unsigned char)fminf(255.0f, val);
        }
    }
    __device__ void processElectricStorm(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.02f) {
            data[idx] = 200;
            data[idx + 1] = 200;
            data[idx + 2] = 255;
        }
    }
    __device__ void processEmbossShift(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            int shift = 1 + (params.frame_count % 3);
            for (int j = 0; j < 3; ++j) {
                int diff = data[idx + j] - data[(y - shift) * step + (x - shift) * 4 + j];
                data[idx + j] = (unsigned char)(128 + diff);
            }
        }
    }
    __device__ void processFiberOptic(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx);
        int fiber = (int)((angle + 3.14159f) * 20) % 3;
        data[idx + fiber] = (unsigned char)fminf(255.0f, data[idx + fiber] * 1.5f);
    }
    __device__ void processFilmGrain(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float grain = (gpu_rand(x, y, params.seed + params.frame_count) - 0.5f) * 30.0f;
        for (int j = 0; j < 3; ++j) {
            float val = data[idx + j] + grain;
            data[idx + j] = (unsigned char)fmaxf(0.0f, fminf(255.0f, val));
        }
    }
    __device__ void processFireWorks(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cx = (params.seed % (width / 2)) + width / 4;
        int cy = (params.seed % (height / 2)) + height / 4;
        float dist = sqrtf((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
        float ring = (float)(params.frame_count % 60);
        if (fabsf(dist - ring) < 3.0f) {
            data[idx + 2] = 255;
            data[idx + 1] = (unsigned char)(200 - ring * 3);
            data[idx] = 0;
        }
    }
    __device__ void processFluidMotion(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float t = params.frame_count * 0.05f;
        float dx = sinf(y * 0.02f + t) * 5.0f;
        float dy = cosf(x * 0.02f + t) * 5.0f;
        int src_x = ((int)(x + dx) + width) % width;
        int src_y = ((int)(y + dy) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processFogRoll(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float fog_line = height * (0.5f + 0.3f * sinf(params.frame_count * 0.02f));
        float fog = 0.0f;
        if (y > fog_line) fog = fminf(0.7f, (y - fog_line) / (height * 0.3f));
        for (int j = 0; j < 3; ++j) {
            float val = data[idx + j] * (1.0f - fog) + 200 * fog;
            data[idx + j] = (unsigned char)val;
        }
    }
    __device__ void processGlassRefract(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float refract = sinf(x * 0.1f + params.frame_count * 0.1f) * cosf(y * 0.1f) * 8.0f;
        int src_x = ((int)(x + refract) + width) % width;
        int src_y = ((int)(y + refract * 0.5f) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processGlowTrails(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float glow = 1.0f + 0.5f * sinf(params.frame_count * 0.1f + x * 0.02f);
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] > 150) {
                data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * glow);
            }
        }
    }
    __device__ void processGridPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int grid_x = x % 32, grid_y = y % 32;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
        if (grid_x < 2 || grid_y < 2) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * pulse + 128 * (1.0f - pulse));
        }
    }
    __device__ void processHalftoneBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int dot_size = 4 + (params.frame_count % 4);
        int cx = (x / dot_size) * dot_size + dot_size / 2;
        int cy = (y / dot_size) * dot_size + dot_size / 2;
        float dist = sqrtf((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
        float threshold = (gray / 255.0f) * dot_size;
        unsigned char val = (dist < threshold) ? 255 : 0;
        for (int j = 0; j < 3; ++j) data[idx + j] = (data[idx + j] + val) / 2;
    }
    __device__ void processHeatDistort(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float heat = sinf(y * 0.1f + params.frame_count * 0.2f) * (1.0f - (float)y / height) * 10.0f;
        int src_x = ((int)(x + heat) + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processHoloGlitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int scanline = (y + params.frame_count * 2) % 100;
        if (scanline < 5) {
            data[idx] = (data[idx] + 50) % 256;
            data[idx + 1] = (data[idx + 1] + 100) % 256;
            data[idx + 2] = (data[idx + 2] + 150) % 256;
        }
    }
    __device__ void processInfraredView(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int intensity = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float pulse = 0.8f + 0.2f * sinf(params.frame_count * 0.05f);
        if (intensity > 180) { data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 200; }
        else if (intensity > 140) { data[idx] = 0; data[idx + 1] = (unsigned char)(255 * pulse); data[idx + 2] = 255; }
        else if (intensity > 100) { data[idx] = 0; data[idx + 1] = 100; data[idx + 2] = (unsigned char)(200 * pulse); }
        else { data[idx] = (unsigned char)(50 * pulse); data[idx + 1] = 0; data[idx + 2] = 100; }
    }
    __device__ void processLavaLamp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float t = params.frame_count * 0.02f;
        float blob1 = sinf(x * 0.02f + t) * cosf(y * 0.02f + t * 0.7f);
        float blob2 = cosf(x * 0.03f - t * 0.5f) * sinf(y * 0.03f + t);
        float blend = (blob1 + blob2 + 2.0f) / 4.0f;
        data[idx + 2] = (unsigned char)(data[idx + 2] * blend + 200 * (1.0f - blend));
        data[idx + 1] = (unsigned char)(data[idx + 1] * (1.0f - blend) + 100 * blend);
    }
    __device__ void processLensFlare(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flare_x = width / 2 + (int)(100.0f * sinf(params.frame_count * 0.03f));
        int flare_y = height / 3;
        float dist = sqrtf((float)((x - flare_x) * (x - flare_x) + (y - flare_y) * (y - flare_y)));
        if (dist < 50) {
            float intensity = 1.0f - dist / 50.0f;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + 200 * intensity);
        }
    }
    __device__ void processLightningBolt(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (params.frame_count % 15 < 2) {
            int bolt_x = width / 2 + (params.seed % 100) - 50;
            bolt_x += (int)(gpu_rand(y, params.seed, params.frame_count) * 30 - 15);
            if (abs(x - bolt_x) < 3) {
                data[idx] = 200;
                data[idx + 1] = 200;
                data[idx + 2] = 255;
            }
        }
    }
    __device__ void processLiquidMetal(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float shine = 0.5f + 0.5f * sinf(gray * 0.1f + params.frame_count * 0.1f);
        unsigned char metal = (unsigned char)(gray * shine + 128 * (1.0f - shine));
        data[idx] = metal;
        data[idx + 1] = metal;
        data[idx + 2] = (unsigned char)fminf(255.0f, metal * 1.1f);
    }
    __device__ void processMatrixCode(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int col = x / 8;
        int drop = ((params.frame_count * 3 + col * 17) % (height + 50)) - 25;
        if (y > drop && y < drop + 25) {
            float intensity = 1.0f - (float)(y - drop) / 25.0f;
            data[idx] = 0;
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 200 * intensity);
            data[idx + 2] = 0;
        }
    }
    __device__ void processMirrorKaleid(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int mx = (x < width / 2) ? x : width - 1 - x;
        int my = (y < height / 2) ? y : height - 1 - y;
        mx = (mx + params.frame_count) % (width / 2);
        int src_idx = my * step + mx * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processNightVision(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] * 30 + data[idx + 1] * 59 + data[idx + 2] * 11) / 100;
        float noise = (gpu_rand(x, y, params.seed + params.frame_count) - 0.5f) * 20.0f;
        gray = (int)fmaxf(0.0f, fminf(255.0f, gray + noise));
        data[idx] = 0;
        data[idx + 1] = (unsigned char)(gray);
        data[idx + 2] = 0;
    }
    __device__ void processOilSlick(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float phase = x * 0.05f + y * 0.03f + params.frame_count * 0.1f;
        float r_shift = sinf(phase) * 50.0f;
        float g_shift = sinf(phase + 2.094f) * 50.0f;
        float b_shift = sinf(phase + 4.189f) * 50.0f;
        data[idx] = (unsigned char)fmaxf(0.0f, fminf(255.0f, data[idx] + b_shift));
        data[idx + 1] = (unsigned char)fmaxf(0.0f, fminf(255.0f, data[idx + 1] + g_shift));
        data[idx + 2] = (unsigned char)fmaxf(0.0f, fminf(255.0f, data[idx + 2] + r_shift));
    }
    __device__ void processParticleField(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 4, y / 4, params.seed);
        if (r < 0.1f) {
            int py = (y + params.frame_count * 2) % 4;
            if (py == 0) {
                for (int j = 0; j < 3; ++j) data[idx + j] = 255;
            }
        }
    }
    __device__ void processPinwheelSpin(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx) + params.frame_count * 0.05f;
        int sector = (int)((angle + 3.14159f) / 0.785f) % 8;
        if (sector % 2 == 0) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.6f);
        }
    }
    __device__ void processPixelStorm(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.03f) {
            int ox = (int)((r * 1000) - 500) % 20;
            int oy = (int)((r * 500) - 250) % 20;
            int src_x = (x + ox + width) % width;
            int src_y = (y + oy + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }

    
    __device__ void processPlaidPattern(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int stripe_x = (x + params.frame_count) % 40;
        int stripe_y = (y + params.frame_count) % 40;
        float blend = 1.0f;
        if (stripe_x < 8) blend *= 1.3f;
        if (stripe_y < 8) blend *= 1.3f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * blend);
    }
    __device__ void processPolarInvert(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float max_dist = sqrtf(cx * cx + cy * cy);
        float threshold = max_dist * (0.5f + 0.3f * sinf(params.frame_count * 0.05f));
        if (dist > threshold) {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255 - data[idx + j];
        }
    }
    __device__ void processPolychromeTint(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int zone = ((x / 50) + (y / 50) + params.frame_count / 10) % 6;
        float tints[6][3] = {{1.2f,0.9f,0.9f},{0.9f,1.2f,0.9f},{0.9f,0.9f,1.2f},{1.2f,1.2f,0.9f},{1.2f,0.9f,1.2f},{0.9f,1.2f,1.2f}};
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * tints[zone][j]);
    }
    __device__ void processPopArtDots(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int dot_size = 8;
        int cx = (x / dot_size) * dot_size + dot_size / 2;
        int cy = (y / dot_size) * dot_size + dot_size / 2;
        float dist = sqrtf((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float radius = (gray / 255.0f) * dot_size * 0.6f;
        if (dist < radius) {
            int color = ((cx / dot_size) + (cy / dot_size) + params.frame_count / 5) % 3;
            data[idx] = (color == 0) ? 255 : 0;
            data[idx + 1] = (color == 1) ? 255 : 0;
            data[idx + 2] = (color == 2) ? 255 : 0;
        } else {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255;
        }
    }
    __device__ void processPrismaticEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - data[(y + 1) * step + (x + 1) * 4 + j]);
                edge += diff;
            }
            if (edge > 50) {
                int hue = (x + y + params.frame_count * 3) % 256;
                data[idx + 2] = hue;
                data[idx + 1] = (hue + 85) % 256;
                data[idx] = (hue + 170) % 256;
            }
        }
    }
    __device__ void processPulseWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float pulse = sinf(dist * 0.05f - params.frame_count * 0.1f) * 10.0f;
        float angle = atan2f(y - cy, x - cx);
        int src_x = (int)(cx + (dist + pulse) * cosf(angle)) % width;
        int src_y = (int)(cy + (dist + pulse) * sinf(angle)) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processQuantumNoise(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.1f) {
            int channel = (int)(r * 30) % 3;
            data[idx + channel] = (unsigned char)((int)(r * 2550) % 256);
        }
    }
    __device__ void processQuiltBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int patch_x = (x / 32) % 2;
        int patch_y = (y / 32) % 2;
        int pattern = (patch_x + patch_y + params.frame_count / 15) % 4;
        float factors[4][3] = {{1.2f,0.8f,0.8f},{0.8f,1.2f,0.8f},{0.8f,0.8f,1.2f},{1.1f,1.1f,0.8f}};
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factors[pattern][j]);
    }
    __device__ void processRadarSweep(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx);
        float sweep_angle = fmodf(params.frame_count * 0.05f, 6.28318f) - 3.14159f;
        float diff = fabsf(angle - sweep_angle);
        if (diff > 3.14159f) diff = 6.28318f - diff;
        if (diff < 0.3f) {
            float intensity = 1.0f - diff / 0.3f;
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 150 * intensity);
        }
    }
    __device__ void processRaindropRipple(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int drop_x = (params.seed % width);
        int drop_y = (params.seed / 2) % height;
        float dist = sqrtf((float)((x - drop_x) * (x - drop_x) + (y - drop_y) * (y - drop_y)));
        float ring = fmodf(params.frame_count * 2.0f, 100.0f);
        if (fabsf(dist - ring) < 5.0f) {
            float wave = sinf((dist - ring) * 0.5f) * 0.3f;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + wave));
        }
    }
    __device__ void processRasterBars(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bar = (y + params.frame_count * 2) % 100;
        if (bar < 20) {
            float intensity = sinf(bar * 0.157f);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (1.0f + intensity));
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (1.0f + intensity * 0.5f));
        }
    }
    __device__ void processRetroTube(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = (x - cx) / cx, dy = (y - cy) / cy;
        float dist_sq = dx * dx + dy * dy;
        float curve = 1.0f + dist_sq * 0.2f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] / curve);
        if ((y + params.frame_count) % 3 == 0) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.95f);
        }
    }
    __device__ void processRingWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float wave = sinf(dist * 0.1f - params.frame_count * 0.15f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (0.8f + wave * 0.2f));
    }
    __device__ void processRippleTank(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave1 = sinf(x * 0.05f + params.frame_count * 0.1f);
        float wave2 = sinf(y * 0.05f + params.frame_count * 0.08f);
        float combined = (wave1 + wave2) * 5.0f;
        int src_x = ((int)(x + combined) + width) % width;
        int src_y = ((int)(y + combined) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processRotatingPrism(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx) + params.frame_count * 0.02f;
        int sector = (int)((angle + 3.14159f) / 1.047f) % 6;
        float hue_shift = sector * 42.0f;
        data[idx + 2] = (unsigned char)fmodf(data[idx + 2] + hue_shift, 256.0f);
        data[idx + 1] = (unsigned char)fmodf(data[idx + 1] + hue_shift * 0.5f, 256.0f);
    }
    __device__ void processSandStorm(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.15f) {
            int ox = (int)((r - 0.075f) * 40);
            int src_x = (x + ox + width) % width;
            int src_idx = y * step + src_x * 4;
            unsigned char sand = (unsigned char)(200 + r * 50);
            data[idx] = (data[src_idx] + sand) / 2;
            data[idx + 1] = (data[src_idx + 1] + sand - 20) / 2;
            data[idx + 2] = (data[src_idx + 2] + sand - 40) / 2;
        }
    }
    __device__ void processSaturationPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.08f);
        for (int j = 0; j < 3; ++j) {
            float diff = data[idx + j] - gray;
            data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, gray + diff * (1.0f + pulse)));
        }
    }
    __device__ void processScatterPixel(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        int scatter = 2 + (params.frame_count % 5);
        int ox = (int)((r - 0.5f) * scatter * 2);
        int oy = (int)((gpu_rand(y, x, params.seed) - 0.5f) * scatter * 2);
        int src_x = (x + ox + width) % width;
        int src_y = (y + oy + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processShadowPlay(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shadow_off = 5 + (params.frame_count % 10);
        int sx = (x + shadow_off) % width;
        int sy = (y + shadow_off) % height;
        int shadow_idx = sy * step + sx * 4;
        int shadow_val = (data[shadow_idx] + data[shadow_idx + 1] + data[shadow_idx + 2]) / 3;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.7f + shadow_val * 0.1f);
    }
    __device__ void processShimmerGlass(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float shimmer = sinf(x * 0.2f + params.frame_count * 0.2f) * cosf(y * 0.2f + params.frame_count * 0.15f) * 3.0f;
        int src_x = ((int)(x + shimmer) + width) % width;
        int src_y = ((int)(y + shimmer) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)((data[idx + j] + data[src_idx + j]) / 2);
    }
    __device__ void processSilhouetteBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int threshold = 100 + (int)(50.0f * sinf(params.frame_count * 0.05f));
        if (gray < threshold) {
            float fade = (float)gray / threshold;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * fade);
        }
    }
    __device__ void processSketchOutline(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) {
                int gx = data[(y) * step + (x + 1) * 4 + j] - data[(y) * step + (x - 1) * 4 + j];
                int gy = data[(y + 1) * step + (x) * 4 + j] - data[(y - 1) * step + (x) * 4 + j];
                edge += abs(gx) + abs(gy);
            }
            edge = edge / 3;
            unsigned char sketch = (edge > 30) ? 0 : 255;
            float blend = 0.5f + 0.5f * sinf(params.frame_count * 0.03f);
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + sketch * blend);
        }
    }
    __device__ void processSliceShift(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int slice = y / 20;
        int shift = (slice % 2 == 0) ? (params.frame_count % 30) : -(params.frame_count % 30);
        int src_x = (x + shift + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processSmearMotion(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int smear_len = 5 + (params.frame_count % 10);
        int sum[3] = {0, 0, 0};
        for (int i = 0; i < smear_len; ++i) {
            int sx = (x + i) % width;
            int sidx = y * step + sx * 4;
            for (int j = 0; j < 3; ++j) sum[j] += data[sidx + j];
        }
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(sum[j] / smear_len);
    }
    __device__ void processSmokeWisp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float t = params.frame_count * 0.03f;
        float wisp = sinf(x * 0.03f + t) * cosf(y * 0.02f + t * 0.7f) * 15.0f;
        int src_y = ((int)(y + wisp) + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            float blend = (data[idx + j] + data[src_idx + j] + 50) / 2.5f;
            data[idx + j] = (unsigned char)fminf(255.0f, blend);
        }
    }
    __device__ void processSnowDrift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count);
        if (r < 0.02f) {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255;
        } else {
            float bright = 1.0f + 0.1f * sinf(params.frame_count * 0.05f);
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * bright);
        }
    }
    __device__ void processSolarFlare(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flare_x = width / 4, flare_y = height / 4;
        float dist = sqrtf((float)((x - flare_x) * (x - flare_x) + (y - flare_y) * (y - flare_y)));
        float flare_radius = 30.0f + 20.0f * sinf(params.frame_count * 0.1f);
        if (dist < flare_radius) {
            float intensity = 1.0f - dist / flare_radius;
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + 200 * intensity);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 150 * intensity);
        }
    }
    __device__ void processSparkShower(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.005f) {
            int fall = (params.frame_count * 5 + (int)(r * 1000)) % height;
            if (abs(y - fall) < 3) {
                data[idx] = 100;
                data[idx + 1] = 200;
                data[idx + 2] = 255;
            }
        }
    }
    __device__ void processSpectrumWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf(x * 0.02f + y * 0.02f + params.frame_count * 0.1f);
        int hue = (int)((wave + 1.0f) * 127.5f);
        data[idx] = (unsigned char)((data[idx] + hue) / 2);
        data[idx + 1] = (unsigned char)((data[idx + 1] + (hue + 85) % 256) / 2);
        data[idx + 2] = (unsigned char)((data[idx + 2] + (hue + 170) % 256) / 2);
    }
    __device__ void processSpiralZoom(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float angle = atan2f(dy, dx) + dist * 0.01f + params.frame_count * 0.02f;
        float zoom = 0.95f + 0.05f * sinf(params.frame_count * 0.05f);
        int src_x = (int)(cx + dist * zoom * cosf(angle)) % width;
        int src_y = (int)(cy + dist * zoom * sinf(angle)) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processSplitMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int split = width / 2 + (int)(50.0f * sinf(params.frame_count * 0.03f));
        if (x > split) {
            int mirror_x = split - (x - split);
            if (mirror_x >= 0) {
                int src_idx = y * step + mirror_x * 4;
                for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
            }
        }
    }
    __device__ void processStarBurst(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx);
        float ray = sinf(angle * 8.0f + params.frame_count * 0.1f);
        if (ray > 0.7f) {
            float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
            float max_dist = sqrtf(cx * cx + cy * cy);
            float fade = 1.0f - dist / max_dist;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + 100 * fade * ray);
        }
    }
    __device__ void processStaticPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pulse = 0.8f + 0.2f * sinf(params.frame_count * 0.15f);
        float r = gpu_rand(x, y, params.seed + params.frame_count / 3);
        if (r < 0.1f) {
            unsigned char noise = (unsigned char)(r * 2550) % 256;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * pulse + noise * (1.0f - pulse));
        }
    }
    __device__ void processStencilCut(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int levels = 4;
        int quantized = (gray / (256 / levels)) * (256 / levels);
        int threshold = 64 + (params.frame_count % 128);
        if (gray < threshold) quantized = 0;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)((data[idx + j] + quantized) / 2);
    }
    __device__ void processStippleShade(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float r = gpu_rand(x, y, params.seed);
        float threshold = gray / 255.0f;
        if (r > threshold) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.3f);
        }
    }
    __device__ void processStormCloud(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cloud = sinf(x * 0.02f + params.frame_count * 0.05f) * cosf(y * 0.03f + params.frame_count * 0.03f);
        if (y < height / 3 && cloud > 0.3f) {
            float gray_factor = 0.5f + cloud * 0.3f;
            int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(avg * gray_factor);
        }
    }
    __device__ void processStreakBlur(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int streak_len = 8 + (params.frame_count % 8);
        int sum[3] = {0, 0, 0};
        int count = 0;
        for (int i = -streak_len; i <= streak_len; ++i) {
            int sx = (x + i + width) % width;
            int sidx = y * step + sx * 4;
            for (int j = 0; j < 3; ++j) sum[j] += data[sidx + j];
            count++;
        }
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(sum[j] / count);
    }
    __device__ void processStrobeEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) {
                int diff = abs(data[idx + j] - data[(y + 1) * step + (x + 1) * 4 + j]);
                edge += diff;
            }
            if (edge > 40 && (params.frame_count % 4) < 2) {
                for (int j = 0; j < 3; ++j) data[idx + j] = 255;
            }
        }
    }
    __device__ void processSubpixelShift(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int r_shift = 1 + (params.frame_count % 3);
        int b_shift = -1 - (params.frame_count % 3);
        int r_x = (x + r_shift + width) % width;
        int b_x = (x + b_shift + width) % width;
        data[idx + 2] = data[y * step + r_x * 4 + 2];
        data[idx] = data[y * step + b_x * 4];
    }
    __device__ void processSwimDistort(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float t = params.frame_count * 0.05f;
        float dx = sinf(y * 0.05f + t) * 8.0f;
        float dy = cosf(x * 0.05f + t * 0.8f) * 8.0f;
        int src_x = ((int)(x + dx) + width) % width;
        int src_y = ((int)(y + dy) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }

    __device__ void processTangentWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float nx = (x - width / 2.0f) / (width / 2.0f);
        float ny = (y - height / 2.0f) / (height / 2.0f);
        float warp = tanf(nx * 0.5f) * tanf(ny * 0.5f) * 20.0f * sinf(params.frame_count * 0.05f);
        int src_x = ((int)(x + warp) + width) % width;
        int src_y = ((int)(y + warp) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processTapeGlitch(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int glitch_line = (params.seed + params.frame_count * 7) % 50;
        if ((y % 50) < glitch_line && (y % 50) > glitch_line - 5) {
            int shift = (params.frame_count % 30) - 15;
            int src_x = (x + shift + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 1.3f);
        }
    }
    __device__ void processTechnoGrid(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gx = x % 16, gy = y % 16;
        int pulse = (params.frame_count % 32);
        if (gx == 0 || gy == 0) {
            float bright = (pulse < 16) ? pulse / 16.0f : (32 - pulse) / 16.0f;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] + 100 * bright);
        }
    }
    __device__ void processTeleportPixel(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 8, y / 8, params.seed + params.frame_count / 5);
        if (r < 0.1f) {
            int tx = (int)(r * width * 10) % width;
            int ty = (int)(r * height * 7) % height;
            int src_idx = ty * step + tx * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processTemporalBlur(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float blur = 0.7f + 0.3f * sinf(params.frame_count * 0.02f + x * 0.01f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * blur + 128 * (1.0f - blur));
    }
    __device__ void processTerraFracture(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block = (x / 40) + (y / 40) * (width / 40);
        float r = gpu_rand(block, params.seed, params.frame_count / 10);
        if (r < 0.3f) {
            int ox = (int)((r - 0.15f) * 20);
            int oy = (int)((r - 0.15f) * 20);
            int src_x = (x + ox + width) % width;
            int src_y = (y + oy + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processTextureWave(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf(x * 0.1f + params.frame_count * 0.1f) * cosf(y * 0.1f);
        float texture = (wave + 1.0f) * 0.5f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.7f + texture * 0.3f));
    }
    __device__ void processThresholdPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int threshold = 64 + (int)(128.0f * fabsf(sinf(params.frame_count * 0.05f)));
        if (gray > threshold) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * 1.3f);
        } else {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.7f);
        }
    }
    __device__ void processTidalWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf(x * 0.02f - params.frame_count * 0.1f) * 30.0f * (1.0f - (float)y / height);
        int src_y = ((int)(y + wave) + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processTintCycle(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cycle = params.frame_count % 90;
        float r_tint = (cycle < 30) ? 1.2f : 0.9f;
        float g_tint = (cycle >= 30 && cycle < 60) ? 1.2f : 0.9f;
        float b_tint = (cycle >= 60) ? 1.2f : 0.9f;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * b_tint);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * g_tint);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * r_tint);
    }
    __device__ void processTraceEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            int edge = 0;
            for (int j = 0; j < 3; ++j) {
                edge += abs(data[y * step + (x + 1) * 4 + j] - data[y * step + (x - 1) * 4 + j]);
                edge += abs(data[(y + 1) * step + x * 4 + j] - data[(y - 1) * step + x * 4 + j]);
            }
            float trace = fminf(1.0f, edge / 200.0f);
            float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - trace * pulse) + 255 * trace * pulse);
        }
    }
    __device__ void processTriangleMosaic(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int size = 20 + (params.frame_count % 20);
        int tx = x / size, ty = y / size;
        int lx = x % size, ly = y % size;
        int cx, cy;
        if (lx + ly < size) { cx = tx * size; cy = ty * size; }
        else { cx = tx * size + size; cy = ty * size + size; }
        cx = cx < width ? cx : width - 1;
        cy = cy < height ? cy : height - 1;
        int src_idx = cy * step + cx * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processTripleSplit(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int third = width / 3;
        int shift = (params.frame_count % 20) - 10;
        if (x < third) {
            int src_x = (x - shift + width) % width;
            data[idx + 2] = data[y * step + src_x * 4 + 2];
        } else if (x >= 2 * third) {
            int src_x = (x + shift + width) % width;
            data[idx] = data[y * step + src_x * 4];
        }
    }
    __device__ void processTurbulentFlow(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float t = params.frame_count * 0.03f;
        float turb_x = sinf(y * 0.03f + t) * cosf(x * 0.02f + t * 0.7f) * 10.0f;
        float turb_y = cosf(x * 0.03f + t * 0.5f) * sinf(y * 0.02f + t) * 10.0f;
        int src_x = ((int)(x + turb_x) + width) % width;
        int src_y = ((int)(y + turb_y) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processUnderwaterCaustic(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float caustic = sinf(x * 0.1f + params.frame_count * 0.15f) * sinf(y * 0.1f + params.frame_count * 0.1f);
        caustic = (caustic + 1.0f) * 0.5f;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * (0.8f + caustic * 0.4f));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (0.9f + caustic * 0.3f));
    }
    __device__ void processUnsharpPulse(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            float amount = 0.5f + 0.5f * sinf(params.frame_count * 0.08f);
            for (int j = 0; j < 3; ++j) {
                int center = data[idx + j] * 5;
                int neighbors = data[(y - 1) * step + x * 4 + j] + data[(y + 1) * step + x * 4 + j] +
                               data[y * step + (x - 1) * 4 + j] + data[y * step + (x + 1) * 4 + j];
                float sharp = center - neighbors;
                data[idx + j] = (unsigned char)fmaxf(0.0f, fminf(255.0f, data[idx + j] + sharp * amount * 0.25f));
            }
        }
    }
    __device__ void processVaporTrail(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int trail_len = 10 + (params.frame_count % 10);
        float sum[3] = {0, 0, 0};
        for (int i = 0; i < trail_len; ++i) {
            int sx = (x - i + width) % width;
            float weight = 1.0f - (float)i / trail_len;
            for (int j = 0; j < 3; ++j) sum[j] += data[y * step + sx * 4 + j] * weight;
        }
        float total_weight = trail_len * 0.5f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(sum[j] / total_weight);
    }
    __device__ void processVectorField(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float angle = sinf(x * 0.02f) * cosf(y * 0.02f) * 3.14159f;
        float mag = 5.0f + 3.0f * sinf(params.frame_count * 0.05f);
        int src_x = ((int)(x + mag * cosf(angle)) + width) % width;
        int src_y = ((int)(y + mag * sinf(angle)) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processVelocityBlur(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float blur_len = dist * 0.02f * (1.0f + sinf(params.frame_count * 0.05f));
        float nx = dx / (dist + 0.001f), ny = dy / (dist + 0.001f);
        int sum[3] = {0, 0, 0};
        int count = 0;
        for (int i = 0; i < 5; ++i) {
            int sx = (int)(x - nx * i * blur_len) % width;
            int sy = (int)(y - ny * i * blur_len) % height;
            sx = (sx + width) % width;
            sy = (sy + height) % height;
            for (int j = 0; j < 3; ++j) sum[j] += data[sy * step + sx * 4 + j];
            count++;
        }
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(sum[j] / count);
    }
    __device__ void processVerticalMelt(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float melt = sinf(x * 0.05f + params.frame_count * 0.1f) * 20.0f;
        int src_y = ((int)(y - melt) + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)((data[idx + j] + data[src_idx + j]) / 2);
    }
    __device__ void processVHSTracking(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int band = (y + params.frame_count * 3) % 200;
        if (band < 10) {
            int shift = (int)(sinf(band * 0.5f + params.frame_count * 0.2f) * 30);
            int src_x = (x + shift + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
        if ((y + params.frame_count) % 3 == 0) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.95f);
        }
    }
    __device__ void processVibrantPop(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float pop = 1.2f + 0.3f * sinf(params.frame_count * 0.08f);
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        for (int j = 0; j < 3; ++j) {
            float diff = data[idx + j] - gray;
            data[idx + j] = (unsigned char)fmaxf(0.0f, fminf(255.0f, gray + diff * pop));
        }
    }
    __device__ void processVignetteFlash(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float max_dist = sqrtf(cx * cx + cy * cy);
        float vignette = 1.0f - (dist / max_dist) * 0.5f;
        float flash = 0.8f + 0.2f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * vignette * flash);
    }
    __device__ void processVoronoiShatter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cell_size = 30 + (params.frame_count % 20);
        int cx = ((x / cell_size) * cell_size + cell_size / 2) % width;
        int cy = ((y / cell_size) * cell_size + cell_size / 2) % height;
        float r = gpu_rand(cx, cy, params.seed);
        cx += (int)((r - 0.5f) * cell_size * 0.5f);
        cy += (int)((r - 0.5f) * cell_size * 0.5f);
        cx = (cx + width) % width;
        cy = (cy + height) % height;
        int src_idx = cy * step + cx * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processWarpSpeed(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float warp = 1.0f + 0.3f * sinf(params.frame_count * 0.1f - dist * 0.02f);
        int src_x = (int)(cx + dx * warp) % width;
        int src_y = (int)(cy + dy * warp) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processWaterColor(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count / 5);
        int ox = (int)((r - 0.5f) * 6);
        int oy = (int)((gpu_rand(y, x, params.seed) - 0.5f) * 6);
        int src_x = (x + ox + width) % width;
        int src_y = (y + oy + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            float blend = (data[idx + j] + data[src_idx + j]) / 2.0f;
            data[idx + j] = (unsigned char)fminf(255.0f, blend * 1.1f);
        }
    }
    __device__ void processWaveCollapse(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave1 = sinf(x * 0.03f + params.frame_count * 0.1f);
        float wave2 = cosf(y * 0.03f + params.frame_count * 0.08f);
        float collapse = wave1 * wave2 * 15.0f;
        int src_x = ((int)(x + collapse) + width) % width;
        int src_y = ((int)(y + collapse) + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processWebPattern(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx);
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        int radial = (int)(dist / 20 + params.frame_count * 0.1f) % 2;
        int angular = (int)((angle + 3.14159f) * 8 / 3.14159f) % 2;
        if (radial == angular) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.7f);
        }
    }
    __device__ void processWhirlpoolSpin(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float angle = atan2f(dy, dx) + dist * 0.01f * sinf(params.frame_count * 0.05f);
        int src_x = (int)(cx + dist * cosf(angle)) % width;
        int src_y = (int)(cy + dist * sinf(angle)) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processWindBlast(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int intensity = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int wind = (255 - intensity) / 10 + (params.frame_count % 10);
        int src_x = (x - wind + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)((data[idx + j] + data[src_idx + j]) / 2);
    }
    __device__ void processWireframePulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int grid = 20;
        int gx = x % grid, gy = y % grid;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
        if (gx < 2 || gy < 2) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + pulse * 0.5f) + 128 * pulse);
        } else {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.5f);
        }
    }
    __device__ void processXRayFlash(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] * 30 + data[idx + 1] * 59 + data[idx + 2] * 11) / 100;
        float flash = 0.5f + 0.5f * sinf(params.frame_count * 0.1f);
        if ((params.frame_count / 10) % 2 == 0) {
            data[idx] = (unsigned char)(255 - gray);
            data[idx + 1] = (unsigned char)(255 - gray);
            data[idx + 2] = (unsigned char)fminf(255.0f, (255 - gray) * 1.2f);
        } else {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * flash);
        }
    }
    __device__ void processZebraStripe(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int stripe = (x + y + params.frame_count) % 20;
        float factor = (stripe < 10) ? 1.2f : 0.8f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
    }
    __device__ void processZenRipple(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float ripple = sinf(dist * 0.05f - params.frame_count * 0.03f) * 0.2f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + j] * (1.0f + ripple)));
    }
    __device__ void processZigzagWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int period = 40;
        int phase = (y / period) % 2;
        int offset = (phase == 0) ? (params.frame_count % period) : -(params.frame_count % period);
        int src_x = (x + offset + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processZoomPulse(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float zoom = 1.0f + 0.1f * sinf(params.frame_count * 0.08f);
        int src_x = (int)(cx + (x - cx) / zoom) % width;
        int src_y = (int)(cy + (y - cy) / zoom) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processZoneTint(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int zone_x = (x * 3) / width;
        int zone_y = (y * 3) / height;
        int zone = zone_x + zone_y * 3;
        zone = (zone + params.frame_count / 20) % 9;
        float tints[9][3] = {{1.3f,0.9f,0.9f},{0.9f,1.3f,0.9f},{0.9f,0.9f,1.3f},
                            {1.2f,1.2f,0.8f},{1.2f,0.8f,1.2f},{0.8f,1.2f,1.2f},
                            {1.1f,1.0f,0.9f},{0.9f,1.1f,1.0f},{1.0f,0.9f,1.1f}};
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * tints[zone][2]);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * tints[zone][1]);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * tints[zone][0]);
    }
    __device__ void processAcidDrip(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drip = sinf(x * 0.1f + params.frame_count * 0.05f) * 20.0f;
        int src_y = ((int)(y + drip * (float)y / height) + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * 1.2f);
    }
    __device__ void processAuroraWave(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wave = sinf(x * 0.02f + params.frame_count * 0.05f) * sinf(y * 0.01f + params.frame_count * 0.03f);
        float aurora = (wave + 1.0f) * 0.5f * (1.0f - (float)y / height);
        if (aurora > 0.3f) {
            data[idx] = (unsigned char)fminf(255.0f, data[idx] + 50 * aurora);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 100 * aurora);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + 30 * aurora);
        }
    }

    
    __device__ void processBandPass(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int low = 60 + (params.frame_count % 60);
        int high = 180 + (params.frame_count % 60);
        for (int j = 0; j < 3; ++j) {
            if (data[idx + j] < low) data[idx + j] = (unsigned char)(data[idx + j] * 0.5f);
            else if (data[idx + j] > high) data[idx + j] = (unsigned char)(255 - (data[idx + j] - high));
        }
    }
    __device__ void processBilinearStretch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float sx = 1.0f + 0.3f * sinf(params.frame_count * 0.05f);
        float sy = 1.0f + 0.3f * cosf(params.frame_count * 0.05f);
        int src_x = (int)(x / sx) % width;
        int src_y = (int)(y / sy) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processBleedThrough(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bleed = 3 + (params.frame_count % 5);
        for (int j = 0; j < 3; ++j) {
            int sum = data[idx + j];
            for (int b = 1; b <= bleed; ++b) {
                int bx = (x + b) % width;
                sum += data[y * step + bx * 4 + j] / (b + 1);
            }
            data[idx + j] = (unsigned char)fminf(255.0f, (float)sum);
        }
    }
    __device__ void processBlockShatter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bsize = 16 + (params.frame_count % 16);
        int bx = x / bsize, by = y / bsize;
        float r = gpu_rand(bx, by, params.seed + params.frame_count / 10);
        if (r < 0.3f) {
            int ox = (int)((r - 0.15f) * bsize);
            int oy = (int)((r - 0.15f) * bsize);
            int src_x = (x + ox + width) % width;
            int src_y = (y + oy + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processBlurMask(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int threshold = 128 + (int)(64.0f * sinf(params.frame_count * 0.05f));
        if (gray > threshold && x > 1 && y > 1 && x < width - 2 && y < height - 2) {
            for (int j = 0; j < 3; ++j) {
                int sum = data[idx + j] * 4;
                sum += data[(y - 1) * step + x * 4 + j];
                sum += data[(y + 1) * step + x * 4 + j];
                sum += data[y * step + (x - 1) * 4 + j];
                sum += data[y * step + (x + 1) * 4 + j];
                data[idx + j] = (unsigned char)(sum / 8);
            }
        }
    }
    __device__ void processBokehBlur(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float max_dist = sqrtf(cx * cx + cy * cy);
        int blur_size = (int)(dist / max_dist * 5.0f * (1.0f + sinf(params.frame_count * 0.05f)));
        if (blur_size > 0 && x > blur_size && y > blur_size && x < width - blur_size && y < height - blur_size) {
            for (int j = 0; j < 3; ++j) {
                int sum = 0, count = 0;
                for (int dy = -blur_size; dy <= blur_size; dy += 2) {
                    for (int dx = -blur_size; dx <= blur_size; dx += 2) {
                        sum += data[(y + dy) * step + (x + dx) * 4 + j];
                        count++;
                    }
                }
                data[idx + j] = (unsigned char)(sum / count);
            }
        }
    }
    __device__ void processBounceWave(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float bounce = fabsf(sinf(x * 0.03f + params.frame_count * 0.1f)) * 15.0f;
        int src_y = ((int)(y + bounce) + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processBrokenGlass(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shard_x = x / 25, shard_y = y / 25;
        float r = gpu_rand(shard_x, shard_y, params.seed);
        float angle = r * 6.28318f;
        float offset = (r - 0.5f) * 10.0f * sinf(params.frame_count * 0.05f);
        int src_x = (int)(x + offset * cosf(angle)) % width;
        int src_y = (int)(y + offset * sinf(angle)) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processBubbleWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float bubble = sinf(dist * 0.05f - params.frame_count * 0.1f) * 10.0f;
        float angle = atan2f(dy, dx);
        int src_x = (int)(x + bubble * cosf(angle)) % width;
        int src_y = (int)(y + bubble * sinf(angle)) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processCRTCurvature(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float nx = (x - width / 2.0f) / (width / 2.0f);
        float ny = (y - height / 2.0f) / (height / 2.0f);
        float curve = 0.1f * (1.0f + 0.5f * sinf(params.frame_count * 0.03f));
        nx = nx * (1.0f + curve * ny * ny);
        ny = ny * (1.0f + curve * nx * nx);
        int src_x = (int)((nx + 1.0f) * width / 2.0f) % width;
        int src_y = (int)((ny + 1.0f) * height / 2.0f) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processCascadeBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int cascade = (y + params.frame_count) % 60;
        float factor = (cascade < 30) ? cascade / 30.0f : (60 - cascade) / 30.0f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(data[idx + j] * (0.5f + factor * 0.5f));
        }
    }
    __device__ void processCelShade(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int levels = 4 + (params.frame_count % 4);
        int step_size = 256 / levels;
        for (int j = 0; j < 3; ++j) {
            int val = data[idx + j];
            val = (val / step_size) * step_size + step_size / 2;
            data[idx + j] = (unsigned char)fminf(255.0f, (float)val);
        }
    }
    __device__ void processChainReaction(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int ring = params.frame_count % 100;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (fabsf(dist - ring * 3) < 5.0f) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * 1.5f);
        }
    }
    __device__ void processChannelDelay(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int r_delay = 5 + (params.frame_count % 10);
        int b_delay = -(5 + (params.frame_count % 10));
        data[idx + 2] = data[y * step + ((x + r_delay + width) % width) * 4 + 2];
        data[idx] = data[y * step + ((x + b_delay + width) % width) * 4];
    }
    __device__ void processChromaBleed(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int bleed = 5 + (params.frame_count % 8);
        int sum_r = 0, sum_g = 0, sum_b = 0;
        for (int i = 0; i < bleed; ++i) {
            int bx = (x + i) % width;
            sum_r += data[y * step + bx * 4 + 2];
            sum_g += data[y * step + bx * 4 + 1];
            sum_b += data[y * step + bx * 4];
        }
        data[idx + 2] = (unsigned char)(sum_r / bleed);
        data[idx + 1] = data[idx + 1];
        data[idx] = (unsigned char)(sum_b / bleed);
    }
    __device__ void processCircuitTrace(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int grid = 12;
        int gx = x % grid, gy = y % grid;
        int node_x = x / grid, node_y = y / grid;
        float r = gpu_rand(node_x, node_y, params.seed);
        int trace = (r < 0.5f) ? (gx < 2) : (gy < 2);
        if (trace) {
            float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.1f + node_x + node_y);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 150 * pulse);
        }
    }
    __device__ void processClockWipe(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float angle = atan2f(y - cy, x - cx) + 3.14159f;
        float wipe_angle = fmodf(params.frame_count * 0.05f, 6.28318f);
        if (angle < wipe_angle) {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255 - data[idx + j];
        }
    }
    __device__ void processCloudShadow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cloud = sinf(x * 0.01f + params.frame_count * 0.02f) * cosf(y * 0.015f + params.frame_count * 0.015f);
        float shadow = (cloud > 0.3f) ? 0.6f : 1.0f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * shadow);
    }
    __device__ void processColorBurn(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float intensity = (data[idx] + data[idx + 1] + data[idx + 2]) / 765.0f;
        float burn = intensity * intensity;
        int channel = params.frame_count % 3;
        data[idx + channel] = (unsigned char)(data[idx + channel] * burn + 255 * (1.0f - burn) * 0.3f);
    }
    __device__ void processColorHalves(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int split = width / 2 + (int)(50.0f * sinf(params.frame_count * 0.03f));
        if (x < split) {
            data[idx] = (unsigned char)(data[idx] * 0.5f);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 1.5f);
        } else {
            data[idx + 2] = (unsigned char)(data[idx + 2] * 0.5f);
            data[idx] = (unsigned char)fminf(255.0f, data[idx] * 1.5f);
        }
    }
    __device__ void processComicDots(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int dot_size = 6;
        int cx = (x / dot_size) * dot_size + dot_size / 2;
        int cy = (y / dot_size) * dot_size + dot_size / 2;
        float dist = sqrtf((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        float radius = (gray / 255.0f) * dot_size * 0.7f;
        if (dist < radius) {
            for (int j = 0; j < 3; ++j) data[idx + j] = 0;
        } else {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255;
        }
    }
    __device__ void processConcentricPulse(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float pulse = sinf(dist * 0.05f - params.frame_count * 0.15f);
        float factor = 0.7f + pulse * 0.3f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * factor);
    }
    __device__ void processCopperTone(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] * 30 + data[idx + 1] * 59 + data[idx + 2] * 11) / 100;
        float pulse = 0.8f + 0.2f * sinf(params.frame_count * 0.05f);
        data[idx] = (unsigned char)(gray * 0.4f * pulse);
        data[idx + 1] = (unsigned char)(gray * 0.6f * pulse);
        data[idx + 2] = (unsigned char)(gray * 0.9f * pulse);
    }
    __device__ void processCornerStretch(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float nx = (float)x / width, ny = (float)y / height;
        float stretch = 0.2f * sinf(params.frame_count * 0.05f);
        float corner_dist = fminf(fminf(nx, 1.0f - nx), fminf(ny, 1.0f - ny));
        float factor = 1.0f + stretch * (1.0f - corner_dist * 4.0f);
        factor = fmaxf(0.5f, fminf(1.5f, factor));
        int src_x = (int)(width / 2.0f + (x - width / 2.0f) * factor) % width;
        int src_y = (int)(height / 2.0f + (y - height / 2.0f) * factor) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processCosmicDust(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed + params.frame_count / 3);
        if (r < 0.01f) {
            float bright = r * 100.0f;
            data[idx] = (unsigned char)fminf(255.0f, data[idx] + 150 * bright);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 100 * bright);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + 200 * bright);
        }
    }
    __device__ void processCrossBlur(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int size = 3 + (params.frame_count % 5);
        if (x > size && y > size && x < width - size && y < height - size) {
            for (int j = 0; j < 3; ++j) {
                int sum = data[idx + j];
                for (int i = 1; i <= size; ++i) {
                    sum += data[y * step + (x - i) * 4 + j];
                    sum += data[y * step + (x + i) * 4 + j];
                    sum += data[(y - i) * step + x * 4 + j];
                    sum += data[(y + i) * step + x * 4 + j];
                }
                data[idx + j] = (unsigned char)(sum / (size * 4 + 1));
            }
        }
    }
    __device__ void processCrossProcess(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = data[idx + 2] / 255.0f;
        float g = data[idx + 1] / 255.0f;
        float b = data[idx] / 255.0f;
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.05f);
        r = powf(r, 0.8f + pulse * 0.2f);
        g = powf(g, 1.0f);
        b = powf(b, 1.2f - pulse * 0.2f);
        data[idx + 2] = (unsigned char)(fminf(1.0f, r * 1.1f) * 255);
        data[idx + 1] = (unsigned char)(g * 255);
        data[idx] = (unsigned char)(fminf(1.0f, b * 0.9f) * 255);
    }
    __device__ void processCrystalEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int facet_size = 20 + (params.frame_count % 20);
        int fx = (x / facet_size) * facet_size;
        int fy = (y / facet_size) * facet_size;
        int edge_x = x - fx, edge_y = y - fy;
        if (edge_x < 2 || edge_y < 2 || edge_x > facet_size - 3 || edge_y > facet_size - 3) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * 1.5f);
        }
    }
    __device__ void processCubeRotate(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float angle = params.frame_count * 0.02f;
        float cos_a = cosf(angle), sin_a = sinf(angle);
        int src_x = (int)(cx + dx * cos_a - dy * sin_a) % width;
        int src_y = (int)(cy + dx * sin_a + dy * cos_a) % height;
        src_x = (src_x + width) % width;
        src_y = (src_y + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processCurtainReveal(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int curtain = (params.frame_count * 3) % width;
        if (x < curtain) {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255 - data[idx + j];
        }
    }
    __device__ void processCyberPunk(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int scan = (y + params.frame_count * 2) % 4;
        float cyan = (scan < 2) ? 1.3f : 0.9f;
        float magenta = (scan >= 2) ? 1.3f : 0.9f;
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * cyan);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * cyan);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * magenta);
    }
    __device__ void processDataCorrupt(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 8, y / 4, params.seed + params.frame_count / 5);
        if (r < 0.05f) {
            int shift = (int)(r * 500) % 50 - 25;
            int src_x = (x + shift + width) % width;
            int src_idx = y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processDebrisField(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 10, y / 10, params.seed);
        if (r < 0.15f) {
            int ox = (int)((r - 0.075f) * 20 + params.frame_count * 0.5f) % 20 - 10;
            int oy = (int)((r - 0.075f) * 20) % 20 - 10;
            int src_x = (x + ox + width) % width;
            int src_y = (y + oy + height) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processDeepFry(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float intensity = 1.5f + 0.5f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) {
            float val = (data[idx + j] - 128) * intensity + 128;
            val = fmaxf(0.0f, fminf(255.0f, val));
            if (val > 200) val = 255;
            else if (val < 55) val = 0;
            data[idx + j] = (unsigned char)val;
        }
    }
    __device__ void processDesyncRGB(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int r_off_x = (params.frame_count % 10) - 5;
        int g_off_y = (params.frame_count % 8) - 4;
        int b_off_x = -((params.frame_count % 10) - 5);
        int r_x = (x + r_off_x + width) % width;
        int g_y = (y + g_off_y + height) % height;
        int b_x = (x + b_off_x + width) % width;
        data[idx + 2] = data[y * step + r_x * 4 + 2];
        data[idx + 1] = data[g_y * step + x * 4 + 1];
        data[idx] = data[y * step + b_x * 4];
    }
    __device__ void processDiagonalWipe(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int diag = x + y;
        int wipe = (params.frame_count * 5) % (width + height);
        if (diag < wipe) {
            for (int j = 0; j < 3; ++j) data[idx + j] = 255 - data[idx + j];
        }
    }
    __device__ void processDigitalArtifact(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int block = (x / 8) + (y / 8) * 100;
        float r = gpu_rand(block, params.seed, params.frame_count / 10);
        if (r < 0.02f) {
            unsigned char artifact = (unsigned char)((int)(r * 12750) % 256);
            for (int j = 0; j < 3; ++j) data[idx + j] = artifact;
        }
    }
    __device__ void processDimensionRift(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float rift = sinf(dist * 0.03f - params.frame_count * 0.1f);
        if (rift > 0.7f) {
            int src_x = (width - 1 - x);
            int src_y = (height - 1 - y);
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = (data[idx + j] + data[src_idx + j]) / 2;
        }
    }
    __device__ void processDotCrawl(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int crawl = ((x + y + params.frame_count) % 4);
        if (crawl == 0) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.8f);
        }
    }
    __device__ void processDualTone(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        int phase = (params.frame_count / 30) % 3;
        if (gray < 128) {
            if (phase == 0) { data[idx] = gray; data[idx + 1] = 0; data[idx + 2] = gray; }
            else if (phase == 1) { data[idx] = 0; data[idx + 1] = gray; data[idx + 2] = gray; }
            else { data[idx] = gray; data[idx + 1] = gray; data[idx + 2] = 0; }
        } else {
            if (phase == 0) { data[idx] = 255 - gray; data[idx + 1] = 255; data[idx + 2] = 255 - gray; }
            else if (phase == 1) { data[idx] = 255; data[idx + 1] = 255 - gray; data[idx + 2] = 255 - gray; }
            else { data[idx] = 255 - gray; data[idx + 1] = 255 - gray; data[idx + 2] = 255; }
        }
    }
    __device__ void processEchoFade(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float fade = 0.9f - (params.frame_count % 30) * 0.02f;
        unsigned char* prev = allFrames[3 % params.numFrames];
        if (prev) {
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - fade) + prev[idx + j] * fade);
        }
    }
    __device__ void processEdgeMelt(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float melt = sinf(x * 0.02f + params.frame_count * 0.05f) * 10.0f;
        int src_y = ((int)(y + melt) % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processElasticWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float elastic = sinf(dist * 0.05f - params.frame_count * 0.08f) * 0.15f;
        int src_x = (int)(x + dx * elastic) % width; if (src_x < 0) src_x += width;
        int src_y = (int)(y + dy * elastic) % height; if (src_y < 0) src_y += height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processEmberGlow(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float glow = 0.5f + 0.5f * sinf(params.frame_count * 0.1f + x * 0.01f);
        data[idx] = (unsigned char)fminf(255.0f, data[idx] * 0.8f);
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (0.5f + glow * 0.3f));
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (0.8f + glow * 0.4f));
    }
    __device__ void processEntropyShift(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float entropy = gpu_rand(x, y, params.seed + params.frame_count);
        int shift = (int)(entropy * 50.0f) - 25;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + j] + shift * (j == params.frame_count % 3 ? 1.0f : 0.3f)));
    }
    __device__ void processErosionBlend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        if (x <= 1 || x >= width - 2 || y <= 1 || y >= height - 2) return;
        int idx = y * step + x * 4;
        unsigned char minVal[3] = {255, 255, 255};
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int n_idx = (y + dy) * step + (x + dx) * 4;
                for (int j = 0; j < 3; ++j) if (data[n_idx + j] < minVal[j]) minVal[j] = data[n_idx + j];
            }
        }
        float blend = 0.5f + 0.3f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + minVal[j] * blend);
    }
    __device__ void processExplosionBurst(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy);
        float burst = fmaxf(0.0f, 1.0f - dist / (width * 0.5f));
        float pulse = 0.5f + 0.5f * sinf(params.frame_count * 0.15f - dist * 0.02f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + burst * pulse * 0.5f));
    }
    __device__ void processFacetMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int facet_size = 40 + (params.frame_count % 20);
        int fx = (x / facet_size) % 2;
        int fy = (y / facet_size) % 2;
        int src_x = fx ? (facet_size - 1 - (x % facet_size)) + (x / facet_size) * facet_size : x;
        int src_y = fy ? (facet_size - 1 - (y % facet_size)) + (y / facet_size) * facet_size : y;
        src_x = (src_x % width + width) % width;
        src_y = (src_y % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processFadeStreak(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float streak = (float)(x + params.frame_count * 3) / width;
        streak = fmodf(streak, 1.0f);
        float fade = 1.0f - streak * 0.5f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * fade);
    }
    __device__ void processFeatherEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float edge_x = fminf((float)x / 50.0f, (float)(width - x) / 50.0f);
        float edge_y = fminf((float)y / 50.0f, (float)(height - y) / 50.0f);
        float feather = fminf(1.0f, fminf(edge_x, edge_y));
        float pulse = 0.8f + 0.2f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * feather * pulse);
    }
    __device__ void processFlashFreeze(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int flash = (params.frame_count / 15) % 4;
        if (flash == 0) {
            int avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            data[idx] = (unsigned char)fminf(255.0f, avg + 50);
            data[idx + 1] = (unsigned char)fminf(255.0f, avg + 60);
            data[idx + 2] = (unsigned char)fminf(255.0f, avg + 40);
        }
    }
    __device__ void processFlipMirror(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int phase = (params.frame_count / 20) % 4;
        int src_x = x, src_y = y;
        if (phase == 1) src_x = width - 1 - x;
        else if (phase == 2) src_y = height - 1 - y;
        else if (phase == 3) { src_x = width - 1 - x; src_y = height - 1 - y; }
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (data[idx + j] + data[src_idx + j]) / 2;
    }
    __device__ void processFloatDrift(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float drift_x = sinf(y * 0.02f + params.frame_count * 0.03f) * 8.0f;
        float drift_y = cosf(x * 0.02f + params.frame_count * 0.04f) * 8.0f;
        int src_x = ((int)(x + drift_x) % width + width) % width;
        int src_y = ((int)(y + drift_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processFlowField(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float angle = sinf(x * 0.01f) * cosf(y * 0.01f) * 3.14159f + params.frame_count * 0.02f;
        float flow = 5.0f;
        int src_x = ((int)(x + cosf(angle) * flow) % width + width) % width;
        int src_y = ((int)(y + sinf(angle) * flow) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (data[idx + j] + data[src_idx + j]) / 2;
    }
    __device__ void processFoldWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int fold_x = (x < width / 2) ? x * 2 : (width - 1 - x) * 2;
        int fold_y = (y < height / 2) ? y * 2 : (height - 1 - y) * 2;
        fold_x = fold_x % width; fold_y = fold_y % height;
        int src_idx = fold_y * step + fold_x * 4;
        float blend = 0.5f + 0.3f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + data[src_idx + j] * blend);
    }
    __device__ void processFragmentScatter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x / 8, y / 8, params.seed);
        int scatter_x = (int)(r * 16.0f) - 8;
        int scatter_y = (int)(gpu_rand(y / 8, x / 8, params.seed) * 16.0f) - 8;
        int src_x = ((x + scatter_x) % width + width) % width;
        int src_y = ((y + scatter_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processFrequencyPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float freq1 = sinf(x * 0.05f + params.frame_count * 0.1f);
        float freq2 = sinf(y * 0.07f + params.frame_count * 0.08f);
        float pulse = 0.7f + 0.3f * freq1 * freq2;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * pulse);
    }
    __device__ void processFrostBite(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float frost = gpu_rand(x, y, params.seed + params.frame_count / 5);
        if (frost > 0.9f) {
            data[idx] = (unsigned char)fminf(255.0f, data[idx] + 80);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + 90);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + 100);
        } else {
            float tint = 0.9f + frost * 0.1f;
            data[idx + 2] = (unsigned char)(data[idx + 2] * tint);
        }
    }
    __device__ void processFuseBlend(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* f1 = allFrames[2 % params.numFrames];
        unsigned char* f2 = allFrames[5 % params.numFrames];
        if (f1 && f2) {
            float fuse = 0.5f + 0.3f * sinf(params.frame_count * 0.05f);
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.4f + f1[idx + j] * fuse * 0.3f + f2[idx + j] * (1.0f - fuse) * 0.3f);
        }
    }
    __device__ void processGalaxySpiral(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float angle = atan2f(dy, dx) + params.frame_count * 0.02f;
        float dist = sqrtf(dx * dx + dy * dy);
        float spiral = sinf(angle * 3.0f - dist * 0.03f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (0.7f + 0.3f * spiral));
    }
    __device__ void processGelWobble(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wobble_x = sinf(y * 0.03f + params.frame_count * 0.08f) * 6.0f;
        float wobble_y = sinf(x * 0.03f + params.frame_count * 0.06f) * 6.0f;
        int src_x = ((int)(x + wobble_x) % width + width) % width;
        int src_y = ((int)(y + wobble_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = (data[idx + j] + data[src_idx + j]) / 2;
    }
    __device__ void processGhostEcho(int x, int y, unsigned char* data, unsigned char** allFrames, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = 20 + (int)(sinf(params.frame_count * 0.05f) * 10.0f);
        int ghost_x = (x + offset) % width;
        unsigned char* prev = allFrames[4 % params.numFrames];
        if (prev) {
            int ghost_idx = y * step + ghost_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.7f + prev[ghost_idx + j] * 0.3f);
        }
    }
    __device__ void processGlassShatter(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shard_size = 25 + (params.frame_count % 15);
        int sx = x / shard_size, sy = y / shard_size;
        float r = gpu_rand(sx, sy, params.seed);
        int offset_x = (int)((r - 0.5f) * shard_size * 0.5f);
        int offset_y = (int)((gpu_rand(sy, sx, params.seed) - 0.5f) * shard_size * 0.5f);
        int src_x = ((x + offset_x) % width + width) % width;
        int src_y = ((y + offset_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processGlimmerPulse(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float glimmer = gpu_rand(x, y, params.seed + params.frame_count / 3);
        float pulse = 0.8f + 0.4f * sinf(params.frame_count * 0.15f + glimmer * 6.28f);
        if (glimmer > 0.95f) pulse = 1.5f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * pulse);
    }
    __device__ void processGlitchMosaic(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int tile = 16 + (params.frame_count % 16);
        int tx = (x / tile) * tile, ty = (y / tile) * tile;
        float r = gpu_rand(tx, ty, params.seed + params.frame_count / 10);
        if (r > 0.7f) {
            int src_x = (tx + (int)(r * tile * 3)) % width;
            int src_y = (ty + (int)(gpu_rand(ty, tx, params.seed) * tile * 3)) % height;
            int src_idx = src_y * step + src_x * 4;
            for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        }
    }
    __device__ void processGlowEdge(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        if (x <= 1 || x >= width - 2 || y <= 1 || y >= height - 2) return;
        int idx = y * step + x * 4;
        float gx = 0, gy = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int n_idx = (y + dy) * step + (x + dx) * 4;
                float val = (data[n_idx] + data[n_idx + 1] + data[n_idx + 2]) / 3.0f;
                gx += val * dx; gy += val * dy;
            }
        }
        float edge = sqrtf(gx * gx + gy * gy) / 255.0f;
        float glow = 1.0f + edge * sinf(params.frame_count * 0.1f) * 0.5f;
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * glow);
    }
    __device__ void processGradientMelt(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float melt = (float)y / height + sinf(x * 0.02f + params.frame_count * 0.05f) * 0.1f;
        float fade = fmaxf(0.3f, 1.0f - melt * 0.7f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * fade);
    }
    __device__ void processGrainStorm(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float grain = gpu_rand(x, y, params.seed + params.frame_count) * 2.0f - 1.0f;
        float intensity = 30.0f + 20.0f * sinf(params.frame_count * 0.1f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, data[idx + j] + grain * intensity));
    }
    __device__ void processGravityPull(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dx = x - cx, dy = y - cy;
        float dist = sqrtf(dx * dx + dy * dy) + 1.0f;
        float pull = 50.0f / dist * sinf(params.frame_count * 0.05f);
        int src_x = (int)(x - dx * pull / dist) % width; if (src_x < 0) src_x += width;
        int src_y = (int)(y - dy * pull / dist) % height; if (src_y < 0) src_y += height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processGridWarp(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int grid = 32;
        float warp_x = sinf((x / grid) * 0.5f + params.frame_count * 0.05f) * 8.0f;
        float warp_y = cosf((y / grid) * 0.5f + params.frame_count * 0.04f) * 8.0f;
        int src_x = ((int)(x + warp_x) % width + width) % width;
        int src_y = ((int)(y + warp_y) % height + height) % height;
        int src_idx = src_y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processHaloRing(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f, cy = height / 2.0f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float ring_dist = 100.0f + 50.0f * sinf(params.frame_count * 0.03f);
        float halo = expf(-fabsf(dist - ring_dist) / 20.0f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * (1.0f + halo * 0.5f));
    }
    __device__ void processHarshLight(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float harsh = 1.3f + 0.3f * sinf(params.frame_count * 0.08f);
        for (int j = 0; j < 3; ++j) {
            float val = (data[idx + j] - 128) * harsh + 128;
            data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    }
    __device__ void processHazeLayer(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float haze = 0.3f + 0.2f * sinf(params.frame_count * 0.03f);
        float depth = (float)y / height;
        float fog = haze * depth;
        unsigned char fog_color = 180 + (unsigned char)(30 * sinf(params.frame_count * 0.05f));
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - fog) + fog_color * fog);
    }
    __device__ void processHeatRipple(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float ripple = sinf(y * 0.05f + params.frame_count * 0.15f) * (3.0f + 2.0f * sinf(params.frame_count * 0.02f));
        int src_x = ((int)(x + ripple) % width + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * 1.1f);
    }
    __device__ void processHexagonBlur(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int hex_size = 20 + (params.frame_count % 10);
        int hx = x / hex_size, hy = y / hex_size;
        int cx = hx * hex_size + hex_size / 2;
        int cy = hy * hex_size + hex_size / 2;
        cx = (cx % width + width) % width;
        cy = (cy % height + height) % height;
        int src_idx = cy * step + cx * 4;
        float blend = 0.6f + 0.2f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * (1.0f - blend) + data[src_idx + j] * blend);
    }
    __device__ void processHighContrast(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float contrast = 1.5f + 0.5f * sinf(params.frame_count * 0.05f);
        for (int j = 0; j < 3; ++j) {
            float val = (data[idx + j] - 128) * contrast + 128;
            data[idx + j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    }
    __device__ void processHologramScan(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int scanline = (params.frame_count * 3) % height;
        float dist = fabsf(y - scanline);
        if (dist < 30) {
            float glow = 1.0f - dist / 30.0f;
            data[idx] = (unsigned char)fminf(255.0f, data[idx] + glow * 100);
            data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] + glow * 150);
            data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] + glow * 80);
        }
        if (y % 3 == 0) for (int j = 0; j < 3; ++j) data[idx + j] = (unsigned char)(data[idx + j] * 0.8f);
    }
    __device__ void processHorizonBend(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float bend = sinf(x * 0.01f + params.frame_count * 0.03f) * 30.0f;
        int src_y = (int)(y + bend * (1.0f - (float)y / height)) % height;
        if (src_y < 0) src_y += height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) data[idx + j] = data[src_idx + j];
    }
    __device__ void processHotSpot(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float cx = width / 2.0f + sinf(params.frame_count * 0.02f) * width * 0.3f;
        float cy = height / 2.0f + cosf(params.frame_count * 0.03f) * height * 0.3f;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        float hot = expf(-dist / 80.0f);
        data[idx + 2] = (unsigned char)fminf(255.0f, data[idx + 2] * (1.0f + hot * 0.8f));
        data[idx + 1] = (unsigned char)fminf(255.0f, data[idx + 1] * (1.0f + hot * 0.4f));
    }
    __device__ void processHueWobble(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float wobble = sinf(x * 0.02f + y * 0.02f + params.frame_count * 0.1f) * 30.0f;
        int shift = (int)wobble;
        unsigned char temp[3] = {data[idx], data[idx + 1], data[idx + 2]};
        int ch = (params.frame_count / 10) % 3;
        data[idx + ch] = (unsigned char)fminf(255.0f, fmaxf(0.0f, temp[ch] + shift));
        data[idx + (ch + 1) % 3] = (unsigned char)fminf(255.0f, fmaxf(0.0f, temp[(ch + 1) % 3] - shift / 2));
    }

    __global__ void unifiedFilterKernel(Filter *filters, size_t count, unsigned char* data, unsigned char** allFrames, int width, int height, size_t step, FilterParams params) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        for (int i = 0; i < count; ++i) {
            int idx = y * step + x * 4;
            switch (filters[i].index) {
                case 0: processSelfAlphaBlend(x, y, data, step, params); break;
                case 1: case 2: processMedianBlend(x, y, data, allFrames, step, params); break;
                case 3: processSquareBlockResize(x, y, data, allFrames, step, params); break;
                case 4: processSelfScaleRefined(x, y, data, step, params); break;
                case 5: processStrangeGlitch(x, y, data, allFrames, step, params); break;
                case 6: processMatrixOutline(x, y, data, allFrames, step, params); break;
                case 7: processAuraTrails(x, y, data, allFrames, step, params); break;
                case 8: {
                    int px = (width - x - 1);
                    int py = (height - y - 1);
                    int i0 = py * step + x * 4;
                    int i1 = py * step + px * 4;
                    int i2 = y * step + px * 4;
                    for (int j = 0; j < 3; j++) {
                        data[idx + (2 - j)] = (unsigned char)fminf(255.0f, (data[idx + j] * 0.25f + data[i0 + j] * 0.25f + data[i1 + j] * 0.25f + data[i2 + j] * 0.25f));
                    }
                } break;
                case 9: processSquareShrink(x, y, data, allFrames, width, height, step, params); break;
                case 10: {
                    unsigned char* ghost = allFrames[0];
                    if (ghost) {
                        for (int j = 0; j < 3; j++) {
                            data[idx + j] = (unsigned char)fminf(255.0f, (data[idx + j] * (1 - params.alpha) + ghost[idx + j] * params.alpha));
                        }
                    }
                } break;
                case 11: case 12: case 13: {
                    int sw = (filters[i].index == 11) ? 8 : (filters[i].index == 12 ? 16 : 32);
                    unsigned char* ref = allFrames[(x / sw) % params.numFrames];
                    if (ref) {
                        for (int j = 0; j < 3; j++) {
                            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + ref[idx + j] * 0.5f);
                        }
                    }
                } break;
                case 14: {
                    float a = (0.5f / height) * y;
                    for (int j = 0; j < 3; j++) {
                        data[idx + j] = (unsigned char)fminf(255.0f, (0.5f * data[idx + j]) + (a * params.float_param1));
                    }
                } break;
                case 15: case 16: {
                    bool hor = (filters[i].index == 15);
                    unsigned char* ref = allFrames[(hor ? y : x) % params.numFrames];
                    if (ref) {
                        for (int j = 0; j < 3; j++) {
                            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + ref[idx + j] * 0.5f);
                        }
                    }
                } break;
                case 17: processWaveTrails(x, y, data, allFrames, step, params); break;
                case 18: {
                    unsigned char* ref = allFrames[(y * width + x + params.seed) % params.numFrames];
                    if (ref) {
                        for (int j = 0; j < 3; j++) {
                            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + ref[idx + j] * 0.5f);
                        }
                    }
                } break;
                case 19: {
                    processAuraTrails(x, y, data, allFrames, step, params);
                    processWaveTrails(x, y, data, allFrames, step, params);
                } break;
                case 20: processParticleSlide(x, y, data, allFrames, width, height, step, params); break;
                case 21: {
                    int sw = 16;
                    int sh = 16;
                    int sx = (x / sw) * sw;
                    int sy = (y / sh) * sh;
                    for (int j = 0; j < 3; j++) {
                        data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[sy * step + sx * 4 + j]);
                    }
                } break;
                case 22: {
                    int sw = params.sw;
                    int sh = params.sh;
                    int sx = (x / sw) * sw;
                    int sy = (y / sh) * sh;
                    for (int j = 0; j < 3; j++) {
                        data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[sy * step + sx * 4 + j]);
                    }
                } break;
                case 23: processRGBShiftTrails(x, y, data, allFrames, step, params); break;
                case 24: {
                    int ny = (y + params.int_param1) % height;
                    if (ny < 0) { ny += height; }
                    for (int j = 0; j < 3; j++) { data[idx + j] = data[ny * step + x * 4 + j]; }
                } break;
                case 25: {
                    int nx = (x + params.int_param1) % width;
                    if (nx < 0) { nx += width; }
                    for (int j = 0; j < 3; j++) { data[idx + j] = data[y * step + nx * 4 + j]; }
                } break;
                case 26: {
                    int nx = (x + params.int_param2) % width;
                    int ny = (y + params.int_param1) % height;
                    if (nx < 0) { nx += width; }
                    if (ny < 0) { ny += height; }
                    for (int j = 0; j < 3; j++) { data[idx + j] = data[ny * step + nx * 4 + j]; }
                } break;
                case 27: {
                    int nx = (x + params.int_param1) % width;
                    if (nx < 0) { nx += width; }
                    data[idx + 2] = data[y * step + nx * 4 + 2];
                } break;
                case 28: {
                    int nx = (x + params.int_param1) % width;
                    if (nx < 0) { nx += width; }
                    data[idx + 1] = data[y * step + nx * 4 + 1];
                } break;
                case 29: {
                    int nx = (x + params.int_param1) % width;
                    if (nx < 0) { nx += width; }
                    data[idx] = data[y * step + nx * 4];
                } break;
                case 30: {
                    int ny = (y + params.int_param1) % height;
                    if (ny < 0) { ny += height; }
                    data[idx + 2] = data[ny * step + x * 4 + 2];
                } break;
                case 31: {
                    int ny = (y + params.int_param1) % height;
                    if (ny < 0) { ny += height; }
                    data[idx + 1] = data[ny * step + x * 4 + 1];
                } break;
                case 32: {
                    int ny = (y + params.int_param1) % height;
                    if (ny < 0) { ny += height; }
                    data[idx] = data[ny * step + x * 4];
                } break;
                case 33: {
                    int nx = (x + ((params.int_param1 % 2 == 1) ? 0 : -16) + (int)(gpu_rand(x, y, params.seed) * 16.0f)) % width;
                    if (nx < 0) { nx += width; }
                    for (int j = 0; j < 3; j++) { data[idx + j] = data[y * step + nx * 4 + j]; }
                } break;
                case 34: {
                    int ny = (y + ((params.int_param1 % 2 == 1) ? 0 : -16) + (int)(gpu_rand(x, y, params.seed) * 16.0f)) % height;
                    if (ny < 0) { ny += height; }
                    for (int j = 0; j < 3; j++) { data[idx + j] = data[ny * step + x * 4 + j]; }
                } break;
                case 35: processTripHSV(x, y, data, step, params.frame_count); break;
                case 36: {
                    data[idx] ^= (unsigned char)params.sumB;
                    data[idx + 1] ^= (unsigned char)params.sumG;
                    data[idx + 2] ^= (unsigned char)params.sumR;
                } break;
                case 37: processSobel(x, y, data, width, height, step, false, params.threshold); break;
                case 38: processSobel(x, y, data, width, height, step, true, params.threshold); break;
                case 39: processSobel(x, y, data, width, height, step, false, params.threshold); break;
                case 40: {
                    int off = params.int_param1 % 100;
                    int sx = (x / (off + 1)) * (off + 1);
                    unsigned char* r = allFrames[params.frame_count % params.numFrames];
                    if (r) {
                        for (int j = 0; j < 3; j++) {
                            data[idx + j] = (unsigned char)(data[idx + j] * 0.5f + r[y * step + sx * 4 + j] * 0.5f);
                        }
                    }
                } break;
                case 41: processLineInLineOut4_Increase(x, y, data, allFrames, width, height, step, params); break;
                case 42: processLineInLineOut_ReverseIncrease(x, y, data, allFrames, width, height, step, params); break;
                case 43: processLineInLineOut_InvertedY(x, y, data, allFrames, width, height, step, params); break;
                case 44: { 
                    int idx = y * step + x * 4;
                    int offset = get_osc_offset(params.frame_count, 400);
                    int zpos = height - y - 1; 
                    int rpos = width - x - 1;
                    unsigned char* hist = allFrames[params.frame_count % params.numFrames];
                    if(hist && zpos >= 0 && rpos >= 0) {
                        for(int j=0; j<3; ++j) 
                            data[idx+j] = (unsigned char)((0.5f * data[idx+j]) + (0.5f * hist[zpos * step + rpos * 4 + j]));
                    }
                } break;
                case 45: processLineInLineOut_Vertical(x, y, data, allFrames, width, height, step, params); break;
                case 46: {
                    int idx = y * step + x * 4;
                    int offset = get_osc_offset(params.frame_count, 400);
                    int src_y = (y + offset) % height;
                    unsigned char* hist = allFrames[params.frame_count % params.numFrames];
                    if(hist) {
                        int src_idx = src_y * step + x * 4;
                        for(int j=0; j<3; ++j) 
                            data[src_idx+j] = (unsigned char)((0.5f * data[src_idx+j]) + (0.5f * hist[idx+j]));
                    }
                } break;
                case 47: processLineInLineOut_IncreaseImage(x, y, data, allFrames, width, height, step, params); break; 
                case 48: processSquareByRow(x, y, data, allFrames, width, height, step, params, false, false); break;
                case 49: processSquareByRow(x, y, data, allFrames, width, height, step, params, true, false); break; 
                case 50: processSquareByRow(x, y, data, allFrames, width, height, step, params, false, true); break; 
                case 51: processDivideByValue(x, y, data, allFrames, step, params); break;
                case 52: processColorCollectionSubtleStrobe(x, y, data, allFrames, step, params); break;
                case 53: processCollectionRandom(x, y, data, allFrames, step, params); break;
                case 54: processCollectionAlphaXor(x, y, data, allFrames, step, params); break;
                case 55: processColorCollection64X(x, y, data, allFrames, step, params); break;
                case 56: processColorCollectionSwitch(x, y, data, allFrames, step, params); break;
                case 57: processColorCollectionRGB_Index(x, y, data, allFrames, step, params); break;
                case 58: processColorCollectionGhostTrails(x, y, data, allFrames, step, params); break;
                case 59: processColorCollectionScale(x, y, data, allFrames, step, params); break;
                case 60: processColorCollectionReverseStrobe(x, y, data, allFrames, step, params); break;
                case 61: processColorCollectionXorPixel(x, y, data, allFrames, step, params); break;
                case 62: processBlendWithSource25(x, y, data, allFrames, step, params); break;
                case 63: processBlendWithSource50(x, y, data, allFrames, step, params); break;
                case 64: processBlendWithSource75(x, y, data, allFrames, step, params); break;
                case 65: processBlendWithSource100(x, y, data, allFrames, step, params); break;
                case 66: processColorCollectionXorOffsetFlash(x, y, data, allFrames, step, params); break;
                case 67: processColorCollectionMatrixGhost(x, y, data, allFrames, step, params); break;
                case 68: processMildStrobe(x, y, data, allFrames, step, params); break;
                case 69: processReduceBy50(x, y, data, step); break;
                case 70: processColorPositionAverageXor(x, y, data, step, params); break;
                case 71: processColorPositionXor(x, y, data, allFrames, step, params); break;
                case 72: processGrayStrobe(x, y, data, step, params); break;
                case 73: processColorStrobeXor(x, y, data, allFrames, step, params); break;
                case 74: processColorGhost(x, y, data, allFrames, step, params); break;
                case 75: processBlurredOutXor(x, y, data, allFrames, step, params); break;
                case 76: processDizzyFilter(x, y, data, allFrames, step, params); break;
                case 77: processBuzzed(x, y, data, allFrames, step, params); break;
                case 78: processBuzzedDark(x, y, data, allFrames, step, params); break;
                case 79: processAllRed(x, y, data, step); break;
                case 80: processAllGreen(x, y, data, step); break;
                case 81: processAllBlue(x, y, data, step); break;
                case 82: processNegativeStrobe(x, y, data, step, params); break;
                case 83: processXorAddMul(x, y, data, step, params); break;
                case 84: processHorizontalLines(x, y, data, step, params); break;
                case 85: processStrobeRedGreenBlue(x, y, data, step, params); break;
                case 86: processPulse(x, y, data, step, params); break;
                case 87: processDiamondPattern(x, y, data, step, params); break;
                case 88: processBitwiseXOR(x, y, data, allFrames, step, params); break;
                case 89: processBitwiseAND(x, y, data, allFrames, step, params); break;
                case 90: processBitwiseOR(x, y, data, allFrames, step, params); break;
                case 91: processBlendSwitch(x, y, data, step, params); break;
                case 92: processLineRGB(x, y, data, step); break;
                case 93: processPixelRGB(x, y, data, width, step); break;
                case 94: processInvertedScanlines(x, y, data, step, params); break;
                case 95: processScanSwitch(x, y, data, step, params); break;
                case 96: processScanAlphaSwitch(x, y, data, step, params); break;
                case 97: processRGBFlash(x, y, data, step, params); break;
                case 98: processDiagonalLines(x, y, data, width, height, step, params); break;
                case 99: processDarken(x, y, data, step); break;
                case 100: processSelfXorBlend(x, y, data, step, params); break;
                case 101: processSelfXorDoubleFlash(x, y, data, step, params); break;
                case 102: processSelfOrDoubleFlash(x, y, data, step, params); break;
                case 103: processBlendRowCurvedSqrt(x, y, data, step, params); break;
                case 104: processXorAlpha(x, y, data, step, params); break;
                case 105: processRandomXorBlend(x, y, data, step, params); break;
                case 106: processAndStrobe(x, y, data, step, params); break;
                case 107: processAndStrobeScale(x, y, data, step, params); break;
                case 108: processAndPixelStrobe(x, y, data, step, params); break;
                case 109: processAndOrXorStrobe(x, y, data, step, params); break;
                case 110: processFadeInAndOut(x, y, data, step, params); break;
                case 111: processBrightStrobe(x, y, data, step, params); break;
                case 112: processDarkStrobe(x, y, data, step, params); break;
                case 113: processRandomXorOpposite(x, y, data, step, params); break;
                case 114: processGradientRainbow(x, y, data, step, params); break;
                case 115: processCossinMultiply(x, y, data, step, params); break;
                case 116: processColorAccumulate1(x, y, data, step, params); break;
                case 117: processColorAccumulate2(x, y, data, step, params); break;
                case 118: processWeakBlend(x, y, data, step, params); break;
                case 119: processStrobeEffect(x, y, data, step, params); break;
                case 120: processBlend3(x, y, data, step, params); break;
                case 121: processNegParadox(x, y, data, step, params); break;
                case 122: processThoughtMode(x, y, data, step, params); break;
                case 123: processTri(x, y, data, step, params); break;
                case 124: processDistort(x, y, data, step, params); break;
                case 125: processColorAccumulate3(x, y, data, step, params); break;
                case 126: processFilter8(x, y, data, step, params); break;
                case 127: processFilter3(x, y, data, step, params); break;
                case 128: processRainbowBlend(x, y, data, step, params); break;
                case 129: processPixelScale(x, y, data, step, params); break;
                case 130: processGradientSelf(x, y, data, step, params); break;
                case 131: processGradientSelfVertical(x, y, data, step, params); break;
                case 132: processGradientDown(x, y, data, step, params); break;
                case 133: processGraidentHorizontal(x, y, data, step, params); break;
                case 134: processInter(x, y, data, step, params); break;
                case 135: processBlendedScanLines(x, y, data, step, params); break;
                case 136: processGradientStripes(x, y, data, step, params); break;
                case 137: processXorSine(x, y, data, step, params); break;
                case 138: processCircular(x, y, data, step, params); break;
                case 139: processRandomPixels(x, y, data, step, params); break;
                case 140: processDarkRandomPixels(x, y, data, step, params); break;
                case 141: processBars(x, y, data, step, params); break;
                case 142: processNegativeByRow(x, y, data, step); break;
                case 143: processXorScale(x, y, data, step, params); break;
                case 144: processSelfAlphaRGB(x, y, data, step, params); break;
                case 145: processBitwiseXorStrobe(x, y, data, step, params); break;
                case 146: processOrStrobe(x, y, data, allFrames, step, params); break;
                case 147: processDivideAndIncH(x, y, data, width, height, step, params); break;
                case 148: processDivideAndIncW(x, y, data, width, height, step, params); break;
                case 149: processRandomIncrease(x, y, data, step, params); break;
                case 150: processSelfAlphaScaleBlend(x, y, data, step, params); break;
                case 151: processFadeBars(x, y, data, width, step, params); break;
                case 152: processStrobeXor(x, y, data, step, params); break;
                case 153: processBlank(x, y, data, step, params); break;
                case 154: processColorVariableBlend(x, y, data, step, params); break;
                case 155: processColorXorBlend(x, y, data, step, params); break;
                case 156: processColorAddBlend(x, y, data, step, params); break;
                case 157: processSurroundingPixels(x, y, data, width, height, step, params); break;
                case 158: processSurroundingPixelsAlpha(x, y, data, width, height, step, params); break;
                case 159: processDarkModBlend(x, y, data, step, params); break;
                case 160: processIncreaseDecreaseGamma(x, y, data, step, params); break;
                case 161: processBlendChannelXor(x, y, data, step, params); break;
                case 162: processIncDifference(x, y, data, allFrames, step, params); break;
                case 163: processIncDifferenceAlpha(x, y, data, allFrames, step, params); break;
                case 164: processMirrorXorAlpha(x, y, data, width, height, step, params); break;
                case 165: processIntertwinedMirror(x, y, data, width, height, step, params); break;
                case 166: processColorFadeFilter(x, y, data, step, params); break;
                case 167: processColorChannelMoveUpAndDown(x, y, data, step, params); break;
                case 168: processMedianStrobe(x, y, data, allFrames, step, params); break;
                case 169: processRGBBlend(x, y, data, step, params); break;
                case 170: processBGRBlend(x, y, data, step, params); break;
                case 171: processFlipAlphaBlend(x, y, data, width, height, step, params); break;
                case 172: processRandomFlipFilter(x, y, data, width, height, step, params); break;
                case 173: processSelfScaleByFrame(x, y, data, step, params); break;
                case 174: processAlphaBlendMirror(x, y, data, width, height, step, params); break;
                case 175: processTwistedVision(x, y, data, width, step, params); break;
                case 176: processTruncateColor(x, y, data, step, params); break;
                case 177: processTruncateVariable(x, y, data, step, params); break;
                case 178: processTruncateVariableScale(x, y, data, step, params); break;
                case 179: processXorFade(x, y, data, step, params); break;
                case 180: processSineValue(x, y, data, step, params); break;
                case 181: processFadeRtoGtoB(x, y, data, step, params); break;
                case 182: processFadeRandomChannel(x, y, data, step, params); break;
                case 183: processVariableLines(x, y, data, width, step, params); break;
                case 184: processVariableLinesVertical(x, y, data, height, step, params); break;
                case 185: processRowMedianBlend(x, y, data, allFrames, step, params); break;
                case 186: processMirrorReverseColor(x, y, data, width, height, step, params); break;
                case 187: processPsychoticVision(x, y, data, step, params); break;
                case 188: processPixelGlitch(x, y, data, step, params); break;
                case 189: processStaticGlitch(x, y, data, step, params); break;
                case 190: processWavePattern(x, y, data, width, step, params); break;
                case 191: processWavePatternXor(x, y, data, step, params); break;
                case 192: processDiagonalXor(x, y, data, step, params); break;
                case 193: processRGBShiftBlend(x, y, data, step, params); break;
                case 194: processChannelShuffle(x, y, data, step, params); break;
                case 195: processChannelShuffleRand(x, y, data, step, params); break;
                case 196: processPixelCounter(x, y, data, step, params); break;
                case 197: processPixelCounterXor(x, y, data, step, params); break;
                case 198: processRowColorBlend(x, y, data, step, params); break;
                case 199: processColumnColorBlend(x, y, data, step, params); break;
                case 200: processCheckerboardXor(x, y, data, step, params); break;
                case 201: processCheckerboardBlend(x, y, data, allFrames, step, params); break;
                case 202: processSineWaveDistort(x, y, data, width, step, params); break;
                case 203: processCosineWaveDistort(x, y, data, height, step, params); break;
                case 204: processSinCosBlend(x, y, data, step, params); break;
                case 205: processPixelReverseXor(x, y, data, width, height, step, params); break;
                case 206: processLinesAcrossX(x, y, data, width, step, params); break;
                case 207: processXorLineX(x, y, data, step, params); break;
                case 208: processAlphaComponentIncrease(x, y, data, step, params); break;
                case 209: processExpandContract(x, y, data, width, step, params); break;
                case 210: processLongLines(x, y, data, step, params); break;
                case 211: processTearRight(x, y, data, width, step, params); break;
                case 212: processTearDown(x, y, data, height, step, params); break;
                case 213: processDistortionByRow(x, y, data, height, step, params); break;
                case 214: processDistortionByCol(x, y, data, width, step, params); break;
                case 215: processAlternateAlpha(x, y, data, step, params); break;
                case 216: processDiagSquareRGB(x, y, data, allFrames, step, params); break;
                case 217: processShiftPixelsRGB(x, y, data, width, step, params); break;
                case 218: processColorWaveTrailsRGB(x, y, data, allFrames, step, params); break;
                case 219: processProperTrails(x, y, data, allFrames, step, params); break;
                case 220: processXorLag(x, y, data, allFrames, step, params); break;
                case 221: processPixelateBlend(x, y, data, allFrames, step, params); break;
                case 222: processDiagPixel(x, y, data, allFrames, step, params); break;
                case 223: processDiagPixelY(x, y, data, allFrames, step, params); break;
                case 224: processExpandLeftRight(x, y, data, allFrames, width, step, params); break;
                case 225: processDiagSquare(x, y, data, allFrames, step, params); break;
                case 226: processHorizontalColorOffset(x, y, data, width, step, params); break;
                case 227: processPrevFrameNotEqual(x, y, data, allFrames, step, params); break;
                case 228: processBlackLines(x, y, data, width, step, params); break;
                case 229: processDizzyMode(x, y, data, allFrames, step, params); break;
                case 230: processGhostShift(x, y, data, allFrames, step, params); break;
                case 231: processRGBSplitFilter(x, y, data, allFrames, step, params); break;
                case 232: processPixelateRect(x, y, data, allFrames, step, params); break;
                case 233: processCollectionXor4(x, y, data, allFrames, step, params); break;
                case 234: processRectangleSpin(x, y, data, width, height, step, params); break;
                case 235: processRectanglePlotXY(x, y, data, width, height, step, params); break;
                case 236: processShiftLinesDown(x, y, data, width, height, step, params); break;
                case 237: processPictureStretch(x, y, data, width, height, step, params); break;
                case 238: processPictureStretchPieces(x, y, data, width, height, step, params); break;
                case 239: processVisualSnow(x, y, data, step, params); break;
                case 240: processVisualSnowX2(x, y, data, step, params); break;
                case 241: processLineGlitch(x, y, data, width, step, params); break;
                case 242: processSlitReverse64(x, y, data, width, step, params); break;
                case 243: processSlitReverse64_Increase(x, y, data, width, step, params); break;
                case 244: processSlitStretch(x, y, data, width, step, params); break;
                case 245: processLineLeftRight(x, y, data, width, step, params); break;
                case 246: processLineLeftRightResize(x, y, data, width, step, params); break;
                case 247: processRGBLineTrails(x, y, data, allFrames, step, params); break;
                case 248: processRGBCollectionBlend(x, y, data, allFrames, step, params); break;
                case 249: processRGBCollectionIncrease(x, y, data, allFrames, step, params); break;
                case 250: processRGBLongTrails(x, y, data, allFrames, step, params); break;
                case 251: processFadeRGB_Speed(x, y, data, step, params); break;
                case 252: processRGBStrobeTrails(x, y, data, allFrames, step, params); break;
                case 253: processBoxGlitch(x, y, data, width, height, step, params); break;
                case 254: processVerticalPictureDistort(x, y, data, height, step, params); break;
                case 255: processShortTrail(x, y, data, allFrames, step, params); break;
                case 256: processDiagInward(x, y, data, width, height, step, params); break;
                case 257: processDiagSquareInward(x, y, data, width, height, step, params); break;
                case 258: processDiagSquareInwardResize(x, y, data, width, height, step, params); break;
                case 259: processPictureShiftDownRight(x, y, data, width, height, step, params); break;
                case 260: processFlipPictureShift(x, y, data, width, height, step, params); break;
                case 261: processRGBWideTrails(x, y, data, allFrames, step, params); break;
                case 262: processLineInLineOut_Increase(x, y, data, width, step, params); break;
                case 263: processLineInLineOut2_Increase(x, y, data, height, step, params); break;
                case 264: processLineInLineOut3_Increase(x, y, data, width, height, step, params); break;
                case 265: processSquareByRow2Plus(x, y, data, width, step, params); break;
                case 266: processFrameSep(x, y, data, allFrames, width, step, params); break;
                case 267: processFrameSep2(x, y, data, allFrames, height, step, params); break;
                case 268: processFrameStopStart(x, y, data, allFrames, step, params); break;
                case 269: processOutOfOrder(x, y, data, allFrames, step, params); break;
                case 270: processTrackingDown(x, y, data, height, step, params); break;
                case 271: processTrackingDownBlend(x, y, data, allFrames, height, step, params); break;
                case 272: processTrackingRev(x, y, data, height, step, params); break;
                case 273: processTrackingMirror(x, y, data, width, height, step, params); break;
                case 274: processBlockPixels(x, y, data, step, params); break;
                case 275: processFrameChop(x, y, data, allFrames, width, step, params); break;
                case 276: processYLineDown(x, y, data, height, step, params); break;
                case 277: processYLineDownBlend(x, y, data, allFrames, height, step, params); break;
                case 278: processSquareDiff1(x, y, data, allFrames, step, params); break;
                case 279: processLineAcrossXNew(x, y, data, width, step, params); break;
                case 280: processColorGlitch(x, y, data, step, params); break;
                case 281: processPixelShiftUp(x, y, data, height, step, params); break;
                case 282: processPixelShiftDown(x, y, data, height, step, params); break;
                case 283: processPixelShiftLeft(x, y, data, width, step, params); break;
                case 284: processPixelShiftRight(x, y, data, width, step, params); break;
                case 285: processPixelShiftDiagonal(x, y, data, width, height, step, params); break;
                case 286: processWaveBlend(x, y, data, width, step, params); break;
                case 287: processWaveBlendX2(x, y, data, width, height, step, params); break;
                case 288: processSineWaveBlend(x, y, data, step, params); break;
                case 289: processCosineWaveBlend(x, y, data, step, params); break;
                case 290: processSpiralWave(x, y, data, width, height, step, params); break;
                case 291: processRadialBlur(x, y, data, width, height, step, params); break;
                case 292: processZoomBlur(x, y, data, width, height, step, params); break;
                case 293: processRotateBlend(x, y, data, width, height, step, params); break;
                case 294: processMirrorWave(x, y, data, width, height, step, params); break;
                case 295: processMirrorWaveX(x, y, data, width, step, params); break;
                case 296: processMirrorWaveY(x, y, data, height, step, params); break;
                case 297: processPixelDrift(x, y, data, allFrames, width, height, step, params); break;
                case 298: processPixelDriftX(x, y, data, width, step, params); break;
                case 299: processPixelDriftY(x, y, data, height, step, params); break;
                case 300: processColorPulse(x, y, data, step, params); break;
                case 301: processColorPulseRGB(x, y, data, step, params); break;
                case 302: processColorPulseXor(x, y, data, step, params); break;
                case 303: processGlitchBlock(x, y, data, width, height, step, params); break;
                case 304: processGlitchBlockXor(x, y, data, step, params); break;
                case 305: processGlitchLine(x, y, data, width, step, params); break;
                case 306: processGlitchLineX(x, y, data, height, step, params); break;
                case 307: processNoiseBlend(x, y, data, step, params); break;
                case 308: processNoiseBlendX2(x, y, data, step, params); break;
                case 309: processNoiseXor(x, y, data, step, params); break;
                case 310: processChannelShift(x, y, data, width, step, params); break;
                case 311: processChannelShiftX(x, y, data, height, step, params); break;
                case 312: processChannelRotate(x, y, data, step, params); break;
                case 313: processDiagonalStretch(x, y, data, width, height, step, params); break;
                case 314: processDiagonalStretchX(x, y, data, width, height, step, params); break;
                case 315: processDiagonalMirror(x, y, data, width, height, step, params); break;
                case 316: processSquareWave(x, y, data, step, params); break;
                case 317: processSquareWaveX(x, y, data, step, params); break;
                case 318: processSquareWaveBlend(x, y, data, allFrames, step, params); break;
                case 319: processTriangleWave(x, y, data, step, params); break;
                case 320: processTriangleWaveBlend(x, y, data, allFrames, step, params); break;
                case 321: processSawtoothWave(x, y, data, step, params); break;
                case 322: processSawtoothWaveBlend(x, y, data, allFrames, step, params); break;
                case 323: processPulseWave(x, y, data, step, params); break;
                case 324: processPulseWaveBlend(x, y, data, allFrames, step, params); break;
                case 325: processStepWave(x, y, data, step, params); break;
                case 326: processStepWaveBlend(x, y, data, allFrames, step, params); break;
                case 327: processRippleEffect(x, y, data, width, height, step, params); break;
                case 328: processRippleEffectX2(x, y, data, width, height, step, params); break;
                case 329: processShockWave(x, y, data, width, height, step, params); break;
                case 330: processShockWaveBlend(x, y, data, allFrames, width, height, step, params); break;
                case 331: processTwistEffect(x, y, data, width, height, step, params); break;
                case 332: processTwistEffectBlend(x, y, data, width, height, step, params); break;
                case 333: processFishEye(x, y, data, width, height, step, params); break;
                case 334: processFishEyeBlend(x, y, data, width, height, step, params); break;
                case 335: processKaleidoscope(x, y, data, width, height, step, params); break;
                case 336: processKaleidoscopeBlend(x, y, data, width, height, step, params); break;
                case 337: processTunnelEffect(x, y, data, width, height, step, params); break;
                case 338: processTunnelEffectBlend(x, y, data, width, height, step, params); break;
                case 339: processVortexEffect(x, y, data, width, height, step, params); break;
                case 340: processVortexEffectBlend(x, y, data, width, height, step, params); break;
                case 341: processColorDrift(x, y, data, step, params); break;
                case 342: processColorDriftX(x, y, data, step, params); break;
                case 343: processRGBShift(x, y, data, width, step, params); break;
                case 344: processRGBShiftX(x, y, data, height, step, params); break;
                case 345: processChromaticAberration(x, y, data, width, step, params); break;
                case 346: processChromaticAberrationX(x, y, data, width, height, step, params); break;
                case 347: processPosterize(x, y, data, step, params); break;
                case 348: processPosterizeBlend(x, y, data, step, params); break;
                case 349: processSolarize(x, y, data, step, params); break;
                case 350: processSolarizeBlend(x, y, data, step, params); break;
                case 351: processGammaBright(x, y, data, step, params); break;
                case 352: processGammaDark(x, y, data, step, params); break;
                case 353: processContrastBoost(x, y, data, step, params); break;
                case 354: processContrastReduce(x, y, data, step, params); break;
                case 355: processEdgeGlowBlend(x, y, data, allFrames, width, height, step, params); break;
                case 356: processFrameBlendMulti(x, y, data, allFrames, step, params); break;
                case 357: processFrameBlendMultiX(x, y, data, allFrames, step, params); break;
                case 358: processAcidTrailsBlend(x, y, data, allFrames, step, params); break;
                case 359: processAcidGlitchX(x, y, data, width, height, step, params); break;
                case 360: processAlphaXorBlend(x, y, data, allFrames, step, params); break;
                case 361: processAlphaXorBlendDouble(x, y, data, allFrames, step, params); break;
                case 362: processAndOrXorStrobeScale(x, y, data, step, params); break;
                case 363: processAveragePixelsXorBlend(x, y, data, step, params); break;
                case 364: processBitwiseRotateBlend(x, y, data, step, params); break;
                case 365: processBitwiseRotateDiffBlend(x, y, data, allFrames, step, params); break;
                case 366: processBitwiseXorScaleBlend(x, y, data, step, params); break;
                case 367: processBlackAndWhiteStrobe(x, y, data, step, params); break;
                case 368: processBlendAlphaXorScale(x, y, data, step, params); break;
                case 369: processBlendBurredXor(x, y, data, allFrames, step, params); break;
                case 370: processBlendCombinedXor(x, y, data, step, params); break;
                case 371: processBlendIncreaseRGB(x, y, data, step, params); break;
                case 372: processBlendThreeXor(x, y, data, allFrames, step, params); break;
                case 373: processBlurDistortionBlend(x, y, data, width, height, step, params); break;
                case 374: processColorAccumulate(x, y, data, step, params); break;
                case 375: processColorAccumulateBlend(x, y, data, allFrames, step, params); break;
                case 376: processColorAccumulateXor(x, y, data, step, params); break;
                case 377: processColorChannelBlend(x, y, data, step, params); break;
                case 378: processColorChannelXor(x, y, data, step, params); break;
                case 379: processColorCollectionEnergy(x, y, data, allFrames, step, params); break;
                case 380: processColorCollectionWave(x, y, data, step, params); break;
                case 381: processColorFadeXor(x, y, data, step, params); break;
                case 382: processColorIntensityBlend(x, y, data, step, params); break;
                case 383: processColorIntensityXor(x, y, data, step, params); break;
                case 384: processColorMoveBlend(x, y, data, width, height, step, params); break;
                case 385: processColorPixelBlend(x, y, data, step, params); break;
                case 386: processColorPixelXor(x, y, data, step, params); break;
                case 387: processColorScaleBlend(x, y, data, step, params); break;
                case 388: processColorWaveXor(x, y, data, step, params); break;
                case 389: processCosineMultiplyBlend(x, y, data, step, params); break;
                case 390: processDarkModBlendXor(x, y, data, step, params); break;
                case 391: processDifferenceBlend(x, y, data, allFrames, step, params); break;
                case 392: processDifferenceXorBlend(x, y, data, allFrames, step, params); break;
                case 393: processDistortBlend(x, y, data, width, height, step, params); break;
                case 394: processDiamondPatternBlend(x, y, data, width, height, step, params); break;
                case 395: processFadeBlendXor(x, y, data, step, params); break;
                case 396: processFlashBlendXor(x, y, data, step, params); break;
                case 397: processGhostTrailsBlend(x, y, data, allFrames, step, params); break;
                case 398: processAddInvert(x, y, data, step, params); break;
                case 399: processAlphaBlendSimple(x, y, data, allFrames, step, params); break;
                case 400: processAlphaBlendDoubleX(x, y, data, allFrames, step, params); break;
                case 401: processAlphaStrobeBlendX(x, y, data, allFrames, step, params); break;
                case 402: processBitwiseAndBlend(x, y, data, step, params); break;
                case 403: processBitwiseXorAverage(x, y, data, step, params); break;
                case 404: processBitwiseXorBlendX(x, y, data, allFrames, step, params); break;
                case 405: processBlackStrobe(x, y, data, step, params); break;
                case 406: processBlendAlphaXorX(x, y, data, step, params); break;
                case 407: processBlendCombinedValuesX(x, y, data, step, params); break;
                case 408: processBlendFor360(x, y, data, width, height, step, params); break;
                case 409: processBlendForward16(x, y, data, allFrames, step, params); break;
                case 410: processBlendForward32(x, y, data, allFrames, step, params); break;
                case 411: processBlendFromXtoY(x, y, data, step, params); break;
                case 412: processBlendIncreaseX(x, y, data, step, params); break;
                case 413: processBlendRedGreenBlue(x, y, data, step, params); break;
                case 414: processBlendWithColorX(x, y, data, step, params); break;
                case 415: processBlendAngle(x, y, data, width, height, step, params); break;
                case 416: processBlockScale(x, y, data, width, height, step, params); break;
                case 417: processBlockStrobe(x, y, data, width, height, step, params); break;
                case 418: processBlockXor(x, y, data, width, height, step, params); break;
                case 419: processBlockyTrails16(x, y, data, allFrames, width, height, step, params); break;
                case 420: processBlockyTrails32(x, y, data, allFrames, width, height, step, params); break;
                case 421: processBlurDistortionX(x, y, data, width, height, step, params); break;
                case 422: processCannyStrobe(x, y, data, width, height, step, params); break;
                case 423: processColorFadeSlow(x, y, data, step, params); break;
                case 424: processColorFibonacci(x, y, data, step, params); break;
                case 425: processCurtainEffect(x, y, data, width, height, step, params); break;
                case 426: processDarkColorFibonacci(x, y, data, step, params); break;
                case 427: processDarkColorsBlend(x, y, data, step, params); break;
                case 428: processEnergizeBlend(x, y, data, allFrames, step, params); break;
                case 429: processAverageLines(x, y, data, width, step, params); break;
                case 430: processAverageLinesBlendX(x, y, data, allFrames, width, step, params); break;
                case 431: processBlendRowAlpha(x, y, data, height, step, params); break;
                case 432: processBlendInOut(x, y, data, allFrames, step, params); break;
                case 433: processColorFlashIncreaseX(x, y, data, step, params); break;
                case 434: processColorIncreaseInOut(x, y, data, step, params); break;
                case 435: processColorLinesX(x, y, data, step, params); break;
                case 436: processColorMoveDownX(x, y, data, height, step, params); break;
                case 437: processColorOrderSwapX(x, y, data, step, params); break;
                case 438: processColorPulseAlphaX(x, y, data, step, params); break;
                case 439: processColorRowShiftX(x, y, data, width, step, params); break;
                case 440: processColorShiftXorX(x, y, data, step, params); break;
                case 441: processCopyXorAlphaX(x, y, data, allFrames, step, params); break;
                case 442: processCycleShiftRGBX(x, y, data, step, params); break;
                case 443: processDarkNegateX(x, y, data, step, params); break;
                case 444: processDarkSelfAlphaX(x, y, data, step, params); break;
                case 445: processDiagonalGlitch(x, y, data, width, height, step, params); break;
                case 446: processDigitalHaze(x, y, data, step, params); break;
                case 447: processDoubleXorBlend(x, y, data, allFrames, step, params); break;
                case 448: processEchoBlend(x, y, data, allFrames, step, params); break;
                case 449: processElectricEdge(x, y, data, width, height, step, params); break;
                case 450: processFlashColorStrobe(x, y, data, step, params); break;
                case 451: processFrameDiffXor(x, y, data, allFrames, step, params); break;
                case 452: processGhostMirror(x, y, data, allFrames, width, step, params); break;
                case 453: processGlitchSort(x, y, data, width, step, params); break;
                case 454: processHeatWave(x, y, data, width, height, step, params); break;
                case 455: processInterlaceBlend(x, y, data, allFrames, step, params); break;
                case 456: processInvertStrobe(x, y, data, step, params); break;
                case 457: processKaleidoBlend(x, y, data, width, height, step, params); break;
                case 458: processLightStrobe(x, y, data, step, params); break;
                case 459: processLineGlitchX(x, y, data, width, step, params); break;
                case 460: processMosaicBlend(x, y, data, width, height, step, params); break;
                case 461: processNegatePulse(x, y, data, step, params); break;
                case 462: processOffsetGhost(x, y, data, allFrames, width, height, step, params); break;
                case 463: processPixelateWave(x, y, data, width, height, step, params); break;
                case 464: processQuantizeBlend(x, y, data, step, params); break;
                case 465: processRandomLines(x, y, data, height, step, params); break;
                case 466: processRippleDisplace(x, y, data, width, height, step, params); break;
                case 467: processRotateShift(x, y, data, width, height, step, params); break;
                case 468: processSaturationGlow(x, y, data, step, params); break;
                case 469: processScaleToCenter(x, y, data, width, height, step, params); break;
                case 470: processShadowMirror(x, y, data, width, height, step, params); break;
                case 471: processShiftChannels(x, y, data, step, params); break;
                case 472: processSliceGlitch(x, y, data, width, height, step, params); break;
                case 473: processSobelGlow(x, y, data, width, height, step, params); break;
                case 474: processSpectralShift(x, y, data, step, params); break;
                case 475: processSpiralTrail(x, y, data, allFrames, width, height, step, params); break;
                case 476: processSquareTrails(x, y, data, allFrames, step, params); break;
                case 477: processStrobeNegate(x, y, data, step, params); break;
                case 478: processThermalBlend(x, y, data, step, params); break;
                case 479: processTintShift(x, y, data, step, params); break;
                case 480: processTrailEcho(x, y, data, allFrames, step, params); break;
                case 481: processTransitionBlend(x, y, data, allFrames, step, params); break;
                case 482: processTwistWarp(x, y, data, width, height, step, params); break;
                case 483: processVerticalShift(x, y, data, height, step, params); break;
                case 484: processVortexBlend(x, y, data, width, height, step, params); break;
                case 485: processWeavePattern(x, y, data, width, height, step, params); break;
                case 486: processWhiteBurst(x, y, data, width, height, step, params); break;
                case 487: processWiggleDisplace(x, y, data, width, height, step, params); break;
                case 488: processXorPulseX(x, y, data, step, params); break;
                case 489: processYellowShift(x, y, data, step, params); break;
                case 490: processZigzagGlitch(x, y, data, width, step, params); break;
                case 491: processAlphaModulate(x, y, data, step, params); break;
                case 492: processBlockSwap(x, y, data, width, height, step, params); break;
                case 493: processColorResonance(x, y, data, step, params); break;
                case 494: processDepthGlitch(x, y, data, width, height, step, params); break;
                case 495: processEchoShift(x, y, data, allFrames, step, params); break;
                case 496: processFractalNoise(x, y, data, step, params); break;
                case 497: processGradientRotate(x, y, data, width, height, step, params); break;
                case 498: processHarmonicShift(x, y, data, step, params); break;
                case 499: processAcidWarp(x, y, data, width, height, step, params); break;
                case 500: processBlendDiagonal(x, y, data, width, height, step, params); break;
                case 501: processChromaFlash(x, y, data, step, params); break;
                case 502: processCircleWave(x, y, data, width, height, step, params); break;
                case 503: processColorCrush(x, y, data, step, params); break;
                case 504: processCrosshatchBlend(x, y, data, step, params); break;
                case 505: processCyberGlitch(x, y, data, width, step, params); break;
                case 506: processDarkPulse(x, y, data, step, params); break;
                case 507: processDiamondPatternX(x, y, data, step, params); break;
                case 508: processDigitalRain(x, y, data, height, step, params); break;
                case 509: processDisplaceX(x, y, data, width, step, params); break;
                case 510: processDriftBlend(x, y, data, allFrames, width, step, params); break;
                case 511: processEdgePulse(x, y, data, width, height, step, params); break;
                case 512: processFlameEffect(x, y, data, height, step, params); break;
                case 513: processFlickerShift(x, y, data, step, params); break;
                case 514: processGhostLayer(x, y, data, allFrames, step, params); break;
                case 515: processGlitchBlockX(x, y, data, width, height, step, params); break;
                case 516: processGlowPulse(x, y, data, step, params); break;
                case 517: processGridDistort(x, y, data, width, height, step, params); break;
                case 518: processHexPattern(x, y, data, step, params); break;
                case 519: processHueRotate(x, y, data, step, params); break;
                case 520: processInterweaveX(x, y, data, allFrames, step, params); break;
                case 521: processJitterBlend(x, y, data, width, step, params); break;
                case 522: processKaleidoScope4(x, y, data, width, height, step, params); break;
                case 523: processLaserScan(x, y, data, width, height, step, params); break;
                case 524: processLightLeak(x, y, data, width, height, step, params); break;
                case 525: processMeltDown(x, y, data, height, step, params); break;
                case 526: processMirrorDiag(x, y, data, width, height, step, params); break;
                case 527: processNeonGlow(x, y, data, width, height, step, params); break;
                case 528: processNoiseBlendX(x, y, data, step, params); break;
                case 529: processPixelDrift(x, y, data, width, step, params); break;
                case 530: processPlasmaWave(x, y, data, step, params); break;
                case 531: processPrismSplit(x, y, data, width, step, params); break;
                case 532: processPulseRadial(x, y, data, width, height, step, params); break;
                case 533: processRainbowStrobe(x, y, data, step, params); break;
                case 534: processRefractionX(x, y, data, width, step, params); break;
                case 535: processScanlineX(x, y, data, step, params); break;
                case 536: processShatterEffect(x, y, data, width, height, step, params); break;
                case 537: processStaticNoise(x, y, data, step, params); break;
                case 538: processTunnelVision(x, y, data, width, height, step, params); break;
                case 539: processAberrationPulse(x, y, data, width, step, params); break;
                case 540: processAquaWave(x, y, data, width, height, step, params); break;
                case 541: processBinaryFlash(x, y, data, step, params); break;
                case 542: processBloomGlow(x, y, data, step, params); break;
                case 543: processCellularNoise(x, y, data, step, params); break;
                case 544: processChromaShift2(x, y, data, width, step, params); break;
                case 545: processColorBands(x, y, data, step, params); break;
                case 546: processColorVortex(x, y, data, width, height, step, params); break;
                case 547: processCrystalMosaic(x, y, data, width, height, step, params); break;
                case 548: processCubicDistort(x, y, data, width, height, step, params); break;
                case 549: processDepthFade(x, y, data, height, step, params); break;
                case 550: processDiscoFlash(x, y, data, step, params); break;
                case 551: processDitherBlend(x, y, data, step, params); break;
                case 552: processDoubleVision(x, y, data, width, step, params); break;
                case 553: processDreamHaze(x, y, data, step, params); break;
                case 554: processElectricStorm(x, y, data, step, params); break;
                case 555: processEmbossShift(x, y, data, width, height, step, params); break;
                case 556: processFiberOptic(x, y, data, width, height, step, params); break;
                case 557: processFilmGrain(x, y, data, step, params); break;
                case 558: processFireWorks(x, y, data, width, height, step, params); break;
                case 559: processFluidMotion(x, y, data, width, height, step, params); break;
                case 560: processFogRoll(x, y, data, height, step, params); break;
                case 561: processGlassRefract(x, y, data, width, height, step, params); break;
                case 562: processGlowTrails(x, y, data, step, params); break;
                case 563: processGridPulse(x, y, data, step, params); break;
                case 564: processHalftoneBlend(x, y, data, step, params); break;
                case 565: processHeatDistort(x, y, data, width, height, step, params); break;
                case 566: processHoloGlitch(x, y, data, step, params); break;
                case 567: processInfraredView(x, y, data, step, params); break;
                case 568: processLavaLamp(x, y, data, width, height, step, params); break;
                case 569: processLensFlare(x, y, data, width, height, step, params); break;
                case 570: processLightningBolt(x, y, data, width, height, step, params); break;
                case 571: processLiquidMetal(x, y, data, step, params); break;
                case 572: processMatrixCode(x, y, data, height, step, params); break;
                case 573: processMirrorKaleid(x, y, data, width, height, step, params); break;
                case 574: processNightVision(x, y, data, step, params); break;
                case 575: processOilSlick(x, y, data, step, params); break;
                case 576: processParticleField(x, y, data, step, params); break;
                case 577: processPinwheelSpin(x, y, data, width, height, step, params); break;
                case 578: processPixelStorm(x, y, data, width, height, step, params); break;
                case 579: processPlaidPattern(x, y, data, step, params); break;
                case 580: processPolarInvert(x, y, data, width, height, step, params); break;
                case 581: processPolychromeTint(x, y, data, step, params); break;
                case 582: processPopArtDots(x, y, data, step, params); break;
                case 583: processPrismaticEdge(x, y, data, width, height, step, params); break;
                case 584: processPulseWarp(x, y, data, width, height, step, params); break;
                case 585: processQuantumNoise(x, y, data, step, params); break;
                case 586: processQuiltBlend(x, y, data, step, params); break;
                case 587: processRadarSweep(x, y, data, width, height, step, params); break;
                case 588: processRaindropRipple(x, y, data, width, height, step, params); break;
                case 589: processRasterBars(x, y, data, step, params); break;
                case 590: processRetroTube(x, y, data, width, height, step, params); break;
                case 591: processRingWave(x, y, data, width, height, step, params); break;
                case 592: processRippleTank(x, y, data, width, height, step, params); break;
                case 593: processRotatingPrism(x, y, data, width, height, step, params); break;
                case 594: processSandStorm(x, y, data, width, height, step, params); break;
                case 595: processSaturationPulse(x, y, data, step, params); break;
                case 596: processScatterPixel(x, y, data, width, height, step, params); break;
                case 597: processShadowPlay(x, y, data, width, height, step, params); break;
                case 598: processShimmerGlass(x, y, data, width, height, step, params); break;
                case 599: processSilhouetteBlend(x, y, data, step, params); break;
                case 600: processSketchOutline(x, y, data, width, height, step, params); break;
                case 601: processSliceShift(x, y, data, width, step, params); break;
                case 602: processSmearMotion(x, y, data, width, step, params); break;
                case 603: processSmokeWisp(x, y, data, width, height, step, params); break;
                case 604: processSnowDrift(x, y, data, step, params); break;
                case 605: processSolarFlare(x, y, data, width, height, step, params); break;
                case 606: processSparkShower(x, y, data, height, step, params); break;
                case 607: processSpectrumWave(x, y, data, step, params); break;
                case 608: processSpiralZoom(x, y, data, width, height, step, params); break;
                case 609: processSplitMirror(x, y, data, width, height, step, params); break;
                case 610: processStarBurst(x, y, data, width, height, step, params); break;
                case 611: processStaticPulse(x, y, data, step, params); break;
                case 612: processStencilCut(x, y, data, step, params); break;
                case 613: processStippleShade(x, y, data, step, params); break;
                case 614: processStormCloud(x, y, data, height, step, params); break;
                case 615: processStreakBlur(x, y, data, width, step, params); break;
                case 616: processStrobeEdge(x, y, data, width, height, step, params); break;
                case 617: processSubpixelShift(x, y, data, width, step, params); break;
                case 618: processSwimDistort(x, y, data, width, height, step, params); break;
                case 619: processTangentWarp(x, y, data, width, height, step, params); break;
                case 620: processTapeGlitch(x, y, data, width, step, params); break;
                case 621: processTechnoGrid(x, y, data, step, params); break;
                case 622: processTeleportPixel(x, y, data, width, height, step, params); break;
                case 623: processTemporalBlur(x, y, data, step, params); break;
                case 624: processTerraFracture(x, y, data, width, height, step, params); break;
                case 625: processTextureWave(x, y, data, step, params); break;
                case 626: processThresholdPulse(x, y, data, step, params); break;
                case 627: processTidalWave(x, y, data, width, height, step, params); break;
                case 628: processTintCycle(x, y, data, step, params); break;
                case 629: processTraceEdge(x, y, data, width, height, step, params); break;
                case 630: processTriangleMosaic(x, y, data, width, height, step, params); break;
                case 631: processTripleSplit(x, y, data, width, step, params); break;
                case 632: processTurbulentFlow(x, y, data, width, height, step, params); break;
                case 633: processUnderwaterCaustic(x, y, data, step, params); break;
                case 634: processUnsharpPulse(x, y, data, width, height, step, params); break;
                case 635: processVaporTrail(x, y, data, width, step, params); break;
                case 636: processVectorField(x, y, data, width, height, step, params); break;
                case 637: processVelocityBlur(x, y, data, width, height, step, params); break;
                case 638: processVerticalMelt(x, y, data, height, step, params); break;
                case 639: processVHSTracking(x, y, data, width, step, params); break;
                case 640: processVibrantPop(x, y, data, step, params); break;
                case 641: processVignetteFlash(x, y, data, width, height, step, params); break;
                case 642: processVoronoiShatter(x, y, data, width, height, step, params); break;
                case 643: processWarpSpeed(x, y, data, width, height, step, params); break;
                case 644: processWaterColor(x, y, data, width, height, step, params); break;
                case 645: processWaveCollapse(x, y, data, width, height, step, params); break;
                case 646: processWebPattern(x, y, data, width, height, step, params); break;
                case 647: processWhirlpoolSpin(x, y, data, width, height, step, params); break;
                case 648: processWindBlast(x, y, data, width, step, params); break;
                case 649: processWireframePulse(x, y, data, step, params); break;
                case 650: processXRayFlash(x, y, data, step, params); break;
                case 651: processZebraStripe(x, y, data, step, params); break;
                case 652: processZenRipple(x, y, data, width, height, step, params); break;
                case 653: processZigzagWave(x, y, data, width, height, step, params); break;
                case 654: processZoomPulse(x, y, data, width, height, step, params); break;
                case 655: processZoneTint(x, y, data, width, height, step, params); break;
                case 656: processAcidDrip(x, y, data, height, step, params); break;
                case 657: processAuroraWave(x, y, data, height, step, params); break;
                case 658: processBandPass(x, y, data, step, params); break;
                case 659: processBilinearStretch(x, y, data, width, height, step, params); break;
                case 660: processBleedThrough(x, y, data, width, step, params); break;
                case 661: processBlockShatter(x, y, data, width, height, step, params); break;
                case 662: processBlurMask(x, y, data, width, height, step, params); break;
                case 663: processBokehBlur(x, y, data, width, height, step, params); break;
                case 664: processBounceWave(x, y, data, width, height, step, params); break;
                case 665: processBrokenGlass(x, y, data, width, height, step, params); break;
                case 666: processBubbleWarp(x, y, data, width, height, step, params); break;
                case 667: processCRTCurvature(x, y, data, width, height, step, params); break;
                case 668: processCascadeBlend(x, y, data, step, params); break;
                case 669: processCelShade(x, y, data, step, params); break;
                case 670: processChainReaction(x, y, data, width, height, step, params); break;
                case 671: processChannelDelay(x, y, data, width, step, params); break;
                case 672: processChromaBleed(x, y, data, width, step, params); break;
                case 673: processCircuitTrace(x, y, data, step, params); break;
                case 674: processClockWipe(x, y, data, width, height, step, params); break;
                case 675: processCloudShadow(x, y, data, step, params); break;
                case 676: processColorBurn(x, y, data, step, params); break;
                case 677: processColorHalves(x, y, data, width, step, params); break;
                case 678: processComicDots(x, y, data, step, params); break;
                case 679: processConcentricPulse(x, y, data, width, height, step, params); break;
                case 680: processCopperTone(x, y, data, step, params); break;
                case 681: processCornerStretch(x, y, data, width, height, step, params); break;
                case 682: processCosmicDust(x, y, data, step, params); break;
                case 683: processCrossBlur(x, y, data, width, height, step, params); break;
                case 684: processCrossProcess(x, y, data, step, params); break;
                case 685: processCrystalEdge(x, y, data, width, height, step, params); break;
                case 686: processCubeRotate(x, y, data, width, height, step, params); break;
                case 687: processCurtainReveal(x, y, data, width, step, params); break;
                case 688: processCyberPunk(x, y, data, step, params); break;
                case 689: processDataCorrupt(x, y, data, width, step, params); break;
                case 690: processDebrisField(x, y, data, width, height, step, params); break;
                case 691: processDeepFry(x, y, data, step, params); break;
                case 692: processDesyncRGB(x, y, data, width, height, step, params); break;
                case 693: processDiagonalWipe(x, y, data, width, height, step, params); break;
                case 694: processDigitalArtifact(x, y, data, step, params); break;
                case 695: processDimensionRift(x, y, data, width, height, step, params); break;
                case 696: processDotCrawl(x, y, data, step, params); break;
                case 697: processDualTone(x, y, data, step, params); break;
                case 698: processEchoFade(x, y, data, allFrames, step, params); break;
                case 699: processEdgeMelt(x, y, data, width, height, step, params); break;
                case 700: processElasticWarp(x, y, data, width, height, step, params); break;
                case 701: processEmberGlow(x, y, data, step, params); break;
                case 702: processEntropyShift(x, y, data, step, params); break;
                case 703: processErosionBlend(x, y, data, width, height, step, params); break;
                case 704: processExplosionBurst(x, y, data, width, height, step, params); break;
                case 705: processFacetMirror(x, y, data, width, height, step, params); break;
                case 706: processFadeStreak(x, y, data, width, step, params); break;
                case 707: processFeatherEdge(x, y, data, width, height, step, params); break;
                case 708: processFlashFreeze(x, y, data, step, params); break;
                case 709: processFlipMirror(x, y, data, width, height, step, params); break;
                case 710: processFloatDrift(x, y, data, width, height, step, params); break;
                case 711: processFlowField(x, y, data, width, height, step, params); break;
                case 712: processFoldWarp(x, y, data, width, height, step, params); break;
                case 713: processFragmentScatter(x, y, data, width, height, step, params); break;
                case 714: processFrequencyPulse(x, y, data, step, params); break;
                case 715: processFrostBite(x, y, data, step, params); break;
                case 716: processFuseBlend(x, y, data, allFrames, step, params); break;
                case 717: processGalaxySpiral(x, y, data, width, height, step, params); break;
                case 718: processGelWobble(x, y, data, width, height, step, params); break;
                case 719: processGhostEcho(x, y, data, allFrames, width, step, params); break;
                case 720: processGlassShatter(x, y, data, width, height, step, params); break;
                case 721: processGlimmerPulse(x, y, data, step, params); break;
                case 722: processGlitchMosaic(x, y, data, width, height, step, params); break;
                case 723: processGlowEdge(x, y, data, width, height, step, params); break;
                case 724: processGradientMelt(x, y, data, height, step, params); break;
                case 725: processGrainStorm(x, y, data, step, params); break;
                case 726: processGravityPull(x, y, data, width, height, step, params); break;
                case 727: processGridWarp(x, y, data, width, height, step, params); break;
                case 728: processHaloRing(x, y, data, width, height, step, params); break;
                case 729: processHarshLight(x, y, data, step, params); break;
                case 730: processHazeLayer(x, y, data, height, step, params); break;
                case 731: processHeatRipple(x, y, data, width, height, step, params); break;
                case 732: processHexagonBlur(x, y, data, width, height, step, params); break;
                case 733: processHighContrast(x, y, data, step, params); break;
                case 734: processHologramScan(x, y, data, height, step, params); break;
                case 735: processHorizonBend(x, y, data, width, height, step, params); break;
                case 736: processHotSpot(x, y, data, width, height, step, params); break;
                case 737: processHueWobble(x, y, data, step, params); break;
            }
        }
        setAlpha(data, y * step + x * 4, params.isNegative);
    }
}
extern "C" void launch_filter(ac_gpu::Filter *f_host, size_t c, unsigned char* data, unsigned char** allFrames, int numFrames, int width, int height, size_t step, float alpha, bool isNegative, int square_size, int start_index, int start_dir, ac_gpu::Filter** d_list_ptr, bool& changed) {
    if (changed || *d_list_ptr == nullptr) {
        if (*d_list_ptr != nullptr) {
            CHECK_CUDA(cudaFree(*d_list_ptr));
        }
        CHECK_CUDA(cudaMalloc(d_list_ptr, sizeof(ac_gpu::Filter) * c));
        CHECK_CUDA(cudaMemcpy(*d_list_ptr, f_host, sizeof(ac_gpu::Filter) * c, cudaMemcpyHostToDevice));
        changed = false; 
    }
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    static int frame_counter = 0;
    ac_gpu::FilterParams params;
    params.alpha = alpha;
    params.isNegative = isNegative;
    params.numFrames = numFrames;
    params.square_size = (square_size > 0) ? square_size : 16;
    params.start_index = start_index;
    params.start_dir = start_dir;
    params.int_param1 = rand() % height;
    params.int_param2 = rand() % width;
    params.float_param1 = (float)(rand() % 255);
    params.seed = rand();
    params.frame_count = frame_counter++;
    params.sumR = rand() % 255;
    params.sumG = rand() % 255;
    params.sumB = rand() % 255;
    params.threshold = 15;
    params.sw = 16 + (frame_counter % 48);
    params.sh = 16 + (frame_counter % 48);
    ac_gpu::unifiedFilterKernel<<<gridSize, blockSize>>>(*d_list_ptr, c, data, allFrames, width, height, step, params);
    CHECK_CUDA(cudaDeviceSynchronize());
}