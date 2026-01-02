#include "ac-gpu/ac-gpu.hpp"
#include <cuda_runtime.h>
#include <string>
namespace ac_gpu {
    Filter filters[] = {
        { 0, "SelfAlphaBlend" },
        { 1, "MedianBlend" },
        { 2, "MedianBlend" },
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
        { 186, "MirrorReverseColor" },
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
        { 285, "PixelShiftDiagonal" }
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
    // Filter 236: ShiftLinesDown
    __device__ void processShiftLinesDown(int x, int y, unsigned char* data, int width, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int shiftAmount = (params.frame_count / 4) % height;
        int src_y = (y + shiftAmount) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)(0.5f * data[idx + j] + 0.5f * data[src_idx + j]);
        }
    }
    // Filter 237: PictureStretch
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
    // Filter 238: PictureStretchPieces
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
    // Filter 239: VisualSnow
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
    // Filter 240: VisualSnowX2
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
    // Filter 241: LineGlitch
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
    // Filter 242: SlitReverse64
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
    // Filter 243: SlitReverse64_Increase
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
    // Filter 244: SlitStretch
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
    // Filter 245: LineLeftRight
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
    // Filter 246: LineLeftRightResize
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
    // Filter 247: RGBLineTrails
    __device__ void processRGBLineTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int channel = y % 3;
        unsigned char* prev = allFrames[0];
        if (prev) {
            data[idx + channel] = (unsigned char)(0.6f * data[idx + channel] + 0.4f * prev[idx + channel]);
        }
    }
    // Filter 248: RGBCollectionBlend
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
    // Filter 249: RGBCollectionIncrease
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
    // Filter 250: RGBLongTrails
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
    // Filter 251: FadeRGB_Speed
    __device__ void processFadeRGB_Speed(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float speedR = fabsf(sinf(params.frame_count * 0.05f));
        float speedG = fabsf(sinf(params.frame_count * 0.07f));
        float speedB = fabsf(sinf(params.frame_count * 0.09f));
        data[idx] = (unsigned char)(data[idx] * speedB);
        data[idx + 1] = (unsigned char)(data[idx + 1] * speedG);
        data[idx + 2] = (unsigned char)(data[idx + 2] * speedR);
    }
    // Filter 252: RGBStrobeTrails
    __device__ void processRGBStrobeTrails(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int strobe = params.frame_count % 3;
        unsigned char* prev = allFrames[0];
        if (prev) {
            data[idx + strobe] = (unsigned char)(0.5f * data[idx + strobe] + 0.5f * prev[idx + strobe]);
        }
    }
    // Filter 253: BoxGlitch
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
    // Filter 254: VerticalPictureDistort
    __device__ void processVerticalPictureDistort(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float distort = sinf(x * 0.05f + params.frame_count * 0.02f) * 10;
        int src_y = ((int)(y + distort) % height + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    // Filter 255: ShortTrail
    __device__ void processShortTrail(int x, int y, unsigned char* data, unsigned char** allFrames, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char* prev = allFrames[0];
        if (prev) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = (unsigned char)(0.8f * data[idx + j] + 0.2f * prev[idx + j]);
            }
        }
    }
    // Filter 256: DiagInward
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
    // Filter 257: DiagSquareInward
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
    // Filter 258: DiagSquareInwardResize
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
    // Filter 259: PictureShiftDownRight
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
    // Filter 260: FlipPictureShift
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
    // Filter 261: RGBWideTrails
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
    // Filter 262: LineInLineOut_Increase
    __device__ void processLineInLineOut_Increase(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineSize = 4 + (params.frame_count / 4) % 32;
        int segment = x / lineSize;
        float factor = (segment % 2 == 0) ? 1.0f + params.alpha * 0.5f : 1.0f - params.alpha * 0.3f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    // Filter 263: LineInLineOut2_Increase
    __device__ void processLineInLineOut2_Increase(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineSize = 4 + (params.frame_count / 4) % 32;
        int segment = y / lineSize;
        float factor = (segment % 2 == 0) ? 1.0f + params.alpha * 0.5f : 1.0f - params.alpha * 0.3f;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = (unsigned char)fminf(255.0f, data[idx + j] * factor);
        }
    }
    // Filter 264: LineInLineOut3_Increase
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
    // Filter 265: SquareByRow2Plus
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
    // Filter 266: FrameSep
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
    // Filter 267: FrameSep2
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
    // Filter 268: FrameStopStart
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
    // Filter 269: OutOfOrder
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
    // Filter 270: TrackingDown
    __device__ void processTrackingDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int track = (params.frame_count * 2) % height;
        if (y == track) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255;
            }
        }
    }
    // Filter 271: TrackingDownBlend
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
    // Filter 272: TrackingRev
    __device__ void processTrackingRev(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int track = height - 1 - ((params.frame_count * 2) % height);
        if (y == track) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255 - data[idx + j];
            }
        }
    }
    // Filter 273: TrackingMirror
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
    // Filter 274: BlockPixels
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
    // Filter 275: FrameChop
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
    // Filter 276: YLineDown
    __device__ void processYLineDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int linePos = (params.frame_count + x) % height;
        if (y == linePos) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255;
            }
        }
    }
    // Filter 277: YLineDownBlend
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
    // Filter 278: SquareDiff1
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
    // Filter 279: LineAcrossX
    __device__ void processLineAcrossXNew(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int lineX = (params.frame_count * 3) % width;
        if (abs(x - lineX) < 2) {
            for (int j = 0; j < 3; ++j) {
                data[idx + j] = 255 - data[idx + j];
            }
        }
    }
    // Filter 280: ColorGlitch
    __device__ void processColorGlitch(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        float r = gpu_rand(x, y, params.seed);
        if (r < 0.05f) {
            int channel = (int)(r * 60) % 3;
            data[idx + channel] = 255 - data[idx + channel];
        }
    }
    // Filter 281: PixelShiftUp
    __device__ void processPixelShiftUp(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param1 % height;
        int src_y = (y + offset) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    // Filter 282: PixelShiftDown
    __device__ void processPixelShiftDown(int x, int y, unsigned char* data, int height, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param1 % height;
        int src_y = (y - offset + height) % height;
        int src_idx = src_y * step + x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    // Filter 283: PixelShiftLeft
    __device__ void processPixelShiftLeft(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param2 % width;
        int src_x = (x + offset) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    // Filter 284: PixelShiftRight
    __device__ void processPixelShiftRight(int x, int y, unsigned char* data, int width, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int offset = params.int_param2 % width;
        int src_x = (x - offset + width) % width;
        int src_idx = y * step + src_x * 4;
        for (int j = 0; j < 3; ++j) {
            data[idx + j] = data[src_idx + j];
        }
    }
    // Filter 285: PixelShiftDiagonal
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