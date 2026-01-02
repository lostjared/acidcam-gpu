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
        { 78, "BuzzedDark" }
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
                case 49: processSquareByRow(x, y, data, allFrames, width, height, step, params, true, false); break; // Rev
                case 50: processSquareByRow(x, y, data, allFrames, width, height, step, params, false, true); break; // SquareByRow2 (Blend)
                case 51: processDivideByValue(x, y, data, allFrames, step, params); break;
                // New filters (52-79)
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