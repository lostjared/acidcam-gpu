#include "ac-gpu/ac-gpu.hpp"
#include <cuda_runtime.h>
#include <string>

namespace ac_gpu {

    Filter filters[] = {
        { 0, "SelfAlphaBlend" },
        { 1, "MedianBlend" },
        { 2, "MedianBlend" },
        { 3, "SquareBlockResize" },
        { 4, "SelfAlphaScaleRefined" },
        { 5, "StrangeGlitch" },
        { 6, "MatrixOutline" },
        { 7, "AuraTrails" },
        { 8, "MirrorReverseColor" },
        { 9, "ImageSquareShrink" },
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
        { 40, "LineInLineOut" }
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