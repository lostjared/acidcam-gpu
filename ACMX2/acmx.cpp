/**
 * @file acmx.cpp
 * @brief ACMX2 — Real-time GPU-accelerated video glitch engine.
 *
 * This file implements the complete ACMX2 command-line application, the core
 * engine of the acidcam-gpu project. It combines CUDA GPU filters, GLSL shader
 * processing, OpenGL rendering, optional 3D model support, audio reactivity,
 * MIDI controller input, and video recording into a single real-time pipeline.
 *
 * @section arch Architecture Overview
 * - **TextureUploader** — Zero-copy CUDA↔OpenGL PBO interop for GPU frames.
 * - **ShaderCache / ShaderLibrary** — Compile, cache, and manage GLSL shader programs.
 * - **FrameCache** — Ring-buffer of recent frames for temporal ("cache") shaders.
 * - **SnapshotThreadPool** — Async PNG snapshot writer.
 * - **ACView** — Main GL object: capture → filter → shade → record pipeline.
 * - **MainWindow** — SDL2/OpenGL window host.
 *
 * @copyright (C) 2026 LostSideDead Software — BSD 2-Clause License
 * @see https://lostsidedead.biz
 */

#include "version_info.hpp"
#include <algorithm>
#include <argz.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <ctime>
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gl.hpp>
#include <iomanip>
#include <mutex>
#include <mx.hpp>
#include "../MXWrite/mxwrite.hpp"
#include <opencv2/opencv.hpp>
#include <optional>
#include <queue>
#include <sstream>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <map>
#include <vector>
#ifdef AUDIO_ENABLED
#include "audio.hpp"
#include "file_audio.hpp"
#endif
#ifdef MIDI_ENABLED
#include <rtmidi/RtMidi.h>
#endif
#include "program.hpp"
#ifdef ACMX2_WITH_WEBP
#include <webp/encode.h>
#endif
#ifdef ACMX2_WITH_TIFF
#include <tiffio.h>
#endif
#ifdef ACMX2_WITH_CUDA
#include <ac-gpu/ac-gpu.hpp>
#include <cuda_gl_interop.h>
#else
// Stubs so code compiled without CUDA still has the symbols it references.
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { (void)(call); } while (0)
#endif
namespace ac_gpu {
    inline constexpr int AC_FILTER_MAX = 0;
    struct Filter { int index; std::string name; };
    struct GPUFilter { int index; };
    struct DynamicFrameBuffer {
        int arraySize = 0;
    };
    // Empty filter table so code referencing ac_gpu::filters still compiles.
    // (Never indexed in no-CUDA builds because AC_FILTER_MAX == 0 guards all uses.)
    inline Filter filters[1] = {{0, ""}};
}
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#ifdef __linux__
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <deque>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <model.hpp>
#ifdef ACMX2_WITH_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
#include <opencv2/opencv.hpp>
#include <string_view>

/// @brief Copy the audio track from one media file to another via FFmpeg.
void transfer_audio(std::string_view, std::string_view);

static std::string safeGLString(GLenum name) {
    const GLubyte *value = glGetString(name);
    if (!value) {
        return "unavailable";
    }
    return reinterpret_cast<const char *>(value);
}

static std::optional<std::string> normalizeShaderIndexEntry(const std::string &raw) {
    const auto first = raw.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return std::nullopt;
    }
    const auto last = raw.find_last_not_of(" \t\r\n");
    std::string entry = raw.substr(first, last - first + 1);

    if (entry.empty() || entry.find("material") != std::string::npos) {
        return std::nullopt;
    }

    std::replace(entry.begin(), entry.end(), '\\', '/');
    std::filesystem::path rel(entry);
    if (rel.is_absolute()) {
        return std::nullopt;
    }

    std::filesystem::path normalized = rel.lexically_normal();
    std::string normalized_str = normalized.generic_string();
    while (normalized_str.rfind("./", 0) == 0) {
        normalized_str.erase(0, 2);
    }

    if (normalized_str.empty() || normalized_str == "." || normalized_str == "..") {
        return std::nullopt;
    }
    if (normalized_str.rfind("../", 0) == 0 ||
        normalized_str.find("/../") != std::string::npos ||
        (normalized_str.size() >= 3 && normalized_str.compare(normalized_str.size() - 3, 3, "/..") == 0)) {
        return std::nullopt;
    }

    return normalized_str;
}

static bool resolveShaderPathInLibrary(const std::string &library_path,
                                       const std::string &relative_path,
                                       std::string &resolved_full_path) {
    std::error_code ec;
    std::filesystem::path base = std::filesystem::weakly_canonical(std::filesystem::path(library_path), ec);
    if (ec) {
        ec.clear();
        base = std::filesystem::absolute(std::filesystem::path(library_path), ec);
        if (ec) {
            return false;
        }
    }

    std::filesystem::path target = std::filesystem::weakly_canonical(base / std::filesystem::path(relative_path), ec);
    if (ec) {
        return false;
    }

    const std::filesystem::path relative = target.lexically_relative(base);
    const std::string relative_str = relative.generic_string();
    if (relative.empty() ||
        relative_str == ".." ||
        relative_str.rfind("../", 0) == 0) {
        return false;
    }

    if (!std::filesystem::exists(target) || !std::filesystem::is_regular_file(target)) {
        return false;
    }

    resolved_full_path = target.string();
    return true;
}

// ---------------------------------------------------------------------------
// Graceful shutdown for headless / --silent mode.
//
// When acmx2 is running without a visible window (offscreen / batch mode),
// pressing Ctrl+C should stop encoding, flush the writer, and close the
// output file cleanly so the partial video is still playable. We install a
// POSIX signal handler that does nothing but set an atomic flag; the main
// render loop polls this flag each frame and triggers ACView::requestStop()
// which unwinds the normal EOF path (flush encoder, close file, trailer).
// ---------------------------------------------------------------------------
namespace {
    std::atomic<bool> g_shutdown_requested{false};

#if defined(__linux__)
    extern "C" void acmx2_signal_handler(int /*sig*/) {
        // Async-signal-safe: only a relaxed atomic store.
        g_shutdown_requested.store(true, std::memory_order_relaxed);
    }

    void installHeadlessSignalHandlers() {
        struct sigaction sa{};
        sa.sa_handler = &acmx2_signal_handler;
        sigemptyset(&sa.sa_mask);
        // No SA_RESTART: let blocking syscalls (e.g. read) return EINTR so
        // the decoder thread can observe shutdown quickly.
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGHUP, &sa, nullptr);
    }
#else
    void installHeadlessSignalHandlers() {}
#endif
}

#if defined(__APPLE__)
class ScopedStderrSilence {
  public:
    ScopedStderrSilence() {
        saved_stderr_fd = dup(STDERR_FILENO);
        if (saved_stderr_fd < 0) {
            return;
        }
        null_fd = open("/dev/null", O_WRONLY);
        if (null_fd < 0) {
            close(saved_stderr_fd);
            saved_stderr_fd = -1;
            return;
        }
        if (dup2(null_fd, STDERR_FILENO) < 0) {
            close(null_fd);
            close(saved_stderr_fd);
            null_fd = -1;
            saved_stderr_fd = -1;
            return;
        }
        active = true;
    }

    ~ScopedStderrSilence() {
        if (saved_stderr_fd >= 0) {
            dup2(saved_stderr_fd, STDERR_FILENO);
            close(saved_stderr_fd);
        }
        if (null_fd >= 0) {
            close(null_fd);
        }
    }

    ScopedStderrSilence(const ScopedStderrSilence &) = delete;
    ScopedStderrSilence &operator=(const ScopedStderrSilence &) = delete;

  private:
    int saved_stderr_fd = -1;
    int null_fd = -1;
    bool active = false;
};
#endif

// ---------------------------------------------------------------------------
// Manual 10-bit BT.2020 YUV -> 16-bit RGBA converter.
//
// swscale's @c sws_setColorspaceDetails is unreliable — on several internal
// paths it silently falls back to BT.601 coefficients even when the caller
// requests BT.2020, which tints HDR decoding pink/green. We sidestep the
// problem entirely by doing the YUV->RGB matrix on the CPU with the exact
// coefficients defined in ITU-R BT.2100 / BT.2020 non-constant-luminance.
//
// Input: AV_PIX_FMT_YUV420P10LE or AV_PIX_FMT_P010LE, limited-range tv levels
// (Y in [64,940], UV in [64,960] at 10-bit). UV is upsampled with nearest-
// neighbour (swscale's default). The output is packed RGBA16 (R,G,B,A, 16
// bits per channel, little-endian host) with A = 0xFFFF — exactly the bytes
// the GL @c uploadHdrFrame path expects.
//
// The transfer function (PQ or HLG) is *not* applied here: the bits stay in
// the non-linear BT.2020 RGB' encoding that the GL @c kHdrDecodeFrag shader
// converts to scene-linear. Bit precision is preserved by keeping the math
// in float and scaling to uint16_t only at the end.
// ---------------------------------------------------------------------------
inline bool convertBt2020Yuv10LimitedToRgba16(const AVFrame *src, cv::Mat &out) {
    if (!src || !src->data[0] || !src->data[1]) {
        return false;
    }
    const int w = src->width;
    const int h = src->height;
    if (w <= 0 || h <= 0) {
        return false;
    }

    const bool is_p010 = (src->format == AV_PIX_FMT_P010LE);
    if (!is_p010 && !src->data[2]) {
        return false;  // planar formats need all three planes.
    }

    // Detect 10-bit sample position within a 16-bit container:
    //   yuv420p10le  -> low 10 bits (shift = 0)
    //   p010le       -> high 10 bits (shift = 6, so divide by 64)
    int sample_shift = 0;
    if (is_p010) {
        sample_shift = 6;
    }

    const int y_stride_b = src->linesize[0];
    const int uv_stride_b = src->linesize[1];   // UV interleaved (p010) or Cb (planar).
    const int v_stride_b = is_p010 ? 0 : src->linesize[2];

    // BT.2020 non-constant-luminance inverse matrix (per ITU-R BT.2020 §4).
    constexpr float kCrR = 1.4746f;
    constexpr float kCbG = -0.16455312684366f;   // -2*(1-0.2627)*0.2627/0.6780
    constexpr float kCrG = -0.57135313725490f;   // -2*(1-0.0593)*0.0593/0.6780
    constexpr float kCbB = 1.8814f;

    out.create(h, w, CV_16UC4);

    auto sample10 = [&](const uint8_t *plane, int stride_b, int x, int y) -> int {
        const uint16_t raw = *reinterpret_cast<const uint16_t *>(
            plane + y * stride_b + x * 2);
        return static_cast<int>(raw >> sample_shift);
    };

    const uint8_t *yp = src->data[0];
    const uint8_t *up = src->data[1];  // planar: Cb plane | p010: interleaved Cb,Cr.
    const uint8_t *vp = is_p010 ? nullptr : src->data[2];

    // 10-bit limited-range BT.2020 scaling:
    //   Y' = (Y_sample - 64)  / 876    (876 = 940-64)
    //   Cb = (Cb_sample - 512)/ 896    (896 = 2*448)
    //   Cr = (Cr_sample - 512)/ 896
    constexpr float kInvY = 1.0f / 876.0f;
    constexpr float kInvC = 1.0f / 896.0f;

    for (int y = 0; y < h; ++y) {
        const int cy = y >> 1;  // 4:2:0 vertical subsampling, nearest.
        uint16_t *dst = out.ptr<uint16_t>(y);
        for (int x = 0; x < w; ++x) {
            const int cx = x >> 1;
            const int Ys = sample10(yp, y_stride_b, x, y);
            int Cbs;
            int Crs;
            if (is_p010) {
                // Interleaved Cb,Cr: each chroma pixel is 2 uint16 samples
                // (Cb then Cr). Row stride is still linesize[1] bytes.
                const uint16_t *row = reinterpret_cast<const uint16_t *>(
                    up + cy * uv_stride_b);
                Cbs = static_cast<int>(row[cx * 2 + 0]) >> sample_shift;
                Crs = static_cast<int>(row[cx * 2 + 1]) >> sample_shift;
            } else {
                Cbs = sample10(up, uv_stride_b, cx, cy);
                Crs = sample10(vp, v_stride_b, cx, cy);
            }

            const float Y  = (Ys  - 64)  * kInvY;
            const float Cb = (Cbs - 512) * kInvC;
            const float Cr = (Crs - 512) * kInvC;

            float R = Y + kCrR * Cr;
            float G = Y + kCbG * Cb + kCrG * Cr;
            float B = Y + kCbB * Cb;

            // Clamp to [0,1] — HLG/PQ code values outside this range are
            // not legal in the non-linear domain.
            R = R < 0.0f ? 0.0f : (R > 1.0f ? 1.0f : R);
            G = G < 0.0f ? 0.0f : (G > 1.0f ? 1.0f : G);
            B = B < 0.0f ? 0.0f : (B > 1.0f ? 1.0f : B);

            dst[x * 4 + 0] = static_cast<uint16_t>(R * 65535.0f + 0.5f);
            dst[x * 4 + 1] = static_cast<uint16_t>(G * 65535.0f + 0.5f);
            dst[x * 4 + 2] = static_cast<uint16_t>(B * 65535.0f + 0.5f);
            dst[x * 4 + 3] = 0xFFFF;
        }
    }
    return true;
}

inline float clamp01f(float v) {
    return (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline float pqToLinearScalar(float e) {
    // SMPTE ST.2084 inverse EOTF, output normalized so 1.0 == 10000 nits.
    constexpr float m1 = 0.1593017578125f;
    constexpr float m2 = 78.84375f;
    constexpr float c1 = 0.8359375f;
    constexpr float c2 = 18.8515625f;
    constexpr float c3 = 18.6875f;
    e = clamp01f(e);
    const float p = std::pow(e, 1.0f / m2);
    const float num = std::max(p - c1, 0.0f);
    const float den = c2 - c3 * p;
    if (den <= 0.0f) {
        return 0.0f;
    }
    return std::pow(num / den, 1.0f / m1);
}

inline float hlgToLinearScalar(float e) {
    // ARIB STD-B67 inverse OETF.
    constexpr float a = 0.17883277f;
    constexpr float b = 0.28466892f;
    constexpr float c = 0.55991073f;
    e = clamp01f(e);
    if (e <= 0.5f) {
        return (e * e) / 3.0f;
    }
    return (std::exp((e - c) / a) + b) / 12.0f;
}

inline unsigned char linearToSrgb8(float v) {
    v = clamp01f(v);
    float srgb = 0.0f;
    if (v <= 0.0031308f) {
        srgb = 12.92f * v;
    } else {
        srgb = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
    const int iv = static_cast<int>(srgb * 255.0f + 0.5f);
    return static_cast<unsigned char>(std::clamp(iv, 0, 255));
}

inline uint16_t linearToSrgb16(float v) {
    v = clamp01f(v);
    float srgb = 0.0f;
    if (v <= 0.0031308f) {
        srgb = 12.92f * v;
    } else {
        srgb = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
    const int iv = static_cast<int>(srgb * 65535.0f + 0.5f);
    return static_cast<uint16_t>(std::clamp(iv, 0, 65535));
}

inline std::vector<unsigned char> toneMapHdrRgba16ToSdrRgba8(const std::vector<unsigned char> &hdr_pixels,
                                                              int w,
                                                              int h,
                                                              int hdr_trc) {
    std::vector<unsigned char> sdr_pixels(static_cast<size_t>(w) * static_cast<size_t>(h) * 4, 0);
    const bool is_hlg = (hdr_trc == AVCOL_TRC_ARIB_STD_B67);

    // Linear BT.2020 -> Linear sRGB (D65).
    constexpr float m00 = 1.6605f, m01 = -0.5876f, m02 = -0.0728f;
    constexpr float m10 = -0.1246f, m11 = 1.1329f, m12 = -0.0083f;
    constexpr float m20 = -0.0182f, m21 = -0.1006f, m22 = 1.1187f;

    auto read_u16_le = [&](size_t byte_index) -> uint16_t {
        return static_cast<uint16_t>(static_cast<uint16_t>(hdr_pixels[byte_index]) |
                                     (static_cast<uint16_t>(hdr_pixels[byte_index + 1]) << 8));
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t px = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            const size_t src_idx = px * 8;
            const size_t dst_idx = px * 4;

            const float enc_r = read_u16_le(src_idx + 0) / 65535.0f;
            const float enc_g = read_u16_le(src_idx + 2) / 65535.0f;
            const float enc_b = read_u16_le(src_idx + 4) / 65535.0f;

            float lin2020_r = 0.0f;
            float lin2020_g = 0.0f;
            float lin2020_b = 0.0f;

            if (is_hlg) {
                lin2020_r = hlgToLinearScalar(enc_r) * 3.77358491f;
                lin2020_g = hlgToLinearScalar(enc_g) * 3.77358491f;
                lin2020_b = hlgToLinearScalar(enc_b) * 3.77358491f;
            } else {
                lin2020_r = pqToLinearScalar(enc_r) * 100.0f;
                lin2020_g = pqToLinearScalar(enc_g) * 100.0f;
                lin2020_b = pqToLinearScalar(enc_b) * 100.0f;
            }

            float sr = m00 * lin2020_r + m01 * lin2020_g + m02 * lin2020_b;
            float sg = m10 * lin2020_r + m11 * lin2020_g + m12 * lin2020_b;
            float sb = m20 * lin2020_r + m21 * lin2020_g + m22 * lin2020_b;

            sr = std::max(sr, 0.0f);
            sg = std::max(sg, 0.0f);
            sb = std::max(sb, 0.0f);

            // Simple global tone map for SDR preview/export.
            sr = sr / (1.0f + sr);
            sg = sg / (1.0f + sg);
            sb = sb / (1.0f + sb);

            sdr_pixels[dst_idx + 0] = linearToSrgb8(sr);
            sdr_pixels[dst_idx + 1] = linearToSrgb8(sg);
            sdr_pixels[dst_idx + 2] = linearToSrgb8(sb);
            sdr_pixels[dst_idx + 3] = 255;
        }
    }

    return sdr_pixels;
}

// 16-bit-per-channel variant: tone-maps PQ/HLG BT.2020 RGBA16 -> sRGB RGBA16
// (gamma-encoded sRGB, contiguous LE uint16 samples). Used by the HDR TIFF
// snapshot path so the resulting file is correctly viewable on any display.
inline std::vector<uint16_t> toneMapHdrRgba16ToSrgbRgba16(const unsigned char *hdr_pixels,
                                                          int w,
                                                          int h,
                                                          int hdr_trc) {
    std::vector<uint16_t> sdr_pixels(static_cast<size_t>(w) * static_cast<size_t>(h) * 4, 0);
    const bool is_hlg = (hdr_trc == AVCOL_TRC_ARIB_STD_B67);

    constexpr float m00 = 1.6605f, m01 = -0.5876f, m02 = -0.0728f;
    constexpr float m10 = -0.1246f, m11 = 1.1329f, m12 = -0.0083f;
    constexpr float m20 = -0.0182f, m21 = -0.1006f, m22 = 1.1187f;

    auto read_u16_le = [&](size_t byte_index) -> uint16_t {
        return static_cast<uint16_t>(static_cast<uint16_t>(hdr_pixels[byte_index]) |
                                     (static_cast<uint16_t>(hdr_pixels[byte_index + 1]) << 8));
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t px = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            const size_t src_idx = px * 8;
            const size_t dst_idx = px * 4;

            const float enc_r = read_u16_le(src_idx + 0) / 65535.0f;
            const float enc_g = read_u16_le(src_idx + 2) / 65535.0f;
            const float enc_b = read_u16_le(src_idx + 4) / 65535.0f;

            float lin2020_r = 0.0f, lin2020_g = 0.0f, lin2020_b = 0.0f;
            if (is_hlg) {
                lin2020_r = hlgToLinearScalar(enc_r) * 3.77358491f;
                lin2020_g = hlgToLinearScalar(enc_g) * 3.77358491f;
                lin2020_b = hlgToLinearScalar(enc_b) * 3.77358491f;
            } else {
                lin2020_r = pqToLinearScalar(enc_r) * 100.0f;
                lin2020_g = pqToLinearScalar(enc_g) * 100.0f;
                lin2020_b = pqToLinearScalar(enc_b) * 100.0f;
            }

            float sr = m00 * lin2020_r + m01 * lin2020_g + m02 * lin2020_b;
            float sg = m10 * lin2020_r + m11 * lin2020_g + m12 * lin2020_b;
            float sb = m20 * lin2020_r + m21 * lin2020_g + m22 * lin2020_b;

            sr = std::max(sr, 0.0f);
            sg = std::max(sg, 0.0f);
            sb = std::max(sb, 0.0f);

            sr = sr / (1.0f + sr);
            sg = sg / (1.0f + sg);
            sb = sb / (1.0f + sb);

            sdr_pixels[dst_idx + 0] = linearToSrgb16(sr);
            sdr_pixels[dst_idx + 1] = linearToSrgb16(sg);
            sdr_pixels[dst_idx + 2] = linearToSrgb16(sb);
            sdr_pixels[dst_idx + 3] = 65535;
        }
    }

    return sdr_pixels;
}

#ifdef ACMX2_WITH_WEBP
inline bool saveSdrWebPFromRgba8(const char *filename,
                                 const unsigned char *rgba8,
                                 int width,
                                 int height) {
    if (filename == nullptr || rgba8 == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    uint8_t *output = nullptr;
    const int stride = width * 4;
    const size_t out_size = WebPEncodeLosslessRGBA(rgba8, width, height, stride, &output);
    if (out_size == 0 || output == nullptr) {
        if (output != nullptr) {
            WebPFree(output);
        }
        return false;
    }

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        WebPFree(output);
        return false;
    }
    ofs.write(reinterpret_cast<const char *>(output), static_cast<std::streamsize>(out_size));
    const bool ok = ofs.good();
    ofs.close();
    WebPFree(output);
    return ok;
}

// Save an HDR snapshot as a WebP file.
//
// libwebp's bitstream is fundamentally 8-bit per channel and has no HDR
// metadata, so we tone-map the PQ/HLG BT.2020 input to sRGB before encoding.
// The resulting lossless RGBA WebP displays correctly on both SDR and HDR
// viewers (including phone HDR displays which would otherwise interpret
// PQ-encoded bytes as sRGB and produce washed-out colours).
inline bool saveHdrWebPFromRgba16(const char *filename,
                                  const unsigned char *rgba16,
                                  int width,
                                  int height,
                                  int hdr_trc) {
    if (filename == nullptr || rgba16 == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    // Reuse the existing tone-mapper. It expects a vector view of the input.
    const std::vector<unsigned char> hdr_view(rgba16, rgba16 + pixel_count * 8);
    const std::vector<unsigned char> rgba8 =
        toneMapHdrRgba16ToSdrRgba8(hdr_view, width, height, hdr_trc);

    uint8_t *output = nullptr;
    const int stride = width * 4;
    const size_t out_size = WebPEncodeLosslessRGBA(rgba8.data(), width, height, stride, &output);
    if (out_size == 0 || output == nullptr) {
        if (output != nullptr) {
            WebPFree(output);
        }
        return false;
    }

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        WebPFree(output);
        return false;
    }
    ofs.write(reinterpret_cast<const char *>(output), static_cast<std::streamsize>(out_size));
    const bool ok = ofs.good();
    ofs.close();
    WebPFree(output);
    return ok;
}
#endif  // ACMX2_WITH_WEBP

#ifdef ACMX2_WITH_TIFF
inline bool saveSdrTiffFromRgba8(const char *filename,
                                 const unsigned char *rgba8,
                                 int width,
                                 int height) {
    if (filename == nullptr || rgba8 == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    TIFF *tif = TIFFOpen(filename, "w");
    if (tif == nullptr) {
        return false;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(width));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(height));
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));

    const uint16_t extra[1] = { EXTRASAMPLE_UNASSALPHA };
    TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, extra);
    TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION,
                 "ACMX2 SDR snapshot: 8-bit RGBA TIFF");

    const tmsize_t row_bytes = static_cast<tmsize_t>(width) * 4;
    bool ok = true;
    for (int y = 0; y < height; ++y) {
        unsigned char *row = const_cast<unsigned char *>(rgba8 + static_cast<size_t>(y) * static_cast<size_t>(row_bytes));
        if (TIFFWriteScanline(tif, row, static_cast<uint32_t>(y), 0) < 0) {
            ok = false;
            break;
        }
    }

    TIFFClose(tif);
    return ok;
}

// Save an HDR snapshot as a 16-bit RGBA TIFF.
//
// We tone-map the PQ/HLG BT.2020 input to sRGB at full 16-bit precision
// before writing. This keeps highlight detail (no 8-bit quantisation) while
// producing a file that displays correctly on every viewer — without the
// PQ-as-sRGB washed-out appearance that bare PQ data would have.
inline bool saveHdrTiffFromRgba16(const char *filename,
                                  const unsigned char *rgba16,
                                  int width,
                                  int height,
                                  int hdr_trc) {
    if (filename == nullptr || rgba16 == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    const std::vector<uint16_t> srgb16 =
        toneMapHdrRgba16ToSrgbRgba16(rgba16, width, height, hdr_trc);

    TIFF *tif = TIFFOpen(filename, "w");
    if (tif == nullptr) {
        return false;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(width));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(height));
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));

    const uint16_t extra[1] = { EXTRASAMPLE_UNASSALPHA };
    TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, extra);
    TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION,
                 "ACMX2 HDR snapshot: 16-bit RGBA, sRGB tone-mapped from BT.2020 PQ/HLG");

    const tmsize_t row_samples = static_cast<tmsize_t>(width) * 4;
    bool ok = true;
    for (int y = 0; y < height; ++y) {
        // libtiff's WriteScanline takes a non-const buffer.
        uint16_t *row = const_cast<uint16_t *>(srgb16.data() + static_cast<size_t>(y) * static_cast<size_t>(row_samples));
        if (TIFFWriteScanline(tif, row, static_cast<uint32_t>(y), 0) < 0) {
            ok = false;
            break;
        }
    }

    TIFFClose(tif);
    return ok;
}
#endif  // ACMX2_WITH_TIFF

// ---------------------------------------------------------------------------
// HDR pipeline shader sources.
//
// kHdrVertPassthrough is a minimal NDC-quad vertex shader that exposes a
// `tc` varying for the fragment shaders below. It matches the 2D sprite
// vertex attribute layout used by gl::GLSprite (position + texcoord).
//
// kHdrDecodeFrag runs once per video frame. It reads the freshly-uploaded
// 16-bit normalised PQ/HLG-encoded BT.2020 source texture and writes linear
// BT.2020 light (reference white = 1.0, so HDR highlights end up as values
// well above 1.0) into an RGBA16F target. Subsequent user shaders sample
// that linear texture via iChannel0/samp — they do not need any changes.
//
// kHdrEncodeFrag runs once after the user's last shader pass. It reads the
// final linear BT.2020 RGBA16F result and produces PQ-encoded values in
// [0,1], ready to be packed into a 16-bit normalised RGBA16 readback
// target (MXWrite converts that to a 10-bit BT.2020 YUV P010 HEVC stream).
//
// The HLG inverse/forward OETFs are gated by a `uniform int transfer` so
// the same two shaders handle both SMPTE ST.2084 (PQ) and ARIB STD-B67
// (HLG) inputs. transfer: 1 = PQ, 2 = HLG.
// ---------------------------------------------------------------------------
constexpr const char *kHdrVertPassthrough =
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in vec2 aTex;\n"
    "out vec2 tc;\n"
    "uniform mat4 mv_matrix;\n"
    "uniform mat4 proj_matrix;\n"
    "void main() {\n"
    "    gl_Position = proj_matrix * mv_matrix * vec4(aPos, 1.0);\n"
    "    tc = aTex;\n"
    "}\n";

constexpr const char *kHdrDecodeFrag =
    "#version 330 core\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "uniform sampler2D samp;\n"
    "uniform int transfer;  // 1 = PQ (SMPTE2084), 2 = HLG (ARIB STD-B67)\n"
    "// SMPTE ST.2084 inverse EOTF. Input: PQ code in [0,1]. Output: linear\n"
    "// fractional luminance where 1.0 = PQ peak (10000 nits).\n"
    "vec3 pqToLinear(vec3 e) {\n"
    "    const float m1 = 2610.0 / 16384.0;\n"
    "    const float m2 = (2523.0 / 4096.0) * 128.0;\n"
    "    const float c1 = 3424.0 / 4096.0;\n"
    "    const float c2 = (2413.0 / 4096.0) * 32.0;\n"
    "    const float c3 = (2392.0 / 4096.0) * 32.0;\n"
    "    vec3 ec = clamp(e, vec3(0.0), vec3(1.0));\n"
    "    vec3 em2 = pow(ec, vec3(1.0 / m2));\n"
    "    vec3 num = max(em2 - c1, vec3(0.0));\n"
    "    // Clamp denominator floor to avoid div-by-zero / sign-flip near\n"
    "    // peak code values which would otherwise propagate NaN through\n"
    "    // pow() and read back as zero on some drivers.\n"
    "    vec3 den = max(c2 - c3 * em2, vec3(1e-6));\n"
    "    return pow(num / den, vec3(1.0 / m1));\n"
    "}\n"
    "// ARIB STD-B67 (HLG) inverse OETF. Input HLG code [0,1]. Output scene\n"
    "// linear [0,1] normalised so 0.5 HLG -> ~0.083 linear (reference white).\n"
    "vec3 hlgToLinear(vec3 e) {\n"
    "    const float a = 0.17883277;\n"
    "    const float b = 0.28466892;\n"
    "    const float c = 0.55991073;\n"
    "    vec3 ec = clamp(e, vec3(0.0), vec3(1.0));\n"
    "    vec3 lo = (ec * ec) / 3.0;\n"
    "    vec3 hi = (exp((ec - c) / a) + b) / 12.0;\n"
    "    return mix(lo, hi, step(vec3(0.5), ec));\n"
    "}\n"
    "void main() {\n"
    "    vec4 raw = texture(samp, tc);\n"
    "    vec3 lin;\n"
    "    if (transfer == 2) {\n"
    "        // HLG: inverse-OETF then scale so reference white == 1.0.\n"
    "        // Reference white sits at ~0.26 scene-linear; scale by 1/0.26\n"
    "        // ~= 3.77 so shader colours match SDR at nominal exposure.\n"
    "        lin = hlgToLinear(raw.rgb) * 3.77358491;\n"
    "    } else {\n"
    "        // PQ: inverse-EOTF, then rescale so 100 nits == 1.0 (reference\n"
    "        // white). 100 / 10000 = 0.01, so we multiply by 100. Highlights\n"
    "        // above reference end up as values > 1.0 (legal in RGBA16F).\n"
    "        lin = pqToLinear(raw.rgb) * 100.0;\n"
    "    }\n"
    "    color = vec4(lin, raw.a);\n"
    "}\n";

constexpr const char *kHdrEncodeFrag =
    "#version 330 core\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "uniform sampler2D samp;\n"
    "uniform int transfer;\n"
    "vec3 linearToPq(vec3 L) {\n"
    "    const float m1 = 2610.0 / 16384.0;\n"
    "    const float m2 = (2523.0 / 4096.0) * 128.0;\n"
    "    const float c1 = 3424.0 / 4096.0;\n"
    "    const float c2 = (2413.0 / 4096.0) * 32.0;\n"
    "    const float c3 = (2392.0 / 4096.0) * 32.0;\n"
    "    vec3 Lm = pow(max(L, vec3(0.0)), vec3(m1));\n"
    "    vec3 num = c1 + c2 * Lm;\n"
    "    vec3 den = 1.0 + c3 * Lm;\n"
    "    return pow(num / den, vec3(m2));\n"
    "}\n"
    "vec3 linearToHlg(vec3 L) {\n"
    "    const float a = 0.17883277;\n"
    "    const float b = 0.28466892;\n"
    "    const float c = 0.55991073;\n"
    "    vec3 Lc = max(L, vec3(0.0));\n"
    "    vec3 lo = sqrt(3.0 * Lc);\n"
    "    // Floor the log argument so the unused 'hi' branch never produces\n"
    "    // NaN. mix() with NaN is mix(lo, NaN, 0) = lo + NaN*0 = NaN, which\n"
    "    // would surface as black pixels after the final clamp/UNORM cast.\n"
    "    vec3 hi = a * log(max(12.0 * Lc - b, vec3(1e-6))) + c;\n"
    "    return mix(lo, hi, step(vec3(1.0 / 12.0), Lc));\n"
    "}\n"
    "void main() {\n"
    "    vec3 lin = texture(samp, tc).rgb;\n"
    "    // Sanitise potentially-NaN/Inf values from prior user shader\n"
    "    // computations so they do not survive the OETF and read back as\n"
    "    // zero (=> black holes in dark/edge regions).\n"
    "    bvec3 nans = isnan(lin);\n"
    "    bvec3 infs = isinf(lin);\n"
    "    lin = mix(lin, vec3(0.0), vec3(nans));\n"
    "    lin = mix(lin, vec3(0.0), vec3(infs));\n"
    "    lin = max(lin, vec3(0.0));\n"
    "    vec3 enc;\n"
    "    if (transfer == 2) {\n"
    "        enc = linearToHlg(lin / 3.77358491);\n"
    "    } else {\n"
    "        enc = linearToPq(lin / 100.0);\n"
    "    }\n"
    "    color = vec4(clamp(enc, 0.0, 1.0), 1.0);\n"
    "}\n";

// Display shader that can optionally flip the Y coordinate of texture sampling.
// Used when --flip is set for windowed display.
constexpr const char *kDisplayVertFlip =
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in vec2 aTex;\n"
    "out vec2 tc;\n"
    "uniform mat4 mv_matrix;\n"
    "uniform mat4 proj_matrix;\n"
    "uniform int flip_y;  // 1 to flip Y coordinate\n"
    "void main() {\n"
    "    gl_Position = proj_matrix * mv_matrix * vec4(aPos, 1.0);\n"
    "    vec2 tex = aTex;\n"
    "    if (flip_y == 1) {\n"
    "        tex.y = 1.0 - tex.y;\n"
    "    }\n"
    "    tc = tex;\n"
    "}\n";

constexpr const char *kDisplayFragPassthrough =
    "#version 330 core\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "uniform sampler2D samp;\n"
    "void main() {\n"
    "    color = texture(samp, tc);\n"
    "}\n";

class FFMpegVideoReader {
  public:
    ~FFMpegVideoReader() {
        close();
    }

    bool open(const std::string &filename, bool prefer_cuda) {
        close();

        if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr) < 0) {
            return false;
        }
        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            return false;
        }

        stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index < 0) {
            return false;
        }

        AVStream *stream = format_ctx->streams[stream_index];
        const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec) {
            return false;
        }

        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            return false;
        }

        if (avcodec_parameters_to_context(codec_ctx, stream->codecpar) < 0) {
            return false;
        }

        codec_ctx->thread_count = std::max(1u, std::thread::hardware_concurrency());

#ifdef ACMX2_WITH_CUDA
        if (prefer_cuda) {
            enableCudaHwDecode(codec);
        }
#else
        (void)prefer_cuda;
#endif

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            return false;
        }

        decoded_frame = av_frame_alloc();
        sw_frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!decoded_frame || !sw_frame || !packet) {
            return false;
        }

        width = codec_ctx->width;
        height = codec_ctx->height;
        fps = av_q2d(stream->avg_frame_rate);
        if (fps <= 0.0) {
            fps = av_q2d(stream->r_frame_rate);
        }
        frame_count = static_cast<double>(stream->nb_frames);
        current_frame = 0;

        // HDR detection: BT.2020 primaries, PQ (SMPTE2084) or HLG (ARIB STD-B67)
        // transfer characteristics, or >=10-bit raw sample depth all indicate HDR.
        {
            const AVCodecParameters *par = stream->codecpar;
            const bool primaries_hdr = (par->color_primaries == AVCOL_PRI_BT2020);
            const bool trc_hdr = (par->color_trc == AVCOL_TRC_SMPTE2084) ||
                                 (par->color_trc == AVCOL_TRC_ARIB_STD_B67) ||
                                 (par->color_trc == AVCOL_TRC_BT2020_10) ||
                                 (par->color_trc == AVCOL_TRC_BT2020_12);
            const bool space_hdr = (par->color_space == AVCOL_SPC_BT2020_NCL) ||
                                   (par->color_space == AVCOL_SPC_BT2020_CL);
            int bpp = par->bits_per_raw_sample;
            if (bpp <= 0) {
                // Fall back to codec context pixel format depth. Many HDR
                // files leave bits_per_raw_sample unset in the codecpar but
                // the decoded format (yuv420p10le etc.) still reports 10-bit.
                const AVPixelFormat pf = codec_ctx->pix_fmt;
                if (pf != AV_PIX_FMT_NONE) {
                    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pf);
                    if (desc && desc->nb_components > 0) {
                        bpp = desc->comp[0].depth;
                    }
                }
            }
            const bool depth_hdr = (bpp >= 10);
            is_hdr = (primaries_hdr || trc_hdr || space_hdr) && depth_hdr;
            hdr_primaries = par->color_primaries;
            hdr_trc = par->color_trc;
            hdr_colorspace = par->color_space;
            hdr_color_range = par->color_range;
            hdr_bit_depth = bpp;

            // Capture mastering display + content light side data from the
            // input stream so they can be attached to the output when in HDR
            // mode. Side data lives on codecpar->coded_side_data in modern
            // ffmpeg.
            hdr_mastering_display.clear();
            hdr_content_light.clear();
            for (int i = 0; i < par->nb_coded_side_data; ++i) {
                const AVPacketSideData &sd = par->coded_side_data[i];
                if (sd.type == AV_PKT_DATA_MASTERING_DISPLAY_METADATA) {
                    hdr_mastering_display.assign(sd.data, sd.data + sd.size);
                } else if (sd.type == AV_PKT_DATA_CONTENT_LIGHT_LEVEL) {
                    hdr_content_light.assign(sd.data, sd.data + sd.size);
                }
            }
        }

        return true;
    }

    bool read(cv::Mat &out_bgr) {
        if (!codec_ctx || !format_ctx) {
            return false;
        }

        while (true) {
            if (!draining) {
                const int read_ret = av_read_frame(format_ctx, packet);
                if (read_ret >= 0) {
                    if (packet->stream_index == stream_index) {
                        if (avcodec_send_packet(codec_ctx, packet) < 0) {
                            av_packet_unref(packet);
                            return false;
                        }
                    }
                    av_packet_unref(packet);
                } else {
                    if (!drain_packet_sent) {
                        if (avcodec_send_packet(codec_ctx, nullptr) < 0) {
                            return false;
                        }
                        drain_packet_sent = true;
                        draining = true;
                    }
                }
            }

            const int receive_ret = avcodec_receive_frame(codec_ctx, decoded_frame);
            if (receive_ret == AVERROR(EAGAIN)) {
                if (draining) {
                    return false;
                }
                continue;
            }
            if (receive_ret == AVERROR_EOF) {
                return false;
            }
            if (receive_ret < 0) {
                return false;
            }

            AVFrame *src = decoded_frame;
            if (hw_decode_enabled && decoded_frame->format == hw_pix_fmt) {
                if (av_hwframe_transfer_data(sw_frame, decoded_frame, 0) < 0) {
                    av_frame_unref(decoded_frame);
                    return false;
                }
                src = sw_frame;
            }

            if (!sws_ctx || sws_src_format != static_cast<AVPixelFormat>(src->format) || sws_w != src->width || sws_h != src->height) {
                if (sws_ctx) {
                    sws_freeContext(sws_ctx);
                }
                sws_src_format = static_cast<AVPixelFormat>(src->format);
                sws_w = src->width;
                sws_h = src->height;
                sws_ctx = sws_getContext(
                    sws_w,
                    sws_h,
                    sws_src_format,
                    sws_w,
                    sws_h,
                    AV_PIX_FMT_BGR24,
                    SWS_BILINEAR,
                    nullptr,
                    nullptr,
                    nullptr);
                if (!sws_ctx) {
                    av_frame_unref(decoded_frame);
                    av_frame_unref(sw_frame);
                    return false;
                }
            }

            out_bgr.create(src->height, src->width, CV_8UC3);
            uint8_t *dst_data[4] = {out_bgr.data, nullptr, nullptr, nullptr};
            int dst_linesize[4] = {static_cast<int>(out_bgr.step), 0, 0, 0};

            sws_scale(
                sws_ctx,
                src->data,
                src->linesize,
                0,
                src->height,
                dst_data,
                dst_linesize);

            av_frame_unref(decoded_frame);
            av_frame_unref(sw_frame);
            ++current_frame;
            return true;
        }
    }

    // HDR read path: convert decoded HDR frames to 16-bit-per-channel RGBA
    // (AV_PIX_FMT_RGBA64LE) so the full 10/12-bit source fidelity and the
    // original PQ/HLG transfer encoding are preserved. The output `cv::Mat`
    // is CV_16UC4 laid out as R,G,B,A (16-bit unsigned, little-endian).
    //
    // Callers are expected to upload this straight to a GL_RGBA16 texture
    // and run the dedicated HDR decode shader to convert PQ/HLG -> linear
    // BT.2020 on the GPU before user shaders see it.
    bool readHdr(cv::Mat &out_rgba16) {
        if (!codec_ctx || !format_ctx) {
            return false;
        }

        while (true) {
            if (!draining) {
                const int read_ret = av_read_frame(format_ctx, packet);
                if (read_ret >= 0) {
                    if (packet->stream_index == stream_index) {
                        if (avcodec_send_packet(codec_ctx, packet) < 0) {
                            av_packet_unref(packet);
                            return false;
                        }
                    }
                    av_packet_unref(packet);
                } else {
                    if (!drain_packet_sent) {
                        if (avcodec_send_packet(codec_ctx, nullptr) < 0) {
                            return false;
                        }
                        drain_packet_sent = true;
                        draining = true;
                    }
                }
            }

            const int receive_ret = avcodec_receive_frame(codec_ctx, decoded_frame);
            if (receive_ret == AVERROR(EAGAIN)) {
                if (draining) {
                    return false;
                }
                continue;
            }
            if (receive_ret == AVERROR_EOF) {
                return false;
            }
            if (receive_ret < 0) {
                return false;
            }

            AVFrame *src = decoded_frame;
            if (hw_decode_enabled && decoded_frame->format == hw_pix_fmt) {
                if (av_hwframe_transfer_data(sw_frame, decoded_frame, 0) < 0) {
                    av_frame_unref(decoded_frame);
                    return false;
                }
                src = sw_frame;
            }

            // Fast / correct path: for 10-bit BT.2020 planar YUV formats we
            // run a hand-rolled BT.2020 NCL matrix on the CPU. This avoids
            // swscale's infamous colorspace-detection quirks (it silently
            // uses BT.601 on several internal paths even after
            // @c sws_setColorspaceDetails), which manifested as a pink tint
            // on HLG output.
            if (src->format == AV_PIX_FMT_YUV420P10LE ||
                src->format == AV_PIX_FMT_P010LE) {
                if (convertBt2020Yuv10LimitedToRgba16(src, out_rgba16)) {
                    av_frame_unref(decoded_frame);
                    av_frame_unref(sw_frame);
                    ++current_frame;
                    return true;
                }
            }

            if (!sws_ctx_hdr || sws_src_format_hdr != static_cast<AVPixelFormat>(src->format) || sws_w_hdr != src->width || sws_h_hdr != src->height) {
                if (sws_ctx_hdr) {
                    sws_freeContext(sws_ctx_hdr);
                }
                sws_src_format_hdr = static_cast<AVPixelFormat>(src->format);
                sws_w_hdr = src->width;
                sws_h_hdr = src->height;
                // Convert to RGBA64LE: 16-bit per channel, interleaved RGBA.
                // swscale does NOT perform transfer-function conversion, so
                // the output bits are still PQ/HLG-encoded BT.2020 values
                // scaled to occupy the full 16-bit range from the source's
                // 10/12-bit depth (shifted up). That is exactly what we want
                // for the GPU decode pass.
                sws_ctx_hdr = sws_getContext(
                    sws_w_hdr,
                    sws_h_hdr,
                    sws_src_format_hdr,
                    sws_w_hdr,
                    sws_h_hdr,
                    AV_PIX_FMT_RGBA64LE,
                    SWS_BILINEAR,
                    nullptr,
                    nullptr,
                    nullptr);
                if (!sws_ctx_hdr) {
                    av_frame_unref(decoded_frame);
                    av_frame_unref(sw_frame);
                    return false;
                }

                // Force swscale to use the source's real YUV->RGB matrix.
                // Without this, YUV420P10LE frames tagged BT.2020 get
                // converted with the default BT.601 coefficients and the
                // output RGB ends up with a pink/green bias. We pick the
                // matrix from the decoded frame's colorspace tag (falling
                // back to BT.2020 NCL for an HDR source) and the range
                // from @c color_range.
                int src_space = SWS_CS_DEFAULT;
                switch (src->colorspace) {
                case AVCOL_SPC_BT2020_NCL:
                case AVCOL_SPC_BT2020_CL:
                    src_space = SWS_CS_BT2020;
                    break;
                case AVCOL_SPC_BT709:
                    src_space = SWS_CS_ITU709;
                    break;
                case AVCOL_SPC_SMPTE170M:
                case AVCOL_SPC_BT470BG:
                    src_space = SWS_CS_ITU601;
                    break;
                default:
                    src_space = SWS_CS_BT2020;  // HDR default.
                    break;
                }
                const int src_range = (src->color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
                const int *src_coefs = sws_getCoefficients(src_space);
                const int *dst_coefs = sws_getCoefficients(SWS_CS_BT2020);
                sws_setColorspaceDetails(
                    sws_ctx_hdr,
                    src_coefs, src_range,
                    dst_coefs, 1 /* full-range RGB output */,
                    0 /* brightness */, 1 << 16 /* contrast 1.0 */, 1 << 16 /* saturation 1.0 */);
            }

            out_rgba16.create(src->height, src->width, CV_16UC4);
            uint8_t *dst_data[4] = {out_rgba16.data, nullptr, nullptr, nullptr};
            int dst_linesize[4] = {static_cast<int>(out_rgba16.step), 0, 0, 0};

            sws_scale(
                sws_ctx_hdr,
                src->data,
                src->linesize,
                0,
                src->height,
                dst_data,
                dst_linesize);

            av_frame_unref(decoded_frame);
            av_frame_unref(sw_frame);
            ++current_frame;
            return true;
        }
    }

    bool seekStart() {
        if (!format_ctx || !codec_ctx || stream_index < 0) {
            return false;
        }
        if (av_seek_frame(format_ctx, stream_index, 0, AVSEEK_FLAG_BACKWARD) < 0) {
            return false;
        }
        avcodec_flush_buffers(codec_ctx);
        drain_packet_sent = false;
        draining = false;
        current_frame = 0;
        return true;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    double getFps() const { return fps; }
    double getFrameCount() const { return frame_count; }
    int64_t getCurrentFrame() const { return current_frame; }
    bool isHwDecodeEnabled() const { return hw_decode_enabled; }
    bool isHdr() const { return is_hdr; }
    int getHdrPrimaries() const { return hdr_primaries; }
    int getHdrTransfer() const { return hdr_trc; }
    int getHdrColorspace() const { return hdr_colorspace; }
    int getHdrColorRange() const { return hdr_color_range; }
    int getHdrBitDepth() const { return hdr_bit_depth; }
    const std::vector<uint8_t> &getHdrMasteringDisplay() const { return hdr_mastering_display; }
    const std::vector<uint8_t> &getHdrContentLight() const { return hdr_content_light; }

  private:
    static AVPixelFormat getHwFormat(AVCodecContext *ctx, const AVPixelFormat *pix_fmts) {
        auto *self = static_cast<FFMpegVideoReader *>(ctx->opaque);
        if (!self) {
            return pix_fmts[0];
        }

        for (const AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
            if (*p == self->hw_pix_fmt) {
                return *p;
            }
        }
        return pix_fmts[0];
    }

    void enableCudaHwDecode(const AVCodec *codec) {
#ifdef ACMX2_WITH_CUDA
        for (int i = 0;; ++i) {
            const AVCodecHWConfig *cfg = avcodec_get_hw_config(codec, i);
            if (!cfg) {
                return;
            }

            if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && cfg->device_type == AV_HWDEVICE_TYPE_CUDA) {
                hw_pix_fmt = cfg->pix_fmt;
                break;
            }
        }

        if (hw_pix_fmt == AV_PIX_FMT_NONE) {
            return;
        }

        if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
            return;
        }

        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        codec_ctx->opaque = this;
        codec_ctx->get_format = &FFMpegVideoReader::getHwFormat;
        hw_decode_enabled = (codec_ctx->hw_device_ctx != nullptr);
#else
        (void)codec;
#endif
    }

    void close() {
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
        if (decoded_frame) {
            av_frame_free(&decoded_frame);
            decoded_frame = nullptr;
        }
        if (sw_frame) {
            av_frame_free(&sw_frame);
            sw_frame = nullptr;
        }
        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
        if (sws_ctx_hdr) {
            sws_freeContext(sws_ctx_hdr);
            sws_ctx_hdr = nullptr;
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
            format_ctx = nullptr;
        }
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
            hw_device_ctx = nullptr;
        }

        stream_index = -1;
        hw_pix_fmt = AV_PIX_FMT_NONE;
        hw_decode_enabled = false;
        draining = false;
        drain_packet_sent = false;
        width = 0;
        height = 0;
        fps = 0.0;
        frame_count = 0.0;
        current_frame = 0;
        sws_src_format = AV_PIX_FMT_NONE;
        sws_w = 0;
        sws_h = 0;
        sws_src_format_hdr = AV_PIX_FMT_NONE;
        sws_w_hdr = 0;
        sws_h_hdr = 0;
        is_hdr = false;
        hdr_primaries = 0;
        hdr_trc = 0;
        hdr_colorspace = 0;
        hdr_color_range = 0;
        hdr_bit_depth = 0;
        hdr_mastering_display.clear();
        hdr_content_light.clear();
    }

    AVFormatContext *format_ctx = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    AVFrame *decoded_frame = nullptr;
    AVFrame *sw_frame = nullptr;
    AVPacket *packet = nullptr;
    SwsContext *sws_ctx = nullptr;
    /// Separate SwsContext for the HDR read path (RGBA64LE output).
    /// Kept independent of @ref sws_ctx so alternating between read() and
    /// readHdr() does not repeatedly re-allocate either context.
    SwsContext *sws_ctx_hdr = nullptr;
    AVBufferRef *hw_device_ctx = nullptr;
    int stream_index = -1;
    AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
    AVPixelFormat sws_src_format = AV_PIX_FMT_NONE;
    int sws_w = 0;
    int sws_h = 0;
    AVPixelFormat sws_src_format_hdr = AV_PIX_FMT_NONE;
    int sws_w_hdr = 0;
    int sws_h_hdr = 0;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    double frame_count = 0.0;
    int64_t current_frame = 0;
    bool hw_decode_enabled = false;
    bool draining = false;
    bool drain_packet_sent = false;
    bool is_hdr = false;
    int hdr_primaries = 0;
    int hdr_trc = 0;
    int hdr_colorspace = 0;
    int hdr_color_range = 0;
    int hdr_bit_depth = 0;
    std::vector<uint8_t> hdr_mastering_display;
    std::vector<uint8_t> hdr_content_light;
};

/**
 * @class SnapshotThreadPool
 * @brief A fixed-size thread pool used for writing PNG snapshots asynchronously.
 *
 * Tasks (snapshot encode + write) are enqueued and executed by worker threads
 * so the render loop is never blocked by disk I/O.
 */
class SnapshotThreadPool {
  public:
    /**
     * @brief Construct a pool with a fixed number of worker threads.
     *
     * Each worker runs a loop that waits on the shared condition variable.
     * When a task is pushed into the queue the condition is signalled and
     * exactly one sleeping worker wakes to execute it.  Workers remain
     * alive until the destructor sets the @c stop flag and joins them.
     *
     * @param threads Number of OS threads to spawn (typically 2).
     */
    SnapshotThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        if (this->tasks.empty()) {
                            continue;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    /**
     * @brief Submit a callable for asynchronous execution.
     *
     * The callable is wrapped in a `std::function<void()>` and pushed
     * onto the shared task queue.  One waiting worker is then notified.
     * Throws `std::runtime_error` if the pool has already been stopped.
     *
     * @tparam F Any callable matching `void()` (typically a lambda
     *           that captures a FrameData by value for PNG writing).
     * @param f  The task to execute on a worker thread.
     */
    template <class F>
    void enqueue(F &&f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped SnapshotThreadPool");
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    /**
     * @brief Drain remaining tasks and join all worker threads.
     *
     * Sets the @c stop flag under the lock, then broadcasts the
     * condition variable so every sleeping worker wakes and exits.
     * Each thread is joined before the destructor returns, guaranteeing
     * that no background I/O is still in flight.
     */
    ~SnapshotThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

  private:
    std::vector<std::thread> workers;              ///< Persistent worker threads.
    std::queue<std::function<void()>> tasks;        ///< FIFO of pending PNG-write tasks.
    std::mutex queue_mutex;                         ///< Protects @c tasks and @c stop.
    std::condition_variable condition;              ///< Signalled when a task is enqueued or pool stops.
    bool stop = false;                             ///< When true, workers exit after draining the queue.
};

/**
 * @class FrameCache
 * @brief Fixed-capacity ring buffer of cv::Mat frames for temporal shaders.
 *
 * Shaders whose filename contains "cache" receive up to 8 previous frames
 * as additional sampler2D uniforms (samp1–samp8). This class stores those
 * frames in a std::deque, evicting the oldest when full.
 */
class FrameCache {
  public:
    /// @param num Maximum number of frames to retain.
    explicit FrameCache(std::size_t num)
        : num_frames(num) {
    }
    ~FrameCache() = default;

    /**
     * @brief Push a frame into the ring buffer.
     *
     * If the buffer has not yet reached capacity the frame is appended.
     * Once full, the oldest frame (front of the deque) is evicted before
     * the new frame is added at the back.  The frame is moved rather than
     * copied to avoid expensive pixel-buffer duplication.
     *
     * @param frame An rvalue reference to the cv::Mat to store.
     */
    void push(cv::Mat &&frame) {
        if (frames.size() < num_frames) {
            frames.emplace_back(std::move(frame));
        } else {
            frames.pop_front();
            frames.emplace_back(std::move(frame));
        }
    }
    /**
     * @brief Bounds-checked element access.
     * @param index Zero-based frame index (0 = oldest retained frame).
     * @return Reference to the cv::Mat at @p index.
     * @throws std::out_of_range if @p index is out of bounds.
     */
    cv::Mat &at(std::size_t index) {
        return frames.at(index);
    }
    /**
     * @brief Subscript access with an explicit out-of-range check.
     * @param index Zero-based frame index.
     * @return Reference to the cv::Mat at @p index.
     * @throws std::out_of_range if @p index >= size().
     */
    cv::Mat &operator[](std::size_t index) {
        if (index >= frames.size()) {
            throw std::out_of_range("FrameCache index out of range");
        }
        return frames[index];
    }
    /**
     * @brief Return the number of frames currently stored.
     * @return Frame count (0 ≤ n ≤ capacity).
     */
    std::size_t size() const {
        return frames.size();
    }

    /**
     * @brief Check whether the buffer has reached its maximum capacity.
     * @return True when exactly @c num_frames frames are stored.
     */
    bool isFull() {
        if (size() == num_frames)
            return true;
        return false;
    }

    /**
     * @brief Pre-fill the buffer with copies of a single frame.
     *
     * Used during initialisation to seed the cache with blank (black)
     * textures so that "cache" shaders have valid sampler data from
     * frame zero.
     *
     * @param frame The cv::Mat to replicate into every slot.
     */
    void fill(cv::Mat &frame) {
        for (size_t i = 0; i < num_frames; ++i) {
            if (frames.size() < num_frames)
                frames.push_back(frame);
        }
    }

  private:
    std::size_t num_frames;
    std::deque<cv::Mat> frames;
};

/**
 * @class TextureUploader
 * @brief Zero-copy CUDA-to-OpenGL texture transfer via Pixel Buffer Object (PBO).
 *
 * Registers an OpenGL PBO with CUDA so that a cv::cuda::GpuMat can be copied
 * directly into an OpenGL texture without passing through host memory.
 * This is the fastest path for getting GPU-filtered frames onto the screen.
 */
class TextureUploader {
  public:
    GLuint textureID = 0;           ///< OpenGL texture receiving the frame data.
    GLuint pboID = 0;               ///< PBO shared between CUDA and OpenGL.
#ifdef ACMX2_WITH_CUDA
    cudaGraphicsResource *cudaPboResource = nullptr; ///< CUDA handle to the mapped PBO.
#endif
    int width = 0;                  ///< Current texture width in pixels.
    int height = 0;                 ///< Current texture height in pixels.

    /**
     * @brief Create (or recreate) the GL texture, PBO, and CUDA registration.
     *
     * Allocates an RGBA OpenGL texture of the requested dimensions, creates a
     * matching Pixel Buffer Object sized to `w * h * 4` bytes, and registers
     * the PBO with the CUDA runtime via cudaGraphicsGLRegisterBuffer so that
     * subsequent update() calls can write GPU memory directly into the PBO
     * without a device-to-host round-trip.
     *
     * If the uploader was previously initialised, cleanup() is called first
     * so that old resources are released before new ones are created.
     *
     * @param w Texture / PBO width in pixels.
     * @param h Texture / PBO height in pixels.
     */
    void init(int w, int h) {
        if (textureID != 0)
            cleanup();
        width = w;
        height = h;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenBuffers(1, &pboID);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#ifdef ACMX2_WITH_CUDA
        CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pboID, cudaGraphicsMapFlagsWriteDiscard));
#endif
    }

    /**
     * @brief Upload a CUDA GpuMat into the OpenGL texture via the shared PBO.
     *
     * The transfer is performed entirely on the GPU:
     *  1. Map the PBO into CUDA address space (cudaGraphicsMapResources).
     *  2. Copy the GpuMat rows into the mapped pointer with cudaMemcpy2D
     *     (device-to-device, respecting the GpuMat stride).
     *  3. Unmap the resource so OpenGL can read it.
     *  4. Bind the PBO as GL_PIXEL_UNPACK_BUFFER and call glTexSubImage2D
     *     with a NULL pointer offset to DMA the data into the texture.
     *
     * If the incoming frame dimensions differ from the current texture,
     * init() is called automatically to reallocate.
     *
     * @param gpuFrame The CUDA GpuMat (CV_8UC4 / RGBA) to upload.
     */
#ifdef ACMX2_WITH_CUDA
    void update(const cv::cuda::GpuMat &gpuFrame) {
        if (gpuFrame.cols != width || gpuFrame.rows != height) {
            init(gpuFrame.cols, gpuFrame.rows);
        }
        void *pboPointer = nullptr;
        size_t numBytes = 0;
        CHECK_CUDA(cudaGraphicsMapResources(1, &cudaPboResource, 0));
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&pboPointer, &numBytes, cudaPboResource));
        CHECK_CUDA(cudaMemcpy2D(pboPointer, width * 4, gpuFrame.data, gpuFrame.step, width * 4, height, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
#endif

    /**
     * @brief Release all GPU resources (CUDA registration, PBO, texture).
     *
     * Unregisters the PBO from CUDA, deletes the OpenGL buffer, and
     * deletes the OpenGL texture.  Safe to call multiple times—each
     * resource handle is tested for non-zero before deletion and
     * reset to zero / nullptr afterwards.
     */
    void cleanup() {
#ifdef ACMX2_WITH_CUDA
        if (cudaPboResource) {
            CHECK_CUDA(cudaGraphicsUnregisterResource(cudaPboResource));
            cudaPboResource = nullptr;
        }
#endif
        if (pboID) {
            glDeleteBuffers(1, &pboID);
            pboID = 0;
        }
        if (textureID) {
            glDeleteTextures(1, &textureID);
            textureID = 0;
        }
    }
};

/**
 * @struct ShaderCacheEntry
 * @brief One shader's precompiled binary data for the on-disk cache.
 *
 * Stores the GL program binary for both 2D and 3D vertex shaders, plus an
 * FNV-1a hash of the source file so stale entries are detected on load.
 */
struct ShaderCacheEntry {
    std::string shader_name;       ///< Stem of the .glsl filename.
    std::vector<char> binary_2d;   ///< GL program binary (2D vertex shader).
    GLenum format_2d = 0;          ///< GL binary format token for 2D.
    std::vector<char> binary_3d;   ///< GL program binary (3D vertex shader).
    GLenum format_3d = 0;          ///< GL binary format token for 3D.
    uint64_t source_hash = 0;      ///< FNV-1a-64 hash of the fragment source.
    bool failed = false;           ///< True if this shader failed to compile;
                                   ///< a passthrough program is substituted at load time
                                   ///< to preserve the user-visible shader index.
};

/**
 * @struct ShaderCache
 * @brief Container for the full on-disk shader cache file.
 *
 * The binary file format is:
 * `[MAGIC][VERSION][gl_renderer][gl_version][dual_mode][count][entries...]`
 *
 * If the GL renderer or driver version changes, the cache is invalidated.
 */
struct ShaderCache {
    static constexpr uint32_t CACHE_MAGIC = 0x53484452;   ///< File magic: "SHDR".
    static constexpr uint32_t CACHE_VERSION = 3;           ///< Current format version.
    std::string gl_renderer;
    std::string gl_version;
    bool dual_mode = false;
    std::vector<ShaderCacheEntry> entries;

    /**
     * @brief Serialise the entire shader cache to a binary file.
     *
     * File layout (all values little-endian, no padding):
     * | Offset | Field                |
     * |--------|----------------------|
     * | 0      | uint32 CACHE_MAGIC   |
     * | 4      | uint32 CACHE_VERSION |
     * | 8      | string gl_renderer   |
     * |        | string gl_version    |
     * |        | bool   dual_mode     |
     * |        | uint32 entry_count   |
     * |        | entries…             |
     *
     * Each string is written as a `uint32` length followed by raw bytes.
     * Each entry contains: shader_name (string), source_hash (uint64),
     * format_2d (GLenum), binary_2d (uint32 size + raw bytes),
     * format_3d (GLenum), binary_3d (uint32 size + raw bytes).
     *
     * @param path Filesystem path for the output file (e.g. `library/.shader_cache`).
     * @return True if the file was written without stream errors.
     */
    bool save(const std::string &path) const {
        std::error_code ec;
        std::filesystem::path parent = std::filesystem::path(path).parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent, ec)) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                mx::system_err << "acmx2: Could not create cache directory '"
                               << parent.string() << "': " << ec.message() << "\n";
                return false;
            }
        }
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            mx::system_err << "acmx2: Could not open cache file for writing: '"
                           << path << "' (errno=" << errno << " - "
                           << std::strerror(errno) << ")\n";
            return false;
        }

        file.write(reinterpret_cast<const char *>(&CACHE_MAGIC), sizeof(CACHE_MAGIC));
        file.write(reinterpret_cast<const char *>(&CACHE_VERSION), sizeof(CACHE_VERSION));

        auto writeString = [&file](const std::string &s) {
            uint32_t len = static_cast<uint32_t>(s.size());
            file.write(reinterpret_cast<const char *>(&len), sizeof(len));
            file.write(s.data(), len);
        };

        writeString(gl_renderer);
        writeString(gl_version);
        file.write(reinterpret_cast<const char *>(&dual_mode), sizeof(dual_mode));

        uint32_t count = static_cast<uint32_t>(entries.size());
        file.write(reinterpret_cast<const char *>(&count), sizeof(count));

        for (const auto &e : entries) {
            writeString(e.shader_name);
            uint8_t failed_flag = e.failed ? 1 : 0;
            file.write(reinterpret_cast<const char *>(&failed_flag), sizeof(failed_flag));
            file.write(reinterpret_cast<const char *>(&e.source_hash), sizeof(e.source_hash));
            file.write(reinterpret_cast<const char *>(&e.format_2d), sizeof(e.format_2d));
            uint32_t size_2d = static_cast<uint32_t>(e.binary_2d.size());
            file.write(reinterpret_cast<const char *>(&size_2d), sizeof(size_2d));
            file.write(e.binary_2d.data(), size_2d);

            file.write(reinterpret_cast<const char *>(&e.format_3d), sizeof(e.format_3d));
            uint32_t size_3d = static_cast<uint32_t>(e.binary_3d.size());
            file.write(reinterpret_cast<const char *>(&size_3d), sizeof(size_3d));
            file.write(e.binary_3d.data(), size_3d);
        }
        return file.good();
    }

    /**
     * @brief Deserialise the shader cache from a binary file.
     *
     * Validates the magic number and version before reading.  If either
     * does not match (e.g. cache was written by an older version or a
     * different application) the method returns false and leaves the
     * object in an indeterminate state—callers should discard it.
     *
     * The layout mirrors save(): magic, version, renderer string,
     * GL version string, dual_mode flag, entry count, then each entry.
     *
     * @param path Filesystem path of the cache file to read.
     * @return True on success; false if the file is missing, corrupt,
     *         or has a version / magic mismatch.
     */
    bool load(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            return false;

        uint32_t magic, version;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (magic != CACHE_MAGIC || version != CACHE_VERSION)
            return false;

        auto readString = [&file]() -> std::string {
            uint32_t len;
            file.read(reinterpret_cast<char *>(&len), sizeof(len));
            std::string s(len, '\0');
            file.read(s.data(), len);
            return s;
        };

        gl_renderer = readString();
        gl_version = readString();
        file.read(reinterpret_cast<char *>(&dual_mode), sizeof(dual_mode));

        uint32_t count;
        file.read(reinterpret_cast<char *>(&count), sizeof(count));
        entries.resize(count);

        for (auto &e : entries) {
            e.shader_name = readString();
            uint8_t failed_flag = 0;
            file.read(reinterpret_cast<char *>(&failed_flag), sizeof(failed_flag));
            e.failed = (failed_flag != 0);
            file.read(reinterpret_cast<char *>(&e.source_hash), sizeof(e.source_hash));
            file.read(reinterpret_cast<char *>(&e.format_2d), sizeof(e.format_2d));
            uint32_t size_2d;
            file.read(reinterpret_cast<char *>(&size_2d), sizeof(size_2d));
            e.binary_2d.resize(size_2d);
            file.read(e.binary_2d.data(), size_2d);

            file.read(reinterpret_cast<char *>(&e.format_3d), sizeof(e.format_3d));
            uint32_t size_3d;
            file.read(reinterpret_cast<char *>(&size_3d), sizeof(size_3d));
            e.binary_3d.resize(size_3d);
            file.read(e.binary_3d.data(), size_3d);
        }
        return file.good();
    }
};

/**
 * @brief Compute an FNV-1a 64-bit hash of a file's contents.
 * @param filepath Path to the file to hash.
 * @return 64-bit hash, or 0 if the file cannot be opened.
 */
static uint64_t fnv1a64_file(const std::string &filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open())
        return 0;

    uint64_t h = 1469598103934665603ull;
    char buf[1 << 15];
    while (f.good()) {
        f.read(buf, sizeof(buf));
        std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= (uint8_t)buf[i];
            h *= 1099511628211ull;
        }
    }
    return h;
}

/// @brief Convenience wrapper—hashes a shader source file for cache validation.
uint64_t hashFileContents(const std::string &filepath) {
    return fnv1a64_file(filepath);
}

typedef void(APIENTRYP PFNGLGETPROGRAMBINARYPROC_LOCAL)(GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary);
typedef void(APIENTRYP PFNGLPROGRAMBINARYPROC_LOCAL)(GLuint program, GLenum binaryFormat, const void *binary, GLsizei length);

static PFNGLGETPROGRAMBINARYPROC_LOCAL glGetProgramBinaryFunc = nullptr;
static PFNGLPROGRAMBINARYPROC_LOCAL glProgramBinaryFunc = nullptr;

/// @brief Dynamically load glGetProgramBinary / glProgramBinary via SDL.
bool loadProgramBinaryFunctions() {
    if (glGetProgramBinaryFunc != nullptr)
        return true;
    glGetProgramBinaryFunc = reinterpret_cast<PFNGLGETPROGRAMBINARYPROC_LOCAL>(SDL_GL_GetProcAddress("glGetProgramBinary"));
    glProgramBinaryFunc = reinterpret_cast<PFNGLPROGRAMBINARYPROC_LOCAL>(SDL_GL_GetProcAddress("glProgramBinary"));

    if (glGetProgramBinaryFunc == nullptr || glProgramBinaryFunc == nullptr) {
        mx::system_err << "acmx2: Failed to load glGetProgramBinary/glProgramBinary functions\n";
        return false;
    }
    return true;
}

#ifdef AUDIO_ENABLED
/**
 * @class SpectrumTexture
 * @brief Manages a 1D OpenGL texture that holds the FFT frequency-magnitude spectrum.
 *
 * ### What this class does
 * Every audio frame, the RtAudio callback captures raw PCM samples into a
 * double buffer (see `push_audio_buffer()`).  On the **render** thread,
 * `update()` calls `compute_audio_fft()` to run a radix-2 FFT and then
 * uploads the resulting magnitude array into a **GL_TEXTURE_1D** so that
 * any GLSL shader can sample it.
 *
 * ### How the 1D texture works
 * A 1D texture is like a single-row image.  Each texel (pixel) stores one
 * float — the energy at that frequency bin.  The texture is `FFT_SIZE/2`
 * texels wide (256 by default) because a real-valued FFT is symmetric and
 * only the first half carries unique information.
 *
 * The internal format is **GL_R32F** (one 32-bit float per texel, red
 * channel only).  In GLSL you read it with:
 * @code{.glsl}
 *   uniform sampler1D spectrum;           // bound to texture unit 9
 *   float energy = texture(spectrum, x).r; // x in [0,1]
 * @endcode
 * where `x = 0.0` is the DC bin and `x = 1.0` is the Nyquist frequency.
 *
 * ### Texture parameters
 * - **GL_LINEAR** filtering — the GPU interpolates between adjacent bins,
 *   giving smooth results even when the shader samples at non-integer
 *   frequency positions.
 * - **GL_CLAMP_TO_EDGE** wrapping — lookups outside [0,1] clamp to the
 *   nearest edge bin instead of wrapping or returning black.
 *
 * ### Texture unit
 * The spectrum is bound to **GL_TEXTURE9** (unit 9).  Units 0 is the main
 * video frame, units 1–8 are the temporal frame cache, so 9 is the first
 * free slot.
 *
 * ### RAII
 * `init()` creates the texture; `cleanup()` deletes it.  The destructor
 * calls `cleanup()` automatically, so you never leak GPU resources.
 *
 * @see push_audio_buffer(), compute_audio_fft(), get_fft_magnitudes()
 */
class SpectrumTexture {
public:
    /**
     * @brief Create the 1D texture and set its sampling parameters.
     *
     * Allocates a `GL_TEXTURE_1D` of width `FFT_SIZE / 2` (256 texels)
     * with the `GL_R32F` internal format (one float per texel).  The
     * initial texel data is nullptr — the texture is filled on the first
     * `update()` call.
     *
     * The two filter parameters (`GL_LINEAR`) tell the GPU to linearly
     * interpolate when a shader samples *between* two bins, which avoids
     * visible staircase artefacts.  `GL_CLAMP_TO_EDGE` prevents wrap-
     * around artefacts if a shader accidentally samples outside [0,1].
     */
    void init() {
        if (textureID != 0) return;
        bins = FFT_SIZE / 2;

        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_1D, textureID);

        // Allocate the texture storage — one 32-bit float per texel.
        // GL_R32F  = internal format (single-channel 32-bit float)
        // GL_RED   = source channel layout
        // GL_FLOAT = source data type
        glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, bins, 0, GL_RED, GL_FLOAT, nullptr);

        // GL_LINEAR makes the GPU interpolate between adjacent frequency
        // bins when the shader samples at a fractional position.
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Clamp so lookups outside [0,1] stick to the edge bin.
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_1D, 0);
    }

    /**
     * @brief Compute a fresh FFT and upload the magnitudes into the texture.
     *
     * This is the per-frame call that bridges the audio and GPU worlds:
     * 1. `compute_audio_fft()` reads the latest PCM snapshot, applies a
     *    Hann window, runs the radix-2 FFT, and writes the magnitudes.
     * 2. `glTexSubImage1D()` copies those magnitudes into the existing
     *    texture **without** reallocating — much cheaper than `glTexImage1D`
     *    every frame.
     *
     * Call this once per frame from the render thread, before binding the
     * texture for shader use.
     */
    void update() {
        if (textureID == 0) return;
        compute_audio_fft();
        const auto &mags = get_fft_magnitudes();
        glBindTexture(GL_TEXTURE_1D, textureID);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, bins, GL_RED, GL_FLOAT, mags.data());
        glBindTexture(GL_TEXTURE_1D, 0);
    }

    void update(float scale) {
        if (textureID == 0) return;
        compute_audio_fft();
        const auto &mags = get_fft_magnitudes();
        scaled_buf.resize(mags.size());
        for (size_t i = 0; i < mags.size(); ++i)
            scaled_buf[i] = mags[i] * scale;
        glBindTexture(GL_TEXTURE_1D, textureID);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, bins, GL_RED, GL_FLOAT, scaled_buf.data());
        glBindTexture(GL_TEXTURE_1D, 0);
    }

    /**
     * @brief Bind the spectrum texture to a specific texture unit.
     *
     * The caller passes a unit index (e.g. 9) and the corresponding
     * `GL_TEXTURE9` unit is activated.  After this call the shader
     * uniform `spectrum` should be set to the same unit index via
     * `glUniform1i(loc, unit)`.
     *
     * @param unit  Texture unit index (0-based).  Default is 9.
     */
    void bind(int unit = SPECTRUM_TEXTURE_UNIT) const {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_1D, textureID);
    }

    /**
     * @brief Delete the OpenGL texture and reset state.
     *
     * Safe to call more than once — the texture ID is checked before
     * deletion and zeroed afterwards.
     */
    void cleanup() {
        if (textureID) {
            glDeleteTextures(1, &textureID);
            textureID = 0;
        }
    }

    ~SpectrumTexture() { cleanup(); }

    /// Texture unit reserved for the spectrum (units 0–8 are taken).
    static constexpr int SPECTRUM_TEXTURE_UNIT = 9;

private:
    GLuint textureID = 0;  ///< OpenGL name for the 1D texture.
    int bins = 0;          ///< Number of texels (== FFT_SIZE / 2).
    std::vector<float> scaled_buf; ///< Scratch buffer for sensitivity-scaled magnitudes.
};
#endif // AUDIO_ENABLED

/**
 * @class ShaderLibrary
 * @brief Manages the complete collection of compiled GLSL shader programs.
 *
 * Responsibilities:
 * - Load a single shader or a full library from an index.txt manifest.
 * - Optionally build and restore a binary shader cache for fast startup.
 * - Track and upload all shader uniforms each frame (time, mouse, audio, etc.).
 * - Support dual-mode (2D + 3D vertex shader) compilation.
 * - Provide navigation (inc/dec/setIndex) and bypass controls.
 *
 * acidcamGL-compatible uniforms (value_alpha_r/g/b, optx, random_var, etc.)
 * are maintained here so legacy shaders work unmodified.
 */
class ShaderLibrary {
    float alpha = 0.1f;
    bool time_active = true;
    float time_f = 1.0;
    float time_speed = 1.0f;
    double video_fps = 0.0;
#ifdef MIDI_ENABLED
    float midi_slider[4] = {0.0f, 0.0f, 0.0f, 0.0f}; ///< MIDI CC slider values (0.0–1.0) for shader uniforms slider1–slider4.
#endif
    bool is3d = false;
    bool dual_mode = false;

    // acidcamGL-compatible uniform state
    float color_alpha_r = 0.1f;
    float color_alpha_g = 0.2f;
    float color_alpha_b = 0.3f;
    bool alpha_dir = true;
    glm::vec4 optx = glm::vec4(0.5f, 0.5f, 0.5f, 0.5f);
    glm::vec4 random_var = glm::vec4(0.0f);
    glm::vec4 inc_value = glm::vec4(0.0f);
    glm::vec4 inc_valuex = glm::vec4(0.0f);
    bool restore_black = false;

    /**
     * @struct ProgramData
     * @brief Cached uniform locations for a single compiled shader program.
     *
     * Querying glGetUniformLocation every frame is expensive; this struct
     * stores all locations once at compile time.
     */
    struct ProgramData {
        std::string name;
        GLint loc = -1, iTime = -1, iMouse = -1, time_f = -1, iResolution = -1;
#ifdef AUDIO_ENABLED
        GLint amp = -1, amp_untouched = -1;
        GLint iamp = -1;
        GLint amp_peak = -1, amp_rms = -1, amp_smooth = -1;
        GLint amp_low = -1, amp_mid = -1, amp_high = -1;
        GLint spectrum_loc = -1; ///< Location of `uniform sampler1D spectrum;` (-1 if unused).
#endif
        GLint texture_cache_loc[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        GLint iFrame = -1;
        GLint iTimeDelta = -1;
        GLint iDate = -1;
        GLint iChannelTime[4] = {-1, -1, -1, -1};
        GLint iChannelResolution[4] = {-1, -1, -1, -1};
        GLint iSampleRate = -1;
        GLint iFrameRate = -1;
        GLint iMouseClick = -1;
        // acidcamGL-compatible uniform locations
        GLint value_alpha_r = -1, value_alpha_g = -1, value_alpha_b = -1;
        GLint alpha_r_loc = -1, alpha_g_loc = -1, alpha_b_loc = -1;
        GLint alpha_value = -1;
        GLint index_value = -1;
        GLint optx_loc = -1;
        GLint random_var_loc = -1;
        GLint restore_black_loc = -1;
        GLint inc_value_loc = -1;
        GLint inc_valuex_loc = -1;
        GLint time_speed_loc = -1;
#ifdef MIDI_ENABLED
        GLint slider_loc[4] = {-1, -1, -1, -1}; ///< Locations of optional `uniform float slider1..slider4;`
#endif
    };
    size_t library_index = 0;
    bool use_cache = false;
    bool rebuild_attempted = false; ///< Set after the first stale-cache rebuild to prevent repeated re-builds in the same session.

    /**
     * @brief Resolve the on-disk path for the shader binary cache file.
     *
     * The cache is preferentially stored under the assets directory (`--path`)
     * so users can point multiple read-only shader libraries at a single
     * writable cache location. If the assets directory is not writable we
     * fall back to placing the cache directly inside the shader library
     * directory (legacy behaviour).
     *
     * The filename is derived from a hash of the absolute, normalised
     * library path so different libraries do not collide.
     *
     * @param assets_path  Assets directory passed via `--path` (may be empty).
     * @param library_path Shader library directory (containing index.txt).
     * @return Absolute or relative path to the shader cache file.
     */
    static std::string shaderCacheFilePath(const std::string &assets_path,
                                           const std::string &library_path) {
        std::error_code ec;
        std::filesystem::path lib(library_path);
        std::filesystem::path abs_lib = std::filesystem::absolute(lib, ec);
        std::string key = ec ? library_path : abs_lib.lexically_normal().string();
        std::hash<std::string> hasher;
        std::ostringstream name_stream;
        name_stream << ".shader_cache_" << std::hex << hasher(key);
        const std::string filename = name_stream.str();

        auto isWritable = [](const std::filesystem::path &dir) {
            std::error_code e;
            if (dir.empty())
                return false;
            if (!std::filesystem::exists(dir, e)) {
                std::filesystem::create_directories(dir, e);
                if (e)
                    return false;
            }
            // Probe writability by creating a temp file.
            std::filesystem::path probe = dir / ".acmx2_write_probe";
            std::ofstream f(probe, std::ios::binary);
            bool ok = f.is_open();
            f.close();
            if (ok)
                std::filesystem::remove(probe, e);
            return ok;
        };

        std::filesystem::path assets(assets_path);
        std::filesystem::path libdir(library_path);

        // Prefer existing cache in assets; otherwise existing cache in lib dir.
        std::filesystem::path assets_cache = assets.empty() ? std::filesystem::path() : assets / filename;
        std::filesystem::path lib_cache = libdir / filename;
        if (!assets.empty() && std::filesystem::exists(assets_cache, ec)) {
            return assets_cache.string();
        }
        if (std::filesystem::exists(lib_cache, ec)) {
            return lib_cache.string();
        }
        // No existing cache; pick a writable location, preferring assets.
        if (!assets.empty() && isWritable(assets)) {
            return assets_cache.string();
        }
        if (isWritable(libdir)) {
            return lib_cache.string();
        }
        // Last resort: return assets path even if it isn't writable so the
        // caller can surface a meaningful error.
        return assets.empty() ? lib_cache.string() : assets_cache.string();
    }

    std::vector<std::unique_ptr<gl::ShaderProgram>> programs_2d;
    std::vector<std::unique_ptr<gl::ShaderProgram>> programs_3d;
    bool time_audio = false;
    bool audio_delta = false;
    std::unordered_map<int, ProgramData> program_names_2d;
    std::unordered_map<int, ProgramData> program_names_3d;
    bool shader_bypass = false;
    bool isDragging = false;
    bool wasClicked = false;
    float clickStartX = 0.0f;
    float clickStartY = 0.0f;
    float lastClickX = 0.0f;
    float lastClickY = 0.0f;

    std::unique_ptr<gl::ShaderProgram> makeProgram() {
        if (use_cache)
            return std::make_unique<ac::ShaderProgram>();
        return std::make_unique<gl::ShaderProgram>();
    }

    /**
     * @brief Compile a minimal passthrough fragment shader as a stand-in.
     *
     * Used when a shader in the library fails to compile (either during
     * cache-build or source-compile).  A placeholder program is inserted
     * at the failing shader's index so that numeric indices remain
     * aligned with the on-disk index.txt listing — this keeps
     * user-selected indices (e.g. from the Qt interface or CLI
     * --shader-index) pointing at the intended slot even when one or
     * more shaders in the library are broken.
     *
     * The fragment shader samples the input frame texture `samp` and
     * writes it unchanged, so the placeholder renders as a no-op
     * passthrough instead of causing a hard error.
     *
     * @param vert_path Path to the vertex shader to pair with the passthrough.
     * @return A compiled ShaderProgram, or an empty unique_ptr on failure.
     */
    std::unique_ptr<gl::ShaderProgram> makePassthroughProgram(const std::string &vert_path) {
        static constexpr const char *kPassthroughFrag =
            "#version 330 core\n"
            "in vec2 tc;\n"
            "out vec4 color;\n"
            "uniform sampler2D samp;\n"
            "void main() {\n"
            "    color = texture(samp, tc);\n"
            "}\n";

        std::ifstream vf(vert_path);
        if (!vf.is_open())
            return {};
        std::stringstream vss;
        vss << vf.rdbuf();
        std::string vert_source = vss.str();

        auto prog = makeProgram();
        prog->setSilent(true);
        if (!prog->loadProgramFromText(vert_source, kPassthroughFrag)) {
            return {};
        }
        return prog;
    }

  public:
    ShaderLibrary() = default;
    ~ShaderLibrary() {}

#ifdef MIDI_ENABLED
    /// Set a MIDI slider value (index 0–3, value 0.0–1.0).
    void setMidiSlider(int idx, float val) { if (idx >= 0 && idx < 4) midi_slider[idx] = val; }
#endif

    /**
     * @brief Enable or disable use of the ac::ShaderProgram binary-cache wrapper.
     *
     * When enabled, makeProgram() returns an ac::ShaderProgram instead of
     * the base gl::ShaderProgram, allowing shader binaries to be loaded
     * from the on-disk cache rather than recompiled from source.
     *
     * @param enable True to use the caching wrapper.
     */
    void enableCache(bool enable) { use_cache = enable; }

    /**
     * @brief Remove all compiled shader programs and reset the library index.
     *
     * Releases every unique_ptr in both the 2D and 3D program vectors,
     * clears the associated uniform-location maps, and resets the
     * current library_index to zero.  Called before rebuilding the
     * shader cache when a count or hash mismatch is detected.
     */
    void clear() {
        programs_2d.clear();
        programs_3d.clear();
        program_names_2d.clear();
        program_names_3d.clear();
        library_index = 0;
    }

    /**
     * @brief Compile a single fragment shader with the 2D (and optionally 3D) vertex shader.
     *
     * Loads `data/vert.glsl` (2D) and optionally `data/vertex.glsl` (3D)
     * paired with the supplied fragment shader path.  After compilation
     * the shader's uniform locations are cached via setupProgramUniforms().
     *
     * @param win  Pointer to the GL window (provides asset path resolution).
     * @param text Full path to the fragment shader source file.
     * @throws mx::Exception if either compile/link stage fails.
     */
    void loadProgram(gl::GLWindow *win, const std::string text) {
        programs_2d.push_back(makeProgram());
        if (!programs_2d.back()->loadProgram(win->util.getFilePath("data/vert.glsl"), text)) {
            throw mx::Exception("Error loading 2D shader program: " + text);
        }
        setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, text);
        if (dual_mode) {
            programs_3d.push_back(makeProgram());
            if (!programs_3d.back()->loadProgram(win->util.getFilePath("data/vertex.glsl"), text)) {
                throw mx::Exception("Error loading 3D shader program: " + text);
            }
            setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, text);
            mx::system_out << "acmx2: Compiled Shader 0 (2D+3D): " << text << " ✔ \n";
        } else {
            mx::system_out << "acmx2: Compiled Shader 0 (2D): " << text << " ✔ \n";
        }
    }

    /**
     * @brief Query and store all uniform locations for a compiled shader program.
     *
     * Called immediately after a program is compiled (or restored from
     * cache).  Activates the program with `useProgram()`, then queries
     * locations for every known uniform—Shadertoy-compatible (iTime,
     * iResolution, iMouse, iFrame, iDate, iTimeDelta, iChannelTime,
     * iChannelResolution, iSampleRate, iFrameRate), audio reactivity
     * (amp, uamp, iamp, amp_peak, amp_rms, amp_smooth, amp_low/mid/high),
     * and acidcamGL legacy (value_alpha_r/g/b, alpha_r/g/b, alpha_value,
     * index_value, optx, random_var, restore_black, inc_value, inc_valuex).
     *
     * Results are stored in the ProgramData struct keyed by program
     * index in the provided map, avoiding per-frame glGetUniformLocation
     * calls.
     *
     * @param win   GL window (used for error reporting).
     * @param prog  The compiled shader program to query.
     * @param names Map to store the resulting ProgramData into.
     * @param pos   Index key under which the data is stored.
     * @param text  Fragment shader file path (stem becomes the display name).
     * @throws mx::Exception on any GL error after useProgram or setUniform.
     */
    void setupProgramUniforms(gl::GLWindow *win, gl::ShaderProgram *prog,
                              std::unordered_map<int, ProgramData> &names, size_t pos,
                              const std::string &text) {
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw mx::Exception("OpenGL Error: on ShaderLibary::loadProgram: " + std::to_string(error));
        }
        prog->useProgram();
        GLint loc = glGetUniformLocation(prog->id(), "iResolution");
        glUniform2f(loc, win->w, win->h);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw mx::Exception("setUniform");
        }

        std::filesystem::path file_path(text);
        std::string name = file_path.stem().string();
        if (!name.empty()) {
            names[pos].name = name;
            names[pos].loc = glGetUniformLocation(prog->id(), "alpha");
            names[pos].iTime = glGetUniformLocation(prog->id(), "iTime");
            names[pos].iMouse = glGetUniformLocation(prog->id(), "iMouse");
            names[pos].time_f = glGetUniformLocation(prog->id(), "time_f");
            names[pos].iResolution = glGetUniformLocation(prog->id(), "iResolution");
            names[pos].iFrame = glGetUniformLocation(prog->id(), "iFrame");
            names[pos].iTimeDelta = glGetUniformLocation(prog->id(), "iTimeDelta");
            names[pos].iDate = glGetUniformLocation(prog->id(), "iDate");
            names[pos].iFrameRate = glGetUniformLocation(prog->id(), "iFrameRate");
            names[pos].iMouseClick = glGetUniformLocation(prog->id(), "iMouseClick");

            for (int i = 0; i < 4; ++i) {
                std::string channelTime = "iChannelTime[" + std::to_string(i) + "]";
                std::string channelRes = "iChannelResolution[" + std::to_string(i) + "]";
                names[pos].iChannelTime[i] = glGetUniformLocation(prog->id(), channelTime.c_str());
                names[pos].iChannelResolution[i] = glGetUniformLocation(prog->id(), channelRes.c_str());
            }

            if (name.find("cache") != std::string::npos) {
                for (int i = 0; i < 8; ++i) {
                    names[pos].texture_cache_loc[i] = glGetUniformLocation(prog->id(), std::string("samp" + std::to_string(i + 1)).c_str());
                }
            }

#ifdef AUDIO_ENABLED
            names[pos].amp = glGetUniformLocation(prog->id(), "amp");
            names[pos].amp_untouched = glGetUniformLocation(prog->id(), "uamp");
            names[pos].iamp = glGetUniformLocation(prog->id(), "iamp");
            names[pos].amp_peak = glGetUniformLocation(prog->id(), "amp_peak");
            names[pos].amp_rms = glGetUniformLocation(prog->id(), "amp_rms");
            names[pos].amp_smooth = glGetUniformLocation(prog->id(), "amp_smooth");
            names[pos].amp_low = glGetUniformLocation(prog->id(), "amp_low");
            names[pos].amp_mid = glGetUniformLocation(prog->id(), "amp_mid");
            names[pos].amp_high = glGetUniformLocation(prog->id(), "amp_high");
            names[pos].iSampleRate = glGetUniformLocation(prog->id(), "iSampleRate");
            names[pos].spectrum_loc = glGetUniformLocation(prog->id(), "spectrum");
#endif
            // acidcamGL-compatible uniform locations
            names[pos].value_alpha_r = glGetUniformLocation(prog->id(), "value_alpha_r");
            names[pos].value_alpha_g = glGetUniformLocation(prog->id(), "value_alpha_g");
            names[pos].value_alpha_b = glGetUniformLocation(prog->id(), "value_alpha_b");
            names[pos].alpha_r_loc = glGetUniformLocation(prog->id(), "alpha_r");
            names[pos].alpha_g_loc = glGetUniformLocation(prog->id(), "alpha_g");
            names[pos].alpha_b_loc = glGetUniformLocation(prog->id(), "alpha_b");
            names[pos].alpha_value = glGetUniformLocation(prog->id(), "alpha_value");
            names[pos].index_value = glGetUniformLocation(prog->id(), "index_value");
            names[pos].optx_loc = glGetUniformLocation(prog->id(), "optx");
            names[pos].random_var_loc = glGetUniformLocation(prog->id(), "random_var");
            names[pos].restore_black_loc = glGetUniformLocation(prog->id(), "restore_black");
            names[pos].inc_value_loc = glGetUniformLocation(prog->id(), "inc_value");
            names[pos].inc_valuex_loc = glGetUniformLocation(prog->id(), "inc_valuex");
            names[pos].time_speed_loc = glGetUniformLocation(prog->id(), "time_speed");
#ifdef MIDI_ENABLED
            names[pos].slider_loc[0] = glGetUniformLocation(prog->id(), "slider1");
            names[pos].slider_loc[1] = glGetUniformLocation(prog->id(), "slider2");
            names[pos].slider_loc[2] = glGetUniformLocation(prog->id(), "slider3");
            names[pos].slider_loc[3] = glGetUniformLocation(prog->id(), "slider4");
#endif
        }
    }

    /**
     * @brief Upload the current frame-rate to the active shader's iFrameRate uniform.
     * @param fps_value Frames per second to send to the GPU.
     */
    void setFPS(float fps_value) {
        auto &names = is3d ? program_names_3d : program_names_2d;
        auto it = names.find(index());
        if (it == names.end())
            return;
        GLint loc = it->second.iFrameRate;
        if (loc != -1)
            glUniform1f(loc, fps_value);
    }

    /**
     * @brief Bind a cache sampler slot for texture-cache shaders.
     *
     * Sets the sampler uniform `samp1`–`samp8` at the given slot to point
     * at texture unit `value + 1`.  Only meaningful for shaders whose
     * filename contains "cache".
     *
     * @param name  Uniform name (unused — the slot index is used directly).
     * @param value Zero-based cache texture slot (0–7).
     */
    void setUniform(const std::string &name, int value) {
        if (value < 0 || value >= 8) {
            return;
        }
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (names.find(index()) == names.end()) {
            return;
        }
        glUniform1i(names[index()].texture_cache_loc[value], value + 1);
    }

    /**
     * @brief Switch between the 2D and 3D program vectors.
     * @param is3d True selects the 3D shader set; false selects 2D.
     */
    void is3D(bool is3d) {
        this->is3d = is3d;
    }

    /// @brief Set the absolute time_f advancement speed multiplier.
    void setTimeSpeed(float speed) {
        time_speed = speed;
    }

    /// @brief Set the video FPS for constant time_f advancement in video mode.
    void setVideoFPS(double fps) {
        video_fps = fps;
    }

    /**
     * @brief Increase the time_f speed multiplier by @p step.
     *
     * Holding Page Up in the render loop calls this every frame,
     * giving the user continuous acceleration control.
     *
     * @param step Amount to add to the speed (e.g. 0.1).
     */
    void incTimeSpeed(float step) {
        time_speed += step;
        mx::system_out << "acmx2: Time speed: " << time_speed << "\n";
        fflush(stdout);
    }

    /**
     * @brief Decrease the time_f speed multiplier by @p step, clamped to zero.
     * @param step Amount to subtract (e.g. 0.1).
     */
    void decTimeSpeed(float step) {
        if (time_speed - step > 0.0f) {
            time_speed -= step;
        } else {
            time_speed = 0.0f;
        }
        mx::system_out << "acmx2: Time speed: " << time_speed << "\n";
        fflush(stdout);
    }

    /**
     * @brief Enable or disable dual-mode compilation (2D + 3D vertex shaders).
     *
     * When dual mode is on, every shader source is compiled twice: once with
     * `data/vert.glsl` (flat quad) and once with `data/vertex.glsl` (3D
     * model).  This doubles compilation time but allows runtime switching
     * between 2D and 3D rendering with a single key press.
     *
     * @param enable True to compile both shader variants.
     */
    void enableDualMode(bool enable) {
        dual_mode = enable;
    }

    /// @brief Return whether dual mode (2D + 3D) is active.
    bool isDualMode() const {
        return dual_mode;
    }

    /**
     * @brief Toggle between 2D and 3D rendering modes.
     *
     * Only effective when dual_mode is enabled; otherwise prints a
     * diagnostic and returns.  Flips the internal is3d flag so that
     * subsequent calls to shader(), useProgram(), size(), etc. use the
     * alternate program vector.
     */
    void toggle3D() {
        if (!dual_mode) {
            mx::system_out << "acmx2: Cannot switch to 3D - dual mode not enabled\n";
            fflush(stdout);
            return;
        }
        is3d = !is3d;
        mx::system_out << "acmx2: Switched to " << (is3d ? "3D" : "2D") << " mode\n";
        fflush(stdout);
    }

    /// @brief Return true if currently in 3D mode.
    bool get3D() const { return is3d; }

    /**
     * @brief Toggle the shader bypass flag.
     *
     * When bypassed, ACView::draw() uses the plain framebuffer pass-through
     * shader instead of the current library shader, showing the raw
     * camera/video feed.  Press Space at runtime to toggle.
     */
    void toggleBypass() {
        shader_bypass = !shader_bypass;
        std::string state = shader_bypass ? "disabled" : "enabled";
        mx::system_out << "acmx2: Shader processing " << state << "\n";
        fflush(stdout);
    }

    /// @brief Return true if shader processing is currently bypassed.
    bool isBypassed() const {
        return shader_bypass;
    }

    /// @brief Compile every shader listed in index.txt, showing a progress overlay.
    void loadPrograms(gl::GLWindow *win, const std::string &text, mx::Font &loadingFont) {
        std::fstream file;
        file.open(text + "/index.txt", std::ios::in);
        if (!file.is_open()) {
            throw mx::Exception("acmx2: Could not load index.txt at shader path: " + text);
        }

        // Collect all shader files first
        std::vector<std::string> shader_files;
        {
            std::string line;
            while (std::getline(file, line)) {
                auto shader_entry = normalizeShaderIndexEntry(line);
                if (!shader_entry) {
                    continue;
                }
                std::string full_path;
                if (resolveShaderPathInLibrary(text, *shader_entry, full_path)) {
                    shader_files.push_back(*shader_entry);
                }
            }
            file.close();
        }

        // Case-insensitive sort to match Qt interface behavior
        std::sort(shader_files.begin(), shader_files.end(),
                  [](const std::string &a, const std::string &b) {
                      return std::lexicographical_compare(
                          a.begin(), a.end(),
                          b.begin(), b.end(),
                          [](unsigned char ca, unsigned char cb) {
                              return std::tolower(ca) < std::tolower(cb);
                          });
                  });

        size_t total_shaders = shader_files.size();

        mx::system_out << "acmx2: Compiling " << total_shaders << " shaders (" << (dual_mode ? "2D+3D" : "2D") << ")...\n";
        fflush(stdout);

        static constexpr const char *kLogoVert =
            "#version 330 core\n"
            "layout(location = 0) in vec3 aPos;\n"
            "layout(location = 1) in vec2 aTex;\n"
            "out vec2 tc;\n"
            "void main() { gl_Position = vec4(aPos, 1.0); tc = aTex; }\n";
        static constexpr const char *kLogoFrag =
            "#version 330 core\n"
            "in vec2 tc;\n"
            "out vec4 color;\n"
            "uniform sampler2D samp;\n"
            "void main() { color = texture(samp, tc); }\n";

        gl::ShaderProgram logo_shader;
        auto logo_sprite = std::make_unique<gl::GLSprite>();
        bool logo_loaded = false;
        {
            std::string logo_path = win->util.getFilePath("data/logo.png");
            if (std::filesystem::exists(logo_path)) {
                GLuint logo_tex = 0;
                try {
                    int lw = 0, lh = 0;
                    logo_tex = gl::loadTexture(logo_path, lw, lh);
                    if (logo_tex && logo_shader.loadProgramFromText(kLogoVert, kLogoFrag)) {
                        logo_sprite->initSize(win->w, win->h);
                        logo_sprite->setName("samp");
                        logo_sprite->setShader(&logo_shader);
                        float scale = std::min((float)win->w / lw, (float)win->h / lh);
                        int dw = static_cast<int>(lw * scale);
                        int dh = static_cast<int>(lh * scale);
                        int lx = (win->w - dw) / 2;
                        int ly = (win->h - dh) / 2;
                        logo_sprite->initWithTexture(&logo_shader, logo_tex, lx, ly, dw, dh);
                        logo_tex = 0;
                        logo_loaded = true;
                    }
                } catch (...) {}
                if (logo_tex) { glDeleteTextures(1, &logo_tex); }
            }
        }

        int last_percent_reported = -1;
        for (size_t shader_index = 0; shader_index < shader_files.size(); ++shader_index) {
            const std::string &line_data = shader_files[shader_index];
            std::string full_path = text + "/" + line_data;
            std::string vert_2d = win->util.getFilePath("data/vert.glsl");
            std::string vert_3d = win->util.getFilePath("data/vertex.glsl");

            bool ok_2d = false;
            programs_2d.push_back(makeProgram());
            try {
                if (programs_2d.back()->loadProgram(vert_2d, full_path)) {
                    ok_2d = true;
                } else {
                    mx::system_out << "acmx2: ⚠ Failed to compile 2D shader: " << line_data
                                   << " — substituting passthrough placeholder\n";
                    fflush(stdout);
                }
            } catch (const std::exception &e) {
                mx::system_out << "acmx2: ⚠ Exception compiling 2D shader: " << line_data
                               << " (" << e.what() << ") — substituting passthrough placeholder\n";
                fflush(stdout);
                ok_2d = false;
            } catch (...) {
                mx::system_out << "acmx2: ⚠ Unknown exception compiling 2D shader: " << line_data
                               << " — substituting passthrough placeholder\n";
                fflush(stdout);
                ok_2d = false;
            }
            if (!ok_2d) {
                // Replace the broken slot with a passthrough program so the
                // shader index stays aligned with index.txt ordering.
                programs_2d.pop_back();
                auto ph = makePassthroughProgram(vert_2d);
                if (!ph) {
                    throw mx::Exception("acmx2: Error could not build 2D passthrough placeholder for: " + line_data);
                }
                programs_2d.push_back(std::move(ph));
            }
            setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, full_path);
            if (dual_mode) {
                bool ok_3d = false;
                programs_3d.push_back(makeProgram());
                try {
                    if (programs_3d.back()->loadProgram(vert_3d, full_path)) {
                        ok_3d = true;
                    } else {
                        mx::system_out << "acmx2: ⚠ Failed to compile 3D shader: " << line_data
                                       << " — substituting passthrough placeholder\n";
                        fflush(stdout);
                    }
                } catch (const std::exception &e) {
                    mx::system_out << "acmx2: ⚠ Exception compiling 3D shader: " << line_data
                                   << " (" << e.what() << ") — substituting passthrough placeholder\n";
                    fflush(stdout);
                    ok_3d = false;
                } catch (...) {
                    mx::system_out << "acmx2: ⚠ Unknown exception compiling 3D shader: " << line_data
                                   << " — substituting passthrough placeholder\n";
                    fflush(stdout);
                    ok_3d = false;
                }
                if (!ok_3d) {
                    programs_3d.pop_back();
                    auto ph = makePassthroughProgram(vert_3d);
                    if (!ph) {
                        throw mx::Exception("acmx2: Error could not build 3D passthrough placeholder for: " + line_data);
                    }
                    programs_3d.push_back(std::move(ph));
                }
                setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, full_path);
            }

            int percent = static_cast<int>((shader_index + 1) * 100 / total_shaders);
            int percent_bucket = (percent / 10) * 10;
            if (percent_bucket > last_percent_reported) {
                last_percent_reported = percent_bucket;
                mx::system_out << "acmx2: Compiling... " << percent_bucket << "% (" << (shader_index + 1) << "/" << total_shaders << " shaders)\n";
                fflush(stdout);

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                if (logo_loaded) {
                    logo_sprite->draw();
                }
                if (loadingFont.handle().has_value()) {
                    std::string loadingText = "Compiling Shader " + std::to_string(shader_index + 1) + "/" + std::to_string(total_shaders) + "...";
                    win->text.printText_Blended(loadingFont, 10, 10, loadingText);
                }
                SDL_GL_SwapWindow(win->getWindow());
                SDL_PumpEvents();
            }
        }
        mx::system_out << "acmx2: Compiled " << shader_files.size() << " shaders (" << (dual_mode ? "2D+3D" : "2D only") << ")\n";
        fflush(stdout);
    }

    /**
     * @brief Compile every shader in a library and rewrite index.txt without
     *        the ones that fail to compile.
     *
     * Reads `<library_path>/index.txt`, attempts to compile each referenced
     * fragment shader against the supplied 2D (and optionally 3D) vertex
     * shaders, and produces a new `index.txt` that omits broken shaders.
     * The original file is preserved as `index.txt.bak` (overwritten each run).
     *
     * Non-shader lines (blank lines and lines containing "material") are
     * preserved verbatim. Files listed in index.txt that do not exist on
     * disk are also dropped.
     *
     * The existing `.shader_cache` is deleted so the library is rebuilt
     * fresh on next launch.
     *
     * @param win           GL window (for asset resolution).
     * @param library_path  Directory containing index.txt and .glsl files.
     * @param vert_2d       Path to the 2D vertex shader.
     * @param vert_3d       Path to the 3D vertex shader (only used when @ref dual_mode is set).
     * @return true if index.txt was rewritten successfully (even when no shaders were removed).
     */
    bool removeBrokenShaders(gl::GLWindow *win,
                             const std::string &library_path,
                             const std::string &vert_2d,
                             const std::string &vert_3d) {
        (void)win;
        if (glGetString(GL_VERSION) == nullptr) {
            mx::system_err << "acmx2: remove-broken requires a valid OpenGL context\n";
            return false;
        }

        std::string index_path = library_path + "/index.txt";
        std::string crash_marker_path = library_path + "/.remove_broken_last_shader";

        auto writeCrashMarker = [&](const std::string &shader_name) {
            std::ofstream marker_out(crash_marker_path, std::ios::trunc);
            if (marker_out.is_open()) {
                marker_out << shader_name;
                marker_out.flush();
            }
        };

        auto clearCrashMarker = [&]() {
            std::error_code marker_ec;
            std::filesystem::remove(crash_marker_path, marker_ec);
        };

        std::string last_crashed_shader;
        {
            std::ifstream marker_in(crash_marker_path);
            if (marker_in.is_open()) {
                std::getline(marker_in, last_crashed_shader);
            }
        }

        std::ifstream in(index_path);
        if (!in.is_open()) {
            mx::system_err << "acmx2: Could not open index.txt at: " << library_path << "\n";
            return false;
        }

        // Preserve ordering and non-shader lines (blank / "material" lines).
        struct Line {
            std::string raw;    ///< Original line text.
            bool is_shader;     ///< True if this line references a fragment shader file.
            bool keep = true;   ///< False if the shader failed to compile.
        };
        std::vector<Line> lines;
        {
            std::string l;
            while (std::getline(in, l)) {
                Line entry;
                entry.raw = l;
                auto shader_entry = normalizeShaderIndexEntry(l);
                std::string full_path;
                bool is_shader_line =
                    shader_entry.has_value() &&
                    resolveShaderPathInLibrary(library_path, *shader_entry, full_path);
                if (shader_entry) {
                    entry.raw = *shader_entry;
                }
                entry.is_shader = is_shader_line;
                if (!is_shader_line && !l.empty() &&
                    l.find("material") == std::string::npos) {
                    // Referenced file is missing — drop this entry too.
                    mx::system_out << "acmx2: ⚠ Removing missing file from index: " << l << "\n";
                    entry.is_shader = false;
                    entry.keep = false;
                }
                lines.push_back(std::move(entry));
            }
        }
        in.close();

        size_t pre_removed_from_crash = 0;
        if (!last_crashed_shader.empty()) {
            for (auto &entry : lines) {
                if (!entry.is_shader || !entry.keep) {
                    continue;
                }
                if (entry.raw == last_crashed_shader) {
                    entry.keep = false;
                    ++pre_removed_from_crash;
                    mx::system_out << "acmx2: ⚠ Previous scan crashed while compiling '"
                                   << last_crashed_shader
                                   << "' — removing it and resuming scan\n";
                    break;
                }
            }
        }

        size_t total_shaders = 0;
        for (const auto &e : lines) {
            if (e.is_shader) ++total_shaders;
        }

        mx::system_out << "acmx2: Scanning " << total_shaders
                       << " shaders in " << library_path
                       << " for compile errors ("
                       << (dual_mode ? "2D+3D" : "2D only") << ")\n";
        fflush(stdout);

        size_t removed = pre_removed_from_crash;
        size_t kept = 0;
        size_t scanned = 0;
        for (auto &entry : lines) {
            if (!entry.is_shader || !entry.keep) continue;
            ++scanned;
            std::string full_path = library_path + "/" + entry.raw;

            // Print the identifying line as a single complete line and
            // fully flush BOTH the C++ ostream buffer and the underlying
            // C stdout buffer before invoking the GL driver. On macOS
            // some shaders cause the Metal-backed GL driver to abort the
            // process; without a complete flush here, the output gets
            // truncated mid-line (e.g. "[790/") and the user can't tell
            // which shader killed the scan.
            {
                std::string scan_line = "acmx2: [" + std::to_string(scanned) + "/" +
                                        std::to_string(total_shaders) + "] " +
                                        entry.raw + " ...\n";
                mx::system_out << scan_line;
                mx::system_out.flush();
                fflush(stdout);
            }

            bool compiled = true;
            try {
                writeCrashMarker(entry.raw);
#if defined(__APPLE__)
                // On macOS, shader compile/link failures can emit massive
                // driver diagnostics to stderr; keep the scan log concise.
                ScopedStderrSilence silence_stderr;
#endif
                gl::ShaderProgram prog_2d;
                prog_2d.setSilent(true);
                if (!prog_2d.loadProgram(vert_2d, full_path)) {
                    compiled = false;
                } else {
                    GLint link_status = 0;
                    glGetProgramiv(prog_2d.id(), GL_LINK_STATUS, &link_status);
                    if (link_status != GL_TRUE) compiled = false;
                }
                if (compiled && dual_mode) {
                    gl::ShaderProgram prog_3d;
                    prog_3d.setSilent(true);
                    if (!prog_3d.loadProgram(vert_3d, full_path)) {
                        compiled = false;
                    } else {
                        GLint link_status = 0;
                        glGetProgramiv(prog_3d.id(), GL_LINK_STATUS, &link_status);
                        if (link_status != GL_TRUE) compiled = false;
                    }
                }
            } catch (const std::exception &e) {
                mx::system_out << "exception: " << e.what() << " ";
                compiled = false;
            } catch (...) {
                compiled = false;
            }
            clearCrashMarker();

            if (compiled) {
                mx::system_out << "acmx2:   -> OK\n";
                ++kept;
            } else {
                mx::system_out << "acmx2:   -> REMOVED\n";
                entry.keep = false;
                ++removed;
            }
            mx::system_out.flush();
            fflush(stdout);
        }

        clearCrashMarker();

        // Back up the original index.txt before rewriting.
        std::error_code ec;
        std::filesystem::copy_file(
            index_path,
            library_path + "/index.txt.bak",
            std::filesystem::copy_options::overwrite_existing,
            ec);
        if (ec) {
            mx::system_out << "acmx2: Warning: could not create index.txt.bak ("
                           << ec.message() << ")\n";
        }

        std::ofstream out(index_path, std::ios::trunc);
        if (!out.is_open()) {
            mx::system_err << "acmx2: Could not rewrite index.txt at: " << library_path << "\n";
            return false;
        }
        for (const auto &entry : lines) {
            if (!entry.keep) continue;
            out << entry.raw << "\n";
        }
        out.close();

        // Invalidate the on-disk cache since the library composition changed.
        std::string cache_file = shaderCacheFilePath(win ? win->util.path : std::string(), library_path);
        if (std::filesystem::exists(cache_file)) {
            std::filesystem::remove(cache_file, ec);
            if (!ec) {
                mx::system_out << "acmx2: Removed stale shader cache: " << cache_file << "\n";
            }
        }

        mx::system_out << "acmx2: Remove-broken complete: kept " << kept
                       << ", removed " << removed << " shader(s). "
                       << "Backup written to index.txt.bak\n";
        fflush(stdout);
        return true;
    }

    /**
     * @brief Build the on-disk shader binary cache for all shaders in a library.
     * @param win    GL window (provides vertex shader paths).
     * @param library_path  Directory containing index.txt and .glsl files.
     * @param vert_2d  Path to the 2D vertex shader.
     * @param vert_3d  Path to the 3D vertex shader.
     * @return true on success.
     */
    bool buildShaderCache(gl::GLWindow *win, const std::string &library_path, const std::string &vert_2d, const std::string &vert_3d) {
        (void)win;
        if (glGetString(GL_VERSION) == nullptr) {
            mx::system_err << "acmx2: build-cache requires a valid OpenGL context\n";
            return false;
        }

        GLint numFormats = 0;
        glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &numFormats);
        if (numFormats == 0) {
            mx::system_err << "acmx2: Error - OpenGL driver does not support program binaries\n";
            return false;
        }
        mx::system_out << "acmx2: OpenGL supports " << numFormats << " program binary format(s)\n";
        fflush(stdout);

        if (!loadProgramBinaryFunctions()) {
            mx::system_err << "acmx2: Error - Failed to load program binary extension functions\n";
            return false;
        }
        mx::system_out << "acmx2: Program binary functions loaded successfully\n";
        fflush(stdout);

        std::string cache_file = shaderCacheFilePath(win ? win->util.path : std::string(), library_path);
        std::fstream file;
        file.open(library_path + "/index.txt", std::ios::in);
        if (!file.is_open()) {
            mx::system_err << "acmx2: Could not open index.txt at: " << library_path << "\n";
            return false;
        }

        ShaderCache cache;
        cache.gl_renderer = safeGLString(GL_RENDERER);
        cache.gl_version = safeGLString(GL_VERSION);
        cache.dual_mode = dual_mode;

        std::vector<std::string> shader_files;
        std::string line;
        while (std::getline(file, line)) {
            auto shader_entry = normalizeShaderIndexEntry(line);
            if (!shader_entry) {
                continue;
            }
            std::string full_path;
            if (resolveShaderPathInLibrary(library_path, *shader_entry, full_path)) {
                shader_files.push_back(*shader_entry);
            }
        }
        file.close();

        // Case-insensitive sort to match Qt interface behavior
        std::sort(shader_files.begin(), shader_files.end(),
                  [](const std::string &a, const std::string &b) {
                      return std::lexicographical_compare(
                          a.begin(), a.end(),
                          b.begin(), b.end(),
                          [](unsigned char ca, unsigned char cb) {
                              return std::tolower(ca) < std::tolower(cb);
                          });
                  });

        mx::system_out << "acmx2: Building shader cache for " << shader_files.size() << " shaders...\n";
        fflush(stdout);

        for (size_t i = 0; i < shader_files.size(); ++i) {
            const std::string &shader_file = shader_files[i];
            std::string full_path = library_path + "/" + shader_file;

            mx::system_out << "acmx2: Caching Shader " << i << "/" << shader_files.size() << ": [" << shader_file << "] \n";
            fflush(stdout);

            ShaderCacheEntry entry;
            std::filesystem::path file_path(shader_file);
            entry.shader_name = file_path.stem().string();

            mx::system_out << "  - Computing hash... ";
            fflush(stdout);
            entry.source_hash = hashFileContents(full_path);
            mx::system_out << "done\n";
            fflush(stdout);

            auto mark_failed = [&](const char *reason) {
                entry.failed = true;
                entry.binary_2d.clear();
                entry.binary_3d.clear();
                entry.format_2d = 0;
                entry.format_3d = 0;
                mx::system_out << "  ⚠ SKIPPED: " << reason
                               << " (slot preserved; will run as passthrough)\n";
                fflush(stdout);
                cache.entries.push_back(std::move(entry));
            };

            try {
                mx::system_out << "  - Compiling 2D shader... ";
                fflush(stdout);

                gl::ShaderProgram prog_2d;
                prog_2d.setSilent(true);
                if (!prog_2d.loadProgram(vert_2d, full_path)) {
                    mx::system_out << " ❌ (2D compile failed)\n";
                    fflush(stdout);
                    mark_failed("2D compile failed");
                    continue;
                }
                mx::system_out << "done (id=" << prog_2d.id() << ")\n";
                fflush(stdout);

                GLint link_status = 0;
                glGetProgramiv(prog_2d.id(), GL_LINK_STATUS, &link_status);
                if (link_status != GL_TRUE) {
                    mx::system_out << "  - ❌ Program not properly linked\n";
                    fflush(stdout);
                    mark_failed("2D link failed");
                    continue;
                }

                GLint binary_retrievable = 0;
                glGetProgramiv(prog_2d.id(), GL_PROGRAM_BINARY_RETRIEVABLE_HINT, &binary_retrievable);
                mx::system_out << "  - Binary retrievable hint: " << binary_retrievable << "\n";
                fflush(stdout);

                mx::system_out << "  - Getting binary length... ";
                fflush(stdout);

                GLint binary_length = 0;
                glGetProgramiv(prog_2d.id(), GL_PROGRAM_BINARY_LENGTH, &binary_length);
                GLenum gl_error = glGetError();

                mx::system_out << binary_length << " bytes\n";
                fflush(stdout);

                if (gl_error != GL_NO_ERROR) {
                    mx::system_out << " ❌ (GL error: " << gl_error << ")\n";
                    fflush(stdout);
                    mark_failed("GL error retrieving 2D binary length");
                    continue;
                }

                if (binary_length > 0) {
                    mx::system_out << "  - Extracting binary... ";
                    fflush(stdout);

                    void *binary_buffer = malloc(binary_length);
                    if (!binary_buffer) {
                        mx::system_out << "❌ (malloc failed)\n";
                        fflush(stdout);
                        mark_failed("malloc failed for 2D binary");
                        continue;
                    }

                    GLsizei actual_length = 0;
                    GLenum format = 0;

                    mx::system_out << "calling glGetProgramBinary... ";
                    fflush(stdout);

                    glGetProgramBinaryFunc(prog_2d.id(), binary_length, &actual_length, &format, binary_buffer);
                    gl_error = glGetError();

                    mx::system_out << "done. actual=" << actual_length << ", format=" << format << "\n";
                    fflush(stdout);

                    if (gl_error != GL_NO_ERROR || actual_length == 0) {
                        mx::system_out << " ❌ (binary extraction failed, gl_error=" << gl_error << ")\n";
                        fflush(stdout);
                        free(binary_buffer);
                        mark_failed("2D binary extraction failed");
                        continue;
                    }

                    entry.binary_2d.resize(actual_length);
                    memcpy(entry.binary_2d.data(), binary_buffer, actual_length);
                    entry.format_2d = format;
                    free(binary_buffer);
                } else {
                    mx::system_out << " ❌ (no binary available)\n";
                    fflush(stdout);
                    mark_failed("no 2D binary available");
                    continue;
                }

                if (dual_mode) {
                    mx::system_out << "  - Compiling 3D shader... ";
                    fflush(stdout);

                    gl::ShaderProgram prog_3d;
                    prog_3d.setSilent(true);
                    if (!prog_3d.loadProgram(vert_3d, full_path)) {
                        mx::system_out << " ❌ (3D compile failed)\n";
                        fflush(stdout);
                        mark_failed("3D compile failed");
                        continue;
                    }

                    mx::system_out << "done (id=" << prog_3d.id() << ")\n";
                    fflush(stdout);

                    mx::system_out << "  - Getting 3D binary... ";
                    fflush(stdout);

                    GLint binary_length_3d = 0;
                    glGetProgramiv(prog_3d.id(), GL_PROGRAM_BINARY_LENGTH, &binary_length_3d);

                    mx::system_out << binary_length_3d << " bytes\n";
                    fflush(stdout);

                    if (binary_length_3d > 0) {
                        entry.binary_3d.resize(binary_length_3d);
                        GLsizei actual_length_3d = 0;
                        glGetProgramBinaryFunc(prog_3d.id(), binary_length_3d, &actual_length_3d, &entry.format_3d, entry.binary_3d.data());
                        entry.binary_3d.resize(actual_length_3d);
                        mx::system_out << "  - 3D binary extracted: " << actual_length_3d << " bytes\n";
                        fflush(stdout);
                    }
                }

                cache.entries.push_back(std::move(entry));
                mx::system_out << "  ✔ SUCCESS\n";
                fflush(stdout);
            } catch (const std::exception &e) {
                mx::system_out << " ❌ (exception: " << e.what() << ")\n";
                fflush(stdout);
                mark_failed("exception during compile");
                continue;
            } catch (...) {
                mx::system_out << " ❌ (unknown exception)\n";
                fflush(stdout);
                mark_failed("unknown exception during compile");
                continue;
            }
        }

        if (cache.save(cache_file)) {
            size_t ok_count = 0;
            size_t failed_count = 0;
            for (const auto &e : cache.entries) {
                if (e.failed) ++failed_count; else ++ok_count;
            }
            mx::system_out << "acmx2: Shader cache saved to: " << cache_file << "\n";
            mx::system_out << "acmx2: Cached " << ok_count << " shaders ("
                           << (dual_mode ? "2D+3D" : "2D only")
                           << "), " << failed_count << " failed (passthrough placeholders)\n";
            fflush(stdout);
            return true;
        } else {
            mx::system_err << "acmx2: Failed to save shader cache to: " << cache_file << "\n";
            return false;
        }
    }

    /**
     * @brief Get a display string for the currently active shader.
     *
     * Returns a string of the form `"42: bloom"` (index + stem name).
     * Used for the on-screen overlay and window title.
     *
     * @return Formatted shader name, or empty string if index is invalid.
     */
    std::string getFullShaderName() {
        std::string name;
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (names.find(library_index) != names.end()) {
            name = std::to_string(library_index) + ": " + names[library_index].name;
        }
        return name;
    }

    /**
     * @brief Get the stem name of a shader by its numeric index.
     * @param idx Zero-based shader program index.
     * @return Shader name, or empty string if @p idx is out of range.
     */
    std::string getShaderNameByIndex(size_t idx) {
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (names.find(idx) != names.end()) {
            return names[idx].name;
        }
        return "";
    }

    /**
     * @brief Get a display string including the active shader and the pass list.
     *
     * Format: `"42: bloom [edge, blur, vhs]"` when multi-pass is active.
     *
     * @param pass_list Vector of additional shader indices applied as post-passes.
     * @return Formatted composite name.
     */
    std::string getFullShaderName(const std::vector<int> &pass_list) {
        std::string name;
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (names.find(library_index) != names.end()) {
            name = std::to_string(library_index) + ": " + names[library_index].name;
        }
        if (!pass_list.empty()) {
            name += " [";
            for (size_t i = 0; i < pass_list.size(); ++i) {
                int idx = pass_list[i];
                if (names.find(idx) != names.end()) {
                    name += names[idx].name;
                } else {
                    name += std::to_string(idx);
                }
                if (i + 1 < pass_list.size()) {
                    name += ", ";
                }
            }
            name += "]";
        }
        return name;
    }

    /// @brief Attempt to load all shader programs from the binary cache file.
    bool loadFromCache(gl::GLWindow *win, const std::string &library_path, mx::Font &loadingFont,
                       const std::string &vert_2d = "", const std::string &vert_3d = "") {
        std::string cache_file = shaderCacheFilePath(win ? win->util.path : std::string(), library_path);

        mx::system_out << "acmx2: Checking for shader cache at: " << cache_file << "\n";
        fflush(stdout);

        if (!std::filesystem::exists(cache_file)) {
            mx::system_out << "acmx2: No shader cache found, will compile shaders\n";
            fflush(stdout);
            return false;
        }

        mx::system_out << "acmx2: Found shader cache, loading...\n";
        fflush(stdout);

        ShaderCache cache;
        if (!cache.load(cache_file)) {
            mx::system_out << "acmx2: Shader cache corrupted or incompatible, will recompile\n";
            return false;
        }

        if (!loadProgramBinaryFunctions()) {
            mx::system_out << "acmx2: Could not load program binary extension, will recompile\n";
            return false;
        }

        std::string current_renderer = safeGLString(GL_RENDERER);
        std::string current_version = safeGLString(GL_VERSION);

        if (cache.gl_renderer != current_renderer) {
            mx::system_out << "acmx2: GPU changed (was: " << cache.gl_renderer << ", now: " << current_renderer << "), will recompile\n";
            return false;
        }

        if (cache.gl_version != current_version) {
            mx::system_out << "acmx2: Driver version changed (was: " << cache.gl_version << ", now: " << current_version << "), will recompile\n";
            return false;
        }

        if (dual_mode && !cache.dual_mode) {
            mx::system_out << "acmx2: Cache was built in 2D-only mode but 3D is enabled, will rebuild with 2D+3D\n";
            return false;
        }

        std::fstream file;
        file.open(library_path + "/index.txt", std::ios::in);
        if (!file.is_open()) {
            return false;
        }

        std::vector<std::string> shader_files;
        std::string line;
        while (std::getline(file, line)) {
            auto shader_entry = normalizeShaderIndexEntry(line);
            if (!shader_entry) {
                continue;
            }
            std::string full_path;
            if (resolveShaderPathInLibrary(library_path, *shader_entry, full_path)) {
                shader_files.push_back(*shader_entry);
            }
        }
        file.close();

        // Case-insensitive sort to match Qt interface behavior
        std::sort(shader_files.begin(), shader_files.end(),
                  [](const std::string &a, const std::string &b) {
                      return std::lexicographical_compare(
                          a.begin(), a.end(),
                          b.begin(), b.end(),
                          [](unsigned char ca, unsigned char cb) {
                              return std::tolower(ca) < std::tolower(cb);
                          });
                  });

        if (shader_files.size() != cache.entries.size()) {
            mx::system_out << "acmx2: Shader count mismatch: index.txt has " << shader_files.size()
                           << " shaders but cache has " << cache.entries.size()
                           << " entries. Rebuilding cache...\n";
            fflush(stdout);
            if (!vert_2d.empty() && !vert_3d.empty()) {
                programs_2d.clear();
                programs_3d.clear();
                program_names_2d.clear();
                program_names_3d.clear();
                buildShaderCache(win, library_path, vert_2d, vert_3d);
                mx::system_out << "acmx2: Cache rebuilt. Loading shaders from source...\n";
                fflush(stdout);
            }
            return false;
        }

        for (size_t i = 0; i < shader_files.size(); ++i) {
            std::string full_path = library_path + "/" + shader_files[i];
            uint64_t current_hash = hashFileContents(full_path);
            if (current_hash != cache.entries[i].source_hash) {
                mx::system_out << "acmx2: Shader source changed: " << shader_files[i] << ", rebuilding cache...\n";
                fflush(stdout);
                if (!vert_2d.empty() && !vert_3d.empty()) {
                    programs_2d.clear();
                    programs_3d.clear();
                    program_names_2d.clear();
                    program_names_3d.clear();
                    buildShaderCache(win, library_path, vert_2d, vert_3d);
                    mx::system_out << "acmx2: Cache rebuilt. Loading shaders from source...\n";
                    fflush(stdout);
                }
                return false;
            }
        }

        mx::system_out << "acmx2: Loading " << cache.entries.size() << " shaders from cache...\n";
        fflush(stdout);

        static constexpr const char *kLogoVertC =
            "#version 330 core\n"
            "layout(location = 0) in vec3 aPos;\n"
            "layout(location = 1) in vec2 aTex;\n"
            "out vec2 tc;\n"
            "void main() { gl_Position = vec4(aPos, 1.0); tc = aTex; }\n";
        static constexpr const char *kLogoFragC =
            "#version 330 core\n"
            "in vec2 tc;\n"
            "out vec4 color;\n"
            "uniform sampler2D samp;\n"
            "void main() { color = texture(samp, tc); }\n";

        gl::ShaderProgram logo_shader_c;
        auto logo_sprite_c = std::make_unique<gl::GLSprite>();
        bool logo_loaded_c = false;
        {
            std::string logo_path = win ? win->util.getFilePath("data/logo.png") : std::string();
            if (!logo_path.empty() && std::filesystem::exists(logo_path)) {
                GLuint logo_tex = 0;
                try {
                    int lw = 0, lh = 0;
                    logo_tex = gl::loadTexture(logo_path, lw, lh);
                    if (logo_tex && logo_shader_c.loadProgramFromText(kLogoVertC, kLogoFragC)) {
                        logo_sprite_c->initSize(win->w, win->h);
                        logo_sprite_c->setName("samp");
                        logo_sprite_c->setShader(&logo_shader_c);
                        float scale = std::min((float)win->w / lw, (float)win->h / lh);
                        int dw = static_cast<int>(lw * scale);
                        int dh = static_cast<int>(lh * scale);
                        int lx = (win->w - dw) / 2;
                        int ly = (win->h - dh) / 2;
                        logo_sprite_c->initWithTexture(&logo_shader_c, logo_tex, lx, ly, dw, dh);
                        logo_tex = 0;
                        logo_loaded_c = true;
                    }
                } catch (...) {}
                if (logo_tex) { glDeleteTextures(1, &logo_tex); }
            }
        }

        int last_percent_reported = -1;
        size_t binary_fail_count = 0;

        // Helper: insert a passthrough program at the current slot to preserve
        // index alignment with index.txt when a cache entry cannot be used.
        auto push_passthrough_2d = [&](size_t i, const char *reason) {
            auto ph = makePassthroughProgram(vert_2d.empty()
                                                 ? win->util.getFilePath("data/vert.glsl")
                                                 : vert_2d);
            if (!ph) {
                mx::system_err << "acmx2: ❌ Failed to build 2D passthrough for slot "
                               << i << " [" << (i < shader_files.size() ? shader_files[i] : std::string("?"))
                               << "]\n";
                return false;
            }
            mx::system_out << "acmx2: ⚠ Slot " << i << " [" << (i < shader_files.size() ? shader_files[i] : std::string("?"))
                           << "] using passthrough (" << reason << ")\n";
            fflush(stdout);
            programs_2d.push_back(std::move(ph));
            setupProgramUniforms(win, programs_2d.back().get(), program_names_2d,
                                 programs_2d.size() - 1,
                                 library_path + "/" + shader_files[i]);
            return true;
        };
        auto push_passthrough_3d = [&](size_t i, const char *reason) {
            auto ph = makePassthroughProgram(vert_3d.empty()
                                                 ? win->util.getFilePath("data/vertex.glsl")
                                                 : vert_3d);
            if (!ph) {
                mx::system_err << "acmx2: ❌ Failed to build 3D passthrough for slot "
                               << i << " [" << (i < shader_files.size() ? shader_files[i] : std::string("?"))
                               << "]\n";
                return false;
            }
            mx::system_out << "acmx2: ⚠ Slot " << i << " [" << (i < shader_files.size() ? shader_files[i] : std::string("?"))
                           << "] using 3D passthrough (" << reason << ")\n";
            fflush(stdout);
            programs_3d.push_back(std::move(ph));
            setupProgramUniforms(win, programs_3d.back().get(), program_names_3d,
                                 programs_3d.size() - 1,
                                 library_path + "/" + shader_files[i]);
            return true;
        };

        for (size_t i = 0; i < cache.entries.size(); ++i) {
            const auto &entry = cache.entries[i];

            // If this entry was marked as failed when the cache was built,
            // substitute a passthrough program so the slot index stays
            // aligned with index.txt.
            if (entry.failed || entry.binary_2d.empty()) {
                if (!push_passthrough_2d(i, entry.failed ? "cached as failed" : "missing 2D binary")) {
                    return false;
                }
                if (dual_mode) {
                    if (!push_passthrough_3d(i, entry.failed ? "cached as failed" : "missing 3D binary")) {
                        return false;
                    }
                }
                continue;
            }

            programs_2d.push_back(makeProgram());
            GLuint prog_id_2d = glCreateProgram();

            GLenum gl_err = glGetError();
            glProgramBinaryFunc(prog_id_2d, entry.format_2d, entry.binary_2d.data(), static_cast<GLsizei>(entry.binary_2d.size()));
            gl_err = glGetError();

            GLint link_status = 0;
            glGetProgramiv(prog_id_2d, GL_LINK_STATUS, &link_status);
            if (link_status != GL_TRUE) {
                GLchar info_log[512];
                glGetProgramInfoLog(prog_id_2d, 512, nullptr, info_log);
                mx::system_out << "acmx2: ❌ Shader " << i << " [" << entry.shader_name << "] 2D binary load failed, gl_err=" << gl_err
                               << ", format=" << entry.format_2d
                               << ", size=" << entry.binary_2d.size()
                               << ", log=" << info_log << "\n";
                fflush(stdout);
                glDeleteProgram(prog_id_2d);
                programs_2d.pop_back();
                ++binary_fail_count;
                // Substitute passthrough to keep slot index valid.
                if (!push_passthrough_2d(i, "2D binary load failed")) {
                    return false;
                }
                if (dual_mode) {
                    if (!push_passthrough_3d(i, "2D binary load failed")) {
                        return false;
                    }
                }
                continue;
            }

            *programs_2d.back() = gl::ShaderProgram(prog_id_2d);
            setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, library_path + "/" + shader_files[i]);

            if (dual_mode) {
                if (entry.binary_3d.empty()) {
                    if (!push_passthrough_3d(i, "missing 3D binary")) {
                        return false;
                    }
                } else {
                    programs_3d.push_back(makeProgram());
                    GLuint prog_id_3d = glCreateProgram();
                    glProgramBinaryFunc(prog_id_3d, entry.format_3d, entry.binary_3d.data(), static_cast<GLsizei>(entry.binary_3d.size()));

                    glGetProgramiv(prog_id_3d, GL_LINK_STATUS, &link_status);
                    if (link_status != GL_TRUE) {
                        mx::system_out << "acmx2: ❌ Shader " << i << " [" << entry.shader_name << "] 3D binary load failed\n";
                        fflush(stdout);
                        glDeleteProgram(prog_id_3d);
                        programs_3d.pop_back();
                        if (!push_passthrough_3d(i, "3D binary load failed")) {
                            return false;
                        }
                    } else {
                        *programs_3d.back() = gl::ShaderProgram(prog_id_3d);
                        setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, library_path + "/" + shader_files[i]);
                    }
                }
            }

            int percent = static_cast<int>((i + 1) * 100 / cache.entries.size());
            int percent_bucket = (percent / 10) * 10;
            if (percent_bucket > last_percent_reported) {
                last_percent_reported = percent_bucket;
                mx::system_out << "acmx2: Cache loading... " << percent_bucket << "% (" << (i + 1) << "/" << cache.entries.size() << " shaders)\n";
                fflush(stdout);

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                if (logo_loaded_c) {
                    logo_sprite_c->draw();
                }
                if (loadingFont.handle().has_value()) {
                    std::string loadingText = "Loading Cached Shader " + std::to_string(i + 1) + "/" + std::to_string(cache.entries.size()) + "...";
                    win->text.printText_Blended(loadingFont, 10, 10, loadingText);
                }
                SDL_GL_SwapWindow(win->getWindow());
                SDL_PumpEvents();
            }
        }

        mx::system_out << "acmx2: Loaded " << cache.entries.size() << " shaders from cache (" << (dual_mode ? "2D+3D" : "2D only") << ")\n";
        fflush(stdout);

        // If more than 10% of the cached binaries failed to load (stale driver/GPU),
        // automatically delete the cache file and rebuild from source.
        // The rebuild is attempted at most once per session to avoid re-entering this
        // path repeatedly when many shaders have genuine compile errors.
        if (binary_fail_count > 0 && cache.entries.size() > 0) {
            size_t fail_pct = (binary_fail_count * 100) / cache.entries.size();
            if (fail_pct >= 10) {
                mx::system_out << "acmx2: ⚠ " << binary_fail_count << "/" << cache.entries.size()
                               << " cached shaders failed to load (" << fail_pct << "%) — cache is stale.\n";
                fflush(stdout);
                std::error_code rm_ec;
                std::filesystem::remove(cache_file, rm_ec);
                programs_2d.clear();
                programs_3d.clear();
                program_names_2d.clear();
                program_names_3d.clear();
                if (!rebuild_attempted && !vert_2d.empty() && !vert_3d.empty()) {
                    rebuild_attempted = true;
                    mx::system_out << "acmx2: Deleting stale cache and rebuilding (first attempt)...\n";
                    fflush(stdout);
                    buildShaderCache(win, library_path, vert_2d, vert_3d);
                    mx::system_out << "acmx2: Cache rebuilt from source.\n";
                    fflush(stdout);
                } else {
                    mx::system_out << "acmx2: Rebuild already attempted once this session — skipping re-build, loading from source directly.\n";
                    fflush(stdout);
                }
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Try loading a full shader library from the binary cache, with fallback.
     *
     * Calls loadFromCache() first; if the cache is missing, stale, or
     * incompatible, falls back to loadPrograms() (compile from source).
     *
     * @param win          GL window for asset resolution.
     * @param text         Shader library directory (containing index.txt).
     * @param loadingFont  Font used for the on-screen progress overlay.
     */
    void loadProgramsWithCache(gl::GLWindow *win, const std::string &text, mx::Font &loadingFont) {
        std::string vert_2d = win->util.getFilePath("data/vert.glsl");
        std::string vert_3d = win->util.getFilePath("data/vertex.glsl");
        if (loadFromCache(win, text, loadingFont, vert_2d, vert_3d)) {
            return;
        }
        loadPrograms(win, text, loadingFont);
    }

    /**
     * @brief Check whether the currently active shader is a texture-cache shader.
     *
     * Returns true when the shader name contains the substring "cache",
     * which signals ACView::draw() to bind the 8 historical textures.
     *
     * @return True if the current shader is a cache shader.
     */
    bool isCache() {
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (library_index < names.size() && names[library_index].name.find("cache") != std::string::npos)
            return true;
        return false;
    }

    /**
     * @brief Set the active shader by index.
     *
     * Bounds-checked against the current program vector (2D or 3D).
     * Logs the shader name to stdout.
     *
     * @param i Zero-based shader index in the active program vector.
     */
    void setIndex(size_t i) {
        auto &progs = is3d ? programs_3d : programs_2d;
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (i < progs.size()) {
            library_index = i;
            mx::system_out << "acmx2: Set Shader to Index: " << i << " [" << names[i].name << "] (" << (is3d ? "3D" : "2D") << ")\n";
            fflush(stdout);
        }
    }
    /// @brief Advance to the next shader in the library (clamped at the end).
    void inc() {
        auto &progs = is3d ? programs_3d : programs_2d;
        if (library_index + 1 < progs.size())
            setIndex(library_index + 1);
    }
    /// @brief Move to the previous shader in the library (clamped at zero).
    void dec() {
        if (library_index > 0)
            setIndex(library_index - 1);
    }
    /// @brief Return the current shader index.
    size_t index() { return library_index; }

    /**
     * @brief Look up a shader index by its stem name.
     * @param name Stem name to search for (case-sensitive).
     * @return Zero-based index, or -1 if not found.
     */
    int findShaderByName(const std::string &name) {
        auto &names = is3d ? program_names_3d : program_names_2d;
        for (auto &[idx, data] : names) {
            if (data.name == name)
                return static_cast<int>(idx);
        }
        return -1;
    }

    /// @brief Number of programs in the currently active set (2D or 3D).
    size_t size() { return is3d ? programs_3d.size() : programs_2d.size(); }
    /// @brief Number of 2D programs.
    size_t size2d() { return programs_2d.size(); }
    /// @brief Number of 3D programs.
    size_t size3d() { return programs_3d.size(); }

    /// @brief Activate the current shader program on the GPU (glUseProgram).
    void useProgram() {
        auto &progs = is3d ? programs_3d : programs_2d;
        progs[index()]->useProgram();
    }
    /// @brief Return a raw pointer to the currently active ShaderProgram.
    gl::ShaderProgram *shader() {
        auto &progs = is3d ? programs_3d : programs_2d;
        return progs[index()].get();
    }

    /**
     * @brief Get a shader by index from the active set (2D or 3D).
     * @param idx Zero-based program index.
     * @return Pointer to the program, or nullptr if out of range.
     */
    gl::ShaderProgram *getShader(size_t idx) {
        auto &progs = is3d ? programs_3d : programs_2d;
        if (idx < progs.size())
            return progs[idx].get();
        return nullptr;
    }

    /**
     * @brief Get a 2D shader by index (regardless of current 2D/3D mode).
     * @param idx Zero-based program index in the 2D vector.
     * @return Pointer, or nullptr if out of range.
     */
    gl::ShaderProgram *getShader2D(size_t idx) {
        if (idx < programs_2d.size())
            return programs_2d[idx].get();
        return nullptr;
    }

    /**
     * @brief Get a 3D shader by index (regardless of current 2D/3D mode).
     * @param idx Zero-based program index in the 3D vector.
     * @return Pointer, or nullptr if out of range.
     */
    gl::ShaderProgram *getShader3D(size_t idx) {
        if (idx < programs_3d.size())
            return programs_3d[idx].get();
        return nullptr;
    }

    /**
     * @brief Upload all uniforms for an arbitrary shader in the active set.
     *
     * Used by the multi-pass pipeline when each pass may target a
     * different shader index.  Computes elapsed time, delta time,
     * frame count, date, mouse state, and all audio metrics, then
     * uploads them to the program at @p idx.
     *
     * @param win GL window (provides resolution and mouse coordinates).
     * @param idx Index of the shader whose uniforms to upload.
     */
    void updateShaderUniforms(gl::GLWindow *win, size_t idx) {
        auto &progs = is3d ? programs_3d : programs_2d;
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (idx >= progs.size())
            return;
        if (names.find(idx) == names.end())
            return;

        static Uint64 start_time = SDL_GetPerformanceCounter();
        static Uint64 last_frame_time = start_time;
        static uint64_t frame_counter = 0;
        Uint64 now_time = SDL_GetPerformanceCounter();
        double elapsed_time = static_cast<double>(now_time - start_time) / SDL_GetPerformanceFrequency();
        double delta_time = static_cast<double>(now_time - last_frame_time) / SDL_GetPerformanceFrequency();
        last_frame_time = now_time;
        frame_counter++;
        auto &n = names[idx];
        progs[idx]->useProgram();
        glUniform1f(n.loc, alpha);
        glUniform1f(n.iTime, static_cast<float>(elapsed_time));
        glUniform1f(n.time_f, time_f);
        glUniform1i(n.iFrame, static_cast<int>(frame_counter % INT_MAX));
        glUniform1f(n.iTimeDelta, static_cast<float>(delta_time));
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm localTime_buf{};
#ifdef _WIN32
        localtime_s(&localTime_buf, &now_c);
#else
        localtime_r(&now_c, &localTime_buf);
#endif
        float year = static_cast<float>(localTime_buf.tm_year + 1900);
        float month = static_cast<float>(localTime_buf.tm_mon + 1);
        float day = static_cast<float>(localTime_buf.tm_mday);
        float seconds = static_cast<float>(localTime_buf.tm_hour * 3600 +
                                           localTime_buf.tm_min * 60 +
                                           localTime_buf.tm_sec);
        glUniform4f(n.iDate, year, month, day, seconds);
        if (n.iFrameRate != -1) {
            glUniform1f(n.iFrameRate, 24.0f);
        }
        int mouseX = 0, mouseY = 0;
        Uint32 mouseState = SDL_GetMouseState(&mouseX, &mouseY);
        float currentY = static_cast<float>(win->h - mouseY);
        float currentX = static_cast<float>(mouseX);
        if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT)) {
            if (!isDragging) {
                clickStartX = currentX;
                clickStartY = currentY;
                lastClickX = currentX;
                lastClickY = currentY;
                isDragging = true;
                wasClicked = true;
            }
        } else {
            isDragging = false;
        }
        if (isDragging) {
            glUniform4f(n.iMouse, currentX, currentY, clickStartX, clickStartY);
        } else {
            glUniform4f(n.iMouse, currentX, currentY, 0.0f, 0.0f);
        }
        if (wasClicked && n.iMouseClick != -1) {
            glUniform2f(n.iMouseClick, lastClickX, lastClickY);
        }
        glUniform2f(n.iResolution, static_cast<float>(win->w), static_cast<float>(win->h));
        if (n.time_speed_loc != -1) {
            glUniform1f(n.time_speed_loc, time_speed);
        }
        uploadAcidCamUniforms(n, idx);
#ifdef AUDIO_ENABLED
        if (time_audio) {
            glUniform1f(n.amp, get_amp());
            glUniform1f(n.amp_untouched, get_sense());
        }
        if (n.iSampleRate != -1) {
            glUniform1f(n.iSampleRate, 44100.0f);
        }
        if (n.iamp != -1) {
            glUniform1f(n.iamp, get_freq());
        }
        {
            float sense = get_sense() * 4.0f;
            if (n.amp_peak != -1) {
                glUniform1f(n.amp_peak, std::sqrt(get_amp_peak()) * sense);
            }
            if (n.amp_rms != -1) {
                glUniform1f(n.amp_rms, std::sqrt(get_amp_rms()) * sense);
            }
            if (n.amp_smooth != -1) {
                glUniform1f(n.amp_smooth, std::sqrt(get_amp_smooth()) * sense);
            }
            if (n.amp_low != -1) {
                glUniform1f(n.amp_low, std::sqrt(get_amp_low()) * sense);
            }
            if (n.amp_mid != -1) {
                glUniform1f(n.amp_mid, std::sqrt(get_amp_mid()) * sense);
            }
            if (n.amp_high != -1) {
                glUniform1f(n.amp_high, std::sqrt(get_amp_high()) * sense);
            }
        }
        // Tell the shader which texture unit holds the spectrum.
        if (n.spectrum_loc != -1) {
            glUniform1i(n.spectrum_loc, SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        }
#endif
#ifdef MIDI_ENABLED
        for (int i = 0; i < 4; ++i) {
            if (n.slider_loc[i] != -1)
                glUniform1f(n.slider_loc[i], midi_slider[i]);
        }
#endif
    }

    /**
     * @brief Upload all uniforms for a shader in the **2D** set specifically.
     *
     * Identical to updateShaderUniforms() but explicitly indexes into
     * `programs_2d` / `program_names_2d`, bypassing the 2D/3D mode
     * switch.  This is used during 3D-mode multi-pass rendering,
     * where the post-processing passes always run in 2D but the
     * final composite runs in 3D.
     *
     * @param win GL window.
     * @param idx Index in the 2D program vector.
     */
    void updateShaderUniforms2D(gl::GLWindow *win, size_t idx) {
        if (idx >= programs_2d.size())
            return;
        if (program_names_2d.find(idx) == program_names_2d.end())
            return;

        static Uint64 start_time = SDL_GetPerformanceCounter();
        static Uint64 last_frame_time = start_time;
        static uint64_t frame_counter = 0;
        Uint64 now_time = SDL_GetPerformanceCounter();
        double elapsed_time = static_cast<double>(now_time - start_time) / SDL_GetPerformanceFrequency();
        double delta_time = static_cast<double>(now_time - last_frame_time) / SDL_GetPerformanceFrequency();
        last_frame_time = now_time;
        frame_counter++;
        auto &n = program_names_2d[idx];
        programs_2d[idx]->useProgram();
        glUniform1f(n.loc, alpha);
        glUniform1f(n.iTime, static_cast<float>(elapsed_time));
        glUniform1f(n.time_f, time_f);
        glUniform1i(n.iFrame, static_cast<int>(frame_counter % INT_MAX));
        glUniform1f(n.iTimeDelta, static_cast<float>(delta_time));
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm localTime_buf{};
#ifdef _WIN32
        localtime_s(&localTime_buf, &now_c);
#else
        localtime_r(&now_c, &localTime_buf);
#endif
        float year = static_cast<float>(localTime_buf.tm_year + 1900);
        float month = static_cast<float>(localTime_buf.tm_mon + 1);
        float day = static_cast<float>(localTime_buf.tm_mday);
        float seconds = static_cast<float>(localTime_buf.tm_hour * 3600 +
                                           localTime_buf.tm_min * 60 +
                                           localTime_buf.tm_sec);
        glUniform4f(n.iDate, year, month, day, seconds);
        if (n.iFrameRate != -1) {
            glUniform1f(n.iFrameRate, 24.0f);
        }
        int mouseX = 0, mouseY = 0;
        Uint32 mouseState = SDL_GetMouseState(&mouseX, &mouseY);
        float currentY = static_cast<float>(win->h - mouseY);
        float currentX = static_cast<float>(mouseX);
        if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT)) {
            if (!isDragging) {
                clickStartX = currentX;
                clickStartY = currentY;
                lastClickX = currentX;
                lastClickY = currentY;
                isDragging = true;
                wasClicked = true;
            }
        } else {
            isDragging = false;
        }
        if (isDragging) {
            glUniform4f(n.iMouse, currentX, currentY, clickStartX, clickStartY);
        } else {
            glUniform4f(n.iMouse, currentX, currentY, 0.0f, 0.0f);
        }
        if (wasClicked && n.iMouseClick != -1) {
            glUniform2f(n.iMouseClick, lastClickX, lastClickY);
        }
        glUniform2f(n.iResolution, static_cast<float>(win->w), static_cast<float>(win->h));
        if (n.time_speed_loc != -1) {
            glUniform1f(n.time_speed_loc, time_speed);
        }
        uploadAcidCamUniforms(n, idx);
#ifdef AUDIO_ENABLED
        if (time_audio) {
            glUniform1f(n.amp, get_amp());
            glUniform1f(n.amp_untouched, get_sense());
        }
        if (n.iSampleRate != -1) {
            glUniform1f(n.iSampleRate, 44100.0f);
        }
        if (n.iamp != -1) {
            glUniform1f(n.iamp, get_freq());
        }
        {
            float sense = get_sense() * 4.0f;
            if (n.amp_peak != -1) {
                glUniform1f(n.amp_peak, std::sqrt(get_amp_peak()) * sense);
            }
            if (n.amp_rms != -1) {
                glUniform1f(n.amp_rms, std::sqrt(get_amp_rms()) * sense);
            }
            if (n.amp_smooth != -1) {
                glUniform1f(n.amp_smooth, std::sqrt(get_amp_smooth()) * sense);
            }
            if (n.amp_low != -1) {
                glUniform1f(n.amp_low, std::sqrt(get_amp_low()) * sense);
            }
            if (n.amp_mid != -1) {
                glUniform1f(n.amp_mid, std::sqrt(get_amp_mid()) * sense);
            }
            if (n.amp_high != -1) {
                glUniform1f(n.amp_high, std::sqrt(get_amp_high()) * sense);
            }
        }
        if (n.spectrum_loc != -1) {
            glUniform1i(n.spectrum_loc, SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        }
#endif
#ifdef MIDI_ENABLED
        for (int i = 0; i < 4; ++i) {
            if (n.slider_loc[i] != -1)
                glUniform1f(n.slider_loc[i], midi_slider[i]);
        }
#endif
    }

    /**
     * @brief Per-frame update: advance time_f, upload all uniforms to the active shader.
     *
     * Called once per frame from ACView::draw().  This method:
     * 1. Computes delta time from SDL performance counters.
     * 2. Advances `time_f` either by wall-clock delta (scaled by time_speed)
     *    or by audio amplitude (when audio-reactive time is enabled).
     * 3. Uploads time_f, iTime, iFrame, iTimeDelta, iDate, iFrameRate,
     *    iMouse (with Shadertoy click semantics), and iResolution.
     * 4. Steps and uploads the acidcamGL-compatible oscillator uniforms.
     * 5. Uploads all audio amplitude and frequency-band uniforms (when
     *    AUDIO_ENABLED is defined and time_audio is true).
     *
     * @param win GL window providing resolution and drawable size.
     */
    void update(gl::GLWindow *win) {
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (names.find(index()) == names.end()) {
            return;
        }
        static Uint64 start_time = SDL_GetPerformanceCounter();
        static Uint64 last_frame_time = start_time;
        static uint64_t frame_counter = 0;
        static double last_good_delta = 1.0 / 60.0;

        Uint64 now_time = SDL_GetPerformanceCounter();
//        double elapsed_time = static_cast<double>(now_time - start_time) / SDL_GetPerformanceFrequency();
        double delta_time = static_cast<double>(now_time - last_frame_time) / SDL_GetPerformanceFrequency();
        if (delta_time <= 0.0)
            delta_time = last_good_delta;
        else
            last_good_delta = delta_time;
        last_frame_time = now_time;
        frame_counter++;

        if (time_audio == false && time_active) {
            if (video_fps > 0.0) {
                time_f += static_cast<float>(1.0 / video_fps) * time_speed;
            } else {
                time_f += static_cast<float>(delta_time) * time_speed;
            }
        } else {
#ifdef AUDIO_ENABLED
            if (time_audio) {
                float dt_scalex = audio_delta ? static_cast<float>(delta_time) : 1.0f;
                float new_ampx = ((get_amp() * get_sense()) * (time_speed * dt_scalex));
                time_f += new_ampx;
            }
#endif
        }
        if (std::isnan(time_f) || std::isinf(time_f))
            time_f = 1.0;

        GLuint time_f_loc = names[index()].time_f;
        glUniform1f(time_f_loc, time_f);
        GLint loc = names[index()].loc;
        glUniform1f(loc, alpha);
        GLint iTimeLoc = names[index()].iTime;
        double currentTime = static_cast<double>(SDL_GetTicks64()) / 1000.0f;
        glUniform1f(iTimeLoc, currentTime);
        GLint iFrameLoc = names[index()].iFrame;
        glUniform1i(iFrameLoc, static_cast<int>(frame_counter % INT_MAX));
        GLint iTimeDeltaLoc = names[index()].iTimeDelta;
        glUniform1f(iTimeDeltaLoc, static_cast<float>(delta_time));
        GLint iDateLoc = names[index()].iDate;
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm localTime_buf{};
#ifdef _WIN32
        localtime_s(&localTime_buf, &now_c);
#else
        localtime_r(&now_c, &localTime_buf);
#endif
        float year = static_cast<float>(localTime_buf.tm_year + 1900);
        float month = static_cast<float>(localTime_buf.tm_mon + 1);
        float day = static_cast<float>(localTime_buf.tm_mday);
        float seconds = static_cast<float>(localTime_buf.tm_hour * 3600 +
                                           localTime_buf.tm_min * 60 +
                                           localTime_buf.tm_sec);
        glUniform4f(iDateLoc, year, month, day, seconds);

        GLint iFrameRateLoc = names[index()].iFrameRate;
        if (iFrameRateLoc != -1) {
            glUniform1f(iFrameRateLoc, 24.0f);
        }

        GLint iMouseLoc = names[index()].iMouse;
        GLint iMouseClickLoc = names[index()].iMouseClick;

        int mouseX = 0, mouseY = 0;
        Uint32 mouseState = SDL_GetMouseState(&mouseX, &mouseY);
        float currentY = static_cast<float>(win->h - mouseY);
        float currentX = static_cast<float>(mouseX);

        if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT)) {
            if (!isDragging) {
                clickStartX = currentX;
                clickStartY = currentY;
                lastClickX = currentX;
                lastClickY = currentY;
                isDragging = true;
                wasClicked = true;
            }
        } else {
            isDragging = false;
        }

        if (isDragging) {
            glUniform4f(iMouseLoc, currentX, currentY, clickStartX, clickStartY);
        } else {
            glUniform4f(iMouseLoc, currentX, currentY, 0.0f, 0.0f);
        }

        if (wasClicked && iMouseClickLoc != -1) {
            glUniform2f(iMouseClickLoc, lastClickX, lastClickY);
        }

        GLint iResolution = names[index()].iResolution;
        glUniform2f(iResolution, win->w, win->h);
        if (names[index()].time_speed_loc != -1) {
            glUniform1f(names[index()].time_speed_loc, time_speed);
        }

        stepAcidCamUniforms();
        uploadAcidCamUniforms(names[index()], index());

#ifdef AUDIO_ENABLED
        GLuint amp_i = names[index()].amp;
        float amplitude = 1.0f;
        float dt_scale = audio_delta ? static_cast<float>(delta_time) : 1.0f;
        float new_amp = (get_amp() * get_sense()) * (time_speed * dt_scale);
        if (std::isnan(new_amp) || std::isinf(new_amp) || new_amp > 1e6f) {
            amplitude = 1.0f;
        } else {
            amplitude = new_amp;
        }
        glUniform1f(amp_i, amplitude);
        GLuint amp_u = names[index()].amp_untouched;
        glUniform1f(amp_u, get_amp());
        GLint iSampleRateLoc = names[index()].iSampleRate;
        if (iSampleRateLoc != -1) {
            glUniform1f(iSampleRateLoc, 44100.0f);
        }
        if (names[index()].iamp != -1) {
            glUniform1f(names[index()].iamp, get_freq());
        }
        {
            float sense = get_sense() * 4.0f;
            auto &n = names[index()];
            if (n.amp_peak != -1) {
                glUniform1f(n.amp_peak, std::sqrt(get_amp_peak()) * sense);
            }
            if (n.amp_rms != -1) {
                glUniform1f(n.amp_rms, std::sqrt(get_amp_rms()) * sense);
            }
            if (n.amp_smooth != -1) {
                glUniform1f(n.amp_smooth, std::sqrt(get_amp_smooth()) * sense);
            }
            if (n.amp_low != -1) {
                glUniform1f(n.amp_low, std::sqrt(get_amp_low()) * sense);
            }
            if (n.amp_mid != -1) {
                glUniform1f(n.amp_mid, std::sqrt(get_amp_mid()) * sense);
            }
            if (n.amp_high != -1) {
                glUniform1f(n.amp_high, std::sqrt(get_amp_high()) * sense);
            }
        }
        if (names[index()].spectrum_loc != -1) {
            glUniform1i(names[index()].spectrum_loc, SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        }
#endif
#ifdef MIDI_ENABLED
        for (int i = 0; i < 4; ++i) {
            if (names[index()].slider_loc[i] != -1)
                glUniform1f(names[index()].slider_loc[i], midi_slider[i]);
        }
#endif
    }

    /**
     * @brief Advance acidcamGL-compatible colour-alpha oscillators and random values.
     *
     * Each colour channel (R, G, B) is incremented by a random 0–0.99
     * and reset to 0.1 when exceeding 1.5.  The `alpha` value
     * oscillates between 1.0 and 6.0.  `random_var` receives four
     * new random bytes.  These values are consumed by legacy acidcamGL
     * shaders that expect per-frame animation in these uniforms.
     */
    void stepAcidCamUniforms() {
        color_alpha_r += (rand() % 100) * 0.01f;
        color_alpha_g += (rand() % 100) * 0.01f;
        color_alpha_b += (rand() % 100) * 0.01f;
        if (color_alpha_r > 1.5f) color_alpha_r = 0.1f;
        if (color_alpha_g > 1.5f) color_alpha_g = 0.1f;
        if (color_alpha_b > 1.5f) color_alpha_b = 0.1f;

        if (alpha_dir) {
            alpha += 0.1f;
            if (alpha >= 6.0f) alpha_dir = false;
        } else {
            alpha -= 0.1f;
            if (alpha <= 1.0f) alpha_dir = true;
        }

        random_var = glm::vec4(rand() % 255, rand() % 255, rand() % 255, rand() % 255);
    }

    /**
     * @brief Upload all acidcamGL-compatible uniforms to the GPU for the given program.
     *
     * Sends value_alpha_r/g/b, alpha_r/g/b, alpha_value, index_value,
     * optx, random_var, restore_black, inc_value, and inc_valuex to
     * the uniform locations cached in @p n.  Each upload is skipped
     * if the location is -1 (uniform not declared in that shader).
     *
     * @param n   ProgramData containing cached uniform locations.
     * @param idx Shader index (passed to index_value uniform).
     */
    void uploadAcidCamUniforms(const ProgramData &n, size_t idx) {
        if (n.value_alpha_r != -1) glUniform1f(n.value_alpha_r, color_alpha_r);
        if (n.value_alpha_g != -1) glUniform1f(n.value_alpha_g, color_alpha_g);
        if (n.value_alpha_b != -1) glUniform1f(n.value_alpha_b, color_alpha_b);
        if (n.alpha_r_loc != -1) glUniform1f(n.alpha_r_loc, color_alpha_r);
        if (n.alpha_g_loc != -1) glUniform1f(n.alpha_g_loc, color_alpha_g);
        if (n.alpha_b_loc != -1) glUniform1f(n.alpha_b_loc, color_alpha_b);
        if (n.alpha_value != -1) glUniform1f(n.alpha_value, alpha);
        if (n.index_value != -1) glUniform1f(n.index_value, static_cast<float>(idx));
        if (n.optx_loc != -1) glUniform4fv(n.optx_loc, 1, glm::value_ptr(optx));
        if (n.random_var_loc != -1) glUniform4fv(n.random_var_loc, 1, glm::value_ptr(random_var));
        if (n.restore_black_loc != -1) glUniform1f(n.restore_black_loc, restore_black ? 1.0f : 0.0f);
        if (n.inc_value_loc != -1) glUniform4fv(n.inc_value_loc, 1, glm::value_ptr(inc_value));
        if (n.inc_valuex_loc != -1) glUniform4fv(n.inc_valuex_loc, 1, glm::value_ptr(inc_valuex));
    }

    /**
     * @brief Step time_f forward manually (when auto-time is paused).
     * @param value Amount to add to time_f.
     */
    void incTime(float value) {
        if (!time_active) {
            time_f += value;
            mx::system_out << "acmx2: Time step forward: " << time_f << "\n";
            fflush(stdout);
        }
    }

    /**
     * @brief Step time_f backward manually (when auto-time is paused).
     *
     * Clamps at 1.0 to avoid negative or zero time values.
     *
     * @param value Amount to subtract from time_f.
     */
    void decTime(float value) {
        if (!time_active) {
            if (time_f - value > 1.0) {
                time_f -= value;
                mx::system_out << "acmx2: Time step back: " << time_f << "\n";
            } else {
                time_f = 1.0f;
                mx::system_out << "acmx2: Time reset to: " << time_f << "\n";
            }
            fflush(stdout);
        }
    }

    /**
     * @brief Enable or disable automatic wall-clock time advancement.
     *
     * When disabled, time_f only changes via manual incTime/decTime
     * or audio-reactive mode, allowing the user to freeze and scrub
     * the shader's temporal state.
     *
     * @param t True to enable, false to pause.
     */
    void activeTime(bool t) {
        time_active = t;
        std::string enabled = ((t == true) ? "on" : "off");
        mx::system_out << "acmx2: active time: " << enabled << "\n";
        fflush(stdout);
    }

    /**
     * @brief Enable or disable audio-reactive time advancement.
     *
     * When enabled, time_f is driven by `get_amp() * get_sense()`
     * instead of wall-clock delta, making the shader evolve in
     * sync with the audio input.
     *
     * @param t True to enable audio-reactive time.
     */
    void audioTime(bool t) {
        time_audio = t;
        std::string enabled = ((t == true) ? "on" : "off");
        mx::system_out << "acmx2: audio time: " << enabled << "\n";
        fflush(stdout);
    }
    /**
     * @brief Toggle delta-time scaling for audio-reactive mode.
     *
     * When on, the audio amplitude contribution to time_f is
     * multiplied by the frame delta time, smoothing the advance
     * rate.  When off, each frame advances by the raw amplitude.
     */
    void toggleAudioDelta() {
        audio_delta = !audio_delta;
        mx::system_out << "acmx2: audio delta time: " << (audio_delta ? "on" : "off") << "\n";
        fflush(stdout);
    }
    bool audioDelta() const { return audio_delta; }
#ifdef AUDIO_ENABLED
    bool timeActive() const { return time_active; }
    bool timeAudio() const { return time_audio; }
    float getAmp() const { return get_amp(); }
    float getAmpUntouched() const { return get_sense(); }
#endif
    /// @brief Reserved for future SDL event handling inside the library.
    void event(SDL_Event &e) {}
};

/**
 * @struct MXArguments
 * @brief Parsed command-line arguments for the ACMX2 application.
 *
 * Populated by the Argz parser in main() and forwarded to ACView.
 */
struct MXArguments {
    std::string path, filename, ofilename;
    std::string graphic_file;
    int audio_input = -1, audio_output = -1;
    int tw = 1280, th = 720;
    std::string crf = "23";
    int camera_device = 0;
    std::string library;
    std::string fragment;
    std::string prefix_path = ".";
    std::string model_file = "cube.mxmod.z";
    int mode = 0;
    int shader_index = 0;
    std::optional<cv::Size> sizev = std::nullopt;
    std::optional<cv::Size> csize = std::nullopt;
    double fps_value = 24.0;
    bool repeat = false;
    std::tuple<int, std::string, int> slib;
    bool full = false;
    bool cache = false;
    int cache_delay = 1;
    bool copy_audio = false;
    bool is3d = false;
#ifdef AUDIO_ENABLED
    bool audio_enabled = false;
    unsigned int audio_channels = 2;
    float audio_sensitivty = 0.25f;
    std::string record_audio_file;
    float record_gain = 1.0f;
    std::string audio_file;
    bool audio_trunc = false; ///< When true, stop playback when file audio reaches the end.
#endif
    bool silent = false;
#ifdef MIDI_ENABLED
    std::string midi_map_file;
    int midi_device = -1;
#endif
    bool gpu_filter_enabled = false;
    std::vector<int> gpu_filter_indices;
    int gpu_frame_buffer_size = 8;
    bool disable_counter = false;
    int cuda_device = 0;
    std::vector<int> shader_pass_list;
    bool shader_pass_enabled = false;
    bool build_cache = false;
    std::string build_library_path;
    bool remove_broken = false;        ///< True when `--remove-broken <path>` was specified.
    std::string remove_broken_path;    ///< Library path passed to `--remove-broken`.
#ifdef __APPLE__
    // The shader binary cache is unsupported on macOS (no usable
    // glProgramBinary path under the system OpenGL framework), so it
    // is permanently disabled on Apple platforms.
    bool use_shader_cache = false;
#else
    bool use_shader_cache = true;
#endif
    float time_speed = 1.0f;
    std::string playlist_file;
    int autopilot_frames = 0;            ///< Frames between random shader switches in autopilot mode (0 = disabled).
    double duration = 0.0;
    float cross_fade_duration = 0.5f; ///< Crossfade duration in seconds when switching playlist shaders (default: 0.5).
    bool use_yuv = false;
    bool flip_output = false;          ///< Vertical flip output frames when set (e.g., for HDR correction).
    bool no_drop = false;              ///< In video mode, block when encoder queue is full instead of dropping.
    bool display_filter = false;       ///< Display current shader/stack and GPU filter overlay in upper-left.
    std::string watermark_text;        ///< User watermark text (--use-watermark). When non-empty, watermark is enabled.
    int watermark_r = 255;             ///< Watermark red channel (0-255), default magenta-pink.
    int watermark_g = 0;               ///< Watermark green channel (0-255).
    int watermark_b = 150;             ///< Watermark blue channel (0-255).
    // User-configurable encoder quality (see EncodeOptions in mxwrite.hpp).
    EncodeOptions encode_opts{};
};

/**
 * @struct FrameData
 * @brief A captured RGBA pixel buffer queued for the writer thread.
 *
 * Holds a vertically-flipped copy of the framebuffer contents
 * plus metadata for the async writer (dimensions, snapshot flag).
 */
struct FrameData {
    std::vector<unsigned char> pixels; ///< RGBA pixel data (SDR 8-bit RGBA, or HDR 16-bit RGBA when @c isHdr).
    int width = 0;                     ///< Frame width in pixels.
    int height = 0;                    ///< Frame height in pixels.
    bool isSnapshot = false;           ///< True if this frame should be saved as a PNG snapshot.
    bool isWebPSnapshot = false;       ///< True if this frame should be saved as a WebP snapshot.
    bool isRawSnapshot = false;        ///< True if this frame should be saved as a raw RGBA file.
    bool isTiffSnapshot = false;       ///< True if this frame should be saved as a TIFF snapshot (16-bit HDR, 8-bit SDR).
    bool isHdr = false;                ///< True when @c pixels holds 16-bit PQ/HLG-encoded BT.2020 RGBA (8 bytes/pixel).
    int hdrTrc = 0;                    ///< AVColorTransferCharacteristic (PQ=16, HLG=18) when @c isHdr.
};

/**
 * @class ACView
 * @brief Core rendering object—drives the capture → filter → shade → record pipeline.
 *
 * Inherits gl::GLObject (libmx2) so it can be hosted inside a GLWindow.
 * Manages:
 * - Video/camera/image capture (OpenCV, with a dedicated capture thread).
 * - GPU filtering via CUDA kernels (ac_gpu).
 * - GLSL shader application (ShaderLibrary), including multipass chains.
 * - 3D model rendering with camera controls (GLM).
 * - PBO-based async frame readback for recording and snapshots.
 * - Audio reactivity (RtAudio) and MIDI controller input (RtMidi).
 * - On-screen overlay (FPS, timer, shader name, MIDI status).
 *
 * @section threading Threading Model
 *
 * ACMX2 uses four concurrency domains.  The diagram below shows data flow
 * and the synchronisation primitives that connect them.
 *
 * @verbatim
 *
 *  ┌─────────────────────┐
 *  │   Capture Thread    │  (camera mode only)
 *  │  cap.read(frame)    │
 *  │  cv::flip → push    │
 *  │                     │
 *  │  captureQueue       │  std::queue<cv::Mat>, max 4 entries
 *  │  captureQueueMutex  │  protects the queue
 *  │  captureQueueCondVar│  signals new frame available
 *  │  captureRunning     │  std::atomic<bool> for shutdown
 *  └────────┬────────────┘
 *           │
 *           │  (main thread pops from captureQueue)
 *           ▼
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │                    Main / GL Thread                         │
 *  │                                                             │
 *  │  draw()  ──── 1. Read frame from captureQueue (camera) or  │
 *  │          │       cap.read() (video file) or clone (image)   │
 *  │          │    2. GPU-filter (CUDA kernels, TextureUploader) │
 *  │          │    3. Upload to GL texture                       │
 *  │          │    4. Render through ShaderLibrary (FBO)          │
 *  │          │    5. 3D model compositing (if enabled)           │
 *  │          │    6. PBO async readback (double-buffered)        │
 *  │          │    7. Push FrameData into frameQueue              │
 *  │          └──── 8. Draw overlay (FPS, timer, shader name)     │
 *  │                                                             │
 *  │  frameQueue       std::queue<FrameData>                     │
 *  │  queueMutex       protects the queue                        │
 *  │  queueCondVar     signals new frame / backpressure wait     │
 *  │                   • camera mode: drops oldest if >30 frames  │
 *  │                   • file mode: waits until queue <30 frames  │
 *  └──────────────┬──────────────────────────────────────────────┘
 *                 │
 *                 │  (writer thread pops from frameQueue)
 *                 ▼
 *  ┌─────────────────────────────────┐
 *  │        Writer Thread            │
 *  │  Dequeue FrameData              │
 *  │   • If isSnapshot: dispatch to  │
 *  │     SnapshotThreadPool (async   │
 *  │     PNG write via enqueue())    │
 *  │   • Else: writer.write() /      │
 *  │     writer.write_ts()           │
 *  │                                 │
 *  │  writerRunning  atomic<bool>    │
 *  │  written_frame_counter          │
 *  │  (skips first N warmup frames)  │
 *  └────────┬────────────────────────┘
 *           │
 *           │  (on shutdown, if audio was recorded)
 *           ▼
 *  ┌─────────────────────────────────┐
 *  │         Mux Thread              │
 *  │  beginMuxing() launches this    │
 *  │   1. Join capture + writer      │
 *  │   2. Close the Writer           │
 *  │   3. runMuxSync(): ffmpeg mux   │
 *  │      audio WAV + video MP4      │
 *  │   4. Set muxComplete = true     │
 *  │                                 │
 *  │  isMuxing    atomic<bool>       │
 *  │  muxComplete atomic<bool>       │
 *  └─────────────────────────────────┘
 *
 * **Synchronisation summary:**
 * - `captureQueueMutex` + `captureQueueCondVar` guard the camera queue.
 *   Max size = 4; oldest frame is dropped when full (non-blocking producer).
 * - `queueMutex` + `queueCondVar` guard the writer queue.
 *   In file mode the main thread *waits* if the queue exceeds 30 frames
 *   (back-pressure).  In camera mode it drops the oldest frame instead.
 * - `running`, `captureRunning`, `writerRunning` are `std::atomic<bool>`
 *   flags tested every iteration to coordinate graceful shutdown.
 * - `isMuxing` and `muxComplete` coordinate the post-recording audio
 *   mux step; the main draw loop shows a "Muxing audio…" overlay while
 *   the mux thread runs ffmpeg.
 * - The SnapshotThreadPool is orthogonal—its own internal mutex/condvar
 *   manage the PNG-writing worker threads.  Tasks are fire-and-forget
 *   from the writer thread's perspective.
 *
 * @endverbatim
 */
class ACView : public gl::GLObject {
#ifdef AUDIO_ENABLED
    bool audio_is_enabled = false;
    int audio_input_device;
    int audio_output_device;
    std::string audio_record_file;
    SpectrumTexture spectrumTex; ///< 1D texture holding the FFT magnitude spectrum for shaders.
    bool spectrum_scale_by_sense = false; ///< When true, scale spectrum 1D buffer by audio sensitivity.
    bool file_audio_mode = false; ///< True when audio comes from a file instead of RtAudio.
    std::string audio_file_path; ///< Path to the audio file used for file_audio_mode.
    bool audio_trunc_mode = false; ///< When true, stop playback when file audio samples are exhausted.
#endif
#ifdef MIDI_ENABLED
    RtMidiIn *midiIn = nullptr;
    bool midiOpen = false;
    struct MidiCode {
        int key1;
        int key2;
        unsigned char b0, b1, b2;
    };
    std::vector<MidiCode> midiCodes;
    // Track last knob CC values for continuous firing
    std::map<std::pair<unsigned char, unsigned char>, unsigned char> knobState;
    // Track previous knob values for delta-based direction (Pitch/Yaw)
    std::map<std::pair<unsigned char, unsigned char>, unsigned char> knobPrevValue;
    // Frame counter for velocity-sensitive knob rate
    std::map<std::pair<unsigned char, unsigned char>, int> knobFrameCount;
    // Last button action string for overlay display
    std::string lastMidiButton;
    std::chrono::steady_clock::time_point lastMidiButtonTime;

    /**
     * @brief Map a virtual key code to a human-readable short name.
     *
     * Virtual codes 262–267 are arrow/page keys, 32–87 are ASCII,
     * and 500–513 are ACMX2 virtual codes for time/speed/rotation
     * knob actions.
     *
     * @param code Virtual key code from the MIDI map file.
     * @return Short label string (e.g. "Right", "SpdUp", "PitchDn").
     */
    static const char* midiKeyName(int code) {
        switch (code) {
        case 262: return "Right";
        case 263: return "Left";
        case 264: return "Down";
        case 265: return "Up";
        case 266: return "PgUp";
        case 267: return "PgDn";
        case 269: return "End";
        case 32:  return "Space";
        case 44:  return "Comma";
        case 45:  return "Minus";
        case 46:  return "Period";
        case 47:  return "Slash";
        case 61:  return "Plus/Eq";
        case 65:  return "A";
        case 66:  return "B";
        case 68:  return "D";
        case 71:  return "G";
        case 70:  return "F";
        case 72:  return "H";
        case 76:  return "L";
        case 78:  return "N";
        case 80:  return "P";
        case 82:  return "R";
        case 83:  return "S";
        case 87:  return "W";
        case 500: return "TimeFwd";
        case 501: return "TimeBack";
        case 502: return "TimePause";
        case 503: return "TimeToggle";
        case 504: return "SpdUp";
        case 505: return "SpdDn";
        case 506: return "PitchUp";
        case 507: return "PitchDn";
        case 508: return "YawR";
        case 509: return "YawL";
        case 510: return "RotSpdUp";
        case 511: return "RotSpdDn";
        case 512: return "RollR";
        case 513: return "RollL";
        case 514: return "ScaleUp";
        case 515: return "ScaleDn";
        case 600: return "Slider1";
        case 601: return "Slider1";
        case 602: return "Slider2";
        case 603: return "Slider2";
        case 604: return "Slider3";
        case 605: return "Slider3";
        case 606: return "Slider4";
        case 607: return "Slider4";
        default:  return "?";
        }
    }

    /**
     * @brief Open a MIDI input port and load the key→action mapping file.
     *
     * Creates an RtMidiIn instance, opens the given device port (or
     * port 0 as fallback), then parses the mapping file line by line.
     * Each line maps a `key1:key2` pair to a MIDI triplet `{b0 b1 b2}`,
     * where key2 != 0 denotes a continuous knob (key1 = CW direction,
     * key2 = CCW direction).
     *
     * @param mapFile    Path to the `.midi_cfg` mapping file.
     * @param deviceIndex MIDI input port index (-1 for default / port 0).
     */
    void initMidi(const std::string &mapFile, int deviceIndex) {
        try {
            midiIn = new RtMidiIn();
            midiIn->ignoreTypes(false, false, false);
            unsigned int ports = midiIn->getPortCount();
            if (ports == 0) {
                mx::system_out << "acmx2: No MIDI input devices found\n";
                delete midiIn;
                midiIn = nullptr;
                return;
            }
            unsigned int port = (deviceIndex >= 0 && deviceIndex < static_cast<int>(ports))
                                    ? static_cast<unsigned int>(deviceIndex) : 0;
            mx::system_out << "acmx2: Opening MIDI port " << port << ": " << midiIn->getPortName(port) << "\n";
            midiIn->openPort(port);
            midiOpen = true;

            std::ifstream file(mapFile);
            if (!file.is_open()) {
                mx::system_err << "acmx2: Could not open MIDI map file: " << mapFile << "\n";
                return;
            }
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                std::string keyPair;
                if (!(iss >> keyPair)) continue;
                auto colonPos = keyPair.find(':');
                if (colonPos == std::string::npos) continue;
                int k1 = std::stoi(keyPair.substr(0, colonPos));
                int k2 = std::stoi(keyPair.substr(colonPos + 1));
                char brace;
                if (!(iss >> brace) || brace != '{') continue;
                int b0, b1, b2;
                if (!(iss >> b0 >> b1 >> b2)) continue;
                midiCodes.push_back({k1, k2,
                    static_cast<unsigned char>(b0),
                    static_cast<unsigned char>(b1),
                    static_cast<unsigned char>(b2)});
            }
            mx::system_out << "acmx2: Loaded " << midiCodes.size() << " MIDI mapping(s)\n";
            fflush(stdout);
        } catch (RtMidiError &e) {
            mx::system_err << "acmx2: MIDI error: " << e.getMessage() << "\n";
            if (midiIn) { delete midiIn; midiIn = nullptr; }
        }
    }

    /**
     * @brief Convert an ACMX2 virtual key code to an SDL_Keycode.
     *
     * Direct keyboard codes (e.g. 262 → SDLK_RIGHT) are mapped;
     * virtual knob codes 504–513 return SDLK_UNKNOWN because they
     * are handled directly in pollMidi().
     *
     * @param code Virtual key code from the MIDI map.
     * @return Corresponding SDL_Keycode, or SDLK_UNKNOWN.
     */
    SDL_Keycode midiKeyToSDL(int code) {
        switch (code) {
        case 262: return SDLK_RIGHT;
        case 263: return SDLK_LEFT;
        case 264: return SDLK_DOWN;
        case 265: return SDLK_UP;
        case 266: return SDLK_PAGEUP;
        case 267: return SDLK_PAGEDOWN;
        case 269: return SDLK_END;
        case 32:  return SDLK_SPACE;
        case 44:  return SDLK_COMMA;
        case 45:  return SDLK_MINUS;
        case 46:  return SDLK_PERIOD;
        case 47:  return SDLK_SLASH;
        case 61:  return SDLK_EQUALS;
        case 65:  return SDLK_a;
        case 66:  return SDLK_b;
        case 68:  return SDLK_d;
        case 71:  return SDLK_g;
        case 70:  return SDLK_f;
        case 72:  return SDLK_h;
        case 76:  return SDLK_l;
        case 78:  return SDLK_n;
        case 80:  return SDLK_p;
        case 82:  return SDLK_r;
        case 83:  return SDLK_s;
        case 75:  return SDLK_k;
        case 87:  return SDLK_w;
        // Virtual codes 504-513 handled directly in pollMidi
        case 504: case 505: case 506: case 507: case 508: case 509:
        case 510: case 511: case 512: case 513:
            return SDLK_UNKNOWN;
        default:  return SDLK_UNKNOWN;
        }
    }

    /**
     * @brief Synthesise an SDL_KEYUP event and dispatch it through event().
     * @param key  The SDL_Keycode to inject.
     * @param win  The GL window that receives the event.
     */
    void injectKey(SDL_Keycode key, gl::GLWindow *win) {
        SDL_Event ev{};
        ev.type = SDL_KEYUP;
        ev.key.keysym.sym = key;
        ev.key.state = SDL_RELEASED;
        event(win, ev);
    }

    /**
     * @brief Drain all pending MIDI messages and dispatch mapped actions.
     *
     * Called at the top of every draw() frame.  Button presses fire
     * immediately via injectKey(); continuous knobs update `knobState`
     * and are rate-limited by a velocity-sensitive frame-skip counter.
     * Knob distance from centre (64) controls speed: far = every frame,
     * near-centre = every ~16 frames.  Pitch/Yaw/Roll knobs use delta
     * from the previous value to determine direction.
     *
     * @param win GL window for injecting key events.
     */
    void pollMidi(gl::GLWindow *win) {
        if (!midiIn || !midiOpen) return;
        // Drain all pending MIDI messages and update knob state
        std::vector<unsigned char> msg;
        while (true) {
            midiIn->getMessage(&msg);
            if (msg.size() < 3) break;
            for (const auto &mc : midiCodes) {
                if (msg[0] == mc.b0 && msg[1] == mc.b1) {
                    if (mc.key2 != 0) {
                        // Knob: store latest value, handled below
                        knobState[{mc.b0, mc.b1}] = msg[2];
                    } else if (msg[2] == mc.b2) {
                        // Button: fire immediately on exact match
                        SDL_Keycode k = midiKeyToSDL(mc.key1);
                        if (k != SDLK_UNKNOWN) {
                            injectKey(k, win);
                            lastMidiButton = midiKeyName(mc.key1);
                            lastMidiButtonTime = std::chrono::steady_clock::now();
                        } else if (mc.key1 == 510) {
                            cameraRotationSpeed += 0.5f;
                            if (cameraRotationSpeed > 50.0f) cameraRotationSpeed = 50.0f;
                            mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                            fflush(stdout);
                            lastMidiButton = "RotSpdUp";
                            lastMidiButtonTime = std::chrono::steady_clock::now();
                        } else if (mc.key1 == 511) {
                            cameraRotationSpeed -= 0.5f;
                            if (cameraRotationSpeed < 0.5f) cameraRotationSpeed = 0.5f;
                            mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                            fflush(stdout);
                            lastMidiButton = "RotSpdDn";
                            lastMidiButtonTime = std::chrono::steady_clock::now();
                        }
                    }
                }
            }
        }
        // Fire keys for knobs with velocity-sensitive rate.
        // Distance from center (64) controls speed:
        //   dist 1-5  -> fire every 16 frames (slowest)
        //   dist 63   -> fire every frame (fastest)
        for (const auto &mc : midiCodes) {
            if (mc.key2 == 0) continue;
            auto key = std::make_pair(mc.b0, mc.b1);
            auto it = knobState.find(key);
            if (it == knobState.end()) continue;
            unsigned char val = it->second;
            // Slider knobs: map CC value (0-127) directly to 0.0-1.0
            if (mc.key1 >= 600 && mc.key1 <= 606 && (mc.key1 % 2 == 0)) {
                int idx = (mc.key1 - 600) / 2;
                library.setMidiSlider(idx, static_cast<float>(val) / 127.0f);
                continue;
            }
            if (val == 64) continue; // dead zone at center
            int dist = (val > 64) ? (val - 64) : (64 - val); // 1..64
            // Map distance to frame skip: max dist (63-64) = 1 (every frame),
            // min dist (1) = 16 (every 16th frame)
            int skipFrames = std::max(1, 17 - (dist * 16 / 63));
            int &counter = knobFrameCount[key];
            if (++counter >= skipFrames) {
                counter = 0;
                // For Pitch/Yaw knobs, use delta from previous value
                // so turning the physical knob one direction always
                // rotates the same way (no ping-pong at end-stops).
                bool useDelta = (mc.key1 == 506 || mc.key1 == 508 || mc.key1 == 512);
                int activeKey;
                if (useDelta) {
                    auto prevIt = knobPrevValue.find(key);
                    unsigned char prev = (prevIt != knobPrevValue.end()) ? prevIt->second : 64;
                    activeKey = (val >= prev) ? mc.key1 : mc.key2;
                    knobPrevValue[key] = val;
                } else {
                    activeKey = (val > 64) ? mc.key1 : mc.key2;
                }
                // Handle direct-action virtual codes
                if (activeKey == 504) {
                    library.incTimeSpeed(0.1f);
                } else if (activeKey == 505) {
                    library.decTimeSpeed(0.1f);
                } else if (activeKey == 506) {
                    modelRotX += cameraRotationSpeed * 0.3f;
                    modelRotX = fmod(modelRotX, 360.0f);
                    mx::system_out << "acmx2: Model RotX: " << modelRotX << "\n";
                    fflush(stdout);
                } else if (activeKey == 507) {
                    modelRotX -= cameraRotationSpeed * 0.33f;
                    modelRotX = fmod(modelRotX + 360.0f, 360.0f);
                    mx::system_out << "acmx2: Model RotX: " << modelRotX << "\n";
                    fflush(stdout);
                } else if (activeKey == 508) {
                    modelRotY += cameraRotationSpeed * 0.3f;
                    modelRotY = fmod(modelRotY, 360.0f);
                    mx::system_out << "acmx2: Model RotY: " << modelRotY << "\n";
                    fflush(stdout);
                } else if (activeKey == 509) {
                    modelRotY -= cameraRotationSpeed * 0.3f;
                    modelRotY = fmod(modelRotY + 360.0f, 360.0f);
                    mx::system_out << "acmx2: Model RotY: " << modelRotY << "\n";
                    fflush(stdout);
                } else if (activeKey == 512) {
                    modelRotZ += cameraRotationSpeed * 0.3f;
                    modelRotZ = fmod(modelRotZ, 360.0f);
                    mx::system_out << "acmx2: Model RotZ: " << modelRotZ << "\n";
                    fflush(stdout);
                } else if (activeKey == 513) {
                    modelRotZ -= cameraRotationSpeed * 0.3f;
                    modelRotZ = fmod(modelRotZ + 360.0f, 360.0f);
                    mx::system_out << "acmx2: Model RotZ: " << modelRotZ << "\n";
                    fflush(stdout);
                } else if (activeKey == 514) {
                    modelRenderScale += 0.05f;
                    mx::system_out << "acmx2: Model scale increased to " << modelRenderScale << "\n";
                    fflush(stdout);
                } else if (activeKey == 515) {
                    modelRenderScale -= 0.05f;
                    if (modelRenderScale < 0.05f) modelRenderScale = 0.05f;
                    mx::system_out << "acmx2: Model scale decreased to " << modelRenderScale << "\n";
                    fflush(stdout);
                } else {
                    SDL_Keycode k = (val > 64)
                        ? midiKeyToSDL(mc.key1)
                        : midiKeyToSDL(mc.key2);
                    if (k != SDLK_UNKNOWN) injectKey(k, win);
                }
            }
        }
    }

    /// @brief Close the MIDI input port and free the RtMidiIn instance.
    void cleanupMidi() {
        if (midiIn) {
            if (midiOpen) midiIn->closePort();
            delete midiIn;
            midiIn = nullptr;
            midiOpen = false;
        }
    }

    /**
     * @brief Draw the MIDI status overlay (active indicator, knob bars, last button).
     *
     * Shows a green "MIDI Active" header, a bar graph for each mapped
     * knob with its current value and direction label, and the last
     * pressed button name with a 2-second fade-out.
     *
     * @param win    GL window (text rendering context).
     * @param font   Font for overlay text.
     * @param startY Y-coordinate to begin drawing.
     */
    void drawMidiOverlay(gl::GLWindow *win, mx::Font &font, int startY) {
        if (!midiOpen || midiCodes.empty()) return;
        int y = startY;
        win->text.setColor({0, 255, 0, 255});
        win->text.printText_Blended(font, 10, y, "MIDI Active");
        y += 25;
        // Find max label width for alignment
        size_t maxLabelLen = 0;
        for (const auto &mc : midiCodes) {
            if (mc.key2 == 0) continue;
            size_t len = std::string(midiKeyName(mc.key1)).size() + 1 + std::string(midiKeyName(mc.key2)).size();
            if (len > maxLabelLen) maxLabelLen = len;
        }
        // Show knob states
        win->text.setColor({0, 255, 0, 255});
        for (const auto &mc : midiCodes) {
            if (mc.key2 == 0) continue;
            auto it = knobState.find({mc.b0, mc.b1});
            unsigned char val = (it != knobState.end()) ? it->second : 64;
            const char *dir = (val == 64) ? "--" : (val > 64) ? midiKeyName(mc.key1) : midiKeyName(mc.key2);
            int barLen = 20;
            int pos = (val * barLen) / 127;
            std::string bar(barLen, '-');
            bar[barLen / 2] = '|';
            if (pos < barLen) bar[pos] = '#';
            std::string label = std::string(midiKeyName(mc.key1)) + "/" + midiKeyName(mc.key2);
            // Pad label to align bars
            while (label.size() < maxLabelLen) label += ' ';
            std::ostringstream oss;
            oss << label << " [" << bar << "] " << std::setw(3) << static_cast<int>(val) << " " << dir;
            win->text.printText_Blended(font, 10, y, oss.str());
            y += 22;
        }
        // Show last button press (fade after 2 seconds)
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - lastMidiButtonTime).count();
        if (!lastMidiButton.empty() && elapsed < 2000) {
            int alpha = (elapsed < 1500) ? 255 : 255 - static_cast<int>((elapsed - 1500) * 255 / 500);
            win->text.setColor({0, 255, 0, static_cast<unsigned char>(std::max(0, alpha))});
            win->text.printText_Blended(font, 10, y, "Button: " + lastMidiButton);
            y += 22;
        }
    }
#endif
    bool isPaused = false;
    bool isFrozen = false;
    bool shaderLocked = false;
    GLuint pboIds[2] = {0, 0};
#ifdef ACMX2_WITH_CUDA
    cudaGraphicsResource *recordCudaPboResources[2] = {nullptr, nullptr};
#endif
    int pboIndex = 0;
    int pboNextIndex = 1;
    SnapshotThreadPool snapshot_pool{2};
    TextureUploader tex_uploader;

  public:
    void requestStop() { running = false; }
    void requestStopNoMux() {
        skip_audio_mux_on_exit = true;
        running = false;
    }
    bool needsAsyncShutdown() {
        return !skip_audio_mux_on_exit.load() && (needsMux() || needsTransferAudio() || needsFileAudioMux());
    }

    // --- HDR pipeline helpers ------------------------------------------------
    //
    // These functions are no-ops when @ref input_is_hdr is false, so the
    // SDR frame path remains untouched. When HDR is active, they build and
    // run the decode/encode fullscreen passes that wrap the existing user
    // shader chain.
    // -------------------------------------------------------------------------

    /**
     * @brief Lazily allocate the HDR decode/encode textures, FBOs and shader
     *        programs. Idempotent.
     * @param w Target width (matches shader pass width).
     * @param h Target height.
     */
    void ensureHdrResources(int w, int h) {
        if (!input_is_hdr) {
            return;
        }
        const bool resize_needed = (hdr_resource_w != w || hdr_resource_h != h);
        if (hdr_linear_video_texture == 0) {
            glGenTextures(1, &hdr_linear_video_texture);
            glBindTexture(GL_TEXTURE_2D, hdr_linear_video_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0,
                         GL_RGBA, GL_HALF_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else if (resize_needed) {
            glBindTexture(GL_TEXTURE_2D, hdr_linear_video_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0,
                         GL_RGBA, GL_HALF_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        if (hdr_linear_video_fbo == 0) {
            glGenFramebuffers(1, &hdr_linear_video_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, hdr_linear_video_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, hdr_linear_video_texture, 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                mx::system_err << "acmx2: HDR linear-video FBO incomplete\n";
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        if (hdr_encoded_texture == 0) {
            glGenTextures(1, &hdr_encoded_texture);
            glBindTexture(GL_TEXTURE_2D, hdr_encoded_texture);
            // GL_RGBA16 = 16-bit unsigned normalised per channel. Readback
            // as GL_UNSIGNED_SHORT gives us 16-bit PQ code values that the
            // writer quantises to 10-bit for P010.
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_SHORT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else if (resize_needed) {
            glBindTexture(GL_TEXTURE_2D, hdr_encoded_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_SHORT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        if (hdr_encoded_fbo == 0) {
            glGenFramebuffers(1, &hdr_encoded_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, hdr_encoded_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, hdr_encoded_texture, 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                mx::system_err << "acmx2: HDR encoded FBO incomplete\n";
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        if (hdr_decode_shader.id() == 0) {
            if (!hdr_decode_shader.loadProgramFromText(kHdrVertPassthrough, kHdrDecodeFrag)) {
                mx::system_err << "acmx2: failed to compile HDR decode shader\n";
            }
        }
        if (hdr_encode_shader.id() == 0) {
            if (!hdr_encode_shader.loadProgramFromText(kHdrVertPassthrough, kHdrEncodeFrag)) {
                mx::system_err << "acmx2: failed to compile HDR encode shader\n";
            }
        }
        if (display_flip_shader.id() == 0) {
            if (!display_flip_shader.loadProgramFromText(kDisplayVertFlip, kDisplayFragPassthrough)) {
                mx::system_err << "acmx2: failed to compile display flip shader\n";
            }
        }
        hdr_resource_w = w;
        hdr_resource_h = h;
    }

    /**
     * @brief Run the PQ/HLG -> linear BT.2020 decode pass, sampling
     *        @c camera_texture and writing into @c hdr_linear_video_texture.
     *
     * The sprite quad is re-used so the pixel coverage matches the rest of
     * the pipeline. After this call, user shaders should sample
     * @c hdr_linear_video_texture (via @c sampleSourceTextureForHdr()).
     */
    void runHdrDecodePass(int w, int h) {
        if (!input_is_hdr || hdr_decode_shader.id() == 0) {
            return;
        }
        const int transfer_mode =
            (input_hdr_trc == AVCOL_TRC_ARIB_STD_B67) ? 2 : 1;

        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
        GLint prev_viewport[4] = {0};
        glGetIntegerv(GL_VIEWPORT, prev_viewport);

        glBindFramebuffer(GL_FRAMEBUFFER, hdr_linear_video_fbo);
        glViewport(0, 0, w, h);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glClear(GL_COLOR_BUFFER_BIT);

        hdr_decode_shader.useProgram();
        hdr_decode_shader.setUniform("mv_matrix", glm::mat4(1.0f));
        hdr_decode_shader.setUniform("proj_matrix", glm::mat4(1.0f));
        glUniform1i(glGetUniformLocation(hdr_decode_shader.id(), "samp"), 0);
        glUniform1i(glGetUniformLocation(hdr_decode_shader.id(), "transfer"), transfer_mode);
        sprite.setShader(&hdr_decode_shader);
        sprite.setName("samp");
        sprite.draw(camera_texture, 0, 0, w, h);

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
    }

    /**
     * @brief Run the linear BT.2020 -> PQ/HLG encode pass from @p src_tex into
     *        @c hdr_encoded_texture. Call this just before the PBO readback.
     */
    void runHdrEncodePass(GLuint src_tex, int w, int h) {
        if (!input_is_hdr || hdr_encode_shader.id() == 0) {
            return;
        }
        const int transfer_mode =
            (input_hdr_trc == AVCOL_TRC_ARIB_STD_B67) ? 2 : 1;

        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
        GLint prev_viewport[4] = {0};
        glGetIntegerv(GL_VIEWPORT, prev_viewport);

        glBindFramebuffer(GL_FRAMEBUFFER, hdr_encoded_fbo);
        glViewport(0, 0, w, h);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glClear(GL_COLOR_BUFFER_BIT);

        hdr_encode_shader.useProgram();
        hdr_encode_shader.setUniform("mv_matrix", glm::mat4(1.0f));
        hdr_encode_shader.setUniform("proj_matrix", glm::mat4(1.0f));
        glUniform1i(glGetUniformLocation(hdr_encode_shader.id(), "samp"), 0);
        glUniform1i(glGetUniformLocation(hdr_encode_shader.id(), "transfer"), transfer_mode);
        sprite.setShader(&hdr_encode_shader);
        sprite.setName("samp");
        sprite.draw(src_tex, 0, 0, w, h);

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
    }

    /**
     * @brief Upload a 16-bit-per-channel RGBA frame (from @c FFMpegVideoReader::readHdr())
     *        into @c camera_texture. The texture must have been allocated as
     *        GL_RGBA16 (done during HDR resource init).
     */
    void uploadHdrFrame(const cv::Mat &rgba16) {
        if (rgba16.empty()) return;
        glBindTexture(GL_TEXTURE_2D, camera_texture);
        if (hdr_upload_tex_w != rgba16.cols || hdr_upload_tex_h != rgba16.rows) {
            // The decoded HDR frame can differ from the output resolution
            // when --resolution is used. Ensure camera_texture matches the
            // source frame dimensions and 16-bit normalized format before
            // sub-image upload; otherwise drivers may crash on invalid
            // glTexSubImage2D parameters.
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16,
                         rgba16.cols, rgba16.rows, 0,
                         GL_RGBA, GL_UNSIGNED_SHORT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            hdr_upload_tex_w = rgba16.cols;
            hdr_upload_tex_h = rgba16.rows;
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        glPixelStorei(GL_UNPACK_ROW_LENGTH,
                      static_cast<GLint>(rgba16.step / (4 * sizeof(uint16_t))));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        rgba16.cols, rgba16.rows,
                        GL_RGBA, GL_UNSIGNED_SHORT, rgba16.ptr());
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * @brief One-time conversion of existing GL resources from SDR (8-bit)
     *        to HDR (16-bit) formats. Called the first time a video is
     *        detected as HDR — after @c load() / @c setupCaptureFBO() have
     *        already allocated SDR-format textures. Re-specifies the
     *        existing texture objects so no IDs change and no callers break.
     *
     *  - @c camera_texture          : GL_RGBA16  (PQ/HLG-encoded source)
     *  - @c fboTexture              : GL_RGBA16F (linear BT.2020 intermediate)
     *  - @c crossfadeTexture        : GL_RGBA16F (if already created)
     *  - @c crossfadePrevTexture    : GL_RGBA16F (if already created)
     *  - @c passTexture[0..1]       : deferred; allocated on demand by the
     *                                 shader-pass branch, which checks
     *                                 @ref input_is_hdr and chooses the
     *                                 correct internal format at that time.
     *  - HDR decode/encode programs + linear/encoded textures: allocated
     *    via @ref ensureHdrResources().
     */
    void convertResourcesToHdr(int w, int h) {
        if (!input_is_hdr) return;

        // Re-spec camera_texture as 16-bit normalised. Keep the same GL
        // name so callers/sprites that already cached the ID stay valid.
        if (camera_texture != 0) {
            glBindTexture(GL_TEXTURE_2D, camera_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_SHORT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            hdr_upload_tex_w = w;
            hdr_upload_tex_h = h;
        }

        // Re-spec fboTexture (colour attachment of captureFBO) as 16F.
        if (fboTexture != 0) {
            glBindTexture(GL_TEXTURE_2D, fboTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0,
                         GL_RGBA, GL_HALF_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // crossfade textures may or may not have been allocated yet.
        auto respecFloat = [&](GLuint tex) {
            if (tex == 0) return;
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0,
                         GL_RGBA, GL_HALF_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
        };
        respecFloat(crossfadeTexture);
        respecFloat(crossfadePrevTexture);

        // If pass textures were already allocated (unusual at this point,
        // but possible on re-entry), re-spec them too.
        for (int p = 0; p < 2; ++p) {
            respecFloat(passTexture[p]);
        }

        ensureHdrResources(w, h);
    }

    /**
     * @brief HDR readback: run the encode pass and read 16-bit PQ-encoded
     *        RGB pixels back into @p out. Output size is 8*w*h bytes
     *        (uint16_t per channel, RGBA).
     *
     * @param src_linear_tex  Texture holding the final linear BT.2020
     *                        result (typically @c fboTexture after the
     *                        shader chain).
     * @param w,h             Frame dimensions.
     * @param out             Output byte vector, resized to 8*w*h.
     */
    void hdrReadback(GLuint src_linear_tex, int w, int h,
                     std::vector<unsigned char> &out) {
        runHdrEncodePass(src_linear_tex, w, h);
        const size_t bytes = static_cast<size_t>(w) * h * 8;
        out.resize(bytes);
        glBindTexture(GL_TEXTURE_2D, hdr_encoded_texture);
        glPixelStorei(GL_PACK_ALIGNMENT, 2);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_SHORT, out.data());
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * @brief Construct the rendering object from parsed CLI arguments.
     *
     * Copies capture parameters, opens the audio subsystem (optional),
     * initialises GPU filter state (CUDA malloc for pointer/filter lists),
     * sets up the shader pass list, loads the MIDI map file, and applies
     * the playlist / duration settings.  No OpenGL calls are made here
     * because the GL context is not yet current—those are deferred to load().
     *
     * @param args Parsed command-line arguments.
     */
    ACView(const MXArguments &args)
        : crf{args.crf},
          encode_opts{args.encode_opts},
          prefix_path{args.prefix_path},
          filename{args.filename},
          ofilename{args.ofilename},
          graphic{args.graphic_file},
          camera_index{args.camera_device},
          flib{args.slib},
          sizev{args.sizev},
          sizec{args.csize},
          fps{args.fps_value},
          repeat{args.repeat},
          full{args.full},
          frame_cache{8},
          texture_cache{args.cache},
          cache_delay{args.cache_delay},
          copy_audio{args.copy_audio},
          gpu_cuda_device{args.cuda_device},
          silent_mode{args.silent},
          no_drop_mode{args.no_drop},
          use_shader_cache_flag{args.use_shader_cache},
          flip_output{args.flip_output},
          display_filter{args.display_filter} {
        if (!args.watermark_text.empty()) {
            enableWatermark = true;
            watermark_text = args.watermark_text;
        }
        watermark_r = std::clamp(args.watermark_r, 0, 255);
        watermark_g = std::clamp(args.watermark_g, 0, 255);
        watermark_b = std::clamp(args.watermark_b, 0, 255);
#ifdef AUDIO_ENABLED
        audio_input_device = args.audio_input;
        audio_output_device = args.audio_output;
        audio_record_file = args.record_audio_file;
        if (!args.audio_file.empty()) {
            if (file_audio_open(args.audio_file)) {
                audio_is_enabled = true;
                file_audio_mode = true;
                audio_file_path = args.audio_file;
                audio_trunc_mode = args.audio_trunc;
                set_sense(args.audio_sensitivty);
                spectrumTex.init();
                mx::system_out << "acmx2: File audio enabled from: " << args.audio_file << "\n";
                mx::system_out << "acmx2: FFT spectrum texture initialised ("
                               << get_fft_bin_count() << " bins on GL_TEXTURE"
                               << SpectrumTexture::SPECTRUM_TEXTURE_UNIT << ")\n";
            } else {
                mx::system_err << "acmx2: Error could not open audio file: " << args.audio_file << "\n";
            }
        } else if (args.audio_enabled) {
            if (init_audio(args.audio_channels, args.audio_sensitivty, audio_input_device, audio_output_device) != 0) {
                mx::system_err << "acmx2: Error could not initalize audio\n";
            } else {
                audio_is_enabled = true;
                set_record_gain(args.record_gain);
                spectrumTex.init();
                mx::system_out << "acmx2: FFT spectrum texture initialised ("
                               << get_fft_bin_count() << " bins on GL_TEXTURE"
                               << SpectrumTexture::SPECTRUM_TEXTURE_UNIT << ")\n";
            }
        }

#endif
        library.is3D(args.is3d);
        library.setTimeSpeed(args.time_speed);
        is3d_enabled = args.is3d;
        m_file = args.model_file;

        gpu_filter_enabled = args.gpu_filter_enabled;
#ifdef ACMX2_WITH_CUDA
        if (gpu_filter_enabled && !args.gpu_filter_indices.empty()) {
            for (int idx : args.gpu_filter_indices) {
                if (idx >= 0 && idx < ac_gpu::AC_FILTER_MAX) {
                    gpu_filters.push_back({idx, ac_gpu::filters[idx].name});
                    mx::system_out << "acmx2: GPU filter added: " << ac_gpu::filters[idx].name << " (index " << idx << ")\n";
                }
            }
            gpu_current_filter_index = args.gpu_filter_indices[0];
            gpu_frame_buffer = std::make_unique<ac_gpu::DynamicFrameBuffer>(args.gpu_frame_buffer_size);
            CHECK_CUDA(cudaMalloc(&d_ptrList, args.gpu_frame_buffer_size * sizeof(unsigned char *)));
            mx::system_out << "acmx2: GPU filtering enabled with " << gpu_filters.size() << " filter(s)\n";
        }
#else
        if (gpu_filter_enabled) {
            mx::system_err << "acmx2: GPU filters requested but this build has no CUDA support; disabling.\n";
            gpu_filter_enabled = false;
        }
#endif
        counter_disabled = args.disable_counter;
        use_yuv = args.use_yuv;

        if (args.shader_pass_enabled && !args.shader_pass_list.empty()) {
            shader_pass_list = args.shader_pass_list;
            shader_pass_enabled = true;
            mx::system_out << "acmx2: Shader pass list enabled with " << shader_pass_list.size() << " shader(s)\n";
            for (int idx : shader_pass_list) {
                mx::system_out << "  - Shader index: " << idx << "\n";
            }
            fflush(stdout);
        }
        playlist_file = args.playlist_file;
        autopilot_frames = args.autopilot_frames;
        duration_limit = args.duration;
        crossfadeDuration = args.cross_fade_duration;
#ifdef MIDI_ENABLED
        if (!args.midi_map_file.empty()) {
            initMidi(args.midi_map_file, args.midi_device);
        }
#endif
    }

    bool is3d_enabled = false;

    bool gpu_filter_enabled = false;
    std::vector<ac_gpu::Filter> gpu_filters;
    int gpu_current_filter_index = 0;
    std::unique_ptr<ac_gpu::DynamicFrameBuffer> gpu_frame_buffer;
#ifdef ACMX2_WITH_CUDA
    cv::cuda::GpuMat gpuWorkingBuffer;
#endif
    cv::Mat gpuFilteredFrame;
    unsigned char **d_ptrList = nullptr;
    ac_gpu::GPUFilter *d_filterList = nullptr;
    bool gpu_filtersChanged = true;
    float gpu_alpha = 1.0f;
    int gpu_alpha_dir = 1;
    int gpu_square_size = 8;
    int gpu_frame_index = 0;
    int gpu_frame_dir = 1;
    std::vector<int> shader_pass_list;
    bool shader_pass_enabled = false;
    std::string cached_shader_name;

    struct PlaylistNode {
        std::string name;
        std::vector<int> shader_indices;
    };
    std::vector<PlaylistNode> playlist_tree;
    std::vector<int> playlist_indices;
    int playlist_index = 0;
    bool playlist_enabled = false;
    std::string playlist_file;
    int autopilot_frames = 0;            ///< Frames between random switches in autopilot mode (0 = unset).
    bool autopilot_enabled = false;       ///< Toggle autopilot via SDLK_j when playlist is enabled.
    bool autopilot_sequential = false;    ///< When true, autopilot advances through the playlist in order instead of randomly (toggle via SDLK_y).
    int autopilot_counter = 0;            ///< Frames elapsed since last autopilot switch.
    std::mt19937 autopilot_rng{std::random_device{}()};
    std::vector<int> saved_pass_list;
    bool saved_pass_enabled = false;
    double duration_limit = 0.0;

    bool random_multipass_mode = false;
    std::vector<int> saved_pass_list_before_random;
    bool saved_pass_enabled_before_random = false;
    size_t saved_shader_index_before_random = 0;

    void generateRandomMultipass(gl::GLWindow *win) {
        static std::mt19937 rng(std::random_device{}());
        size_t shader_count = library.size();
        if (shader_count == 0) return;
        std::uniform_int_distribution<int> count_dist(1, 5);
        std::uniform_int_distribution<int> shader_dist(0, static_cast<int>(shader_count) - 1);
        int chain_len = count_dist(rng);
        beginCrossfade(win);
        shader_pass_list.clear();
        for (int i = 0; i < chain_len; ++i) {
            shader_pass_list.push_back(shader_dist(rng));
        }
        shader_pass_enabled = true;
        if (is3d_enabled)
            cube.setShaderProgram(library.shader());
        sprite.setShader(library.shader());
        updateShaderNameCache();
        mx::system_out << "acmx2: Random multipass [";
        for (size_t i = 0; i < shader_pass_list.size(); ++i) {
            mx::system_out << library.getShaderNameByIndex(shader_pass_list[i]);
            if (i + 1 < shader_pass_list.size()) mx::system_out << ", ";
        }
        mx::system_out << "]\n";
        fflush(stdout);
    }

    void generateRandomMultipassShort(gl::GLWindow *win) {
        static std::mt19937 rng(std::random_device{}());
        size_t shader_count = library.size();
        if (shader_count == 0) return;
        std::uniform_int_distribution<int> shader_dist(0, static_cast<int>(shader_count) - 1);
        beginCrossfade(win);
        shader_pass_list.clear();
        for (int i = 0; i < 2; ++i) {
            shader_pass_list.push_back(shader_dist(rng));
        }
        shader_pass_enabled = true;
        if (is3d_enabled)
            cube.setShaderProgram(library.shader());
        sprite.setShader(library.shader());
        updateShaderNameCache();
        mx::system_out << "acmx2: Short random multipass [";
        for (size_t i = 0; i < shader_pass_list.size(); ++i) {
            mx::system_out << library.getShaderNameByIndex(shader_pass_list[i]);
            if (i + 1 < shader_pass_list.size()) mx::system_out << ", ";
        }
        mx::system_out << "]\n";
        fflush(stdout);
    }

    void generateRandomMultipassLong(gl::GLWindow *win) {
        static std::mt19937 rng(std::random_device{}());
        size_t shader_count = library.size();
        if (shader_count == 0) return;
        std::uniform_int_distribution<int> count_dist(1, 10);
        std::uniform_int_distribution<int> shader_dist(0, static_cast<int>(shader_count) - 1);
        int chain_len = count_dist(rng);
        beginCrossfade(win);
        shader_pass_list.clear();
        for (int i = 0; i < chain_len; ++i) {
            shader_pass_list.push_back(shader_dist(rng));
        }
        shader_pass_enabled = true;
        if (is3d_enabled)
            cube.setShaderProgram(library.shader());
        sprite.setShader(library.shader());
        updateShaderNameCache();
        mx::system_out << "acmx2: Long random multipass [";
        for (size_t i = 0; i < shader_pass_list.size(); ++i) {
            mx::system_out << library.getShaderNameByIndex(shader_pass_list[i]);
            if (i + 1 < shader_pass_list.size()) mx::system_out << ", ";
        }
        mx::system_out << "]\n";
        fflush(stdout);
    }

    /**
     * @brief Refresh the cached shader name string for the HUD overlay.
     *
     * Called whenever the active shader or pass list changes so that
     * the overlay does not need to rebuild the string every frame.
     */
    void updateShaderNameCache() {
        cached_shader_name = shader_pass_enabled
                                 ? library.getFullShaderName(shader_pass_list)
                                 : library.getFullShaderName();
    }

    /**
     * @brief Pick a random entry from the active playlist and apply it.
     *
     * Used by autopilot mode. Mirrors the right-arrow advance logic but
     * selects a uniformly random index. No-op unless playlist mode is
     * active and at least one entry exists.
     *
     * @param win Active window (used for crossfade animation).
     */
    void autopilotRandomSwitch(gl::GLWindow *win) {
        if (!playlist_enabled)
            return;
        if (shaderLocked)
            return;
        if (!playlist_tree.empty()) {
            const int n = static_cast<int>(playlist_tree.size());
            if (n <= 0) return;
            std::uniform_int_distribution<int> dist(0, n - 1);
            int r = dist(autopilot_rng);
            if (n > 1 && r == playlist_index)
                r = (r + 1) % n;
            beginCrossfade(win);
            playlist_index = r;
            const auto &node = playlist_tree[playlist_index];
            shader_pass_list = node.shader_indices;
            shader_pass_enabled = !shader_pass_list.empty();
            if (is3d_enabled)
                cube.setShaderProgram(library.shader());
            sprite.setShader(library.shader());
            updateShaderNameCache();
            mx::system_out << "acmx2: Autopilot -> Node: " << node.name
                           << " [" << node.shader_indices.size() << " shaders] ("
                           << (playlist_index + 1) << "/" << n << ")\n";
            fflush(stdout);
        } else if (!playlist_indices.empty()) {
            const int n = static_cast<int>(playlist_indices.size());
            std::uniform_int_distribution<int> dist(0, n - 1);
            int r = dist(autopilot_rng);
            if (n > 1 && r == playlist_index)
                r = (r + 1) % n;
            beginCrossfade(win);
            playlist_index = r;
            library.setIndex(playlist_indices[playlist_index]);
            if (is3d_enabled)
                cube.setShaderProgram(library.shader());
            sprite.setShader(library.shader());
            updateShaderNameCache();
            mx::system_out << "acmx2: Autopilot -> Playlist [" << (playlist_index + 1) << "/" << n << "]\n";
            fflush(stdout);
        }
    }

    /**
     * @brief Advance to the next entry in the active playlist (wrapping).
     *
     * Used by sequential autopilot mode. Steps @c playlist_index forward by
     * one and wraps to zero when the end is reached, applying the same
     * crossfade and shader-pass updates as @ref autopilotRandomSwitch.
     *
     * @param win Active window (used for crossfade animation).
     */
    void autopilotSequentialAdvance(gl::GLWindow *win) {
        if (!playlist_enabled)
            return;
        if (shaderLocked)
            return;
        if (!playlist_tree.empty()) {
            const int n = static_cast<int>(playlist_tree.size());
            if (n <= 0) return;
            beginCrossfade(win);
            playlist_index = (playlist_index + 1) % n;
            const auto &node = playlist_tree[playlist_index];
            shader_pass_list = node.shader_indices;
            shader_pass_enabled = !shader_pass_list.empty();
            if (is3d_enabled)
                cube.setShaderProgram(library.shader());
            sprite.setShader(library.shader());
            updateShaderNameCache();
            mx::system_out << "acmx2: Autopilot (sequential) -> Node: " << node.name
                           << " [" << node.shader_indices.size() << " shaders] ("
                           << (playlist_index + 1) << "/" << n << ")\n";
            fflush(stdout);
        } else if (!playlist_indices.empty()) {
            const int n = static_cast<int>(playlist_indices.size());
            beginCrossfade(win);
            playlist_index = (playlist_index + 1) % n;
            library.setIndex(playlist_indices[playlist_index]);
            if (is3d_enabled)
                cube.setShaderProgram(library.shader());
            sprite.setShader(library.shader());
            updateShaderNameCache();
            mx::system_out << "acmx2: Autopilot (sequential) -> Playlist ["
                           << (playlist_index + 1) << "/" << n << "]\n";
            fflush(stdout);
        }
    }

    /**
     * @brief Lazily create the crossfade FBO and its two textures.
     *
     * Allocates a framebuffer object (@c crossfadeFBO) with an RGBA colour
     * attachment (@c crossfadeTexture) and a second texture
     * (@c crossfadePrevTexture) used to store the previous frame.
     * Subsequent calls are no-ops once the FBO has been created.
     *
     * @param width  Framebuffer width in pixels.
     * @param height Framebuffer height in pixels.
     * @throws mx::Exception if the framebuffer is incomplete.
     */
    void ensureCrossfadeFBO(int width, int height) {
        if (crossfadeFBO)
            return;
        // In HDR mode the pipeline stores LINEAR BT.2020 light in RGBA16F
        // (values frequently exceed 1.0). Allocating an 8-bit UNORM target
        // here would clamp HDR highlights to 1.0 and crush the crossfade
        // result. Match the rest of the HDR intermediate chain.
        const GLint cf_internal = input_is_hdr ? GL_RGBA16F : GL_RGBA;
        const GLenum cf_type    = input_is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
        glGenFramebuffers(1, &crossfadeFBO);
        glGenTextures(1, &crossfadeTexture);
        glBindTexture(GL_TEXTURE_2D, crossfadeTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, cf_internal, width, height, 0, GL_RGBA, cf_type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, crossfadeFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, crossfadeTexture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw mx::Exception("acmx2: crossfade framebuffer is not complete");
        }
        glGenTextures(1, &crossfadePrevTexture);
        glBindTexture(GL_TEXTURE_2D, crossfadePrevTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, cf_internal, width, height, 0, GL_RGBA, cf_type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    /**
     * @brief Start a crossfade transition from the current frame.
     *
     * Snapshots the current capture FBO contents into @c crossfadePrevTexture,
     * resets @c crossfadeAlpha to zero, and records the start time so that
     * applyCrossfade() can linearly interpolate over @c crossfadeDuration
     * seconds.
     *
     * @param win The GL window whose dimensions define the framebuffer size.
     */
    void beginCrossfade(gl::GLWindow *win) {
        ensureCrossfadeFBO(win->w, win->h);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, captureFBO);
        glBindTexture(GL_TEXTURE_2D, crossfadePrevTexture);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, win->w, win->h);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        crossfadeAlpha = 0.0f;
        crossfadeActive = true;
        crossfadeStartTime = std::chrono::steady_clock::now();
    }

    /**
     * @brief Render the crossfade blend for the current frame.
     *
     * Computes a linear alpha from elapsed time and @c crossfadeDuration,
     * then draws a full-screen quad into @c crossfadeFBO using the
     * @c crossfadeShader.  The shader mixes @c crossfadePrevTexture
     * (the old frame) with @p currentTexture (the new frame) via
     * the @c fade_alpha uniform.  When the transition completes
     * (@c crossfadeAlpha >= 1.0), @c crossfadeActive is set to false.
     *
     * @param win            The GL window (provides viewport dimensions).
     * @param currentTexture The texture containing the newly rendered frame.
     */
    void applyCrossfade(gl::GLWindow *win, GLuint currentTexture) {
        if (!crossfadeActive)
            return;
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - crossfadeStartTime).count();
        crossfadeAlpha = elapsed / crossfadeDuration;
        if (crossfadeAlpha >= 1.0f) {
            crossfadeAlpha = 1.0f;
            crossfadeActive = false;
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, crossfadeFBO);
        glViewport(0, 0, win->w, win->h);
        glClear(GL_COLOR_BUFFER_BIT);
        crossfadeShader.useProgram();
        crossfadeShader.setUniform("mv_matrix", glm::mat4(1.0f));
        crossfadeShader.setUniform("proj_matrix", glm::mat4(1.0f));
        crossfadeShader.setUniform("fade_alpha", crossfadeAlpha);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentTexture);
        crossfadeShader.setUniform("samp", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, crossfadePrevTexture);
        crossfadeShader.setUniform("prev_samp", 1);
        sprite.setShader(&crossfadeShader);
        sprite.setName("samp");
        sprite.draw(currentTexture, 0, 0, win->w, win->h);
        glActiveTexture(GL_TEXTURE0);
    }

    mx::Font overlayFont;
    mx::Font waterFont;
    std::chrono::steady_clock::time_point sessionStartTime;
    double displayFPS = 0.0;
    int fpsFrameCount = 0;
    std::chrono::steady_clock::time_point fpsLastTime;
    bool counter_disabled = false;

    /**
     * @brief Destroy the rendering object: stop threads, release GPU resources.
     *
     * Destruction order:
     * 1. Clean up MIDI (if enabled).
     * 2. Release TextureUploader (CUDA↔OpenGL PBO).
     * 3. Free CUDA filter arrays.
     * 4. Stop the capture thread (join).
     * 5. Flush any remaining PBO frames into the writer queue.
     * 6. Stop the writer thread (join), close the Writer, mux audio
     *    if applicable.
     * 7. Join the mux thread if it was started.
     * 8. Close the audio subsystem.
     * 9. Delete PBOs, FBOs, textures, depth buffer.
     * 10.Release the VideoCapture.
     */
    ~ACView() override {
#ifdef MIDI_ENABLED
        cleanupMidi();
#endif
        tex_uploader.cleanup();
#ifdef ACMX2_WITH_CUDA
        if (d_ptrList) {
            cudaFree(d_ptrList);
            d_ptrList = nullptr;
        }
        if (d_filterList) {
            cudaFree(d_filterList);
            d_filterList = nullptr;
        }
#endif
        gpu_frame_buffer.reset();

        stopCaptureThread();

        if (pboIds[0] && writer.is_open() && win_w > 0 && win_h > 0) {
            for (int i = 0; i < 2; i++) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
                GLubyte *src = static_cast<GLubyte *>(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));

                if (src) {
                    std::vector<unsigned char> pixels(win_w * win_h * 4);
                    std::memcpy(pixels.data(), src, pixels.size());
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

                    std::vector<unsigned char> flipped_pixels(win_w * win_h * 4);
                    for (int y = 0; y < win_h; ++y) {
                        int src_row_start = y * win_w * 4;
                        int dest_row_start = (win_h - 1 - y) * win_w * 4;
                        std::copy(pixels.begin() + src_row_start,
                                  pixels.begin() + src_row_start + (win_w * 4),
                                  flipped_pixels.begin() + dest_row_start);
                    }

                    FrameData fd;
                    fd.pixels = std::move(flipped_pixels);
                    fd.width = win_w;
                    fd.height = win_h;
                    fd.isSnapshot = false;

                    {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        frameQueue.push(std::move(fd));
                    }
                    queueCondVar.notify_one();
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (!isMuxing.load()) {
            bool shouldMux = !skip_audio_mux_on_exit.load() && needsMux() && writer.is_open();
            bool shouldFileAudioMux = !skip_audio_mux_on_exit.load() && needsFileAudioMux() && writer.is_open();
            stopWriterThread();
            if (shouldMux) {
                runMuxSync();
            }
            if (shouldFileAudioMux) {
                runFileAudioMuxSync();
            }
        }
        if (muxThread.joinable()) {
            muxThread.join();
        }

#ifdef AUDIO_ENABLED
        if (audio_is_enabled) {
            if (is_audio_recording()) {
                stop_audio_recording();
            }
            if (file_audio_mode)
                file_audio_close();
            else
                close_audio();
            spectrumTex.cleanup();
        }
#endif

        for (int i = 0; i < 2; ++i) {
#ifdef ACMX2_WITH_CUDA
            if (recordCudaPboResources[i]) {
                CHECK_CUDA(cudaGraphicsUnregisterResource(recordCudaPboResources[i]));
                recordCudaPboResources[i] = nullptr;
            }
#else
            (void)i;
#endif
        }

        if (pboIds[0]) {
            glDeleteBuffers(2, pboIds);
            pboIds[0] = pboIds[1] = 0;
        }

        if (captureFBO) {
            glDeleteFramebuffers(1, &captureFBO);
            captureFBO = 0;
        }
        if (fboTexture) {
            glDeleteTextures(1, &fboTexture);
            fboTexture = 0;
        }
        if (preOverlayTexture) {
            glDeleteTextures(1, &preOverlayTexture);
            preOverlayTexture = 0;
        }
        if (preOverlayFBO) {
            glDeleteFramebuffers(1, &preOverlayFBO);
            preOverlayFBO = 0;
        }
        for (int p = 0; p < 2; ++p) {
            if (passFBO[p]) {
                glDeleteFramebuffers(1, &passFBO[p]);
                passFBO[p] = 0;
            }
            if (passTexture[p]) {
                glDeleteTextures(1, &passTexture[p]);
                passTexture[p] = 0;
            }
        }
        if (crossfadeFBO) {
            glDeleteFramebuffers(1, &crossfadeFBO);
            crossfadeFBO = 0;
        }
        if (crossfadeTexture) {
            glDeleteTextures(1, &crossfadeTexture);
            crossfadeTexture = 0;
        }
        if (crossfadePrevTexture) {
            glDeleteTextures(1, &crossfadePrevTexture);
            crossfadePrevTexture = 0;
        }
        if (depthBuffer) {
            glDeleteRenderbuffers(1, &depthBuffer);
            depthBuffer = 0;
        }
        if (camera_texture) {
            glDeleteTextures(1, &camera_texture);
            camera_texture = 0;
        }

        if (texture_cache) {
            glDeleteTextures(8, cache_textures);
            for (int i = 0; i < 8; i++) {
                cache_textures[i] = 0;
            }
        }

        if (cap.isOpened())
            cap.release();
    }

    mx::Model cube;
    gl::ShaderProgram fshader, fshader3d;
    std::string m_file;

    /**
     * @brief Called once by the GLWindow to initialise capture, shaders, FBOs, and recording.
     *
     * Performs the full OpenGL-dependent setup:
     * 1. Set the CUDA device.
     * 2. Load fonts.
     * 3. Open the input source (image file, camera, or video file);
     *    set resolution, FPS, and window size.
     * 4. Open the output Writer (FFmpeg pipe) if `-o` was specified.
     * 5. Load and compile the shader library (from cache if available).
     * 6. Load the 3D model (if --enable-3d).
     * 7. Initialise framebuffer shader, texture cache, sprites.
     * 8. Set up the capture FBO and double-buffered PBOs.
     * 9. Start the writer thread (always) and the capture thread
     *    (camera mode only).
     *
     * @param win Pointer to the hosting GLWindow.
     */
    virtual void load(gl::GLWindow *win) override {
#ifdef ACMX2_WITH_CUDA
        cudaError_t cuda_err = cudaSetDevice(gpu_cuda_device);
        if (cuda_err != cudaSuccess) {
            throw mx::Exception("Failed to set CUDA device " + std::to_string(gpu_cuda_device) + ": " + std::string(cudaGetErrorString(cuda_err)));
        }
        mx::system_out << "acmx2: Using CUDA device: " << gpu_cuda_device << "\n";
        fflush(stdout);
#else
        mx::system_out << "acmx2: CUDA support disabled (compiled without ACMX2_WITH_CUDA)\n";
        fflush(stdout);
#endif

        frame_counter = 0;
        sessionStartTime = std::chrono::steady_clock::now();
        fpsLastTime = sessionStartTime;
        fpsFrameCount = 0;
        displayFPS = 0.0;

        overlayFont.tryLoadFont(win->util.getFilePath("data/font.ttf"), 24);

        int w = 1280, h = 720;
        int frame_w = w, frame_h = h;

        if (!graphic.empty()) {
            graphic_frame = cv::imread(graphic);
            if (graphic_frame.empty()) {
                throw mx::Exception("Graphics file not found: " + graphic);
            }

            w = graphic_frame.cols;
            h = graphic_frame.rows;
            frame_w = w;
            frame_h = h;
            mx::system_out << "acmx2: Graphics file loaded: " << w << "x" << h << " at FPS: " << fps << "\n";
            fflush(stdout);
            fflush(stderr);
            if (sizev.has_value()) {
                w = sizev.value().width;
                h = sizev.value().height;
                mx::system_out << "acmx2: Resolution stretched to: " << w << "x" << h << "\n";
                fflush(stdout);
                fflush(stderr);
            }

            win->setWindowSize(w, h);
            SDL_PumpEvents();
            SDL_Delay(50);
            SDL_PumpEvents();
            SDL_GL_GetDrawableSize(win->getWindow(), &win->w, &win->h);
            if (win->w != w || win->h != h) {
                win->w = w;
                win->h = h;
            }
            glViewport(0, 0, win->w, win->h);
            SDL_SetWindowPosition(win->getWindow(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
            if (!ofilename.empty()) {
                if (writer.open(ofilename, w, h, fps, encode_opts)) {
                    if (silent_mode || no_drop_mode) {
                        // Block the producer when the encoder queue fills
                        // so no encoded frames are dropped.
                        writer.set_block_when_full(true);
                    }
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << encode_opts.crf
                                   << " preset: " << encode_opts.preset
                                   << " tune: " << (encode_opts.tune.empty() ? "none" : encode_opts.tune)
                                   << " codec: " << encode_opts.codec
                                   << (encode_opts.realtime ? " [realtime]" : "")
                                   << " FPS: " << fps << "\n";
#ifdef AUDIO_ENABLED
                    startAudioRecordingIfNeeded();
#endif
                    mx::system_out << "acmx2: Pipeline mode => decode: graphic/image, encode: "
                                   << (writer.is_hardware_encode() ? "h264_nvenc (hardware)" : "h264 (software)") << "\n";

                    fflush(stdout);
                    fflush(stderr);
                } else {
                    throw mx::Exception("Could not open output video file: " + ofilename);
                }
            }
        } else if (filename.empty()) {
#ifdef _WIN32
            cap.open(camera_index, cv::CAP_DSHOW);
#elif defined(__linux__)
            cap.open(camera_index, cv::CAP_V4L2);
#else
            cap.open(camera_index);
#endif
            if (!cap.isOpened()) {
                throw mx::Exception("Could not open camera index: " + std::to_string(camera_index));
            }
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            if (sizec.has_value()) {
                cap.set(cv::CAP_PROP_FRAME_WIDTH, sizec.value().width);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, sizec.value().height);
            } else {
                cap.set(cv::CAP_PROP_FRAME_WIDTH, win->w);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, win->h);
            }
            if (use_yuv)
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
            else
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FPS, fps);
            w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            fps = cap.get(cv::CAP_PROP_FPS);
            frame_w = w;
            frame_h = h;
            mx::system_out << "acmx2: Camera opened: " << w << "x" << h << " at FPS: " << fps << "\n";
            fflush(stderr);
            fflush(stdout);

            if (sizev.has_value()) {
                w = sizev.value().width;
                h = sizev.value().height;
                mx::system_out << "acmx2: Resolution stretched to: " << w << "x" << h << "\n";
            }
            win->setWindowSize(w, h);
            SDL_GL_GetDrawableSize(win->getWindow(), &win->w, &win->h);
            if (win->w != w || win->h != h) {
                win->w = w;
                win->h = h;
            }
            glViewport(0, 0, w, h);

            SDL_SetWindowPosition(win->getWindow(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

            if (!ofilename.empty()) {
                // Camera capture path — force realtime semantics to avoid
                // encoder stalls on live input regardless of user preset.
                EncodeOptions cam_opts = encode_opts;
                cam_opts.realtime = true;
                if (writer.open_ts(ofilename, w, h, fps, cam_opts)) {
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << cam_opts.crf
                                   << " preset: " << cam_opts.preset
                                   << " codec: " << cam_opts.codec
                                   << " [realtime]"
                                   << " FPS: " << fps << "\n";
#ifdef AUDIO_ENABLED
                    startAudioRecordingIfNeeded();
#endif
                    mx::system_out << "acmx2: Pipeline mode => decode: camera, encode: "
                                   << (writer.is_hardware_encode() ? "h264_nvenc (hardware)" : "h264 (software)") << "\n";
                } else {
                    throw mx::Exception("Could not open output video file: " + ofilename);
                }
            }
        } else if (!filename.empty() && graphic.empty()) {
            use_ffmpeg_reader = ffmpeg_reader.open(filename, true);
            const char *decode_mode = nullptr;
            if (use_ffmpeg_reader) {
                decode_mode = ffmpeg_reader.isHwDecodeEnabled() ? "ffmpeg-cuda" : "ffmpeg-software";
                w = ffmpeg_reader.getWidth();
                h = ffmpeg_reader.getHeight();
                fps = ffmpeg_reader.getFps();
                totalFrames = ffmpeg_reader.getFrameCount();
                library.setVideoFPS(fps);

                frame_w = w;
                frame_h = h;

                mx::system_out << "acmx2: Video opened (FFmpeg decode): " << w << "x" << h
                               << " at FPS: " << fps
                               << " Total Frames: " << totalFrames << "\n";
                mx::system_out << "acmx2: FFmpeg CUDA decode: " << (ffmpeg_reader.isHwDecodeEnabled() ? "enabled" : "unavailable/fallback") << "\n";

                // If the input video carries HDR metadata (BT.2020 primaries /
                // PQ or HLG transfer / >=10-bit depth), the CUDA glitch filters
                // are not colour-correct on that data (they assume 8-bit sRGB).
                // Force-disable the CUDA filter stage and run shaders only.
                if (ffmpeg_reader.isHdr()) {
                    // Record HDR state so the GL pipeline switches to the
                    // 16-bit linear BT.2020 path (allocated further below).
                    input_is_hdr = true;
                    input_hdr_trc = ffmpeg_reader.getHdrTransfer();
                    const int trc = input_hdr_trc;
                    const char *trc_label =
                        (trc == AVCOL_TRC_SMPTE2084)    ? "PQ (SMPTE2084)" :
                        (trc == AVCOL_TRC_ARIB_STD_B67) ? "HLG (ARIB STD-B67)" :
                        (trc == AVCOL_TRC_BT2020_10)    ? "BT.2020 10-bit" :
                        (trc == AVCOL_TRC_BT2020_12)    ? "BT.2020 12-bit" : "unknown";
                    mx::system_out << "acmx2: ============================================================\n"
                                   << "acmx2: *** PROCESSING IN HDR MODE ***\n"
                                   << "acmx2:   Source is HDR: " << trc_label
                                   << ", " << ffmpeg_reader.getHdrBitDepth() << "-bit,"
                                   << " BT.2020 primaries\n"
                                   << "acmx2: ============================================================\n";
                    mx::system_out << "acmx2: HDR input detected (primaries=" << ffmpeg_reader.getHdrPrimaries()
                                   << ", transfer=" << trc
                                   << ", bit_depth=" << ffmpeg_reader.getHdrBitDepth() << ")\n";
                    if (gpu_filter_enabled) {
                        mx::system_out << "acmx2: *** CUDA GPU filters DISABLED for HDR input "
                                          "(colour math assumes 8-bit sRGB). Shader processing only. ***\n";
                        gpu_filter_enabled = false;
                        gpu_filters.clear();
                    } else {
                        mx::system_out << "acmx2: CUDA GPU filters not in use; shader processing only for HDR input.\n";
                    }

                    // Populate encode_opts.hdr so the writer switches to the
                    // HEVC Main10 + BT.2020 output path for both write paths
                    // below. We force libx265 + Main10 regardless of the
                    // user's codec preference when the input is HDR.
                    encode_opts.hdr.enabled = true;
                    encode_opts.hdr.color_primaries = ffmpeg_reader.getHdrPrimaries();
                    encode_opts.hdr.color_trc = ffmpeg_reader.getHdrTransfer();
                    encode_opts.hdr.color_space = ffmpeg_reader.getHdrColorspace();
                    encode_opts.hdr.color_range = ffmpeg_reader.getHdrColorRange();
                    encode_opts.hdr.mastering_display = ffmpeg_reader.getHdrMasteringDisplay();
                    encode_opts.hdr.content_light = ffmpeg_reader.getHdrContentLight();
                    mx::system_out << "acmx2: HDR output mode enabled: HEVC Main10 + BT.2020 + "
                                   << (encode_opts.hdr.color_trc == AVCOL_TRC_ARIB_STD_B67 ? "HLG" : "PQ")
                                   << " (mastering side-data bytes="
                                   << encode_opts.hdr.mastering_display.size()
                                   << ", content-light bytes="
                                   << encode_opts.hdr.content_light.size() << ")\n";

                    // Upgrade existing SDR GL resources (camera_texture,
                    // fboTexture, crossfade textures) to 16-bit formats
                    // and lazily create the HDR decode/encode machinery.
                    convertResourcesToHdr(w, h);
                    mx::system_out << "acmx2: HDR GL pipeline: camera_texture=GL_RGBA16, "
                                      "fboTexture=GL_RGBA16F, pass textures=GL_RGBA16F, "
                                      "decode/encode shaders ready.\n";
                }
            } else {
                decode_mode = "opencv-ffmpeg";
                std::vector<int> file_params = {
                    cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY};
                cap.open(filename, cv::CAP_FFMPEG, file_params);
                if (!cap.isOpened()) {
                    throw mx::Exception("Could not open video file: " + filename);
                }
                w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                fps = cap.get(cv::CAP_PROP_FPS);
                totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
                library.setVideoFPS(fps);

                frame_w = w;
                frame_h = h;

                mx::system_out << "acmx2: Video opened (OpenCV/FFmpeg fallback): " << w << "x" << h
                               << " at FPS: " << fps
                               << " Total Frames: " << totalFrames << "\n";

                int hw_accel = static_cast<int>(cap.get(cv::CAP_PROP_HW_ACCELERATION));
                mx::system_out << "acmx2: HW Acceleration result: " << hw_accel
                               << (hw_accel == cv::VIDEO_ACCELERATION_NONE ? " (software/fallback)" : hw_accel == cv::VIDEO_ACCELERATION_ANY ? " (auto preference)"
                                                                                                  : hw_accel == cv::VIDEO_ACCELERATION_VAAPI ? " (VAAPI)"
                                                                                                  : hw_accel == cv::VIDEO_ACCELERATION_D3D11 ? " (D3D11)"
                                                                                                  : hw_accel == cv::VIDEO_ACCELERATION_MFX   ? " (MFX)"
                                                                                                  : hw_accel == cv::VIDEO_ACCELERATION_DRM   ? " (DRM)"
                                                                                                                                             : " (other)")
                               << "\n";
            }

            fflush(stdout);
            fflush(stderr);

            if (sizev.has_value()) {
                w = sizev.value().width;
                h = sizev.value().height;
                mx::system_out << "acmx2: Resolution stretched to: "
                               << w << "x" << h << "\n";
                fflush(stdout);
                fflush(stderr);
            }

            win->setWindowSize(w, h);
            SDL_GL_GetDrawableSize(win->getWindow(), &win->w, &win->h);
            if (win->w != w || win->h != h) {
                win->w = w;
                win->h = h;
            }
            glViewport(0, 0, w, h);
            SDL_SetWindowPosition(win->getWindow(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

            if (!ofilename.empty()) {
                if (writer.open(ofilename, w, h, fps, encode_opts)) {
                    if (silent_mode || no_drop_mode) {
                        // Batch transcoding or --no-drop: block the producer
                        // when the encoder queue fills instead of dropping
                        // frames.
                        writer.set_block_when_full(true);
                    }
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << encode_opts.crf
                                   << " preset: " << encode_opts.preset
                                   << " tune: " << (encode_opts.tune.empty() ? "none" : encode_opts.tune)
                                   << " codec: " << encode_opts.codec
                                   << (encode_opts.realtime ? " [realtime]" : "")
                                   << "\n";
                    if (no_drop_mode) {
                        mx::system_out << "acmx2: --no-drop active (video mode): producer blocks when encoder queue is full\n";
                    }
                    if (encode_opts.hdr.enabled) {
                        mx::system_out << "acmx2: *** HDR OUTPUT ENABLED: writing HEVC Main10 + BT.2020 "
                                       << (encode_opts.hdr.color_trc == AVCOL_TRC_ARIB_STD_B67 ? "HLG" : "PQ")
                                       << " ***\n";
                    }
#ifdef AUDIO_ENABLED
                    startAudioRecordingIfNeeded();
#endif
                    mx::system_out << "acmx2: Pipeline mode => decode: " << decode_mode
                                   << ", encode: "
                                   << (writer.is_hardware_encode() ? "h264_nvenc (hardware)" : "h264 (software)") << "\n";
                    fflush(stdout);
                    fflush(stderr);
                } else {
                    throw mx::Exception("Could not open output video file: " + ofilename);
                }
            }
        } else if (graphic.empty() && filename.empty()) {
            throw mx::Exception("Requires input from a file, or camera.");
        }

        library.is3D(is3d_enabled);
        library.enableDualMode(is3d_enabled);
        if (overlayFont.handle().has_value()) {
            win->text.init(win->w, win->h);
            win->text.setColor({255, 255, 255, 255});
        }
        waterFontSize = std::max(12, static_cast<int>(win->h / 40.0f));
        waterFont.tryLoadFont(win->util.getFilePath("data/font.ttf"), waterFontSize);
        mx::system_out << "acmx2: Watermark font loaded at size: " << waterFontSize << " for " << win->w << "x" << win->h << "\n";
        fflush(stdout);

        library.enableCache(use_shader_cache_flag);
        if (std::get<0>(flib) == 1) {
            if (use_shader_cache_flag)
                library.loadProgramsWithCache(win, std::get<1>(flib), overlayFont);
            else
                library.loadPrograms(win, std::get<1>(flib), overlayFont);
        } else {
            library.loadProgram(win, std::get<1>(flib));
        }
        library.setIndex(std::get<2>(flib));
        if (!playlist_file.empty()) {
            std::ifstream pfile(playlist_file);
            if (!pfile.is_open()) {
                mx::system_err << "acmx2: Error could not open playlist: " << playlist_file << "\n";
            } else {
                std::string line;
                PlaylistNode *currentNode = nullptr;
                while (std::getline(pfile, line)) {
                    if (line.empty()) continue;
                    if (line.front() == '[' && line.back() == ']') {
                        playlist_tree.push_back({line.substr(1, line.size() - 2), {}});
                        currentNode = &playlist_tree.back();
                        continue;
                    }
                    std::string name = std::filesystem::path(line).stem().string();
                    int idx = library.findShaderByName(name);
                    if (idx >= 0) {
                        playlist_indices.push_back(idx);
                        if (currentNode) {
                            currentNode->shader_indices.push_back(idx);
                        }
                    } else {
                        mx::system_err << "acmx2: Playlist shader not found: " << line << "\n";
                    }
                }
                if (playlist_tree.empty() && !playlist_indices.empty()) {
                    playlist_tree.push_back({"Default", playlist_indices});
                }
                mx::system_out << "acmx2: Playlist loaded [" << playlist_indices.size() << "] shaders in ["
                               << playlist_tree.size() << "] nodes from: " << playlist_file << "\n";
                for (const auto &node : playlist_tree) {
                    mx::system_out << "  Node: " << node.name << " [" << node.shader_indices.size() << " shaders]\n";
                }
                fflush(stdout);
            }
        }
        updateShaderNameCache();

        std::string m_file_path;
        if (std::filesystem::exists(m_file)) {
            m_file_path = m_file;
        } else {
            m_file_path = win->util.getFilePath("data/" + m_file);
        }

        if (is3d_enabled && !cube.openModel(m_file_path)) {
            throw mx::Exception("Could not open model: cube.mxmod.z");
        }
        cube.setShaderProgram(library.shader(), "samp");
        cube.saveOriginal();

        if (is3d_enabled && !cube.meshes.empty()) {
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float minZ = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float maxY = std::numeric_limits<float>::lowest();
            float maxZ = std::numeric_limits<float>::lowest();
            for (const auto &mesh : cube.meshes) {
                for (size_t i = 0; i + 2 < mesh.vert.size(); i += 3) {
                    float x = mesh.vert[i];
                    float y = mesh.vert[i + 1];
                    float z = mesh.vert[i + 2];
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    minZ = std::min(minZ, z);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                    maxZ = std::max(maxZ, z);
                }
            }
            float dx = maxX - minX;
            float dy = maxY - minY;
            float dz = maxZ - minZ;
            modelSize = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (modelSize < 0.001f)
                modelSize = 1.0f;
            glm::vec3 modelCenter = glm::vec3((minX + maxX) * 0.5f, (minY + maxY) * 0.5f, (minZ + maxZ) * 0.5f);
            modelCenterOffset = -modelCenter;
            float maxExtent = std::max(dx, std::max(dy, dz));
            const float targetSize = 2.5f;
            if (maxExtent > 1e-6f)
                modelRenderScale = targetSize / maxExtent;
            mx::system_out << "acmx2: Model bounding diagonal: " << modelSize << "\n";
            fflush(stdout);
        }

        if (!fshader.loadProgram(win->util.getFilePath("data/vert.glsl"), win->util.getFilePath("data/framebuffer.glsl"))) {
            throw mx::Exception("Error loading shader");
        }
        if (!fshader3d.loadProgram(win->util.getFilePath("data/vertex.glsl"), win->util.getFilePath("data/framebuffer.glsl"))) {
            throw mx::Exception("Error loading shader");
        }
        if (!crossfadeShader.loadProgram(win->util.getFilePath("data/vert.glsl"), win->util.getFilePath("data/crossfade.glsl"))) {
            throw mx::Exception("Error loading crossfade shader");
        }
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw mx::Exception("OpenGL error occurred: GL Error: " + std::to_string(error));
        }

        library.useProgram();
        if (texture_cache) {
            cv::Mat blankMat = cv::Mat::zeros(frame_h, frame_w, CV_8UC3);
            for (int i = 0; i < 8; ++i) {
                cache_textures[i] = loadTexture(blankMat);
            }
            frame_cache.fill(blankMat);
            mx::system_out << "acmx2: Texture cache initalized.\n";
            fflush(stdout);
        }
        sprite.initSize(win->w, win->h);
        tex_uploader.init(win->w, win->h);
        camera_texture = tex_uploader.textureID;
        if (input_is_hdr) {
            // HDR path uploads decoded frames directly via glTex(Sub)Image2D.
            // Do not reuse TextureUploader's texture object here: when
            // --resolution differs from source size, re-specifying that
            // texture has caused driver crashes in headless EGL on NVIDIA.
            // Keep an independent texture dedicated to HDR uploads.
            glGenTextures(1, &camera_texture);
            glBindTexture(GL_TEXTURE_2D, camera_texture);
            const int upload_w = (frame_w > 0) ? frame_w : win->w;
            const int upload_h = (frame_h > 0) ? frame_h : win->h;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16,
                         upload_w, upload_h, 0,
                         GL_RGBA, GL_UNSIGNED_SHORT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            hdr_upload_tex_w = upload_w;
            hdr_upload_tex_h = upload_h;
        }
        sprite.setName("samp");
        sprite.initWithTexture(library.shader(), camera_texture, 0, 0, win->w, win->h);
        setupCaptureFBO(win->w, win->h);
        if (input_is_hdr) {
            // At HDR detection time, GL resources may not exist yet; run the
            // conversion now with actual allocated textures/FBOs and the
            // final output size.
            convertResourcesToHdr(win->w, win->h);
        }
        glGenBuffers(2, pboIds);
        size_t pboSize = win->w * win->h * 4;
        for (int i = 0; i < 2; i++) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, pboSize, nullptr, GL_STREAM_READ);
#ifdef ACMX2_WITH_CUDA
            CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&recordCudaPboResources[i], pboIds[i], cudaGraphicsMapFlagsReadOnly));
#endif
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        if (!graphic.empty())
            win->setWindowTitle("ACMX2 - Graphics Input");
        else if (filename.empty())
            win->setWindowTitle("ACMX2 - Capture Input");
        else
            win->setWindowTitle("ACMX2 - [" + filename + "]");

        if (full) {
            win->setFullScreen(true);
        }
        running = true;
        if (writer.is_open() || true /* snapshots possible */) {
            startWriterThread();
        }

        if (filename.empty() && cap.isOpened()) {
            startCaptureThread();
        }
    }

    cv::Mat newFrame;
    float movementSpeed = 0.1f;

    /**
     * @brief Called every frame: capture, GPU-filter, shade, composite, record, display overlay.
     *
     * This is the main rendering loop body.  On each invocation:
     * 1. Rate-limit to the configured FPS.
     * 2. Poll MIDI (if enabled).
     * 3. If muxing, draw the overlay and return early.
     * 4. Check the duration limit.
     * 5. Read a new frame: from capture queue (camera), cap.read()
     *    (video), or clone (image).
     * 6. Apply GPU CUDA filters (if enabled) via TextureUploader /
     *    launch_filter(), otherwise upload via glTexSubImage2D.
     * 7. Bind the capture FBO, activate the shader, upload uniforms.
     * 8. Render in 2D (sprite quad) or 3D (model with camera/wave),
     *    including multi-pass chains through ping-pong FBOs.
     * 9. Blit the FBO result to the default framebuffer.
     * 10. PBO double-buffered readback into FrameData, push to writer queue.
     * 11. Draw the HUD overlay (shader name, FPS, timer).
     * 12. Update the window title with progress info.
     *
     * @param win Pointer to the hosting GLWindow.
     */
    virtual void draw(gl::GLWindow *win) override {
        // Skip FPS pacing in headless/silent batch mode so transcoding runs
        // at full speed instead of being capped at the input's frame rate.
        if (fps > 0.0 && !silent_mode) {
            auto now = std::chrono::steady_clock::now();
            auto frame_duration = std::chrono::microseconds(static_cast<long long>(1000000.0 / fps));
            if (now > lastFrameTime + (frame_duration * 4)) {
                lastFrameTime = now;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastFrameTime);
            if (elapsed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed);
            }
            lastFrameTime += frame_duration;
        }

#ifdef MIDI_ENABLED
        pollMidi(win);
#endif

        if (isMuxing.load()) {
            if (muxComplete.load()) {
                if (muxThread.joinable())
                    muxThread.join();
                isMuxing = false;
                win->quit();
                return;
            }
            glViewport(0, 0, win->w, win->h);
            if (overlayFont.handle().has_value()) {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                win->text.setColor({255, 255, 255, 255});
                win->text.printText_Blended(overlayFont, 10, 10, "Muxing audio...");
                glDisable(GL_BLEND);
            }
            return;
        }

        if (duration_limit > 0.0 && writer.is_open() && writerRunning) {
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - captureStartTime).count();
            if (elapsed >= duration_limit) {
                mx::system_out << "acmx2: Duration limit reached (" << duration_limit << "s), stopping recording...\n";
                fflush(stdout);
                running = false;
            }
        }

        if (!running) {
            if (!skip_audio_mux_on_exit.load() && (needsMux() || needsTransferAudio() || needsFileAudioMux())) {
                beginMuxing(win);
                return;
            }
            win->quit();
            return;
        }

        if (!isPaused && !isFrozen) {
            if (!graphic.empty()) {
                newFrame = graphic_frame.clone();
                cv::flip(newFrame, newFrame, 0);
            } else if (filename.empty()) {
                std::unique_lock<std::mutex> lock(captureQueueMutex);
                if (!captureQueue.empty()) {
                    newFrame = std::move(captureQueue.front());
                    captureQueue.pop();
                }
            } else {
                bool read_ok = false;
                if (use_ffmpeg_reader) {
                    if (input_is_hdr) {
                        read_ok = ffmpeg_reader.readHdr(hdr_frame_mat);
                        if (!read_ok && !filename.empty() && repeat) {
                            mx::system_out << "acmx2: video loop...\n";
                            if (ffmpeg_reader.seekStart()) {
                                read_ok = ffmpeg_reader.readHdr(hdr_frame_mat);
                            }
                            if (!read_ok) {
                                mx::system_out << "acmx2: cannot read after looping.\n";
                            }
                        }
                    } else {
                        read_ok = ffmpeg_reader.read(newFrame);
                        if (!read_ok && !filename.empty() && repeat) {
                            mx::system_out << "acmx2: video loop...\n";
                            if (ffmpeg_reader.seekStart()) {
                                read_ok = ffmpeg_reader.read(newFrame);
                            }
                            if (!read_ok) {
                                mx::system_out << "acmx2: cannot read after looping.\n";
                            }
                        }
                    }
                } else {
                    read_ok = cap.read(newFrame);
                    if (!read_ok && !filename.empty() && repeat) {
                        mx::system_out << "acmx2: video loop...\n";
                        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        read_ok = cap.read(newFrame);
                        if (!read_ok) {
                            mx::system_out << "acmx2: cannot read after looping.\n";
                        }
                    }
                }

                if (!read_ok) {
                    if (silent_mode) {
                        std::cout << "\n";
                    }
                    running = false;
                    finished = true;
                    return;
                }

                if (!newFrame.empty())
                    cv::flip(newFrame, newFrame, 0);
                // HDR path: leave @c hdr_frame_mat top-down. The SDR path
                // pre-flips because its shader chain produces a Y-flipped
                // readback that the CPU loop then re-flips; the HDR path
                // has an additional sprite.draw pass (hdr_encode) before
                // the PBO-free readback, and empirically the single CPU
                // flip in @c hdrReadback's caller is all that is needed
                // to deliver top-down rows to the HEVC encoder.
            }
        }
        if (library.isBypassed()) {
            if (is3d_enabled) {
                fshader3d.useProgram();
            } else {
                fshader.useProgram();
            }
        } else {
            library.useProgram();
        }
        if (!isFrozen && input_is_hdr) {
            // HDR branch: upload 16-bit RGBA (PQ/HLG-encoded BT.2020) and
            // run the decode fullscreen pass so the user-shader chain
            // samples linear BT.2020 light via @c hdr_linear_video_texture.
            if (!hdr_frame_mat.empty()) {
                glActiveTexture(GL_TEXTURE0);
                uploadHdrFrame(hdr_frame_mat);
                runHdrDecodePass(win->w, win->h);
            }
        } else if (!isFrozen && !newFrame.empty()) {
#ifdef ACMX2_WITH_CUDA
            if (gpu_filter_enabled && !gpu_filters.empty() && gpu_frame_buffer) {
                gpu_frame_buffer->update(newFrame);

                if (gpuWorkingBuffer.empty() || gpuWorkingBuffer.cols != newFrame.cols || gpuWorkingBuffer.rows != newFrame.rows) {
                    gpuWorkingBuffer.create(newFrame.rows, newFrame.cols, CV_8UC4);
                }

                if (gpu_alpha_dir == 1) {
                    gpu_alpha += 0.01f;
                    if (gpu_alpha >= 3.0f)
                        gpu_alpha_dir = 0;
                } else {
                    gpu_alpha -= 0.01f;
                    if (gpu_alpha <= 1.0f)
                        gpu_alpha_dir = 1;
                }

                if (gpu_frame_dir == 1) {
                    gpu_frame_index++;
                    if (gpu_frame_index >= gpu_frame_buffer->arraySize - 1) {
                        gpu_frame_index = gpu_frame_buffer->arraySize - 1;
                        gpu_frame_dir = 0;
                    }
                } else {
                    gpu_frame_index--;
                    if (gpu_frame_index <= 0) {
                        gpu_frame_index = 0;
                        gpu_frame_dir = 1;
                    }
                }

                CHECK_CUDA(cudaMemcpy(d_ptrList, gpu_frame_buffer->rawPointers.data(),
                                      gpu_frame_buffer->arraySize * sizeof(unsigned char *),
                                      cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaMemcpy2D(gpuWorkingBuffer.ptr<unsigned char>(), gpuWorkingBuffer.step,
                                        gpu_frame_buffer->deviceFrames[gpu_frame_buffer->arraySize - 1].data,
                                        gpu_frame_buffer->framePitch,
                                        gpu_frame_buffer->w * 4, gpu_frame_buffer->h,
                                        cudaMemcpyDeviceToDevice));

                launch_filter(
                    gpu_filters.data(),
                    gpu_filters.size(),
                    gpuWorkingBuffer.ptr<unsigned char>(),
                    d_ptrList,
                    gpu_frame_buffer->arraySize,
                    gpuWorkingBuffer.cols,
                    gpuWorkingBuffer.rows,
                    gpuWorkingBuffer.step,
                    gpu_alpha,
                    false,
                    gpu_square_size,
                    gpu_frame_index,
                    gpu_frame_dir,
                    &d_filterList,
                    gpu_filtersChanged);
                gpu_filtersChanged = false;
                tex_uploader.update(gpuWorkingBuffer);
            } else {
                glActiveTexture(GL_TEXTURE0);
                updateTexture(camera_texture, newFrame);
            }
#else
            {
                glActiveTexture(GL_TEXTURE0);
                updateTexture(camera_texture, newFrame);
            }
#endif
            if (texture_cache && library.isCache() && (!filename.empty() || !graphic.empty())) {
                static int counter = 0;
                if (++counter > cache_delay) {
                    frame_cache.push(std::move(newFrame));
                    counter = 0;
                }
                if (frame_cache.isFull()) {
                    for (int i = 0; i < 8; ++i) {
                        library.setUniform("samp" + std::to_string(i + 1), i);
                        glActiveTexture(GL_TEXTURE1 + i);
                        updateTexture(cache_textures[i], frame_cache.at(i));
                        glBindTexture(GL_TEXTURE_2D, cache_textures[i]);
                    }
                }
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        glViewport(0, 0, win->w, win->h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!isFrozen && !library.isBypassed()) {
            library.useProgram();
#ifdef AUDIO_ENABLED
            if (audio_is_enabled) {
                if (file_audio_mode) {
                    file_audio_process_frame(fps);
                    if (audio_trunc_mode && !file_audio_is_active()) {
                        mx::system_out << "acmx2: Audio file finished, stopping (--audio-trunc).\n";
                        fflush(stdout);
                        running = false;
                    }
                }
                if (spectrum_scale_by_sense)
                    spectrumTex.update(get_sense());
                else
                    spectrumTex.update();
                spectrumTex.bind();
            }
#endif
            library.update(win);
            library.setFPS(static_cast<float>(fps));
        }

        {
            const Uint8 *keystate = SDL_GetKeyboardState(NULL);
            if (keystate[SDL_SCANCODE_PAGEUP]) {
                library.incTimeSpeed(0.1f);
                fflush(stdout);
            }
            if (keystate[SDL_SCANCODE_PAGEDOWN]) {
                library.decTimeSpeed(0.1f);
                fflush(stdout);
            }

            if(keystate[SDL_SCANCODE_PERIOD]) {
                cameraRotationSpeed += 0.5f;
                if (cameraRotationSpeed > 50.0f) cameraRotationSpeed = 50.0f;
                    mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                    fflush(stdout);
            }
            if(keystate[SDL_SCANCODE_COMMA]) {
                cameraRotationSpeed -= 0.5f;
                if (cameraRotationSpeed < 0.5f) cameraRotationSpeed = 0.5f;
                    mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                fflush(stdout);
            }

        }

        if (is3d_enabled) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glDisable(GL_CULL_FACE);

            static auto last3DTime = std::chrono::steady_clock::now();
            auto now3D = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now3D - last3DTime).count();
            if (dt > 0.1f) dt = 0.1f;
            last3DTime = now3D;

            static float rotation = 0.0f;
            rotation = fmod(rotation + 0.5f, 360.0f);

            const Uint8 *keystate = SDL_GetKeyboardState(NULL);
            if (!oscillateScale) {

                if (keystate[SDL_SCANCODE_B]) {
                    movementSpeed += 0.1f * dt * 30.0f;
                    mx::system_out << "acmx2: movement increased: " << movementSpeed << "\n";
                    fflush(stdout);
                }

                if (keystate[SDL_SCANCODE_N]) {
                    movementSpeed -= 0.1f * dt * 30.0f;
                    mx::system_out << "acmx2: movement decreased: " << movementSpeed << "\n";
                    fflush(stdout);
                }

                if (keystate[SDL_SCANCODE_EQUALS] || keystate[SDL_SCANCODE_KP_PLUS]) {
                    cameraDistance += movementSpeed * dt;
                    mx::system_out << "acmx2: cameraDistance increased: " << cameraDistance << "\n";
                    fflush(stdout);
                }
                if (keystate[SDL_SCANCODE_MINUS] || keystate[SDL_SCANCODE_KP_MINUS]) {
                    cameraDistance -= movementSpeed * dt;
                    mx::system_out << "acmx2: cameraDistance decreased: " << cameraDistance << "\n";
                    fflush(stdout);
                }
                if (keystate[SDL_SCANCODE_RIGHTBRACKET]) {
                    modelRenderScale += 0.5f * dt;
                    mx::system_out << "acmx2: Model scale increased to " << modelRenderScale << "\n";
                    fflush(stdout);
                }
                if (keystate[SDL_SCANCODE_LEFTBRACKET]) {
                    modelRenderScale -= 0.5f * dt;
                    if (modelRenderScale < 0.05f) modelRenderScale = 0.05f;
                    mx::system_out << "acmx2: Model scale decreased to " << modelRenderScale << "\n";
                    fflush(stdout);
                }
            }
            static float t = 0.0f;
            float oscOffset = 0.0f;
            if (oscillateScale) {
                t += 0.016f;
                oscOffset = 0.3f * std::sin(t);
            }

            if (!viewRotationActive) {
                if (keystate[SDL_SCANCODE_W]) {
                    cameraPitch += cameraRotationSpeed * 0.3f * dt * 30.0f;
                    if (cameraPitch > 89.0f)
                        cameraPitch = 89.0f;
                }
                if (keystate[SDL_SCANCODE_S]) {
                    cameraPitch -= cameraRotationSpeed * 0.33f * dt * 30.0f;
                    if (cameraPitch < -89.0f)
                        cameraPitch = -89.0f;
                }
                if (keystate[SDL_SCANCODE_A]) {
                    cameraYaw -= cameraRotationSpeed * 0.3f * dt * 30.0f;
                    cameraYaw = fmod(cameraYaw + 360.0f, 360.0f);
                }
                if (keystate[SDL_SCANCODE_D]) {
                    cameraYaw += cameraRotationSpeed * 0.3f * dt * 30.0f;
                    cameraYaw = fmod(cameraYaw, 360.0f);
                }
            }
            glm::vec3 lookDirection;
            if (viewRotationActive) {
                static float viewRotation = 0.0f;
                viewRotation = fmod(viewRotation + 0.3f, 360.0f);
                float lookX = 0.48f * sin(glm::radians(viewRotation));
                float lookY = 0.48f * sin(glm::radians(viewRotation * 0.7f));
                float lookZ = 0.48f * cos(glm::radians(viewRotation));
                lookDirection = glm::vec3(lookX, lookY, lookZ);
            } else {
                lookDirection.x = cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
                lookDirection.y = sin(glm::radians(cameraPitch));
                lookDirection.z = cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
                lookDirection = glm::normalize(lookDirection) * 0.48f;
            }

            float finalOffset = oscillateScale ? oscOffset : cameraDistance;
            glm::vec3 cameraPosBase = glm::vec3(0.0f, 0.0f, 0.0f);
            glm::vec3 cameraPos = cameraPosBase - glm::normalize(lookDirection) * finalOffset;
            glm::vec3 cameraTarget = cameraPos + lookDirection;
            glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
            glm::mat4 viewMatrix = glm::lookAt(cameraPos, cameraTarget, cameraUp);
            glm::mat4 projectionMatrix = glm::perspective(
                glm::radians(120.0f),
                static_cast<float>(win->w) / static_cast<float>(win->h),
                0.01f,
                1000.0f);

            glm::mat4 modelMatrix = glm::mat4(1.0f);
            modelMatrix = glm::scale(modelMatrix, glm::vec3(modelRenderScale));
            modelMatrix = glm::rotate(modelMatrix, glm::radians(modelRotX), glm::vec3(1.0f, 0.0f, 0.0f));
            modelMatrix = glm::rotate(modelMatrix, glm::radians(modelRotY), glm::vec3(0.0f, 1.0f, 0.0f));
            modelMatrix = glm::rotate(modelMatrix, glm::radians(modelRotZ), glm::vec3(0.0f, 0.0f, 1.0f));
            modelMatrix = glm::translate(modelMatrix, modelCenterOffset);

            glm::mat4 mvMatrix = viewMatrix * modelMatrix;
            GLuint textureForMesh = input_is_hdr ? hdr_linear_video_texture : camera_texture;
            if (shader_pass_enabled && !shader_pass_list.empty() && !library.isBypassed()) {
                glDisable(GL_DEPTH_TEST);
                auto logPassWarning3D = [](const std::string &msg) {
                    static Uint64 last_warn_tick = 0;
                    Uint64 now_tick = SDL_GetTicks64();
                    if (now_tick - last_warn_tick >= 1000) {
                        mx::system_out << "acmx2: multipass(3D) " << msg << "\n";
                        fflush(stdout);
                        last_warn_tick = now_tick;
                    }
                };

                if (passFBO[0] == 0) {
                    const GLint pass_internal = input_is_hdr ? GL_RGBA16F : GL_RGBA;
                    const GLenum pass_type = input_is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
                    for (int p = 0; p < 2; ++p) {
                        glGenFramebuffers(1, &passFBO[p]);
                        glGenTextures(1, &passTexture[p]);
                        glBindTexture(GL_TEXTURE_2D, passTexture[p]);
                        glTexImage2D(GL_TEXTURE_2D, 0, pass_internal, win->w, win->h, 0, GL_RGBA, pass_type, nullptr);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glBindFramebuffer(GL_FRAMEBUFFER, passFBO[p]);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, passTexture[p], 0);
                        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                            throw mx::Exception("acmx2: 3D pass framebuffer is not complete");
                        }
                    }
                }

                GLuint inputTex = input_is_hdr ? hdr_linear_video_texture : camera_texture;
                int pingpong = 0;
                bool pass_applied = false;
                for (size_t i = 0; i < shader_pass_list.size(); ++i) {
                    int shader_idx = shader_pass_list[i];
                    if (shader_idx >= 0 && shader_idx < static_cast<int>(library.size2d())) {
                        gl::ShaderProgram *pass_shader = library.getShader2D(shader_idx);
                        if (pass_shader) {
                            glBindFramebuffer(GL_FRAMEBUFFER, passFBO[pingpong]);
                            glViewport(0, 0, win->w, win->h);
                            glClear(GL_COLOR_BUFFER_BIT);
                            pass_shader->useProgram();
                            library.updateShaderUniforms2D(win, shader_idx);
                            pass_shader->setUniform("mv_matrix", glm::mat4(1.0f));
                            pass_shader->setUniform("proj_matrix", glm::mat4(1.0f));
                            glActiveTexture(GL_TEXTURE0);
                            glBindTexture(GL_TEXTURE_2D, inputTex);
                            glUniform1i(glGetUniformLocation(pass_shader->id(), "samp"), 0);
                            sprite.setShader(pass_shader);
                            sprite.setName("samp");
                            sprite.draw(inputTex, 0, 0, win->w, win->h);
                            pass_applied = true;
                            inputTex = passTexture[pingpong];
                            pingpong = 1 - pingpong;
                        } else {
                            logPassWarning3D("skipping pass index " + std::to_string(shader_idx) + " (null shader)");
                        }
                    } else {
                        logPassWarning3D("skipping invalid pass index " + std::to_string(shader_idx) + " (valid range 0-" + std::to_string(static_cast<int>(library.size2d()) - 1) + ")");
                    }
                }
                if (!pass_applied) {
                    logPassWarning3D("no pass applied; using base camera texture");
                }

                textureForMesh = inputTex;
                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                glViewport(0, 0, win->w, win->h);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
            }

            gl::ShaderProgram *activeShader;
            if (library.isBypassed()) {
                activeShader = &fshader3d;
            } else {
                activeShader = library.shader();
            }
            activeShader->useProgram();
            activeShader->setUniform("mv_matrix", mvMatrix);
            activeShader->setUniform("proj_matrix", projectionMatrix);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureForMesh);
            glUniform1i(glGetUniformLocation(activeShader->id(), "samp"), 0);
            if (!library.isBypassed()) {
                cube.setShaderProgram(activeShader);
            } else {
                cube.setShaderProgram(&fshader3d);
            }
            cube.resetToOriginal();
            if (waveActive) {
                static float amp_x = 0.0f;
                static float amp_y = 0.0f;
                static float amp_z = 0.0f;
                static int dir_x = 1;
                static int dir_y = 1;
                static int dir_z = 1;
                static const float wave_speed = 0.005f;
                static const float wave_max = 0.5f;
                static const float wave_min = 0.0f;
                static float phase = 0.0f;

#ifdef AUDIO_ENABLED
                if (audio_is_enabled && library.timeAudio())
                    phase += (library.getAmp() * library.getAmpUntouched());
                else
                    phase += 0.05f;
#else
                phase += 0.05f;
#endif
                if (phase > 360.0f)
                    phase -= 360.0f;

                amp_x += wave_speed * dir_x;
                if (amp_x >= wave_max) {
                    amp_x = wave_max;
                    dir_x = -1;
                } else if (amp_x <= wave_min) {
                    amp_x = wave_min;
                    dir_x = 1;
                }

                amp_y += wave_speed * dir_y;
                if (amp_y >= wave_max) {
                    amp_y = wave_max;
                    dir_y = -1;
                } else if (amp_y <= wave_min) {
                    amp_y = wave_min;
                    dir_y = 1;
                }

                amp_z += wave_speed * dir_z;
                if (amp_z >= wave_max) {
                    amp_z = wave_max;
                    dir_z = -1;
                } else if (amp_z <= wave_min) {
                    amp_z = wave_min;
                    dir_z = 1;
                }
                cube.wave(mx::DeformAxis::X, amp_x, 2.0f, phase);
                cube.wave(mx::DeformAxis::Y, amp_y, 2.0f, phase + 120.0f);
                cube.wave(mx::DeformAxis::Z, amp_z, 2.0f, phase + 240.0f);
            }

            cube.updateBuffers();
            cube.recalculateNormals();

            for (auto &m : cube.meshes) {
                m.draw();
            }
            glFrontFace(GL_CCW);
        } else {
            glDisable(GL_DEPTH_TEST);
            GLuint textureForSprite = input_is_hdr ? hdr_linear_video_texture : camera_texture;
            if (shader_pass_enabled && !shader_pass_list.empty() && !library.isBypassed()) {
                auto logPassWarning2D = [](const std::string &msg) {
                    static Uint64 last_warn_tick = 0;
                    Uint64 now_tick = SDL_GetTicks64();
                    if (now_tick - last_warn_tick >= 1000) {
                        mx::system_out << "acmx2: multipass(2D) " << msg << "\n";
                        fflush(stdout);
                        last_warn_tick = now_tick;
                    }
                };
                if (passFBO[0] == 0) {
                    const GLint pass_internal = input_is_hdr ? GL_RGBA16F : GL_RGBA;
                    const GLenum pass_type = input_is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
                    for (int p = 0; p < 2; ++p) {
                        glGenFramebuffers(1, &passFBO[p]);
                        glGenTextures(1, &passTexture[p]);
                        glBindTexture(GL_TEXTURE_2D, passTexture[p]);
                        glTexImage2D(GL_TEXTURE_2D, 0, pass_internal, win->w, win->h, 0, GL_RGBA, pass_type, nullptr);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glBindFramebuffer(GL_FRAMEBUFFER, passFBO[p]);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, passTexture[p], 0);
                        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                            throw mx::Exception("acmx2: pass framebuffer is not complete");
                        }
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                }

                GLuint inputTex = input_is_hdr ? hdr_linear_video_texture : camera_texture;
                int pingpong = 0;
                bool pass_applied = false;

                for (size_t i = 0; i < shader_pass_list.size(); ++i) {
                    int shader_idx = shader_pass_list[i];
                    if (shader_idx >= 0 && shader_idx < static_cast<int>(library.size())) {
                        gl::ShaderProgram *pass_shader = library.getShader(shader_idx);
                        if (pass_shader) {
                            glBindFramebuffer(GL_FRAMEBUFFER, passFBO[pingpong]);
                            glViewport(0, 0, win->w, win->h);
                            glClear(GL_COLOR_BUFFER_BIT);
                            pass_shader->useProgram();
                            library.updateShaderUniforms(win, shader_idx);
                            pass_shader->setUniform("mv_matrix", glm::mat4(1.0f));
                            pass_shader->setUniform("proj_matrix", glm::mat4(1.0f));
                            glActiveTexture(GL_TEXTURE0);
                            glBindTexture(GL_TEXTURE_2D, inputTex);
                            glUniform1i(glGetUniformLocation(pass_shader->id(), "samp"), 0);
                            sprite.setShader(pass_shader);
                            sprite.setName("samp");
                            sprite.draw(inputTex, 0, 0, win->w, win->h);
                            pass_applied = true;
                            inputTex = passTexture[pingpong];
                            pingpong = 1 - pingpong;
                        } else {
                            logPassWarning2D("skipping pass index " + std::to_string(shader_idx) + " (null shader)");
                        }
                    } else {
                        logPassWarning2D("skipping invalid pass index " + std::to_string(shader_idx) + " (valid range 0-" + std::to_string(static_cast<int>(library.size()) - 1) + ")");
                    }
                }
                if (!pass_applied) {
                    logPassWarning2D("no pass applied; using base camera texture");
                } else {
                    textureForSprite = inputTex;
                }
                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                glViewport(0, 0, win->w, win->h);
                glClear(GL_COLOR_BUFFER_BIT);
            }

            gl::ShaderProgram *activeShader;
            if (library.isBypassed()) {
                activeShader = &fshader;
            } else {
                activeShader = library.shader();
            }
            activeShader->useProgram();
            activeShader->setUniform("mv_matrix", glm::mat4(1.0f));
            activeShader->setUniform("proj_matrix", glm::mat4(1.0f));
            sprite.setShader(activeShader);
            sprite.setName("samp");
            sprite.draw(textureForSprite, 0, 0, win->w, win->h);
        }

        if (crossfadeActive) {
            applyCrossfade(win, fboTexture);
            if (crossfadeActive) {
                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                glViewport(0, 0, win->w, win->h);
                glClear(GL_COLOR_BUFFER_BIT);
                fshader.useProgram();
                fshader.setUniform("mv_matrix", glm::mat4(1.0f));
                fshader.setUniform("proj_matrix", glm::mat4(1.0f));
                sprite.setShader(&fshader);
                sprite.setName("samp");
                sprite.draw(crossfadeTexture, 0, 0, win->w, win->h);
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, win->w, win->h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        // In HDR windowed mode, flip by default to match headless orientation.
        // When --flip is set, invert this (don't flip) to show upside down.
        gl::ShaderProgram *display_shader = &fshader;
        int flip_y_uniform = 0;
        if (input_is_hdr && display_flip_shader.id() != 0) {
            // Apply flip unless --flip flag is set (which inverts the behavior)
            if (!flip_output) {
                display_shader = &display_flip_shader;
                flip_y_uniform = 1;
            }
        }

        display_shader->useProgram();
        display_shader->setUniform("mv_matrix", glm::mat4(1.0f));
        display_shader->setUniform("proj_matrix", glm::mat4(1.0f));
        if (flip_y_uniform == 1) {
            glUniform1i(glGetUniformLocation(display_shader->id(), "flip_y"), flip_y_uniform);
        }
        sprite.setShader(display_shader);
        sprite.draw(fboTexture, 0, 0, win->w, win->h);

        // Drawing the watermark / display-filter overlay into captureFBO would
        // pollute fboTexture and bleed into the next frame's crossfade
        // snapshot (beginCrossfade samples fboTexture). Save the un-watermarked
        // image now and restore it after the writer/snapshot readback so
        // recordings include the overlay but the on-screen / crossfade source
        // does not.
        const bool overlay_will_draw =
            writer.is_open() && waterFont.handle().has_value() &&
            (display_filter || enableWatermark);
        if (overlay_will_draw && preOverlayFBO != 0) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, captureFBO);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, preOverlayFBO);
            glBlitFramebuffer(0, 0, win->w, win->h, 0, 0, win->w, win->h,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        }

        int watermarkY = 10;
        if (display_filter && writer.is_open() && waterFont.handle().has_value()) {
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glViewport(0, 0, win->w, win->h);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            win->text.setColor({255, 0, 255, 255});
            int dfY = 10;
            const int lineH = waterFontSize + 4;
            std::string shaderName = library.getShaderNameByIndex(library.index());
            if (!shaderName.empty()) {
                win->text.printText_Solid(waterFont, 10, dfY, "Shader: " + shaderName);
                dfY += lineH;
            }
            if (shader_pass_enabled && !shader_pass_list.empty()) {
                std::string mpLine = "Multipass: ";
                for (size_t i = 0; i < shader_pass_list.size(); ++i) {
                    if (i > 0) mpLine += ", ";
                    std::string n = library.getShaderNameByIndex(shader_pass_list[i]);
                    mpLine += n.empty() ? std::to_string(shader_pass_list[i]) : n;
                }
                win->text.printText_Solid(waterFont, 10, dfY, mpLine);
                dfY += lineH;
            }
            if (gpu_filter_enabled && !gpu_filters.empty()) {
                std::string gpuLine = "GPU: ";
                for (size_t i = 0; i < gpu_filters.size(); ++i) {
                    if (i > 0) gpuLine += ", ";
                    gpuLine += gpu_filters[i].name;
                }
                win->text.printText_Solid(waterFont, 10, dfY, gpuLine);
                dfY += lineH;
            }
            glDisable(GL_BLEND);
            watermarkY = dfY;
        }
        if (enableWatermark && writer.is_open() && waterFont.handle().has_value()) {
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glViewport(0, 0, win->w, win->h);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            win->text.setColor({static_cast<Uint8>(watermark_r),
                                static_cast<Uint8>(watermark_g),
                                static_cast<Uint8>(watermark_b), 255});
            win->text.printText_Blended(waterFont, 10, watermarkY, watermark_text);
            glDisable(GL_BLEND);
        }

        bool needWriter = (writer.is_open() || snapshot_state > 0 || hdr_snapshot_state > 0 || raw_snapshot_state > 0 || tiff_snapshot_state > 0) && !isFrozen;

        bool has_snapshot_request = (snapshot_state > 0);
        bool has_hdr_snapshot_request = (hdr_snapshot_state > 0);
        bool has_raw_snapshot_request = (raw_snapshot_state > 0);
        bool has_tiff_snapshot_request = (tiff_snapshot_state > 0);
        if (needWriter && input_is_hdr
            && (writer.is_open() || has_snapshot_request || has_hdr_snapshot_request || has_raw_snapshot_request || has_tiff_snapshot_request)) {
            // HDR writer readback path. Bypasses the 8-bit PBO ring entirely
            // and reads 16-bit PQ-encoded BT.2020 RGBA from
            // @c hdr_encoded_texture via a synchronous glGetTexImage. The
            // resulting buffer (8 bytes/pixel) is pushed through the same
            // frame queue with @c FrameData::isHdr set so the writer thread
            // dispatches @c writer.write_hdr_rgba16() instead of the SDR
            // @c writer.write().
            //
            // Orientation: The HDR input frame is NOT pre-flipped (unlike SDR),
            // and the hdr_encode shader pass produces top-down output. If
            // flip_output is set, apply vertical flipping here to correct
            // the output orientation.
            std::vector<unsigned char> pixels;
            hdrReadback(fboTexture, win->w, win->h, pixels);

            // Apply vertical flip if requested (e.g., for HDR correction).
            if (flip_output) {
                const int row_bytes = win->w * 8; // 4 channels * 2 bytes
                std::vector<unsigned char> flipped_pixels(pixels.size());
                for (int y = 0; y < win->h; ++y) {
                    std::copy(pixels.begin() + y * row_bytes,
                              pixels.begin() + (y + 1) * row_bytes,
                              flipped_pixels.begin() + (win->h - 1 - y) * row_bytes);
                }
                pixels = std::move(flipped_pixels);
            }

            if (writer.is_open() || has_hdr_snapshot_request || has_raw_snapshot_request) {
                FrameData fd;
                fd.pixels = pixels;
                fd.width = win->w;
                fd.height = win->h;
                fd.isHdr = true;
                fd.hdrTrc = input_hdr_trc;
                fd.isSnapshot = has_hdr_snapshot_request;
                fd.isWebPSnapshot = has_hdr_snapshot_request;
                fd.isRawSnapshot = has_raw_snapshot_request;

                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    queueCondVar.wait(lock, [this] { return frameQueue.size() < 30 || !writerRunning; });
                    frameQueue.push(std::move(fd));
                }
                queueCondVar.notify_one();
            }

            if (has_tiff_snapshot_request) {
                FrameData tiff_fd;
                tiff_fd.pixels = pixels;
                tiff_fd.width = win->w;
                tiff_fd.height = win->h;
                tiff_fd.isHdr = true;
                tiff_fd.hdrTrc = input_hdr_trc;
                tiff_fd.isSnapshot = false;
                tiff_fd.isWebPSnapshot = false;
                tiff_fd.isRawSnapshot = false;
                tiff_fd.isTiffSnapshot = true;

                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    queueCondVar.wait(lock, [this] { return frameQueue.size() < 30 || !writerRunning; });
                    frameQueue.push(std::move(tiff_fd));
                }
                queueCondVar.notify_one();
            }

            if (has_snapshot_request) {
                // In HDR mode, normal PNG snapshots intentionally use the
                // non-HDR 8-bit readback path so viewers can open them as
                // standard SDR PNGs.
                std::vector<unsigned char> sdr_pixels(static_cast<size_t>(win->w) * static_cast<size_t>(win->h) * 4);
                glBindTexture(GL_TEXTURE_2D, fboTexture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, sdr_pixels.data());
                glBindTexture(GL_TEXTURE_2D, 0);

                std::vector<unsigned char> flipped_pixels(static_cast<size_t>(win->w) * static_cast<size_t>(win->h) * 4);
                for (int y = 0; y < win->h; ++y) {
                    const int src_row_start = y * win->w * 4;
                    const int dest_row_start = (win->h - 1 - y) * win->w * 4;
                    std::copy(sdr_pixels.begin() + src_row_start,
                              sdr_pixels.begin() + src_row_start + (win->w * 4),
                              flipped_pixels.begin() + dest_row_start);
                }

                FrameData sdr_fd;
                sdr_fd.pixels = std::move(flipped_pixels);
                sdr_fd.width = win->w;
                sdr_fd.height = win->h;
                sdr_fd.isHdr = false;
                sdr_fd.isSnapshot = true;
                sdr_fd.isWebPSnapshot = false;
                sdr_fd.isRawSnapshot = false;

                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    queueCondVar.wait(lock, [this] { return frameQueue.size() < 30 || !writerRunning; });
                    frameQueue.push(std::move(sdr_fd));
                }
                queueCondVar.notify_one();
            }

            if (has_snapshot_request) {
                snapshot_state = 0;
            }

            if (has_hdr_snapshot_request) {
                hdr_snapshot_state = 0;
            }

            if (has_raw_snapshot_request) {
                raw_snapshot_state = 0;
            }

            if (has_tiff_snapshot_request) {
                tiff_snapshot_state = 0;
            }
        } else if (needWriter) {

            if (snapshot_state == 1 || hdr_snapshot_state == 1 || raw_snapshot_state == 1 || tiff_snapshot_state == 1) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pboIndex]);
                glBindTexture(GL_TEXTURE_2D, fboTexture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);

                pboIndex = (pboIndex + 1) % 2;
                pboNextIndex = (pboNextIndex + 1) % 2;
                if (snapshot_state == 1) snapshot_state = 2;
                if (hdr_snapshot_state == 1) hdr_snapshot_state = 2;
                if (raw_snapshot_state == 1) raw_snapshot_state = 2;
                if (tiff_snapshot_state == 1) tiff_snapshot_state = 2;
            } else {
                bool is_snapshot_frame = (snapshot_state == 2);
                bool is_webp_snapshot_frame = (hdr_snapshot_state == 2);
                bool is_raw_snapshot_frame = (raw_snapshot_state == 2);
                bool is_tiff_snapshot_frame = (tiff_snapshot_state == 2);

                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pboIndex]);
                glBindTexture(GL_TEXTURE_2D, fboTexture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

                if (writer.is_open() || is_snapshot_frame || is_webp_snapshot_frame || is_raw_snapshot_frame || is_tiff_snapshot_frame) {
                    bool used_zero_copy = false;

#ifdef ACMX2_WITH_CUDA
                    if (writer.is_open() && !is_snapshot_frame && !is_webp_snapshot_frame && !is_raw_snapshot_frame && !is_tiff_snapshot_frame && recordCudaPboResources[pboNextIndex]) {
                        cudaGraphicsResource *resource = recordCudaPboResources[pboNextIndex];
                        void *devPtr = nullptr;
                        size_t mappedBytes = 0;

                        CHECK_CUDA(cudaGraphicsMapResources(1, &resource, 0));
                        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedBytes, resource));

                        const size_t requiredBytes = static_cast<size_t>(win->w) * static_cast<size_t>(win->h) * 4;
                        if (devPtr && mappedBytes >= requiredBytes) {
                            used_zero_copy = writer.write_cuda_rgba(devPtr, static_cast<int>(win->w) * 4, true);
                        }

                        CHECK_CUDA(cudaGraphicsUnmapResources(1, &resource, 0));
                    }
#endif

                    if (!used_zero_copy) {
                        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pboNextIndex]);
                        GLubyte *src = static_cast<GLubyte *>(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));

                        if (src) {
                            std::vector<unsigned char> pixels(win->w * win->h * 4);
                            std::memcpy(pixels.data(), src, pixels.size());
                            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

                            std::vector<unsigned char> flipped_pixels(win->w * win->h * 4);
                            for (int y = 0; y < win->h; ++y) {
                                int src_row_start = y * win->w * 4;
                                int dest_row_start = (win->h - 1 - y) * win->w * 4;
                                std::copy(pixels.begin() + src_row_start,
                                          pixels.begin() + src_row_start + (win->w * 4),
                                          flipped_pixels.begin() + dest_row_start);
                            }

                            FrameData fd;
                            fd.pixels = std::move(flipped_pixels);
                            fd.width = win->w;
                            fd.height = win->h;
                            fd.isSnapshot = (is_snapshot_frame || is_webp_snapshot_frame);
                            fd.isWebPSnapshot = is_webp_snapshot_frame;
                            fd.isRawSnapshot = is_raw_snapshot_frame;
                            fd.isTiffSnapshot = is_tiff_snapshot_frame;

                            if (is_snapshot_frame) {
                                snapshot_state = 0;
                            }
                            if (is_webp_snapshot_frame) {
                                hdr_snapshot_state = 0;
                            }
                            if (is_raw_snapshot_frame) {
                                raw_snapshot_state = 0;
                            }
                            if (is_tiff_snapshot_frame) {
                                tiff_snapshot_state = 0;
                            }

                            {
                                std::unique_lock<std::mutex> lock(queueMutex);
                                bool is_camera_mode = filename.empty() && graphic.empty();
                                if (is_camera_mode && !is_snapshot_frame && !is_webp_snapshot_frame && !is_raw_snapshot_frame && !is_tiff_snapshot_frame) {
                                    if (frameQueue.size() > 30) {
                                        frames_dropped++;
                                        frameQueue.pop();
                                    }
                                } else {
                                    queueCondVar.wait(lock, [this] { return frameQueue.size() < 30 || !writerRunning; });
                                }
                                frameQueue.push(std::move(fd));
                            }
                            queueCondVar.notify_one();
                        }
                    }

                    if (used_zero_copy) {
                        snapshot_state = 0;
                        hdr_snapshot_state = 0;
                        raw_snapshot_state = 0;
                        tiff_snapshot_state = 0;
                    }
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);
                pboIndex = (pboIndex + 1) % 2;
                pboNextIndex = (pboNextIndex + 1) % 2;
            }
        }

        // Restore fboTexture so the next frame (and any crossfade snapshot)
        // sees the un-watermarked frame; the writer has already captured the
        // overlaid pixels above.
        if (overlay_will_draw && preOverlayFBO != 0) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, preOverlayFBO);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, captureFBO);
            glBlitFramebuffer(0, 0, win->w, win->h, 0, 0, win->w, win->h,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, win->w, win->h);

        if (!counter_disabled && overlayFont.handle().has_value()) {
            fpsFrameCount++;
            auto currentTime = std::chrono::steady_clock::now();
            auto fpsDelta = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - fpsLastTime).count();
            if (fpsDelta >= 500) {
                displayFPS = (fpsFrameCount * 1000.0) / fpsDelta;
                fpsFrameCount = 0;
                fpsLastTime = currentTime;
            }

            std::string timerStr = getTimeString();
            std::ostringstream fpsStr;
            fpsStr << std::fixed << std::setprecision(1) << displayFPS << " FPS";
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            win->text.setColor({0, 0, 255, 255});
            win->text.printText_Blended(overlayFont, 10, 10, cached_shader_name);
            int overlayY = 40;
            if (gpu_filter_enabled && !gpu_filters.empty()) {
                win->text.setColor({255, 0, 255, 255});
                std::string gpuLine = "GPU: ";
                for (size_t i = 0; i < gpu_filters.size(); ++i) {
                    if (i > 0) gpuLine += ", ";
                    gpuLine += gpu_filters[i].name;
                }
                win->text.printText_Blended(overlayFont, 10, overlayY, gpuLine);
                overlayY += 30;
            }
            win->text.setColor({255, 255, 255, 255});
            win->text.printText_Blended(overlayFont, 10, overlayY, timerStr);
            win->text.printText_Blended(overlayFont, 10, overlayY + 30, fpsStr.str());
            win->text.setColor({128, 128, 128, 255});
            win->text.printText_Blended(overlayFont, 10, overlayY + 55, "F9: Toggle overlay");
#ifdef MIDI_ENABLED
            drawMidiOverlay(win, overlayFont, overlayY + 80);
#endif
            glDisable(GL_BLEND);
        }

        static auto lastUpdate = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();

        if (!graphic.empty()) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() >= 250) {
                std::string timeStr = getTimeString();
                int64_t currentFrames = getFrameCount();

                std::ostringstream stream;
                stream << "ACMX2 - Graphics Mode - "
                       << timeStr
                       << " [" << currentFrames << " frames]";
                if (writer.is_open()) {
                    stream << " (Recording)";
                }
                win->setWindowTitle(stream.str());
                lastUpdate = now;
            }

        } else if (!filename.empty()) {
            if (use_ffmpeg_reader) {
                frame_counter = static_cast<unsigned int>(std::max<int64_t>(0, ffmpeg_reader.getCurrentFrame()));
            } else if (cap.isOpened()) {
                frame_counter = static_cast<unsigned int>(cap.get(cv::CAP_PROP_POS_FRAMES));
            }

            if (silent_mode && totalFrames > 0.0) {
                int current_percent = static_cast<int>((static_cast<double>(frame_counter) / totalFrames) * 100.0);
                // Emit progress at least every ~500 ms, and on every percent
                // boundary, so the user sees continuous headless progress
                // even when stdout is piped / redirected / captured by a
                // logger (no TTY, so carriage-return tricks don't render).
                // We always write newline-terminated lines for headless mode
                // to guarantee each update is flushed through line-buffered
                // pipes and visible in real time.
                static auto lastProgressEmit = std::chrono::steady_clock::now();
                bool percent_changed = (current_percent > last_progress_percent && current_percent <= 100);
                bool time_elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressEmit).count() >= 500);
                if (percent_changed || time_elapsed) {
                    if (percent_changed) {
                        last_progress_percent = current_percent;
                    }
                    lastProgressEmit = now;
                    int64_t frames_written = writer.is_open() ? writer.get_frame_count() : 0;
                    double elapsed_secs = static_cast<double>(frame_counter) / fps;
                    uint64_t hours = static_cast<uint64_t>(elapsed_secs / 3600);
                    uint64_t minutes = static_cast<uint64_t>(elapsed_secs / 60) % 60;
                    uint64_t seconds = static_cast<uint64_t>(elapsed_secs) % 60;

                    std::cout << "acmx2: [" << std::setw(3) << current_percent << "%] "
                              << "Frame " << frame_counter << "/" << static_cast<int>(totalFrames)
                              << " | Written: " << frames_written
                              << " | Time: " << std::setfill('0') << std::setw(2) << hours << ":"
                              << std::setfill('0') << std::setw(2) << minutes << ":"
                              << std::setfill('0') << std::setw(2) << seconds
                              << std::setfill(' ') << "\n" << std::flush;
                }
            } else if (silent_mode) {
                // Fallback: input reports unknown frame count (e.g. some MKV
                // / streaming containers). No percentage possible, so emit
                // an elapsed-style progress line every 500 ms with what we
                // do know: current frame number, frames written, elapsed
                // time based on the input FPS.
                static auto lastProgressEmitUnk = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressEmitUnk).count() >= 500) {
                    lastProgressEmitUnk = now;
                    int64_t frames_written = writer.is_open() ? writer.get_frame_count() : 0;
                    double elapsed_secs = (fps > 0.0) ? static_cast<double>(frame_counter) / fps : 0.0;
                    uint64_t hours = static_cast<uint64_t>(elapsed_secs / 3600);
                    uint64_t minutes = static_cast<uint64_t>(elapsed_secs / 60) % 60;
                    uint64_t seconds = static_cast<uint64_t>(elapsed_secs) % 60;

                    std::cout << "acmx2: [  ?%] "
                              << "Frame " << frame_counter << "/?"
                              << " | Written: " << frames_written
                              << " | Time: " << std::setfill('0') << std::setw(2) << hours << ":"
                              << std::setfill('0') << std::setw(2) << minutes << ":"
                              << std::setfill('0') << std::setw(2) << seconds
                              << std::setfill(' ') << "\n" << std::flush;
                }
            }

            if (!silent_mode && std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count() >= 1) {
                if (totalFrames <= 0.0 && !use_ffmpeg_reader && cap.isOpened()) {
                    totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
                }
                std::string timeStr = getTimeString();
                std::ostringstream stream;
                stream << "ACMX2 - ["
                       << frame_counter << "/"
                       << static_cast<int>(totalFrames) << "] - "
                       << timeStr << " - Video Mode";
                if (writer.is_open()) {
                    stream << " (Recording)";
                }
                win->setWindowTitle(stream.str());
                lastUpdate = now;
            }

        } else if (cap.isOpened() && filename.empty()) {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count() >= 1) {
                std::string timeStr = getTimeString();
                int64_t currentFrames = getFrameCount();
                std::ostringstream stream;
                stream << "ACMX2 - Capture Mode - "
                       << timeStr
                       << " [" << currentFrames << " frames]";
                if (writer.is_open()) {
                    stream << " (Recording)";
                }
                win->setWindowTitle(stream.str());
                lastUpdate = now;
            }
        }
        if (playlist_enabled && autopilot_enabled && autopilot_frames > 0) {
            if (++autopilot_counter >= autopilot_frames) {
                autopilot_counter = 0;
                if (autopilot_sequential)
                    autopilotSequentialAdvance(win);
                else
                    autopilotRandomSwitch(win);
            }
        }
        frame_counter++;
    }

    /**
     * @brief Format the current session time as `HH:MM:SS`.
     *
     * When a writer is open the time is based on the number of
     * frames actually written (at the configured FPS).  For video
     * file input, a percentage prefix is prepended.
     *
     * @return Formatted time string for display.
     */
    std::string getTimeString() {
        int64_t frameCount = 0;
        double timeSeconds = 0.0;

        if (writer.is_open()) {
            frameCount = writer.get_frame_count();
            timeSeconds = (fps > 0.0) ? static_cast<double>(frameCount) / fps : 0.0;
        } else {
            frameCount = static_cast<int64_t>(frame_counter);
            timeSeconds = (fps > 0.0) ? static_cast<double>(frameCount) / fps : 0.0;
        }

        uint64_t hours = static_cast<uint64_t>(timeSeconds / 3600);
        uint64_t minutes = static_cast<uint64_t>(timeSeconds / 60) % 60;
        uint64_t seconds = static_cast<uint64_t>(timeSeconds) % 60;

        std::ostringstream timerStr;
        if (!filename.empty() && totalFrames > 0.0) {
            double currentFrame = static_cast<double>(frame_counter);
            double percentage = (currentFrame / totalFrames) * 100.0;
            timerStr << std::fixed << std::setprecision(1) << percentage << "% - ";
        }

        timerStr << std::setfill('0') << std::setw(2) << hours << ":"
                 << std::setfill('0') << std::setw(2) << minutes << ":"
                 << std::setfill('0') << std::setw(2) << seconds;
        return timerStr.str();
    }

    /**
     * @brief Return the current frame count (writer count or display count).
     * @return Number of frames written if recording, else the display counter.
     */
    int64_t getFrameCount() {
        if (writer.is_open()) {
            return writer.get_frame_count();
        }
        return static_cast<int64_t>(frame_counter);
    }

    /**
     * @brief Handle SDL keyboard events (shader navigation, mode toggles, etc.).
     *
     * Key bindings (SDL_KEYUP):
     * - Up/Down: Previous/next shader (or playlist entry if playlist enabled,
     *   or change main shader with crossfade while in random multipass mode).
     * - Left/Right: Previous/next GPU CUDA filter.
     * - Space: Toggle shader bypass.
     * - P: Toggle playlist mode or pause video.
     * - L: Freeze frame (stop updating texture but keep time advancing).
    * - Z: Take a PNG snapshot (8-bit non-HDR readback when HDR input is active).
    * - 5: Take an HDR PNG snapshot (HDR mode only).
     * - T: Toggle active time.  Q: Toggle audio time.  Home: Toggle audio delta.
    * - V: Toggle view rotation (3D).  O: Oscillation.  C: Wave.
    * - X: Reset camera.  Ctrl+X: Quit immediately without audio mux/transfer.
     * - 3: Toggle 2D/3D mode.  M: Toggle multi-pass.  E: Watermark.
     * - R: Toggle random multipass mode (generates 1-5 random shader chain).
     * - G: Generate new random shader chain (while in random multipass mode).
     * - H: Generate long random shader chain up to 10 (while in random multipass mode).
     * - F: Generate short random shader pair of 2 (while in random multipass mode).
     * - F9: Toggle HUD overlay visibility.
     *
     * Key bindings (SDL_KEYDOWN):
     * - U/I: Manual time step forward/backward.
     * - Insert/Delete: Audio sensitivity +/-.
     * - End: Toggle spectrum sensitivity scaling on/off.
     * - Page Up/Down: Time speed (handled in draw() via key-state polling).
     *
     * @param win Hosting GLWindow.
     * @param e   The SDL event to process.
     */
    virtual void event(gl::GLWindow *win, SDL_Event &e) override {
        switch (e.type) {
        case SDL_KEYUP:
            switch (e.key.keysym.sym) {
            case SDLK_UP:
                if (shaderLocked) break;
                if (random_multipass_mode) {
                    beginCrossfade(win);
                    library.dec();
                    mx::system_out << "acmx2: Random mode shader: " << library.getFullShaderName(shader_pass_list) << "\n";
                    fflush(stdout);
                } else if (playlist_enabled && !playlist_tree.empty()) {
                    if (playlist_index > 0) {
                        beginCrossfade(win);
                        --playlist_index;
                        const auto &node = playlist_tree[playlist_index];
                        shader_pass_list = node.shader_indices;
                        shader_pass_enabled = !shader_pass_list.empty();
                        mx::system_out << "acmx2: Playlist Node: " << node.name
                                       << " [" << node.shader_indices.size() << " shaders] ("
                                       << (playlist_index + 1) << "/" << playlist_tree.size() << ")\n";
                        fflush(stdout);
                    }
                } else if (playlist_enabled && !playlist_indices.empty()) {
                    if (playlist_index > 0) {
                        beginCrossfade(win);
                        --playlist_index;
                    }
                    library.setIndex(playlist_indices[playlist_index]);
                    mx::system_out << "acmx2: Playlist [" << playlist_index << "/" << playlist_indices.size() << "]\n";
                    fflush(stdout);
                } else {
                    library.dec();
                }
                if (is3d_enabled)
                    cube.setShaderProgram(library.shader());
                sprite.setShader(library.shader());
                updateShaderNameCache();
                break;
            case SDLK_DOWN:
                if (shaderLocked) break;
                if (random_multipass_mode) {
                    beginCrossfade(win);
                    library.inc();
                    mx::system_out << "acmx2: Random mode shader: " << library.getFullShaderName(shader_pass_list) << "\n";
                    fflush(stdout);
                } else if (playlist_enabled && !playlist_tree.empty()) {
                    if (playlist_index + 1 < static_cast<int>(playlist_tree.size())) {
                        beginCrossfade(win);
                        ++playlist_index;
                        const auto &node = playlist_tree[playlist_index];
                        shader_pass_list = node.shader_indices;
                        shader_pass_enabled = !shader_pass_list.empty();
                        mx::system_out << "acmx2: Playlist Node: " << node.name
                                       << " [" << node.shader_indices.size() << " shaders] ("
                                       << (playlist_index + 1) << "/" << playlist_tree.size() << ")\n";
                        fflush(stdout);
                    }
                } else if (playlist_enabled && !playlist_indices.empty()) {
                    if (playlist_index + 1 < static_cast<int>(playlist_indices.size())) {
                        beginCrossfade(win);
                        ++playlist_index;
                    }
                    library.setIndex(playlist_indices[playlist_index]);
                    mx::system_out << "acmx2: Playlist [" << playlist_index << "/" << playlist_indices.size() << "]\n";
                    fflush(stdout);
                } else {
                    library.inc();
                }
                if (is3d_enabled)
                    cube.setShaderProgram(library.shader());
                sprite.setShader(library.shader());
                updateShaderNameCache();
                break;
            case SDLK_LEFT:
                if (gpu_filter_enabled && !gpu_filters.empty()) {
                    gpu_current_filter_index--;
                    if (gpu_current_filter_index < 0)
                        gpu_current_filter_index = ac_gpu::AC_FILTER_MAX - 1;
                    gpu_filters.clear();
                    gpu_filters.push_back({gpu_current_filter_index, ac_gpu::filters[gpu_current_filter_index].name});
                    gpu_filtersChanged = true;
                    mx::system_out << "acmx2: GPU Filter: " << ac_gpu::filters[gpu_current_filter_index].name << " [" << gpu_current_filter_index << "]\n";
                    fflush(stdout);
                }
                break;
            case SDLK_RIGHT:
                if (gpu_filter_enabled && !gpu_filters.empty()) {
                    gpu_current_filter_index++;
                    if (gpu_current_filter_index >= ac_gpu::AC_FILTER_MAX)
                        gpu_current_filter_index = 0;
                    gpu_filters.clear();
                    gpu_filters.push_back({gpu_current_filter_index, ac_gpu::filters[gpu_current_filter_index].name});
                    gpu_filtersChanged = true;
                    mx::system_out << "acmx2: GPU Filter: " << ac_gpu::filters[gpu_current_filter_index].name << " [" << gpu_current_filter_index << "]\n";
                    fflush(stdout);
                }
                break;
            case SDLK_SPACE:
                library.toggleBypass();
                updateShaderNameCache();
                break;
            case SDLK_p:
                if (!playlist_tree.empty()) {
                    playlist_enabled = !playlist_enabled;
                    if (playlist_enabled) {
                        beginCrossfade(win);
                        saved_pass_list = shader_pass_list;
                        saved_pass_enabled = shader_pass_enabled;
                        if (playlist_index < 0 || playlist_index >= static_cast<int>(playlist_tree.size()))
                            playlist_index = 0;
                        const auto &node = playlist_tree[playlist_index];
                        shader_pass_list = node.shader_indices;
                        shader_pass_enabled = !shader_pass_list.empty();
                        if (is3d_enabled)
                            cube.setShaderProgram(library.shader());
                        sprite.setShader(library.shader());
                        updateShaderNameCache();
                        mx::system_out << "acmx2: Playlist mode enabled - Node: " << node.name
                                       << " [" << node.shader_indices.size() << " shaders] ("
                                       << (playlist_index + 1) << "/" << playlist_tree.size() << ")\n";
                    } else {
                        beginCrossfade(win);
                        shader_pass_list = saved_pass_list;
                        shader_pass_enabled = saved_pass_enabled;
                        if (is3d_enabled)
                            cube.setShaderProgram(library.shader());
                        sprite.setShader(library.shader());
                        updateShaderNameCache();
                        mx::system_out << "acmx2: Playlist mode disabled - restored original"
                                       << (shader_pass_enabled ? " multi-pass" : " single shader") << "\n";
                    }
                    fflush(stdout);
                } else if (!playlist_indices.empty()) {
                    playlist_enabled = !playlist_enabled;
                    if (playlist_enabled) {
                        beginCrossfade(win);
                        if (playlist_index >= 0 && playlist_index < static_cast<int>(playlist_indices.size())) {
                            library.setIndex(playlist_indices[playlist_index]);
                            if (is3d_enabled)
                                cube.setShaderProgram(library.shader());
                            sprite.setShader(library.shader());
                            updateShaderNameCache();
                        }
                        mx::system_out << "acmx2: Playlist mode enabled [" << playlist_indices.size() << " shaders]\n";
                    } else {
                        beginCrossfade(win);
                        mx::system_out << "acmx2: Playlist mode disabled\n";
                    }
                    fflush(stdout);
                } else if (!filename.empty() || !graphic.empty()) {
                    isPaused = !isPaused;
                    mx::system_out << "acmx2: paused: " << ((isPaused == true) ? "enabled" : "disabled") << "\n";
                    fflush(stdout);
                    fflush(stderr);
                }
                break;
            case SDLK_l:
                if (!filename.empty() || !graphic.empty()) {
                    isFrozen = !isFrozen;
                    mx::system_out << "acmx2: frozen: " << ((isFrozen == true) ? "enabled" : "disabled") << "\n";
                    fflush(stdout);
                    fflush(stderr);
                }
                break;
            case SDLK_k:
                shaderLocked = !shaderLocked;
                mx::system_out << "acmx2: Shader lock: " << (shaderLocked ? "enabled" : "disabled") << "\n";
                fflush(stdout);
                break;
            case SDLK_j:
                if (!playlist_enabled) {
                    mx::system_out << "acmx2: Autopilot requires playlist mode (press P first)\n";
                    fflush(stdout);
                    break;
                }
                if (playlist_tree.empty() && playlist_indices.empty()) {
                    mx::system_out << "acmx2: Autopilot has no playlist entries\n";
                    fflush(stdout);
                    break;
                }
                autopilot_enabled = !autopilot_enabled;
                autopilot_counter = 0;
                if (autopilot_enabled) {
                    autopilot_sequential = false;
                    if (autopilot_frames <= 0) {
                        autopilot_frames = 300; // sensible default if user never set it
                    }
                }
                mx::system_out << "acmx2: Autopilot " << (autopilot_enabled ? "enabled (random)" : "disabled")
                               << " (every " << autopilot_frames << " frames)\n";
                fflush(stdout);
                break;
            case SDLK_y:
                if (!playlist_enabled) {
                    mx::system_out << "acmx2: Sequential autopilot requires playlist mode (press P first)\n";
                    fflush(stdout);
                    break;
                }
                if (playlist_tree.empty() && playlist_indices.empty()) {
                    mx::system_out << "acmx2: Sequential autopilot has no playlist entries\n";
                    fflush(stdout);
                    break;
                }
                if (autopilot_enabled && autopilot_sequential) {
                    autopilot_enabled = false;
                    autopilot_sequential = false;
                    mx::system_out << "acmx2: Autopilot disabled\n";
                } else {
                    autopilot_enabled = true;
                    autopilot_sequential = true;
                    autopilot_counter = 0;
                    if (autopilot_frames <= 0) {
                        autopilot_frames = 300;
                    }
                    mx::system_out << "acmx2: Autopilot enabled (sequential) (every "
                                   << autopilot_frames << " frames)\n";
                }
                fflush(stdout);
                break;
            case SDLK_z:
                if (snapshot_state == 0) {
                    snapshot_state = 1;
                }
                break;
            case SDLK_4:
#ifdef ACMX2_WITH_TIFF
                if (tiff_snapshot_state == 0) {
                    tiff_snapshot_state = 1;
                }
#endif
                break;
            case SDLK_5:
                if (hdr_snapshot_state == 0) {
                    hdr_snapshot_state = 1;
                }
                break;
            case SDLK_6:
                if (raw_snapshot_state == 0) {
                    raw_snapshot_state = 1;
                }
                break;
#ifdef AUDIO_ENABLED
            case SDLK_t:
                library.activeTime(!library.timeActive());
                break;
            case SDLK_q:
                library.audioTime(!library.timeAudio());
                break;
            case SDLK_HOME:
                library.toggleAudioDelta();
                break;
#endif
            case SDLK_v:
                viewRotationActive = !viewRotationActive;
                mx::system_out << "acmx2: View rotation: " << (viewRotationActive ? "enabled" : "disabled") << "\n";
                fflush(stdout);
                break;
            case SDLK_x:
                if ((e.key.keysym.mod & KMOD_CTRL) != 0) {
                    requestStopNoMux();
                    mx::system_out << "acmx2: Ctrl+X pressed, exiting without audio mux\n";
                    fflush(stdout);
                    win->quit();
                    break;
                }
                cameraDistance = 0.0f;
                mx::system_out << "acmx2: Camera distance reset\n";
                fflush(stdout);
                break;
            case SDLK_o:
                oscillateScale = !oscillateScale;
                mx::system_out << "acmx2: Scale oscillation "
                               << (oscillateScale ? "enabled" : "disabled") << "\n";
                fflush(stdout);
                break;
            case SDLK_c:
                waveActive = !waveActive;
                mx::system_out << "acmx2: Wave effect "
                               << (waveActive ? "enabled" : "disabled") << "\n";
                fflush(stdout);
                break;
            case SDLK_e:
                enableWatermark = !enableWatermark;
                mx::system_out << "acmx2: Watermark "
                               << (enableWatermark ? "enabled" : "disabled") << "\n";
                fflush(stdout);
                break;
            case SDLK_m:
                if (!shader_pass_list.empty()) {
                    shader_pass_enabled = !shader_pass_enabled;
                    updateShaderNameCache();
                    mx::system_out << "acmx2: Multi-shader pass "
                                   << (shader_pass_enabled ? "enabled" : "disabled") << "\n";
                    fflush(stdout);
                } else {
                    mx::system_out << "acmx2: No shader pass list defined (use --shader-pass)\n";
                    fflush(stdout);
                }
                break;
            case SDLK_3:
                if (!library.isDualMode()) {
                    mx::system_out << "acmx2: Cannot switch to 3D mode - 3D shaders not compiled (use --enable-3d at startup)\n";
                    fflush(stdout);
                } else if (cube.meshes.empty()) {
                    mx::system_out << "acmx2: Cannot switch to 3D mode - no model loaded (use --enable-3d)\n";
                    fflush(stdout);
                } else {
                    is3d_enabled = !is3d_enabled;
                    library.is3D(is3d_enabled);
                    updateShaderNameCache();
                    mx::system_out << "acmx2: " << (is3d_enabled ? "3D" : "2D") << " mode "
                                   << (is3d_enabled ? "enabled" : "disabled") << "\n";
                    fflush(stdout);
                }
                break;
            case SDLK_r:
                if (std::get<0>(flib) == 0) {
                    mx::system_out << "acmx2: Random multipass mode not available in single shader mode\n";
                    fflush(stdout);
                } else if (!random_multipass_mode) {
                    random_multipass_mode = true;
                    saved_pass_list_before_random = shader_pass_list;
                    saved_pass_enabled_before_random = shader_pass_enabled;
                    saved_shader_index_before_random = library.index();
                    generateRandomMultipass(win);
                    mx::system_out << "acmx2: Random multipass mode enabled\n";
                    fflush(stdout);
                } else {
                    random_multipass_mode = false;
                    beginCrossfade(win);
                    shader_pass_list = saved_pass_list_before_random;
                    shader_pass_enabled = saved_pass_enabled_before_random;
                    library.setIndex(saved_shader_index_before_random);
                    if (is3d_enabled)
                        cube.setShaderProgram(library.shader());
                    sprite.setShader(library.shader());
                    updateShaderNameCache();
                    mx::system_out << "acmx2: Random multipass mode disabled - restored original\n";
                    fflush(stdout);
                }
                break;
            case SDLK_g:
                if (random_multipass_mode) {
                    generateRandomMultipass(win);
                } else {
                    mx::system_out << "acmx2: Press R first to enable random multipass mode\n";
                    fflush(stdout);
                }
                break;
            case SDLK_h:
                if (random_multipass_mode) {
                    generateRandomMultipassLong(win);
                } else {
                    mx::system_out << "acmx2: Press R first to enable random multipass mode\n";
                    fflush(stdout);
                }
                break;
            case SDLK_f:
                if (random_multipass_mode) {
                    generateRandomMultipassShort(win);
                } else {
                    mx::system_out << "acmx2: Press R first to enable random multipass mode\n";
                    fflush(stdout);
                }
                break;
            }
            break;
        case SDL_KEYDOWN:
            switch (e.key.keysym.sym) {
            case SDLK_u:
                library.incTime(0.05f);
                break;
            case SDLK_i:
                library.decTime(0.05f);
                break;
#ifdef AUDIO_ENABLED
            case SDLK_INSERT: {
                float s = get_sense() + 0.1f;
                if (s > 5.0f) s = 5.0f;
                set_sense(s);
                mx::system_out << "acmx2: Audio sensitivity increased to " << s << "\n";
                fflush(stdout);
                break;
            }
            case SDLK_DELETE: {
                float s = get_sense() - 0.1f;
                if (s < 0.1f) s = 0.1f;
                set_sense(s);
                mx::system_out << "acmx2: Audio sensitivity decreased to " << s << "\n";
                fflush(stdout);
                break;
            }
            case SDLK_END:
                spectrum_scale_by_sense = !spectrum_scale_by_sense;
                mx::system_out << "acmx2: Spectrum sensitivity scaling "
                               << (spectrum_scale_by_sense ? "enabled" : "disabled") << "\n";
                fflush(stdout);
                break;
#endif
            case SDLK_F9:
                counter_disabled = !counter_disabled;
                mx::system_out << "acmx2: Overlay " << (counter_disabled ? "hidden" : "shown") << " (F9)\n";
                fflush(stdout);
                break;
            }
            break;
        }
        library.event(e);
    }

  private:
    unsigned int frame_counter = 0;
    unsigned int written_frame_counter = 0;
    std::string crf = "23";
    EncodeOptions encode_opts{};
    std::string prefix_path;
    std::string filename, ofilename, graphic;
    int camera_index = 0;
    std::tuple<int, std::string, int> flib;
    std::optional<cv::Size> sizev, sizec;
    ShaderLibrary library;
    Writer writer;
    double fps = 30;
    bool repeat = false;
    bool full = false;
    int snapshot_state = 0;
    int hdr_snapshot_state = 0;
    int raw_snapshot_state = 0;
    int tiff_snapshot_state = 0;
    double totalFrames = 0;
    cv::VideoCapture cap;
    FFMpegVideoReader ffmpeg_reader;
    bool use_ffmpeg_reader = false;
    cv::Mat graphic_frame;
    gl::GLSprite sprite;
    GLuint camera_texture = 0;
    GLuint captureFBO = 0;
    GLuint fboTexture = 0;
    GLuint preOverlayFBO = 0;     ///< FBO whose color attachment is preOverlayTexture; used to save/restore fboTexture around the overlay+writer step.
    GLuint preOverlayTexture = 0; ///< Snapshot of fboTexture taken before drawing the watermark/display-filter overlay, used to restore the un-watermarked image after the writer readback so the next frame's crossfade snapshot does not pick up the overlay.
    GLuint depthBuffer = 0;
    GLuint passFBO[2] = {0, 0};
    GLuint passTexture[2] = {0, 0};
    GLuint crossfadeFBO = 0;                                  ///< FBO used for the crossfade compositing pass.
    GLuint crossfadeTexture = 0;                               ///< Colour attachment of @c crossfadeFBO (blended output).
    GLuint crossfadePrevTexture = 0;                           ///< Snapshot of the previous frame used as the blend source.

    // --- HDR-mode GL resources ----------------------------------------------
    // Only allocated / used when @ref input_is_hdr is true. SDR path is
    // entirely unchanged. Internal formats: GL_RGBA16 for textures that
    // hold PQ/HLG-encoded normalised values (source upload + pre-encode
    // readback target), GL_RGBA16F for linear-BT.2020 intermediates.
    bool input_is_hdr = false;                                 ///< Active HDR pipeline for this input.
    int input_hdr_trc = 0;                                     ///< AVColorTransferCharacteristic (PQ/HLG/BT2020).
    int hdr_upload_tex_w = 0;                                  ///< Current GL size of @ref camera_texture in HDR upload mode.
    int hdr_upload_tex_h = 0;                                  ///< Current GL size of @ref camera_texture in HDR upload mode.
    int hdr_resource_w = 0;                                    ///< Width of HDR intermediate/encoded textures.
    int hdr_resource_h = 0;                                    ///< Height of HDR intermediate/encoded textures.
    GLuint hdr_linear_video_texture = 0;                       ///< GL_RGBA16F: PQ/HLG-decoded linear BT.2020 video.
    GLuint hdr_linear_video_fbo = 0;                           ///< FBO writing into @ref hdr_linear_video_texture.
    GLuint hdr_encoded_texture = 0;                            ///< GL_RGBA16: final PQ-re-encoded output for readback.
    GLuint hdr_encoded_fbo = 0;                                ///< FBO writing into @ref hdr_encoded_texture.
    gl::ShaderProgram hdr_decode_shader;                       ///< PQ/HLG -> linear BT.2020 fullscreen pass.
    gl::ShaderProgram hdr_encode_shader;                       ///< Linear BT.2020 -> PQ (or HLG) fullscreen pass.
    gl::ShaderProgram display_flip_shader;                     ///< Display shader with optional Y-flip for windowed output.
    cv::Mat hdr_frame_mat;                                     ///< Scratch CV_16UC4 RGBA frame for HDR decode.
    gl::ShaderProgram crossfadeShader;                         ///< Shader that mixes prev_samp and samp via fade_alpha.
    float crossfadeAlpha = 1.0f;                               ///< Current blend factor (0 = old frame, 1 = new frame).
    bool crossfadeActive = false;                              ///< True while a crossfade transition is in progress.
    float crossfadeDuration = 0.5f;                            ///< Duration of the crossfade transition in seconds.
    std::chrono::steady_clock::time_point crossfadeStartTime;  ///< Wall-clock time the current crossfade began.
    std::thread writerThread;
    std::atomic<bool> running{false};
    std::atomic<bool> captureRunning{false};
    std::atomic<bool> writerRunning{false};
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondVar;
    std::thread captureThread;
    std::queue<cv::Mat> captureQueue;
    std::mutex captureQueueMutex;
    std::condition_variable captureQueueCondVar;
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point captureStartTime;
    FrameCache frame_cache;
    bool texture_cache = false;
    GLuint cache_textures[8] = {0};
    int cache_delay = 1;
    std::atomic<bool> finished{false};
    std::atomic<bool> copy_audio{false};
    std::atomic<bool> skip_audio_mux_on_exit{false};
    std::atomic<bool> isMuxing{false};
    std::atomic<bool> muxComplete{false};
    std::thread muxThread;
    float cameraYaw = 270.0f;
    float cameraPitch = 0.0f;
    float cameraRotationSpeed = 5.0f;
    bool viewRotationActive = false;
    bool oscillateScale = false;
    bool waveActive = false;
    float cameraDistance = 0.0f;
    float modelSize = 1.0f;
    float modelRenderScale = 1.0f;
    glm::vec3 modelCenterOffset = glm::vec3(0.0f);
    float modelRotX = 0.0f;
    float modelRotY = 0.0f;
    float modelRotZ = 0.0f;
    std::atomic<uint64_t> snapshotOffset{0};
    [[maybe_unused]] int gpu_cuda_device = 0;
    bool silent_mode = false;
    bool no_drop_mode = false;
#ifdef __APPLE__
    bool use_shader_cache_flag = false;
#else
    bool use_shader_cache_flag = true;
#endif
    bool use_yuv = false;
    bool flip_output = false;
    int last_progress_percent = -1;
    bool enableWatermark = false;
    bool display_filter = false;
    int waterFontSize = 12;
    std::string watermark_text = "LostSideDead.biz"; ///< Active watermark text (overridden by --use-watermark).
    int watermark_r = 255;                            ///< Watermark color red.
    int watermark_g = 0;                              ///< Watermark color green.
    int watermark_b = 150;                            ///< Watermark color blue.

  private:
    std::atomic<uint64_t> frames_dropped{0};
    int win_w = 0;
    int win_h = 0;

    /**
     * @brief Read any remaining pixels from both PBOs and enqueue them.
     *
     * Called during destruction to ensure the last two frames that
     * were in-flight via double-buffered PBO readback are not lost.
     * Each PBO is mapped, its contents copied and vertically flipped,
     * then pushed into the writer queue.
     *
     * @param win GL window (provides width/height).
     */
    void flushPBOs(gl::GLWindow *win) {
        if (!pboIds[0])
            return;
        for (int i = 0; i < 2; i++) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
            GLubyte *src = static_cast<GLubyte *>(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));

            if (src) {
                std::vector<unsigned char> pixels(win->w * win->h * 4);
                std::memcpy(pixels.data(), src, pixels.size());
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

                std::vector<unsigned char> flipped_pixels(win->w * win->h * 4);
                for (int y = 0; y < win->h; ++y) {
                    int src_row_start = y * win->w * 4;
                    int dest_row_start = (win->h - 1 - y) * win->w * 4;
                    std::copy(pixels.begin() + src_row_start,
                              pixels.begin() + src_row_start + (win->w * 4),
                              flipped_pixels.begin() + dest_row_start);
                }

                FrameData fd;
                fd.pixels = std::move(flipped_pixels);
                fd.width = win->w;
                fd.height = win->h;
                fd.isSnapshot = false;

                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    frameQueue.push(std::move(fd));
                }
                queueCondVar.notify_one();
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }
    }

    /**
     * @brief Create the off-screen FBO used for capture/recording.
     *
     * Allocates a GL_RGBA colour texture and a depth renderbuffer,
     * both sized to the requested dimensions, and attaches them to
     * a new framebuffer.  The capture FBO is the target of all
     * shader rendering; its colour attachment (fboTexture) is then
     * blitted to the default framebuffer and read back via PBOs.
     *
     * @param width  FBO width in pixels.
     * @param height FBO height in pixels.
     * @throws mx::Exception if the FBO completeness check fails.
     */
    void setupCaptureFBO(int width, int height) {
        win_w = width;
        win_h = height;
        glGenFramebuffers(1, &captureFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);

        glGenTextures(1, &fboTexture);
        glBindTexture(GL_TEXTURE_2D, fboTexture);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA,
                     width,
                     height,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               fboTexture,
                               0);

        glGenRenderbuffers(1, &depthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw mx::Exception("FBO is not complete.");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        if (preOverlayTexture == 0) {
            glGenTextures(1, &preOverlayTexture);
        }
        glBindTexture(GL_TEXTURE_2D, preOverlayTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        if (preOverlayFBO == 0) {
            glGenFramebuffers(1, &preOverlayFBO);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, preOverlayFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, preOverlayTexture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw mx::Exception("preOverlayFBO is not complete.");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    /**
     * @brief Create a new GL texture from a cv::Mat (BGR → RGBA upload).
     *
     * Generates a texture, converts the frame from BGR to RGBA, and
     * uploads via glTexImage2D.  Used to initialise the 8 cache
     * textures with blank frames at startup.
     *
     * @param frame Input cv::Mat (BGR, 8-bit).
     * @return Newly created GL texture ID.
     * @throws mx::Exception on any GL error.
     */
    GLuint loadTexture(cv::Mat &frame) {
        GLuint texture = 0;
        glGenTextures(1, &texture);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw mx::Exception("OpenGL error: glGenTextures() returned " + std::to_string(error));
        }

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        cv::Mat temp;
        cv::cvtColor(frame, temp, cv::COLOR_BGR2RGBA);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, temp.cols, temp.rows,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, temp.ptr());

        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw mx::Exception("OpenGL error: glTexImage2D() returned " + std::to_string(error));
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        return texture;
    }

    /**
     * @brief Update an existing GL texture with new cv::Mat data (BGR → RGBA).
     *
     * Uses glTexSubImage2D when dimensions match (fast path) or
     * falls back to glTexImage2D when the frame size has changed
     * (reallocates the texture storage).
     *
     * @param texture GL texture ID to update.
     * @param frame   New frame (BGR, 8-bit).
     */
    void updateTexture(GLuint texture, cv::Mat &frame) {
        glBindTexture(GL_TEXTURE_2D, texture);
        cv::Mat temp;
        cv::cvtColor(frame, temp, cv::COLOR_BGR2RGBA);
        GLint texWidth = 0, texHeight = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texWidth);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texHeight);
        if (texWidth != temp.cols || texHeight != temp.rows) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, temp.cols, temp.rows,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, temp.ptr());
        } else {
            glTexSubImage2D(GL_TEXTURE_2D,
                            0, 0, 0,
                            temp.cols, temp.rows,
                            GL_RGBA,
                            GL_UNSIGNED_BYTE,
                            temp.ptr());
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * @brief Update a GL texture with pre-converted RGBA data (no colour conversion).
     *
     * Faster than updateTexture() when the frame is already RGBA.
     * Does not handle size changes—caller must ensure dimensions match.
     *
     * @param texture GL texture ID.
     * @param frame   cv::Mat in RGBA format.
     */
    void updateTextureRGBA(GLuint texture, cv::Mat &frame) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D,
                        0, 0, 0,
                        frame.cols, frame.rows,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        frame.ptr());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * @brief Start the background camera capture thread (camera mode only).
     *
     * Spawns a `std::thread` that continuously calls `cap.read()` and
     * pushes the flipped frame into `captureQueue`.  The queue is
     * capped at 4 entries; when full the oldest frame is dropped
     * so the capture thread never blocks.  The main thread pops
     * from this queue inside draw() under `captureQueueMutex`.
     *
     * The thread exits when `captureRunning` is set to false or
     * a read failure occurs.
     *
     * @see stopCaptureThread()
     */
    void startCaptureThread() {
        if (captureThread.joinable()) {
            return;
        }
        captureRunning = true;
        captureThread = std::thread([this]() {
            try {

                while (captureRunning) {
                    cv::Mat localFrame;
                    if (!cap.read(localFrame)) {
                        mx::system_err << "acmx2: camera read failed.\n";
                        captureRunning = false;
                        running = false;
                        break;
                    }
                    if (localFrame.empty()) {
                        continue;
                    }
                    cv::flip(localFrame, localFrame, 0);
                    {
                        std::lock_guard<std::mutex> lock(captureQueueMutex);
                        if (captureQueue.size() >= 4) {
                            captureQueue.pop();
                        }
                        captureQueue.push(std::move(localFrame));
                    }
                    captureQueueCondVar.notify_one();
                }
            } catch (const std::exception &e) {
                mx::system_err << "acmx2: Capture thread exception: " << e.what() << "\n";
                captureRunning = false;
                running = false;
            }
        });
    }

    /**
     * @brief Signal the capture thread to stop and wait for it to join.
     *
     * Sets `captureRunning = false`, wakes the condition variable
     * (in case the thread is waiting), and joins the thread.
     */
    void stopCaptureThread() {
        captureRunning = false;
        captureQueueCondVar.notify_all();
        if (captureThread.joinable()) {
            captureThread.join();
        }
    }

    /**
     * @brief Start the background writer thread (video recording + snapshots).
     *
     * Spawns a `std::thread` that dequeues FrameData from `frameQueue`
     * and either:
     * - Dispatches snapshot frames to the SnapshotThreadPool for async
     *   PNG writing.
     * - Passes video frames to `writer.write()` (file/image mode) or
     *   `writer.write_ts()` (camera mode) for H.264 encoding via FFmpeg.
     *
     * The first 1 frame (file mode) or 30 frames (camera mode) are
     * discarded as warmup to avoid capturing startup artefacts.
     *
     * The thread blocks on `queueCondVar` when the queue is empty.
     * It exits when `writerRunning` is set to false and the queue is
     * drained.
     *
     * Audio recording (if enabled) is started on the first frame
     * after warmup, keeping audio and video synchronised.
     *
     * @see stopWriterThread()
     */
    void startWriterThread() {
        if (writerThread.joinable())
            return;
        writerRunning = true;
        written_frame_counter = 0;
        writerThread = std::thread([this]() {
            try {
                captureStartTime = std::chrono::steady_clock::now();

                // Drain queued frames before exiting. shutdown paths set
                // writerRunning=false first, then notify; if we only loop on
                // writerRunning we can leave tail frames unwritten.
                while (true) {
                    FrameData fd;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        queueCondVar.wait(lock, [this]() {
                            return !frameQueue.empty() || !writerRunning;
                        });

                        if (!writerRunning && frameQueue.empty()) {
                            break;
                        }

                        fd = std::move(frameQueue.front());
                        frameQueue.pop();

                        queueCondVar.notify_all();
                    }

                    if (fd.isSnapshot) {
                        uint64_t current_offset = snapshotOffset.fetch_add(1);
                        std::string snap_prefix = prefix_path;
                        snapshot_pool.enqueue([snap_prefix, fd, current_offset] {
                            auto now1 = std::chrono::system_clock::now();
                            std::time_t now_c = std::chrono::system_clock::to_time_t(now1);
                            std::tm localTime{};
#ifdef _WIN32
                            localtime_s(&localTime, &now_c);
#else
                            localtime_r(&now_c, &localTime);
#endif

                            std::ostringstream oss;
                            oss << std::put_time(&localTime, "%Y.%m.%d-%H.%M.%S");
                            std::string snapshot_type = fd.isHdr ? "ACMX2.HDR.Snapshot" : "ACMX2.Snapshot";
#ifdef ACMX2_WITH_WEBP
                            const bool write_webp = fd.isWebPSnapshot || fd.isHdr;
                            const char *snap_ext = write_webp ? ".webp" : ".png";
#else
                            const char *snap_ext = ".png";
#endif
                            std::string name = snap_prefix + "/" + snapshot_type + "-" + oss.str() + "-" + std::to_string(fd.width) + "x" + std::to_string(fd.height) + "-" + std::to_string(current_offset) + snap_ext;

                            if (fd.isHdr) {
#ifdef ACMX2_WITH_WEBP
                                if (!saveHdrWebPFromRgba16(name.c_str(), fd.pixels.data(), fd.width, fd.height, fd.hdrTrc)) {
                                    mx::system_err << "acmx2: ERROR: failed to write HDR WebP snapshot: " << name << "\n";
                                }
#else
                                png::SavePNG_RGBA16(name.c_str(), fd.pixels.data(), fd.width, fd.height);
#endif
                            } else {
                                if (fd.isWebPSnapshot) {
#ifdef ACMX2_WITH_WEBP
                                    if (!saveSdrWebPFromRgba8(name.c_str(), fd.pixels.data(), fd.width, fd.height)) {
                                        mx::system_err << "acmx2: ERROR: failed to write SDR WebP snapshot: " << name << "\n";
                                    }
#else
                                    png::SavePNG_RGBA(name.c_str(),
                                                      const_cast<unsigned char *>(fd.pixels.data()),
                                                      fd.width, fd.height);
#endif
                                } else {
                                    png::SavePNG_RGBA(name.c_str(),
                                                      const_cast<unsigned char *>(fd.pixels.data()),
                                                      fd.width, fd.height);
                                }
                            }

                            mx::system_out << "acmx2: Took snapshot: " << name << "\n";
                            fflush(stdout);
                        });
                    }
                    if (fd.isRawSnapshot) {
                        uint64_t current_offset = snapshotOffset.fetch_add(1);
                        std::string snap_prefix = prefix_path;
                        snapshot_pool.enqueue([snap_prefix, fd, current_offset] {
                            auto now1 = std::chrono::system_clock::now();
                            std::time_t now_c = std::chrono::system_clock::to_time_t(now1);
                            std::tm localTime{};
#ifdef _WIN32
                            localtime_s(&localTime, &now_c);
#else
                            localtime_r(&now_c, &localTime);
#endif
                            std::ostringstream oss;
                            oss << std::put_time(&localTime, "%Y.%m.%d-%H.%M.%S");
                            std::string snapshot_type = fd.isHdr ? "ACMX2.HDR.Snapshot" : "ACMX2.Raw";
                            std::string name = snap_prefix + "/" + snapshot_type + "-" + oss.str() + "-" + std::to_string(fd.width) + "x" + std::to_string(fd.height) + "-" + std::to_string(current_offset) + ".raw";

                            size_t bpp = fd.isHdr ? 8 : 4;
                            png::SaveRawBytes(name.c_str(), fd.pixels.data(),
                                              static_cast<size_t>(fd.width),
                                              static_cast<size_t>(fd.height), bpp);

                            mx::system_out << "acmx2: Saved raw frame: " << name << "\n";
                            fflush(stdout);
                        });
                    }
#ifdef ACMX2_WITH_TIFF
                    if (fd.isTiffSnapshot) {
                        uint64_t current_offset = snapshotOffset.fetch_add(1);
                        std::string snap_prefix = prefix_path;
                        snapshot_pool.enqueue([snap_prefix, fd, current_offset] {
                            auto now1 = std::chrono::system_clock::now();
                            std::time_t now_c = std::chrono::system_clock::to_time_t(now1);
                            std::tm localTime{};
#ifdef _WIN32
                            localtime_s(&localTime, &now_c);
#else
                            localtime_r(&now_c, &localTime);
#endif
                            std::ostringstream oss;
                            oss << std::put_time(&localTime, "%Y.%m.%d-%H.%M.%S");
                            std::string snapshot_type = fd.isHdr ? "ACMX2.HDR.Snapshot" : "ACMX2.Snapshot";
                            std::string name = snap_prefix + "/" + snapshot_type + "-" + oss.str() + "-" +
                                               std::to_string(fd.width) + "x" + std::to_string(fd.height) + "-" +
                                               std::to_string(current_offset) + ".tiff";
                            bool ok = false;
                            if (fd.isHdr) {
                                ok = saveHdrTiffFromRgba16(name.c_str(), fd.pixels.data(), fd.width, fd.height, fd.hdrTrc);
                            } else {
                                ok = saveSdrTiffFromRgba8(name.c_str(), fd.pixels.data(), fd.width, fd.height);
                            }
                            if (!ok) {
                                mx::system_err << "acmx2: ERROR: failed to write TIFF snapshot: " << name << "\n";
                            } else {
                                mx::system_out << "acmx2: Took snapshot: " << name << "\n";
                            }
                            fflush(stdout);
                        });
                    }
#endif
                    if (writer.is_open() && (!filename.empty() || !graphic.empty()) && written_frame_counter == 0) {
                        written_frame_counter++;
                        continue;
                    } else if (writer.is_open() && written_frame_counter <= 30 && filename.empty() && graphic.empty()) {
                        written_frame_counter++;
                        continue;
                    }
#ifdef AUDIO_ENABLED
                    startAudioRecordingIfNeeded();
#endif

                    if (writer.is_open() && !fd.isSnapshot && !fd.isTiffSnapshot) {
                        if (fd.isHdr) {
                            writer.write_hdr_rgba16(fd.pixels.data());
                        } else if (!filename.empty() || !graphic.empty())
                            writer.write(fd.pixels.data());
                        else
                            writer.write_ts(fd.pixels.data());
                    }
                }
            } catch (const std::exception &e) {
                mx::system_err << "acmx2: writer thread exception: " << e.what() << "\n";
            }
            writerRunning = false;
        });
    }

    /**
     * @brief Start WAV audio capture if recording is configured and not already active.
     *
     * Used from both writer startup and writer loop paths to ensure
     * `--record-audio` is activated reliably for file, image, and
     * camera recording modes.
     */
    void startAudioRecordingIfNeeded() {
#ifdef AUDIO_ENABLED
        if (audio_is_enabled && !file_audio_mode && !audio_record_file.empty() && !is_audio_recording()) {
            if (!start_audio_recording(audio_record_file)) {
                mx::system_err << "acmx2: Error could not start audio recording to: " << audio_record_file << "\n";
            }
        }
#endif
    }

    /**
     * @brief Determine if recorded-audio mux should run at shutdown.
     *
     * Returns true only when recorded audio mode is enabled, an output
     * file is set, and either recording is still active or the recorded
     * WAV file already exists.
     */
    bool needsMux() {
#ifdef AUDIO_ENABLED
        return audio_is_enabled && !file_audio_mode && !audio_record_file.empty() && !ofilename.empty() &&
               (is_audio_recording() || std::filesystem::exists(audio_record_file));
#else
        return false;
#endif
    }

    /**
     * @brief Determine if source-track audio copy should run at shutdown.
     *
     * This applies to non-repeating video-file input with `--copy-audio`.
     */
    bool needsTransferAudio() {
        return !filename.empty() && !repeat && copy_audio && writer.is_open();
    }

    bool needsFileAudioMux() {
#ifdef AUDIO_ENABLED
        /// @brief Check whether the output video needs file-audio muxing.
        /// @return @c true when file_audio_mode is active, recording was explicitly
        /// requested (used as mux opt-in), and an output file is set.
        return file_audio_mode && !audio_file_path.empty() && !audio_record_file.empty() && !ofilename.empty();
#else
        return false;
#endif
    }

    /**
     * @brief Run ffmpeg synchronously to mux the recorded audio WAV into the video MP4.
     *
     * Builds an ffmpeg command line that copies the video stream,
     * encodes the audio as AAC 192 kbps, and trims to the video
     * duration.  The output is written to a temporary file which
     * replaces the original on success.
     *
     * Called either from the mux thread (beginMuxing) or directly
     * from the destructor when no mux thread was launched.
     */
    void runMuxSync() {
#ifdef AUDIO_ENABLED
        if (!audio_is_enabled || audio_record_file.empty() || ofilename.empty())
            return;
        if (!is_audio_recording() && !std::filesystem::exists(audio_record_file)) {
            mx::system_out << "acmx2: recorded audio file not found, skipping recorded-audio mux: " << audio_record_file << "\n";
            fflush(stdout);
            return;
        }
        if (is_audio_recording()) {
            stop_audio_recording();
        }
        std::string out_ext = std::filesystem::path(ofilename).extension().string();
        if (out_ext.empty()) out_ext = ".mp4";
        std::string tmp_out = ofilename + ".tmp" + out_ext;
        bool is_mp4_like = (out_ext == ".mp4" || out_ext == ".MP4" || out_ext == ".mov" || out_ext == ".MOV" || out_ext == ".m4v" || out_ext == ".M4V");
        int64_t fc = writer.get_frame_count();
        double video_duration = (fps > 0.0 && fc > 0) ? static_cast<double>(fc) / fps : 0.0;
        std::ostringstream cmd;
        cmd << "ffmpeg -y -i \"" << ofilename << "\" -i \"" << audio_record_file
            << "\" -map 0:v:0 -map 1:a:0"
            << " -c:v copy -c:a aac -b:a 192k";
        if (video_duration > 0.0) {
            cmd << " -t " << std::fixed << std::setprecision(3) << video_duration;
        }
        if (is_mp4_like) {
            cmd << " -movflags +faststart";
        }
        cmd << " \"" << tmp_out << "\" 2>&1";
        mx::system_out << "acmx2: muxing recorded audio into video...\n";
        fflush(stdout);
        int ret = std::system(cmd.str().c_str());
        if (ret == 0) {
            std::remove(ofilename.c_str());
            std::rename(tmp_out.c_str(), ofilename.c_str());
            mx::system_out << "acmx2: muxed recorded audio from: " << audio_record_file << " to " << ofilename << "\n";
            if (std::remove(audio_record_file.c_str()) == 0) {
                mx::system_out << "acmx2: removed temporary recorded audio: " << audio_record_file << "\n";
            } else {
                mx::system_err << "acmx2: warning could not remove recorded audio file: " << audio_record_file << "\n";
            }
        } else {
            mx::system_err << "acmx2: ffmpeg mux failed (exit code " << ret << ")\n";
            std::remove(tmp_out.c_str());
        }
        fflush(stdout);
        fflush(stderr);
#endif
    }

    /**
     * @brief Run ffmpeg synchronously to mux the audio file into the output video.
     *
     * Copies the video stream from the output file and encodes the audio
     * file track as AAC 192 kbps.  If the video is shorter than the audio
     * the audio track is truncated to match the video duration.
     * The result is written to a temporary file which replaces the
     * original on success.
     */
    void runFileAudioMuxSync() {
#ifdef AUDIO_ENABLED
        if (!file_audio_mode || audio_file_path.empty() || ofilename.empty())
            return;
        std::string out_ext = std::filesystem::path(ofilename).extension().string();
        if (out_ext.empty()) out_ext = ".mp4";
        std::string tmp_out = ofilename + ".tmp" + out_ext;
        bool is_mp4_like = (out_ext == ".mp4" || out_ext == ".MP4" || out_ext == ".mov" || out_ext == ".MOV" || out_ext == ".m4v" || out_ext == ".M4V");
        int64_t fc = writer.get_frame_count();
        double video_duration = (fps > 0.0 && fc > 0) ? static_cast<double>(fc) / fps : 0.0;
        std::ostringstream cmd;
        cmd << "ffmpeg -y -i \"" << ofilename << "\" -i \"" << audio_file_path
            << "\" -map 0:v:0 -map 1:a:0"
            << " -c:v copy -c:a aac -b:a 192k";
        if (video_duration > 0.0) {
            cmd << " -t " << std::fixed << std::setprecision(3) << video_duration;
        }
        if (is_mp4_like) {
            cmd << " -movflags +faststart";
        }
        cmd << " \"" << tmp_out << "\" 2>&1";
        mx::system_out << "acmx2: muxing audio file into video...\n";
        fflush(stdout);
        int ret = std::system(cmd.str().c_str());
        if (ret == 0) {
            std::remove(ofilename.c_str());
            std::rename(tmp_out.c_str(), ofilename.c_str());
            mx::system_out << "acmx2: muxed audio from: " << audio_file_path << " to " << ofilename << "\n";
        } else {
            mx::system_err << "acmx2: ffmpeg audio file mux failed (exit code " << ret << ")\n";
            std::remove(tmp_out.c_str());
        }
        fflush(stdout);
        fflush(stderr);
#endif
    }

    /**
     * @brief Launch the asynchronous audio-mux thread and show a progress overlay.
     *
     * Signals both capture and writer threads to stop, then spawns
     * a new `muxThread` that:
     * 1. Joins the capture and writer threads.
     * 2. Closes the Writer (flushes the MP4 trailer).
     * 3. Calls runMuxSync() to invoke ffmpeg.
     * 4. Sets `muxComplete = true`.
     *
     * While `isMuxing` is true, draw() renders a "Muxing audio…"
     * overlay and skips normal frame processing.  When muxComplete
     * becomes true, draw() joins the mux thread and quits the window.
     *
     * @param win GL window for the overlay display.
     */
    void beginMuxing(gl::GLWindow *win) {
        captureRunning = false;
        writerRunning = false;
        queueCondVar.notify_all();
        captureQueueCondVar.notify_all();
        isMuxing = true;
        muxComplete = false;
        muxThread = std::thread([this]() {
            const bool shouldTransferAudio = !filename.empty() && !repeat && copy_audio;
#ifdef AUDIO_ENABLED
            const bool shouldRecordedMux = audio_is_enabled && !file_audio_mode && !audio_record_file.empty() &&
                                           (is_audio_recording() || std::filesystem::exists(audio_record_file));
            const bool shouldFileAudioMux = file_audio_mode && !audio_file_path.empty() && !audio_record_file.empty() && !ofilename.empty();
#else
            const bool shouldRecordedMux = false;
            const bool shouldFileAudioMux = false;
#endif

            if (captureThread.joinable())
                captureThread.join();
            if (writerThread.joinable())
                writerThread.join();
            if (writer.is_open()) {
                writer.close();
                int64_t fc = writer.get_frame_count();
                double ts = (fps > 0.0) ? static_cast<double>(fc) / fps : 0.0;
                mx::system_out << "acmx2: wrote " << fc << " frames ("
                               << static_cast<int>(ts / 3600) << ":"
                               << static_cast<int>(ts / 60) % 60 << ":"
                               << static_cast<int>(ts) % 60 << ") to file: " << ofilename << "\n";
                fflush(stdout);
            }
            if (shouldTransferAudio) {
                transfer_audio(filename, ofilename);
                mx::system_out << "acmx2: copied audio track from: " << filename << " to " << ofilename << "\n";
            }
            if (shouldRecordedMux)
                runMuxSync();
            if (shouldFileAudioMux)
                runFileAudioMuxSync();
            muxComplete = true;
        });
    }

    /**
     * @brief Stop the writer thread, close the Writer, and handle audio transfer.
     *
     * Sets `writerRunning = false`, wakes the condition variable,
     * joins the writer thread, then closes the Writer (which flushes
     * the MP4 container trailer).  Logs the total duration and frame
     * count.  If `copy_audio` is set and a video file was the input,
     * the audio track is copied from the input to the output via
     * `transfer_audio()`.  Any active audio recording is stopped.
     */
    void stopWriterThread() {
        bool recording = writer.is_open();
        writerRunning = false;
        queueCondVar.notify_all();
        if (writerThread.joinable()) {
            writerThread.join();
        }
        if (recording) {
            writer.close();
            int64_t final_frame_count = writer.get_frame_count();
            double total_secs = (fps > 0.0) ? static_cast<double>(final_frame_count) / fps : 0.0;
            uint64_t hours = 0, minutes = 0, seconds = 0;
            hours = static_cast<uint64_t>(total_secs / 3600);
            minutes = static_cast<uint64_t>(total_secs / 60) % 60;
            seconds = static_cast<uint64_t>(total_secs) % 60;
            std::ostringstream timerStr;
            timerStr << std::setfill('0') << std::setw(2) << hours << ":"
                     << std::setfill('0') << std::setw(2) << minutes << ":"
                     << std::setfill('0') << std::setw(2) << seconds;

            mx::system_out << "acmx2: " << " wrote " << timerStr.str() << " (" << final_frame_count << " frames) to file: " << ofilename << "\n";
            if (!skip_audio_mux_on_exit.load() && !filename.empty() && repeat == false && copy_audio) {
                transfer_audio(filename, ofilename);
                mx::system_out << "acmx2: copied audio track from: " << filename << " to " << ofilename << "\n";
            }
#ifdef AUDIO_ENABLED
            if (audio_is_enabled && is_audio_recording()) {
                stop_audio_recording();
            }
#endif
            fflush(stdout);
            fflush(stderr);
        }
    }
};

/**
 * @class MainWindow
 * @brief Top-level SDL2/OpenGL window that hosts the ACView rendering object.
 *
 * Supports both visible and headless (silent) modes. In silent mode an
 * off-screen GL context is used for batch video processing without a window.
 */
class MainWindow : public gl::GLWindow {
    bool silent_mode = false;

    static int eventFilter(void *userdata, SDL_Event *event) {
        if (event->type == SDL_QUIT ||
            (event->type == SDL_KEYDOWN && event->key.keysym.sym == SDLK_ESCAPE)) {
            auto *win = static_cast<MainWindow *>(userdata);
            auto *view = static_cast<ACView *>(win->object.get());
            if (view && view->needsAsyncShutdown()) {
                view->requestStop();
                return 0;
            }
        }
        return 1;
    }

    void initCommon(const MXArguments &args) {
        util.path = args.path;
        if (!std::filesystem::exists(util.path + "/data/win-icon.png")) {
            if (std::filesystem::exists("/usr/local/share/acmx2/data"))
                util.path = "/usr/local/share/acmx2";
        }

        if (!silent_mode) {
            SDL_Surface *ico = png::LoadPNG(util.getFilePath("data/win-icon.png").c_str());
            if (!ico) {
                throw mx::Exception("Could not load icon: " + util.getFilePath("data/win-icon.png"));
            }
            setWindowIcon(ico);
            SDL_FreeSurface(ico);
        }

        setObject(new ACView(args));
        object->load(this);
        SDL_SetEventFilter(eventFilter, this);
        fflush(stdout);
        fflush(stderr);
    }

  public:
    /**
     * @brief Construct a visible (windowed) MainWindow.
     *
     * Creates an SDL2/OpenGL window of the requested size, then
     * calls initCommon() to set up the rendering pipeline.
     *
     * @param args Parsed CLI arguments (resolution, asset path, etc.).
     */
    MainWindow(const MXArguments &args) : gl::GLWindow("ACMX2", args.tw, args.th, false), silent_mode(args.silent) {
        initCommon(args);
    }

    /**
     * @brief Construct a headless (off-screen) MainWindow for silent batch processing.
     *
     * Uses gl::GLMode::DESKTOP to create an OpenGL context without a
     * visible window.  Intended for `--silent` mode where video is
     * processed and recorded without any display.
     *
     * @param args     Parsed CLI arguments.
     * @param headless Unused disambiguator parameter.
     */
    MainWindow(const MXArguments &args, bool headless) : gl::GLWindow(args.tw, args.th, gl::GLMode::DESKTOP), silent_mode(true) {
        initCommon(args);
    }

    ~MainWindow() override {
        SDL_SetEventFilter(nullptr, nullptr);
    }

    /**
     * @brief Per-frame callback: clear, draw ACView, swap buffers.
     *
     * Called by the GLWindow event loop.  Clears the default framebuffer,
     * asks the ACView object to render one frame, then swaps and delays
     * to maintain the target frame rate.
     */
    void draw() override {
        glClearColor(0.f, 0.f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, w, h);
#if defined(__linux__)
        if (silent_mode && g_shutdown_requested.load(std::memory_order_relaxed)) {
            auto *view = static_cast<ACView *>(object.get());
            if (view) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) {
                    mx::system_out << "\nacmx2: Ctrl+C received - finishing current frame and "
                                      "flushing output file...\n";
                    mx::system_out.flush();
                    view->requestStopNoMux();
                }
            }
            this->quit();
        }
#endif
        object->draw(this);
        swap();
        // In headless/silent batch mode there is no user watching: skip the
        // frame-pacing delay() so processing runs at full GPU/encoder speed
        // instead of being throttled to the target playback FPS.
        if (!silent_mode) {
            delay();
        }
    }

    /// @brief Placeholder—SDL events are forwarded to ACView::event() by libmx2.
    void event(SDL_Event &e) override {
    }
};

/// @brief Verify CUDA device availability and print GPU info.
void checkDevices(bool list_only = false) {
#ifdef ACMX2_WITH_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "OpenCV Cuda Support not found." << std::endl;
        std::cerr << "Reason: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "Check: Are NVIDIA drivers installed? Is the GPU seated?" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "🚀 GPU Acceleration Active: " << device_count << " device(s) found." << std::endl;
        if (list_only) {
            for (int i = 0; i < device_count; ++i) {
                cudaSetDevice(i);
                cv::cuda::printShortCudaDeviceInfo(i);
            }
        } else {
            cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
        }
    }
#else
    (void)list_only;
    std::cerr << "acmx2: CUDA support was disabled at build time (ACMX2_WITH_CUDA not defined).\n";
#endif
}

namespace {
    struct CliColors {
        bool enabled = false;
        std::string_view reset = "";
        std::string_view title = "";
        std::string_view section = "";
        std::string_view flag = "";
        std::string_view arg = "";
        std::string_view desc = "";
        std::string_view example = "";
    };

    struct HelpEntry {
        std::string_view flags;
        std::string_view description;
        std::string_view example;
    };

    bool terminalSupportsColor() {
        const char *no_color = std::getenv("NO_COLOR");
        if (no_color && no_color[0] != '\0') {
            return false;
        }

        const char *force = std::getenv("CLICOLOR_FORCE");
        if (force && force[0] != '0' && force[0] != '\0') {
            return true;
        }

#if defined(__linux__) || defined(__APPLE__)
        if (!isatty(fileno(stdout))) {
            return false;
        }
#else
        return false;
#endif

        const char *term = std::getenv("TERM");
        if (!term || std::strcmp(term, "dumb") == 0) {
            return false;
        }
        return true;
    }

    CliColors makeCliColors() {
        CliColors c;
        c.enabled = terminalSupportsColor();
        if (c.enabled) {
            c.reset = "\033[0m";
            c.title = "\033[1;96m";
            c.section = "\033[1;93m";
            c.flag = "\033[1;92m";
            c.arg = "\033[36m";
            c.desc = "\033[0;97m";
            c.example = "\033[95m";
        }
        return c;
    }

    template <typename Stream>
    void printSection(Stream &out, const CliColors &c, std::string_view name, const std::vector<HelpEntry> &entries) {
        out << c.section << "\n" << name << c.reset << "\n";
        for (const auto &entry : entries) {
            out << "  " << c.flag << entry.flags << c.reset << "\n";
            out << "    " << c.desc << entry.description << c.reset << "\n";
            if (!entry.example.empty()) {
                out << "    " << c.example << "example: " << entry.example << c.reset << "\n";
            }
        }
    }

    template <typename Stream>
    void printDetailedArguments(Stream &out) {
        const CliColors c = makeCliColors();
        out << c.title << "\nArguments" << c.reset << "\n";
        out << c.example << "Short and long forms are equivalent; values shown in <> are required." << c.reset << "\n";

        printSection(out, c, "General", {
            {"-v, --help", "Show this help screen and keyboard controls.", "acmx2 --help"},
            {"-p <path>, --path <path>", "Set assets root directory (shaders, data files, defaults).", "acmx2 --path ./data"},
            {"-r <WxH>, --resolution <WxH>", "Set output/window resolution (for display and recording).", "acmx2 --resolution 1920x1080"},
            {"-N, --fullscreen", "Start in fullscreen mode (Escape to exit fullscreen).", "acmx2 --fullscreen"},
            {"--silent", "Run headless (no preview window). Intended for file-to-file rendering.", "acmx2 -i in.mp4 -o out.mp4 --silent"},
            {"--duration <seconds>", "Auto-stop recording/output after elapsed seconds.", "acmx2 -i in.mp4 -o out.mp4 --duration 30"}
        });

        printSection(out, c, "Input Source", {
            {"-i <file>, --input <file>", "Input video file.", "acmx2 --input clip.mp4"},
            {"-g <file>, --graphic <file>", "Input still image instead of camera/video.", "acmx2 --graphic frame.png"},
            {"-d <idx>, --device <idx>", "Camera device index to open.", "acmx2 --device 0"},
            {"-c <WxH>, --camera-res <WxH>", "Request camera capture resolution.", "acmx2 --camera-res 1280x720"},
            {"--enumerate-device <idx>", "Print camera resolutions/formats supported by device and exit.", "acmx2 --enumerate-device 0"},
            {"--use-yuv", "Prefer YUYV camera capture over MJPG for compatible devices.", "acmx2 --device 0 --use-yuv"}
        });

        printSection(out, c, "Shaders And Visual Pipeline", {
            {"-s <index.txt>, --shaders <index.txt>", "Use shader library index file (playlist-able shader set).", "acmx2 --shaders ./shaders/index.txt"},
            {"-f <frag.glsl>, --fragment <frag.glsl>", "Use a single fragment shader file directly.", "acmx2 --fragment ./shaders/wave.glsl"},
            {"-h <index>, --shader <index>", "Select initial shader index from the active library.", "acmx2 --shaders index.txt --shader 3"},
            {"--shader-pass <list>", "Run multiple shader indices per frame (comma-separated).", "acmx2 --shader-pass 0,4,7"},
            {"--playlist <file>", "Load shader playlist text file (one shader name per line).", "acmx2 --playlist live_set.txt"},
            {"--cross-fade <seconds>", "Set smooth transition time between playlist shader switches.", "acmx2 --playlist live_set.txt --cross-fade 1.25"},
            {"--autopilot-frames <N>", "Auto-switch to random playlist shader every N rendered frames.", "acmx2 --playlist live_set.txt --autopilot-frames 240"},
            {"--time-speed <mult>", "Scale shader time uniform speed (1.0 = normal).", "acmx2 --time-speed 0.5"},
            {"--build <library-path>", "Compile shader library into cache, then exit.", "acmx2 --build ./shaders"},
            {"--remove-broken <library-path>", "Compile-check each shader, remove failing entries from index.txt, then exit.", "acmx2 --remove-broken ./shaders"},
            {"--no-cache", "Disable shader binary cache and always compile at startup.", "acmx2 --no-cache"},
            {"--texture-cache", "Enable texture/frame cache for cache-aware shader effects.", "acmx2 --texture-cache"},
            {"--cache-delay <frames>", "Delay frame cache feed by N frames for temporal effects.", "acmx2 --texture-cache --cache-delay 6"},
            {"--enable-3d", "Enable 3D object rendering pipeline.", "acmx2 --enable-3d"},
            {"--model <file>", "Load a custom 3D model file for the 3D scene.", "acmx2 --enable-3d --model scene.obj"},
            {"--flip", "Flip final output vertically before display/encode.", "acmx2 --flip"}
        });

        printSection(out, c, "GPU And CUDA", {
            {"--gpu-filter <list>", "Apply CUDA filter chain by index list (comma-separated).", "acmx2 --gpu-filter 1,12,18"},
            {"--gpu-buffer <N>", "Set GPU temporal frame buffer size (4..32).", "acmx2 --gpu-buffer 12"},
            {"--list-filters", "List all built-in GPU filters and their indices.", "acmx2 --list-filters"},
            {"-m <idx>, --cuda-device <idx>", "Select CUDA device index to run processing on.", "acmx2 --cuda-device 0"},
            {"--list-cuda-devices", "List CUDA devices visible to the runtime.", "acmx2 --list-cuda-devices"},
            {"--check-cuda", "Report whether this build has CUDA support enabled.", "acmx2 --check-cuda"}
        });

        printSection(out, c, "Recording And Encoding", {
            {"-o <file>, --output <file>", "Write processed video to output file.", "acmx2 -i in.mp4 -o out.mp4"},
            {"-e <prefix>, --prefix <prefix>", "Snapshot filename prefix for captured frames.", "acmx2 --prefix snap/frame_"},
            {"-u <fps>, --fps <fps>", "Set output frame rate for recording.", "acmx2 --fps 60"},
            {"-b <crf>, --bitrate <crf>", "Legacy CRF quality option for encoder.", "acmx2 --bitrate 20"},
            {"--encode-preset <name>", "Encoder speed/quality preset (ultrafast .. veryslow).", "acmx2 --encode-preset fast"},
            {"--encode-tune <name>", "Tune encoder for content type or low latency.", "acmx2 --encode-tune film"},
            {"--encode-crf <0-51>", "Set encoder quality directly (lower = better quality/larger file).", "acmx2 --encode-crf 18"},
            {"--encode-codec <mode>", "Codec backend: auto, software, or nvenc.", "acmx2 --encode-codec nvenc"},
            {"--encode-realtime", "Enable low-latency encoder settings for live pipelines.", "acmx2 --encode-realtime"},
            {"--no-drop", "Never drop frames; block producer when encoder queue is full.", "acmx2 --no-drop"},
            {"--display-filter", "Show current shader/stack and GPU filter in upper-left corner.", "acmx2 --display-filter"},
            {"--use-watermark <text>", "Enable watermark with given text in recorded videos (upper-left).", "acmx2 --use-watermark \"My Channel\""},
            {"--use-watermark-color <r,g,b>", "Watermark text color as 0-255 components.", "acmx2 --use-watermark-color 255,255,0"},
            {"--copy-audio", "Mux input audio track into encoded output when possible.", "acmx2 -i in.mp4 -o out.mp4 --copy-audio"},
            {"-a, --repeat", "Loop video input source continuously.", "acmx2 -i loop.mp4 --repeat"}
        });

#ifdef AUDIO_ENABLED
        printSection(out, c, "Audio Reactivity", {
            {"-w, --enable-audio", "Enable audio-reactive shader modulation.", "acmx2 --enable-audio"},
            {"-l <N>, --channels <N>", "Number of audio channels to capture/process.", "acmx2 --channels 2"},
            {"-q <value>, --sense <value>", "Set audio sensitivity multiplier for visual response.", "acmx2 --sense 1.4"},
            {"-y, --pass-through", "Pass captured input audio directly to selected output device.", "acmx2 --pass-through"},
            {"--audio-input <device>", "Select input audio device name/id.", "acmx2 --audio-input \"USB Audio\""},
            {"--audio-output <device>", "Select output audio device name/id.", "acmx2 --audio-output \"Built-in Output\""},
            {"--list-devices", "List available audio input/output devices.", "acmx2 --list-devices"},
            {"--record-audio <wav-file>", "Record captured audio stream to a WAV file.", "acmx2 --record-audio take.wav"},
            {"--record-gain <0.0-2.0>", "Set recording gain multiplier (1.0 = unity).", "acmx2 --record-gain 1.2"},
            {"--audio-file <file>", "Use an audio file as reactivity source instead of microphone input.", "acmx2 --audio-file soundtrack.mp3"},
            {"--audio-trunc", "Stop playback/output when the audio file reaches EOF.", "acmx2 --audio-file soundtrack.mp3 --audio-trunc"},
            {"--check-audio", "Report whether this build has audio support enabled.", "acmx2 --check-audio"}
        });
#endif

#ifdef MIDI_ENABLED
        printSection(out, c, "MIDI Control", {
            {"--midi-map <file>", "Load MIDI mapping configuration file.", "acmx2 --midi-map midi.midi_cfg"},
            {"--midi-device <idx>", "Select MIDI input device index.", "acmx2 --midi-device 0"},
            {"--list-midi", "List available MIDI input devices.", "acmx2 --list-midi"},
            {"--check-midi", "Report whether this build has MIDI support enabled.", "acmx2 --check-midi"}
        });
#endif

        printSection(out, c, "Runtime Overlay", {
            {"--disable-counter", "Hide timer and FPS overlay text.", "acmx2 --disable-counter"}
        });
    }

    template <typename Stream>
    void printKeyboardControls(Stream &out) {
        const CliColors c = makeCliColors();
        out << c.title << "\nKeyboard Controls" << c.reset << "\n";

        printSection(out, c, "Main", {
            {"Escape", "Quit.", ""},
            {"Ctrl+X", "Quit without audio mux.", ""},
            {"Up Arrow", "Previous shader.", ""},
            {"Down Arrow", "Next shader.", ""},
            {"Left Arrow", "Previous GPU filter (if enabled).", ""},
            {"Right Arrow", "Next GPU filter (if enabled).", ""},
            {"Space", "Enable/disable processing.", ""},
            {"L", "Toggle video freeze (Video/Image modes).", ""},
            {"P", "Toggle pause (Video/Image) or toggle shader playlist.", ""},
            {"J", "Toggle autopilot mode (requires playlist).", ""},
            {"Y", "Toggle sequential autopilot (cycles playlist in order, requires playlist).", ""},
            {"T", "Enable/disable time.", ""},
            {"U / I", "Step time when time is disabled.", ""},
            {"Page Up / Page Down", "Increase/decrease time speed.", ""},
            {"M", "Toggle multi-pass / multi-shader pass.", ""},
            {"F", "Toggle fullscreen.", ""},
            {"Q", "Toggle reactive time (if AUDIO_ENABLED).", ""},
            {"Insert", "Increase audio sensitivity.", ""},
            {"Delete", "Decrease audio sensitivity.", ""},
            {"End", "Toggle spectrum sensitivity scaling.", ""},
            {"Home", "Toggle audio delta time scaling.", ""},
            {"3", "Toggle 2D/3D mode.", ""}
        });

        printSection(out, c, "Snapshots", {
            {"Z", "Save PNG snapshot (SDR 8-bit; HDR mode still outputs SDR PNG).", ""},
            {"4", "Save TIFF snapshot (SDR: 8-bit RGBA; HDR: 16-bit RGBA; requires ACMX2_WITH_TIFF).", ""},
            {"5", "Save lossless WebP snapshot (HDR is tone-mapped; requires ACMX2_WITH_WEBP).", ""},
            {"6", "Save raw RGBA snapshot (HDR: 16-bit RGBA, otherwise 8-bit RGBA).", "ffplay -f rawvideo -pixel_format rgba64le -video_size WxH file.raw"}
        });

        printSection(out, c, "3D Mode", {
            {"W / A / S / D", "Look around.", ""},
            {"V", "Toggle view rotation.", ""},
            {"O", "Toggle oscillation.", ""},
            {"X", "Reset camera distance.", ""},
            {"+ / -", "Increase/decrease camera distance.", ""},
            {"B", "Increase movement speed.", ""},
            {"N", "Decrease movement speed.", ""},
            {"C", "Toggle object wave.", ""},
            {"E", "Enable/disable watermark.", ""},
            {"]", "Increase model scale.", ""},
            {"[", "Decrease model scale.", ""},
            {". (period)", "Increase camera rotation speed.", ""},
            {", (comma)", "Decrease camera rotation speed.", ""}
        });
        printSection(out, c, "Environment Variables", {
            {"ACMX2_PATH", "Default assets root directory (equivalent to --path). Used when --path is not specified.", "export ACMX2_PATH=/usr/local/share/acmx2"},
            {"ACMX2_SHADER_PATH", "Default shader library index file or directory (equivalent to --shaders). Used when neither --shaders nor --fragment is specified.", "export ACMX2_SHADER_PATH=/usr/local/share/acmx2/filters"}
        });
    }
}

/// @brief Print program version, author, arguments, and keyboard controls.
void printAbout() {
    mx::system_out << PROGRAM_NAME << ": " << VERSION_INFO << "\n";
    mx::system_out << "(C) 2026 " << VERSION_AUTHOR << "\n";
    mx::system_out << "https://lostsidedead.biz\n";
    printDetailedArguments(mx::system_out);
    printKeyboardControls(mx::system_out);
}

/**
 * @brief Application entry point.
 *
 * Parses command-line arguments with Argz, validates inputs, then either
 * builds a shader cache and exits or launches the main render loop.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on clean exit, EXIT_FAILURE on error.
 */
int main(int argc, char **argv) {
    fflush(stdout);
    Argz<std::string> parser(argc, argv);
    parser.addOptionSingle('v', "Display help message")
        .addOptionSingleValue('p', "assets path")
        .addOptionDoubleValue('P', "path", "assets path")
        .addOptionSingleValue('r', "Resolution WidthxHeight")
        .addOptionDoubleValue('R', "resolution", "Resolution WidthxHeight")
        .addOptionSingleValue('d', "Camera Device")
        .addOptionDoubleValue('D', "device", "Device Index")
        .addOptionSingleValue('c', "Camera Resolution")
        .addOptionDoubleValue('C', "camera-res", "Camera Resolution")
        .addOptionSingleValue('i', "Input file")
        .addOptionSingleValue('g', "Input Image")
        .addOptionDoubleValue('G', "graphic", "Input graphics file")
        .addOptionDoubleValue('I', "input", "Input file")
        .addOptionSingleValue('s', "Shader Library Index File")
        .addOptionDoubleValue('S', "shaders", "Shader Library Index File")
        .addOptionSingleValue('f', "Fragment Shader")
        .addOptionDoubleValue('F', "fragment", "Fragment Shader")
        .addOptionSingleValue('h', "Shader Index")
        .addOptionDoubleValue('H', "shader", "Shader Index")
        .addOptionSingleValue('e', "Save Prefix")
        .addOptionDoubleValue('E', "prefix", "Save Prefix")
        .addOptionSingleValue('o', "output file")
        .addOptionDoubleValue('O', "output", "output file")
        .addOptionSingleValue('b', "Bitrate in CRF")
        .addOptionDoubleValue('B', "bitrate", "Bitrate in CRF")
        .addOptionSingleValue('u', "frames per second")
        .addOptionDoubleValue('U', "fps", "Frames per second")
        .addOptionSingle('a', "Repeat")
        .addOptionDouble('A', "repeat", "Video repeat")
        .addOptionSingle('n', "fullscreen")
        .addOptionDouble(256, "texture-cache", "Enable texture cache")
        .addOptionDoubleValue(257, "cache-delay", "Cache delay in frames")
        .addOptionDouble(258, "copy-audio", "Copy audio track")
        .addOptionDouble(259, "enable-3d", "Enable 3D cube")
        .addOptionDoubleValue(260, "model", "Model file")
        .addOptionDouble(261, "help", "print help info")
        .addOptionDoubleValue(400, "gpu-filter", "GPU filter indices (comma-separated)")
        .addOptionDoubleValue(401, "gpu-buffer", "GPU frame buffer size (4-32)")
        .addOptionDouble(402, "list-filters", "List available GPU filters")
        .addOptionDouble(403, "disable-counter", "Disable timer and FPS counter overlay")
        .addOptionSingleValue('m', "CUDA device index")
        .addOptionDoubleValue('M', "cuda-device", "CUDA device index")
        .addOptionDouble(404, "list-cuda-devices", "List available CUDA devices")
        .addOptionDouble(415, "check-cuda", "Report whether CUDA support is compiled in")
#ifdef AUDIO_ENABLED
        .addOptionSingle('w', "Enable Audio Reactivity")
        .addOptionDouble('W', "enable-audio", "enabled audio reacitivty")
        .addOptionSingleValue('l', "Audio channels")
        .addOptionDoubleValue('L', "channels", "Audio channels")
        .addOptionSingleValue('q', "Audio Sensitivty")
        .addOptionDoubleValue('Q', "sense", "Audio sensitivty")
        .addOptionSingle('y', "Enable Audio Pass-through")
        .addOptionDouble('Y', "pass-through", "Enable audio pass through")
        .addOptionDoubleValue(300, "audio-input", "Audio input device")
        .addOptionDoubleValue(301, "audio-output", "Audio output device")
        .addOptionDouble(302, "list-devices", "list audio devices")
        .addOptionDoubleValue(303, "record-audio", "Record captured audio to WAV file")
        .addOptionDoubleValue(304, "record-gain", "Recording volume gain 0.0-2.0 (default: 1.0)")
        .addOptionDoubleValue(305, "audio-file", "Use audio from file (WAV/MP3/etc.) for reactivity instead of mic")
        .addOptionDouble(306, "audio-trunc", "Stop playback when the audio file reaches the end")
#endif
        .addOptionDouble('N', "fullscreen", "Fullscreen Window (Escape to quit)")
        .addOptionDouble(405, "silent", "Silent mode - process video without window, (video files only)")
        .addOptionDoubleValue(406, "shader-pass", "Shader pass indices (comma-separated, e.g. 0,1,2)")
        .addOptionDoubleValue(407, "build", "Build shader cache for specified library path (compiles shaders and exits)")
        .addOptionDouble(408, "no-cache", "Disable shader caching (always recompile shaders)")
        .addOptionDoubleValue(416, "remove-broken", "Compile each shader in library path; remove shaders that fail to compile from index.txt, then exit")
        .addOptionDoubleValue(409, "time-speed", "Constant time_f speed multiplier (default: 1.0)")
        .addOptionDoubleValue(410, "playlist", "Shader playlist text file (one shader name per line, P to toggle)")
        .addOptionDoubleValue(417, "autopilot-frames", "Autopilot frame interval; switch to a random playlist shader every N frames (J toggles)")
        .addOptionDoubleValue(411, "duration", "Recording duration in seconds (float); stop recording and exit after elapsed")
        .addOptionDoubleValue(412, "cross-fade", "Crossfade duration in seconds when switching playlist shaders (default: 0.5)")
        .addOptionDoubleValue(413, "enumerate-device", "List supported resolutions for a camera device index")
        .addOptionDouble(414, "use-yuv", "Use YUV (YUYV) camera format instead of MJPG")
        .addOptionDoubleValue(600, "encode-preset", "Encoder preset: ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow (default: medium)")
        .addOptionDoubleValue(601, "encode-tune", "Encoder tune: none,film,animation,grain,stillimage,psnr,ssim,fastdecode,zerolatency (default: none)")
        .addOptionDoubleValue(602, "encode-crf", "Encoder CRF quality 0 (best) .. 51 (worst), default 18")
        .addOptionDoubleValue(603, "encode-codec", "Encoder codec: auto,software,nvenc (default: auto)")
        .addOptionDouble(604, "encode-realtime", "Enable low-latency realtime encoding flags")
        .addOptionDouble(605, "flip", "Vertical flip output frames")
        .addOptionDouble(606, "no-drop", "Video mode: never drop frames; block when encoder queue is full")
        .addOptionDouble(607, "display-filter", "Display current shader/stack and GPU filter in upper-left corner")
        .addOptionDoubleValue(608, "use-watermark", "Enable watermark with the given text in upper-left corner of recorded video")
        .addOptionDoubleValue(609, "use-watermark-color", "Watermark color as r,g,b each 0-255 (default: 255,0,150)")
#ifdef MIDI_ENABLED
        .addOptionDoubleValue(500, "midi-map", "MIDI config file (.midi_cfg)")
        .addOptionDoubleValue(501, "midi-device", "MIDI input device index")
        .addOptionDouble(502, "list-midi", "List available MIDI input devices")
#endif
    ;

    if (argc == 1) {
        printAbout();
        exit(EXIT_SUCCESS);
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--check-cuda") {
#ifdef ACMX2_WITH_CUDA
            std::cout << "CUDA: enabled" << std::endl;
#else
            std::cout << "CUDA: disabled" << std::endl;
#endif
            return EXIT_SUCCESS;
        }
        if (std::string(argv[i]) == "--check-midi") {
#ifdef MIDI_ENABLED
            std::cout << "MIDI: enabled" << std::endl;
#else
            std::cout << "MIDI: disabled" << std::endl;
#endif
            return EXIT_SUCCESS;
        }
        if (std::string(argv[i]) == "--check-audio") {
#ifdef AUDIO_ENABLED
            std::cout << "AUDIO: enabled" << std::endl;
#else
            std::cout << "AUDIO: disabled" << std::endl;
#endif
            return EXIT_SUCCESS;
        }
    }

    mx::system_out << PROGRAM_NAME << " " << VERSION_INFO << "\n";
    mx::system_out << "(C) 2026 " << VERSION_AUTHOR << "\n";
    mx::system_out << "https://lostsidedead.biz\n";

    Argument<std::string> arg;
    MXArguments args;
    int value = 0;
    try {
        while ((value = parser.proc(arg)) != -1) {
            switch (value) {
            case 'v':
            case 261:
                printAbout();
                exit(EXIT_SUCCESS);
                break;
            case 'p':
            case 'P':
                args.path = arg.arg_value;
                break;
            case 'r':
            case 'R': {
                auto pos = arg.arg_value.find("x");
                if (pos == std::string::npos) {
                    mx::system_err << "Error invalid resolution use WidthxHeight\n";
                    mx::system_err.flush();
                    exit(EXIT_FAILURE);
                }
                std::string left, right;
                left = arg.arg_value.substr(0, pos);
                right = arg.arg_value.substr(pos + 1);
                args.tw = atoi(left.c_str());
                args.th = atoi(right.c_str());
                args.sizev = cv::Size(args.tw, args.th);
            } break;
            case 'G':
            case 'g':
                args.graphic_file = arg.arg_value;
                break;
            case 'C':
            case 'c': {
                auto pos = arg.arg_value.find("x");
                if (pos == std::string::npos) {
                    mx::system_err << "Error invalid camera resolution use WidthxHeight\n";
                    mx::system_err.flush();
                    exit(EXIT_FAILURE);
                }
                std::string left, right;
                left = arg.arg_value.substr(0, pos);
                right = arg.arg_value.substr(pos + 1);
                int xw = atoi(left.c_str());
                int xh = atoi(right.c_str());
                args.csize = cv::Size(xw, xh);
            } break;
            case 'd':
            case 'D':
                args.camera_device = atoi(arg.arg_value.c_str());
                break;
            case 's':
            case 'S':
                args.mode = 1;
                args.library = arg.arg_value;
                break;
            case 'F':
            case 'f':
                args.mode = 0;
                args.fragment = arg.arg_value;
                break;
            case 'h':
            case 'H':
                args.shader_index = atoi(arg.arg_value.c_str());
                break;
            case 'e':
            case 'E':
                args.prefix_path = arg.arg_value;
                break;
            case 'i':
            case 'I':
                args.filename = arg.arg_value;
                break;
            case 'o':
            case 'O':
                args.ofilename = arg.arg_value;
                break;
            case 'b':
            case 'B':
                args.crf = arg.arg_value;
                try {
                    int v = std::stoi(arg.arg_value);
                    if (v < 0) v = 0;
                    if (v > 51) v = 51;
                    args.encode_opts.crf = v;
                } catch (...) {}
                break;
            case 'u':
            case 'U':
                args.fps_value = atof(arg.arg_value.c_str());
                break;
            case 'a':
            case 'A':
                args.repeat = true;
                break;
            case 'n':
            case 'N':
                args.full = true;
                break;
            case 256:
                args.cache = true;
                mx::system_out << "acmx2: Texture cache enabled.\n";
                break;
            case 257:
                args.cache_delay = atoi(arg.arg_value.c_str());
                mx::system_out << "acmx2: Cache delay set to: " << args.cache_delay << "\n";
                break;
            case 258:
                args.copy_audio = true;
                break;
            case 259:
                args.is3d = true;
                mx::system_out << "acmx2: 3D cube enabled.\n";
                break;
            case 260:
                args.model_file = arg.arg_value;
                break;
            case 400: {
                args.gpu_filter_enabled = true;
                std::string list = arg.arg_value;
                size_t start = 0;
                while (true) {
                    size_t pos = list.find(',', start);
                    std::string tok = (pos == std::string::npos) ? list.substr(start) : list.substr(start, pos - start);
                    if (!tok.empty()) {
                        int idx = std::stoi(tok);
                        if (idx >= 0 && idx < ac_gpu::AC_FILTER_MAX) {
                            args.gpu_filter_indices.push_back(idx);
                        } else {
                            mx::system_err << "acmx2: Invalid GPU filter index: " << idx << " (max: " << ac_gpu::AC_FILTER_MAX - 1 << ")\n";
                        }
                    }
                    if (pos == std::string::npos)
                        break;
                    start = pos + 1;
                }
            } break;
            case 401:
                args.gpu_frame_buffer_size = std::stoi(arg.arg_value);
                if (args.gpu_frame_buffer_size < 4)
                    args.gpu_frame_buffer_size = 4;
                if (args.gpu_frame_buffer_size > 32)
                    args.gpu_frame_buffer_size = 32;
                mx::system_out << "acmx2: GPU frame buffer size: " << args.gpu_frame_buffer_size << "\n";
                break;
            case 402:
                mx::system_out << "Available GPU Filters (" << ac_gpu::AC_FILTER_MAX << " total):\n";
                for (int i = 0; i < ac_gpu::AC_FILTER_MAX; ++i) {
                    mx::system_out << "  " << i << ": " << ac_gpu::filters[i].name << "\n";
                }
                exit(EXIT_SUCCESS);
                break;
            case 403:
                args.disable_counter = true;
                break;
            case 'm':
            case 'M':
                args.cuda_device = atoi(arg.arg_value.c_str());
                break;
            case 404:
                checkDevices(true);
                exit(EXIT_SUCCESS);
                break;
            case 415:
#ifdef ACMX2_WITH_CUDA
                std::cout << "CUDA: enabled\n";
#else
                std::cout << "CUDA: disabled\n";
#endif
                exit(EXIT_SUCCESS);
                break;
#ifdef AUDIO_ENABLED
            case 'W':
            case 'w':
                args.audio_enabled = true;
                break;
            case 'l':
            case 'L':
                args.audio_channels = atoi(arg.arg_value.c_str());
                break;
            case 'Q':
            case 'q':
                args.audio_sensitivty = atof(arg.arg_value.c_str());
                break;
            case 'Y':
            case 'y':
                set_output(true);
                break;
            case 300:
                if (arg.arg_value == "default")
                    args.audio_input = -1;
                else
                    args.audio_input = atoi(arg.arg_value.c_str());
                break;
            case 301:
                if (arg.arg_value == "default")
                    args.audio_output = -1;
                else
                    args.audio_output = atoi(arg.arg_value.c_str());
                break;
            case 302:
                list_audio_devices();
                exit(EXIT_SUCCESS);
                break;
            case 303:
                args.record_audio_file = arg.arg_value;
                break;
            case 304:
                args.record_gain = static_cast<float>(atof(arg.arg_value.c_str()));
                break;
            case 305:
                args.audio_file = arg.arg_value;
                args.audio_enabled = true;
                break;
            case 306:
                args.audio_trunc = true;
                break;
#endif
            case 405:
                args.silent = true;
                break;
            case 406: {
                std::string pass_list = arg.arg_value;
                size_t start = 0;
                while (true) {
                    size_t pos = pass_list.find(',', start);
                    std::string tok = (pos == std::string::npos)
                                          ? pass_list.substr(start)
                                          : pass_list.substr(start, pos - start);
                    if (!tok.empty()) {
                        try {
                            int idx = std::stoi(tok);
                            if (idx >= 0) {
                                args.shader_pass_list.push_back(idx);
                            }
                        } catch (...) {
                            mx::system_err << "acmx2: Warning: Invalid shader pass index: " << tok << "\n";
                        }
                    }
                    if (pos == std::string::npos)
                        break;
                    start = pos + 1;
                }
                if (!args.shader_pass_list.empty()) {
                    args.shader_pass_enabled = true;
                    mx::system_out << "acmx2: Shader pass list enabled with " << args.shader_pass_list.size() << " passes\n";
                }
                break;
            }
            case 407:
                args.build_cache = true;
                args.build_library_path = arg.arg_value;
                break;
            case 416:
                args.remove_broken = true;
                args.remove_broken_path = arg.arg_value;
                break;
            case 408:
                args.use_shader_cache = false;
                mx::system_out << "acmx2: Shader caching disabled\n";
                break;
            case 409:
                args.time_speed = static_cast<float>(atof(arg.arg_value.c_str()));
                mx::system_out << "acmx2: Time speed set to: " << args.time_speed << "\n";
                break;
            case 410:
                args.playlist_file = arg.arg_value;
                mx::system_out << "acmx2: Playlist file: " << args.playlist_file << "\n";
                break;
            case 417:
                args.autopilot_frames = atoi(arg.arg_value.c_str());
                if (args.autopilot_frames < 0)
                    args.autopilot_frames = 0;
                mx::system_out << "acmx2: Autopilot frames: " << args.autopilot_frames << "\n";
                break;
            case 411:
                args.duration = atof(arg.arg_value.c_str());
                if (args.duration > 0.0) {
                    mx::system_out << "acmx2: Duration set to: " << args.duration << " seconds\n";
                }
                break;
            case 412:
                args.cross_fade_duration = static_cast<float>(atof(arg.arg_value.c_str()));
                mx::system_out << "acmx2: Crossfade duration set to: " << args.cross_fade_duration << " seconds\n";
                break;
            case 413: {
#ifdef __linux__
                int dev_idx = atoi(arg.arg_value.c_str());
                std::string dev_path = "/dev/video" + std::to_string(dev_idx);
                int fd = open(dev_path.c_str(), O_RDWR);
                if (fd < 0) {
                    mx::system_err << "acmx2: Cannot open " << dev_path << ": " << strerror(errno) << "\n";
                    exit(EXIT_FAILURE);
                }
                v4l2_capability cap{};
                if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
                    mx::system_out << "Device " << dev_idx << ": " << dev_path << "\n";
                    mx::system_out << "  Driver : " << cap.driver << "\n";
                    mx::system_out << "  Card   : " << cap.card << "\n";
                    mx::system_out << "  Bus    : " << cap.bus_info << "\n";
                }
                v4l2_fmtdesc fmt{};
                fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                fmt.index = 0;
                while (ioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0) {
                    char fourcc[5] = {
                        static_cast<char>(fmt.pixelformat & 0xFF),
                        static_cast<char>((fmt.pixelformat >> 8) & 0xFF),
                        static_cast<char>((fmt.pixelformat >> 16) & 0xFF),
                        static_cast<char>((fmt.pixelformat >> 24) & 0xFF),
                        '\0'
                    };
                    mx::system_out << "\n  Format: " << fourcc << " (" << fmt.description << ")\n";
                    v4l2_frmsizeenum fsize{};
                    fsize.pixel_format = fmt.pixelformat;
                    fsize.index = 0;
                    while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fsize) == 0) {
                        if (fsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                            mx::system_out << "    " << fsize.discrete.width << "x" << fsize.discrete.height;
                            v4l2_frmivalenum fival{};
                            fival.pixel_format = fmt.pixelformat;
                            fival.width = fsize.discrete.width;
                            fival.height = fsize.discrete.height;
                            fival.index = 0;
                            bool first = true;
                            while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fival) == 0) {
                                if (fival.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
                                    double fps_val = static_cast<double>(fival.discrete.denominator) / fival.discrete.numerator;
                                    mx::system_out << (first ? " @ " : ", ") << std::fixed << std::setprecision(1) << fps_val << " fps";
                                    first = false;
                                }
                                fival.index++;
                            }
                            mx::system_out << "\n";
                        } else if (fsize.type == V4L2_FRMSIZE_TYPE_STEPWISE || fsize.type == V4L2_FRMSIZE_TYPE_CONTINUOUS) {
                            mx::system_out << "    " << fsize.stepwise.min_width << "x" << fsize.stepwise.min_height
                                           << " to " << fsize.stepwise.max_width << "x" << fsize.stepwise.max_height
                                           << " (step " << fsize.stepwise.step_width << "x" << fsize.stepwise.step_height << ")\n";
                            break;
                        }
                        fsize.index++;
                    }
                    fmt.index++;
                }
                close(fd);
#else
                mx::system_out << "acmx2: --enumerate-device is only supported on Linux\n";
#endif
                fflush(stdout);
                exit(EXIT_SUCCESS);
                break;
            }
            case 414:
                args.use_yuv = true;
                mx::system_out << "acmx2: Using YUV (YUYV) camera format\n";
                break;
            case 600:
                args.encode_opts.preset = arg.arg_value;
                mx::system_out << "acmx2: Encoder preset: " << args.encode_opts.preset << "\n";
                break;
            case 601:
                args.encode_opts.tune = (arg.arg_value == "none") ? std::string{} : arg.arg_value;
                mx::system_out << "acmx2: Encoder tune: " << (args.encode_opts.tune.empty() ? "none" : args.encode_opts.tune) << "\n";
                break;
            case 602: {
                int v = atoi(arg.arg_value.c_str());
                if (v < 0) v = 0;
                if (v > 51) v = 51;
                args.encode_opts.crf = v;
                args.crf = std::to_string(v);
                mx::system_out << "acmx2: Encoder CRF: " << v << "\n";
                break;
            }
            case 603:
                args.encode_opts.codec = arg.arg_value;
                mx::system_out << "acmx2: Encoder codec: " << args.encode_opts.codec << "\n";
                break;
            case 604:
                args.encode_opts.realtime = true;
                mx::system_out << "acmx2: Encoder realtime mode enabled\n";
                break;
            case 605:
                args.flip_output = true;
                mx::system_out << "acmx2: Output frame flipping enabled\n";
                break;
            case 606:
                args.no_drop = true;
                mx::system_out << "acmx2: --no-drop enabled (video mode)\n";
                break;
            case 607:
                args.display_filter = true;
                mx::system_out << "acmx2: --display-filter enabled\n";
                break;
            case 608:
                args.watermark_text = arg.arg_value;
                mx::system_out << "acmx2: --use-watermark text: \"" << args.watermark_text << "\"\n";
                break;
            case 609: {
                const std::string &v = arg.arg_value;
                int r = 255, g = 0, b = 150;
                size_t c1 = v.find(',');
                size_t c2 = (c1 == std::string::npos) ? std::string::npos : v.find(',', c1 + 1);
                if (c1 != std::string::npos && c2 != std::string::npos) {
                    try {
                        r = std::stoi(v.substr(0, c1));
                        g = std::stoi(v.substr(c1 + 1, c2 - c1 - 1));
                        b = std::stoi(v.substr(c2 + 1));
                    } catch (...) {
                        mx::system_err << "acmx2: --use-watermark-color: invalid value '"
                                       << v << "'; expected r,g,b\n";
                        break;
                    }
                    args.watermark_r = std::clamp(r, 0, 255);
                    args.watermark_g = std::clamp(g, 0, 255);
                    args.watermark_b = std::clamp(b, 0, 255);
                    mx::system_out << "acmx2: --use-watermark-color: "
                                   << args.watermark_r << ","
                                   << args.watermark_g << ","
                                   << args.watermark_b << "\n";
                } else {
                    mx::system_err << "acmx2: --use-watermark-color: invalid value '"
                                   << v << "'; expected r,g,b\n";
                }
                break;
            }
#ifdef MIDI_ENABLED
            case 500:
                args.midi_map_file = arg.arg_value;
                mx::system_out << "acmx2: MIDI map file: " << args.midi_map_file << "\n";
                break;
            case 501:
                args.midi_device = atoi(arg.arg_value.c_str());
                mx::system_out << "acmx2: MIDI device index: " << args.midi_device << "\n";
                break;
            case 502: {
                try {
                    RtMidiIn midi;
                    unsigned int ports = midi.getPortCount();
                    if (ports == 0) {
                        mx::system_out << "No MIDI input devices found.\n";
                    } else {
                        mx::system_out << "MIDI Input Devices (" << ports << "):\n";
                        for (unsigned int i = 0; i < ports; ++i) {
                            mx::system_out << "  " << i << ": " << midi.getPortName(i) << "\n";
                        }
                    }
                } catch (RtMidiError &e) {
                    mx::system_err << "MIDI error: " << e.getMessage() << "\n";
                }
                fflush(stdout);
                exit(EXIT_SUCCESS);
                break;
            }
#endif
            }
        }
    } catch (const ArgException<std::string> &e) {
        mx::system_err << e.text() << "\n";
        mx::system_err.flush();
        return EXIT_FAILURE;
    }
    checkDevices();
    if (args.path.empty()) {
        const char *env_path = std::getenv("ACMX2_PATH");
        if (env_path && env_path[0] != '\0') {
            args.path = env_path;
            mx::system_out << "acmx2: Using ACMX2_PATH environment variable: " << args.path << "\n";
        } else {
            args.path = ".";
            mx::system_out << "acmx2: Path name not provided, using current path...\n";
        }
    }
    if (args.library.empty() && args.fragment.empty()) {
        const char *env_shader_path = std::getenv("ACMX2_SHADER_PATH");
        if (env_shader_path && env_shader_path[0] != '\0') {
            args.library = env_shader_path;
            args.mode = 1;
            mx::system_out << "acmx2: Using ACMX2_SHADER_PATH environment variable: " << args.library << "\n";
        }
    }
    if (args.library.empty() && args.fragment.empty()) {
        args.fragment = args.path + "/frag.glsl";
    }
    if (args.library.empty() && args.mode == 1) {
        args.library = args.path + "/filters";
    }
    if (args.remove_broken) {
        if (args.remove_broken_path.empty()) {
            mx::system_err << "acmx2: Error: --remove-broken requires a shader library path\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }
        if (!std::filesystem::exists(args.remove_broken_path + "/index.txt")) {
            mx::system_err << "acmx2: Error: No index.txt found at: " << args.remove_broken_path << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }
        try {
#if defined(__linux__)
            if (args.silent) {
                // Make remove-broken use the same offscreen path as silent batch mode.
                setenv("SDL_VIDEODRIVER", "offscreen", 0);
                setenv("SDL_AUDIODRIVER", "dummy", 0);
                SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
                installHeadlessSignalHandlers();
                mx::system_out << "acmx2: remove-broken headless mode enabled (Linux)\n";
            }
#endif
            mx::system_out << "acmx2: Creating scan window for remove-broken...\n";
            fflush(stdout);

            /**
             * @brief Headless GL context used exclusively by the
             *        `--remove-broken` CLI flag.
             *
             * Mirrors `BuildWindow`: opens a minimal hidden 640x480 window
             * just long enough to get a valid OpenGL context, invokes
             * `ShaderLibrary::removeBrokenShaders()` once, and then exits.
             */
            class RemoveBrokenWindow : public gl::GLWindow {
              public:
                ShaderLibrary library;
                std::string lib_path;
                bool enable_3d;
                std::string assets_path;
                bool success = false;
                bool done = false;
                bool active = true;

                RemoveBrokenWindow(const std::string &path, bool is3d, const std::string &assets)
                    : gl::GLWindow("ACMX2 Remove-Broken", 640, 480, false),
                      lib_path(path), enable_3d(is3d), assets_path(assets) {
                    util.path = assets_path;
                    library.enableDualMode(enable_3d);
                }

                                RemoveBrokenWindow(const std::string &path, bool is3d, const std::string &assets, bool)
                                        : gl::GLWindow(640, 480, gl::GLMode::DESKTOP),
                                            lib_path(path), enable_3d(is3d), assets_path(assets) {
                                        util.path = assets_path;
                                        library.enableDualMode(enable_3d);
                                }

                void draw() override {
                    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT);
                    if (!done) {
                        done = true;
                        std::string vert_2d = util.getFilePath("data/vert.glsl");
                        std::string vert_3d = util.getFilePath("data/vertex.glsl");
                        mx::system_out << "acmx2: Scanning library: " << lib_path << "\n";
                        mx::system_out << "acmx2: Mode: " << (enable_3d ? "2D+3D" : "2D only") << "\n";
                        mx::system_out << "acmx2: OpenGL Renderer: " << safeGLString(GL_RENDERER) << "\n";
                        mx::system_out << "acmx2: OpenGL Version: " << safeGLString(GL_VERSION) << "\n";
                        fflush(stdout);
                        success = library.removeBrokenShaders(this, lib_path, vert_2d, vert_3d);
                        library.clear();
                        active = false;
                    }
                    swap();
                }

                void event(SDL_Event &) override {}

                void scanLoop() {
                    SDL_Event ev;
                    while (active) {
                        while (SDL_PollEvent(&ev)) {
                            if (ev.type == SDL_QUIT) active = false;
                            event(ev);
                        }
                        draw();
                    }
                }
            };

#if defined(__linux__)
            if (args.silent) {
                RemoveBrokenWindow rb_win(args.remove_broken_path, args.is3d, args.path, true);
                rb_win.scanLoop();
                return rb_win.success ? EXIT_SUCCESS : EXIT_FAILURE;
            }
#endif
            RemoveBrokenWindow rb_win(args.remove_broken_path, args.is3d, args.path);
            rb_win.scanLoop();
            return rb_win.success ? EXIT_SUCCESS : EXIT_FAILURE;
        } catch (const mx::Exception &e) {
            mx::system_err << "acmx2: Remove-broken failed: " << e.text() << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        } catch (std::exception &e) {
            mx::system_err << "acmx2: Remove-broken failed: " << e.what() << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }
    }

    if (args.build_cache) {
        if (args.build_library_path.empty()) {
            mx::system_err << "acmx2: Error: --build requires a shader library path\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }
        if (!std::filesystem::exists(args.build_library_path + "/index.txt")) {
            mx::system_err << "acmx2: Error: No index.txt found at: " << args.build_library_path << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }

        try {
#if defined(__linux__)
            if (args.silent) {
                // Make build-cache use the same offscreen path as silent batch mode.
                setenv("SDL_VIDEODRIVER", "offscreen", 0);
                setenv("SDL_AUDIODRIVER", "dummy", 0);
                SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
                installHeadlessSignalHandlers();
                mx::system_out << "acmx2: build headless mode enabled (Linux)\n";
            }
#endif
            mx::system_out << "acmx2: Creating build window...\n";
            fflush(stdout);

            /**
             * @brief Headless-style GLWindow used exclusively by the `--build` CLI flag.
             *
             * Creates a minimal 640×480 hidden window just long enough to
             * establish an OpenGL context, compile every shader in the
             * library, serialise the resulting GPU program binaries to
             * disk via ShaderLibrary::buildShaderCache(), then exit.
             *
             * The window's draw() fires exactly once (guarded by
             * `build_done`), performs the full cache build, and then
             * sets `active = false` so buildLoop() terminates.
             */
            class BuildWindow : public gl::GLWindow {
              public:
                ShaderLibrary library;   ///< Temporary shader library for compilation.
                std::string lib_path;    ///< Path to the shader source directory.
                bool enable_3d;          ///< Whether to include 3-D vertex shaders.
                std::string assets_path; ///< Base asset path (for locating vert.glsl).
                bool success = false;    ///< True if buildShaderCache() succeeded.
                bool build_done = false; ///< Guard: ensures draw() builds only once.
                bool active = true;      ///< Controls the buildLoop() pump.

                /**
                 * @brief Construct a build window and configure the shader library.
                 * @param path   Path to the shader source directory.
                 * @param is3d   Include 3-D shaders in the cache.
                 * @param assets Base asset path for vertex shader lookup.
                 */
                BuildWindow(const std::string &path, bool is3d, const std::string &assets)
                    : gl::GLWindow("ACMX2 Shader Builder", 640, 480, false),
                      lib_path(path), enable_3d(is3d), assets_path(assets) {
                    mx::system_out << "acmx2: Window created, setting up...\n";
                    fflush(stdout);
                    util.path = assets_path;
                    library.enableDualMode(enable_3d);
                }

                                BuildWindow(const std::string &path, bool is3d, const std::string &assets, bool)
                                        : gl::GLWindow(640, 480, gl::GLMode::DESKTOP),
                                            lib_path(path), enable_3d(is3d), assets_path(assets) {
                                        mx::system_out << "acmx2: Window created, setting up...\n";
                                        fflush(stdout);
                                        util.path = assets_path;
                                        library.enableDualMode(enable_3d);
                                }

                /**
                 * @brief Single-shot draw: compile all shaders, write cache, then signal exit.
                 *
                 * On the first (and only) call, resolves vertex shader paths,
                 * invokes ShaderLibrary::buildShaderCache(), clears GPU
                 * resources, and sets `active = false` to break the pump.
                 */
                void draw() override {
                    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT);

                    if (!build_done) {
                        build_done = true;

                        // Display logo.png while the build runs
                        static constexpr const char *kBldLogoVert =
                            "#version 330 core\n"
                            "layout(location = 0) in vec3 aPos;\n"
                            "layout(location = 1) in vec2 aTex;\n"
                            "out vec2 tc;\n"
                            "void main() { gl_Position = vec4(aPos, 1.0); tc = aTex; }\n";
                        static constexpr const char *kBldLogoFrag =
                            "#version 330 core\n"
                            "in vec2 tc;\n"
                            "out vec4 color;\n"
                            "uniform sampler2D samp;\n"
                            "void main() { color = texture(samp, tc); }\n";
                        gl::ShaderProgram logo_sh;
                        gl::GLSprite logo_sp;
                        std::string logo_path = util.getFilePath("data/logo.png");
                        if (std::filesystem::exists(logo_path)) {
                            GLuint logo_tex = 0;
                            try {
                                int lw = 0, lh = 0;
                                logo_tex = gl::loadTexture(logo_path, lw, lh);
                                if (logo_tex && logo_sh.loadProgramFromText(kBldLogoVert, kBldLogoFrag)) {
                                    logo_sp.initSize(w, h);
                                    logo_sp.setName("samp");
                                    logo_sp.setShader(&logo_sh);
                                    float scale = std::min((float)w / lw, (float)h / lh);
                                    int dw = static_cast<int>(lw * scale);
                                    int dh = static_cast<int>(lh * scale);
                                    int lx = (w - dw) / 2;
                                    int ly = (h - dh) / 2;
                                    logo_sp.initWithTexture(&logo_sh, logo_tex, lx, ly, dw, dh);
                                    logo_tex = 0;
                                    logo_sp.draw();
                                }
                            } catch (...) {}
                            if (logo_tex) { glDeleteTextures(1, &logo_tex); }
                        }
                        swap();
                        SDL_PumpEvents();

                        std::string vert_2d = util.getFilePath("data/vert.glsl");
                        std::string vert_3d = util.getFilePath("data/vertex.glsl");

                        mx::system_out << "acmx2: Building shader cache for: " << lib_path << "\n";
                        mx::system_out << "acmx2: Mode: " << (enable_3d ? "2D+3D" : "2D only") << "\n";
                        mx::system_out << "acmx2: OpenGL Renderer: " << safeGLString(GL_RENDERER) << "\n";
                        mx::system_out << "acmx2: OpenGL Version: " << safeGLString(GL_VERSION) << "\n";
                        fflush(stdout);
                        success = library.buildShaderCache(this, lib_path, vert_2d, vert_3d);
                        library.clear();
                        active = false;
                    }

                    swap();
                }

                /// @brief No-op; build mode ignores all user input.
                void event(SDL_Event &e) override {}
                /**
                 * @brief Minimal SDL event pump that drives draw() until `active` is cleared.
                 *
                 * Polls SDL events (honouring SDL_QUIT), calls draw() once per
                 * iteration, and exits when the build is complete.
                 */
                void buildLoop() {
                    SDL_Event ev;
                    while (active) {
                        while (SDL_PollEvent(&ev)) {
                            if (ev.type == SDL_QUIT) {
                                active = false;
                            }
                            event(ev);
                        }
                        draw();
                    }
                }
            };

#if defined(__linux__)
            if (args.silent) {
                BuildWindow build_win(args.build_library_path, args.is3d, args.path, true);
                build_win.buildLoop();
                return build_win.success ? EXIT_SUCCESS : EXIT_FAILURE;
            }
#endif
            BuildWindow build_win(args.build_library_path, args.is3d, args.path);
            build_win.buildLoop();

            return build_win.success ? EXIT_SUCCESS : EXIT_FAILURE;
        } catch (const mx::Exception &e) {
            mx::system_err << "acmx2: Build failed: " << e.text() << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        } catch (std::exception &e) {
            mx::system_err << "acmx2: Build failed: " << e.what() << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }
    }

    try {
        args.slib = std::make_tuple(args.mode,
                                    (args.mode == 0) ? args.fragment : args.library,
                                    (args.mode == 0) ? 0 : args.shader_index);
        if (args.filename.empty() && args.cache) {
            throw mx::Exception("Texture cache only works in video mode\n");
        }

        if (args.silent) {
            if (args.filename.empty()) {
                mx::system_err << "acmx2: Error: --silent mode requires a video input file (-i/--input)\n";
                mx::system_err << "       Silent mode only works with video files, not camera or graphics input.\n";
                mx::system_err.flush();
                return EXIT_FAILURE;
            }
            if (args.ofilename.empty()) {
                mx::system_err << "acmx2: Error: --silent mode requires an output file (-o/--output)\n";
                mx::system_err.flush();
                return EXIT_FAILURE;
            }
            if (!args.graphic_file.empty()) {
                mx::system_err << "acmx2: Error: --silent mode cannot be used with graphics input (-g/--graphic)\n";
                mx::system_err << "       Silent mode only works with video files.\n";
                mx::system_err.flush();
                return EXIT_FAILURE;
            }
            mx::system_out << "acmx2: Silent mode enabled - processing without window\n";

            // In headless/batch mode stdout is usually a pipe (tee, logger,
            // CI capture). Force line buffering so every progress line is
            // delivered as soon as it's written, regardless of the child
            // process's default fully-buffered pipe mode.
            std::setvbuf(stdout, nullptr, _IOLBF, 0);

            // True headless mode: force SDL to use the 'offscreen' video
            // driver so the process runs without any X11 / Wayland display
            // server. The offscreen driver uses EGL surfaceless rendering to
            // provide an OpenGL context. Use setenv with overwrite=0 so a
            // user who explicitly set SDL_VIDEODRIVER (e.g. to force a
            // specific backend for debugging) is respected. Also disable
            // joystick / gamecontroller / audio subsystems that can block or
            // fail on a pure headless server.
            setenv("SDL_VIDEODRIVER", "offscreen", 0);
            setenv("SDL_AUDIODRIVER", "dummy", 0);
            SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
            mx::system_out << "acmx2: Headless: SDL_VIDEODRIVER="
                           << (getenv("SDL_VIDEODRIVER") ? getenv("SDL_VIDEODRIVER") : "(unset)")
                           << ", SDL_AUDIODRIVER="
                           << (getenv("SDL_AUDIODRIVER") ? getenv("SDL_AUDIODRIVER") : "(unset)") << "\n";
#if defined(__linux__)
            // Install Ctrl+C / SIGTERM / SIGHUP handlers so batch/headless runs
            // can be interrupted cleanly: the writer flushes, mp4 trailer is
            // written, and the partial output file stays playable.
            installHeadlessSignalHandlers();
            mx::system_out << "acmx2: Headless: signal handlers installed (SIGINT, SIGTERM, SIGHUP)\n";
#endif
        }

        SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
        SDL_SetHint("SDL_VIDEO_WAYLAND_WMCLASS", "acmx2");
        SDL_SetHint("SDL_VIDEO_X11_WMCLASS", "ACMX2");

        if (args.silent) {
            MainWindow main_window(args, true);
            main_window.loop();
        } else {
            MainWindow main_window(args);
            main_window.loop();
        }
    } catch (const mx::Exception &e) {
        mx::system_err << "acmx2: Exception: " << e.text() << "\n";
        mx::system_err.flush();
        return EXIT_FAILURE;
    } catch (std::exception &e) {
        mx::system_err << "acmx2: Exception: " << e.what() << "\n";
        mx::system_err.flush();
        return EXIT_FAILURE;
    } catch (...) {
        mx::system_err << "acmx2: Unknown exception occurred.\n";
        mx::system_err.flush();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}