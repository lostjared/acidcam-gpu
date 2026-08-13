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
 * - **TextureUploader** — Host-zero-copy CUDA↔OpenGL image interop (direct cudaArray binding) for GPU frames.
 * - **ShaderCache / ShaderLibrary** — Compile, cache, and manage GLSL shader programs.
 * - **FrameCache** — Ring-buffer of recent frames for temporal ("cache") shaders.
 * - **SnapshotThreadPool** — Async PNG snapshot writer.
 * - **ACView** — Main GL object: capture → filter → shade → record pipeline.
 * - **MainWindow** — SDL2/OpenGL window host.
 *
 * @copyright (C) 2026 LostSideDead Software — BSD 2-Clause License
 * @see https://lostsidedead.biz
 */

#include "mxwrite.hpp"
#include "shader_selection_shm.hpp"
#include "version_info.hpp"
#include <algorithm>
#include <argz.hpp>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gl.hpp>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <mx.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#ifdef AUDIO_ENABLED
#include "audio.hpp"
#include "file_audio.hpp"

using acmx2::audio::FFT_SIZE;
#endif
#ifdef MIDI_ENABLED
#include <rtmidi/RtMidi.h>
#endif
#include "program.hpp"
#ifdef ACMX2_WITH_DNN
#include "dnn.hpp"
#include <memory>
#endif
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
#define CHECK_CUDA(call)         \
    do {                         \
        static_cast<void>(call); \
    } while (0)
#endif
namespace ac_gpu {
    inline constexpr int AC_FILTER_MAX = 0;
    struct Filter {
        int index;
        std::string name;
    };
    struct GPUFilter {
        int index;
    };
    struct DynamicFrameBuffer {
        int arraySize = 0;
    };
    // Empty filter table so code referencing ac_gpu::filters still compiles.
    // (Never indexed in no-CUDA builds because AC_FILTER_MAX == 0 guards all uses.)
    inline Filter filters[1] = {{0, ""}};
} // namespace ac_gpu
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <deque>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <model.hpp>
#ifdef ACMX2_WITH_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
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

struct OpenGLContextConfig {
    int major = 4;
    int minor = 1;
};

/** True when the active ACMX2 context can run OpenGL compute shaders. */
static bool compute_shader_supported = false;

#if defined(__linux__)
/**
 * @brief Test whether SDL can create a core-profile context of a given version.
 *
 * OpenGL capabilities cannot be queried without first creating a context, so
 * Linux startup uses a small hidden probe window before constructing the real
 * ACMX2 window. The probe uses the already-selected SDL video driver, including
 * the offscreen driver selected by silent mode.
 */
static bool probe_open_gl_context(int major, int minor, std::string &error_message) {
    const bool video_was_initialized = (SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO) != 0;
    if (!video_was_initialized && SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        error_message = SDL_GetError();
        return false;
    }

    SDL_GL_ResetAttributes();
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    SDL_Window *probe_window = SDL_CreateWindow(
        "ACMX2 OpenGL Probe", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        32, 32, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!probe_window) {
        error_message = SDL_GetError();
        if (!video_was_initialized) {
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
        return false;
    }

    SDL_GLContext probe_context = SDL_GL_CreateContext(probe_window);
    bool version_supported = false;
    if (probe_context) {
        int actual_major = 0;
        int actual_minor = 0;
        SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &actual_major);
        SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &actual_minor);
        version_supported = actual_major > major ||
                            (actual_major == major && actual_minor >= minor);
        if (!version_supported) {
            error_message = "driver returned OpenGL " + std::to_string(actual_major) +
                            "." + std::to_string(actual_minor);
        }
        SDL_GL_DeleteContext(probe_context);
    } else {
        error_message = SDL_GetError();
    }

    SDL_DestroyWindow(probe_window);
    if (!video_was_initialized) {
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
    return version_supported;
}

/** Select OpenGL 4.3 for compute shaders, falling back to ACMX2's 4.1 baseline. */
static OpenGLContextConfig select_open_gl_context() {
    std::string error_message;
    if (probe_open_gl_context(4, 3, error_message)) {
        mx::system_out << "acmx2: OpenGL 4.3 context available; compute shaders enabled\n";
        return {4, 3};
    }

    mx::system_out << "acmx2: OpenGL 4.3 unavailable (" << error_message
                   << "); falling back to OpenGL 4.1 with compute shaders disabled\n";
    error_message.clear();
    if (!probe_open_gl_context(4, 1, error_message)) {
        throw std::runtime_error("OpenGL 4.1 or newer is required: " + error_message);
    }
    return {4, 1};
}
#else
static OpenGLContextConfig select_open_gl_context() {
    return {4, 1};
}
#endif

/** Update the feature flag from the real context rather than only the probe. */
static void update_compute_shader_support() {
#if defined(__linux__)
    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    compute_shader_supported = major > 4 || (major == 4 && minor >= 3);
#else
    compute_shader_supported = false;
#endif
    mx::system_out << "acmx2: Compute shader support: "
                   << (compute_shader_supported ? "enabled" : "disabled") << "\n";
}

/**
 * @brief Print the active driver's shader-uniform capacity in one line.
 *
 * GL_MAX_UNIFORM_LOCATIONS became core in OpenGL 4.3. Older contexts still
 * expose the per-stage component limits, so report locations as unsupported
 * instead of issuing an invalid glGetIntegerv query.
 */
static void print_open_gl_uniform_limits() {
    GLint max_vertex_components = 0;
    GLint max_fragment_components = 0;
    GLint max_uniform_locations = 0;

    glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_vertex_components);
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &max_fragment_components);

    GLint major_version = 0;
    GLint minor_version = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major_version);
    glGetIntegerv(GL_MINOR_VERSION, &minor_version);
    bool uniform_locations_supported =
        major_version > 4 || (major_version == 4 && minor_version >= 3);
#ifdef GL_ARB_explicit_uniform_location
    uniform_locations_supported =
        uniform_locations_supported ||
        SDL_GL_ExtensionSupported("GL_ARB_explicit_uniform_location") == SDL_TRUE;
#endif

#ifdef GL_MAX_UNIFORM_LOCATIONS
    if (uniform_locations_supported) {
        glGetIntegerv(GL_MAX_UNIFORM_LOCATIONS, &max_uniform_locations);
    }
#else
    uniform_locations_supported = false;
#endif

    mx::system_out
        << "acmx2: OpenGL uniform limits: vertex components="
        << max_vertex_components << " (GL_MAX_VERTEX_UNIFORM_COMPONENTS), "
        << "fragment components=" << max_fragment_components
        << " (GL_MAX_FRAGMENT_UNIFORM_COMPONENTS), uniform locations=";
    if (uniform_locations_supported) {
        mx::system_out << max_uniform_locations;
    } else {
        mx::system_out << "unsupported by this OpenGL context";
    }
    mx::system_out << " (GL_MAX_UNIFORM_LOCATIONS)\n";
}

#ifdef __linux__
/**
 * @brief Return whether a numbered V4L2 device is provided by v4l2loopback.
 *
 * Loopback devices expose only their current frame interval through
 * VIDIOC_ENUM_FRAMEINTERVALS, even though consumers may select another
 * time-per-frame value with VIDIOC_S_PARM.  Detecting them lets camera
 * enumeration expose useful high-speed choices to the Qt interface.
 */
static bool isV4l2LoopbackDevice(int device_index) {
    const std::string device_path = "/dev/video" + std::to_string(device_index);
    const int fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        return false;
    }

    v4l2_capability capability{};
    const bool queried = ioctl(fd, VIDIOC_QUERYCAP, &capability) == 0;
    close(fd);
    if (!queried) {
        return false;
    }

    const std::string driver(reinterpret_cast<const char *>(capability.driver));
    return driver.find("v4l2loopback") != std::string::npos ||
           driver.find("v4l2 loopback") != std::string::npos;
}
#endif

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

enum class ShaderManifestFormat { Json,
                                  Text };

enum class ShaderProgramKind { Fragment,
                               Compute,
                               ComputeUnavailable };

static bool isComputeShaderFile(const std::string &path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return extension == ".comp";
}

struct ShaderManifestData {
    ShaderManifestFormat format = ShaderManifestFormat::Text;
    std::string path;
    std::vector<std::string> entries;
    struct CustomUniform {
        std::string name;
        double minimum = 0.0;
        double maximum = 1.0;
        double step = 0.01;
        double value = 0.0;
    };
    std::vector<CustomUniform> custom_uniforms;
};

static bool isValidCustomUniformName(const std::string &name) {
    if (name.empty() ||
        name.size() >= acmx2::ipc::kShaderSelectionMaxUniformName ||
        name.rfind("gl_", 0) == 0 ||
        !(std::isalpha(static_cast<unsigned char>(name.front())) ||
          name.front() == '_')) {
        return false;
    }
    return std::all_of(name.begin() + 1, name.end(), [](unsigned char ch) {
        return std::isalnum(ch) || ch == '_';
    });
}

static std::string shaderManifestPath(const std::string &library_path) {
    const std::filesystem::path root(library_path);
    const std::filesystem::path json_path = root / "library.json";
    if (std::filesystem::is_regular_file(json_path))
        return json_path.string();
    const std::filesystem::path text_path = root / "index.txt";
    if (std::filesystem::is_regular_file(text_path))
        return text_path.string();
    return {};
}

static bool loadShaderManifest(const std::string &library_path,
                               ShaderManifestData &manifest,
                               std::string &error) {
    manifest = {};
    error.clear();
    manifest.path = shaderManifestPath(library_path);
    if (manifest.path.empty()) {
        error = "No library.json or index.txt found at: " + library_path;
        return false;
    }

    if (std::filesystem::path(manifest.path).extension() == ".json") {
        manifest.format = ShaderManifestFormat::Json;
        try {
            cv::FileStorage storage(manifest.path,
                                    cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            if (!storage.isOpened()) {
                error = "Could not open shader manifest: " + manifest.path;
                return false;
            }
            const cv::FileNode shaders = storage["shaders"];
            if (shaders.type() == cv::FileNode::NONE || !shaders.isSeq()) {
                error = manifest.path + " must contain a 'shaders' array";
                return false;
            }
            for (const cv::FileNode &entry : shaders) {
                std::string file;
                if (entry.isString()) {
                    entry >> file;
                } else if (entry.isMap() && !entry["file"].empty()) {
                    entry["file"] >> file;
                } else {
                    error = manifest.path + " contains a shader entry without a file name";
                    return false;
                }
                manifest.entries.push_back(std::move(file));
            }
            const cv::FileNode uniforms = storage["custom_uniforms"];
            if (!uniforms.empty()) {
                if (!uniforms.isMap()) {
                    error = manifest.path + " field 'custom_uniforms' must be an object";
                    return false;
                }
                for (auto it = uniforms.begin(); it != uniforms.end(); ++it) {
                    if (manifest.custom_uniforms.size() >=
                        acmx2::ipc::kShaderSelectionMaxCustomUniforms) {
                        error = manifest.path + " contains too many custom uniforms";
                        return false;
                    }
                    const cv::FileNode entry = *it;
                    ShaderManifestData::CustomUniform uniform;
                    uniform.name = entry.name();
                    if (!entry.isMap() || !isValidCustomUniformName(uniform.name)) {
                        error = manifest.path + " contains an invalid custom uniform: " +
                                uniform.name;
                        return false;
                    }
                    if (!entry["minimum"].empty())
                        entry["minimum"] >> uniform.minimum;
                    if (!entry["maximum"].empty())
                        entry["maximum"] >> uniform.maximum;
                    if (!entry["step"].empty())
                        entry["step"] >> uniform.step;
                    uniform.value = uniform.minimum;
                    if (!entry["value"].empty())
                        entry["value"] >> uniform.value;
                    if (!std::isfinite(uniform.minimum) ||
                        !std::isfinite(uniform.maximum) ||
                        !std::isfinite(uniform.step) ||
                        !std::isfinite(uniform.value) ||
                        uniform.maximum <= uniform.minimum || uniform.step <= 0.0) {
                        error = manifest.path + " contains an invalid range for custom uniform: " +
                                uniform.name;
                        return false;
                    }
                    uniform.value = std::clamp(uniform.value, uniform.minimum,
                                               uniform.maximum);
                    manifest.custom_uniforms.push_back(std::move(uniform));
                }
            }
            return true;
        } catch (const cv::Exception &e) {
            error = "Could not parse " + manifest.path + ": " + e.what();
            return false;
        }
    }

    manifest.format = ShaderManifestFormat::Text;
    std::ifstream file(manifest.path);
    if (!file.is_open()) {
        error = "Could not open shader manifest: " + manifest.path;
        return false;
    }
    std::string line;
    while (std::getline(file, line))
        manifest.entries.push_back(std::move(line));
    return true;
}

static bool writeJsonShaderManifest(const std::string &path,
                                    const std::vector<std::string> &entries,
                                    const std::vector<ShaderManifestData::CustomUniform> &custom_uniforms,
                                    std::string &error) {
    try {
        cv::FileStorage storage(path,
                                cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!storage.isOpened()) {
            error = "Could not write shader manifest: " + path;
            return false;
        }
        storage << "version" << 1;
        storage << "shaders"
                << "[";
        for (const std::string &entry : entries)
            storage << entry;
        storage << "]";
        if (!custom_uniforms.empty()) {
            storage << "custom_uniforms"
                    << "{";
            for (const auto &uniform : custom_uniforms) {
                storage << uniform.name
                        << "{"
                        << "minimum" << uniform.minimum
                        << "maximum" << uniform.maximum
                        << "step" << uniform.step
                        << "value" << uniform.value
                        << "}";
            }
            storage << "}";
        }
        storage.release();
        return true;
    } catch (const cv::Exception &e) {
        error = "Could not write " + path + ": " + e.what();
        return false;
    }
}

static bool collectShaderLibraryEntries(const std::string &library_path,
                                        std::vector<std::string> &shader_files,
                                        std::string &error) {
    ShaderManifestData manifest;
    if (!loadShaderManifest(library_path, manifest, error))
        return false;
    shader_files.clear();
    for (const std::string &entry : manifest.entries) {
        const auto shader_entry = normalizeShaderIndexEntry(entry);
        if (!shader_entry)
            continue;
        std::string full_path;
        if (resolveShaderPathInLibrary(library_path, *shader_entry, full_path))
            shader_files.push_back(*shader_entry);
    }
    std::sort(shader_files.begin(), shader_files.end(),
              [](const std::string &a, const std::string &b) {
                  return std::lexicographical_compare(
                      a.begin(), a.end(), b.begin(), b.end(),
                      [](unsigned char ca, unsigned char cb) {
                          return std::tolower(ca) < std::tolower(cb);
                      });
              });
    return true;
}

static std::vector<std::string> sortedShaderLibraryEntries(const std::string &library_path) {
    std::vector<std::string> shader_files;
    std::string error;
    if (!collectShaderLibraryEntries(library_path, shader_files, error))
        mx::system_err << "acmx2: " << error << "\n";
    return shader_files;
}

static int shaderIndexForFile(const std::vector<std::string> &shader_files,
                              const std::string &shader_file) {
    const auto selected = std::find(shader_files.begin(), shader_files.end(),
                                    shader_file);
    if (selected == shader_files.end())
        return -1;
    return static_cast<int>(std::distance(shader_files.begin(), selected));
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
} // namespace

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
// The transfer function (PQ or `) is *not* applied here: the bits stay in
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
        return false; // planar formats need all three planes.
    }

    // Detect 10-bit sample position within a 16-bit container:
    //   yuv420p10le  -> low 10 bits (shift = 0)
    //   p010le       -> high 10 bits (shift = 6, so divide by 64)
    int sample_shift = 0;
    if (is_p010) {
        sample_shift = 6;
    }

    const int y_stride_b = src->linesize[0];
    const int uv_stride_b = src->linesize[1]; // UV interleaved (p010) or Cb (planar).
    const int v_stride_b = is_p010 ? 0 : src->linesize[2];

    // BT.2020 non-constant-luminance inverse matrix (per ITU-R BT.2020 §4).
    constexpr float kCrR = 1.4746f;
    constexpr float kCbG = -0.16455312684366f; // -2*(1-0.2627)*0.2627/0.6780
    constexpr float kCrG = -0.57135313725490f; // -2*(1-0.0593)*0.0593/0.6780
    constexpr float kCbB = 1.8814f;

    out.create(h, w, CV_16UC4);

    auto sample10 = [&](const uint8_t *plane, int stride_b, int x, int y) -> int {
        const uint16_t raw = *reinterpret_cast<const uint16_t *>(
            plane + y * stride_b + x * 2);
        return static_cast<int>(raw >> sample_shift);
    };

    const uint8_t *yp = src->data[0];
    const uint8_t *up = src->data[1]; // planar: Cb plane | p010: interleaved Cb,Cr.
    const uint8_t *vp = is_p010 ? nullptr : src->data[2];

    // 10-bit limited-range BT.2020 scaling:
    //   Y' = (Y_sample - 64)  / 876    (876 = 940-64)
    //   Cb = (Cb_sample - 512)/ 896    (896 = 2*448)
    //   Cr = (Cr_sample - 512)/ 896
    constexpr float kInvY = 1.0f / 876.0f;
    constexpr float kInvC = 1.0f / 896.0f;

    for (int y = 0; y < h; ++y) {
        const int cy = y >> 1; // 4:2:0 vertical subsampling, nearest.
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

            const float Y = (Ys - 64) * kInvY;
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
#endif // ACMX2_WITH_WEBP

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

    const uint16_t extra[1] = {EXTRASAMPLE_UNASSALPHA};
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

    const uint16_t extra[1] = {EXTRASAMPLE_UNASSALPHA};
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
#endif // ACMX2_WITH_TIFF

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

constexpr const char *kMuxOverlayFrag =
    "#version 330 core\n"
    "out vec4 color;\n"
    "in vec2 tc;\n"
    "uniform sampler2D samp;\n"
    "uniform float time_f;\n"
    "uniform vec2 iResolution;\n"
    "uniform float alpha;\n"
    "void main(void) {\n"
    "    vec2 uv = (tc * 2.0 - 1.0);\n"
    "    float aspect = iResolution.x / iResolution.y;\n"
    "    uv.x *= aspect;\n"
    "    float d = length(uv);\n"
    "    float lensStrength = 1.5;\n"
    "    vec3 normal = normalize(vec3(uv, 1.0 / lensStrength));\n"
    "    float fisheyeRadius = atan(d, 1.0);\n"
    "    vec2 distortedUV = normalize(uv + 1e-6) * fisheyeRadius;\n"
    "    float t = time_f * 0.8;\n"
    "    float r_dist = length(distortedUV);\n"
    "    float angle = atan(distortedUV.y, distortedUV.x);\n"
    "    float spiral = angle + (log(r_dist + 0.1) * 3.0) - t * 1.5;\n"
    "    float r = sin(spiral * 3.0 + t);\n"
    "    float g = sin(spiral * 3.0 + t + 2.094);\n"
    "    float b = sin(spiral * 3.0 + t + 4.188);\n"
    "    vec3 spiralCol = vec3(r, g, b) * 0.5 + 0.5;\n"
    "    vec3 lightDir = normalize(vec3(sin(time_f), cos(time_f), 1.0));\n"
    "    float diff = max(dot(normal, lightDir), 0.0);\n"
    "    float spec = pow(max(dot(reflect(-lightDir, normal), vec3(0,0,1)), 0.0), 16.0);\n"
    "    vec4 texColor = texture(samp, tc);\n"
    "    vec3 finalCol = mix(texColor.rgb, spiralCol * (diff + 0.5) + spec, 0.7);\n"
    "    finalCol *= smoothstep(2.0, 0.5, d);\n"
    "    float finalAlpha = texColor.a * alpha;\n"
    "    color = vec4(finalCol, finalAlpha);\n"
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
        static_cast<void>(prefer_cuda);
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

    /**
     * @brief Decode and discard one frame without transferring or converting it.
     *
     * Used when an external playback clock has advanced beyond the next video
     * timestamp. Hardware frames stay on the decoder device and software
     * frames avoid swscale/OpenCV allocation entirely.
     */
    bool skip() {
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
                } else if (!drain_packet_sent) {
                    if (avcodec_send_packet(codec_ctx, nullptr) < 0) {
                        return false;
                    }
                    drain_packet_sent = true;
                    draining = true;
                }
            }

            const int receive_ret =
                avcodec_receive_frame(codec_ctx, decoded_frame);
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

            av_frame_unref(decoded_frame);
            av_frame_unref(sw_frame);
            ++current_frame;
            return true;
        }
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
                    src_space = SWS_CS_BT2020; // HDR default.
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
        static_cast<void>(codec);
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
    std::vector<std::thread> workers;        ///< Persistent worker threads.
    std::queue<std::function<void()>> tasks; ///< FIFO of pending PNG-write tasks.
    std::mutex queue_mutex;                  ///< Protects @c tasks and @c stop.
    std::condition_variable condition;       ///< Signalled when a task is enqueued or pool stops.
    bool stop = false;                       ///< When true, workers exit after draining the queue.
};

/**
 * @class FrameCache
 * @brief Fixed-capacity ring buffer of GL textures for temporal shaders.
 *
 * Legacy shaders whose filename contains "cache" receive previous frames as
 * separate sampler2D uniforms. Array-cache shaders instead receive one
 * sampler2DArray whose layers form the ring. In either mode only the newest
 * frame is uploaded into the next physical slot. The "oldest → newest"
 * indexing used by shaders is preserved via a logical-to-physical offset.
 *
 * This mirrors the texture-ring approach used by gl_compute_cv and
 * eliminates 7 BGR→RGBA conversions plus 7 texture uploads per push.
 */
class FrameCache {
  public:
    /**
     * @param num Maximum number of frames to retain.
     * @param use_array Store the ring in one `GL_TEXTURE_2D_ARRAY` instead of
     *                  separate `GL_TEXTURE_2D` objects.
     */
    explicit FrameCache(std::size_t num, bool use_array = false)
        : num_frames(num),
          use_history_array(use_array) {
    }
    ~FrameCache() { cleanup(); }

    FrameCache(const FrameCache &) = delete;
    FrameCache &operator=(const FrameCache &) = delete;

    /**
     * @brief Allocate the ring of GL textures sized to @p w x @p h.
     *
     * Must be called once a GL context is current. Existing textures (if
     * any) are released first. The ring remains empty until the first source
     * frame is replicated into every slot by one of the push methods.
     *
     * @param w   Texture width in pixels.
     * @param h   Texture height in pixels.
     * @param hdr When true, allocate textures as @c GL_RGBA16F (matching
     *            the HDR linear-light pipeline) instead of @c GL_RGBA.
     */
    void init(int w, int h, bool hdr = false) {
        cleanup();
        if (num_frames == 0)
            return;
        width = w;
        height = h;
        is_hdr = hdr;
        if (use_history_array) {
            allocateHistoryTexture();
            head = 0;
            count = 0;
            return;
        }

        textures.assign(num_frames, 0);
        glGenTextures(static_cast<GLsizei>(num_frames), textures.data());
        const GLint internal = hdr ? GL_RGBA16F : GL_RGBA;
        const GLenum type = hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
        const std::size_t bytes_per_pixel = hdr ? 8u : 4u;
        std::vector<unsigned char> zeros(
            static_cast<size_t>(w) * static_cast<size_t>(h) * bytes_per_pixel, 0);
        for (std::size_t i = 0; i < num_frames; ++i) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0,
                         GL_RGBA, type, zeros.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        head = 0;
        count = 0;
    }

    /// Release all GL textures and reset the ring state.
    void cleanup() {
        if (!textures.empty()) {
            glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
            textures.clear();
        }
        if (scratch_fbo != 0) {
            glDeleteFramebuffers(1, &scratch_fbo);
            scratch_fbo = 0;
        }
        if (history_texture != 0) {
            glDeleteTextures(1, &history_texture);
            history_texture = 0;
        }
        head = 0;
        count = 0;
        width = 0;
        height = 0;
        is_hdr = false;
    }

    /**
     * @brief GPU-side push from a source FBO into the next ring slot.
     *
     * Used by the HDR pipeline to copy the post-decode linear BT.2020
     * texture into an @c RGBA16F ring without any CPU readback or pixel
     * format conversion. The source FBO must have a colour attachment
     * sized at least @p w x @p h and using a compatible internal format
     * (e.g. @c GL_RGBA16F when the cache was initialised with @c hdr=true).
     *
     * @param src_fbo Read framebuffer object containing the new frame.
     * @param w,h     Region to copy (typically the full FBO dimensions).
     */
    void pushFromFBO(GLuint src_fbo, int w, int h) {
        if (!hasStorage())
            return;
        GLint prev_read = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prev_read);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, src_fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        if (use_history_array) {
            if (w != width || h != height) {
                width = w;
                height = h;
                allocateHistoryTexture();
                head = 0;
                count = 0;
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, history_texture);
            if (count == 0) {
                for (std::size_t i = 0; i < num_frames; ++i) {
                    glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0,
                                        static_cast<GLint>(i), 0, 0, w, h);
                }
            } else {
                glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0,
                                    static_cast<GLint>(head), 0, 0, w, h);
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read);
            finishPush();
            return;
        }

        if (w != width || h != height) {
            const GLint internal = is_hdr ? GL_RGBA16F : GL_RGBA;
            const GLenum type = is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
            for (GLuint texture : textures) {
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0,
                             GL_RGBA, type, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            width = w;
            height = h;
            head = 0;
            count = 0;
        }
        if (count == 0) {
            for (GLuint texture : textures) {
                glBindTexture(GL_TEXTURE_2D, texture);
                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
            }
        } else {
            glBindTexture(GL_TEXTURE_2D, textures[head]);
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read);
        finishPush();
    }

    /**
     * @brief Upload @p frame (BGR) into the next ring slot.
     *
     * When the ring is full this overwrites the oldest slot in place,
     * which becomes the new "newest" frame. Only one BGR→RGBA conversion
     * and one texture upload occur per call.
     */
    void push(const cv::Mat &frame) {
        if (!hasStorage())
            return;
        if (count == 0 || frame.cols != width || frame.rows != height) {
            fill(frame);
            return;
        }
        cv::Mat tmp;
        cv::cvtColor(frame, tmp, cv::COLOR_BGR2RGBA);
        if (use_history_array) {
            glBindTexture(GL_TEXTURE_2D_ARRAY, history_texture);
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0,
                            static_cast<GLint>(head), tmp.cols, tmp.rows, 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, tmp.ptr());
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
            advance();
            return;
        }

        glBindTexture(GL_TEXTURE_2D, textures[head]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tmp.cols, tmp.rows,
                        GL_RGBA, GL_UNSIGNED_BYTE, tmp.ptr());
        glBindTexture(GL_TEXTURE_2D, 0);
        advance();
    }

    /**
     * @brief GPU-side push from an existing GL texture into the next ring slot.
     *
     * Uses a small scratch FBO so the per-frame BGR->RGBA conversion and
     * second host->GPU upload that @ref push() performs can be avoided
     * entirely once the camera texture has already been populated for
     * the shader chain. This is the path that keeps zero-copy interop
     * intact when expensive CPU passes (e.g. ONNX) run earlier in the
     * frame and the redundant cv::Mat upload starts dropping frames.
     *
     * @param src_tex GL texture currently containing the new frame.
     * @param w,h     Region to copy (typically the full texture size).
     */
    void pushFromTexture(GLuint src_tex, int w, int h) {
        if (!hasStorage() || src_tex == 0)
            return;
        if (scratch_fbo == 0) {
            glGenFramebuffers(1, &scratch_fbo);
        }
        GLint prev_read = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prev_read);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, scratch_fbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, src_tex, 0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        if (use_history_array) {
            if (w != width || h != height) {
                width = w;
                height = h;
                allocateHistoryTexture();
                head = 0;
                count = 0;
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, history_texture);
            if (count == 0) {
                for (std::size_t i = 0; i < num_frames; ++i) {
                    glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0,
                                        static_cast<GLint>(i), 0, 0, w, h);
                }
            } else {
                glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0,
                                    static_cast<GLint>(head), 0, 0, w, h);
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, 0, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read);
            finishPush();
            return;
        }

        if (w != width || h != height) {
            const GLint internal = is_hdr ? GL_RGBA16F : GL_RGBA;
            const GLenum type = is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
            for (GLuint texture : textures) {
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0,
                             GL_RGBA, type, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            width = w;
            height = h;
            head = 0;
            count = 0;
        }
        if (count == 0) {
            for (GLuint texture : textures) {
                glBindTexture(GL_TEXTURE_2D, texture);
                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
            }
        } else {
            glBindTexture(GL_TEXTURE_2D, textures[head]);
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        // Detach to avoid keeping a stale reference to the source texture.
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, 0, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, prev_read);
        finishPush();
    }

    /**
     * @brief Return the GL texture at logical index @p index.
     * @param index 0 = oldest retained frame, @c size()-1 = newest.
     */
    GLuint textureAt(std::size_t index) const {
        const std::size_t base = (count == num_frames) ? head : 0;
        return textures[(base + index) % num_frames];
    }

    /// Number of frames currently retained (0 ≤ n ≤ capacity).
    std::size_t size() const { return count; }

    /// Maximum number of frames retained by the ring.
    std::size_t capacity() const { return num_frames; }

    /// True when the ring is fully populated.
    bool isFull() const { return count == num_frames; }

    /// Array texture containing the physical history ring layers.
    GLuint historyTexture() const { return history_texture; }

    /// Physical array layer corresponding to logical history index zero.
    int oldestLayer() const {
        return isFull() ? static_cast<int>(head) : 0;
    }

    /**
     * @brief Pre-fill every slot with copies of a single frame.
     *
     * Seeds the entire cache with one frame and marks the ring as full.
     */
    void fill(const cv::Mat &frame) {
        if (!hasStorage())
            return;
        cv::Mat tmp;
        cv::cvtColor(frame, tmp, cv::COLOR_BGR2RGBA);
        const bool size_matches = (tmp.cols == width && tmp.rows == height);
        if (use_history_array) {
            if (!size_matches) {
                width = tmp.cols;
                height = tmp.rows;
                allocateHistoryTexture();
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, history_texture);
            for (std::size_t i = 0; i < num_frames; ++i) {
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0,
                                static_cast<GLint>(i), tmp.cols, tmp.rows, 1,
                                GL_RGBA, GL_UNSIGNED_BYTE, tmp.ptr());
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
            head = 0;
            count = num_frames;
            return;
        }

        for (std::size_t i = 0; i < num_frames; ++i) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            if (!size_matches) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tmp.cols, tmp.rows, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, tmp.ptr());
            } else {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tmp.cols, tmp.rows,
                                GL_RGBA, GL_UNSIGNED_BYTE, tmp.ptr());
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        if (!size_matches) {
            width = tmp.cols;
            height = tmp.rows;
        }
        head = 0;
        count = num_frames;
    }

  private:
    bool hasStorage() const {
        return use_history_array ? history_texture != 0 : !textures.empty();
    }

    void advance() {
        head = (head + 1) % num_frames;
        if (count < num_frames)
            ++count;
    }

    void finishPush() {
        if (count == 0) {
            head = 0;
            count = num_frames;
        } else {
            advance();
        }
    }

    /**
     * @brief Allocate the optional array texture using the cache's dimensions
     * and pixel format.
     */
    void allocateHistoryTexture() {
        if (!use_history_array || num_frames == 0 || width <= 0 || height <= 0)
            return;
        if (history_texture == 0) {
            glGenTextures(1, &history_texture);
        }
        glBindTexture(GL_TEXTURE_2D_ARRAY, history_texture);
        const GLint internal = is_hdr ? GL_RGBA16F : GL_RGBA;
        const GLenum type = is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, internal, width, height,
                     static_cast<GLsizei>(num_frames), 0, GL_RGBA, type, nullptr);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

    std::size_t num_frames;
    int width = 0;
    int height = 0;
    bool is_hdr = false;
    bool use_history_array = false;
    std::size_t head = 0;
    std::size_t count = 0;
    std::vector<GLuint> textures;
    GLuint scratch_fbo = 0;     ///< Scratch FBO used by pushFromTexture().
    GLuint history_texture = 0; ///< Optional array-backed history ring.
};

/**
 * @class TextureUploader
 * @brief Host-zero-copy CUDA-to-OpenGL texture transfer via direct image interop.
 *
 * Registers the OpenGL texture directly with CUDA (cudaGraphicsGLRegisterImage)
 * so that a cv::cuda::GpuMat can be copied straight into the texture's backing
 * cudaArray with a single device-to-device cudaMemcpy2DToArrayAsync. This
 * removes the intermediate PBO and the per-frame glTexSubImage2D DMA that the
 * old PBO path required, halving the per-frame interop cost.
 *
 * The copy is issued on a dedicated CUDA stream so it does not serialise with
 * other default-stream work (e.g. the GPU filter chain that produced the
 * GpuMat). A CUDA event orders that copy after the producer's default-stream
 * work, while cudaGraphicsUnmapResources provides the GL↔CUDA synchronisation
 * that subsequent OpenGL draws need.
 */
class TextureUploader {
  public:
    GLuint textureID = 0; ///< OpenGL texture receiving the frame data.
#ifdef ACMX2_WITH_CUDA
    cudaGraphicsResource *cudaTexResource = nullptr; ///< CUDA handle to the mapped GL texture.
    cudaStream_t uploadStream = nullptr;             ///< Dedicated stream for the device→array copy.
    cudaEvent_t inputReadyEvent = nullptr;           ///< Orders the copy after GpuMat production.
#endif
    int width = 0;  ///< Current texture width in pixels.
    int height = 0; ///< Current texture height in pixels.

    /**
     * @brief Create (or recreate) the GL texture and CUDA image registration.
     *
     * Allocates an RGBA8 OpenGL texture of the requested dimensions and
     * registers it with the CUDA runtime via cudaGraphicsGLRegisterImage,
     * which exposes the texture's backing storage as a cudaArray that
     * subsequent update() calls write into directly.
     *
     * If the uploader was previously initialised, cleanup() is called first
     * so that old resources are released before new ones are created.
     *
     * @param w Texture width in pixels.
     * @param h Texture height in pixels.
     */
    void init(int w, int h) {
        if (textureID != 0)
            cleanup();
        width = w;
        height = h;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        // Use a sized internal format so cudaGraphicsGLRegisterImage can match
        // it deterministically across drivers.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
#ifdef ACMX2_WITH_CUDA
        // WriteDiscard: tell the driver we don't need the previous texture
        // contents preserved on map — that is true for streaming frames and
        // lets the driver skip a possible read-back.
        CHECK_CUDA(cudaGraphicsGLRegisterImage(&cudaTexResource, textureID, GL_TEXTURE_2D,
                                               cudaGraphicsRegisterFlagsWriteDiscard));
        if (!uploadStream) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking));
        }
        if (!inputReadyEvent) {
            CHECK_CUDA(cudaEventCreateWithFlags(&inputReadyEvent,
                                                cudaEventDisableTiming));
        }
#endif
    }

#ifdef ACMX2_WITH_CUDA
    /**
     * @brief Upload a CUDA GpuMat into the OpenGL texture without a host copy.
     *
     * Steps:
     *  1. Map the GL texture as a cudaArray (cudaGraphicsMapResources).
     *  2. cudaMemcpy2DToArrayAsync the GpuMat rows into that array on the
     *     uploader's dedicated stream — a single device-to-device DMA into
     *     the texture's own storage, no PBO intermediate.
     *  3. Unmap the resource so OpenGL can sample it.  cudaGraphicsUnmapResources
     *     inserts the GL↔CUDA dependency needed before the next draw.
     *
     * If the incoming frame dimensions differ from the current texture,
     * init() is called automatically to reallocate.
     *
     * @param gpuFrame The CUDA GpuMat (CV_8UC4 / RGBA) to upload.
     */
    void update(const cv::cuda::GpuMat &gpuFrame) {
        if (gpuFrame.cols != width || gpuFrame.rows != height) {
            init(gpuFrame.cols, gpuFrame.rows);
        }
        cudaArray_t texArray = nullptr;
        // OpenCV CUDA calls in this pipeline and launch_filter() produce their
        // output on the default stream. The uploader stream is nonblocking, so
        // establish the dependency explicitly before reading gpuFrame.
        CHECK_CUDA(cudaEventRecord(inputReadyEvent, nullptr));
        CHECK_CUDA(cudaStreamWaitEvent(uploadStream, inputReadyEvent, 0));
        CHECK_CUDA(cudaGraphicsMapResources(1, &cudaTexResource, uploadStream));
        CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&texArray, cudaTexResource, 0, 0));
        CHECK_CUDA(cudaMemcpy2DToArrayAsync(texArray, 0, 0,
                                            gpuFrame.data, gpuFrame.step,
                                            static_cast<size_t>(width) * 4,
                                            static_cast<size_t>(height),
                                            cudaMemcpyDeviceToDevice, uploadStream));
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &cudaTexResource, uploadStream));
    }
#endif

    /**
     * @brief Release all GPU resources (CUDA registration, stream, texture).
     *
     * Unregisters the texture from CUDA, destroys the upload stream, and
     * deletes the OpenGL texture.  Safe to call multiple times—each
     * resource handle is tested for non-zero before deletion and
     * reset to zero / nullptr afterwards.
     */
    void cleanup() {
#ifdef ACMX2_WITH_CUDA
        if (cudaTexResource) {
            CHECK_CUDA(cudaGraphicsUnregisterResource(cudaTexResource));
            cudaTexResource = nullptr;
        }
        if (uploadStream) {
            cudaStreamDestroy(uploadStream);
            uploadStream = nullptr;
        }
        if (inputReadyEvent) {
            cudaEventDestroy(inputReadyEvent);
            inputReadyEvent = nullptr;
        }
#endif
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
 * Stores the GL program binary for a fragment or compute shader, plus an
 * FNV-1a hash of the source file so stale entries are detected on load.
 * Fragment entries can contain both 2D and 3D variants; compute entries use
 * binary_2d for their standalone program and use a 3D passthrough slot.
 */
struct ShaderCacheEntry {
    std::string shader_name;     ///< Stem of the .glsl filename.
    std::vector<char> binary_2d; ///< GL program binary (2D vertex shader).
    GLenum format_2d = 0;        ///< GL binary format token for 2D.
    std::vector<char> binary_3d; ///< GL program binary (3D vertex shader).
    GLenum format_3d = 0;        ///< GL binary format token for 3D.
    uint64_t source_hash = 0;    ///< FNV-1a-64 hash of the fragment source.
    ShaderProgramKind kind = ShaderProgramKind::Fragment;
    bool failed = false; ///< True if this shader failed to compile;
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
    static constexpr uint32_t CACHE_MAGIC = 0x53484452; ///< File magic: "SHDR".
    static constexpr uint32_t CACHE_VERSION = 4;        ///< Current format version.
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
     * Each entry contains: shader_name (string), kind (uint8), failed (uint8),
     * source_hash (uint64),
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
            const uint8_t kind = static_cast<uint8_t>(e.kind);
            file.write(reinterpret_cast<const char *>(&kind), sizeof(kind));
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
            uint8_t kind = 0;
            file.read(reinterpret_cast<char *>(&kind), sizeof(kind));
            if (kind > static_cast<uint8_t>(ShaderProgramKind::ComputeUnavailable))
                return false;
            e.kind = static_cast<ShaderProgramKind>(kind);
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
 * thread-safe analyzer buffer. On the **render** thread, `update()` computes
 * a radix-2 FFT and uploads the resulting magnitude array into a
 * **GL_TEXTURE_1D** so that
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
 * @see acmx2::audio::AudioAnalyzer
 */
class SpectrumTexture {
  public:
    explicit SpectrumTexture(acmx2::audio::AudioAnalyzer &analyzer)
        : analyzer(analyzer) {}

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
        if (textureID != 0)
            return;
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
     * 1. AudioAnalyzer reads the latest PCM snapshot, applies a
     *    Hann window, runs the radix-2 FFT, and writes the magnitudes.
     * 2. `glTexSubImage1D()` copies those magnitudes into the existing
     *    texture **without** reallocating — much cheaper than `glTexImage1D`
     *    every frame.
     *
     * Call this once per frame from the render thread, before binding the
     * texture for shader use.
     */
    void update() {
        if (textureID == 0)
            return;
        analyzer.compute_spectrum();
        const auto &mags = analyzer.spectrum();
        glBindTexture(GL_TEXTURE_1D, textureID);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, bins, GL_RED, GL_FLOAT, mags.data());
        glBindTexture(GL_TEXTURE_1D, 0);
    }

    void update(float scale) {
        if (textureID == 0)
            return;
        analyzer.compute_spectrum();
        const auto &mags = analyzer.spectrum();
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
    acmx2::audio::AudioAnalyzer &analyzer;
    GLuint textureID = 0;          ///< OpenGL name for the 1D texture.
    int bins = 0;                  ///< Number of texels (== FFT_SIZE / 2).
    std::vector<float> scaled_buf; ///< Scratch buffer for sensitivity-scaled magnitudes.
};

/**
 * @class SpectrumHistory
 * @brief Runtime-sized ring buffer stored in one 1D array texture.
 *
 * Enabled via the `--enable-audio-buffers <N>` CLI option. Allocates one
 * `GL_TEXTURE_1D_ARRAY` with @p N `GL_R32F` layers and exposes it to GLSL as
 * `uniform sampler1DArray spectrum_history;`. The requested layer count is
 * clamped only to the active GPU's `GL_MAX_ARRAY_TEXTURE_LAYERS` limit.
 *
 * Each call to `update()` writes the current FFT magnitudes into the
 * layer at the ring head and advances the head. No texture data is copied
 * between layers. Shaders use `spectrum_history_head` and
 * `spectrum_history_size` to convert a logical age into a physical layer.
 *
 * The complete array uses only texture unit 10. Unit 9 remains the live
 * `spectrum`/`spectrum0` texture.
 */
class SpectrumHistory {
  public:
    explicit SpectrumHistory(const acmx2::audio::AudioAnalyzer &analyzer)
        : analyzer(analyzer) {}

    /// Texture unit assigned to the complete spectrum history array.
    static constexpr int TEXTURE_UNIT = 10;

    /**
     * @brief Allocate one array texture with @p requested_count history layers.
     *
     * @param requested_count Number of requested history frames.
     * @return The allocated layer count after applying the GPU limit.
     */
    int init(int requested_count) {
        cleanup();
        if (requested_count <= 0)
            return 0;

        GLint max_layers = 0;
        glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &max_layers);
        if (max_layers <= 0) {
            mx::system_err
                << "acmx2: GL_TEXTURE_1D_ARRAY is unavailable on this context\n";
            return 0;
        }
        layer_count = std::min(requested_count, static_cast<int>(max_layers));
        if (layer_count != requested_count) {
            mx::system_err << "acmx2: --enable-audio-buffers clamped to GPU "
                              "array-layer limit "
                           << layer_count << " (was " << requested_count << ")\n";
        }

        bins = FFT_SIZE / 2;
        std::vector<float> zeros(
            static_cast<size_t>(bins) * static_cast<size_t>(layer_count),
            0.0f);
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_1D_ARRAY, texture_id);
        glTexImage2D(GL_TEXTURE_1D_ARRAY, 0, GL_R32F, bins, layer_count, 0,
                     GL_RED, GL_FLOAT, zeros.data());
        glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S,
                        GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_1D_ARRAY, 0);
        write_idx = 0;
        return layer_count;
    }

    /**
     * @brief Push the current FFT magnitudes into the head of the ring buffer.
     *
     * Reuses magnitudes already computed by `SpectrumTexture::update()` for
     * the same frame (caller must invoke that first).  When @p scale is
     * non-zero, magnitudes are multiplied by it (matching the live
     * spectrum's sensitivity scaling).
     */
    void update(float scale = 0.0f) {
        if (texture_id == 0 || layer_count == 0)
            return;
        const auto &mags = analyzer.spectrum();
        const float *src = mags.data();
        if (scale != 0.0f) {
            scaled_buf.resize(mags.size());
            for (size_t i = 0; i < mags.size(); ++i)
                scaled_buf[i] = mags[i] * scale;
            src = scaled_buf.data();
        }
        glBindTexture(GL_TEXTURE_1D_ARRAY, texture_id);
        glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, write_idx, bins, 1, GL_RED,
                        GL_FLOAT, src);
        glBindTexture(GL_TEXTURE_1D_ARRAY, 0);
        write_idx = (write_idx + 1) % layer_count;
    }

    /// Bind the complete spectrum history array to its single texture unit.
    void bind() const {
        if (texture_id == 0)
            return;
        glActiveTexture(GL_TEXTURE0 + TEXTURE_UNIT);
        glBindTexture(GL_TEXTURE_1D_ARRAY, texture_id);
        glActiveTexture(GL_TEXTURE0);
    }

    /// Physical layer containing the newest spectrum frame.
    int newestLayer() const {
        return layer_count > 0 ? (write_idx - 1 + layer_count) % layer_count
                               : 0;
    }

    /// Number of history layers currently allocated.
    int count() const { return layer_count; }

    /// Delete the array texture and reset state.
    void cleanup() {
        if (texture_id != 0) {
            glDeleteTextures(1, &texture_id);
            texture_id = 0;
        }
        layer_count = 0;
        write_idx = 0;
    }

    ~SpectrumHistory() { cleanup(); }

  private:
    const acmx2::audio::AudioAnalyzer &analyzer;
    GLuint texture_id = 0;         ///< One `GL_TEXTURE_1D_ARRAY` object.
    int bins = 0;                  ///< Texels per layer (== FFT_SIZE / 2).
    int layer_count = 0;           ///< Runtime-selected array depth.
    int write_idx = 0;             ///< Next physical layer to overwrite.
    std::vector<float> scaled_buf; ///< Scratch for sensitivity-scaled magnitudes.
};
#endif // AUDIO_ENABLED

/**
 * @class ShaderLibrary
 * @brief Manages the complete collection of compiled GLSL shader programs.
 *
 * Responsibilities:
 * - Load a single shader or a full library from JSON or text manifests.
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
    bool normalized_time = false;
    double video_fps = 0.0;
#ifdef AUDIO_ENABLED
    const acmx2::audio::AudioAnalyzer *audio_analyzer = nullptr;
    int audio_buffer_count = 0;         ///< Runtime depth of the spectrum history array.
    int spectrum_history_head = 0;      ///< Physical layer containing the newest spectrum.
    float audio_warmup_envelope = 1.0f; ///< Startup ramp [0..1] used to soften initial audio-reactive intensity.
#endif
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
    std::vector<ShaderManifestData::CustomUniform> custom_uniforms;
    std::vector<std::regex> custom_uniform_declarations;
    std::vector<std::regex> custom_uniform_references;

    /**
     * @struct ProgramData
     * @brief Cached uniform locations for a single compiled shader program.
     *
     * Querying glGetUniformLocation every frame is expensive; this struct
     * stores all locations once at compile time.
     */
    struct ProgramData {
        std::string name;
        GLint compute_work_group_size[3] = {1, 1, 1};
        GLint loc = -1, iTime = -1, iMouse = -1, time_f = -1, iResolution = -1;
#ifdef AUDIO_ENABLED
        GLint amp = -1, amp_untouched = -1;
        GLint iamp = -1;
        GLint amp_peak = -1, amp_rms = -1, amp_smooth = -1;
        GLint amp_low = -1, amp_mid = -1, amp_high = -1;
        GLint spectrum_loc = -1;              ///< Live `sampler1D spectrum`.
        GLint spectrum_zero_loc = -1;         ///< Current-frame `spectrum0` alias.
        GLint spectrum_history_loc = -1;      ///< `sampler1DArray spectrum_history`.
        GLint spectrum_history_head_loc = -1; ///< Newest physical array layer.
        GLint spectrum_history_size_loc = -1; ///< Runtime array depth.
#endif
        GLint texture_cache_loc[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        // Array-form alias of texture_cache_loc: locations of `textures[0]..textures[N-1]`
        // for shaders that declare `uniform sampler2D textures[SIZE];`. Aliases the same
        // texture units as samp1..samp8 (for the first 8 entries) — no extra binds or
        // copies needed. Sized at runtime to match the active --texture-cache-size so
        // shaders can scale beyond the legacy 8-frame ceiling.
        std::vector<GLint> texture_array_loc;
        GLint texture_array_base_loc = -1; ///< Base location of `textures[0]` for bulk glUniform1iv assignment.
        GLint history_loc = -1;            ///< Location of optional `uniform sampler2DArray history;`.
        GLint history_head_loc = -1;       ///< Location of the array ring's oldest physical layer.
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
        std::vector<GLint> custom_uniform_locs;
#ifdef MIDI_ENABLED
        GLint slider_loc[4] = {-1, -1, -1, -1}; ///< Locations of optional `uniform float slider1..slider4;`
#endif
    };
    size_t library_index = 0;
    bool use_cache = false;

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
     * @param library_path Shader library directory containing a supported manifest.
     * @param cache_size   Active texture-cache size used to key binary-cache compatibility.
     * @param use_array    Whether texture history uses a sampler array; included in the cache key.
     * @return Absolute or relative path to the shader cache file.
     */
    static std::string shaderCacheFilePath(const std::string &assets_path,
                                           const std::string &library_path,
                                           int cache_size = 8,
                                           bool use_array = false) {
        std::error_code ec;
        std::filesystem::path lib(library_path);
        std::filesystem::path abs_lib = std::filesystem::absolute(lib, ec);
        std::string key = ec ? library_path : abs_lib.lexically_normal().string();
        // Append cache_size to the hash key so each --texture-cache-size value
        // gets its own cache file. Shader binaries compiled with `#define SIZE N`
        // for one N must not be reused for a different N — the sampler array
        // declaration would mismatch the linked program.
        key += "|s=" + std::to_string(cache_size);
        key += "|a=" + std::to_string(use_array ? 1 : 0);
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

    /**
     * @brief Active runtime size of the texture-cache ring buffer.
     *
     * Set via setCacheSize() before any program load; defaults to 8 to
     * preserve the legacy samp1..samp8 binding behaviour. Used to:
     *  - inject `#define SIZE N` into fragment sources at compile time
     *    so `uniform sampler2D textures[SIZE];` declarations resolve to
     *    the right array length;
     *  - size the per-program `texture_array_loc` lookup vector;
     *  - bound the runtime binding loop in the cache-render path.
     */
    int cache_size = 8;
    bool use_history_array = false;
    int history_head = 0;

    /**
     * @brief Read a fragment shader file and inject `#define SIZE N`
     *        immediately after its `#version` directive.
     *
     * GLSL sampler array sizes must be compile-time constants, so the
     * cache-frame count has to be baked into the source string before
     * `glShaderSource`. This helper preserves the original source line
     * numbering as best it can: the inserted line replaces no existing
     * line, only shifts everything after `#version` down by one.
     *
     * If the file cannot be opened the returned string is empty (the
     * caller falls through to the file-based loader, which surfaces the
     * proper error).
     *
     * @param frag_path  Filesystem path to the fragment .glsl source.
     * @param size       Cache ring length to bake as `SIZE`.
     * @param use_array  Whether to inject the texture-array compatibility define.
     * @return Modified fragment source, or empty string on read failure.
     */
    std::string injectShaderSize(const std::string &frag_path, int size,
                                 bool use_array) const {
        std::ifstream in(frag_path);
        if (!in.is_open())
            return {};
        std::ostringstream ss;
        ss << in.rdbuf();
        std::string src = ss.str();
        // Locate the first `#version` directive and insert after its newline.
        std::size_t v_pos = src.find("#version");
        std::size_t insert_pos = 0;
        if (v_pos != std::string::npos) {
            std::size_t nl = src.find('\n', v_pos);
            insert_pos = (nl == std::string::npos) ? src.size() : nl + 1;
        }
        std::string define =
            "#define SIZE " + std::to_string(size) + "\n" +
            "#define USE_HISTORY_TEXTURE_ARRAY " +
            std::to_string(use_array ? 1 : 0) + "\n";
        std::string sourceWithoutComments;
        sourceWithoutComments.reserve(src.size());
        bool lineComment = false;
        bool blockComment = false;
        for (std::size_t i = 0; i < src.size(); ++i) {
            if (lineComment) {
                if (src[i] == '\n') {
                    lineComment = false;
                    sourceWithoutComments.push_back('\n');
                } else {
                    sourceWithoutComments.push_back(' ');
                }
                continue;
            }
            if (blockComment) {
                if (src[i] == '*' && i + 1 < src.size() && src[i + 1] == '/') {
                    sourceWithoutComments.append("  ");
                    ++i;
                    blockComment = false;
                } else {
                    sourceWithoutComments.push_back(src[i] == '\n' ? '\n' : ' ');
                }
                continue;
            }
            if (src[i] == '/' && i + 1 < src.size() && src[i + 1] == '/') {
                sourceWithoutComments.append("  ");
                ++i;
                lineComment = true;
            } else if (src[i] == '/' && i + 1 < src.size() && src[i + 1] == '*') {
                sourceWithoutComments.append("  ");
                ++i;
                blockComment = true;
            } else {
                sourceWithoutComments.push_back(src[i]);
            }
        }
        for (std::size_t i = 0; i < custom_uniforms.size(); ++i) {
            const bool referenced =
                i < custom_uniform_references.size() &&
                std::regex_search(sourceWithoutComments,
                                  custom_uniform_references[i]);
            const bool declared =
                i < custom_uniform_declarations.size() &&
                std::regex_search(sourceWithoutComments,
                                  custom_uniform_declarations[i]);
            if (referenced && !declared) {
                define += "uniform float " + custom_uniforms[i].name + ";\n";
            }
        }
        src.insert(insert_pos, define);
        return src;
    }

    uint64_t preparedFragmentHash(const std::string &frag_path) const {
        const std::string source =
            injectShaderSize(frag_path, cache_size, use_history_array);
        if (source.empty())
            return 0;
        uint64_t hash = 1469598103934665603ull;
        for (unsigned char byte : source) {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    /**
     * @brief Compile a shader from file pair, injecting `#define SIZE N`
     *        into the fragment source. Falls back to the plain file-based
     *        loader if the fragment cannot be read in-process.
     *
     * The vertex source is not modified — SIZE is only meaningful for
     * fragment-side cache shaders.
     *
     * @return true if the program compiled and linked successfully.
     */
    bool loadProgramWithSize(gl::ShaderProgram *prog,
                             const std::string &vert_path,
                             const std::string &frag_path) const {
        std::string frag_src =
            injectShaderSize(frag_path, cache_size, use_history_array);
        if (frag_src.empty()) {
            // Could not read the fragment file; fall back so the loader can
            // produce its own diagnostic.
            return prog->loadProgram(vert_path, frag_path);
        }
        std::ifstream vin(vert_path);
        if (!vin.is_open()) {
            return prog->loadProgram(vert_path, frag_path);
        }
        std::ostringstream vss;
        vss << vin.rdbuf();
        return prog->loadProgramFromText(vss.str(), frag_src);
    }

    std::vector<std::unique_ptr<gl::ShaderProgram>> programs_2d;
    std::vector<std::unique_ptr<gl::ShaderProgram>> programs_3d;
    std::vector<ShaderProgramKind> program_kinds;
    bool time_audio = false;
    bool audio_delta = false;
    std::unordered_map<int, ProgramData> program_names_2d;
    std::unordered_map<int, ProgramData> program_names_3d;
    bool shader_bypass = false;
    bool isDraggingLeft = false;
    bool isDraggingRight = false;
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

    /** Compile and link a standalone OpenGL compute program. */
    std::unique_ptr<gl::ShaderProgram> makeComputeProgram(
        const std::string &compute_path, std::string &error) const {
        if (!compute_shader_supported) {
            error = "OpenGL 4.3 compute shaders are unavailable";
            return {};
        }

        const std::string source =
            injectShaderSize(compute_path, cache_size, use_history_array);
        if (source.empty()) {
            error = "Could not read compute shader: " + compute_path;
            return {};
        }

        if (use_cache) {
            GLuint cached_program = 0;
            if (ac::loadComputeProgramBinaryFromCache(source, cached_program)) {
                mx::system_out << "acmx2: Loaded cached compute shader: "
                               << compute_path << "\n";
                return std::make_unique<gl::ShaderProgram>(cached_program);
            }
        }

        const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        if (shader == 0) {
            error = "Could not create compute shader object";
            return {};
        }
        const char *source_ptr = source.c_str();
        glShaderSource(shader, 1, &source_ptr, nullptr);
        glCompileShader(shader);
        GLint compiled = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != GL_TRUE) {
            error = "Compute shader compilation failed: " + compute_path;
            const std::string log = shaderInfoLog(shader);
            if (!log.empty())
                error += "\n" + log;
            glDeleteShader(shader);
            return {};
        }

        const GLuint program = glCreateProgram();
        if (program == 0) {
            glDeleteShader(shader);
            error = "Could not create compute program";
            return {};
        }
        glAttachShader(program, shader);
#if !defined(__APPLE__)
        if (glProgramParameteri != nullptr) {
            glProgramParameteri(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT,
                                GL_TRUE);
        }
#endif
        glLinkProgram(program);
        GLint linked = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE) {
            error = "Compute program link failed: " + compute_path;
            const std::string log = programInfoLog(program);
            if (!log.empty())
                error += "\n" + log;
            glDeleteProgram(program);
            glDeleteShader(shader);
            return {};
        }

        glDetachShader(program, shader);
        glDeleteShader(shader);
        if (use_cache)
            ac::saveComputeProgramBinaryToCache(source, program);
        error.clear();
        return std::make_unique<gl::ShaderProgram>(program);
    }

    static std::string shaderInfoLog(GLuint shader) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        if (length <= 1)
            return {};
        std::vector<char> buffer(static_cast<std::size_t>(length), '\0');
        GLsizei written = 0;
        glGetShaderInfoLog(shader, length, &written, buffer.data());
        std::size_t used = static_cast<std::size_t>(std::max<GLsizei>(written, 0));
        while (used > 0 && buffer[used - 1] == '\0')
            --used;
        return std::string(buffer.data(), used);
    }

    static std::string programInfoLog(GLuint program) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        if (length <= 1)
            return {};
        std::vector<char> buffer(static_cast<std::size_t>(length), '\0');
        GLsizei written = 0;
        glGetProgramInfoLog(program, length, &written, buffer.data());
        std::size_t used = static_cast<std::size_t>(std::max<GLsizei>(written, 0));
        while (used > 0 && buffer[used - 1] == '\0')
            --used;
        return std::string(buffer.data(), used);
    }

    /** Compile a replacement program while retaining complete driver diagnostics. */
    std::unique_ptr<gl::ShaderProgram> compileProgramForReload(
        const std::string &vert_path,
        const std::string &frag_path,
        std::string &error) const {
        if (isComputeShaderFile(frag_path))
            return makeComputeProgram(frag_path, error);
        std::ifstream vertex_file(vert_path);
        if (!vertex_file.is_open()) {
            error = "Could not read vertex shader: " + vert_path;
            return {};
        }
        std::ostringstream vertex_stream;
        vertex_stream << vertex_file.rdbuf();
        const std::string vertex_source = vertex_stream.str();
        const std::string fragment_source =
            injectShaderSize(frag_path, cache_size, use_history_array);
        if (fragment_source.empty()) {
            error = "Could not read fragment shader: " + frag_path;
            return {};
        }

        const auto compile_stage = [&error](GLenum type,
                                            const std::string &source,
                                            const std::string &label) -> GLuint {
            const GLuint shader = glCreateShader(type);
            if (shader == 0) {
                error = "Could not create " + label + " shader object";
                return 0;
            }
            const char *source_ptr = source.c_str();
            glShaderSource(shader, 1, &source_ptr, nullptr);
            glCompileShader(shader);
            GLint compiled = GL_FALSE;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
            if (compiled != GL_TRUE) {
                error = label + " shader compilation failed";
                const std::string log = shaderInfoLog(shader);
                if (!log.empty())
                    error += ":\n" + log;
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        };

        const GLuint vertex_shader =
            compile_stage(GL_VERTEX_SHADER, vertex_source, vert_path);
        if (vertex_shader == 0)
            return {};
        const GLuint fragment_shader =
            compile_stage(GL_FRAGMENT_SHADER, fragment_source, frag_path);
        if (fragment_shader == 0) {
            glDeleteShader(vertex_shader);
            return {};
        }

        const GLuint program = glCreateProgram();
        if (program == 0) {
            glDeleteShader(vertex_shader);
            glDeleteShader(fragment_shader);
            error = "Could not create shader program";
            return {};
        }
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);
        GLint linked = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE) {
            error = "Shader program link failed";
            const std::string log = programInfoLog(program);
            if (!log.empty())
                error += ":\n" + log;
            glDeleteProgram(program);
            glDeleteShader(vertex_shader);
            glDeleteShader(fragment_shader);
            return {};
        }

        glDetachShader(program, vertex_shader);
        glDetachShader(program, fragment_shader);
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        error.clear();
        return std::make_unique<gl::ShaderProgram>(program);
    }

    /**
     * @brief Compile a minimal passthrough fragment shader as a stand-in.
     *
     * Used when a shader in the library fails to compile (either during
     * cache-build or source-compile).  A placeholder program is inserted
     * at the failing shader's index so that numeric indices remain
     * aligned with the on-disk shader manifest — this keeps
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
#ifdef AUDIO_ENABLED
    explicit ShaderLibrary(const acmx2::audio::AudioAnalyzer *analyzer = nullptr)
        : audio_analyzer(analyzer) {}
#else
    ShaderLibrary() = default;
#endif

    void setCustomUniformValues(
        const std::vector<ShaderManifestData::CustomUniform> &uniforms) {
        const bool namesChanged = custom_uniforms.size() != uniforms.size() ||
                                  !std::equal(
                                      custom_uniforms.begin(), custom_uniforms.end(),
                                      uniforms.begin(),
                                      [](const auto &left, const auto &right) {
                                          return left.name == right.name;
                                      });
        custom_uniforms = uniforms;
        if (!namesChanged)
            return;
        custom_uniform_declarations.clear();
        custom_uniform_references.clear();
        custom_uniform_declarations.reserve(custom_uniforms.size());
        custom_uniform_references.reserve(custom_uniforms.size());
        for (const auto &uniform : custom_uniforms) {
            custom_uniform_declarations.emplace_back(
                "\\buniform\\s+(?:(?:lowp|mediump|highp)\\s+)?float\\s+" +
                uniform.name + "\\b");
            custom_uniform_references.emplace_back("\\b" + uniform.name + "\\b");
        }
        const auto refreshLocations = [this](
                                          auto &names,
                                          const auto &programs) {
            for (auto &[index, data] : names) {
                data.custom_uniform_locs.assign(custom_uniforms.size(), -1);
                if (index < 0 || static_cast<std::size_t>(index) >= programs.size())
                    continue;
                for (std::size_t i = 0; i < custom_uniforms.size(); ++i) {
                    data.custom_uniform_locs[i] = glGetUniformLocation(
                        programs[static_cast<std::size_t>(index)]->id(),
                        custom_uniforms[i].name.c_str());
                }
            }
        };
        refreshLocations(program_names_2d, programs_2d);
        refreshLocations(program_names_3d, programs_3d);
    }
    ~ShaderLibrary() {}

#ifdef MIDI_ENABLED
    /// Set a MIDI slider value (index 0–3, value 0.0–1.0).
    void setMidiSlider(int idx, float val) {
        if (idx >= 0 && idx < 4)
            midi_slider[idx] = val;
    }
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
     * @brief Set the active texture-cache ring length.
     *
     * Must be called before loadPrograms / loadProgramsWithCache so the
     * value is in effect when shaders are compiled (it is baked into the
     * fragment source as `#define SIZE N`) and when uniform locations are
     * queried (it sizes the per-program `texture_array_loc` lookup).
     *
     * Values outside [1, 64] are clamped silently to keep us under any
     * reasonable `GL_MAX_TEXTURE_IMAGE_UNITS` minimum (driver minimum for
     * GL 3.3 core is 16; the runtime caller already clamps to 1..64).
     *
     * @param size Ring buffer size in frames.
     */
    void setCacheSize(int size) {
        if (size < 1)
            size = 1;
        if (size > 64)
            size = 64;
        cache_size = size;
    }

    /// @brief Active cache ring size as seen by the shader-compile path.
    int cacheSize() const { return cache_size; }

    /**
     * @brief Select the cache sampler representation before compiling shaders.
     *
     * Array mode binds a single `sampler2DArray history` at texture unit 1.
     * Legacy mode binds `samp1..samp8` and `textures[SIZE]` to units 1..SIZE.
     */
    void setHistoryTextureArray(bool enabled) {
        use_history_array = enabled;
    }

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
        program_kinds.clear();
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
        const bool is_compute = isComputeShaderFile(text);
        if (is_compute && compute_shader_supported) {
            std::string error;
            auto compute_program = makeComputeProgram(text, error);
            if (!compute_program)
                throw mx::Exception(error);
            programs_2d.push_back(std::move(compute_program));
            program_kinds.push_back(ShaderProgramKind::Compute);
        } else {
            programs_2d.push_back(makeProgram());
            if (!is_compute &&
                !loadProgramWithSize(programs_2d.back().get(),
                                     win->util.getFilePath("data/vert.glsl"), text)) {
                throw mx::Exception("Error loading 2D shader program: " + text);
            }
            if (is_compute) {
                programs_2d.pop_back();
                auto passthrough = makePassthroughProgram(
                    win->util.getFilePath("data/vert.glsl"));
                if (!passthrough)
                    throw mx::Exception("Could not create compute fallback: " + text);
                programs_2d.push_back(std::move(passthrough));
                program_kinds.push_back(ShaderProgramKind::ComputeUnavailable);
                mx::system_out << "acmx2: Skipping compute shader on this context: "
                               << text << "\n";
            } else {
                program_kinds.push_back(ShaderProgramKind::Fragment);
            }
        }
        setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, text);
        if (dual_mode) {
            if (is_compute) {
                auto passthrough = makePassthroughProgram(
                    win->util.getFilePath("data/vertex.glsl"));
                if (!passthrough)
                    throw mx::Exception("Could not create 3D compute passthrough: " + text);
                programs_3d.push_back(std::move(passthrough));
            } else {
                programs_3d.push_back(makeProgram());
                if (!loadProgramWithSize(programs_3d.back().get(),
                                         win->util.getFilePath("data/vertex.glsl"),
                                         text)) {
                    throw mx::Exception("Error loading 3D shader program: " + text);
                }
            }
            setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, text);
            mx::system_out << "acmx2: Compiled Shader 0 (2D+3D): " << text << " ✔ \n";
        } else {
            mx::system_out << "acmx2: Compiled Shader 0 (2D): " << text << " ✔ \n";
        }
    }

    /**
     * @brief Recompile one library slot and atomically replace it on success.
     *
     * Both 2D and 3D variants are compiled into temporary programs first.
     * The currently running programs remain installed if compilation, linking,
     * or uniform setup fails.
     */
    bool reloadProgram(gl::GLWindow *win, size_t shader_index,
                       const std::string &fragment_path, std::string &error) {
        if (shader_index >= programs_2d.size()) {
            error = "Shader reload index is outside the loaded library: " +
                    std::to_string(shader_index);
            return false;
        }
        if (dual_mode && shader_index >= programs_3d.size()) {
            error = "Shader reload index is outside the loaded 3D library: " +
                    std::to_string(shader_index);
            return false;
        }

        const bool is_compute = isComputeShaderFile(fragment_path);
        auto replacement_2d = compileProgramForReload(
            win->util.getFilePath("data/vert.glsl"), fragment_path, error);
        if (!replacement_2d)
            return false;

        std::unique_ptr<gl::ShaderProgram> replacement_3d;
        if (dual_mode) {
            if (is_compute) {
                replacement_3d = makePassthroughProgram(
                    win->util.getFilePath("data/vertex.glsl"));
                if (!replacement_3d)
                    error = "Could not create 3D passthrough for compute shader";
            } else {
                replacement_3d = compileProgramForReload(
                    win->util.getFilePath("data/vertex.glsl"), fragment_path, error);
            }
            if (!replacement_3d)
                return false;
        }

        std::unordered_map<int, ProgramData> replacement_names_2d;
        std::unordered_map<int, ProgramData> replacement_names_3d;
        try {
            setupProgramUniforms(win, replacement_2d.get(), replacement_names_2d,
                                 shader_index, fragment_path);
            if (dual_mode) {
                setupProgramUniforms(win, replacement_3d.get(), replacement_names_3d,
                                     shader_index, fragment_path);
            }
        } catch (const std::exception &e) {
            error = std::string("Shader uniform setup failed: ") + e.what();
            shader()->useProgram();
            return false;
        } catch (...) {
            error = "Shader uniform setup failed with an unknown error";
            shader()->useProgram();
            return false;
        }

        const int index_key = static_cast<int>(shader_index);
        const auto names_2d_it = replacement_names_2d.find(index_key);
        const auto current_names_2d_it = program_names_2d.find(index_key);
        if (names_2d_it == replacement_names_2d.end()) {
            error = "Shader reload did not produce 2D uniform metadata";
            shader()->useProgram();
            return false;
        }
        if (current_names_2d_it == program_names_2d.end()) {
            error = "Loaded shader is missing 2D uniform metadata";
            shader()->useProgram();
            return false;
        }

        auto names_3d_it = replacement_names_3d.end();
        auto current_names_3d_it = program_names_3d.end();
        if (dual_mode) {
            names_3d_it = replacement_names_3d.find(index_key);
            current_names_3d_it = program_names_3d.find(index_key);
            if (names_3d_it == replacement_names_3d.end() ||
                current_names_3d_it == program_names_3d.end()) {
                error = "Shader reload is missing 3D uniform metadata";
                shader()->useProgram();
                return false;
            }
        }

        programs_2d[shader_index].swap(replacement_2d);
        if (shader_index < program_kinds.size()) {
            program_kinds[shader_index] =
                is_compute ? ShaderProgramKind::Compute
                           : ShaderProgramKind::Fragment;
        }
        std::swap(current_names_2d_it->second, names_2d_it->second);
        if (dual_mode) {
            programs_3d[shader_index].swap(replacement_3d);
            std::swap(current_names_3d_it->second, names_3d_it->second);
        }
        shader()->useProgram();
        error.clear();
        return true;
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
            if (&names == &program_names_2d && pos < program_kinds.size() &&
                program_kinds[pos] == ShaderProgramKind::Compute) {
                glGetProgramiv(prog->id(), GL_COMPUTE_WORK_GROUP_SIZE,
                               names[pos].compute_work_group_size);
            }
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
                names[pos].history_loc =
                    glGetUniformLocation(prog->id(), "history");
                names[pos].history_head_loc =
                    glGetUniformLocation(prog->id(), "history_head");
                if (use_history_array && names[pos].history_loc != -1) {
                    glUniform1i(names[pos].history_loc, 1);
                }
                if (use_history_array && names[pos].history_head_loc != -1) {
                    glUniform1i(names[pos].history_head_loc, history_head);
                }
                // samp1..samp8 are the legacy slot-named samplers (capped
                // at 8 because the engine only ever declared that many).
                for (int i = 0; i < 8; ++i) {
                    names[pos].texture_cache_loc[i] = glGetUniformLocation(prog->id(), std::string("samp" + std::to_string(i + 1)).c_str());
                }
                // Prefer assigning the whole sampler array through its base
                // location; some drivers do not reliably expose locations for
                // every `textures[i]` element queried individually.
                names[pos].texture_array_base_loc = glGetUniformLocation(prog->id(), "textures[0]");
                if (!use_history_array &&
                    names[pos].texture_array_base_loc != -1) {
                    std::vector<GLint> units(static_cast<std::size_t>(cache_size), 0);
                    for (int i = 0; i < cache_size; ++i) {
                        units[static_cast<std::size_t>(i)] = i + 1;
                    }
                    glUniform1iv(names[pos].texture_array_base_loc, cache_size, units.data());
                }
                // textures[0..N-1] is the array-form alias and scales with
                // the runtime --texture-cache-size. glGetUniformLocation
                // returns -1 for any element the shader doesn't actually
                // declare/reference; glUniform1i on -1 is a silent no-op.
                names[pos].texture_array_loc.assign(static_cast<std::size_t>(cache_size), -1);
                for (int i = 0; i < cache_size; ++i) {
                    names[pos].texture_array_loc[i] = glGetUniformLocation(
                        prog->id(),
                        std::string("textures[" + std::to_string(i) + "]").c_str());
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
            names[pos].spectrum_zero_loc =
                glGetUniformLocation(prog->id(), "spectrum0");
            names[pos].spectrum_history_loc =
                glGetUniformLocation(prog->id(), "spectrum_history");
            names[pos].spectrum_history_head_loc =
                glGetUniformLocation(prog->id(), "spectrum_history_head");
            names[pos].spectrum_history_size_loc =
                glGetUniformLocation(prog->id(), "spectrum_history_size");
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
            names[pos].custom_uniform_locs.assign(custom_uniforms.size(), -1);
            for (std::size_t i = 0; i < custom_uniforms.size(); ++i) {
                names[pos].custom_uniform_locs[i] = glGetUniformLocation(
                    prog->id(), custom_uniforms[i].name.c_str());
            }
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
        if (value < 0 || value >= cache_size) {
            return;
        }
        auto &names = is3d ? program_names_3d : program_names_2d;
        if (names.find(index()) == names.end()) {
            return;
        }
        if (use_history_array) {
            if (value == 0 && names[index()].history_loc != -1) {
                glUniform1i(names[index()].history_loc, 1);
            }
            return;
        }
        // samp1..samp8 only exist for the first 8 slots (legacy ceiling).
        if (value < 8) {
            glUniform1i(names[index()].texture_cache_loc[value], value + 1);
        }
        // If a base location exists, the whole array was assigned in
        // setupProgramUniforms() via glUniform1iv, so no per-element update
        // is needed here.
        if (names[index()].texture_array_base_loc != -1) {
            return;
        }
        // Array-form `textures[value]` scales to whatever cache_size is
        // active. glUniform1i on -1 (uniform absent or optimized out) is
        // a no-op, so the bounds check above is the only guard needed.
        if (value < static_cast<int>(names[index()].texture_array_loc.size())) {
            glUniform1i(names[index()].texture_array_loc[value], value + 1);
        }
    }

    /**
     * @brief Set the physical array layer that represents logical history zero.
     *
     * The array texture is updated as a ring, so this offset advances whenever
     * the oldest frame is overwritten. Shaders map a logical index with
     * `(history_head + index) % SIZE`.
     */
    void setHistoryHead(int layer) {
        history_head = layer;
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

    /// @brief Select fixed-per-frame or elapsed real-time advancement.
    void setNormalizedTime(bool enabled) {
        normalized_time = enabled;
    }

#ifdef AUDIO_ENABLED
    /**
     * @brief Configure the runtime depth of `spectrum_history`.
     */
    void setAudioBufferCount(int n) {
        audio_buffer_count = std::max(n, 0);
        if (audio_buffer_count == 0)
            spectrum_history_head = 0;
    }

    /// @brief Number of spectrum history array layers currently configured.
    int audioBufferCount() const { return audio_buffer_count; }

    /// @brief Set the physical array layer containing the newest FFT frame.
    void setSpectrumHistoryHead(int layer) {
        spectrum_history_head =
            audio_buffer_count > 0 ? layer % audio_buffer_count : 0;
    }

    /// @brief Set startup audio warmup envelope in [0,1] for uniform scaling.
    void setAudioWarmupEnvelope(float env) {
        audio_warmup_envelope = std::clamp(env, 0.0f, 1.0f);
    }
#endif

    /// @brief Set the frame rate used by normalized time advancement.
    void setVideoFPS(double fps) {
        if (std::isfinite(fps) && fps > 0.0)
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
        time_speed -= step;

        // Optional: Add a 'deadzone' to make it easier to stop the animation
        if (std::abs(time_speed) < 0.01f) {
            time_speed = 0.0f;
        }

        mx::system_out << "acmx2: Time speed (decelerating): " << time_speed << "\n";
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

    /// @brief Compile every shader listed in the preferred manifest.
    void loadPrograms(gl::GLWindow *win, const std::string &text, mx::Font &loadingFont) {
        ShaderManifestData manifest;
        std::string manifest_error;
        if (!loadShaderManifest(text, manifest, manifest_error))
            throw mx::Exception("acmx2: " + manifest_error);
        setCustomUniformValues(manifest.custom_uniforms);

        std::vector<std::string> shader_files;
        for (const std::string &entry : manifest.entries) {
            auto shader_entry = normalizeShaderIndexEntry(entry);
            if (!shader_entry)
                continue;
            std::string full_path;
            if (resolveShaderPathInLibrary(text, *shader_entry, full_path)) {
                shader_files.push_back(*shader_entry);
            }
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

        const char *load_action = use_cache ? "Loading" : "Compiling";
        mx::system_out << "acmx2: " << load_action << " " << total_shaders
                       << " shaders (" << (dual_mode ? "2D+3D" : "2D")
                       << ")...\n";
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
                        float scale = std::min(static_cast<float>(win->w) / lw, static_cast<float>(win->h) / lh);
                        int dw = static_cast<int>(lw * scale);
                        int dh = static_cast<int>(lh * scale);
                        int lx = (win->w - dw) / 2;
                        int ly = (win->h - dh) / 2;
                        logo_sprite->initWithTexture(&logo_shader, logo_tex, lx, ly, dw, dh);
                        logo_tex = 0;
                        logo_loaded = true;
                    }
                } catch (...) {
                }
                if (logo_tex) {
                    glDeleteTextures(1, &logo_tex);
                }
            }
        }

        int last_percent_reported = -1;
        for (size_t shader_index = 0; shader_index < shader_files.size(); ++shader_index) {
            const std::string &line_data = shader_files[shader_index];
            std::string full_path = text + "/" + line_data;
            std::string vert_2d = win->util.getFilePath("data/vert.glsl");
            std::string vert_3d = win->util.getFilePath("data/vertex.glsl");

            const bool is_compute = isComputeShaderFile(line_data);
            bool ok_2d = false;
            if (is_compute && compute_shader_supported) {
                std::string compute_error;
                auto compute_program = makeComputeProgram(full_path, compute_error);
                if (compute_program) {
                    programs_2d.push_back(std::move(compute_program));
                    ok_2d = true;
                } else {
                    mx::system_out << "acmx2: ⚠ " << compute_error
                                   << " — substituting passthrough placeholder\n";
                }
            } else if (!is_compute) {
                programs_2d.push_back(makeProgram());
                try {
                    ok_2d = loadProgramWithSize(programs_2d.back().get(),
                                                vert_2d, full_path);
                } catch (const std::exception &e) {
                    mx::system_out << "acmx2: ⚠ Exception compiling 2D shader: "
                                   << line_data << " (" << e.what() << ")\n";
                } catch (...) {
                    mx::system_out << "acmx2: ⚠ Unknown exception compiling 2D shader: "
                                   << line_data << "\n";
                }
            }
            if (!ok_2d) {
                if (!programs_2d.empty() && programs_2d.size() > shader_index)
                    programs_2d.pop_back();
                auto ph = makePassthroughProgram(vert_2d);
                if (!ph)
                    throw mx::Exception("acmx2: Error could not build 2D passthrough placeholder for: " + line_data);
                programs_2d.push_back(std::move(ph));
                if (is_compute && !compute_shader_supported) {
                    mx::system_out << "acmx2: Skipping unsupported compute shader: "
                                   << line_data << "\n";
                }
            }
            program_kinds.push_back(
                is_compute ? (ok_2d ? ShaderProgramKind::Compute
                                    : ShaderProgramKind::ComputeUnavailable)
                           : ShaderProgramKind::Fragment);
            setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, full_path);
            if (dual_mode) {
                if (is_compute) {
                    auto ph = makePassthroughProgram(vert_3d);
                    if (!ph)
                        throw mx::Exception("acmx2: Error could not build 3D compute passthrough for: " + line_data);
                    programs_3d.push_back(std::move(ph));
                } else {
                    bool ok_3d = false;
                    programs_3d.push_back(makeProgram());
                    try {
                        ok_3d = loadProgramWithSize(programs_3d.back().get(),
                                                    vert_3d, full_path);
                    } catch (...) {
                    }
                    if (!ok_3d) {
                        programs_3d.pop_back();
                        auto ph = makePassthroughProgram(vert_3d);
                        if (!ph)
                            throw mx::Exception("acmx2: Error could not build 3D passthrough placeholder for: " + line_data);
                        programs_3d.push_back(std::move(ph));
                    }
                }
                setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, full_path);
            }

            int percent = static_cast<int>((shader_index + 1) * 100 / total_shaders);
            int percent_bucket = (percent / 10) * 10;
            if (percent_bucket > last_percent_reported) {
                last_percent_reported = percent_bucket;
                mx::system_out << "acmx2: " << load_action << "... "
                               << percent_bucket << "% (" << (shader_index + 1)
                               << "/" << total_shaders << " shaders)\n";
                fflush(stdout);

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                if (logo_loaded) {
                    logo_sprite->draw();
                }
                if (loadingFont.handle().has_value()) {
                    std::string loadingText = std::string(load_action) + " Shader " +
                                              std::to_string(shader_index + 1) + "/" +
                                              std::to_string(total_shaders) + "...";
                    win->text.printText_Blended(loadingFont, 10, 10, loadingText);
                }
                SDL_GL_SwapWindow(win->getWindow());
                SDL_PumpEvents();
            }
        }
        mx::system_out << "acmx2: " << (use_cache ? "Loaded " : "Compiled ")
                       << shader_files.size() << " shaders ("
                       << (dual_mode ? "2D+3D" : "2D only") << ")\n";
        fflush(stdout);
    }

    /**
     * @brief Compile every shader in a library and rewrite its manifest without
     *        the ones that fail to compile.
     *
     * Reads `library.json` when present, otherwise `index.txt`, and compiles each
     * fragment shader against the supplied 2D (and optionally 3D) vertex
     * shader. The selected manifest is rewritten without broken entries and
     * preserved as a `.bak` file before replacement.
     *
     * Non-shader lines (blank lines and lines containing "material") are
     * preserved verbatim. Files listed in the manifest that do not exist on
     * disk are also dropped.
     *
     * The existing `.shader_cache` is deleted so the library is rebuilt
     * fresh on next launch.
     *
     * @param win           GL window (for asset resolution).
     * @param library_path  Directory containing a shader manifest and GLSL files.
     * @param vert_2d       Path to the 2D vertex shader.
     * @param vert_3d       Path to the 3D vertex shader (only used when @ref dual_mode is set).
     * @return true if the manifest was rewritten successfully.
     */
    bool removeBrokenShaders(gl::GLWindow *win,
                             const std::string &library_path,
                             const std::string &vert_2d,
                             const std::string &vert_3d) {
        static_cast<void>(win);
        if (glGetString(GL_VERSION) == nullptr) {
            mx::system_err << "acmx2: remove-broken requires a valid OpenGL context\n";
            return false;
        }

        ShaderManifestData manifest;
        std::string manifest_error;
        if (!loadShaderManifest(library_path, manifest, manifest_error)) {
            mx::system_err << "acmx2: " << manifest_error << "\n";
            return false;
        }
        setCustomUniformValues(manifest.custom_uniforms);
        const std::string manifest_path = manifest.path;
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

        // Preserve ordering and, for legacy manifests, non-shader lines.
        struct Line {
            std::string raw;  ///< Original line text.
            bool is_shader;   ///< True if this line references a fragment shader file.
            bool keep = true; ///< False if the shader failed to compile.
        };
        std::vector<Line> lines;
        for (const std::string &manifest_entry : manifest.entries) {
            Line entry;
            entry.raw = manifest_entry;
            auto shader_entry = normalizeShaderIndexEntry(manifest_entry);
            std::string full_path;
            bool is_shader_line =
                shader_entry.has_value() &&
                resolveShaderPathInLibrary(library_path, *shader_entry, full_path);
            if (shader_entry)
                entry.raw = *shader_entry;
            entry.is_shader = is_shader_line;
            if (!is_shader_line && !manifest_entry.empty() &&
                manifest_entry.find("material") == std::string::npos) {
                mx::system_out << "acmx2: ⚠ Removing missing file from manifest: "
                               << manifest_entry << "\n";
                entry.is_shader = false;
                entry.keep = false;
            }
            lines.push_back(std::move(entry));
        }

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
            if (e.is_shader)
                ++total_shaders;
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
            if (!entry.is_shader || !entry.keep)
                continue;
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
                if (isComputeShaderFile(entry.raw)) {
                    if (compute_shader_supported) {
                        std::string compute_error;
                        compiled = static_cast<bool>(
                            makeComputeProgram(full_path, compute_error));
                        if (!compiled && !compute_error.empty())
                            mx::system_out << compute_error << " ";
                    } else {
                        mx::system_out << "compute unavailable; kept without validation ";
                    }
                } else {
                    gl::ShaderProgram prog_2d;
                    prog_2d.setSilent(true);
                    if (!loadProgramWithSize(&prog_2d, vert_2d, full_path)) {
                        compiled = false;
                    } else {
                        GLint link_status = 0;
                        glGetProgramiv(prog_2d.id(), GL_LINK_STATUS, &link_status);
                        if (link_status != GL_TRUE)
                            compiled = false;
                    }
                    if (compiled && dual_mode) {
                        gl::ShaderProgram prog_3d;
                        prog_3d.setSilent(true);
                        if (!loadProgramWithSize(&prog_3d, vert_3d, full_path)) {
                            compiled = false;
                        } else {
                            GLint link_status = 0;
                            glGetProgramiv(prog_3d.id(), GL_LINK_STATUS, &link_status);
                            if (link_status != GL_TRUE)
                                compiled = false;
                        }
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

        // Back up the selected manifest before rewriting.
        std::error_code ec;
        std::filesystem::copy_file(
            manifest_path,
            manifest_path + ".bak",
            std::filesystem::copy_options::overwrite_existing,
            ec);
        if (ec) {
            mx::system_out << "acmx2: Warning: could not create manifest backup ("
                           << ec.message() << ")\n";
        }

        std::vector<std::string> kept_entries;
        for (const auto &entry : lines) {
            if (entry.keep)
                kept_entries.push_back(entry.raw);
        }
        if (manifest.format == ShaderManifestFormat::Json) {
            if (!writeJsonShaderManifest(manifest_path, kept_entries,
                                         manifest.custom_uniforms,
                                         manifest_error)) {
                mx::system_err << "acmx2: " << manifest_error << "\n";
                return false;
            }
        } else {
            std::ofstream out(manifest_path, std::ios::trunc);
            if (!out.is_open()) {
                mx::system_err << "acmx2: Could not rewrite manifest at: "
                               << manifest_path << "\n";
                return false;
            }
            for (const std::string &entry : kept_entries)
                out << entry << "\n";
            out.close();
        }

        // Invalidate the on-disk cache since the library composition changed.
        std::string cache_file =
            shaderCacheFilePath(win ? win->util.path : std::string(),
                                library_path, cache_size, use_history_array);
        if (std::filesystem::exists(cache_file)) {
            std::filesystem::remove(cache_file, ec);
            if (!ec) {
                mx::system_out << "acmx2: Removed stale shader cache: " << cache_file << "\n";
            }
        }

        mx::system_out << "acmx2: Remove-broken complete: kept " << kept
                       << ", removed " << removed << " shader(s). "
                       << "Backup written to " << manifest_path << ".bak\n";
        fflush(stdout);
        return true;
    }

    /**
     * @brief Compile one shader and replace a single binary-cache entry.
     *
     * A failed compile is recorded in the entry so its manifest slot remains
     * stable and can use the normal passthrough fallback.
     */
    bool compileShaderCacheEntry(const std::string &shader_file,
                                 const std::string &full_path,
                                 const std::string &vert_2d,
                                 const std::string &vert_3d,
                                 bool include_3d,
                                 ShaderCacheEntry &entry) {
        entry = {};
        entry.shader_name = std::filesystem::path(shader_file).stem().string();
        entry.source_hash = preparedFragmentHash(full_path);
        entry.kind = isComputeShaderFile(shader_file)
                         ? ShaderProgramKind::Compute
                         : ShaderProgramKind::Fragment;

        const auto markFailed = [&](const std::string &reason) {
            entry.failed = true;
            entry.binary_2d.clear();
            entry.binary_3d.clear();
            entry.format_2d = 0;
            entry.format_3d = 0;
            mx::system_err << "acmx2: Incremental cache compile failed for "
                           << shader_file << ": " << reason << "\n";
            mx::system_err.flush();
        };
        const auto extractBinary = [](GLuint program, std::vector<char> &binary,
                                      GLenum &format) {
            GLint binaryLength = 0;
            glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &binaryLength);
            if (binaryLength <= 0)
                return false;
            binary.resize(static_cast<std::size_t>(binaryLength));
            GLsizei actualLength = 0;
            format = 0;
            while (glGetError() != GL_NO_ERROR) {
            }
            glGetProgramBinaryFunc(program, binaryLength, &actualLength,
                                   &format, binary.data());
            if (glGetError() != GL_NO_ERROR || actualLength <= 0) {
                binary.clear();
                format = 0;
                return false;
            }
            binary.resize(static_cast<std::size_t>(actualLength));
            return true;
        };

        try {
            if (entry.kind == ShaderProgramKind::Compute) {
                if (!compute_shader_supported) {
                    entry.kind = ShaderProgramKind::ComputeUnavailable;
                    markFailed("OpenGL 4.3 compute shaders are unavailable");
                    return false;
                }

                std::string compute_error;
                auto compute_program = makeComputeProgram(full_path, compute_error);
                if (!compute_program) {
                    markFailed(compute_error);
                    return false;
                }
                if (!extractBinary(compute_program->id(), entry.binary_2d,
                                   entry.format_2d)) {
                    markFailed("could not extract compute program binary");
                    return false;
                }
                entry.failed = false;
                return true;
            }

            gl::ShaderProgram program2d;
            program2d.setSilent(true);
            if (!loadProgramWithSize(&program2d, vert_2d, full_path)) {
                markFailed("2D compile failed");
                return false;
            }
            if (!extractBinary(program2d.id(), entry.binary_2d,
                               entry.format_2d)) {
                markFailed("could not extract 2D program binary");
                return false;
            }

            if (include_3d) {
                gl::ShaderProgram program3d;
                program3d.setSilent(true);
                if (!loadProgramWithSize(&program3d, vert_3d, full_path)) {
                    markFailed("3D compile failed");
                    return false;
                }
                if (!extractBinary(program3d.id(), entry.binary_3d,
                                   entry.format_3d)) {
                    markFailed("could not extract 3D program binary");
                    return false;
                }
            }
        } catch (const std::exception &error) {
            markFailed(error.what());
            return false;
        } catch (...) {
            markFailed("unknown exception");
            return false;
        }

        entry.failed = false;
        return true;
    }

    /**
     * @brief Build the on-disk shader binary cache for all shaders in a library.
     * @param win    GL window (provides vertex shader paths).
     * @param library_path  Directory containing a shader manifest and GLSL files.
     * @param vert_2d  Path to the 2D vertex shader.
     * @param vert_3d  Path to the 3D vertex shader.
     * @param loadingFont Optional font used to render rebuild progress.
     * @return true on success.
     */
    bool buildShaderCache(gl::GLWindow *win, const std::string &library_path,
                          const std::string &vert_2d,
                          const std::string &vert_3d,
                          mx::Font *loadingFont = nullptr) {
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

        std::string cache_file =
            shaderCacheFilePath(win ? win->util.path : std::string(),
                                library_path, cache_size, use_history_array);
        ShaderManifestData manifest;
        std::string manifest_error;
        if (!loadShaderManifest(library_path, manifest, manifest_error)) {
            mx::system_err << "acmx2: " << manifest_error << "\n";
            return false;
        }
        setCustomUniformValues(manifest.custom_uniforms);
        std::vector<std::string> shader_files;
        if (!collectShaderLibraryEntries(library_path, shader_files, manifest_error)) {
            mx::system_err << "acmx2: " << manifest_error << "\n";
            return false;
        }

        ShaderCache cache;
        cache.gl_renderer = safeGLString(GL_RENDERER);
        cache.gl_version = safeGLString(GL_VERSION);
        cache.dual_mode = dual_mode;

        mx::system_out << "acmx2: Building shader cache for " << shader_files.size() << " shaders...\n";
        fflush(stdout);

        static constexpr const char *kCacheLogoVert =
            "#version 330 core\n"
            "layout(location = 0) in vec3 aPos;\n"
            "layout(location = 1) in vec2 aTex;\n"
            "out vec2 tc;\n"
            "void main() { gl_Position = vec4(aPos, 1.0); tc = aTex; }\n";
        static constexpr const char *kCacheLogoFrag =
            "#version 330 core\n"
            "in vec2 tc;\n"
            "out vec4 color;\n"
            "uniform sampler2D samp;\n"
            "void main() { color = texture(samp, tc); }\n";

        gl::ShaderProgram cache_logo_shader;
        auto cache_logo_sprite = std::make_unique<gl::GLSprite>();
        bool cache_logo_loaded = false;
        if (win != nullptr) {
            const std::string logo_path = win->util.getFilePath("data/logo.png");
            if (std::filesystem::exists(logo_path)) {
                GLuint logo_texture = 0;
                try {
                    int logo_width = 0;
                    int logo_height = 0;
                    logo_texture = gl::loadTexture(
                        logo_path, logo_width, logo_height);
                    if (logo_texture &&
                        cache_logo_shader.loadProgramFromText(
                            kCacheLogoVert, kCacheLogoFrag)) {
                        cache_logo_sprite->initSize(win->w, win->h);
                        cache_logo_sprite->setName("samp");
                        cache_logo_sprite->setShader(&cache_logo_shader);
                        const float scale = std::min(
                            static_cast<float>(win->w) / logo_width,
                            static_cast<float>(win->h) / logo_height);
                        const int draw_width =
                            static_cast<int>(logo_width * scale);
                        const int draw_height =
                            static_cast<int>(logo_height * scale);
                        const int draw_x = (win->w - draw_width) / 2;
                        const int draw_y = (win->h - draw_height) / 2;
                        cache_logo_sprite->initWithTexture(
                            &cache_logo_shader, logo_texture,
                            draw_x, draw_y, draw_width, draw_height);
                        logo_texture = 0;
                        cache_logo_loaded = true;
                    }
                } catch (...) {
                }
                if (logo_texture)
                    glDeleteTextures(1, &logo_texture);
            }
        }

        int last_progress_percent = -1;
        const auto present_cache_progress = [&](std::size_t completed) {
            // Pump native window events for every shader so desktop
            // environments do not mark the synchronous rebuild as hung.
            SDL_PumpEvents();
            if (win == nullptr)
                return;

            const int percent = shader_files.empty()
                                    ? 100
                                    : static_cast<int>(
                                          completed * 100 /
                                          shader_files.size());
            if (percent == last_progress_percent)
                return;
            last_progress_percent = percent;

            glViewport(0, 0, win->w, win->h);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            if (cache_logo_loaded)
                cache_logo_sprite->draw();
            if (loadingFont != nullptr &&
                loadingFont->handle().has_value()) {
                const std::string progress_text =
                    "Building Shader Cache " +
                    std::to_string(completed) + "/" +
                    std::to_string(shader_files.size()) + "...";
                win->text.printText_Blended(
                    *loadingFont, 10, 10, progress_text);
            }
            SDL_GL_SwapWindow(win->getWindow());
            SDL_PumpEvents();
        };

        present_cache_progress(0);

        for (size_t i = 0; i < shader_files.size(); ++i) {
            present_cache_progress(i);
            const std::string &shader_file = shader_files[i];
            std::string full_path = library_path + "/" + shader_file;

            mx::system_out << "acmx2: Caching Shader " << i << "/" << shader_files.size() << ": [" << shader_file << "] \n";
            fflush(stdout);

            ShaderCacheEntry entry;
            std::filesystem::path file_path(shader_file);
            entry.shader_name = file_path.stem().string();

            mx::system_out << "  - Computing hash... ";
            fflush(stdout);
            entry.source_hash = preparedFragmentHash(full_path);
            mx::system_out << "done\n";
            fflush(stdout);

            if (isComputeShaderFile(shader_file)) {
                const bool compiled = compileShaderCacheEntry(
                    shader_file, full_path, vert_2d, vert_3d, false, entry);
                cache.entries.push_back(std::move(entry));
                mx::system_out
                    << (compiled ? "  ✔ COMPUTE SUCCESS\n"
                                 : "  ⚠ COMPUTE SKIPPED (passthrough placeholder)\n");
                fflush(stdout);
                continue;
            }

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
                if (!loadProgramWithSize(&prog_2d, vert_2d, full_path)) {
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
                    if (!loadProgramWithSize(&prog_3d, vert_3d, full_path)) {
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

        present_cache_progress(shader_files.size());

        if (cache.save(cache_file)) {
            size_t ok_count = 0;
            size_t failed_count = 0;
            for (const auto &e : cache.entries) {
                if (e.failed)
                    ++failed_count;
                else
                    ++ok_count;
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
        ShaderManifestData manifest;
        std::string manifest_error;
        if (!loadShaderManifest(library_path, manifest, manifest_error)) {
            mx::system_out << "acmx2: " << manifest_error << "\n";
            return false;
        }
        setCustomUniformValues(manifest.custom_uniforms);
        std::string cache_file =
            shaderCacheFilePath(win ? win->util.path : std::string(),
                                library_path, cache_size, use_history_array);

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

        std::vector<std::string> shader_files;
        if (!collectShaderLibraryEntries(library_path, shader_files, manifest_error)) {
            mx::system_out << "acmx2: " << manifest_error << "\n";
            return false;
        }

        if (shader_files.size() != cache.entries.size()) {
            mx::system_out << "acmx2: Shader count mismatch: manifest has " << shader_files.size()
                           << " shaders but cache has " << cache.entries.size()
                           << " entries. Rebuilding cache...\n";
            fflush(stdout);
            return false;
        }

        std::vector<std::size_t> staleIndices;
        for (std::size_t i = 0; i < shader_files.size(); ++i) {
            const std::string fullPath = library_path + "/" + shader_files[i];
            const uint64_t currentHash = preparedFragmentHash(fullPath);
            const ShaderProgramKind expectedKind = isComputeShaderFile(shader_files[i])
                                                       ? (compute_shader_supported
                                                              ? ShaderProgramKind::Compute
                                                              : ShaderProgramKind::ComputeUnavailable)
                                                       : ShaderProgramKind::Fragment;
            if (currentHash != cache.entries[i].source_hash ||
                cache.entries[i].shader_name !=
                    std::filesystem::path(shader_files[i]).stem().string() ||
                cache.entries[i].kind != expectedKind) {
                staleIndices.push_back(i);
            }
        }

        if (!staleIndices.empty()) {
            mx::system_out << "acmx2: Updating " << staleIndices.size()
                           << " changed shader cache entr"
                           << (staleIndices.size() == 1 ? "y" : "ies") << "...\n";
            mx::system_out.flush();
            for (const std::size_t index : staleIndices) {
                const std::string fullPath =
                    library_path + "/" + shader_files[index];
                mx::system_out << "acmx2: Incremental cache update: "
                               << shader_files[index] << "\n";
                mx::system_out.flush();
                compileShaderCacheEntry(shader_files[index], fullPath,
                                        vert_2d, vert_3d, cache.dual_mode,
                                        cache.entries[index]);
            }
            if (!cache.save(cache_file)) {
                mx::system_err << "acmx2: Could not save incrementally updated shader cache\n";
                return false;
            }
        }

        mx::system_out << "acmx2: Loading " << cache.entries.size() << " shaders from cache...\n";
        fflush(stdout);
        program_kinds.resize(cache.entries.size(), ShaderProgramKind::Fragment);

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
                        float scale = std::min(static_cast<float>(win->w) / lw, static_cast<float>(win->h) / lh);
                        int dw = static_cast<int>(lw * scale);
                        int dh = static_cast<int>(lh * scale);
                        int lx = (win->w - dw) / 2;
                        int ly = (win->h - dh) / 2;
                        logo_sprite_c->initWithTexture(&logo_shader_c, logo_tex, lx, ly, dw, dh);
                        logo_tex = 0;
                        logo_loaded_c = true;
                    }
                } catch (...) {
                }
                if (logo_tex) {
                    glDeleteTextures(1, &logo_tex);
                }
            }
        }

        int last_percent_reported = -1;
        size_t binary_fail_count = 0;

        // Helper: insert a passthrough program at the current slot to preserve
        // index alignment with the manifest when a cache entry cannot be used.
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
            program_kinds[i] = entry.failed && entry.kind == ShaderProgramKind::Compute
                                   ? ShaderProgramKind::ComputeUnavailable
                                   : entry.kind;

            // If this entry was marked as failed when the cache was built,
            // substitute a passthrough program so the slot index stays
            // aligned with the manifest.
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
                if (entry.kind == ShaderProgramKind::Compute)
                    program_kinds[i] = ShaderProgramKind::ComputeUnavailable;
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
        // delete the cache so loadProgramsWithCache() can rebuild it once.
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
                program_kinds.clear();
                program_names_2d.clear();
                program_names_3d.clear();
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
     * @param text         Shader library directory containing a supported manifest.
     * @param loadingFont  Font used for the on-screen progress overlay.
     */
    void loadProgramsWithCache(gl::GLWindow *win, const std::string &text, mx::Font &loadingFont) {
        ShaderManifestData manifest;
        std::string manifest_error;
        if (!loadShaderManifest(text, manifest, manifest_error))
            throw mx::Exception("acmx2: " + manifest_error);
        setCustomUniformValues(manifest.custom_uniforms);
        std::string vert_2d = win->util.getFilePath("data/vert.glsl");
        std::string vert_3d = win->util.getFilePath("data/vertex.glsl");
        if (loadFromCache(win, text, loadingFont, vert_2d, vert_3d)) {
            return;
        }
        // Cache miss (file absent / corrupt / source changed). Try to build the
        // cache now and reload from it so subsequent runs hit the binary cache
        // instead of recompiling 1700+ shaders every launch. If building or
        // reloading fails for any reason, fall back to a plain source compile.
        std::string cache_file =
            shaderCacheFilePath(win ? win->util.path : std::string(), text,
                                cache_size, use_history_array);
        mx::system_out << "acmx2: Building shader cache at: " << cache_file << "\n";
        fflush(stdout);
        programs_2d.clear();
        programs_3d.clear();
        program_kinds.clear();
        program_names_2d.clear();
        program_names_3d.clear();
        if (buildShaderCache(win, text, vert_2d, vert_3d, &loadingFont) &&
            loadFromCache(win, text, loadingFont, vert_2d, vert_3d)) {
            return;
        }
        mx::system_out << "acmx2: Cache build/reload failed; compiling from source.\n";
        fflush(stdout);
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
     * @brief Check whether the 2D shader at @p idx is a texture-cache shader.
     *
     * Used by the multipass pipeline (which always renders pass shaders
     * out of the 2D set) to decide per-pass whether to bind the cache
     * texture units.
     */
    bool isCache2D(size_t idx) const {
        auto it = program_names_2d.find(static_cast<int>(idx));
        if (it == program_names_2d.end())
            return false;
        return it->second.name.find("cache") != std::string::npos;
    }

    /// Return whether a library slot contains an executable compute program.
    bool isCompute(size_t idx) const {
        return idx < program_kinds.size() &&
               program_kinds[idx] == ShaderProgramKind::Compute;
    }

    /**
     * Run a compute library slot as a full-frame image pass.
     *
     * Compute shaders read the input through `uniform sampler2D samp` on
     * texture unit 0 and write an RGBA16F image on image unit 0. The image may
     * be declared with explicit `binding = 0`, or use one of the conventional
     * names outputImage, output_image, destTex, or img_output.
     */
    bool dispatchCompute2D(gl::GLWindow *win, size_t idx, GLuint input_texture,
                           GLuint output_texture) {
        if (!compute_shader_supported || !isCompute(idx) ||
            idx >= programs_2d.size() || input_texture == 0 ||
            output_texture == 0 || input_texture == output_texture) {
            return false;
        }

        gl::ShaderProgram *program = programs_2d[idx].get();
        while (glGetError() != GL_NO_ERROR) {
        }
        program->useProgram();
        updateShaderUniforms2D(win, idx);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, input_texture);
        const GLint input_location = glGetUniformLocation(program->id(), "samp");
        if (input_location != -1)
            glUniform1i(input_location, 0);

        static constexpr const char *OUTPUT_IMAGE_NAMES[] = {
            "outputImage", "output_image", "destTex", "img_output"};
        for (const char *name : OUTPUT_IMAGE_NAMES) {
            const GLint location = glGetUniformLocation(program->id(), name);
            if (location != -1)
                glUniform1i(location, 0);
        }
        glBindImageTexture(0, output_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                           GL_RGBA16F);

        const ProgramData &data = program_names_2d.at(static_cast<int>(idx));
        const GLuint local_x = static_cast<GLuint>(
            std::max(data.compute_work_group_size[0], 1));
        const GLuint local_y = static_cast<GLuint>(
            std::max(data.compute_work_group_size[1], 1));
        const GLuint groups_x =
            (static_cast<GLuint>(win->w) + local_x - 1) / local_x;
        const GLuint groups_y =
            (static_cast<GLuint>(win->h) + local_y - 1) / local_y;
        glDispatchCompute(groups_x, groups_y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                        GL_TEXTURE_FETCH_BARRIER_BIT |
                        GL_FRAMEBUFFER_BARRIER_BIT);
        glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        return glGetError() == GL_NO_ERROR;
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
            if (!isDraggingLeft) {
                clickStartX = currentX;
                clickStartY = currentY;
                lastClickX = currentX;
                lastClickY = currentY;
                isDraggingLeft = true;
                wasClicked = true;
            }
        } else {
            isDraggingLeft = false;
        }
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
            if (!isDraggingRight) {
                isDraggingRight = true;
            }
        } else {
            isDraggingRight = false;
        }
        float leftClickFlag = isDraggingLeft ? 1.0f : 0.0f;
        float rightClickFlag = isDraggingRight ? 1.0f : 0.0f;
        glUniform4f(n.iMouse, currentX, currentY, leftClickFlag, rightClickFlag);
        if (wasClicked && n.iMouseClick != -1) {
            glUniform2f(n.iMouseClick, lastClickX, lastClickY);
        }
        glUniform2f(n.iResolution, static_cast<float>(win->w), static_cast<float>(win->h));
        if (n.time_speed_loc != -1) {
            glUniform1f(n.time_speed_loc, time_speed);
        }
        uploadAcidCamUniforms(n, idx);
#ifdef AUDIO_ENABLED
        const auto audio_metrics = audio_analyzer != nullptr
                                       ? audio_analyzer->metrics()
                                       : acmx2::audio::AudioMetrics{};
        const float audio_sensitivity =
            audio_analyzer != nullptr ? audio_analyzer->sensitivity() : 1.0f;
        if (time_audio) {
            glUniform1f(n.amp, audio_metrics.amplitude * audio_warmup_envelope);
            glUniform1f(n.amp_untouched, audio_sensitivity);
        }
        if (n.iSampleRate != -1) {
            const float sample_rate = audio_analyzer != nullptr
                                          ? static_cast<float>(audio_analyzer->sample_rate())
                                          : 44100.0f;
            glUniform1f(n.iSampleRate, sample_rate);
        }
        if (n.iamp != -1) {
            glUniform1f(n.iamp, audio_metrics.frequency);
        }
        {
            float sense = audio_sensitivity * 4.0f * audio_warmup_envelope;
            if (n.amp_peak != -1) {
                glUniform1f(n.amp_peak, std::sqrt(audio_metrics.peak) * sense);
            }
            if (n.amp_rms != -1) {
                glUniform1f(n.amp_rms, std::sqrt(audio_metrics.rms) * sense);
            }
            if (n.amp_smooth != -1) {
                glUniform1f(n.amp_smooth, std::sqrt(audio_metrics.smooth) * sense);
            }
            if (n.amp_low != -1) {
                glUniform1f(n.amp_low, std::sqrt(audio_metrics.low) * sense);
            }
            if (n.amp_mid != -1) {
                glUniform1f(n.amp_mid, std::sqrt(audio_metrics.mid) * sense);
            }
            if (n.amp_high != -1) {
                glUniform1f(n.amp_high, std::sqrt(audio_metrics.high) * sense);
            }
        }
        // Tell the shader which texture unit holds the spectrum.
        if (n.spectrum_loc != -1) {
            glUniform1i(n.spectrum_loc, SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        }
        if (n.spectrum_zero_loc != -1)
            glUniform1i(n.spectrum_zero_loc,
                        SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        if (n.spectrum_history_loc != -1)
            glUniform1i(n.spectrum_history_loc, SpectrumHistory::TEXTURE_UNIT);
        if (n.spectrum_history_head_loc != -1)
            glUniform1i(n.spectrum_history_head_loc, spectrum_history_head);
        if (n.spectrum_history_size_loc != -1)
            glUniform1i(n.spectrum_history_size_loc, audio_buffer_count);
#endif
        // Re-assert the selected texture-cache sampler representation.
        if (use_history_array) {
            if (n.history_loc != -1)
                glUniform1i(n.history_loc, 1);
            if (n.history_head_loc != -1)
                glUniform1i(n.history_head_loc, history_head);
        } else {
            if (n.texture_array_base_loc != -1) {
                std::vector<GLint> units(static_cast<std::size_t>(cache_size), 0);
                for (int i = 0; i < cache_size; ++i) {
                    units[static_cast<std::size_t>(i)] = i + 1;
                }
                glUniform1iv(n.texture_array_base_loc, cache_size, units.data());
            } else {
                for (int i = 0;
                     i < static_cast<int>(n.texture_array_loc.size()); ++i) {
                    if (n.texture_array_loc[i] != -1)
                        glUniform1i(n.texture_array_loc[i], i + 1);
                }
            }
            for (int i = 0; i < 8 && i < cache_size; ++i) {
                if (n.texture_cache_loc[i] != -1)
                    glUniform1i(n.texture_cache_loc[i], i + 1);
            }
        }
#ifdef MIDI_ENABLED
        for (int i = 0; i < 4; ++i) {
            if (n.slider_loc[i] != -1)
                glUniform1f(n.slider_loc[i], midi_slider[i]);
        }
#endif
        uploadCustomUniforms(n);
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
            if (!isDraggingLeft) {
                clickStartX = currentX;
                clickStartY = currentY;
                lastClickX = currentX;
                lastClickY = currentY;
                isDraggingLeft = true;
                wasClicked = true;
            }
        } else {
            isDraggingLeft = false;
        }
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
            if (!isDraggingRight) {
                isDraggingRight = true;
            }
        } else {
            isDraggingRight = false;
        }
        float leftClickFlag = isDraggingLeft ? 1.0f : 0.0f;
        float rightClickFlag = isDraggingRight ? 1.0f : 0.0f;
        glUniform4f(n.iMouse, currentX, currentY, leftClickFlag, rightClickFlag);
        if (wasClicked && n.iMouseClick != -1) {
            glUniform2f(n.iMouseClick, lastClickX, lastClickY);
        }
        glUniform2f(n.iResolution, static_cast<float>(win->w), static_cast<float>(win->h));
        if (n.time_speed_loc != -1) {
            glUniform1f(n.time_speed_loc, time_speed);
        }
        uploadAcidCamUniforms(n, idx);
#ifdef AUDIO_ENABLED
        const auto audio_metrics = audio_analyzer != nullptr
                                       ? audio_analyzer->metrics()
                                       : acmx2::audio::AudioMetrics{};
        const float audio_sensitivity =
            audio_analyzer != nullptr ? audio_analyzer->sensitivity() : 1.0f;
        if (time_audio) {
            glUniform1f(n.amp, audio_metrics.amplitude * audio_warmup_envelope);
            glUniform1f(n.amp_untouched, audio_sensitivity);
        }
        if (n.iSampleRate != -1) {
            const float sample_rate = audio_analyzer != nullptr
                                          ? static_cast<float>(audio_analyzer->sample_rate())
                                          : 44100.0f;
            glUniform1f(n.iSampleRate, sample_rate);
        }
        if (n.iamp != -1) {
            glUniform1f(n.iamp, audio_metrics.frequency);
        }
        {
            float sense = audio_sensitivity * 4.0f * audio_warmup_envelope;
            if (n.amp_peak != -1) {
                glUniform1f(n.amp_peak, std::sqrt(audio_metrics.peak) * sense);
            }
            if (n.amp_rms != -1) {
                glUniform1f(n.amp_rms, std::sqrt(audio_metrics.rms) * sense);
            }
            if (n.amp_smooth != -1) {
                glUniform1f(n.amp_smooth, std::sqrt(audio_metrics.smooth) * sense);
            }
            if (n.amp_low != -1) {
                glUniform1f(n.amp_low, std::sqrt(audio_metrics.low) * sense);
            }
            if (n.amp_mid != -1) {
                glUniform1f(n.amp_mid, std::sqrt(audio_metrics.mid) * sense);
            }
            if (n.amp_high != -1) {
                glUniform1f(n.amp_high, std::sqrt(audio_metrics.high) * sense);
            }
        }
        if (n.spectrum_loc != -1) {
            glUniform1i(n.spectrum_loc, SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        }
        if (n.spectrum_zero_loc != -1)
            glUniform1i(n.spectrum_zero_loc,
                        SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        if (n.spectrum_history_loc != -1)
            glUniform1i(n.spectrum_history_loc, SpectrumHistory::TEXTURE_UNIT);
        if (n.spectrum_history_head_loc != -1)
            glUniform1i(n.spectrum_history_head_loc, spectrum_history_head);
        if (n.spectrum_history_size_loc != -1)
            glUniform1i(n.spectrum_history_size_loc, audio_buffer_count);
#endif
        // Re-assert the selected texture-cache sampler representation.
        if (use_history_array) {
            if (n.history_loc != -1)
                glUniform1i(n.history_loc, 1);
            if (n.history_head_loc != -1)
                glUniform1i(n.history_head_loc, history_head);
        } else {
            if (n.texture_array_base_loc != -1) {
                std::vector<GLint> units(static_cast<std::size_t>(cache_size), 0);
                for (int i = 0; i < cache_size; ++i) {
                    units[static_cast<std::size_t>(i)] = i + 1;
                }
                glUniform1iv(n.texture_array_base_loc, cache_size, units.data());
            } else {
                for (int i = 0;
                     i < static_cast<int>(n.texture_array_loc.size()); ++i) {
                    if (n.texture_array_loc[i] != -1)
                        glUniform1i(n.texture_array_loc[i], i + 1);
                }
            }
            for (int i = 0; i < 8 && i < cache_size; ++i) {
                if (n.texture_cache_loc[i] != -1)
                    glUniform1i(n.texture_cache_loc[i], i + 1);
            }
        }
#ifdef MIDI_ENABLED
        for (int i = 0; i < 4; ++i) {
            if (n.slider_loc[i] != -1)
                glUniform1f(n.slider_loc[i], midi_slider[i]);
        }
#endif
        uploadCustomUniforms(n);
    }

    /**
     * @brief Per-frame update: advance time_f, upload all uniforms to the active shader.
     *
     * Called once per frame from ACView::draw().  This method:
     * 1. Computes delta time from SDL performance counters.
     * 2. Advances `time_f` either by a normalized fixed frame interval or by
     *    wall-clock delta (scaled by time_speed), or by audio amplitude when
     *    audio-reactive time is enabled.
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
            const double time_delta =
                normalized_time && video_fps > 0.0 ? 1.0 / video_fps
                                                   : delta_time;
            const float step = static_cast<float>(time_delta) * time_speed;
            time_f += step;
        } else {
#ifdef AUDIO_ENABLED
            if (time_audio) {
                const double time_delta =
                    normalized_time && video_fps > 0.0 ? 1.0 / video_fps
                                                       : delta_time;
                float dt_scalex =
                    audio_delta ? static_cast<float>(time_delta) : 1.0f;
                const auto audio_metrics = audio_analyzer != nullptr
                                               ? audio_analyzer->metrics()
                                               : acmx2::audio::AudioMetrics{};
                const float audio_sensitivity =
                    audio_analyzer != nullptr ? audio_analyzer->sensitivity() : 1.0f;
                float new_ampx =
                    audio_metrics.amplitude * audio_sensitivity * time_speed * dt_scalex;
                time_f += new_ampx;
            }
#endif
        }

        // Wrap at a large multiple of 2*PI so cos/sin remain continuous across
        // the wrap (any integer-multiple-of-2*PI bound preserves trig phase) and
        // shaders using mod(time_f, N) with small N still see a freely-growing
        // time_f. 65536 * 2*PI keeps float32 precision high (well under 2^20).
        constexpr float TIME_F_WRAP = 65536.0f * 6.2831853f;
        if (time_f >= TIME_F_WRAP)
            time_f -= TIME_F_WRAP;
        else if (time_f < 0.0f)
            time_f += TIME_F_WRAP;

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
            if (!isDraggingLeft) {
                clickStartX = currentX;
                clickStartY = currentY;
                lastClickX = currentX;
                lastClickY = currentY;
                isDraggingLeft = true;
                wasClicked = true;
            }
        } else {
            isDraggingLeft = false;
        }
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
            if (!isDraggingRight) {
                isDraggingRight = true;
            }
        } else {
            isDraggingRight = false;
        }
        float leftClickFlag = isDraggingLeft ? 1.0f : 0.0f;
        float rightClickFlag = isDraggingRight ? 1.0f : 0.0f;
        glUniform4f(iMouseLoc, currentX, currentY, leftClickFlag, rightClickFlag);

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
        const auto audio_metrics = audio_analyzer != nullptr
                                       ? audio_analyzer->metrics()
                                       : acmx2::audio::AudioMetrics{};
        const float audio_sensitivity =
            audio_analyzer != nullptr ? audio_analyzer->sensitivity() : 1.0f;
        GLuint amp_i = names[index()].amp;
        float amplitude = 1.0f;
        float dt_scale = audio_delta ? static_cast<float>(delta_time) : 1.0f;
        float new_amp =
            audio_metrics.amplitude * audio_sensitivity * time_speed * dt_scale;
        if (std::isnan(new_amp) || std::isinf(new_amp) || new_amp > 1e6f) {
            amplitude = 1.0f;
        } else {
            amplitude = new_amp;
        }
        glUniform1f(amp_i, amplitude * audio_warmup_envelope);
        GLuint amp_u = names[index()].amp_untouched;
        glUniform1f(amp_u, audio_metrics.amplitude);
        GLint iSampleRateLoc = names[index()].iSampleRate;
        if (iSampleRateLoc != -1) {
            const float sample_rate = audio_analyzer != nullptr
                                          ? static_cast<float>(audio_analyzer->sample_rate())
                                          : 44100.0f;
            glUniform1f(iSampleRateLoc, sample_rate);
        }
        if (names[index()].iamp != -1) {
            glUniform1f(names[index()].iamp, audio_metrics.frequency);
        }
        {
            float sense = audio_sensitivity * 4.0f * audio_warmup_envelope;
            auto &n = names[index()];
            if (n.amp_peak != -1) {
                glUniform1f(n.amp_peak, std::sqrt(audio_metrics.peak) * sense);
            }
            if (n.amp_rms != -1) {
                glUniform1f(n.amp_rms, std::sqrt(audio_metrics.rms) * sense);
            }
            if (n.amp_smooth != -1) {
                glUniform1f(n.amp_smooth, std::sqrt(audio_metrics.smooth) * sense);
            }
            if (n.amp_low != -1) {
                glUniform1f(n.amp_low, std::sqrt(audio_metrics.low) * sense);
            }
            if (n.amp_mid != -1) {
                glUniform1f(n.amp_mid, std::sqrt(audio_metrics.mid) * sense);
            }
            if (n.amp_high != -1) {
                glUniform1f(n.amp_high, std::sqrt(audio_metrics.high) * sense);
            }
        }
        if (names[index()].spectrum_loc != -1) {
            glUniform1i(names[index()].spectrum_loc, SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        }
        auto &spectrum_names = names[index()];
        if (spectrum_names.spectrum_zero_loc != -1)
            glUniform1i(spectrum_names.spectrum_zero_loc,
                        SpectrumTexture::SPECTRUM_TEXTURE_UNIT);
        if (spectrum_names.spectrum_history_loc != -1)
            glUniform1i(spectrum_names.spectrum_history_loc,
                        SpectrumHistory::TEXTURE_UNIT);
        if (spectrum_names.spectrum_history_head_loc != -1)
            glUniform1i(spectrum_names.spectrum_history_head_loc,
                        spectrum_history_head);
        if (spectrum_names.spectrum_history_size_loc != -1)
            glUniform1i(spectrum_names.spectrum_history_size_loc,
                        audio_buffer_count);
#endif
        if (use_history_array) {
            auto &n = names[index()];
            if (n.history_loc != -1)
                glUniform1i(n.history_loc, 1);
            if (n.history_head_loc != -1)
                glUniform1i(n.history_head_loc, history_head);
        }
#ifdef MIDI_ENABLED
        for (int i = 0; i < 4; ++i) {
            if (names[index()].slider_loc[i] != -1)
                glUniform1f(names[index()].slider_loc[i], midi_slider[i]);
        }
#endif
        uploadCustomUniforms(names[index()]);
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
        if (color_alpha_r > 1.5f)
            color_alpha_r = 0.1f;
        if (color_alpha_g > 1.5f)
            color_alpha_g = 0.1f;
        if (color_alpha_b > 1.5f)
            color_alpha_b = 0.1f;

        if (alpha_dir) {
            alpha += 0.1f;
            if (alpha >= 6.0f)
                alpha_dir = false;
        } else {
            alpha -= 0.1f;
            if (alpha <= 1.0f)
                alpha_dir = true;
        }

        random_var = glm::vec4(rand() % 255, rand() % 255, rand() % 255, rand() % 255);
    }

    /**
     * @brief Upload configured custom uniforms to the GPU for the given program.
     *
     * Uploads each configured custom uniform to its location cached in
     * @p data. Each upload is skipped if the location is -1 (uniform not
     * declared in that shader).
     *
     * @param data ProgramData containing cached custom-uniform locations.
     */
    void uploadCustomUniforms(const ProgramData &data) const {
        const std::size_t count = std::min(custom_uniforms.size(),
                                           data.custom_uniform_locs.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (data.custom_uniform_locs[i] != -1) {
                glUniform1f(data.custom_uniform_locs[i],
                            static_cast<float>(custom_uniforms[i].value));
            }
        }
    }

    void uploadAcidCamUniforms(const ProgramData &n, size_t idx) {
        if (n.value_alpha_r != -1)
            glUniform1f(n.value_alpha_r, color_alpha_r);
        if (n.value_alpha_g != -1)
            glUniform1f(n.value_alpha_g, color_alpha_g);
        if (n.value_alpha_b != -1)
            glUniform1f(n.value_alpha_b, color_alpha_b);
        if (n.alpha_r_loc != -1)
            glUniform1f(n.alpha_r_loc, color_alpha_r);
        if (n.alpha_g_loc != -1)
            glUniform1f(n.alpha_g_loc, color_alpha_g);
        if (n.alpha_b_loc != -1)
            glUniform1f(n.alpha_b_loc, color_alpha_b);
        if (n.alpha_value != -1)
            glUniform1f(n.alpha_value, alpha);
        if (n.index_value != -1)
            glUniform1f(n.index_value, static_cast<float>(idx));
        if (n.optx_loc != -1)
            glUniform4fv(n.optx_loc, 1, glm::value_ptr(optx));
        if (n.random_var_loc != -1)
            glUniform4fv(n.random_var_loc, 1, glm::value_ptr(random_var));
        if (n.restore_black_loc != -1)
            glUniform1f(n.restore_black_loc, restore_black ? 1.0f : 0.0f);
        if (n.inc_value_loc != -1)
            glUniform4fv(n.inc_value_loc, 1, glm::value_ptr(inc_value));
        if (n.inc_valuex_loc != -1)
            glUniform4fv(n.inc_valuex_loc, 1, glm::value_ptr(inc_valuex));
    }

    /**
     * @brief Step time_f forward manually (when auto-time is paused).
     * @param value Amount to add to time_f.
     */
    void incTime(float value) {
        if (!time_active) {
            // Wrap at a large 2*PI multiple to preserve trig continuity while
            // letting time_f grow large enough for mod(time_f, N) shader usage.
            constexpr float TIME_F_WRAP = 65536.0f * 6.2831853f;
            time_f += value;
            if (time_f >= TIME_F_WRAP)
                time_f = std::fmod(time_f, TIME_F_WRAP);
            if (time_f < 0.0f)
                time_f += TIME_F_WRAP;
            mx::system_out << "acmx2: Time stepped forward: " << time_f << "\n";
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
            constexpr float TWO_PI = 6.2831853f;

            // Subtract the value and apply a true modulo wrap
            // The double fmod + addition ensures the result is always positive
            time_f = std::fmod(std::fmod(time_f - value, TWO_PI) + TWO_PI, TWO_PI);

            // If you specifically need to avoid 0.0 (e.g. to prevent division by zero in shaders)
            if (time_f < 0.0001f)
                time_f = TWO_PI;

            mx::system_out << "acmx2: Time stepped back (wrapped): " << time_f << "\n";
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
     * When enabled, time_f is driven by analyzer amplitude and sensitivity
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
    float getAmp() const {
        return audio_analyzer != nullptr ? audio_analyzer->metrics().amplitude : 0.0f;
    }
    float getAmpUntouched() const {
        return audio_analyzer != nullptr ? audio_analyzer->sensitivity() : 1.0f;
    }
#endif
    /// @brief Reserved for future SDL event handling inside the library.
    void event(SDL_Event &e) {}
};

enum class FrameRotation {
    None,
    Clockwise90,
    Rotate180,
    Counterclockwise90
};

/**
 * @brief Read a video's coded dimensions without opening a decoder.
 *
 * This metadata-only probe runs before the SDL/OpenGL window is created so
 * Default-resolution video playback can construct the window at its native
 * size. Resizing an already-created OpenGL window is asynchronous on some
 * window managers and can leave its drawable at the constructor's fallback
 * size.
 *
 * @return Native video dimensions, or @c std::nullopt when metadata cannot be
 * read. The normal decoder path reports the definitive error later.
 */
static std::optional<cv::Size> probe_video_size(const std::string &filename) {
    AVFormatContext *format_context = nullptr;
    if (avformat_open_input(&format_context, filename.c_str(), nullptr,
                            nullptr) < 0) {
        return std::nullopt;
    }

    std::optional<cv::Size> dimensions;
    if (avformat_find_stream_info(format_context, nullptr) >= 0) {
        const int stream_index = av_find_best_stream(
            format_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index >= 0) {
            const AVCodecParameters *parameters =
                format_context->streams[stream_index]->codecpar;
            if (parameters->width > 0 && parameters->height > 0) {
                dimensions = cv::Size(parameters->width, parameters->height);
            }
        }
    }

    avformat_close_input(&format_context);
    return dimensions;
}

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
    std::string shader_file;
    std::string prefix_path = ".";
    std::string model_file = "cube.mxmod.z";
    std::string human_model;            ///< PPHS human-segmentation ONNX model path (--human). Empty when disabled.
    bool human_background_only = false; ///< When true (with --human), shaders apply only to the background; person composited on top.
    float human_black = 0.35f;          ///< --black: mask black point (shadow crush threshold).
    float human_white = 0.75f;          ///< --white: mask white point (opacity saturation threshold).
    std::string edge_model;             ///< Dexined edge-detection ONNX model path (--edge). Empty when disabled.
    std::string onnx_model;             ///< Generic OnnxWrapper YAML config path (--onnx). Empty when disabled.
    int mode = 0;
    int shader_index = 0;
    std::optional<cv::Size> sizev = std::nullopt;
    std::optional<cv::Size> csize = std::nullopt;
    double fps_value = 24.0;
    bool repeat = false;
    std::tuple<int, std::string, int> slib;
    bool full = false;
    bool cache = false;
    bool cache_array = false;
    int cache_delay = 1;
    int cache_size = 8;
    bool copy_audio = false;
    bool is3d = false;
#ifdef AUDIO_ENABLED
    bool audio_enabled = false;
    bool audio_pass_through = false;
    unsigned int audio_channels = 2;
    float audio_sensitivty = 0.25f;
    float audio_warm_rate = 0.5f; ///< Startup warmup envelope rate (1/sec). 0.5 ~= 2s to full strength.
    std::string record_audio_file;
    float record_gain = 1.0f;
    std::string audio_file;
    bool audio_trunc = false;  ///< When true, stop playback when file audio reaches the end.
    bool audio_repeat = false; ///< When true, restart file audio from the beginning at EOF.
    int audio_buffers = 0;     ///< Requested spectrum-history array depth.
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
    std::vector<std::string> shader_pass_files;
    bool shader_pass_enabled = false;
    bool build_cache = false;
    std::string build_library_path;
    bool remove_broken = false;     ///< True when `--remove-broken <path>` was specified.
    std::string remove_broken_path; ///< Library path passed to `--remove-broken`.
#ifdef __APPLE__
    // The shader binary cache is unsupported on macOS (no usable
    // glProgramBinary path under the system OpenGL framework), so it
    // is permanently disabled on Apple platforms.
    bool use_shader_cache = false;
#else
    bool use_shader_cache = true;
#endif
    float time_speed = 1.0f;
    bool normalized_time = false;
    std::string playlist_file;
    int autopilot_frames = 0;               ///< Frames between random shader switches in autopilot mode (0 = disabled).
    bool autopilot_random_interval = false; ///< When true, randomize autopilot frame interval after each switch.
    int autopilot_random_timeout = 0;       ///< Upper bound (inclusive) for randomized autopilot interval.
    double duration = 0.0;
    double max_size_mb = 0.0;
    float cross_fade_duration = 0.5f; ///< Crossfade duration in seconds when switching playlist shaders (default: 0.5).
    bool use_yuv = false;
    bool flip_output = false;                           ///< Vertical flip output frames when set (e.g., for HDR correction).
    FrameRotation frame_rotation = FrameRotation::None; ///< Optional input-frame rotation.
    bool png_output = false;                            ///< Video-file mode only: write PNG frames to a subdirectory instead of encoding video.
    int generate_interval = 0;                          ///< Save a PNG frame every N frames to a subdirectory (video or camera mode, 0 = disabled).
    bool no_drop = false;                               ///< In video mode, pace frame production to encoder throughput.
    bool display_filter = false;                        ///< Display current shader/stack and GPU filter overlay in upper-left.
    std::string watermark_text;                         ///< User watermark text (--use-watermark). When non-empty, watermark is enabled.
    int watermark_r = 255;                              ///< Watermark red channel (0-255), default magenta-pink.
    int watermark_g = 0;                                ///< Watermark green channel (0-255).
    int watermark_b = 150;                              ///< Watermark blue channel (0-255).
    bool interface_shm = false;                         ///< Enable interface shared-memory control channel (Qt launcher use).
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
    bool usesTimelineClock = false;    ///< Frame timing follows an external real-time clock.
    uint64_t timelineFrame = 0;        ///< Clock position expressed in nominal video frames.
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
 *  │     writer.write_at_pts()       │
 *  │                                 │
 *  │  writerRunning  atomic<bool>    │
 *  │  Starts only after the first    │
 *  │  valid source frame is ready    │
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
    acmx2::audio::AudioEngine audio_engine;
    SpectrumTexture spectrumTex{audio_engine.analyzer()};     ///< 1D FFT magnitude texture for shaders.
    SpectrumHistory spectrumHistory{audio_engine.analyzer()}; ///< Runtime-sized FFT history array.
    int audio_buffer_count = 0;                               ///< Allocated history-array layer count.
    float audio_warmup_envelope = 0.0f;                       ///< Startup fade for audio-driven uniforms/textures.
    float audio_warmup_rate = 0.5f;                           ///< Warmup envelope slope in 1/sec (higher = faster ramp).
    std::chrono::steady_clock::time_point audio_warmup_last_tick = std::chrono::steady_clock::now();
    bool spectrum_scale_by_sense = false; ///< When true, scale spectrum 1D buffer by audio sensitivity.
    bool file_audio_mode = false;         ///< True when audio comes from a file instead of RtAudio.
    std::string audio_file_path;          ///< Path to the audio file used for file_audio_mode.
    bool audio_trunc_mode = false;        ///< When true, stop playback when file audio samples are exhausted.
    bool audio_repeat_mode = false;       ///< When true, restart file audio when its samples are exhausted.

    /// Reset the startup envelope used to tame initial audio intensity.
    void resetAudioWarmupEnvelope() {
        audio_warmup_envelope = 0.0f;
        audio_warmup_last_tick = std::chrono::steady_clock::now();
    }

    /// Advance and return the startup warmup envelope in [0,1].
    float updateAudioWarmupEnvelope() {
        if (!audio_is_enabled)
            return 1.0f;
        if (audio_warmup_rate <= 0.0f) {
            audio_warmup_envelope = 1.0f;
            return audio_warmup_envelope;
        }
        auto now = std::chrono::steady_clock::now();
        float delta_time = std::chrono::duration<float>(now - audio_warmup_last_tick).count();
        audio_warmup_last_tick = now;
        if (delta_time < 0.0f)
            delta_time = 0.0f;
        audio_warmup_envelope += delta_time * audio_warmup_rate;
        if (audio_warmup_envelope > 1.0f)
            audio_warmup_envelope = 1.0f;
        return audio_warmup_envelope;
    }
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
    static const char *midiKeyName(int code) {
        switch (code) {
        case 262:
            return "Right";
        case 263:
            return "Left";
        case 264:
            return "Down";
        case 265:
            return "Up";
        case 266:
            return "PgUp";
        case 267:
            return "PgDn";
        case 269:
            return "End";
        case 268:
            return "Home";
        case 260:
            return "Insert";
        case 261:
            return "Delete";
        case 298:
            return "F9";
        case 32:
            return "Space";
        case 44:
            return "Comma";
        case 45:
            return "Minus";
        case 46:
            return "Period";
        case 47:
            return "Slash";
        case 61:
            return "Plus/Eq";
        case 91:
            return "[";
        case 93:
            return "]";
        case 51:
            return "3";
        case 52:
            return "4";
        case 53:
            return "5";
        case 54:
            return "6";
        case 65:
            return "A";
        case 66:
            return "B";
        case 67:
            return "C";
        case 68:
            return "D";
        case 69:
            return "E";
        case 71:
            return "G";
        case 70:
            return "F";
        case 72:
            return "H";
        case 73:
            return "I";
        case 74:
            return "J";
        case 75:
            return "K";
        case 76:
            return "L";
        case 77:
            return "M";
        case 78:
            return "N";
        case 79:
            return "O";
        case 80:
            return "P";
        case 81:
            return "Q";
        case 82:
            return "R";
        case 83:
            return "S";
        case 84:
            return "T";
        case 85:
            return "U";
        case 86:
            return "V";
        case 87:
            return "W";
        case 88:
            return "X";
        case 89:
            return "Y";
        case 90:
            return "Z";
        case 500:
            return "TimeFwd";
        case 501:
            return "TimeBack";
        case 502:
            return "TimePause";
        case 503:
            return "TimeToggle";
        case 504:
            return "SpdUp";
        case 505:
            return "SpdDn";
        case 506:
            return "PitchUp";
        case 507:
            return "PitchDn";
        case 508:
            return "YawR";
        case 509:
            return "YawL";
        case 510:
            return "RotSpdUp";
        case 511:
            return "RotSpdDn";
        case 512:
            return "RollR";
        case 513:
            return "RollL";
        case 514:
            return "ScaleUp";
        case 515:
            return "ScaleDn";
        case 600:
            return "Slider1";
        case 601:
            return "Slider1";
        case 602:
            return "Slider2";
        case 603:
            return "Slider2";
        case 604:
            return "Slider3";
        case 605:
            return "Slider3";
        case 606:
            return "Slider4";
        case 607:
            return "Slider4";
        default:
            return "?";
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
                                    ? static_cast<unsigned int>(deviceIndex)
                                    : 0;
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
                if (!(iss >> keyPair))
                    continue;
                auto colonPos = keyPair.find(':');
                if (colonPos == std::string::npos)
                    continue;
                int k1 = std::stoi(keyPair.substr(0, colonPos));
                int k2 = std::stoi(keyPair.substr(colonPos + 1));
                char brace;
                if (!(iss >> brace) || brace != '{')
                    continue;
                int b0, b1, b2;
                if (!(iss >> b0 >> b1 >> b2))
                    continue;
                midiCodes.push_back({k1, k2,
                                     static_cast<unsigned char>(b0),
                                     static_cast<unsigned char>(b1),
                                     static_cast<unsigned char>(b2)});
            }
            mx::system_out << "acmx2: Loaded " << midiCodes.size() << " MIDI mapping(s)\n";
            fflush(stdout);
        } catch (RtMidiError &e) {
            mx::system_err << "acmx2: MIDI error: " << e.getMessage() << "\n";
            if (midiIn) {
                delete midiIn;
                midiIn = nullptr;
            }
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
        case 262:
            return SDLK_RIGHT;
        case 263:
            return SDLK_LEFT;
        case 264:
            return SDLK_DOWN;
        case 265:
            return SDLK_UP;
        case 266:
            return SDLK_PAGEUP;
        case 267:
            return SDLK_PAGEDOWN;
        case 268:
            return SDLK_HOME;
        case 269:
            return SDLK_END;
        case 260:
            return SDLK_INSERT;
        case 261:
            return SDLK_DELETE;
        case 298:
            return SDLK_F9;
        case 32:
            return SDLK_SPACE;
        case 44:
            return SDLK_COMMA;
        case 45:
            return SDLK_MINUS;
        case 46:
            return SDLK_PERIOD;
        case 47:
            return SDLK_SLASH;
        case 61:
            return SDLK_EQUALS;
        case 91:
            return SDLK_LEFTBRACKET;
        case 93:
            return SDLK_RIGHTBRACKET;
        case 51:
            return SDLK_3;
        case 52:
            return SDLK_4;
        case 53:
            return SDLK_5;
        case 54:
            return SDLK_6;
        case 65:
            return SDLK_a;
        case 66:
            return SDLK_b;
        case 67:
            return SDLK_c;
        case 68:
            return SDLK_d;
        case 69:
            return SDLK_e;
        case 71:
            return SDLK_g;
        case 70:
            return SDLK_f;
        case 72:
            return SDLK_h;
        case 73:
            return SDLK_i;
        case 74:
            return SDLK_j;
        case 76:
            return SDLK_l;
        case 77:
            return SDLK_m;
        case 78:
            return SDLK_n;
        case 79:
            return SDLK_o;
        case 80:
            return SDLK_p;
        case 81:
            return SDLK_q;
        case 82:
            return SDLK_r;
        case 83:
            return SDLK_s;
        case 84:
            return SDLK_t;
        case 85:
            return SDLK_u;
        case 75:
            return SDLK_k;
        case 86:
            return SDLK_v;
        case 87:
            return SDLK_w;
        case 88:
            return SDLK_x;
        case 89:
            return SDLK_y;
        case 90:
            return SDLK_z;
        // Virtual codes 504-513 handled directly in pollMidi
        case 504:
        case 505:
        case 506:
        case 507:
        case 508:
        case 509:
        case 510:
        case 511:
        case 512:
        case 513:
            return SDLK_UNKNOWN;
        default:
            return SDLK_UNKNOWN;
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
        if (!midiIn || !midiOpen)
            return;
        // Drain all pending MIDI messages and update knob state
        std::vector<unsigned char> msg;
        while (true) {
            midiIn->getMessage(&msg);
            if (msg.size() < 3)
                break;
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
                            if (cameraRotationSpeed > 50.0f)
                                cameraRotationSpeed = 50.0f;
                            mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                            fflush(stdout);
                            lastMidiButton = "RotSpdUp";
                            lastMidiButtonTime = std::chrono::steady_clock::now();
                        } else if (mc.key1 == 511) {
                            cameraRotationSpeed -= 0.5f;
                            if (cameraRotationSpeed < 0.5f)
                                cameraRotationSpeed = 0.5f;
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
            if (mc.key2 == 0)
                continue;
            auto key = std::make_pair(mc.b0, mc.b1);
            auto it = knobState.find(key);
            if (it == knobState.end())
                continue;
            unsigned char val = it->second;
            // Slider knobs: map CC value (0-127) directly to 0.0-1.0
            if (mc.key1 >= 600 && mc.key1 <= 606 && (mc.key1 % 2 == 0)) {
                int idx = (mc.key1 - 600) / 2;
                library.setMidiSlider(idx, static_cast<float>(val) / 127.0f);
                continue;
            }
            if (val == 64)
                continue;                                    // dead zone at center
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
                    if (modelRenderScale < 0.05f)
                        modelRenderScale = 0.05f;
                    mx::system_out << "acmx2: Model scale decreased to " << modelRenderScale << "\n";
                    fflush(stdout);
                } else {
                    SDL_Keycode k = (val > 64)
                                        ? midiKeyToSDL(mc.key1)
                                        : midiKeyToSDL(mc.key2);
                    if (k != SDLK_UNKNOWN)
                        injectKey(k, win);
                }
            }
        }
    }

    /// @brief Close the MIDI input port and free the RtMidiIn instance.
    void cleanupMidi() {
        if (midiIn) {
            if (midiOpen)
                midiIn->closePort();
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
        if (!midiOpen || midiCodes.empty())
            return;
        int y = startY;
        win->text.setColor({0, 255, 0, 255});
        win->text.printText_Blended(font, 10, y, "MIDI Active");
        y += 25;
        // Find max label width for alignment
        size_t maxLabelLen = 0;
        for (const auto &mc : midiCodes) {
            if (mc.key2 == 0)
                continue;
            size_t len = std::string(midiKeyName(mc.key1)).size() + 1 + std::string(midiKeyName(mc.key2)).size();
            if (len > maxLabelLen)
                maxLabelLen = len;
        }
        // Show knob states
        win->text.setColor({0, 255, 0, 255});
        for (const auto &mc : midiCodes) {
            if (mc.key2 == 0)
                continue;
            auto it = knobState.find({mc.b0, mc.b1});
            unsigned char val = (it != knobState.end()) ? it->second : 64;
            const char *dir = (val == 64) ? "--" : (val > 64) ? midiKeyName(mc.key1)
                                                              : midiKeyName(mc.key2);
            int barLen = 20;
            int pos = (val * barLen) / 127;
            std::string bar(barLen, '-');
            bar[barLen / 2] = '|';
            if (pos < barLen)
                bar[pos] = '#';
            std::string label = std::string(midiKeyName(mc.key1)) + "/" + midiKeyName(mc.key2);
            // Pad label to align bars
            while (label.size() < maxLabelLen)
                label += ' ';
            std::ostringstream oss;
            oss << label << " [" << bar << "] " << std::setw(3) << static_cast<int>(val) << " " << dir;
            win->text.printText_Blended(font, 10, y, oss.str());
            y += 22;
        }
        // Show last button press (fade after 2 seconds)
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - lastMidiButtonTime)
                           .count();
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
    bool recording_pbo_uses_timeline_clock[2] = {false, false};
    uint64_t recording_pbo_timeline_frame[2] = {0, 0};
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
        if (rgba16.empty())
            return;
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

    /** Allocate the shared fragment/compute ping-pong targets on demand. */
    void ensurePassTargets(gl::GLWindow *win) {
        if (passFBO[0] != 0 && pass_target_width == win->w &&
            pass_target_height == win->h) {
            return;
        }
        for (int pass = 0; pass < 2; ++pass) {
            if (passFBO[pass] != 0)
                glDeleteFramebuffers(1, &passFBO[pass]);
            if (passTexture[pass] != 0)
                glDeleteTextures(1, &passTexture[pass]);
            passFBO[pass] = 0;
            passTexture[pass] = 0;
        }
        for (int pass = 0; pass < 2; ++pass) {
            glGenFramebuffers(1, &passFBO[pass]);
            glGenTextures(1, &passTexture[pass]);
            glBindTexture(GL_TEXTURE_2D, passTexture[pass]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, win->w, win->h, 0,
                         GL_RGBA, GL_HALF_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindFramebuffer(GL_FRAMEBUFFER, passFBO[pass]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, passTexture[pass], 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
                GL_FRAMEBUFFER_COMPLETE) {
                throw mx::Exception("acmx2: shader pass framebuffer is not complete");
            }
        }
        pass_target_width = win->w;
        pass_target_height = win->h;
    }

    /** Bind history textures required by a cache-aware pass. */
    void bindPassHistoryTextures(size_t shader_index) {
        if (!texture_cache || !frame_cache.isFull() ||
            !library.isCache2D(shader_index)) {
            return;
        }
        if (texture_cache_array) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D_ARRAY, frame_cache.historyTexture());
        } else {
            for (int cache_index = 0; cache_index < library.cacheSize();
                 ++cache_index) {
                glActiveTexture(GL_TEXTURE1 + cache_index);
                glBindTexture(GL_TEXTURE_2D,
                              frame_cache.textureAt(cache_index));
            }
        }
        glActiveTexture(GL_TEXTURE0);
    }

    bool runComputePass(gl::GLWindow *win, size_t shader_index,
                        GLuint input_texture, GLuint output_texture) {
        bindPassHistoryTextures(shader_index);
        return library.dispatchCompute2D(win, shader_index, input_texture,
                                         output_texture);
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
        if (!input_is_hdr)
            return;

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
            if (tex == 0)
                return;
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
          frame_cache{
              static_cast<std::size_t>(args.cache_size > 0 ? args.cache_size : 8),
              args.cache_array},
          texture_cache{args.cache},
          texture_cache_array{args.cache_array},
          cache_delay{args.cache_delay},
          copy_audio{args.copy_audio},
          gpu_cuda_device{args.cuda_device},
          silent_mode{args.silent},
          no_drop_mode{args.no_drop},
          use_shader_cache_flag{args.use_shader_cache},
          flip_output{args.flip_output},
          frame_rotation{args.frame_rotation},
          png_video_mode{args.png_output && !args.filename.empty()},
          generate_mode{args.generate_interval > 0},
          generate_interval{args.generate_interval},
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
        audio_warmup_rate = std::max(args.audio_warm_rate, 0.0f);
        audio_record_file = args.record_audio_file;
        if (!args.audio_file.empty()) {
            if (file_audio_open(args.audio_file)) {
                audio_is_enabled = true;
                file_audio_mode = true;
                audio_file_path = args.audio_file;
                audio_trunc_mode = args.audio_trunc;
                audio_repeat_mode = args.audio_repeat;
                file_audio_set_repeat(audio_repeat_mode);
                audio_engine.analyzer().reset();
                audio_engine.analyzer().set_sample_rate(44100);
                audio_engine.analyzer().set_sensitivity(args.audio_sensitivty);
                if (args.audio_pass_through) {
                    if (!file_audio_enable_output(audio_output_device)) {
                        mx::system_err
                            << "acmx2: File audio playback could not be opened; "
                               "continuing with visual reactivity only\n";
                    } else {
                        mx::system_out
                            << "acmx2: File audio output is the master clock; "
                               "late video frames will be dropped\n";
                    }
                }
                resetAudioWarmupEnvelope();
                spectrumTex.init();
                audio_buffer_count =
                    spectrumHistory.init(std::max(args.audio_buffers, 0));
                if (audio_buffer_count > 0) {
                    library.setAudioBufferCount(audio_buffer_count);
                }
                mx::system_out << "acmx2: File audio enabled from: " << args.audio_file << "\n";
                mx::system_out << "acmx2: FFT spectrum texture initialised ("
                               << acmx2::audio::AudioAnalyzer::spectrum_bin_count()
                               << " bins on GL_TEXTURE"
                               << SpectrumTexture::SPECTRUM_TEXTURE_UNIT << ")\n";
                if (audio_buffer_count > 0) {
                    mx::system_out
                        << "acmx2: Audio spectrum history array enabled ("
                        << audio_buffer_count
                        << " layers in one GL_TEXTURE_1D_ARRAY on GL_TEXTURE"
                        << SpectrumHistory::TEXTURE_UNIT << ")\n";
                }
            } else {
                mx::system_err << "acmx2: Error could not open audio file: " << args.audio_file << "\n";
            }
        } else if (args.audio_enabled) {
            const acmx2::audio::AudioStreamConfig audio_config{
                args.audio_channels,
                args.audio_sensitivty,
                audio_input_device,
                audio_output_device,
                args.audio_pass_through,
            };
            if (!audio_engine.open(audio_config)) {
                mx::system_err << "acmx2: Error could not initialize audio\n";
            } else {
                audio_is_enabled = true;
                audio_engine.recorder().set_gain(args.record_gain);
                resetAudioWarmupEnvelope();
                spectrumTex.init();
                audio_buffer_count =
                    spectrumHistory.init(std::max(args.audio_buffers, 0));
                if (audio_buffer_count > 0) {
                    library.setAudioBufferCount(audio_buffer_count);
                }
                mx::system_out << "acmx2: FFT spectrum texture initialised ("
                               << acmx2::audio::AudioAnalyzer::spectrum_bin_count()
                               << " bins on GL_TEXTURE"
                               << SpectrumTexture::SPECTRUM_TEXTURE_UNIT << ")\n";
                if (audio_buffer_count > 0) {
                    mx::system_out
                        << "acmx2: Audio spectrum history array enabled ("
                        << audio_buffer_count
                        << " layers in one GL_TEXTURE_1D_ARRAY on GL_TEXTURE"
                        << SpectrumHistory::TEXTURE_UNIT << ")\n";
                }
            }
        }

#endif
        library.is3D(args.is3d);
        library.setTimeSpeed(args.time_speed);
        library.setVideoFPS(args.fps_value);
        library.setNormalizedTime(args.normalized_time);
        is3d_enabled = args.is3d;
        m_file = args.model_file;

#ifdef ACMX2_WITH_DNN
        if (!args.human_model.empty()) {
            try {
                human_model_path = args.human_model;
                human_background_only = args.human_background_only;
                human_black_point = args.human_black;
                human_white_point = args.human_white;
                human_seg_model = std::make_unique<ac_dnn::PPHS>(human_model_path);
                mx::system_out << "acmx2: Human segmentation (PPHS) enabled with model: "
                               << human_model_path
                               << " [automatic CPU/CUDA selection]"
                               << (human_background_only ? " [background-only shader mode]" : "")
                               << "\n";
            } catch (const std::exception &e) {
                mx::system_err << "acmx2: Failed to load human segmentation model '"
                               << args.human_model << "': " << e.what() << "\n";
                human_seg_model.reset();
            }
        } else if (args.human_background_only) {
            mx::system_err << "acmx2: --background was specified without --human; ignoring.\n";
        }
        if (!args.edge_model.empty()) {
            try {
                edge_det_model = std::make_unique<ac_dnn::Dexined>(args.edge_model);
                mx::system_out << "acmx2: Edge detection (Dexined) enabled with model: "
                               << args.edge_model
                               << " [automatic CPU/CUDA selection]\n";
            } catch (const std::exception &e) {
                mx::system_err << "acmx2: Failed to load edge detection model '"
                               << args.edge_model << "': " << e.what() << "\n";
                edge_det_model.reset();
            }
        }
        if (!args.onnx_model.empty()) {
            try {
                onnx_proc_model = std::make_unique<ac_dnn::OnnxWrapper>(args.onnx_model);
                mx::system_out << "acmx2: Generic ONNX model enabled from YAML: " << args.onnx_model << "\n";
            } catch (const std::exception &e) {
                mx::system_err << "acmx2: Failed to load ONNX model '" << args.onnx_model << "': " << e.what() << "\n";
                onnx_proc_model.reset();
            }
        }
#else
        if (!args.human_model.empty()) {
            mx::system_err << "acmx2: --human requested but this build has no OpenCV DNN support (configure with -DWITH_OPENCV_DNN=ON).\n";
        }
        if (args.human_background_only) {
            mx::system_err << "acmx2: --background requested but this build has no OpenCV DNN support.\n";
        }
        if (!args.edge_model.empty()) {
            mx::system_err << "acmx2: --edge requested but this build has no OpenCV DNN support (configure with -DWITH_OPENCV_DNN=ON).\n";
        }
        if (!args.onnx_model.empty()) {
            mx::system_err << "acmx2: --onnx requested but this build has no OpenCV DNN support (configure with -DWITH_OPENCV_DNN=ON).\n";
        }
#endif

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
        autopilot_random_interval = args.autopilot_random_interval;
        autopilot_random_timeout = args.autopilot_random_timeout;
        resetAutopilotInterval();
        duration_limit = args.duration;
        max_size_limit_mb = args.max_size_mb;
        if (max_size_limit_mb > 0.0) {
            max_size_limit_bytes = max_size_limit_mb * 1024.0 * 1024.0;
        }
        crossfadeDuration = args.cross_fade_duration;
#ifdef MIDI_ENABLED
        if (!args.midi_map_file.empty()) {
            initMidi(args.midi_map_file, args.midi_device);
        }
#endif
#if defined(__linux__) || defined(__APPLE__)
        if (args.interface_shm) {
            initShaderSelectionSharedMemory();
        }
#endif
    }

    bool is3d_enabled = false;

#ifdef ACMX2_WITH_DNN
    std::unique_ptr<ac_dnn::PPHS> human_seg_model;
    std::string human_model_path;
    bool human_background_only = false;
    float human_black_point = 0.35f;
    float human_white_point = 0.75f;
    std::unique_ptr<ac_dnn::Dexined> edge_det_model;
    std::unique_ptr<ac_dnn::OnnxWrapper> onnx_proc_model;
    GLuint human_overlay_tex = 0;
    int human_overlay_w = 0;
    int human_overlay_h = 0;
    bool human_overlay_ready = false; ///< True once a mask has been produced for the current frame.
#endif

    bool gpu_filter_enabled = false;
    std::vector<ac_gpu::Filter> gpu_filters;
    int gpu_current_filter_index = 0;
    std::unique_ptr<ac_gpu::DynamicFrameBuffer> gpu_frame_buffer;
#ifdef ACMX2_WITH_CUDA
    cv::cuda::GpuMat gpuWorkingBuffer;
    cv::cuda::GpuMat gpu_rotation_input;
    cv::cuda::GpuMat gpu_rotation_output;
    cv::cuda::GpuMat onnxGpuOutput;
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
    int autopilot_frames = 0;                ///< Frames between random switches in autopilot mode (0 = unset).
    bool autopilot_enabled = false;          ///< Toggle autopilot via SDLK_j when playlist is enabled.
    bool autopilot_sequential = false;       ///< When true, autopilot advances through the playlist in order instead of randomly (toggle via SDLK_y).
    int autopilot_counter = 0;               ///< Frames elapsed since last autopilot switch.
    bool autopilot_random_interval = false;  ///< When true, choose a new interval from [4, autopilot_random_timeout] after each switch.
    int autopilot_random_timeout = 0;        ///< Inclusive upper bound for random autopilot interval.
    int autopilot_interval_frames = 0;       ///< Active interval currently used by autopilot tick.
    bool autopilot_random_crossfade = false; ///< When true, autopilot picks a random crossfade shader on each switch.
    std::mt19937 autopilot_rng{std::random_device{}()};
    std::vector<int> saved_pass_list;
    bool saved_pass_enabled = false;
    double duration_limit = 0.0;
    size_t frames_proc = 0;
    double max_size_limit_mb = 0.0;
    double max_size_limit_bytes = 0.0;

    void resetAutopilotInterval() {
        if (autopilot_random_interval) {
            const int lower = 4;
            const int upper = std::max(lower, autopilot_random_timeout);
            std::uniform_int_distribution<int> dist(lower, upper);
            autopilot_interval_frames = dist(autopilot_rng);
        } else {
            autopilot_interval_frames = autopilot_frames;
        }
    }

    int activePlaylistSize() const {
        if (!playlist_tree.empty()) {
            return static_cast<int>(playlist_tree.size());
        }
        if (!playlist_indices.empty()) {
            return static_cast<int>(playlist_indices.size());
        }
        return 0;
    }

    void maybeRandomizeAutopilotCrossfade() {
        if (!autopilot_random_crossfade || crossfadeShaders.empty())
            return;
        const int n = static_cast<int>(crossfadeShaders.size());
        std::uniform_int_distribution<int> dist(0, n - 1);
        int next = dist(autopilot_rng);
        if (n > 1 && next == crossfadeShaderIndex)
            next = (next + 1) % n;
        crossfadeShaderIndex = next;
    }

    bool random_multipass_mode = false;
    std::vector<int> saved_pass_list_before_random;
    bool saved_pass_enabled_before_random = false;
    size_t saved_shader_index_before_random = 0;

#if defined(__linux__) || defined(__APPLE__)
    int shaderSelectionShmFd = -1;
    acmx2::ipc::ShaderSelectionShmData *shaderSelectionShm = nullptr;
    uint32_t shaderSelectionLastSequence = 0;
    uint32_t shaderReloadLastSequence = 0;
    uint32_t audioFileLastSequence = 0;

    std::vector<ShaderManifestData::CustomUniform>
    customUniformsFromSharedMemory() const {
        std::vector<ShaderManifestData::CustomUniform> uniforms;
        if (!shaderSelectionShm)
            return uniforms;
        const uint32_t count = std::min<uint32_t>(
            shaderSelectionShm->custom_uniform_count,
            acmx2::ipc::kShaderSelectionMaxCustomUniforms);
        uniforms.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const char *nameData = shaderSelectionShm->custom_uniform_names[i];
            std::size_t length = 0;
            while (length < acmx2::ipc::kShaderSelectionMaxUniformName &&
                   nameData[length] != '\0') {
                ++length;
            }
            ShaderManifestData::CustomUniform uniform;
            uniform.name.assign(nameData, length);
            if (!isValidCustomUniformName(uniform.name))
                continue;
            uniform.minimum = -std::numeric_limits<double>::max();
            uniform.maximum = std::numeric_limits<double>::max();
            uniform.step = 1.0;
            uniform.value = shaderSelectionShm->custom_uniform_values[i];
            if (!std::isfinite(uniform.value))
                uniform.value = 0.0;
            uniforms.push_back(std::move(uniform));
        }
        return uniforms;
    }

    void initShaderSelectionSharedMemory() {
        if (shaderSelectionShm)
            return;
        shaderSelectionShmFd = ::shm_open(acmx2::ipc::kShaderSelectionShmName, O_RDWR, 0666);
        if (shaderSelectionShmFd < 0)
            return;

        void *mapped = ::mmap(nullptr,
                              sizeof(acmx2::ipc::ShaderSelectionShmData),
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED,
                              shaderSelectionShmFd,
                              0);
        if (mapped == MAP_FAILED) {
            ::close(shaderSelectionShmFd);
            shaderSelectionShmFd = -1;
            return;
        }

        shaderSelectionShm = static_cast<acmx2::ipc::ShaderSelectionShmData *>(mapped);
        if (shaderSelectionShm->magic != acmx2::ipc::kShaderSelectionMagic ||
            shaderSelectionShm->version != acmx2::ipc::kShaderSelectionVersion) {
            cleanupShaderSelectionSharedMemory();
            return;
        }

        shaderSelectionLastSequence = shaderSelectionShm->sequence;
        shaderReloadLastSequence = shaderSelectionShm->reload_sequence;
        audioFileLastSequence = shaderSelectionShm->audio_file_sequence;
        library.setCustomUniformValues(customUniformsFromSharedMemory());
    }

    void cleanupShaderSelectionSharedMemory() {
        if (shaderSelectionShm) {
            ::munmap(shaderSelectionShm, sizeof(acmx2::ipc::ShaderSelectionShmData));
            shaderSelectionShm = nullptr;
        }
        if (shaderSelectionShmFd >= 0) {
            ::close(shaderSelectionShmFd);
            shaderSelectionShmFd = -1;
        }
    }

    void syncShaderSelectionFromInterface(gl::GLWindow *win) {
        if (!shaderSelectionShm)
            return;
        if (shaderSelectionShm->magic != acmx2::ipc::kShaderSelectionMagic ||
            shaderSelectionShm->version != acmx2::ipc::kShaderSelectionVersion) {
            return;
        }
        if (shaderSelectionShm->sequence == shaderSelectionLastSequence)
            return;

        shaderSelectionLastSequence = shaderSelectionShm->sequence;
        const auto readBoundedText = [](const char *buf, std::size_t cap) {
            std::size_t len = 0;
            while (len < cap && buf[len] != '\0') {
                ++len;
            }
            return std::string(buf, len);
        };
        const std::vector<std::string> interfaceShaderFiles =
            std::get<0>(flib) == 1
                ? sortedShaderLibraryEntries(std::get<1>(flib))
                : std::vector<std::string>{};

        library.setCustomUniformValues(customUniformsFromSharedMemory());

        if (shaderSelectionShm->audio_file_sequence != audioFileLastSequence) {
            audioFileLastSequence = shaderSelectionShm->audio_file_sequence;
#ifdef AUDIO_ENABLED
            const std::string requestedAudioPath = readBoundedText(
                shaderSelectionShm->audio_file_path,
                acmx2::ipc::kShaderSelectionMaxAudioFilePath);
            if (!file_audio_mode) {
                mx::system_err
                    << "acmx2: Ignoring live audio-file change because this process "
                       "was not started in audio-file mode\n";
            } else if (requestedAudioPath.empty()) {
                mx::system_err << "acmx2: Ignoring empty live audio-file request\n";
            } else if (file_audio_open(requestedAudioPath)) {
                audio_file_path = requestedAudioPath;
                audio_output_device = shaderSelectionShm->audio_output_device;
                audio_trunc_mode = shaderSelectionShm->audio_trunc != 0;
                audio_repeat_mode = shaderSelectionShm->audio_repeat != 0;
                file_audio_set_repeat(audio_repeat_mode);
                audio_engine.analyzer().reset();
                audio_engine.analyzer().set_sample_rate(44100);
                resetAudioWarmupEnvelope();
                if (shaderSelectionShm->audio_pass_through != 0 &&
                    !file_audio_enable_output(audio_output_device)) {
                    mx::system_err
                        << "acmx2: Live audio-file output could not be opened; "
                           "continuing with visual reactivity only\n";
                }
                mx::system_out << "acmx2: Switched file audio to: "
                               << requestedAudioPath << "\n";
            } else {
                mx::system_err << "acmx2: Could not switch file audio to: "
                               << requestedAudioPath << "\n";
            }
            mx::system_out.flush();
            mx::system_err.flush();
#endif
        }

        if (shaderSelectionShm->reload_sequence != shaderReloadLastSequence) {
            shaderReloadLastSequence = shaderSelectionShm->reload_sequence;
            const int requestedReloadIndex = shaderSelectionShm->reload_shader_index;
            const std::string requestedPath = readBoundedText(
                shaderSelectionShm->reload_shader_path,
                acmx2::ipc::kShaderSelectionMaxReloadPath);
            std::string reloadPath;
            size_t reloadIndex = 0;
            std::string reloadError;

            if (requestedReloadIndex < 0 || requestedPath.empty()) {
                reloadError = "Invalid shader reload request from interface";
            } else if (std::get<0>(flib) == 1) {
                const auto shaderFiles = sortedShaderLibraryEntries(std::get<1>(flib));
                if (static_cast<size_t>(requestedReloadIndex) >= shaderFiles.size()) {
                    reloadError = "Shader reload index is outside the shader manifest: " +
                                  std::to_string(requestedReloadIndex);
                } else if (!resolveShaderPathInLibrary(
                               std::get<1>(flib),
                               shaderFiles[static_cast<size_t>(requestedReloadIndex)],
                               reloadPath)) {
                    reloadError = "Could not resolve shader reload path from the manifest";
                } else {
                    std::error_code requestedError;
                    const auto canonicalRequested = std::filesystem::weakly_canonical(
                        std::filesystem::path(requestedPath), requestedError);
                    std::error_code expectedError;
                    const auto canonicalExpected = std::filesystem::weakly_canonical(
                        std::filesystem::path(reloadPath), expectedError);
                    if (requestedError || expectedError ||
                        canonicalRequested != canonicalExpected) {
                        reloadError = "Shader reload path does not match the requested library index";
                    } else {
                        reloadIndex = static_cast<size_t>(requestedReloadIndex);
                    }
                }
            } else {
                std::error_code requestedError;
                const auto canonicalRequested = std::filesystem::weakly_canonical(
                    std::filesystem::path(requestedPath), requestedError);
                std::error_code loadedError;
                const auto canonicalLoaded = std::filesystem::weakly_canonical(
                    std::filesystem::path(std::get<1>(flib)), loadedError);
                if (requestedError || loadedError || canonicalRequested != canonicalLoaded) {
                    reloadError = "Saved shader is not the shader loaded by this process";
                } else {
                    reloadPath = canonicalLoaded.string();
                }
            }

            if (reloadError.empty() &&
                library.reloadProgram(win, reloadIndex, reloadPath, reloadError)) {
                if (is3d_enabled)
                    cube.setShaderProgram(library.shader());
                sprite.setShader(library.shader());
                updateShaderNameCache();
                mx::system_out << "acmx2: Live reloaded shader " << reloadPath << "\n";
                mx::system_out.flush();
                fflush(stdout);
            } else {
                mx::system_err << "acmx2: Live shader reload failed for "
                               << requestedPath << ":\n"
                               << reloadError << "\n";
                mx::system_err.flush();
                fflush(stderr);
            }
        }

        std::vector<int> requestedPassList;
        requestedPassList.reserve(shaderSelectionShm->shader_pass_count);
        const uint32_t clampedPassCount = std::min<uint32_t>(
            shaderSelectionShm->shader_pass_count,
            acmx2::ipc::kShaderSelectionMaxPassCount);
        for (uint32_t i = 0; i < clampedPassCount; ++i) {
            int passIndex = shaderSelectionShm->shader_pass_indices[i];
            if (std::get<0>(flib) == 1) {
                const std::string passName = readBoundedText(
                    shaderSelectionShm->shader_pass_names[i],
                    acmx2::ipc::kShaderSelectionMaxShaderName);
                if (!passName.empty())
                    passIndex = shaderIndexForFile(interfaceShaderFiles,
                                                   passName);
            }
            if (passIndex < 0)
                continue;
            if (static_cast<size_t>(passIndex) >= library.size())
                continue;
            requestedPassList.push_back(passIndex);
        }
        const bool requestedPassEnabled = shaderSelectionShm->shader_pass_enabled != 0 && !requestedPassList.empty();
        const bool multipassChanged = (shader_pass_enabled != requestedPassEnabled) ||
                                      (shader_pass_list != requestedPassList);
        // Playlist mode owns the active pass list. A right-click shader
        // selection from the interface should change only the post/main
        // shader, just like Shift+Up/Down, without replacing the playlist
        // node with the interface's standalone multipass settings.
        if (!random_multipass_mode && !playlist_enabled) {
            if (multipassChanged && requestedPassEnabled)
                beginCrossfade(win);
            shader_pass_list = requestedPassList;
            shader_pass_enabled = requestedPassEnabled;
            updateShaderNameCache();
        }

        repeat = (shaderSelectionShm->repeat_enabled != 0);
        display_filter = (shaderSelectionShm->display_filter_enabled != 0);
        library.setNormalizedTime(
            shaderSelectionShm->normalized_time_enabled != 0);

        const std::string requestedWatermark = readBoundedText(
            shaderSelectionShm->watermark_text,
            acmx2::ipc::kShaderSelectionMaxWatermarkText);
        const bool requestedWatermarkEnabled =
            (shaderSelectionShm->watermark_enabled != 0) && !requestedWatermark.empty();
        enableWatermark = requestedWatermarkEnabled;
        watermark_text = requestedWatermark;
        watermark_r = std::clamp<int>(shaderSelectionShm->watermark_r, 0, 255);
        watermark_g = std::clamp<int>(shaderSelectionShm->watermark_g, 0, 255);
        watermark_b = std::clamp<int>(shaderSelectionShm->watermark_b, 0, 255);

#ifdef ACMX2_WITH_CUDA
        std::vector<int> requestedGpuFilters;
        const uint32_t clampedGpuCount = std::min<uint32_t>(
            shaderSelectionShm->gpu_filter_count,
            acmx2::ipc::kShaderSelectionMaxGpuFilterCount);
        requestedGpuFilters.reserve(clampedGpuCount);
        for (uint32_t i = 0; i < clampedGpuCount; ++i) {
            const int idx = shaderSelectionShm->gpu_filter_indices[i];
            if (idx < 0 || idx >= ac_gpu::AC_FILTER_MAX)
                continue;
            requestedGpuFilters.push_back(idx);
        }

        const bool requestedGpuEnabled =
            (shaderSelectionShm->gpu_filter_enabled != 0) && !requestedGpuFilters.empty();
        const int requestedGpuBufferSize =
            std::clamp<int>(static_cast<int>(shaderSelectionShm->gpu_buffer_size), 4, 32);

        std::vector<int> currentGpuFilters;
        currentGpuFilters.reserve(gpu_filters.size());
        for (const auto &f : gpu_filters) {
            currentGpuFilters.push_back(f.index);
        }

        const bool gpuBufferChanged =
            requestedGpuEnabled &&
            (!gpu_frame_buffer || gpu_frame_buffer->arraySize != requestedGpuBufferSize);
        const bool gpuConfigChanged =
            (gpu_filter_enabled != requestedGpuEnabled) ||
            (currentGpuFilters != requestedGpuFilters) ||
            gpuBufferChanged;

        if (gpuConfigChanged) {
            if (requestedGpuEnabled) {
                gpu_filters.clear();
                for (int idx : requestedGpuFilters) {
                    gpu_filters.push_back({idx, ac_gpu::filters[idx].name});
                }
                gpu_filter_enabled = !gpu_filters.empty();
                if (gpu_filter_enabled) {
                    gpu_current_filter_index = gpu_filters.front().index;
                    if (!gpu_frame_buffer || gpu_frame_buffer->arraySize != requestedGpuBufferSize) {
                        gpu_frame_buffer = std::make_unique<ac_gpu::DynamicFrameBuffer>(requestedGpuBufferSize);
                        if (d_ptrList) {
                            CHECK_CUDA(cudaFree(d_ptrList));
                            d_ptrList = nullptr;
                        }
                        CHECK_CUDA(cudaMalloc(&d_ptrList, requestedGpuBufferSize * sizeof(unsigned char *)));
                        gpu_frame_index = 0;
                        gpu_frame_dir = 1;
                    }
                    gpu_filtersChanged = true;
                } else {
                    gpu_filter_enabled = false;
                }
            } else {
                gpu_filter_enabled = false;
                gpu_filters.clear();
                gpu_current_filter_index = 0;
                gpu_filtersChanged = true;
            }
        }
#endif

        int requestedIndex = shaderSelectionShm->selected_index;
        if (std::get<0>(flib) == 1) {
            const std::string requestedName = readBoundedText(
                shaderSelectionShm->selected_shader_name,
                acmx2::ipc::kShaderSelectionMaxShaderName);
            if (!requestedName.empty())
                requestedIndex = shaderIndexForFile(interfaceShaderFiles,
                                                    requestedName);
        }
        if (requestedIndex < 0)
            return;
        if (static_cast<size_t>(requestedIndex) >= library.size())
            return;
        if (shaderLocked || random_multipass_mode)
            return;
        if (static_cast<size_t>(requestedIndex) == library.index())
            return;

        beginCrossfade(win);
        library.setIndex(static_cast<size_t>(requestedIndex));
        if (is3d_enabled)
            cube.setShaderProgram(library.shader());
        sprite.setShader(library.shader());
        updateShaderNameCache();
    }
#endif

    void generateRandomMultipass(gl::GLWindow *win) {
        static std::mt19937 rng(std::random_device{}());
        size_t shader_count = library.size();
        if (shader_count == 0)
            return;
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
            if (i + 1 < shader_pass_list.size())
                mx::system_out << ", ";
        }
        mx::system_out << "]\n";
        fflush(stdout);
    }

    void generateRandomMultipassShort(gl::GLWindow *win) {
        static std::mt19937 rng(std::random_device{}());
        size_t shader_count = library.size();
        if (shader_count == 0)
            return;
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
            if (i + 1 < shader_pass_list.size())
                mx::system_out << ", ";
        }
        mx::system_out << "]\n";
        fflush(stdout);
    }

    void generateRandomMultipassLong(gl::GLWindow *win) {
        static std::mt19937 rng(std::random_device{}());
        size_t shader_count = library.size();
        if (shader_count == 0)
            return;
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
            if (i + 1 < shader_pass_list.size())
                mx::system_out << ", ";
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
            if (n <= 0)
                return;
            std::uniform_int_distribution<int> dist(0, n - 1);
            int r = dist(autopilot_rng);
            if (n > 1 && r == playlist_index)
                r = (r + 1) % n;
            maybeRandomizeAutopilotCrossfade();
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
            maybeRandomizeAutopilotCrossfade();
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
            if (n <= 0)
                return;
            maybeRandomizeAutopilotCrossfade();
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
            maybeRandomizeAutopilotCrossfade();
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
        const GLenum cf_type = input_is_hdr ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE;
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
        if (crossfadeShaders.empty()) {
            crossfadeActive = false;
            return;
        }
        if (crossfadeShaderIndex < 0 || crossfadeShaderIndex >= static_cast<int>(crossfadeShaders.size()))
            crossfadeShaderIndex = 0;
        gl::ShaderProgram &activeCrossfade = crossfadeShaders[crossfadeShaderIndex];
        activeCrossfade.useProgram();
        activeCrossfade.setUniform("mv_matrix", glm::mat4(1.0f));
        activeCrossfade.setUniform("proj_matrix", glm::mat4(1.0f));
        activeCrossfade.setUniform("fade_alpha", crossfadeAlpha);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentTexture);
        activeCrossfade.setUniform("samp", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, crossfadePrevTexture);
        activeCrossfade.setUniform("prev_samp", 1);
        sprite.setShader(&activeCrossfade);
        sprite.setName("samp");
        sprite.draw(currentTexture, 0, 0, win->w, win->h);
        glActiveTexture(GL_TEXTURE0);
    }

    mx::Font overlayFont;
    mx::Font waterFont;
    gl::ShaderProgram muxOverlayShader;
    gl::GLSprite muxOverlaySprite;
    GLuint muxDummyTex = 0;
    float mux_time_f = 0.0f;
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
#if defined(__linux__) || defined(__APPLE__)
        cleanupShaderSelectionSharedMemory();
#endif
#ifdef MIDI_ENABLED
        cleanupMidi();
#endif
#ifdef ACMX2_WITH_DNN
        if (human_overlay_tex != 0) {
            glDeleteTextures(1, &human_overlay_tex);
            human_overlay_tex = 0;
        }
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

        if (pboIds[0] && (writer.is_open() || png_video_mode) &&
            recording_pbo_primed &&
            win_w > 0 && win_h > 0) {
            // Double-buffered readback always leaves exactly one completed
            // recording frame pending. Flushing both PBOs duplicated the
            // preceding frame and could make a duration-limited silent render
            // exceed its configured maximum by one frame.
            const int pending_pbo_index = pboNextIndex;
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pending_pbo_index]);
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
                fd.usesTimelineClock =
                    recording_pbo_uses_timeline_clock[pending_pbo_index];
                fd.timelineFrame =
                    recording_pbo_timeline_frame[pending_pbo_index];

                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    frameQueue.push(std::move(fd));
                }
                queueCondVar.notify_one();
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

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
            if (audio_engine.recorder().is_recording())
                audio_engine.recorder().stop();
            if (file_audio_mode)
                file_audio_close();
            else
                audio_engine.close();
            spectrumTex.cleanup();
            spectrumHistory.cleanup();
        }
#endif

        for (int i = 0; i < 2; ++i) {
#ifdef ACMX2_WITH_CUDA
            if (recordCudaPboResources[i]) {
                CHECK_CUDA(cudaGraphicsUnregisterResource(recordCudaPboResources[i]));
                recordCudaPboResources[i] = nullptr;
            }
#else
            static_cast<void>(i);
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
        pass_target_width = 0;
        pass_target_height = 0;
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
            frame_cache.cleanup();
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
        print_open_gl_uniform_limits();
        mx::system_out.flush();

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
            if (rotation_swaps_dimensions()) {
                std::swap(w, h);
                std::swap(frame_w, frame_h);
            }
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
                        // Pace the producer to encoder capacity so no encoded
                        // frames are dropped and no large queue burst forms.
                        writer.set_block_when_full(true);
                    }
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << encode_opts.crf
                                   << " preset: " << encode_opts.preset
                                   << " tune: " << (encode_opts.tune.empty() ? "none" : encode_opts.tune)
                                   << " codec: " << encode_opts.codec
                                   << (encode_opts.realtime ? " [realtime]" : "")
                                   << " FPS: " << fps << "\n";
                    mx::system_out << "acmx2: Pipeline mode => decode: graphic/image, encode: "
                                   << (writer.is_hardware_encode() ? "h264_nvenc (hardware)" : "h264 (software)") << "\n";

                    fflush(stdout);
                    fflush(stderr);
                } else {
                    throw mx::Exception("Could not open output video file: " + ofilename);
                }
            }
        } else if (filename.empty()) {
            if (no_drop_mode) {
                mx::system_out
                    << "acmx2: --no-drop is ignored in webcam mode; "
                       "wall-clock timestamps and late-frame dropping remain active\n";
                no_drop_mode = false;
            }
#ifdef __linux__
            const bool loopback_device = isV4l2LoopbackDevice(camera_index);
#else
            const bool loopback_device = false;
#endif
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
            const double requested_fps = fps;
            const bool fps_configured = cap.set(cv::CAP_PROP_FPS, requested_fps);
            w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            const double reported_fps = cap.get(cv::CAP_PROP_FPS);
            // v4l2loopback may continue to report the producer's initial
            // 30 fps interval even after accepting a high-speed consumer
            // interval.  Keep the explicitly requested rate in that case so
            // the render loop does not impose an artificial 30 fps cap.
            if (loopback_device && requested_fps > 0.0) {
                fps = requested_fps;
            } else if (reported_fps > 0.0) {
                fps = reported_fps;
            } else {
                fps = requested_fps;
            }
            frame_w = w;
            frame_h = h;
            mx::system_out << "acmx2: Camera opened: " << w << "x" << h
                           << " at FPS: " << fps;
            if (loopback_device && reported_fps > 0.0 &&
                std::abs(reported_fps - requested_fps) > 0.05) {
                mx::system_out << " (loopback reports " << reported_fps
                               << ", requested " << requested_fps << ")";
            } else if (!fps_configured && requested_fps > 0.0) {
                mx::system_out << " (driver rejected requested " << requested_fps << ")";
            }
            mx::system_out << "\n";
            fflush(stderr);
            fflush(stdout);

            if (rotation_swaps_dimensions()) {
                std::swap(w, h);
                std::swap(frame_w, frame_h);
            }

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
                    mx::system_out << "acmx2: Pipeline mode => decode: camera, encode: "
                                   << (writer.is_hardware_encode() ? "(hardware)" : "(software)") << "\n";
                    if (!no_drop_mode && fps > 0.0) {
                        mx::system_out
                            << "acmx2: Webcam recording follows wall-clock timestamps; "
                               "late frames will be dropped\n";
                    }
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
                        (trc == AVCOL_TRC_SMPTE2084) ? "PQ (SMPTE2084)" : (trc == AVCOL_TRC_ARIB_STD_B67) ? "HLG (ARIB STD-B67)"
                                                                      : (trc == AVCOL_TRC_BT2020_10)      ? "BT.2020 10-bit"
                                                                      : (trc == AVCOL_TRC_BT2020_12)      ? "BT.2020 12-bit"
                                                                                                          : "unknown";
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

            if (fps > 60.0) {
                if (SDL_GL_SetSwapInterval(0) == 0) {
                    mx::system_out << "acmx2: VSync disabled for high-frame-rate video input ("
                                   << fps << " FPS)\n";
                } else {
                    mx::system_err << "acmx2: Could not disable VSync for " << fps
                                   << " FPS video input: " << SDL_GetError() << "\n";
                }
            }

            if (rotation_swaps_dimensions()) {
                std::swap(w, h);
                std::swap(frame_w, frame_h);
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
                if (png_video_mode) {
                    std::filesystem::path out_path(ofilename);
                    std::filesystem::path out_parent = out_path.parent_path();
                    if (out_parent.empty()) {
                        out_parent = ".";
                    }
                    std::string out_name = out_path.filename().string();
                    if (out_name.empty()) {
                        out_name = "output";
                    }
                    std::filesystem::path frame_dir = out_parent / ("video_file-" + out_name + "-png");
                    std::error_code mk_err;
                    if (!std::filesystem::exists(frame_dir, mk_err) && !std::filesystem::create_directories(frame_dir, mk_err)) {
                        throw mx::Exception("Could not create PNG output directory: " + frame_dir.string());
                    }
                    png_video_dir = frame_dir.string();
                    mx::system_out << "acmx2: --png enabled: writing video frames to " << png_video_dir << "\n";
                } else {
                    if (writer.open(ofilename, w, h, fps, encode_opts)) {
                        bool block_encoder_queue =
                            silent_mode || no_drop_mode;
                        if (block_encoder_queue) {
                            // Batch transcoding or --no-drop: pace the producer
                            // to encoder capacity instead of filling a large
                            // queue or dropping frames.
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
                            mx::system_out << "acmx2: --no-drop active (video mode): frame processing paced to encoder throughput\n";
                        }
                        if (encode_opts.hdr.enabled) {
                            mx::system_out << "acmx2: *** HDR OUTPUT ENABLED: writing HEVC Main10 + BT.2020 "
                                           << (encode_opts.hdr.color_trc == AVCOL_TRC_ARIB_STD_B67 ? "HLG" : "PQ")
                                           << " ***\n";
                        }
                        mx::system_out << "acmx2: Pipeline mode => decode: " << decode_mode
                                       << ", encode: "
                                       << (writer.is_hardware_encode() ? "(hardware)" : "(software)") << "\n";
                        fflush(stdout);
                        fflush(stderr);
                    } else {
                        throw mx::Exception("Could not open output video file: " + ofilename);
                    }
                }
            }
        } else if (graphic.empty() && filename.empty()) {
            throw mx::Exception("Requires input from a file, or camera.");
        }

        if (generate_mode) {
            std::filesystem::path gen_dir;
            if (!ofilename.empty()) {
                std::filesystem::path out_path(ofilename);
                std::filesystem::path out_parent = out_path.parent_path();
                if (out_parent.empty())
                    out_parent = ".";
                std::string out_name = out_path.filename().string();
                if (out_name.empty())
                    out_name = "output";
                gen_dir = out_parent / ("video_file-" + out_name + "-generate");
            } else if (!filename.empty()) {
                std::filesystem::path in_path(filename);
                std::filesystem::path in_parent = in_path.parent_path();
                if (in_parent.empty())
                    in_parent = ".";
                std::string in_name = in_path.filename().string();
                if (in_name.empty())
                    in_name = "input";
                gen_dir = in_parent / ("video_file-" + in_name + "-generate");
            } else {
                gen_dir = std::filesystem::path("camera-generate");
            }
            std::error_code mk_err;
            if (!std::filesystem::exists(gen_dir, mk_err) && !std::filesystem::create_directories(gen_dir, mk_err)) {
                throw mx::Exception("Could not create generate output directory: " + gen_dir.string());
            }
            generate_dir = gen_dir.string();
            mx::system_out << "acmx2: --generate " << generate_interval
                           << ": saving PNG frames every " << generate_interval
                           << " frames to " << generate_dir << "\n";
            fflush(stdout);
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

        if (muxOverlayShader.loadProgramFromText(kHdrVertPassthrough, kMuxOverlayFrag)) {
            static constexpr unsigned char kWhite[4] = {255, 255, 255, 255};
            glGenTextures(1, &muxDummyTex);
            glBindTexture(GL_TEXTURE_2D, muxDummyTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, kWhite);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);
            muxOverlaySprite.initSize(win->w, win->h);
            muxOverlaySprite.setName("samp");
            muxOverlaySprite.initWithTexture(&muxOverlayShader, muxDummyTex, 0.0f, 0.0f, win->w, win->h);
        }

        library.enableCache(use_shader_cache_flag);
        // Tell the library how many cache textures we'll bind so it can
        // (a) inject `#define SIZE N` into fragment sources before compile,
        // (b) size the per-program `texture_array_loc` lookup vector,
        // (c) keep the on-disk binary cache keyed per-size to avoid
        //     reusing a binary compiled for a different SIZE.
        library.setCacheSize(static_cast<int>(frame_cache.capacity()));
        library.setHistoryTextureArray(texture_cache_array);
        if (std::get<0>(flib) == 1) {
            if (use_shader_cache_flag)
                library.loadProgramsWithCache(win, std::get<1>(flib), overlayFont);
            else
                library.loadPrograms(win, std::get<1>(flib), overlayFont);
        } else {
            library.loadProgram(win, std::get<1>(flib));
        }
#if defined(__linux__) || defined(__APPLE__)
        if (shaderSelectionShm)
            library.setCustomUniformValues(customUniformsFromSharedMemory());
#endif
        library.setIndex(std::get<2>(flib));
        if (!playlist_file.empty()) {
            std::ifstream pfile(playlist_file);
            if (!pfile.is_open()) {
                mx::system_err << "acmx2: Error could not open playlist: " << playlist_file << "\n";
            } else {
                std::string line;
                PlaylistNode *currentNode = nullptr;
                const auto shaderFiles =
                    sortedShaderLibraryEntries(std::get<1>(flib));
                while (std::getline(pfile, line)) {
                    if (line.empty())
                        continue;
                    if (line.front() == '[' && line.back() == ']') {
                        playlist_tree.push_back({line.substr(1, line.size() - 2), {}});
                        currentNode = &playlist_tree.back();
                        continue;
                    }
                    int idx = shaderIndexForFile(shaderFiles, line);
                    if (idx < 0) {
                        const std::string name =
                            std::filesystem::path(line).stem().string();
                        idx = library.findShaderByName(name);
                    }
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
            std::string m_file_base = m_file;
            if (m_file_base.size() > 5 && m_file_base.substr(0, 5) == "data/")
                m_file_base = m_file_base.substr(5);
            m_file_path = win->util.getFilePath("data/" + m_file_base);
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
        {
            static const char *kCrossfadeFiles[] = {
                "data/xfade_01_linear.glsl",
                "data/xfade_02_block.glsl",
                "data/xfade_03_wipe.glsl",
                "data/xfade_04_radial.glsl",
                "data/xfade_05_pixelate.glsl",
                "data/xfade_06_dissolve.glsl",
                "data/xfade_07_swirl.glsl",
                "data/xfade_08_glitch.glsl",
                "data/xfade_09_diamond.glsl",
                "data/xfade_10_burn.glsl",
                "data/xfade_11_fade_black.glsl",
                "data/xfade_12_fade_white.glsl",
                "data/xfade_13_slide_left.glsl",
                "data/xfade_14_slide_right.glsl",
                "data/xfade_15_slide_up.glsl",
                "data/xfade_16_slide_down.glsl",
                "data/xfade_17_diagonal_wipe.glsl",
                "data/xfade_18_iris_open.glsl",
                "data/xfade_19_iris_close.glsl",
                "data/xfade_20_checker.glsl",
                "data/xfade_21_blinds_h.glsl",
                "data/xfade_22_blinds_v.glsl",
                "data/xfade_23_zoom_in.glsl",
                "data/xfade_24_zoom_out.glsl",
                "data/xfade_25_rotate.glsl",
                "data/xfade_26_ripple.glsl",
                "data/xfade_27_wave.glsl",
                "data/xfade_28_chroma.glsl",
                "data/xfade_29_invert.glsl",
                "data/xfade_30_flash.glsl",
                "data/xfade_31_explode.glsl",
                "data/xfade_32_mosaic.glsl",
                "data/xfade_33_shutter.glsl",
                "data/xfade_34_luma.glsl",
                "data/xfade_35_noise.glsl",
            };
            crossfadeShaders.clear();
            crossfadeShaderNames.clear();
            const std::string vertPath = win->util.getFilePath("data/vert.glsl");
            for (const char *frag : kCrossfadeFiles) {
                gl::ShaderProgram prog;
                if (!prog.loadProgram(vertPath, win->util.getFilePath(frag))) {
                    throw mx::Exception(std::string("Error loading crossfade shader: ") + frag);
                }
                crossfadeShaders.push_back(prog);
                std::string nm = frag;
                auto slash = nm.find_last_of('/');
                if (slash != std::string::npos)
                    nm = nm.substr(slash + 1);
                auto dot = nm.find_last_of('.');
                if (dot != std::string::npos)
                    nm = nm.substr(0, dot);
                crossfadeShaderNames.push_back(nm);
            }
            if (crossfadeShaderIndex < 0 || crossfadeShaderIndex >= static_cast<int>(crossfadeShaders.size()))
                crossfadeShaderIndex = 0;
        }
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw mx::Exception("OpenGL error occurred: GL Error: " + std::to_string(error));
        }

        library.useProgram();
        if (texture_cache) {
            if (input_is_hdr) {
                // HDR cache rides on the post-decode linear BT.2020 RGBA16F
                // texture, which is allocated at win->w x win->h by
                // ensureHdrResources(). The ring uses the same dimensions
                // and format so glCopyTexSubImage2D can copy GPU->GPU.
                frame_cache.init(win->w, win->h, true);
            } else {
                frame_cache.init(frame_w, frame_h);
            }
            mx::system_out << "acmx2: Texture cache initalized.\n";
            fflush(stdout);
        }
        sprite.initSize(win->w, win->h);
        // Initialise the uploader at the SOURCE frame dimensions, not the
        // output (win->w/h) dimensions. When --resolution differs from the
        // camera/video source size, the per-frame TextureUploader::update()
        // would otherwise hit its size-mismatch branch on the very first
        // frame, call init() again, delete the original GL texture and
        // create a new textureID. `camera_texture` (captured below) would
        // then point at a deleted name and every subsequent sample would
        // produce black. Initialising at the source size keeps the
        // CUDA<->GL registration stable for the common case.
        {
            const int up_w = (frame_w > 0) ? frame_w : win->w;
            const int up_h = (frame_h > 0) ? frame_h : win->h;
            tex_uploader.init(up_w, up_h);
        }
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
            // Give the camera ~2 seconds to stabilize after loading before
            // pushing real content into the frame cache. This ensures that
            // any loading-screen remnants are not captured into samp1-samp8.
            cache_warmup_frames = 60;
        }
    }

    cv::Mat newFrame;

    bool rotation_swaps_dimensions() const {
        return frame_rotation == FrameRotation::Clockwise90 ||
               frame_rotation == FrameRotation::Counterclockwise90;
    }

    void rotate_frame(cv::Mat &frame) {
        if (frame.empty() || frame_rotation == FrameRotation::None) {
            return;
        }

        // SDR frames are vertically corrected before upload and again during
        // GL readback. Conjugating a 90-degree rotation by those flips
        // reverses its visible direction, so use the opposite source-space
        // transform to preserve the direction selected by the user.
        FrameRotation source_rotation = frame_rotation;
        if (!input_is_hdr) {
            if (source_rotation == FrameRotation::Clockwise90) {
                source_rotation = FrameRotation::Counterclockwise90;
            } else if (source_rotation ==
                       FrameRotation::Counterclockwise90) {
                source_rotation = FrameRotation::Clockwise90;
            }
        }

#ifdef ACMX2_WITH_CUDA
        gpu_rotation_input.upload(frame);
        cv::Size destination_size = frame.size();
        double angle = 180.0;
        double shift_x = static_cast<double>(frame.cols - 1);
        double shift_y = static_cast<double>(frame.rows - 1);

        if (source_rotation == FrameRotation::Clockwise90) {
            destination_size = cv::Size(frame.rows, frame.cols);
            angle = -90.0;
            shift_x = static_cast<double>(frame.rows - 1);
            shift_y = 0.0;
        } else if (source_rotation == FrameRotation::Counterclockwise90) {
            destination_size = cv::Size(frame.rows, frame.cols);
            angle = 90.0;
            shift_x = 0.0;
            shift_y = static_cast<double>(frame.cols - 1);
        }

        cv::cuda::rotate(gpu_rotation_input, gpu_rotation_output,
                         destination_size, angle, shift_x, shift_y,
                         cv::INTER_NEAREST);
        gpu_rotation_output.download(frame);
#else
        int rotation_code = cv::ROTATE_180;
        if (source_rotation == FrameRotation::Clockwise90) {
            rotation_code = cv::ROTATE_90_CLOCKWISE;
        } else if (source_rotation == FrameRotation::Counterclockwise90) {
            rotation_code = cv::ROTATE_90_COUNTERCLOCKWISE;
        }
        cv::rotate(frame, frame, rotation_code);
#endif
    }

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
        // at full speed. Real-time file-audio output is the exception: its
        // hardware clock must pace video even when the window is hidden.
        bool use_realtime_pacing = !silent_mode;
#ifdef AUDIO_ENABLED
        use_realtime_pacing =
            use_realtime_pacing ||
            (file_audio_mode && file_audio_has_output_clock());
#endif
        if (fps > 0.0 && use_realtime_pacing) {
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

#if defined(__linux__) || defined(__APPLE__)
        syncShaderSelectionFromInterface(win);
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
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            if (muxDummyTex) {
                muxOverlayShader.useProgram();
                muxOverlayShader.setUniform("mv_matrix", glm::mat4(1.0f));
                muxOverlayShader.setUniform("proj_matrix", glm::mat4(1.0f));
                glUniform1f(glGetUniformLocation(muxOverlayShader.id(), "time_f"), mux_time_f);
                glUniform2f(glGetUniformLocation(muxOverlayShader.id(), "iResolution"),
                            static_cast<float>(win->w), static_cast<float>(win->h));
                glUniform1f(glGetUniformLocation(muxOverlayShader.id(), "alpha"), 1.0f);
                muxOverlaySprite.draw(muxDummyTex, 0, 0, win->w, win->h);
                mux_time_f += 0.016f;
            }
            if (overlayFont.handle().has_value()) {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                win->text.setColor({255, 255, 255, 255});
                win->text.printText_Blended(overlayFont, 10, 10, "Muxing audio...");
                glDisable(GL_BLEND);
            }
            return;
        }

        if (duration_limit > 0.0 && media_timeline_started &&
            (writer.is_open() || png_video_mode) && writerRunning) {
            double time_passed = 0.0;
            bool has_media_clock = false;
            if (filename.empty() && graphic.empty() && !no_drop_mode &&
                fps > 0.0) {
                time_passed = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() -
                                  media_timeline_start_time)
                                  .count();
                has_media_clock = true;
            }
#ifdef AUDIO_ENABLED
            if (file_audio_mode && file_audio_has_output_clock()) {
                time_passed = file_audio_playback_time();
                has_media_clock = true;
            }
#endif
            if (!has_media_clock && fps > 0.0) {
                frames_proc++;
                time_passed = static_cast<double>(frames_proc) / fps;
            }
            if (time_passed >= duration_limit) {
                if (silent_mode && !graphic.empty()) {
                    emitSilentGraphicsProgress(true);
                } else if (silent_mode && !filename.empty()) {
                    emitSilentVideoProgress(true);
                }
                mx::system_out << "acmx2: Duration limit reached (" << duration_limit << "s), stopping recording...\n";
                fflush(stdout);
                running = false;
            }
        }

        if (max_size_limit_bytes > 0.0 && writer.is_open() && writerRunning && !ofilename.empty()) {
            struct stat out_stat{};
            if (::stat(ofilename.c_str(), &out_stat) == 0) {
                const double current_size = static_cast<double>(out_stat.st_size);
                if (current_size > max_size_limit_bytes) {
                    mx::system_out << "acmx2: Max size reached ("
                                   << std::fixed << std::setprecision(2) << max_size_limit_mb
                                   << " MB), stopping recording...\n";
                    fflush(stdout);
                    running = false;
                }
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

        bool received_source_frame = false;
        bool file_audio_clock_controls_video = false;
        bool source_frame_uses_timeline_clock = false;
        uint64_t source_timeline_frame = 0;
        bool stop_after_recording_drain = false;
        if (!isPaused && !isFrozen) {
            if (!graphic.empty()) {
                newFrame = graphic_frame.clone();
                cv::flip(newFrame, newFrame, 0);
                received_source_frame = !newFrame.empty();
            } else if (filename.empty()) {
                std::unique_lock<std::mutex> lock(captureQueueMutex);
                if (!captureQueue.empty()) {
                    newFrame = std::move(captureQueue.front());
                    captureQueue.pop();
                    received_source_frame = !newFrame.empty();
                    if (received_source_frame && writer.is_open() &&
                        !no_drop_mode && fps > 0.0) {
                        // Live webcam recording follows wall-clock time. If
                        // rendering or encoding falls behind, explicit PTS
                        // gaps keep the video duration aligned with live
                        // audio instead of slowing playback down.
                        source_frame_uses_timeline_clock = true;
                        if (media_timeline_started) {
                            const double elapsed_seconds =
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    media_timeline_start_time)
                                    .count();
                            source_timeline_frame = static_cast<uint64_t>(
                                std::floor(std::max(0.0, elapsed_seconds) *
                                           fps));
                        }
                    }
                }
            } else {
                bool audio_clock_available = false;
                double audio_clock_time = 0.0;
#ifdef AUDIO_ENABLED
                audio_clock_available = file_audio_mode && file_audio_has_output_clock();
                file_audio_clock_controls_video = audio_clock_available;
                if (audio_clock_available) {
                    audio_clock_time = file_audio_playback_time();
                }
#endif

                uint64_t target_frame = decoded_video_frame_count;
                if (audio_clock_available && media_timeline_started && fps > 0.0) {
                    target_frame = static_cast<uint64_t>(
                        std::floor(std::max(0.0, audio_clock_time) * fps));
                }

                const bool frame_due =
                    !audio_clock_available || !media_timeline_started ||
                    target_frame >= decoded_video_frame_count;
                if (frame_due) {
                    const uint64_t frames_to_decode =
                        audio_clock_available && media_timeline_started
                            ? target_frame - decoded_video_frame_count + 1
                            : 1;
                    bool read_ok = false;
                    bool decoded_any_frame = false;

                    auto read_next_frame = [&](bool discard) {
                        bool ok = false;
                        if (use_ffmpeg_reader) {
                            ok = discard
                                     ? ffmpeg_reader.skip()
                                 : input_is_hdr
                                     ? ffmpeg_reader.readHdr(hdr_frame_mat)
                                     : ffmpeg_reader.read(newFrame);
                            if (!ok && repeat) {
                                mx::system_out << "acmx2: video loop...\n";
                                if (ffmpeg_reader.seekStart()) {
                                    ok = discard
                                             ? ffmpeg_reader.skip()
                                         : input_is_hdr
                                             ? ffmpeg_reader.readHdr(hdr_frame_mat)
                                             : ffmpeg_reader.read(newFrame);
                                }
                            }
                        } else {
                            ok = discard ? cap.grab() : cap.read(newFrame);
                            if (!ok && repeat) {
                                mx::system_out << "acmx2: video loop...\n";
                                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                                ok = discard ? cap.grab()
                                             : cap.read(newFrame);
                            }
                        }
                        if (ok) {
                            decoded_video_frame_count++;
                        }
                        return ok;
                    };

                    for (uint64_t frame = 0; frame < frames_to_decode; ++frame) {
                        const bool discard =
                            frame + 1 < frames_to_decode;
                        read_ok = read_next_frame(discard);
                        if (!read_ok) {
                            break;
                        }
                        decoded_any_frame = true;
                    }

                    if (!read_ok) {
                        if (decoded_any_frame) {
                            // The audio clock jumped beyond the final source
                            // timestamp. Render the last successfully decoded
                            // frame, then detect EOF again on the next draw so
                            // the asynchronous PBO can be drained.
                            read_ok = true;
                        } else if (repeat) {
                            mx::system_out << "acmx2: cannot read after looping.\n";
                        }
                        if (!read_ok) {
                            const bool can_drain_recording =
                                !input_is_hdr && recording_pbo_primed &&
                                (writer.is_open() || png_video_mode ||
                                 generate_mode);
                            if (can_drain_recording) {
                                stop_after_recording_drain = true;
                            } else {
                                if (silent_mode) {
                                    std::cout << "\n";
                                }
                                running = false;
                                finished = true;
                                return;
                            }
                        }
                    }

                    if (read_ok) {
                        received_source_frame =
                            input_is_hdr ? !hdr_frame_mat.empty() : !newFrame.empty();
                        source_frame_uses_timeline_clock = audio_clock_available;
                        source_timeline_frame =
                            audio_clock_available
                                ? std::min(target_frame,
                                           decoded_video_frame_count - 1)
                                : decoded_video_frame_count - 1;
                        if (!newFrame.empty())
                            cv::flip(newFrame, newFrame, 0);
                    }
                }
                // HDR path: leave @c hdr_frame_mat top-down. The SDR path
                // pre-flips because its shader chain produces a Y-flipped
                // readback that the CPU loop then re-flips; the HDR path
                // has an additional sprite.draw pass (hdr_encode) before
                // the PBO-free readback, and empirically the single CPU
                // flip in @c hdrReadback's caller is all that is needed
                // to deliver top-down rows to the HEVC encoder.
            }
        }

        if (received_source_frame && !isFrozen) {
            if (input_is_hdr) {
                rotate_frame(hdr_frame_mat);
            } else {
                rotate_frame(newFrame);
            }
        }
#ifdef ACMX2_WITH_CUDA
        bool onnxGpuFrameReady = false;
#endif
#ifdef ACMX2_WITH_DNN
        // Human segmentation pass (PPHS):
        //  * Default (--human only): isolate the person, blacken background,
        //    then run the entire shader pipeline on the cutout.
        //  * --human --background: leave @c newFrame untouched so shaders
        //    process the full frame; build an RGBA overlay (original frame
        //    + hardened alpha mask) and composite it on top of the shader
        //    output via a final GL blend pass below.
        human_overlay_ready = false;
        if (human_seg_model && !isFrozen && !input_is_hdr && !newFrame.empty()) {
            try {
                cv::Mat mask = human_seg_model->infer(newFrame);
                if (human_background_only) {
                    cv::Mat alpha8 = ac_dnn::hardenedAlphaMask(newFrame, mask, human_black_point, human_white_point);
                    if (!alpha8.empty() && alpha8.size() == newFrame.size()) {
                        cv::Mat rgba;
                        cv::cvtColor(newFrame, rgba, cv::COLOR_BGR2RGBA);
                        cv::insertChannel(alpha8, rgba, 3);

                        // Lazy-allocate / resize the overlay GL texture.
                        if (human_overlay_tex == 0) {
                            glGenTextures(1, &human_overlay_tex);
                            glBindTexture(GL_TEXTURE_2D, human_overlay_tex);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                            human_overlay_w = 0;
                            human_overlay_h = 0;
                        }
                        glBindTexture(GL_TEXTURE_2D, human_overlay_tex);
                        if (human_overlay_w != rgba.cols || human_overlay_h != rgba.rows) {
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows,
                                         0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.ptr());
                            human_overlay_w = rgba.cols;
                            human_overlay_h = rgba.rows;
                        } else {
                            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                            rgba.cols, rgba.rows,
                                            GL_RGBA, GL_UNSIGNED_BYTE, rgba.ptr());
                        }
                        glBindTexture(GL_TEXTURE_2D, 0);
                        human_overlay_ready = true;

                        // Remove the person from newFrame so the shader chain
                        // processes only the background.  The GL blend pass
                        // below composites the original person back on top
                        // using straight-alpha blending.
                        cv::Mat inverseAlpha;
                        cv::subtract(cv::Scalar::all(255), alpha8, inverseAlpha);
                        cv::Mat inverseBgr;
                        cv::cvtColor(inverseAlpha, inverseBgr, cv::COLOR_GRAY2BGR);
                        cv::multiply(newFrame, inverseBgr, newFrame,
                                     1.0 / 255.0, CV_8UC3);
                    }
                } else {
                    cv::Mat isolated = ac_dnn::isolateBody(newFrame, mask, human_black_point, human_white_point);
                    if (!isolated.empty()) {
                        newFrame = isolated;
                    }
                }
            } catch (const cv::Exception &e) {
                mx::system_err << "acmx2: PPHS inference error: " << e.what() << "\n";
            }
        }
        // Edge detection pass (Dexined): replaces newFrame with a 3-channel
        // edge map so the shader chain renders on top of the edge output.
        if (edge_det_model && !isFrozen && !input_is_hdr && !newFrame.empty()) {
            try {
                cv::Mat edges;
                edge_det_model->processFrame(newFrame, edges);
                if (!edges.empty()) {
                    if (edges.channels() == 1)
                        cv::cvtColor(edges, newFrame, cv::COLOR_GRAY2BGR);
                    else
                        newFrame = edges;
                }
            } catch (const cv::Exception &e) {
                mx::system_err << "acmx2: Dexined inference error: " << e.what() << "\n";
            }
        }
        // Generic ONNX pass. When no CUDA filter needs a host frame, keep the
        // final normalize/resize/colour conversion in VRAM and hand RGBA
        // directly to TextureUploader below. GPU filters currently own a
        // host-input frame history, so retain the BGR CPU path for that case.
        if (onnx_proc_model && !isFrozen && !input_is_hdr && !newFrame.empty()) {
            try {
#ifdef ACMX2_WITH_CUDA
                const bool gpuFiltersNeedHostFrame =
                    gpu_filter_enabled && !gpu_filters.empty() && gpu_frame_buffer;
                if (!gpuFiltersNeedHostFrame)
                    onnxGpuFrameReady =
                        onnx_proc_model->procGpu(newFrame, onnxGpuOutput);
                if (!onnxGpuFrameReady)
#endif
                {
                    cv::Mat onnx_out;
                    onnx_proc_model->proc(newFrame, onnx_out);
                    if (!onnx_out.empty())
                        newFrame = onnx_out;
                }
            } catch (const cv::Exception &e) {
                mx::system_err << "acmx2: OnnxWrapper inference error: " << e.what() << "\n";
            }
        }
#endif
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
            const bool multipass_uses_cache = shader_pass_enabled && [&]() {
                for (int idx : shader_pass_list) {
                    if (idx >= 0 && library.isCache2D(static_cast<size_t>(idx)))
                        return true;
                }
                return false;
            }();
            if (texture_cache && (library.isCache() || multipass_uses_cache)) {
                static int hdr_counter = 0;
                if (frame_cache.size() == 0) {
                    if (received_source_frame) {
                        frame_cache.pushFromFBO(hdr_linear_video_fbo,
                                                win->w, win->h);
                        hdr_counter = 0;
                    }
                } else if (++hdr_counter > cache_delay) {
                    if (cache_warmup_frames <= 0) {
                        // GPU->GPU copy of the freshly decoded linear-light
                        // frame into the next ring slot. No CPU readback.
                        frame_cache.pushFromFBO(hdr_linear_video_fbo,
                                                win->w, win->h);
                    }
                    hdr_counter = 0;
                }
                if (frame_cache.isFull()) {
                    if (texture_cache_array) {
                        library.setHistoryHead(frame_cache.oldestLayer());
                        library.setUniform("history", 0);
                        glActiveTexture(GL_TEXTURE1);
                        glBindTexture(GL_TEXTURE_2D_ARRAY,
                                      frame_cache.historyTexture());
                    } else {
                        const int n_bind = library.cacheSize();
                        for (int i = 0; i < n_bind; ++i) {
                            // setUniform(name, slot) routes through ProgramData
                            // and assigns BOTH `samp(i+1)` (for i<8) and
                            // `textures[i]` to texture unit i+1.
                            library.setUniform(
                                "samp" + std::to_string(i + 1), i);
                            glActiveTexture(GL_TEXTURE1 + i);
                            glBindTexture(GL_TEXTURE_2D,
                                          frame_cache.textureAt(i));
                        }
                    }
                }
            }
        } else if (!isFrozen && !newFrame.empty()) {
#ifdef ACMX2_WITH_CUDA
            if (onnxGpuFrameReady) {
                tex_uploader.update(onnxGpuOutput);
                if (camera_texture != tex_uploader.textureID)
                    camera_texture = tex_uploader.textureID;
            } else if (gpu_filter_enabled && !gpu_filters.empty() && gpu_frame_buffer) {
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
                // If the incoming GpuMat size ever differs from the
                // uploader's current size, update() calls init() which
                // deletes the old GL texture and allocates a new one with
                // a fresh textureID. Re-sync our cached camera_texture so
                // it never dangles at a deleted name (would render black).
                if (camera_texture != tex_uploader.textureID) {
                    camera_texture = tex_uploader.textureID;
                }
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
            if (texture_cache && (library.isCache() || ([&]() {
                                      if (!shader_pass_enabled)
                                          return false;
                                      for (int idx : shader_pass_list) {
                                          if (idx >= 0 && library.isCache2D(static_cast<size_t>(idx)))
                                              return true;
                                      }
                                      return false;
                                  }()))) {
                static int counter = 0;
                if (frame_cache.size() == 0) {
                    frame_cache.pushFromTexture(camera_texture,
                                                newFrame.cols, newFrame.rows);
                    counter = 0;
                } else if (++counter > cache_delay) {
                    // Only push frames into cache after the post-load warmup period
                    if (cache_warmup_frames <= 0) {
                        // GPU->GPU copy from the camera texture that was just
                        // updated above. Avoids a second BGR->RGBA conversion
                        // and host upload per frame, which became a major
                        // bottleneck once CPU ONNX passes were added.
                        frame_cache.pushFromTexture(camera_texture,
                                                    newFrame.cols, newFrame.rows);
                    }
                    counter = 0;
                }
                if (frame_cache.isFull()) {
                    if (texture_cache_array) {
                        library.setHistoryHead(frame_cache.oldestLayer());
                        library.setUniform("history", 0);
                        glActiveTexture(GL_TEXTURE1);
                        glBindTexture(GL_TEXTURE_2D_ARRAY,
                                      frame_cache.historyTexture());
                    } else {
                        const int n_bind = library.cacheSize();
                        for (int i = 0; i < n_bind; ++i) {
                            library.setUniform(
                                "samp" + std::to_string(i + 1), i);
                            glActiveTexture(GL_TEXTURE1 + i);
                            glBindTexture(GL_TEXTURE_2D,
                                          frame_cache.textureAt(i));
                        }
                    }
                }
            }
        }
        if (received_source_frame) {
            source_frame_ready = true;
            startMediaTimelineIfReady();
        }

        // Count cache warmup in real source frames, not loading-screen draws.
        if (cache_warmup_frames > 0 && received_source_frame) {
            cache_warmup_frames--;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        glViewport(0, 0, win->w, win->h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef AUDIO_ENABLED
        if (!isFrozen && audio_is_enabled && media_timeline_started) {
            float audio_warmup = updateAudioWarmupEnvelope();
            library.setAudioWarmupEnvelope(audio_warmup);
            if (file_audio_mode) {
                const bool process_file_audio =
                    file_audio_has_output_clock() || received_source_frame;
                if (process_file_audio) {
                    file_audio_process_frame(fps, audio_engine.analyzer());
                }
                if (audio_trunc_mode && !file_audio_is_active()) {
                    mx::system_out << "acmx2: Audio file finished, stopping (--audio-trunc).\n";
                    fflush(stdout);
                    running = false;
                }
            }
            float spectrum_scale = spectrum_scale_by_sense
                                       ? (audio_engine.analyzer().sensitivity() * audio_warmup)
                                       : audio_warmup;
            spectrumTex.update(spectrum_scale);
            spectrumTex.bind();
            if (audio_buffer_count > 0) {
                spectrumHistory.update(spectrum_scale);
                spectrumHistory.bind();
                library.setSpectrumHistoryHead(spectrumHistory.newestLayer());
            }
        }
#endif

        if (!isFrozen && !library.isBypassed()) {
            library.useProgram();
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

            if (keystate[SDL_SCANCODE_PERIOD]) {
                cameraRotationSpeed += 0.5f;
                if (cameraRotationSpeed > 50.0f)
                    cameraRotationSpeed = 50.0f;
                mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                fflush(stdout);
            }
            if (keystate[SDL_SCANCODE_COMMA]) {
                cameraRotationSpeed -= 0.5f;
                if (cameraRotationSpeed < 0.5f)
                    cameraRotationSpeed = 0.5f;
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
            if (dt > 0.1f)
                dt = 0.1f;
            last3DTime = now3D;

            static float rotation = 0.0f;
            rotation = fmod(rotation + 0.5f, 360.0f);

            const Uint8 *keystate = SDL_GetKeyboardState(NULL);
            if (!oscillateScale) {

                if (keystate[SDL_SCANCODE_1]) {
                    movementSpeed += 0.1f * dt * 30.0f;
                    mx::system_out << "acmx2: camera movement speed increased: " << movementSpeed << "\n";
                    fflush(stdout);
                }

                if (keystate[SDL_SCANCODE_2]) {
                    movementSpeed -= 0.1f * dt * 30.0f;
                    mx::system_out << "acmx2: camera movement speed decreased: " << movementSpeed << "\n";
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
                    if (modelRenderScale < 0.05f)
                        modelRenderScale = 0.05f;
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
                }
                if (keystate[SDL_SCANCODE_S]) {
                    cameraPitch -= cameraRotationSpeed * 0.33f * dt * 30.0f;
                }
                cameraPitch = fmod(cameraPitch, 360.0f);
                if (cameraPitch < 0.0f)
                    cameraPitch += 360.0f;
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
            if (!viewRotationActive) {
                const float pitch = glm::radians(cameraPitch);
                const float yaw = glm::radians(cameraYaw);
                cameraUp = glm::vec3(-sin(pitch) * cos(yaw),
                                     cos(pitch),
                                     -sin(pitch) * sin(yaw));
            }
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

                ensurePassTargets(win);

                GLuint inputTex = input_is_hdr ? hdr_linear_video_texture : camera_texture;
                int pingpong = 0;
                bool pass_applied = false;
                for (size_t i = 0; i < shader_pass_list.size(); ++i) {
                    int shader_idx = shader_pass_list[i];
                    if (shader_idx >= 0 && shader_idx < static_cast<int>(library.size2d())) {
                        gl::ShaderProgram *pass_shader = library.getShader2D(shader_idx);
                        if (pass_shader) {
                            bool applied = false;
                            if (library.isCompute(static_cast<size_t>(shader_idx))) {
                                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                                applied = runComputePass(
                                    win, static_cast<size_t>(shader_idx), inputTex,
                                    passTexture[pingpong]);
                            } else {
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
                                bindPassHistoryTextures(
                                    static_cast<size_t>(shader_idx));
                                sprite.setShader(pass_shader);
                                sprite.setName("samp");
                                sprite.draw(inputTex, 0, 0, win->w, win->h);
                                applied = true;
                            }
                            if (applied) {
                                pass_applied = true;
                                inputTex = passTexture[pingpong];
                                pingpong = 1 - pingpong;
                            }
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

            const bool active_compute = !library.isBypassed() &&
                                        library.isCompute(library.index());
            if (active_compute) {
                ensurePassTargets(win);
                const int output_index =
                    textureForMesh == passTexture[0] ? 1 : 0;
                if (runComputePass(win, library.index(), textureForMesh,
                                   passTexture[output_index])) {
                    textureForMesh = passTexture[output_index];
                }
                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                glViewport(0, 0, win->w, win->h);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
            }

            gl::ShaderProgram *activeShader;
            if (library.isBypassed() || active_compute) {
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
                ensurePassTargets(win);
                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);

                GLuint inputTex = input_is_hdr ? hdr_linear_video_texture : camera_texture;
                int pingpong = 0;
                bool pass_applied = false;

                for (size_t i = 0; i < shader_pass_list.size(); ++i) {
                    int shader_idx = shader_pass_list[i];
                    if (shader_idx >= 0 && shader_idx < static_cast<int>(library.size())) {
                        gl::ShaderProgram *pass_shader = library.getShader(shader_idx);
                        if (pass_shader) {
                            bool applied = false;
                            if (library.isCompute(static_cast<size_t>(shader_idx))) {
                                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                                applied = runComputePass(
                                    win, static_cast<size_t>(shader_idx), inputTex,
                                    passTexture[pingpong]);
                            } else {
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
                                bindPassHistoryTextures(
                                    static_cast<size_t>(shader_idx));
                                sprite.setShader(pass_shader);
                                sprite.setName("samp");
                                sprite.draw(inputTex, 0, 0, win->w, win->h);
                                applied = true;
                            }
                            if (applied) {
                                pass_applied = true;
                                inputTex = passTexture[pingpong];
                                pingpong = 1 - pingpong;
                            }
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

            const bool active_compute = !library.isBypassed() &&
                                        library.isCompute(library.index());
            if (active_compute) {
                ensurePassTargets(win);
                const int output_index =
                    textureForSprite == passTexture[0] ? 1 : 0;
                if (runComputePass(win, library.index(), textureForSprite,
                                   passTexture[output_index])) {
                    textureForSprite = passTexture[output_index];
                }
                glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
                glViewport(0, 0, win->w, win->h);
                glClear(GL_COLOR_BUFFER_BIT);
            }

            gl::ShaderProgram *activeShader;
            if (library.isBypassed() || active_compute) {
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

#ifdef ACMX2_WITH_DNN
        // --human --background : composite the original person on top of the
        // shaded captureFBO using the hardened alpha mask. The overlay
        // texture stores the original frame in RGB and the cleaned mask in
        // its alpha channel, so a single straight-alpha blend reproduces the
        // person over whatever the shader chain just rendered to captureFBO.
        if (human_seg_model && human_background_only && human_overlay_ready &&
            human_overlay_tex != 0 && !input_is_hdr) {
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glViewport(0, 0, win->w, win->h);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            fshader.useProgram();
            fshader.setUniform("mv_matrix", glm::mat4(1.0f));
            fshader.setUniform("proj_matrix", glm::mat4(1.0f));
            sprite.setShader(&fshader);
            sprite.setName("samp");
            sprite.draw(human_overlay_tex, 0, 0, win->w, win->h);
            glDisable(GL_BLEND);
        }
#endif

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
        if (display_filter && (writer.is_open() || png_video_mode) && waterFont.handle().has_value()) {
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
                    if (i > 0)
                        mpLine += ", ";
                    std::string n = library.getShaderNameByIndex(shader_pass_list[i]);
                    mpLine += n.empty() ? std::to_string(shader_pass_list[i]) : n;
                }
                win->text.printText_Solid(waterFont, 10, dfY, mpLine);
                dfY += lineH;
            }
            if (gpu_filter_enabled && !gpu_filters.empty()) {
                std::string gpuLine = "GPU: ";
                for (size_t i = 0; i < gpu_filters.size(); ++i) {
                    if (i > 0)
                        gpuLine += ", ";
                    gpuLine += gpu_filters[i].name;
                }
                win->text.printText_Solid(waterFont, 10, dfY, gpuLine);
                dfY += lineH;
            }
            glDisable(GL_BLEND);
            watermarkY = dfY;
        }
        if (enableWatermark && (writer.is_open() || png_video_mode) && waterFont.handle().has_value()) {
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

        const bool normal_output_frame_due =
            media_timeline_started &&
            (!file_audio_clock_controls_video || received_source_frame ||
             stop_after_recording_drain);
        bool needWriter = (((writer.is_open() || png_video_mode || generate_mode) &&
                            normal_output_frame_due) ||
                           snapshot_state > 0 || hdr_snapshot_state > 0 || raw_snapshot_state > 0 ||
                           tiff_snapshot_state > 0) &&
                          !isFrozen;

        bool has_snapshot_request = (snapshot_state > 0);
        bool has_hdr_snapshot_request = (hdr_snapshot_state > 0);
        bool has_raw_snapshot_request = (raw_snapshot_state > 0);
        bool has_tiff_snapshot_request = (tiff_snapshot_state > 0);
        if (needWriter && input_is_hdr && (writer.is_open() || png_video_mode || generate_mode || has_snapshot_request || has_hdr_snapshot_request || has_raw_snapshot_request || has_tiff_snapshot_request)) {
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

            if (writer.is_open() || png_video_mode || generate_mode || has_hdr_snapshot_request || has_raw_snapshot_request) {
                FrameData fd;
                fd.pixels = pixels;
                fd.width = win->w;
                fd.height = win->h;
                fd.isHdr = true;
                fd.hdrTrc = input_hdr_trc;
                fd.isSnapshot = has_hdr_snapshot_request;
                fd.isWebPSnapshot = has_hdr_snapshot_request;
                fd.isRawSnapshot = has_raw_snapshot_request;
                fd.usesTimelineClock =
                    source_frame_uses_timeline_clock && !has_hdr_snapshot_request &&
                    !has_raw_snapshot_request;
                fd.timelineFrame = source_timeline_frame;

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
                if (snapshot_state == 1)
                    snapshot_state = 2;
                if (hdr_snapshot_state == 1)
                    hdr_snapshot_state = 2;
                if (raw_snapshot_state == 1)
                    raw_snapshot_state = 2;
                if (tiff_snapshot_state == 1)
                    tiff_snapshot_state = 2;
            } else {
                bool is_snapshot_frame = (snapshot_state == 2);
                bool is_webp_snapshot_frame = (hdr_snapshot_state == 2);
                bool is_raw_snapshot_frame = (raw_snapshot_state == 2);
                bool is_tiff_snapshot_frame = (tiff_snapshot_state == 2);
                const bool has_normal_output =
                    normal_output_frame_due && (writer.is_open() || png_video_mode || generate_mode);
                const bool previous_recording_frame_ready = recording_pbo_primed;

                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pboIndex]);
                glBindTexture(GL_TEXTURE_2D, fboTexture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

                if (has_normal_output) {
                    recording_pbo_uses_timeline_clock[pboIndex] =
                        source_frame_uses_timeline_clock;
                    recording_pbo_timeline_frame[pboIndex] =
                        source_timeline_frame;
                    recording_pbo_primed = true;
                }

                if ((has_normal_output && previous_recording_frame_ready) || is_snapshot_frame ||
                    is_webp_snapshot_frame || is_raw_snapshot_frame || is_tiff_snapshot_frame) {
                    bool used_zero_copy = false;

#ifdef ACMX2_WITH_CUDA
                    if (writer.is_open() && !generate_mode &&
                        !is_snapshot_frame && !is_webp_snapshot_frame &&
                        !is_raw_snapshot_frame && !is_tiff_snapshot_frame &&
                        recordCudaPboResources[pboNextIndex]) {
                        cudaGraphicsResource *resource = recordCudaPboResources[pboNextIndex];
                        void *devPtr = nullptr;
                        size_t mappedBytes = 0;

                        CHECK_CUDA(cudaGraphicsMapResources(1, &resource, 0));
                        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedBytes, resource));

                        const size_t requiredBytes = static_cast<size_t>(win->w) * static_cast<size_t>(win->h) * 4;
                        if (devPtr && mappedBytes >= requiredBytes) {
                            if (recording_pbo_uses_timeline_clock[pboNextIndex]) {
                                used_zero_copy =
                                    writer.write_cuda_rgba_at_pts(
                                        devPtr, static_cast<int>(win->w) * 4,
                                        static_cast<int64_t>(
                                            recording_pbo_timeline_frame
                                                [pboNextIndex]),
                                        true);
                            } else {
                                used_zero_copy =
                                    writer.write_cuda_rgba(
                                        devPtr, static_cast<int>(win->w) * 4,
                                        true);
                            }
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
                            fd.usesTimelineClock =
                                recording_pbo_uses_timeline_clock[pboNextIndex] &&
                                !fd.isSnapshot && !fd.isRawSnapshot &&
                                !fd.isTiffSnapshot;
                            fd.timelineFrame =
                                recording_pbo_timeline_frame[pboNextIndex];

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

        if (stop_after_recording_drain) {
            // The EOF draw consumed the last valid PBO. The replacement
            // readback contains the already-rendered final frame, not a new
            // source frame, so it must not be flushed again at shutdown.
            recording_pbo_primed = false;
            if (silent_mode) {
                emitSilentVideoProgress(true);
                std::cout << "\n";
            }
            running = false;
            finished = true;
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
#ifdef AUDIO_ENABLED
            if (file_audio_mode && file_audio_has_output_clock()) {
                const std::string trackPath = file_audio_current_source_path();
                const std::string trackName =
                    std::filesystem::path(trackPath).filename().string();
                if (!trackName.empty()) {
                    win->text.setColor({255, 0, 255, 255});
                    win->text.printText_Blended(
                        overlayFont, 10, overlayY, "Track: " + trackName);
                    overlayY += 30;
                }
            }
#endif
            if (gpu_filter_enabled && !gpu_filters.empty()) {
                win->text.setColor({255, 0, 255, 255});
                std::string gpuLine = "GPU: ";
                for (size_t i = 0; i < gpu_filters.size(); ++i) {
                    if (i > 0)
                        gpuLine += ", ";
                    gpuLine += gpu_filters[i].name;
                }
                win->text.printText_Blended(overlayFont, 10, overlayY, gpuLine);
                overlayY += 30;
            }
            if (autopilot_enabled) {
                const int activeInterval = autopilot_random_interval ? autopilot_interval_frames : autopilot_frames;
                const int remainingFrames = std::max(0, activeInterval - autopilot_counter);
                const int playlistCount = activePlaylistSize();
                std::ostringstream autopilotLine;
                if (autopilot_random_interval) {
                    autopilotLine << "Autopilot "
                                  << (autopilot_sequential ? "seq" : "rnd")
                                  << " [4-" << std::max(4, autopilot_random_timeout)
                                  << "] cur=" << activeInterval
                                  << " next=" << remainingFrames << "f";
                } else {
                    autopilotLine << "Autopilot "
                                  << (autopilot_sequential ? "seq" : "rnd")
                                  << " every " << activeInterval
                                  << "f next=" << remainingFrames << "f";
                }
                if (playlistCount > 0) {
                    autopilotLine << " idx=" << (playlist_index + 1) << "/" << playlistCount;
                }
                win->text.setColor({0, 255, 255, 255});
                win->text.printText_Blended(overlayFont, 10, overlayY, autopilotLine.str());
                overlayY += 30;
            }
            if (!crossfadeShaderNames.empty()) {
                int n = static_cast<int>(crossfadeShaderNames.size());
                std::string xfadeLine = "XFade [" + std::to_string(crossfadeShaderIndex + 1) + "/" +
                                        std::to_string(n) + "]: " +
                                        crossfadeShaderNames[crossfadeShaderIndex];
                win->text.setColor({255, 200, 0, 255});
                win->text.printText_Blended(overlayFont, 10, overlayY, xfadeLine);
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
            if (silent_mode) {
                emitSilentGraphicsProgress(false);
            } else if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() >= 250) {
                std::string timeStr = getTimeString();
                int64_t currentFrames = getFrameCount();

                std::ostringstream stream;
                stream << "ACMX2 - Graphics Mode - "
                       << timeStr
                       << " [" << currentFrames << " frames]";
                appendRecordingTitleSuffix(stream);
                win->setWindowTitle(stream.str());
                lastUpdate = now;
            }

        } else if (!filename.empty()) {
            if (use_ffmpeg_reader) {
                frame_counter = static_cast<unsigned int>(std::max<int64_t>(0, ffmpeg_reader.getCurrentFrame()));
            } else if (cap.isOpened()) {
                frame_counter = static_cast<unsigned int>(cap.get(cv::CAP_PROP_POS_FRAMES));
            }

            if (silent_mode &&
                (totalFrames > 0.0 ||
                 (duration_limit > 0.0 && fps > 0.0))) {
                emitSilentVideoProgress(false);
            } else if (silent_mode) {
                // Fallback: input reports unknown frame count (e.g. some MKV
                // / streaming containers). No percentage possible, so emit
                // an elapsed-style progress line every 500 ms with what we
                // do know: current frame number, frames written, elapsed
                // time based on the input FPS.
                static auto lastProgressEmitUnk = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressEmitUnk).count() >= 500) {
                    lastProgressEmitUnk = now;
                    int64_t frames_written = png_video_mode ? static_cast<int64_t>(png_video_frame_counter.load()) : (writer.is_open() ? writer.get_frame_count() : 0);
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
                              << std::setfill(' ');
                    appendSilentProgressFileSize(std::cout);
                    std::cout << "\n"
                              << std::flush;
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
                appendRecordingTitleSuffix(stream);
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
                appendRecordingTitleSuffix(stream);
                win->setWindowTitle(stream.str());
                lastUpdate = now;
            }
        }
        const bool autopilotTickEnabled = autopilot_random_interval || autopilot_frames > 0;
        if (playlist_enabled && autopilot_enabled && autopilotTickEnabled) {
            const int interval = autopilot_random_interval ? autopilot_interval_frames : autopilot_frames;
            if (++autopilot_counter >= interval) {
                autopilot_counter = 0;
                if (autopilot_sequential && !autopilot_random_interval) {
                    autopilotSequentialAdvance(win);
                } else {
                    autopilotRandomSwitch(win);
                }
                if (autopilot_random_interval) {
                    resetAutopilotInterval();
                }
            }
        }
        frame_counter++;
    }

    /**
     * @brief Emit progress for headless video rendering.
     *
     * When a duration limit is active, its frame boundary becomes the
     * progress total if it occurs before source EOF. This makes a capped
     * silent render finish at 100% instead of stopping partway through the
     * input video's progress range.
     *
     * @param complete Force the final 100% progress update.
     */
    void emitSilentVideoProgress(bool complete) {
        if (!silent_mode || filename.empty() || fps <= 0.0) {
            return;
        }

        uint64_t expected_frames = totalFrames > 0.0
                                       ? static_cast<uint64_t>(std::ceil(totalFrames))
                                       : 0;
        if (duration_limit > 0.0) {
            const uint64_t duration_frames = std::max<uint64_t>(
                1, static_cast<uint64_t>(std::ceil(duration_limit * fps)));
            if (repeat || expected_frames == 0) {
                expected_frames = duration_frames;
            } else {
                expected_frames = std::min(expected_frames, duration_frames);
            }
        }
        if (expected_frames == 0) {
            return;
        }

        const uint64_t current_frame =
            duration_limit > 0.0 && repeat
                ? static_cast<uint64_t>(frames_proc) + 1
                : static_cast<uint64_t>(frame_counter);
        const uint64_t processed_frames = complete
                                              ? expected_frames
                                              : std::min(current_frame,
                                                         expected_frames);
        int current_percent = static_cast<int>(
            (static_cast<double>(processed_frames) / expected_frames) * 100.0);
        if (!complete) {
            current_percent = std::min(current_percent, 99);
        }

        const auto now = std::chrono::steady_clock::now();
        const bool percent_changed = current_percent > last_progress_percent;
        const bool time_elapsed =
            last_video_progress_emit.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_video_progress_emit)
                    .count() >= 500;
        if (!complete && !percent_changed && !time_elapsed) {
            return;
        }

        last_progress_percent = current_percent;
        last_video_progress_emit = now;
        const int64_t frames_written =
            png_video_mode
                ? static_cast<int64_t>(png_video_frame_counter.load())
                : (writer.is_open() ? writer.get_frame_count() : 0);
        const double elapsed_secs = static_cast<double>(processed_frames) / fps;
        const uint64_t hours = static_cast<uint64_t>(elapsed_secs / 3600.0);
        const uint64_t minutes = static_cast<uint64_t>(elapsed_secs / 60.0) % 60;
        const uint64_t seconds = static_cast<uint64_t>(elapsed_secs) % 60;

        std::cout << "acmx2: [" << std::setw(3) << current_percent << "%] "
                  << "Frame " << processed_frames << "/" << expected_frames
                  << " | Written: " << frames_written
                  << " | Time: " << std::setfill('0') << std::setw(2) << hours << ":"
                  << std::setw(2) << minutes << ":" << std::setw(2) << seconds
                  << std::setfill(' ');
        appendSilentProgressFileSize(std::cout);
        std::cout << "\n"
                  << std::flush;
    }

    /**
     * @brief Emit bounded progress updates for duration-limited headless graphics rendering.
     *
     * Reports after approximately one second of output frames or 500 ms of
     * wall time, whichever happens first. The duration limit provides the
     * expected frame count and allows a final 100% update.
     *
     * @param complete Force the final 100% progress update.
     */
    void emitSilentGraphicsProgress(bool complete) {
        if (!silent_mode || graphic.empty() || duration_limit <= 0.0 || fps <= 0.0) {
            return;
        }

        const uint64_t expected_frames = std::max<uint64_t>(
            1, static_cast<uint64_t>(std::ceil(duration_limit * fps)));
        const uint64_t processed_frames = complete
                                              ? expected_frames
                                              : std::min<uint64_t>(
                                                    static_cast<uint64_t>(frame_counter) + 1,
                                                    expected_frames);
        const uint64_t frame_interval = std::max<uint64_t>(
            1, static_cast<uint64_t>(std::ceil(fps)));
        const auto now = std::chrono::steady_clock::now();
        const bool frame_interval_elapsed =
            processed_frames >= last_graphics_progress_frame + frame_interval;
        const bool time_interval_elapsed =
            last_graphics_progress_emit.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_graphics_progress_emit)
                    .count() >= 500;

        if (!complete && !frame_interval_elapsed && !time_interval_elapsed) {
            return;
        }

        int percent = static_cast<int>(
            (static_cast<double>(processed_frames) / expected_frames) * 100.0);
        if (!complete) {
            percent = std::min(percent, 99);
        }

        const double elapsed_secs = static_cast<double>(processed_frames) / fps;
        const uint64_t hours = static_cast<uint64_t>(elapsed_secs / 3600.0);
        const uint64_t minutes = static_cast<uint64_t>(elapsed_secs / 60.0) % 60;
        const uint64_t seconds = static_cast<uint64_t>(elapsed_secs) % 60;
        const int64_t frames_written = writer.is_open() ? writer.get_frame_count() : 0;

        std::cout << "acmx2: [" << std::setw(3) << percent << "%] "
                  << "Frame " << processed_frames << "/" << expected_frames
                  << " | Written: " << frames_written
                  << " | Time: " << std::setfill('0') << std::setw(2) << hours << ":"
                  << std::setw(2) << minutes << ":" << std::setw(2) << seconds
                  << std::setfill(' ');
        appendSilentProgressFileSize(std::cout);
        std::cout << "\n"
                  << std::flush;

        last_graphics_progress_frame = processed_frames;
        last_graphics_progress_emit = now;
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

    void appendRecordingTitleSuffix(std::ostringstream &stream) {
        if (!writer.is_open() && !png_video_mode) {
            return;
        }

        stream << " (Recording)";
#ifdef __linux__
        if (const auto file_size_bytes = getOutputFileSizeBytes(); file_size_bytes.has_value()) {
            constexpr double kBytesPerMB = 1024.0 * 1024.0;
            const double file_size_mb = static_cast<double>(*file_size_bytes) / kBytesPerMB;
            stream << " [File: " << std::fixed << std::setprecision(2)
                   << file_size_mb << " MB]";
        }
#endif
    }

#ifdef __linux__
    std::optional<uintmax_t> getOutputFileSizeBytes() const {
        if (ofilename.empty()) {
            return std::nullopt;
        }

        struct stat file_stat{};
        if (::stat(ofilename.c_str(), &file_stat) != 0) {
            return std::nullopt;
        }
        if (!S_ISREG(file_stat.st_mode)) {
            return std::nullopt;
        }

        return static_cast<uintmax_t>(file_stat.st_size);
    }
#endif

    /// @brief Append the current encoded output size to a silent progress line.
    void appendSilentProgressFileSize([[maybe_unused]] std::ostream &stream) const {
#ifdef __linux__
        const auto file_size_bytes = getOutputFileSizeBytes();
        if (!file_size_bytes.has_value()) {
            return;
        }

        constexpr double kBytesPerMB = 1024.0 * 1024.0;
        const double file_size_mb = static_cast<double>(*file_size_bytes) / kBytesPerMB;
        std::ostringstream size_stream;
        size_stream << std::fixed << std::setprecision(2) << file_size_mb;
        stream << " | Size: " << size_stream.str() << " MB";
#endif
    }

    /**
     * @brief Return the current frame count (writer count or display count).
     * @return Number of frames written if recording, else the display counter.
     */
    int64_t getFrameCount() {
        if (png_video_mode) {
            return static_cast<int64_t>(png_video_frame_counter.load());
        }
        if (writer.is_open()) {
            return writer.get_frame_count();
        }
        return static_cast<int64_t>(frame_counter);
    }

    /**
     * @brief Handle SDL keyboard events (shader navigation, mode toggles, etc.).
     *
     * Key bindings (SDL_KEYUP):
     * - Up/Down: Crossfade to the previous/next shader (or playlist entry if
     *   playlist mode is enabled).
     * - Shift+Up/Down: In playlist or autopilot mode, change the post-multipass
     *   shader without altering the current playlist position.
     * - Left/Right: Previous/next GPU CUDA filter.
     * - Space: Toggle shader bypass.
     * - P: Toggle playlist mode or pause video.
     * - L: Freeze frame (stop updating texture but keep time advancing).
     * - Z: Take a PNG snapshot (8-bit non-HDR readback when HDR input is active).
     * - 5: Take an HDR PNG snapshot (HDR mode only).
     * - T: Toggle active time.  Q: Toggle audio time.  Home: Toggle audio delta.
     * - V: Toggle view rotation (3D).  O: Oscillation.  C: Wave.
     * - X: Reset camera.  Ctrl+X: Quit immediately without audio mux/transfer.
     * - N: Toggle random autopilot crossfade.  1: Camera movement speed up (3D).  2: Camera movement speed down (3D).
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
                if (shaderLocked)
                    break;
                if ((e.key.keysym.mod & KMOD_SHIFT) &&
                    (playlist_enabled || autopilot_frames > 0)) {
                    beginCrossfade(win);
                    library.dec();
                    mx::system_out << "acmx2: Post-shader (Shift+Up): " << library.getFullShaderName() << "\n";
                    fflush(stdout);
                    if (is3d_enabled)
                        cube.setShaderProgram(library.shader());
                    sprite.setShader(library.shader());
                    updateShaderNameCache();
                    break;
                }
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
                    if (library.index() > 0) {
                        beginCrossfade(win);
                        library.dec();
                    }
                }
                if (is3d_enabled)
                    cube.setShaderProgram(library.shader());
                sprite.setShader(library.shader());
                updateShaderNameCache();
                break;
            case SDLK_DOWN:
                if (shaderLocked)
                    break;
                if ((e.key.keysym.mod & KMOD_SHIFT) &&
                    (playlist_enabled || autopilot_frames > 0)) {
                    beginCrossfade(win);
                    library.inc();
                    mx::system_out << "acmx2: Post-shader (Shift+Down): " << library.getFullShaderName() << "\n";
                    fflush(stdout);
                    if (is3d_enabled)
                        cube.setShaderProgram(library.shader());
                    sprite.setShader(library.shader());
                    updateShaderNameCache();
                    break;
                }
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
                    if (library.index() + 1 < library.size()) {
                        beginCrossfade(win);
                        library.inc();
                    }
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
                    if (!autopilot_random_interval && autopilot_frames <= 0) {
                        autopilot_frames = 300; // sensible default if user never set it
                    }
                    resetAutopilotInterval();
                }
                if (autopilot_enabled) {
                    if (autopilot_random_interval) {
                        mx::system_out << "acmx2: Autopilot enabled (random) (interval 4-"
                                       << std::max(4, autopilot_random_timeout)
                                       << " frames, current " << autopilot_interval_frames << ")\n";
                    } else {
                        mx::system_out << "acmx2: Autopilot enabled (random) (every "
                                       << autopilot_frames << " frames)\n";
                    }
                } else {
                    mx::system_out << "acmx2: Autopilot disabled\n";
                }
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
                    if (!autopilot_random_interval && autopilot_frames <= 0) {
                        autopilot_frames = 300;
                    }
                    resetAutopilotInterval();
                    if (autopilot_random_interval) {
                        mx::system_out << "acmx2: Autopilot enabled (sequential timing + random index) (interval 4-"
                                       << std::max(4, autopilot_random_timeout)
                                       << " frames, current " << autopilot_interval_frames << ")\n";
                    } else {
                        mx::system_out << "acmx2: Autopilot enabled (sequential) (every "
                                       << autopilot_frames << " frames)\n";
                    }
                }
                fflush(stdout);
                break;
            case SDLK_n:
                autopilot_random_crossfade = !autopilot_random_crossfade;
                mx::system_out << "acmx2: Random autopilot crossfade "
                               << (autopilot_random_crossfade ? "enabled" : "disabled") << "\n";
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
                float s = audio_engine.analyzer().sensitivity() + 0.1f;
                if (s > 5.0f)
                    s = 5.0f;
                audio_engine.analyzer().set_sensitivity(s);
                mx::system_out << "acmx2: Audio sensitivity increased to " << s << "\n";
                fflush(stdout);
                break;
            }
            case SDLK_DELETE: {
                float s = audio_engine.analyzer().sensitivity() - 0.1f;
                if (s < 0.1f)
                    s = 0.1f;
                audio_engine.analyzer().set_sensitivity(s);
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
            case SDLK_LEFTBRACKET:
                if (!crossfadeShaders.empty()) {
                    int n = static_cast<int>(crossfadeShaders.size());
                    crossfadeShaderIndex = (crossfadeShaderIndex - 1 + n) % n;
                    mx::system_out << "acmx2: Crossfade shader: " << crossfadeShaderNames[crossfadeShaderIndex]
                                   << " (" << (crossfadeShaderIndex + 1) << "/" << n << ")\n";
                    fflush(stdout);
                }
                break;
            case SDLK_RIGHTBRACKET:
                if (!crossfadeShaders.empty()) {
                    int n = static_cast<int>(crossfadeShaders.size());
                    crossfadeShaderIndex = (crossfadeShaderIndex + 1) % n;
                    mx::system_out << "acmx2: Crossfade shader: " << crossfadeShaderNames[crossfadeShaderIndex]
                                   << " (" << (crossfadeShaderIndex + 1) << "/" << n << ")\n";
                    fflush(stdout);
                }
                break;
            }
            break;
        }
        library.event(e);
    }

  private:
    unsigned int frame_counter = 0;
    std::string crf = "23";
    EncodeOptions encode_opts{};
    std::string prefix_path;
    std::string filename, ofilename, graphic;
    int camera_index = 0;
    std::tuple<int, std::string, int> flib;
    std::optional<cv::Size> sizev, sizec;
#ifdef AUDIO_ENABLED
    ShaderLibrary library{&audio_engine.analyzer()};
#else
    ShaderLibrary library;
#endif
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
    int pass_target_width = 0;
    int pass_target_height = 0;
    GLuint crossfadeFBO = 0;         ///< FBO used for the crossfade compositing pass.
    GLuint crossfadeTexture = 0;     ///< Colour attachment of @c crossfadeFBO (blended output).
    GLuint crossfadePrevTexture = 0; ///< Snapshot of the previous frame used as the blend source.

    // --- HDR-mode GL resources ----------------------------------------------
    // Only allocated / used when @ref input_is_hdr is true. SDR path is
    // entirely unchanged. Internal formats: GL_RGBA16 for textures that
    // hold PQ/HLG-encoded normalised values (source upload + pre-encode
    // readback target), GL_RGBA16F for linear-BT.2020 intermediates.
    bool input_is_hdr = false;                                ///< Active HDR pipeline for this input.
    int input_hdr_trc = 0;                                    ///< AVColorTransferCharacteristic (PQ/HLG/BT2020).
    int hdr_upload_tex_w = 0;                                 ///< Current GL size of @ref camera_texture in HDR upload mode.
    int hdr_upload_tex_h = 0;                                 ///< Current GL size of @ref camera_texture in HDR upload mode.
    int hdr_resource_w = 0;                                   ///< Width of HDR intermediate/encoded textures.
    int hdr_resource_h = 0;                                   ///< Height of HDR intermediate/encoded textures.
    GLuint hdr_linear_video_texture = 0;                      ///< GL_RGBA16F: PQ/HLG-decoded linear BT.2020 video.
    GLuint hdr_linear_video_fbo = 0;                          ///< FBO writing into @ref hdr_linear_video_texture.
    GLuint hdr_encoded_texture = 0;                           ///< GL_RGBA16: final PQ-re-encoded output for readback.
    GLuint hdr_encoded_fbo = 0;                               ///< FBO writing into @ref hdr_encoded_texture.
    gl::ShaderProgram hdr_decode_shader;                      ///< PQ/HLG -> linear BT.2020 fullscreen pass.
    gl::ShaderProgram hdr_encode_shader;                      ///< Linear BT.2020 -> PQ (or HLG) fullscreen pass.
    gl::ShaderProgram display_flip_shader;                    ///< Display shader with optional Y-flip for windowed output.
    cv::Mat hdr_frame_mat;                                    ///< Scratch CV_16UC4 RGBA frame for HDR decode.
    std::vector<gl::ShaderProgram> crossfadeShaders;          ///< Available crossfade transition shaders (cycle with [ and ]).
    std::vector<std::string> crossfadeShaderNames;            ///< Display names matching @ref crossfadeShaders by index.
    int crossfadeShaderIndex = 0;                             ///< Active index into @ref crossfadeShaders.
    float crossfadeAlpha = 1.0f;                              ///< Current blend factor (0 = old frame, 1 = new frame).
    bool crossfadeActive = false;                             ///< True while a crossfade transition is in progress.
    float crossfadeDuration = 0.5f;                           ///< Duration of the crossfade transition in seconds.
    std::chrono::steady_clock::time_point crossfadeStartTime; ///< Wall-clock time the current crossfade began.
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
    FrameCache frame_cache;
    bool texture_cache = false;
    bool texture_cache_array = false;
    int cache_delay = 1;
    int cache_warmup_frames = 0; // Frames to skip before pushing into cache after load
    bool source_frame_ready = false;
    bool media_timeline_started = false;
    std::chrono::steady_clock::time_point media_timeline_start_time{};
    uint64_t last_graphics_progress_frame = 0;
    std::chrono::steady_clock::time_point last_graphics_progress_emit{};
    std::chrono::steady_clock::time_point last_video_progress_emit{};
    bool recording_pbo_primed = false;
    uint64_t decoded_video_frame_count = 0;
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
    FrameRotation frame_rotation = FrameRotation::None;
    bool png_video_mode = false;
    std::string png_video_dir;
    std::atomic<uint64_t> png_video_frame_counter{0};
    bool generate_mode = false;
    int generate_interval = 0;
    std::string generate_dir;
    std::atomic<uint64_t> generate_all_frames{0};
    std::atomic<uint64_t> generate_saved_counter{0};
    int last_progress_percent = -1;
    bool enableWatermark = false;
    bool display_filter = false;
    int waterFontSize = 12;
    std::string watermark_text = "LostSideDead.biz"; ///< Active watermark text (overridden by --use-watermark).
    int watermark_r = 255;                           ///< Watermark color red.
    int watermark_g = 0;                             ///< Watermark color green.
    int watermark_b = 150;                           ///< Watermark color blue.

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
                // Drain any frames buffered by the OS/driver during shader
                // loading.  Without this, frames captured while the loading
                // screen was visible are the first ones pushed into the cache
                // ring, making samp1-samp8 shaders show loading-screen content.
                {
                    cv::Mat drain;
                    for (int i = 0; i < 8 && captureRunning; ++i) {
                        if (!cap.read(drain) || drain.empty())
                            break;
                    }
                }

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
     * - Passes video frames to `writer.write()` for sequential output or
     *   `writer.write_at_pts()` for clock-synchronised output.
     *
     * The thread blocks on `queueCondVar` when the queue is empty.
     * It exits when `writerRunning` is set to false and the queue is
     * drained.
     *
     * @see stopWriterThread()
     */
    void startWriterThread() {
        if (writerThread.joinable())
            return;
        writerRunning = true;
        writerThread = std::thread([this]() {
            try {
                auto write_video_frame = [this](const FrameData &fd) {
                    if (writer.is_open() || png_video_mode) {
                        if (png_video_mode) {
                            uint64_t frame_index =
                                png_video_frame_counter.fetch_add(1);
                            std::ostringstream frame_name;
                            frame_name << png_video_dir << "/frame-"
                                       << std::setfill('0') << std::setw(8)
                                       << frame_index << ".png";
                            std::string frame_path = frame_name.str();
                            if (fd.isHdr) {
                                png::SavePNG_RGBA16(frame_path.c_str(),
                                                    fd.pixels.data(), fd.width,
                                                    fd.height);
                            } else {
                                png::SavePNG_RGBA(
                                    frame_path.c_str(),
                                    const_cast<unsigned char *>(fd.pixels.data()),
                                    fd.width, fd.height);
                            }
                        } else if (fd.isHdr) {
                            if (fd.usesTimelineClock) {
                                writer.write_hdr_rgba16_at_pts(
                                    const_cast<unsigned char *>(fd.pixels.data()),
                                    static_cast<int64_t>(fd.timelineFrame));
                            } else {
                                writer.write_hdr_rgba16(
                                    const_cast<unsigned char *>(fd.pixels.data()));
                            }
                        } else if (!filename.empty() || !graphic.empty()) {
                            if (fd.usesTimelineClock) {
                                writer.write_at_pts(
                                    const_cast<unsigned char *>(fd.pixels.data()),
                                    static_cast<int64_t>(fd.timelineFrame));
                            } else {
                                writer.write(
                                    const_cast<unsigned char *>(fd.pixels.data()));
                            }
                        } else if (fd.usesTimelineClock) {
                            writer.write_at_pts(
                                const_cast<unsigned char *>(fd.pixels.data()),
                                static_cast<int64_t>(fd.timelineFrame));
                        } else {
                            writer.write(
                                const_cast<unsigned char *>(fd.pixels.data()));
                        }
                    }

                    if (generate_mode) {
                        uint64_t all_idx = generate_all_frames.fetch_add(1);
                        if (generate_interval > 0 &&
                            all_idx %
                                    static_cast<uint64_t>(generate_interval) ==
                                0) {
                            uint64_t saved_idx =
                                generate_saved_counter.fetch_add(1);
                            std::ostringstream frame_name;
                            frame_name << generate_dir << "/frame-"
                                       << std::setfill('0') << std::setw(8)
                                       << saved_idx << ".png";
                            std::string frame_path = frame_name.str();
                            if (fd.isHdr) {
                                png::SavePNG_RGBA16(frame_path.c_str(),
                                                    fd.pixels.data(), fd.width,
                                                    fd.height);
                            } else {
                                png::SavePNG_RGBA(
                                    frame_path.c_str(),
                                    const_cast<unsigned char *>(fd.pixels.data()),
                                    fd.width, fd.height);
                            }
                        }
                    }
                };

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
                    const bool is_snapshot_task =
                        fd.isSnapshot || fd.isRawSnapshot ||
                        fd.isTiffSnapshot;
                    if (is_snapshot_task) {
                        continue;
                    }

                    write_video_frame(fd);
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
     * Called when the first valid source frame starts the shared media
     * timeline so captured audio and encoded video have the same origin.
     */
    void startAudioRecordingIfNeeded() {
#ifdef AUDIO_ENABLED
        auto &recorder = audio_engine.recorder();
        if (audio_is_enabled && !file_audio_mode && !audio_record_file.empty() &&
            !recorder.is_recording()) {
            if (!recorder.start(audio_record_file,
                                audio_engine.analyzer().sample_rate(),
                                audio_engine.input_channels())) {
                mx::system_err << "acmx2: Error could not start audio recording to: " << audio_record_file << "\n";
            }
        }
#endif
    }

    /**
     * @brief Start playback and recording timing on the first valid source frame.
     *
     * Writer setup, shader compilation, and camera startup can all render
     * before an input frame exists. This one-shot gate prevents those draws
     * from advancing file audio or entering the video writer.
     */
    void startMediaTimelineIfReady() {
        if (media_timeline_started || !source_frame_ready) {
            return;
        }

        media_timeline_started = true;
        media_timeline_start_time = std::chrono::steady_clock::now();
        frames_proc = 0;
#ifdef AUDIO_ENABLED
        resetAudioWarmupEnvelope();
        startAudioRecordingIfNeeded();
#endif
        mx::system_out << "acmx2: Media timeline started on first source frame\n";
        fflush(stdout);
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
               (audio_engine.recorder().is_recording() || std::filesystem::exists(audio_record_file));
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
        auto &recorder = audio_engine.recorder();
        if (!recorder.is_recording() && !std::filesystem::exists(audio_record_file)) {
            mx::system_out << "acmx2: recorded audio file not found, skipping recorded-audio mux: " << audio_record_file << "\n";
            fflush(stdout);
            return;
        }
        if (recorder.is_recording())
            recorder.stop();
        std::string out_ext = std::filesystem::path(ofilename).extension().string();
        if (out_ext.empty())
            out_ext = ".mp4";
        std::string tmp_out = ofilename + ".tmp" + out_ext;
        bool is_mp4_like = (out_ext == ".mp4" || out_ext == ".MP4" || out_ext == ".mov" || out_ext == ".MOV" || out_ext == ".m4v" || out_ext == ".M4V");
        int64_t fc = writer.get_frame_count();
        if (fc <= 0) {
            mx::system_err << "acmx2: no encoded video frames; skipping recorded-audio mux for: " << ofilename << "\n";
            fflush(stderr);
            return;
        }
        double video_duration = (fps > 0.0 && fc > 0) ? static_cast<double>(fc) / fps : 0.0;
        // Sequential video output may need a final duration correction when
        // the webcam delivered fewer frames than its configured FPS. The
        // normal live webcam path already carries wall-clock PTS, so rescaling
        // that stream would destroy its synchronization; only trim its audio
        // to the timestamped video duration.
        const double audio_duration = recorder.duration_seconds();
        double itsscale = 1.0;
        const bool timestamped_webcam =
            filename.empty() && graphic.empty() && !no_drop_mode && fps > 0.0;
        if (!timestamped_webcam && video_duration > 0.0 &&
            audio_duration > 0.0) {
            const double s = audio_duration / video_duration;
            if (s >= 0.5 && s <= 2.0) {
                itsscale = s;
            }
        }
        const bool apply_itsscale = std::abs(itsscale - 1.0) > 0.001;
        std::ostringstream cmd;
        cmd << "ffmpeg -y";
        if (apply_itsscale) {
            cmd << " -itsscale " << std::fixed << std::setprecision(6) << itsscale;
        }
        cmd << " -i \"" << ofilename << "\" -i \"" << audio_record_file
            << "\" -map 0:v:0? -map 1:a:0?"
            << " -c:v copy -c:a aac -b:a 192k";
        if (!apply_itsscale && video_duration > 0.0) {
            // Without itsscale the original video duration is correct,
            // so cap audio to it to avoid trailing audio past picture.
            cmd << " -t " << std::fixed << std::setprecision(3) << video_duration;
        }
        if (is_mp4_like) {
            cmd << " -movflags +faststart";
        }
        cmd << " \"" << tmp_out << "\" 2>&1";
        mx::system_out << "acmx2: muxing recorded audio into video";
        if (apply_itsscale) {
            mx::system_out << " (A/V resync itsscale=" << std::fixed << std::setprecision(4)
                           << itsscale << ", video=" << std::setprecision(3) << video_duration
                           << "s, audio=" << audio_duration << "s)";
        }
        mx::system_out << "...\n";
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
     * file track as AAC 192 kbps. Output is always limited to the shorter
     * stream. The video timeline is also capped to the original source-video
     * duration when that duration is known, preventing a longer audio file or
     * repeated/queued tail frame from extending the result.
     * The result is written to a temporary file which replaces the
     * original on success.
     */
    void runFileAudioMuxSync() {
#ifdef AUDIO_ENABLED
        if (!file_audio_mode || audio_file_path.empty() || ofilename.empty())
            return;
        std::string out_ext = std::filesystem::path(ofilename).extension().string();
        if (out_ext.empty())
            out_ext = ".mp4";
        std::string tmp_out = ofilename + ".tmp" + out_ext;
        bool is_mp4_like = (out_ext == ".mp4" || out_ext == ".MP4" || out_ext == ".mov" || out_ext == ".MOV" || out_ext == ".m4v" || out_ext == ".M4V");
        int64_t fc = writer.get_frame_count();
        if (fc <= 0) {
            mx::system_err << "acmx2: no encoded video frames; skipping file-audio mux for: " << ofilename << "\n";
            fflush(stderr);
            return;
        }
        double video_duration =
            (fps > 0.0 && fc > 0) ? static_cast<double>(fc) / fps
                                  : 0.0;
        double mux_duration = video_duration;
        if (!filename.empty() && fps > 0.0 && totalFrames > 0.0) {
            const double source_video_duration = totalFrames / fps;
            mux_duration =
                mux_duration > 0.0
                    ? std::min(mux_duration, source_video_duration)
                    : source_video_duration;
        }
        std::ostringstream cmd;
        cmd << "ffmpeg -y -i \"" << ofilename << "\"";
        if (audio_repeat_mode) {
            cmd << " -stream_loop -1";
        }
        const std::vector<std::string> audioSources =
            file_audio_source_paths();
        std::filesystem::path concatPath;
        if (audioSources.size() > 1) {
            concatPath = std::filesystem::temp_directory_path() /
                         ("acmx2-audio-" +
                          std::to_string(std::chrono::steady_clock::now()
                                             .time_since_epoch()
                                             .count()) +
                          ".ffconcat");
            std::ofstream concatFile(concatPath);
            concatFile << "ffconcat version 1.0\n";
            for (const std::string &source : audioSources) {
                std::string escapedSource;
                for (char value : source) {
                    if (value == '\'')
                        escapedSource += "'\\''";
                    else
                        escapedSource += value;
                }
                concatFile << "file '" << escapedSource << "'\n";
            }
            concatFile.close();
            cmd << " -f concat -safe 0 -i \"" << concatPath.string()
                << "\"";
        } else {
            const std::string audioSource =
                audioSources.empty() ? audio_file_path : audioSources.front();
            cmd << " -i \"" << audioSource << "\"";
        }
        cmd << " -map 0:v:0? -map 1:a:0?"
            << " -c:v copy -c:a aac -b:a 192k";
        if (mux_duration > 0.0) {
            cmd << " -t " << std::fixed << std::setprecision(6)
                << mux_duration;
        }
        cmd << " -shortest";
        if (is_mp4_like) {
            cmd << " -movflags +faststart";
        }
        cmd << " \"" << tmp_out << "\" 2>&1";
        mx::system_out << "acmx2: muxing audio file into video"
                       << (audio_repeat_mode ? " (repeating)" : "")
                       << " (shortest stream"
                       << (mux_duration > 0.0
                               ? ", max " + std::to_string(mux_duration) + "s"
                               : std::string())
                       << ")...\n";
        fflush(stdout);
        int ret = std::system(cmd.str().c_str());
        if (!concatPath.empty()) {
            std::error_code removeError;
            std::filesystem::remove(concatPath, removeError);
        }
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
#ifdef AUDIO_ENABLED
        // End live audio at the video capture boundary. Draining queued
        // frames and encoder packets must not add an audio-only tail.
        if (audio_is_enabled && audio_engine.recorder().is_recording()) {
            audio_engine.recorder().stop();
        }
#endif
        queueCondVar.notify_all();
        captureQueueCondVar.notify_all();
        isMuxing = true;
        muxComplete = false;
        muxThread = std::thread([this]() {
            const bool shouldTransferAudio = !png_video_mode && !filename.empty() && !repeat && copy_audio;
#ifdef AUDIO_ENABLED
            const bool shouldRecordedMux = !png_video_mode && audio_is_enabled && !file_audio_mode && !audio_record_file.empty() &&
                                           (audio_engine.recorder().is_recording() || std::filesystem::exists(audio_record_file));
            const bool shouldFileAudioMux = !png_video_mode && file_audio_mode && !audio_file_path.empty() && !audio_record_file.empty() && !ofilename.empty();
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
            } else if (png_video_mode) {
                mx::system_out << "acmx2: wrote " << png_video_frame_counter.load()
                               << " PNG frames to directory: " << png_video_dir << "\n";
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
#ifdef AUDIO_ENABLED
        if (audio_is_enabled && audio_engine.recorder().is_recording()) {
            audio_engine.recorder().stop();
        }
#endif
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
            fflush(stdout);
            fflush(stderr);
        } else if (png_video_mode) {
            mx::system_out << "acmx2: wrote " << png_video_frame_counter.load()
                           << " PNG frames to directory: " << png_video_dir << "\n";
            fflush(stdout);
        }
        if (generate_mode) {
            mx::system_out << "acmx2: --generate: saved " << generate_saved_counter.load()
                           << " PNG frames to directory: " << generate_dir << "\n";
            fflush(stdout);
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
        update_compute_shader_support();
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
     * @param context_config Requested OpenGL context version.
     */
    MainWindow(const MXArguments &args, const OpenGLContextConfig &context_config)
        : gl::GLWindow("ACMX2", args.tw, args.th, false, gl::GLMode::DESKTOP,
                       context_config.major, context_config.minor,
                       args.fps_value <= 60.0),
          silent_mode(args.silent) {
        initCommon(args);
    }

    /**
     * @brief Construct a headless (off-screen) MainWindow for silent batch processing.
     *
     * Uses gl::GLMode::DESKTOP to create an OpenGL context without a
     * visible window.  Intended for `--silent` mode where video is
     * processed and recorded without any display.
     *
     * @param args           Parsed CLI arguments.
     * @param context_config Requested OpenGL context version.
     * @param headless       Unused disambiguator parameter.
     */
    MainWindow(const MXArguments &args, const OpenGLContextConfig &context_config,
               bool headless)
        : gl::GLWindow("ACMX2", args.tw, args.th, false, gl::GLMode::DESKTOP,
                       context_config.major, context_config.minor, false),
          silent_mode(true) {
        static_cast<void>(headless);
        SDL_HideWindow(getWindow());
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
    static_cast<void>(list_only);
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
        out << c.section << "\n"
            << name << c.reset << "\n";
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

        printSection(out, c, "General", {{"-v, -h, --help, --version", "Show this information screen and keyboard controls.", "acmx2 --help"}, {"-p <path>, --path <path>", "Set assets root directory (shaders, data files, defaults).", "acmx2 --path ./data"}, {"-r <WxH>, --resolution <WxH>", "Set output/window resolution (for display and recording).", "acmx2 --resolution 1920x1080"}, {"-N, --fullscreen", "Start in fullscreen mode (Escape to exit fullscreen).", "acmx2 --fullscreen"}, {"--silent", "Run video or graphics-file rendering headlessly (no preview window).", "acmx2 -g image.png -o out.mp4 --duration 10 --silent"}, {"--duration <seconds>", "Auto-stop recording/output after elapsed seconds.", "acmx2 -i in.mp4 -o out.mp4 --duration 30"}, {"--max-size <MB>", "Auto-stop when output file size exceeds MB.", "acmx2 -i in.mp4 -o out.mp4 --max-size 500.0"}});

        printSection(out, c, "Input Source", {{"-i <file>, --input <file>", "Input video file.", "acmx2 --input clip.mp4"}, {"-g <file>, --graphic <file>", "Input still image instead of camera/video.", "acmx2 --graphic frame.png"}, {"-d <idx>, --device <idx>", "Camera device index to open.", "acmx2 --device 0"}, {"-c <WxH>, --camera-res <WxH>", "Request camera capture resolution.", "acmx2 --camera-res 1280x720"}, {"--enumerate-device <idx>", "Print camera resolutions/formats supported by device and exit.", "acmx2 --enumerate-device 0"}, {"--use-yuv", "Prefer YUYV camera capture over MJPG for compatible devices.", "acmx2 --device 0 --use-yuv"}});

        printSection(out, c, "Shaders And Visual Pipeline", {{"-s <library-dir>, --shaders <library-dir>", "Use a shader library directory (library.json preferred, index.txt fallback).", "acmx2 --shaders ./shaders"}, {"-f <frag.glsl>, --fragment <frag.glsl>", "Use a single fragment shader file directly.", "acmx2 --fragment ./shaders/wave.glsl"}, {"--shader <index>", "Select initial shader index from the active library.", "acmx2 --shaders ./shaders --shader 3"}, {"--shader-pass <list>", "Run multiple shader indices per frame (comma-separated).", "acmx2 --shader-pass 0,4,7"}, {"--playlist <file>", "Load shader playlist text file (one shader name per line).", "acmx2 --playlist live_set.txt"}, {"--cross-fade <seconds>", "Set smooth transition time between playlist shader switches.", "acmx2 --playlist live_set.txt --cross-fade 1.25"}, {"--autopilot-frames <N>", "Auto-switch to random playlist shader every N rendered frames (minimum 4).", "acmx2 --shaders ./shaders --autopilot-frames 240"}, {"--autopilot-timeout <N>", "Alias for --autopilot-frames (minimum 4).", "acmx2 --shaders ./shaders --autopilot-timeout 240"}, {"--autopilot-random <N>", "Use random autopilot interval 4..N frames for each J/Y autoplay switch.", "acmx2 --shaders ./shaders --autopilot-random 300"}, {"--time-speed <mult>", "Scale shader time uniform speed (1.0 = normal).", "acmx2 --time-speed 0.5"}, {"--normalized", "Advance time_f by a fixed output-frame interval instead of wall time.", "acmx2 --normalized --time-speed 0.5"}, {"--build <library-path>", "Compile shader library into cache, then exit.", "acmx2 --build ./shaders"}, {"--remove-broken <library-path>", "Compile-check each shader and remove failing manifest entries, then exit.", "acmx2 --remove-broken ./shaders"}, {"--no-cache", "Disable shader binary cache and always compile at startup.", "acmx2 --no-cache"}, {"--texture-cache", "Enable texture/frame cache for cache-aware shader effects.", "acmx2 --texture-cache"}, {"--cache-delay <frames>", "Delay frame cache feed by N frames for temporal effects.", "acmx2 --texture-cache --cache-delay 6"}, {"--texture-cache-size <N>", "Set texture cache ring buffer size (1-64, default 8).", "acmx2 --texture-cache --texture-cache-size 16"}, {"--enable-3d", "Enable 3D object rendering pipeline.", "acmx2 --enable-3d"}, {"--model <file>", "Load a custom 3D model file for the 3D scene.", "acmx2 --enable-3d --model scene.obj"}, {"--flip", "Flip final output vertically before display/encode.", "acmx2 --flip"}, {"--rotate <mode>", "Rotate input frames clockwise, 180 degrees, or counterclockwise.", "acmx2 --rotate clockwise"}});

        printSection(out, c, "Texture Array Cache", {{"--texture-cache-array", "Store frame history in one sampler2DArray named history.", "acmx2 --texture-cache-array"}});

        printSection(out, c, "DNN And ONNX Models", {{"--human <file>", "Load ONNX human segmentation model (e.g., pphumanseg .onnx) to isolate foreground person.", "acmx2 --human human_seg.onnx -i input.mp4 -o output.mp4"}, {"--background", "When --human is used, apply shaders only to background; composite person on top.", "acmx2 --human model.onnx --background"}, {"--black <threshold>", "Set mask black point / shadow crush threshold for color/segmentation masks (default: 0.35).", "acmx2 --human seg.onnx --black 0.25"}, {"--white <threshold>", "Set mask white point / opacity saturation threshold for color/segmentation masks (default: 0.75).", "acmx2 --human seg.onnx --white 0.85"}, {"--edge <file>", "Load ONNX edge detection model (e.g., Dexined .onnx) to replace frame with edge map.", "acmx2 --edge edges.onnx -i video.mp4 -o edges.mp4"}, {"--onnx <file>", "Load generic ONNX model from YAML config file; replaces frame with model output.", "acmx2 --onnx bubble.yaml -i input.mp4 -o output.mp4"}, {"--check-dnn", "Report whether this build has OpenCV DNN support enabled.", "acmx2 --check-dnn"}});

        printSection(out, c, "GPU And CUDA", {{"--gpu-filter <list>", "Apply CUDA filter chain by index list (comma-separated).", "acmx2 --gpu-filter 1,12,18"}, {"--gpu-buffer <N>", "Set GPU temporal frame buffer size (4..32).", "acmx2 --gpu-buffer 12"}, {"--list-filters", "List all built-in GPU filters and their indices.", "acmx2 --list-filters"}, {"-m <idx>, --cuda-device <idx>", "Select CUDA device index to run processing on.", "acmx2 --cuda-device 0"}, {"--list-cuda-devices", "List CUDA devices visible to the runtime.", "acmx2 --list-cuda-devices"}, {"--check-cuda", "Report whether this build has CUDA support enabled.", "acmx2 --check-cuda"}});

        printSection(out, c, "Recording And Encoding", {{"-o <file>, --output <file>", "Write processed video to output file.", "acmx2 -i in.mp4 -o out.mp4"}, {"--png", "Video file mode: write output as PNG frame sequence in an output subdirectory.", "acmx2 -i in.mp4 -o out.mp4 --png"}, {"--generate <N>", "Save a PNG frame every N frames in an output subdirectory (video or camera mode).", "acmx2 -i in.mp4 --generate 30"}, {"-e <prefix>, --prefix <prefix>", "Snapshot filename prefix for captured frames.", "acmx2 --prefix snap/frame_"}, {"-u <fps>, --fps <fps>", "Set output frame rate for recording.", "acmx2 --fps 60"}, {"-b <crf>, --bitrate <crf>", "Legacy CRF quality option for encoder.", "acmx2 --bitrate 20"}, {"--encode-preset <name>", "Encoder speed/quality preset (ultrafast .. veryslow).", "acmx2 --encode-preset fast"}, {"--encode-tune <name>", "Tune encoder for content type or low latency.", "acmx2 --encode-tune film"}, {"--encode-crf <0-51>", "Set encoder quality directly (lower = better quality/larger file).", "acmx2 --encode-crf 18"}, {"--encode-codec <name>", "Select auto/software/nvenc or an exact installed FFmpeg encoder.", "acmx2 --encode-codec libx265"}, {"--list-encoders", "List FFmpeg video encoders visible to MXWrite and exit.", "acmx2 --list-encoders"}, {"--list-encoder-options <name>", "List AVOptions accepted by one exact encoder and exit.", "acmx2 --list-encoder-options libx265"}, {"--encode-realtime", "Enable low-latency encoder settings for live pipelines.", "acmx2 --encode-realtime"}, {"--no-drop", "File/graphics mode: never drop frames; ignored for webcams.", "acmx2 -i in.mp4 -o out.mp4 --no-drop"}, {"--display-filter", "Show current shader/stack and GPU filter in upper-left corner.", "acmx2 --display-filter"}, {"--use-watermark <text>", "Enable watermark with given text in recorded videos (upper-left).", "acmx2 --use-watermark \"My Channel\""}, {"--use-watermark-color <r,g,b>", "Watermark text color as 0-255 components.", "acmx2 --use-watermark-color 255,255,0"}, {"--copy-audio", "Mux input audio track into encoded output when possible.", "acmx2 -i in.mp4 -o out.mp4 --copy-audio"}, {"-a, --repeat", "Loop video input source continuously.", "acmx2 -i loop.mp4 --repeat"}});

        printSection(out, c, "Advanced Encoder Parameters", {{"--encode-params <string>", "Pass additional FFmpeg-style video encoder options through MXWrite.", "acmx2 --encode-codec hevc_nvenc --encode-params \"-preset p6 -tune lossless -profile:v rext -pix_fmt yuv444p\""}});

#ifdef AUDIO_ENABLED
        printSection(out, c, "Audio Reactivity", {{"-w, --enable-audio", "Enable audio-reactive shader modulation.", "acmx2 --enable-audio"}, {"-l <N>, --channels <N>", "Number of audio channels to capture/process.", "acmx2 --channels 2"}, {"-q <value>, --sense <value>", "Set audio sensitivity multiplier for visual response.", "acmx2 --sense 1.4"}, {"--audio-warm-rate <value>", "Startup audio warmup rate in 1/sec (0.5 ~= 2s fade-in, 1.0 ~= 1s, 0 disables warmup).", "acmx2 --enable-audio --audio-warm-rate 0.35"}, {"-y, --pass-through", "Play live input or file audio through the selected output device.", "acmx2 --audio-file soundtrack.mp3 --pass-through"}, {"--audio-input <device>", "Select input audio device name/id.", "acmx2 --audio-input \"USB Audio\""}, {"--audio-output <device>", "Select pass-through output device name/id.", "acmx2 --audio-output \"Built-in Output\""}, {"--list-devices", "List available audio input/output devices.", "acmx2 --list-devices"}, {"--record-audio <wav-file>", "Record captured audio stream to a WAV file.", "acmx2 --record-audio take.wav"}, {"--record-gain <0.0-2.0>", "Set recording gain multiplier (1.0 = unity).", "acmx2 --record-gain 1.2"}, {"--audio-file <file>", "Use an audio file or M3U playlist as reactivity source instead of microphone input.", "acmx2 --audio-file soundtrack.m3u"}, {"--audio-trunc", "Stop playback/output when the audio source reaches EOF.", "acmx2 --audio-file soundtrack.m3u --audio-trunc"}, {"--audio-repeat", "Restart file audio or the full playlist at EOF.", "acmx2 --audio-file soundtrack.m3u --audio-repeat"}, {"--enable-audio-buffers <N>", "Allocate one sampler1DArray with N spectrum-history layers (GPU-limited).", "acmx2 --enable-audio --enable-audio-buffers 8"}, {"--check-audio", "Report whether this build has audio support enabled.", "acmx2 --check-audio"}});
#endif

#ifdef MIDI_ENABLED
        printSection(out, c, "MIDI Control", {{"--midi-map <file>", "Load MIDI mapping configuration file.", "acmx2 --midi-map midi.midi_cfg"}, {"--midi-device <idx>", "Select MIDI input device index.", "acmx2 --midi-device 0"}, {"--list-midi", "List available MIDI input devices.", "acmx2 --list-midi"}, {"--check-midi", "Report whether this build has MIDI support enabled.", "acmx2 --check-midi"}});
#endif

        printSection(out, c, "Runtime Overlay", {{"--disable-counter", "Hide timer and FPS overlay text.", "acmx2 --disable-counter"}});
    }

    template <typename Stream>
    void printKeyboardControls(Stream &out) {
        const CliColors c = makeCliColors();
        out << c.title << "\nKeyboard Controls" << c.reset << "\n";

        printSection(out, c, "Main", {{"Escape", "Quit.", ""}, {"Ctrl+X", "Quit without audio mux.", ""}, {"Up Arrow", "Crossfade to the previous shader (or playlist entry in playlist/autopilot mode).", ""}, {"Down Arrow", "Crossfade to the next shader (or playlist entry in playlist/autopilot mode).", ""}, {"Shift+Up Arrow", "In playlist/autopilot mode: change post-multipass shader backward.", ""}, {"Shift+Down Arrow", "In playlist/autopilot mode: change post-multipass shader forward.", ""}, {"Left Arrow", "Previous GPU filter (if enabled).", ""}, {"Right Arrow", "Next GPU filter (if enabled).", ""}, {"Space", "Enable/disable processing.", ""}, {"L", "Toggle video freeze (Video/Image modes).", ""}, {"P", "Toggle pause (Video/Image) or toggle shader playlist.", ""}, {"J", "Toggle autopilot mode (requires playlist).", ""}, {"Y", "Toggle sequential autopilot (cycles playlist in order, requires playlist).", ""}, {"N", "Toggle random crossfade selection for autopilot shader switches.", ""}, {"T", "Enable/disable time.", ""}, {"U / I", "Step time when time is disabled.", ""}, {"Page Up / Page Down", "Increase/decrease time speed.", ""}, {"M", "Toggle multi-pass / multi-shader pass.", ""}, {"F", "Toggle fullscreen.", ""}, {"Q", "Toggle reactive time (if AUDIO_ENABLED).", ""}, {"Insert", "Increase audio sensitivity.", ""}, {"Delete", "Decrease audio sensitivity.", ""}, {"End", "Toggle spectrum sensitivity scaling.", ""}, {"Home", "Toggle audio delta time scaling.", ""}, {"3", "Toggle 2D/3D mode.", ""}});

        printSection(out, c, "Snapshots", {{"Z", "Save PNG snapshot (SDR 8-bit; HDR mode still outputs SDR PNG).", ""}, {"4", "Save TIFF snapshot (SDR: 8-bit RGBA; HDR: 16-bit RGBA; requires ACMX2_WITH_TIFF).", ""}, {"5", "Save lossless WebP snapshot (HDR is tone-mapped; requires ACMX2_WITH_WEBP).", ""}, {"6", "Save raw RGBA snapshot (HDR: 16-bit RGBA, otherwise 8-bit RGBA).", "ffplay -f rawvideo -pixel_format rgba64le -video_size WxH file.raw"}});

        printSection(out, c, "3D Mode", {{"W / A / S / D", "Look around.", ""}, {"V", "Toggle view rotation.", ""}, {"O", "Toggle oscillation.", ""}, {"X", "Reset camera distance.", ""}, {"+ / -", "Increase/decrease camera distance.", ""}, {"B", "Increase movement speed.", ""}, {"N (held in 3D)", "Decrease movement speed.", ""}, {"C", "Toggle object wave.", ""}, {"E", "Enable/disable watermark.", ""}, {"]", "Increase model scale.", ""}, {"[", "Decrease model scale.", ""}, {". (period)", "Increase camera rotation speed.", ""}, {", (comma)", "Decrease camera rotation speed.", ""}});
        printSection(out, c, "Environment Variables", {{"ACMX2_PATH", "Default assets root directory (equivalent to --path). Used when --path is not specified.", "export ACMX2_PATH=/usr/local/share/acmx2"}, {"ACMX2_SHADER_PATH", "Default shader library index file or directory (equivalent to --shaders). Used when neither --shaders nor --fragment is specified.", "export ACMX2_SHADER_PATH=/usr/local/share/acmx2/filters"}});
    }
} // namespace

/// @brief Print the program name, software brand, and project URL once.
void printBranding() {
    mx::system_out << PROGRAM_NAME << " " << VERSION_INFO << "\n";
    mx::system_out << "(C) 2026 " << VERSION_AUTHOR << "\n";
    mx::system_out << "https://lostsidedead.biz\n";
}

/// @brief Print program information, arguments, and keyboard controls.
void printAbout(bool include_branding = true) {
    if (include_branding)
        printBranding();
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
        .addOptionSingle('h', "Display help message")
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
        .addOptionDoubleValue('H', "shader", "Shader Index")
        .addOptionDoubleValue(622, "shader-file", "Shader filename in the active library")
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
        .addOptionDoubleValue(275, "texture-cache-size", "Ring buffer size for texture cache (default 8)")
        .addOptionDouble(276, "texture-cache-array", "Expose texture cache as sampler2DArray history")
        .addOptionDouble(258, "copy-audio", "Copy audio track")
        .addOptionDouble(259, "enable-3d", "Enable 3D cube")
        .addOptionDoubleValue(260, "model", "Model file")
        .addOptionDoubleValue(700, "human", "Human segmentation model (PPHS .onnx) -- isolate person via DNN")
        .addOptionDouble(701, "background", "With --human: apply shaders only to the background; composite person on top")
        .addOptionDoubleValue(702, "black", "Mask black point / shadow crush threshold (default 0.35)")
        .addOptionDoubleValue(703, "white", "Mask white point / opacity saturation threshold (default 0.75)")
        .addOptionDoubleValue(704, "edge", "Edge detection model (Dexined .onnx) -- replace frame with edge map")
        .addOptionDoubleValue(705, "onnx", "Generic ONNX model YAML config -- replace frame with DNN output")
        .addOptionDouble(621, "check-dnn", "Report whether OpenCV DNN support is compiled in")
        .addOptionDouble(261, "help", "print help info")
        .addOptionDouble(262, "version", "print version and help info")
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
        .addOptionDoubleValue(305, "audio-file", "Use an audio file or M3U playlist for reactivity instead of mic")
        .addOptionDouble(306, "audio-trunc", "Stop playback when the audio file reaches the end")
        .addOptionDoubleValue(307, "enable-audio-buffers", "Allocate a spectrum-history sampler1DArray with N layers")
        .addOptionDoubleValue(308, "audio-warm-rate", "Startup audio warmup rate (1/sec, default: 0.5)")
        .addOptionDouble(309, "audio-repeat", "Restart audio file playback at the end")
#endif
        .addOptionDouble('N', "fullscreen", "Fullscreen Window (Escape to quit)")
        .addOptionDouble(405, "silent", "Silent mode - process video without window, (video files only)")
        .addOptionDoubleValue(406, "shader-pass", "Shader pass indices (comma-separated, e.g. 0,1,2)")
        .addOptionDoubleValue(623, "shader-pass-files", "Shader pass filenames (length-prefixed)")
        .addOptionDoubleValue(407, "build", "Build shader cache for specified library path (compiles shaders and exits)")
        .addOptionDouble(408, "no-cache", "Disable shader caching (always recompile shaders)")
        .addOptionDoubleValue(416, "remove-broken", "Compile each shader in library path; remove failures from its manifest, then exit")
        .addOptionDoubleValue(409, "time-speed", "Constant time_f speed multiplier (default: 1.0)")
        .addOptionDouble(620, "normalized", "Use deterministic output-frame time_f advancement")
        .addOptionDoubleValue(410, "playlist", "Shader playlist text file (one shader name per line, P to toggle)")
        .addOptionDoubleValue(417, "autopilot-frames", "Autopilot frame interval; switch to a random playlist shader every N frames (minimum 4, J toggles)")
        .addOptionDoubleValue(420, "autopilot-timeout", "Alias for --autopilot-frames")
        .addOptionDoubleValue(418, "autopilot-random", "Autopilot random interval upper bound; each J/Y switch picks 4..N frames")
        .addOptionDoubleValue(419, "autiopilot-random", "Alias for --autopilot-random")
        .addOptionDoubleValue(411, "duration", "Recording duration in seconds (float); stop recording and exit after elapsed")
        .addOptionDoubleValue(610, "max-size", "Stop recording when output file exceeds size in MB (float)")
        .addOptionDouble(611, "png", "Video file mode: write PNG frames to output subdirectory instead of encoding video")
        .addOptionDoubleValue(612, "generate", "Save a PNG frame every N frames to an output subdirectory (video or camera mode)")
        .addOptionDouble(613, "interface-shm", "Enable Qt interface shared-memory control channel")
        .addOptionDoubleValue(412, "cross-fade", "Crossfade duration in seconds when switching playlist shaders (default: 0.5)")
        .addOptionDoubleValue(413, "enumerate-device", "List supported resolutions for a camera device index")
        .addOptionDouble(414, "use-yuv", "Use YUV (YUYV) camera format instead of MJPG")
        .addOptionDoubleValue(600, "encode-preset", "Encoder preset: ultrafast..veryslow or NVENC p1..p7 (default: medium)")
        .addOptionDoubleValue(601, "encode-tune", "Encoder tune: software tunes or NVENC hq,uhq,ll,ull,lossless (default: none)")
        .addOptionDoubleValue(602, "encode-crf", "Encoder CRF quality 0 (best) .. 51 (worst), default 18")
        .addOptionDoubleValue(603, "encode-codec", "Encoder policy or exact FFmpeg encoder name (default: auto)")
        .addOptionDouble(604, "encode-realtime", "Enable low-latency realtime encoding flags")
        .addOptionDouble(605, "flip", "Vertical flip output frames")
        .addOptionDoubleValue(617, "rotate", "Rotate input frames: clockwise, 180, or counterclockwise")
        .addOptionDouble(606, "no-drop", "File/graphics mode: never drop frames; ignored for webcams")
        .addOptionDouble(607, "display-filter", "Display current shader/stack and GPU filter in upper-left corner")
        .addOptionDoubleValue(608, "use-watermark", "Enable watermark with the given text in upper-left corner of recorded video")
        .addOptionDoubleValue(609, "use-watermark-color", "Watermark color as r,g,b each 0-255 (default: 255,0,150)")
        .addOptionDoubleValue(614, "encode-params", "Additional FFmpeg-style video encoder parameters passed through MXWrite")
        .addOptionDouble(615, "list-encoders", "List FFmpeg video encoders available to MXWrite")
        .addOptionDoubleValue(616, "list-encoder-options", "List options for an exact FFmpeg video encoder")
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
        if (std::string(argv[i]) == "--check-dnn") {
#ifdef ACMX2_WITH_DNN
            std::cout << "OpenCV DNN: enabled" << std::endl;
#else
            std::cout << "OpenCV DNN: disabled" << std::endl;
#endif
            return EXIT_SUCCESS;
        }
    }

    printBranding();

    Argument<std::string> arg;
    MXArguments args;
    int value = 0;
    try {
        while ((value = parser.proc(arg)) != -1) {
            switch (value) {
            case 'v':
            case 'h':
            case 261:
            case 262:
                printAbout(false);
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
            case 'H':
                args.shader_index = atoi(arg.arg_value.c_str());
                break;
            case 622:
                args.shader_file = arg.arg_value;
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
                    if (v < 0)
                        v = 0;
                    if (v > 51)
                        v = 51;
                    args.encode_opts.crf = v;
                } catch (...) {
                }
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
            case 275: {
                int sz = atoi(arg.arg_value.c_str());
                if (sz >= 1 && sz <= 64)
                    args.cache_size = sz;
                mx::system_out << "acmx2: Texture cache size set to: " << args.cache_size << "\n";
                break;
            }
            case 276:
                args.cache = true;
                args.cache_array = true;
                mx::system_out
                    << "acmx2: Texture cache array enabled as uniform history.\n";
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
            case 700:
                args.human_model = arg.arg_value;
                break;
            case 701:
                args.human_background_only = true;
                break;
            case 702:
                args.human_black = static_cast<float>(std::stod(arg.arg_value));
                break;
            case 703:
                args.human_white = static_cast<float>(std::stod(arg.arg_value));
                break;
            case 704:
                args.edge_model = arg.arg_value;
                break;
            case 705:
                args.onnx_model = arg.arg_value;
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
            case 621:
#ifdef ACMX2_WITH_DNN
                std::cout << "OpenCV DNN: enabled\n";
#else
                std::cout << "OpenCV DNN: disabled\n";
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
                args.audio_pass_through = true;
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
                acmx2::audio::AudioEngine::list_devices();
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
            case 307: {
                int n = atoi(arg.arg_value.c_str());
                if (n < 0)
                    n = 0;
                args.audio_buffers = n;
            } break;
            case 308: {
                float rate = static_cast<float>(atof(arg.arg_value.c_str()));
                if (!std::isfinite(rate) || rate < 0.0f) {
                    mx::system_err << "acmx2: --audio-warm-rate must be >= 0.0\n";
                    mx::system_err.flush();
                    exit(EXIT_FAILURE);
                }
                args.audio_warm_rate = rate;
            } break;
            case 309:
                args.audio_repeat = true;
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
            case 623: {
                const std::string passFiles = arg.arg_value;
                size_t start = 0;
                while (start < passFiles.size()) {
                    const size_t separator = passFiles.find(':', start);
                    if (separator == std::string::npos)
                        break;
                    size_t consumed = 0;
                    size_t nameLength = 0;
                    try {
                        nameLength = std::stoull(
                            passFiles.substr(start, separator - start),
                            &consumed);
                    } catch (...) {
                        break;
                    }
                    if (consumed != separator - start ||
                        nameLength > passFiles.size() - separator - 1)
                        break;
                    const size_t nameStart = separator + 1;
                    args.shader_pass_files.push_back(
                        passFiles.substr(nameStart, nameLength));
                    start = nameStart + nameLength;
                }
                if (start != passFiles.size()) {
                    mx::system_err
                        << "acmx2: Invalid --shader-pass-files payload\n";
                    return EXIT_FAILURE;
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
            case 620:
                args.normalized_time = true;
                mx::system_out << "acmx2: Normalized time enabled\n";
                break;
            case 410:
                args.playlist_file = arg.arg_value;
                mx::system_out << "acmx2: Playlist file: " << args.playlist_file << "\n";
                break;
            case 417:
            case 420:
                args.autopilot_frames = atoi(arg.arg_value.c_str());
                if (args.autopilot_frames < 4)
                    args.autopilot_frames = 4;
                mx::system_out << "acmx2: Autopilot frames: " << args.autopilot_frames << "\n";
                break;
            case 418:
            case 419:
                args.autopilot_random_interval = true;
                args.autopilot_random_timeout = atoi(arg.arg_value.c_str());
                if (args.autopilot_random_timeout < 4)
                    args.autopilot_random_timeout = 4;
                mx::system_out << "acmx2: Autopilot random interval enabled (4-"
                               << args.autopilot_random_timeout << " frames)\n";
                break;
            case 411:
                args.duration = atof(arg.arg_value.c_str());
                if (args.duration > 0.0) {
                    mx::system_out << "acmx2: Duration set to: " << args.duration << " seconds\n";
                }
                break;
            case 610:
                args.max_size_mb = atof(arg.arg_value.c_str());
                if (args.max_size_mb > 0.0) {
                    mx::system_out << "acmx2: Max output size set to: "
                                   << std::fixed << std::setprecision(2) << args.max_size_mb
                                   << " MB\n";
                } else {
                    args.max_size_mb = 0.0;
                }
                break;
            case 611:
                args.png_output = true;
                mx::system_out << "acmx2: --png enabled (video-file mode only; ignored for camera mode)\n";
                break;
            case 612: {
                int n = atoi(arg.arg_value.c_str());
                if (n < 1) {
                    mx::system_err << "acmx2: --generate requires a positive integer frame interval\n";
                    exit(EXIT_FAILURE);
                }
                args.generate_interval = n;
                mx::system_out << "acmx2: --generate " << n << ": will save a PNG frame every " << n << " frames\n";
                break;
            }
            case 613:
                args.interface_shm = true;
                mx::system_out << "acmx2: Qt shared-memory control channel enabled\n";
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
                bool loopback_device = false;
                if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
                    mx::system_out << "Device " << dev_idx << ": " << dev_path << "\n";
                    mx::system_out << "  Driver : " << cap.driver << "\n";
                    mx::system_out << "  Card   : " << cap.card << "\n";
                    mx::system_out << "  Bus    : " << cap.bus_info << "\n";
                    const std::string driver(reinterpret_cast<const char *>(cap.driver));
                    loopback_device = driver.find("v4l2loopback") != std::string::npos ||
                                      driver.find("v4l2 loopback") != std::string::npos;
                }

                double current_fps = 0.0;
                v4l2_streamparm stream_parameters{};
                stream_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (ioctl(fd, VIDIOC_G_PARM, &stream_parameters) == 0 &&
                    stream_parameters.parm.capture.timeperframe.numerator != 0) {
                    const v4l2_fract &interval =
                        stream_parameters.parm.capture.timeperframe;
                    current_fps = static_cast<double>(interval.denominator) /
                                  interval.numerator;
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
                        '\0'};
                    mx::system_out << "\n  Format: " << fourcc << " (" << fmt.description << ")\n";
                    v4l2_frmsizeenum fsize{};
                    fsize.pixel_format = fmt.pixelformat;
                    fsize.index = 0;
                    while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fsize) == 0) {
                        if (fsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                            mx::system_out << "    " << fsize.discrete.width << "x" << fsize.discrete.height;
                            std::vector<double> frame_rates;
                            auto append_frame_rate = [&frame_rates](double frame_rate) {
                                if (frame_rate <= 0.0) {
                                    return;
                                }
                                const auto existing = std::find_if(
                                    frame_rates.begin(), frame_rates.end(),
                                    [frame_rate](double value) {
                                        return std::abs(value - frame_rate) < 0.05;
                                    });
                                if (existing == frame_rates.end()) {
                                    frame_rates.push_back(frame_rate);
                                }
                            };
                            v4l2_frmivalenum fival{};
                            fival.pixel_format = fmt.pixelformat;
                            fival.width = fsize.discrete.width;
                            fival.height = fsize.discrete.height;
                            fival.index = 0;
                            while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fival) == 0) {
                                if (fival.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
                                    if (fival.discrete.numerator != 0) {
                                        append_frame_rate(
                                            static_cast<double>(fival.discrete.denominator) /
                                            fival.discrete.numerator);
                                    }
                                } else if (fival.type == V4L2_FRMIVAL_TYPE_STEPWISE ||
                                           fival.type == V4L2_FRMIVAL_TYPE_CONTINUOUS) {
                                    if (fival.stepwise.min.numerator != 0) {
                                        append_frame_rate(
                                            static_cast<double>(fival.stepwise.min.denominator) /
                                            fival.stepwise.min.numerator);
                                    }
                                    if (fival.stepwise.max.numerator != 0) {
                                        append_frame_rate(
                                            static_cast<double>(fival.stepwise.max.denominator) /
                                            fival.stepwise.max.numerator);
                                    }
                                }
                                fival.index++;
                            }

                            append_frame_rate(current_fps);
                            if (loopback_device) {
                                // v4l2loopback reports only its current interval,
                                // but accepts consumer-selected time-per-frame
                                // values. Include common real-time and constrained
                                // high-speed camera rates for the interface.
                                constexpr double LOOPBACK_FRAME_RATES[] = {
                                    24.0, 25.0, 30.0, 50.0, 60.0,
                                    90.0, 120.0, 144.0, 240.0};
                                for (double frame_rate : LOOPBACK_FRAME_RATES) {
                                    append_frame_rate(frame_rate);
                                }
                            }

                            std::sort(frame_rates.begin(), frame_rates.end(),
                                      std::greater<double>());
                            bool first = true;
                            for (double frame_rate : frame_rates) {
                                mx::system_out << (first ? " @ " : ", ")
                                               << std::fixed << std::setprecision(1)
                                               << frame_rate << " fps";
                                first = false;
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
                if (v < 0)
                    v = 0;
                if (v > 51)
                    v = 51;
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
            case 617: {
                std::string rotation = arg.arg_value;
                std::transform(rotation.begin(), rotation.end(),
                               rotation.begin(), [](unsigned char character) {
                                   return static_cast<char>(std::tolower(character));
                               });
                if (rotation == "clockwise" || rotation == "cw" ||
                    rotation == "90" || rotation == "90cw") {
                    args.frame_rotation = FrameRotation::Clockwise90;
                    mx::system_out
                        << "acmx2: Input frame rotation: 90 degrees clockwise\n";
                } else if (rotation == "180") {
                    args.frame_rotation = FrameRotation::Rotate180;
                    mx::system_out
                        << "acmx2: Input frame rotation: 180 degrees\n";
                } else if (rotation == "counterclockwise" ||
                           rotation == "ccw" || rotation == "90ccw" ||
                           rotation == "270") {
                    args.frame_rotation = FrameRotation::Counterclockwise90;
                    mx::system_out
                        << "acmx2: Input frame rotation: 90 degrees counterclockwise\n";
                } else {
                    mx::system_err
                        << "acmx2: --rotate requires clockwise, 180, or "
                           "counterclockwise\n";
                    mx::system_err.flush();
                    exit(EXIT_FAILURE);
                }
                break;
            }
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
            case 614:
                args.encode_opts.ffmpeg_options = arg.arg_value;
                mx::system_out << "acmx2: Extra FFmpeg encoder parameters: "
                               << args.encode_opts.ffmpeg_options << "\n";
                break;
            case 615: {
                auto clean_field = [](std::string value) {
                    std::replace_if(value.begin(), value.end(), [](char ch) { return ch == '\t' || ch == '\r' || ch == '\n'; }, ' ');
                    return value;
                };
                mx::system_out << "MXWRITE_ENCODERS\t1\n";
                for (const EncoderInfo &encoder : available_video_encoders()) {
                    mx::system_out << "ENCODER\t" << clean_field(encoder.name) << '\t'
                                   << clean_field(encoder.long_name) << '\t'
                                   << clean_field(encoder.codec_name) << '\t'
                                   << (encoder.hardware ? "hardware" : "software") << '\t'
                                   << (encoder.experimental ? "experimental" : "stable") << '\t'
                                   << clean_field(encoder.pixel_formats) << '\n';
                }
                mx::system_out.flush();
                return EXIT_SUCCESS;
            }
            case 616: {
                auto clean_field = [](std::string value) {
                    std::replace_if(value.begin(), value.end(), [](char ch) { return ch == '\t' || ch == '\r' || ch == '\n'; }, ' ');
                    return value;
                };
                const std::vector<EncoderOptionInfo> options =
                    video_encoder_options(arg.arg_value);
                if (options.empty() &&
                    !avcodec_find_encoder_by_name(arg.arg_value.c_str())) {
                    mx::system_err << "acmx2: encoder not found: " << arg.arg_value << '\n';
                    return EXIT_FAILURE;
                }
                mx::system_out << "MXWRITE_ENCODER_OPTIONS\t1\t"
                               << clean_field(arg.arg_value) << '\n';
                for (const EncoderOptionInfo &option : options) {
                    mx::system_out << "OPTION\t" << clean_field(option.name) << '\t'
                                   << clean_field(option.type) << '\t'
                                   << clean_field(option.default_value) << '\t'
                                   << clean_field(option.minimum) << '\t'
                                   << clean_field(option.maximum) << '\t'
                                   << clean_field(option.choices) << '\t'
                                   << clean_field(option.help) << '\n';
                }
                mx::system_out.flush();
                return EXIT_SUCCESS;
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
        if (shaderManifestPath(args.remove_broken_path).empty()) {
            mx::system_err << "acmx2: Error: No library.json or index.txt found at: "
                           << args.remove_broken_path << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }
        try {
#if defined(__linux__)
            if (args.silent) {
                // Make remove-broken use the same offscreen path as silent batch mode.
                setenv("SDL_VIDEODRIVER", "offscreen", 0);
                setenv("SDL_AUDIODRIVER", "dummy", 0);
                SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "0");
                installHeadlessSignalHandlers();
                mx::system_out << "acmx2: remove-broken headless mode enabled (Linux)\n";
            }
#endif
            mx::system_out << "acmx2: Creating scan window for remove-broken...\n";
            fflush(stdout);
            const OpenGLContextConfig context_config = select_open_gl_context();

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

                RemoveBrokenWindow(const std::string &path, bool is3d,
                                   const std::string &assets,
                                   const OpenGLContextConfig &context_config,
                                   bool headless)
                    : gl::GLWindow("ACMX2 Remove-Broken", 640, 480, false,
                                   gl::GLMode::DESKTOP, context_config.major,
                                   context_config.minor, false),
                      lib_path(path), enable_3d(is3d), assets_path(assets) {
                    if (headless)
                        SDL_HideWindow(getWindow());
                    update_compute_shader_support();
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
                            if (ev.type == SDL_QUIT)
                                active = false;
                            event(ev);
                        }
                        draw();
                    }
                }
            };

#if defined(__linux__)
            if (args.silent) {
                RemoveBrokenWindow rb_win(args.remove_broken_path, args.is3d,
                                          args.path, context_config, true);
                rb_win.scanLoop();
                return rb_win.success ? EXIT_SUCCESS : EXIT_FAILURE;
            }
#endif
            RemoveBrokenWindow rb_win(args.remove_broken_path, args.is3d,
                                      args.path, context_config, false);
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
        if (shaderManifestPath(args.build_library_path).empty()) {
            mx::system_err << "acmx2: Error: No library.json or index.txt found at: "
                           << args.build_library_path << "\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }

        try {
#if defined(__linux__)
            if (args.silent) {
                // Make build-cache use the same offscreen path as silent batch mode.
                setenv("SDL_VIDEODRIVER", "offscreen", 0);
                setenv("SDL_AUDIODRIVER", "dummy", 0);
                SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "0");
                installHeadlessSignalHandlers();
                mx::system_out << "acmx2: build headless mode enabled (Linux)\n";
            }
#endif
            mx::system_out << "acmx2: Creating build window...\n";
            fflush(stdout);
            const OpenGLContextConfig context_config = select_open_gl_context();

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

                /*
                 * @brief Construct a build window and configure the shader library.
                 * @param path   Path to the shader source directory.
                 * @param is3d   Include 3-D shaders in the cache.
                 * @param assets Base asset path for vertex shader lookup.
                 */
                BuildWindow(const std::string &path, bool is3d,
                            const std::string &assets, int tex_cache_size,
                            bool use_array,
                            const OpenGLContextConfig &context_config,
                            bool headless)
                    : gl::GLWindow("ACMX2 Shader Builder", 640, 480, false,
                                   gl::GLMode::DESKTOP, context_config.major,
                                   context_config.minor, false),
                      lib_path(path), enable_3d(is3d), assets_path(assets) {
                    if (headless)
                        SDL_HideWindow(getWindow());
                    update_compute_shader_support();
                    mx::system_out << "acmx2: Window created, setting up...\n";
                    fflush(stdout);
                    util.path = assets_path;
                    library.enableDualMode(enable_3d);
                    library.setCacheSize(tex_cache_size > 0 ? tex_cache_size : 8);
                    library.setHistoryTextureArray(use_array);
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
                                    float scale = std::min(static_cast<float>(w) / lw, static_cast<float>(h) / lh);
                                    int dw = static_cast<int>(lw * scale);
                                    int dh = static_cast<int>(lh * scale);
                                    int lx = (w - dw) / 2;
                                    int ly = (h - dh) / 2;
                                    logo_sp.initWithTexture(&logo_sh, logo_tex, lx, ly, dw, dh);
                                    logo_tex = 0;
                                    logo_sp.draw();
                                }
                            } catch (...) {
                            }
                            if (logo_tex) {
                                glDeleteTextures(1, &logo_tex);
                            }
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
                BuildWindow build_win(args.build_library_path, args.is3d,
                                      args.path, args.cache_size,
                                      args.cache_array, context_config, true);
                build_win.buildLoop();
                return build_win.success ? EXIT_SUCCESS : EXIT_FAILURE;
            }
#endif
            BuildWindow build_win(args.build_library_path, args.is3d,
                                  args.path, args.cache_size,
                                  args.cache_array, context_config, false);
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
        const std::vector<std::string> requestedShaderFiles =
            args.mode == 1 &&
                    (!args.shader_file.empty() || !args.shader_pass_files.empty())
                ? sortedShaderLibraryEntries(args.library)
                : std::vector<std::string>{};
        if (args.mode == 1 && !args.shader_file.empty()) {
            const int selectedIndex = shaderIndexForFile(requestedShaderFiles,
                                                         args.shader_file);
            if (selectedIndex < 0) {
                mx::system_err << "acmx2: Shader file is not present in the active library: "
                               << args.shader_file << "\n";
                return EXIT_FAILURE;
            }
            args.shader_index = selectedIndex;
        }
        if (args.mode == 1 && !args.shader_pass_files.empty()) {
            args.shader_pass_list.clear();
            for (const std::string &shaderFile : args.shader_pass_files) {
                const int passIndex = shaderIndexForFile(requestedShaderFiles,
                                                         shaderFile);
                if (passIndex < 0) {
                    mx::system_err << "acmx2: Shader pass file is not present in the active library: "
                                   << shaderFile << "\n";
                    return EXIT_FAILURE;
                }
                args.shader_pass_list.push_back(passIndex);
            }
            args.shader_pass_enabled = !args.shader_pass_list.empty();
        }
        args.slib = std::make_tuple(args.mode,
                                    (args.mode == 0) ? args.fragment : args.library,
                                    (args.mode == 0) ? 0 : args.shader_index);
        // Texture cache works in video, graphics, and camera modes.

        if (args.silent) {
            if (args.filename.empty() && args.graphic_file.empty()) {
                mx::system_err << "acmx2: Error: --silent mode requires a video (-i/--input) "
                                  "or graphics (-g/--graphic) input file\n";
                mx::system_err << "       Silent mode does not support camera input.\n";
                mx::system_err.flush();
                return EXIT_FAILURE;
            }
            if (args.ofilename.empty()) {
                mx::system_err << "acmx2: Error: --silent mode requires an output file (-o/--output)\n";
                mx::system_err.flush();
                return EXIT_FAILURE;
            }
            if (!args.graphic_file.empty() && args.duration <= 0.0) {
                mx::system_err << "acmx2: Error: silent graphics mode requires a positive "
                                  "maximum duration (--duration <seconds>)\n";
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
            SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "0");
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

        if (args.png_output && !args.filename.empty() && args.ofilename.empty()) {
            mx::system_err << "acmx2: Error: --png in video-file mode requires -o/--output to derive the PNG frame directory\n";
            mx::system_err.flush();
            return EXIT_FAILURE;
        }

        // Create file-input windows at the source's native dimensions when
        // --resolution was not supplied. Some window managers do not reliably
        // apply an SDL resize immediately after an OpenGL window is created,
        // so determine the initial size before constructing MainWindow.
        if (!args.graphic_file.empty() && !args.sizev.has_value()) {
            const cv::Mat graphic_size_probe = cv::imread(args.graphic_file);
            if (graphic_size_probe.empty()) {
                mx::system_err << "acmx2: Error: graphics file not found or unreadable: "
                               << args.graphic_file << "\n";
                mx::system_err.flush();
                return EXIT_FAILURE;
            }

            args.tw = graphic_size_probe.cols;
            args.th = graphic_size_probe.rows;
            if (args.frame_rotation == FrameRotation::Clockwise90 ||
                args.frame_rotation == FrameRotation::Counterclockwise90) {
                std::swap(args.tw, args.th);
            }
            mx::system_out << "acmx2: Graphics window initial size: "
                           << args.tw << "x" << args.th << "\n";
        } else if (!args.filename.empty() && !args.sizev.has_value()) {
            const std::optional<cv::Size> video_size =
                probe_video_size(args.filename);
            if (video_size.has_value()) {
                args.tw = video_size->width;
                args.th = video_size->height;
                if (args.frame_rotation == FrameRotation::Clockwise90 ||
                    args.frame_rotation == FrameRotation::Counterclockwise90) {
                    std::swap(args.tw, args.th);
                }
                mx::system_out << "acmx2: Video window initial size: "
                               << args.tw << "x" << args.th << "\n";
            } else {
                mx::system_out
                    << "acmx2: Could not probe video dimensions before window "
                       "creation; using the startup fallback size\n";
            }
        }

        SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
        SDL_SetHint("SDL_VIDEO_WAYLAND_WMCLASS", "acmx2");
        SDL_SetHint("SDL_VIDEO_X11_WMCLASS", "ACMX2");

        const OpenGLContextConfig context_config = select_open_gl_context();

        if (args.silent) {
            MainWindow main_window(args, context_config, true);
            main_window.loop();
        } else {
            MainWindow main_window(args, context_config);
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
