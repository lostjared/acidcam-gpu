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
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gl.hpp>
#include <iomanip>
#include <mutex>
#include <mx.hpp>
#include <mxwrite.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <map>
#include <vector>
#ifdef AUDIO_ENABLED
#include "audio.hpp"
#endif
#ifdef MIDI_ENABLED
#include <rtmidi/RtMidi.h>
#endif
#include "program.hpp"
#include <ac-gpu/ac-gpu.hpp>
#include <cuda_gl_interop.h>
#include <deque>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <model.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string_view>

/// @brief Copy the audio track from one media file to another via FFmpeg.
void transfer_audio(std::string_view, std::string_view);

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
    cudaGraphicsResource *cudaPboResource = nullptr; ///< CUDA handle to the mapped PBO.
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
        CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pboID, cudaGraphicsMapFlagsWriteDiscard));
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

    /**
     * @brief Release all GPU resources (CUDA registration, PBO, texture).
     *
     * Unregisters the PBO from CUDA, deletes the OpenGL buffer, and
     * deletes the OpenGL texture.  Safe to call multiple times—each
     * resource handle is tested for non-zero before deletion and
     * reset to zero / nullptr afterwards.
     */
    void cleanup() {
        if (cudaPboResource) {
            CHECK_CUDA(cudaGraphicsUnregisterResource(cudaPboResource));
            cudaPboResource = nullptr;
        }
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
    GLenum format_2d;              ///< GL binary format token for 2D.
    std::vector<char> binary_3d;   ///< GL program binary (3D vertex shader).
    GLenum format_3d;              ///< GL binary format token for 3D.
    uint64_t source_hash;          ///< FNV-1a-64 hash of the fragment source.
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
    static constexpr uint32_t CACHE_VERSION = 2;           ///< Current format version.
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
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open())
            return false;

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
    };
    size_t library_index = 0;
    bool use_cache = false;
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

  public:
    ShaderLibrary() = default;
    ~ShaderLibrary() {}

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
        size_t total_shaders = 0;
        {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && std::filesystem::exists(text + "/" + line) && line.find("material") == std::string::npos) {
                    total_shaders++;
                }
            }
            file.clear();
            file.seekg(0);
        }

        mx::system_out << "acmx2: Compiling " << total_shaders << " shaders (" << (dual_mode ? "2D+3D" : "2D") << ")...\n";
        fflush(stdout);

        int last_percent_reported = -1;
        size_t shader_index = 0;
        while (!file.eof()) {
            std::string line_data;
            std::getline(file, line_data);
            if (file && !line_data.empty() && std::filesystem::exists(text + "/" + line_data) && line_data.find("material") == std::string::npos) {
                programs_2d.push_back(makeProgram());
                try {
                    if (!programs_2d.back()->loadProgram(win->util.getFilePath("data/vert.glsl"), text + "/" + line_data)) {
                        mx::system_out << "acmx2: ❌ Failed to compile 2D shader: " << line_data << "\n";
                        fflush(stdout);
                        throw mx::Exception("\nacmx2: Error could not load 2D shader: " + line_data);
                    }
                } catch (mx::Exception &e) {
                    fflush(stdout);
                    fflush(stderr);
                    throw;
                }
                setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, text + "/" + line_data);
                if (dual_mode) {
                    programs_3d.push_back(makeProgram());
                    try {
                        if (!programs_3d.back()->loadProgram(win->util.getFilePath("data/vertex.glsl"), text + "/" + line_data)) {
                            mx::system_out << "acmx2: ❌ Failed to compile 3D shader: " << line_data << "\n";
                            fflush(stdout);
                            throw mx::Exception("acmx2: Error could not load 3D shader: " + line_data);
                        }
                    } catch (mx::Exception &e) {
                        fflush(stdout);
                        fflush(stderr);
                        throw;
                    }
                    setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, text + "/" + line_data);
                }
                shader_index++;

                int percent = static_cast<int>(shader_index * 100 / total_shaders);
                int percent_bucket = (percent / 10) * 10;
                if (percent_bucket > last_percent_reported) {
                    last_percent_reported = percent_bucket;
                    mx::system_out << "acmx2: Compiling... " << percent_bucket << "% (" << shader_index << "/" << total_shaders << " shaders)\n";
                    fflush(stdout);

                    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT);
                    if (loadingFont.handle().has_value()) {
                        std::string loadingText = "Compiling Shader " + std::to_string(shader_index) + "/" + std::to_string(total_shaders) + "...";
                        win->text.printText_Blended(loadingFont, 10, 10, loadingText);
                    }
                    SDL_GL_SwapWindow(win->getWindow());
                    SDL_PumpEvents();
                }
            }
        }
        file.close();
        mx::system_out << "acmx2: Compiled " << shader_index << " shaders (" << (dual_mode ? "2D+3D" : "2D only") << ")\n";
        fflush(stdout);
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

        std::string cache_file = library_path + "/.shader_cache";
        std::fstream file;
        file.open(library_path + "/index.txt", std::ios::in);
        if (!file.is_open()) {
            mx::system_err << "acmx2: Could not open index.txt at: " << library_path << "\n";
            return false;
        }

        ShaderCache cache;
        cache.gl_renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
        cache.gl_version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
        cache.dual_mode = dual_mode;

        std::vector<std::string> shader_files;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && std::filesystem::exists(library_path + "/" + line) && line.find("material") == std::string::npos) {
                shader_files.push_back(line);
            }
        }
        file.close();

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

            try {
                mx::system_out << "  - Compiling 2D shader... ";
                fflush(stdout);

                gl::ShaderProgram prog_2d;
                prog_2d.setSilent(true);
                if (!prog_2d.loadProgram(vert_2d, full_path)) {
                    mx::system_out << " ❌ (2D compile failed)\n";
                    fflush(stdout);
                    continue;
                }
                mx::system_out << "done (id=" << prog_2d.id() << ")\n";
                fflush(stdout);

                GLint link_status = 0;
                glGetProgramiv(prog_2d.id(), GL_LINK_STATUS, &link_status);
                if (link_status != GL_TRUE) {
                    mx::system_out << "  - ❌ Program not properly linked\n";
                    fflush(stdout);
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
                    continue;
                }

                if (binary_length > 0) {
                    mx::system_out << "  - Extracting binary... ";
                    fflush(stdout);

                    void *binary_buffer = malloc(binary_length);
                    if (!binary_buffer) {
                        mx::system_out << "❌ (malloc failed)\n";
                        fflush(stdout);
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
                        continue;
                    }

                    entry.binary_2d.resize(actual_length);
                    memcpy(entry.binary_2d.data(), binary_buffer, actual_length);
                    entry.format_2d = format;
                    free(binary_buffer);
                } else {
                    mx::system_out << " ❌ (no binary available)\n";
                    fflush(stdout);
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
                continue;
            } catch (...) {
                mx::system_out << " ❌ (unknown exception)\n";
                fflush(stdout);
                continue;
            }
        }

        if (cache.save(cache_file)) {
            mx::system_out << "acmx2: Shader cache saved to: " << cache_file << "\n";
            mx::system_out << "acmx2: Cached " << cache.entries.size() << " shaders (" << (dual_mode ? "2D+3D" : "2D only") << ")\n";
            fflush(stdout);
            return true;
        } else {
            mx::system_err << "acmx2: Failed to save shader cache\n";
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
        std::string cache_file = library_path + "/.shader_cache";

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

        std::string current_renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
        std::string current_version = reinterpret_cast<const char *>(glGetString(GL_VERSION));

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
            if (!line.empty() && std::filesystem::exists(library_path + "/" + line) && line.find("material") == std::string::npos) {
                shader_files.push_back(line);
            }
        }
        file.close();

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

        int last_percent_reported = -1;

        for (size_t i = 0; i < cache.entries.size(); ++i) {
            const auto &entry = cache.entries[i];

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
                return false;
            }

            *programs_2d.back() = gl::ShaderProgram(prog_id_2d);
            setupProgramUniforms(win, programs_2d.back().get(), program_names_2d, programs_2d.size() - 1, library_path + "/" + shader_files[i]);

            if (dual_mode && !entry.binary_3d.empty()) {
                programs_3d.push_back(makeProgram());
                GLuint prog_id_3d = glCreateProgram();
                glProgramBinaryFunc(prog_id_3d, entry.format_3d, entry.binary_3d.data(), static_cast<GLsizei>(entry.binary_3d.size()));

                glGetProgramiv(prog_id_3d, GL_LINK_STATUS, &link_status);
                if (link_status != GL_TRUE) {
                    mx::system_out << "acmx2: ❌ Shader " << i << " [" << entry.shader_name << "] 3D binary load failed\n";
                    fflush(stdout);
                    glDeleteProgram(prog_id_3d);
                    programs_3d.pop_back();
                    return false;
                }

                *programs_3d.back() = gl::ShaderProgram(prog_id_3d);
                setupProgramUniforms(win, programs_3d.back().get(), program_names_3d, programs_3d.size() - 1, library_path + "/" + shader_files[i]);
            }

            int percent = static_cast<int>((i + 1) * 100 / cache.entries.size());
            int percent_bucket = (percent / 10) * 10;
            if (percent_bucket > last_percent_reported) {
                last_percent_reported = percent_bucket;
                mx::system_out << "acmx2: Cache loading... " << percent_bucket << "% (" << (i + 1) << "/" << cache.entries.size() << " shaders)\n";
                fflush(stdout);

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
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
            time_f += static_cast<float>(delta_time) * time_speed;
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
    std::string library = "./filters";
    std::string fragment = "./frag.glsl";
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
    bool use_shader_cache = true;
    float time_speed = 1.0f;
    std::string playlist_file;
    double duration = 0.0;
    float cross_fade_duration = 0.5f;
};

/**
 * @struct FrameData
 * @brief A captured RGBA pixel buffer queued for the writer thread.
 *
 * Holds a vertically-flipped copy of the framebuffer contents
 * plus metadata for the async writer (dimensions, snapshot flag).
 */
struct FrameData {
    std::vector<unsigned char> pixels; ///< RGBA pixel data.
    int width = 0;                     ///< Frame width in pixels.
    int height = 0;                    ///< Frame height in pixels.
    bool isSnapshot = false;           ///< True if this frame should be saved as a PNG snapshot.
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
        case 32:  return "Space";
        case 44:  return "Comma";
        case 45:  return "Minus";
        case 46:  return "Period";
        case 47:  return "Slash";
        case 61:  return "Plus/Eq";
        case 65:  return "A";
        case 66:  return "B";
        case 68:  return "D";
        case 72:  return "H";
        case 76:  return "L";
        case 78:  return "N";
        case 80:  return "P";
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
        case 32:  return SDLK_SPACE;
        case 44:  return SDLK_COMMA;
        case 45:  return SDLK_MINUS;
        case 46:  return SDLK_PERIOD;
        case 47:  return SDLK_SLASH;
        case 61:  return SDLK_EQUALS;
        case 65:  return SDLK_a;
        case 66:  return SDLK_b;
        case 68:  return SDLK_d;
        case 72:  return SDLK_h;
        case 76:  return SDLK_l;
        case 78:  return SDLK_n;
        case 80:  return SDLK_p;
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
    int pboIndex = 0;
    int pboNextIndex = 1;
    SnapshotThreadPool snapshot_pool{2};
    TextureUploader tex_uploader;

  public:
    void requestStop() { running = false; }
    bool needsAsyncShutdown() { return needsMux() || needsTransferAudio(); }
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
          use_shader_cache_flag{args.use_shader_cache} {
#ifdef AUDIO_ENABLED
        audio_input_device = args.audio_input;
        audio_output_device = args.audio_output;
        audio_record_file = args.record_audio_file;
        if (args.audio_enabled) {
            if (init_audio(args.audio_channels, args.audio_sensitivty, audio_input_device, audio_output_device) != 0) {
                mx::system_err << "acmx2: Error could not initalize audio\n";
            } else {
                audio_is_enabled = true;
                set_record_gain(args.record_gain);
            }
        }

#endif
        library.is3D(args.is3d);
        library.setTimeSpeed(args.time_speed);
        is3d_enabled = args.is3d;
        m_file = args.model_file;

        gpu_filter_enabled = args.gpu_filter_enabled;
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
        counter_disabled = args.disable_counter;

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
    cv::cuda::GpuMat gpuWorkingBuffer;
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
    std::vector<int> saved_pass_list;
    bool saved_pass_enabled = false;
    double duration_limit = 0.0;

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

    void ensureCrossfadeFBO(int width, int height) {
        if (crossfadeFBO)
            return;
        glGenFramebuffers(1, &crossfadeFBO);
        glGenTextures(1, &crossfadeTexture);
        glBindTexture(GL_TEXTURE_2D, crossfadeTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, crossfadeFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, crossfadeTexture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw mx::Exception("acmx2: crossfade framebuffer is not complete");
        }
        glGenTextures(1, &crossfadePrevTexture);
        glBindTexture(GL_TEXTURE_2D, crossfadePrevTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

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
        if (d_ptrList) {
            cudaFree(d_ptrList);
            d_ptrList = nullptr;
        }
        if (d_filterList) {
            cudaFree(d_filterList);
            d_filterList = nullptr;
        }
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
            bool shouldMux = needsMux() && writer.is_open();
            stopWriterThread();
            if (shouldMux) {
                runMuxSync();
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
            close_audio();
        }
#endif

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
        cudaError_t cuda_err = cudaSetDevice(gpu_cuda_device);
        if (cuda_err != cudaSuccess) {
            throw mx::Exception("Failed to set CUDA device " + std::to_string(gpu_cuda_device) + ": " + std::string(cudaGetErrorString(cuda_err)));
        }
        mx::system_out << "acmx2: Using CUDA device: " << gpu_cuda_device << "\n";
        fflush(stdout);

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
                if (writer.open(ofilename, w, h, fps, crf.c_str())) {
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << crf
                                   << " FPS: " << fps << "\n";

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
                if (writer.open_ts(ofilename, w, h, fps, crf.c_str())) {
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << crf
                                   << " FPS: " << fps << "\n";
                } else {
                    throw mx::Exception("Could not open output video file: " + ofilename);
                }
            }
        } else if (!filename.empty() && graphic.empty()) {
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

            frame_w = w;
            frame_h = h;

            mx::system_out << "acmx2: Video opened: " << w << "x" << h
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
                if (writer.open(ofilename, w, h, fps, crf.c_str())) {
                    mx::system_out << "acmx2: Opened: " << ofilename
                                   << " for writing at: CRF: " << crf << "\n";
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
        int waterFontSize = std::max(12, static_cast<int>(win->h / 40.0f));
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
        sprite.setName("samp");
        sprite.initWithTexture(library.shader(), camera_texture, 0, 0, win->w, win->h);
        setupCaptureFBO(win->w, win->h);
        glGenBuffers(2, pboIds);
        size_t pboSize = win->w * win->h * 4;
        for (int i = 0; i < 2; i++) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, pboSize, nullptr, GL_STREAM_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        if (!graphic.empty())
            win->setWindowTitle("ACMX2 - Graphics Input");
        else if (filename.empty())
            win->setWindowTitle("ACMX2 - Capture Input");
        else
            win->setWindowTitle("ACMX2 - [" + filename + "] 0 seconds, frame 0");

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
        if (fps > 0.0) {
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
            if (needsMux() || needsTransferAudio()) {
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
                if (!cap.read(newFrame)) {
                    if (!filename.empty() && repeat) {
                        mx::system_out << "acmx2: video loop...\n";
                        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        if (!cap.read(newFrame)) {
                            mx::system_out << "acmx2: cannot read after looping.\n";
                        }
                    } else {
                        if (silent_mode) {
                            std::cout << "\n";
                        }
                        running = false;
                        finished = true;
                        return;
                    }
                }
                if (!newFrame.empty())
                    cv::flip(newFrame, newFrame, 0);
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
        if (!isFrozen && !newFrame.empty()) {
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

            if(keystate[SDL_SCANCODE_J]) {
                cameraRotationSpeed += 0.5f;
                if (cameraRotationSpeed > 50.0f) cameraRotationSpeed = 50.0f;
                    mx::system_out << "acmx2: Camera rotation speed: " << cameraRotationSpeed << "\n";
                    fflush(stdout);
            }
            if(keystate[SDL_SCANCODE_K]) {
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
            GLuint textureForMesh = camera_texture;
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
                    for (int p = 0; p < 2; ++p) {
                        glGenFramebuffers(1, &passFBO[p]);
                        glGenTextures(1, &passTexture[p]);
                        glBindTexture(GL_TEXTURE_2D, passTexture[p]);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, win->w, win->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glBindFramebuffer(GL_FRAMEBUFFER, passFBO[p]);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, passTexture[p], 0);
                        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                            throw mx::Exception("acmx2: 3D pass framebuffer is not complete");
                        }
                    }
                }

                GLuint inputTex = camera_texture;
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
            GLuint textureForSprite = camera_texture;
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
                    for (int p = 0; p < 2; ++p) {
                        glGenFramebuffers(1, &passFBO[p]);
                        glGenTextures(1, &passTexture[p]);
                        glBindTexture(GL_TEXTURE_2D, passTexture[p]);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, win->w, win->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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

                GLuint inputTex = camera_texture;
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

        fshader.useProgram();
        fshader.setUniform("mv_matrix", glm::mat4(1.0f));
        fshader.setUniform("proj_matrix", glm::mat4(1.0f));
        sprite.setShader(&fshader);
        sprite.draw(fboTexture, 0, 0, win->w, win->h);

        if (enableWatermark && writer.is_open() && waterFont.handle().has_value()) {
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glViewport(0, 0, win->w, win->h);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            win->text.setColor({255, 0, 150, 255});
            win->text.printText_Blended(waterFont, 10, 10, "LostSideDead.biz");
            glDisable(GL_BLEND);
        }

        bool needWriter = (writer.is_open() || snapshot_state > 0) && !isFrozen;

        if (needWriter) {

            if (snapshot_state == 1) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pboIndex]);
                glBindTexture(GL_TEXTURE_2D, fboTexture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);

                pboIndex = (pboIndex + 1) % 2;
                pboNextIndex = (pboNextIndex + 1) % 2;
                snapshot_state = 2;
            } else {
                bool is_snapshot_frame = (snapshot_state == 2);

                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[pboIndex]);
                glBindTexture(GL_TEXTURE_2D, fboTexture);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

                if (writer.is_open() || is_snapshot_frame) {
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
                        fd.isSnapshot = is_snapshot_frame;

                        if (is_snapshot_frame) {
                            snapshot_state = 0;
                        }

                        {
                            std::unique_lock<std::mutex> lock(queueMutex);
                            bool is_camera_mode = filename.empty() && graphic.empty();
                            if (is_camera_mode && !is_snapshot_frame) {
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

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);
                pboIndex = (pboIndex + 1) % 2;
                pboNextIndex = (pboNextIndex + 1) % 2;
            }
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

        } else if (cap.isOpened() && !filename.empty()) {
            frame_counter = static_cast<unsigned int>(cap.get(cv::CAP_PROP_POS_FRAMES));

            if (silent_mode && totalFrames > 0.0) {
                int current_percent = static_cast<int>((static_cast<double>(frame_counter) / totalFrames) * 100.0);
                if (current_percent > last_progress_percent && current_percent <= 100) {
                    last_progress_percent = current_percent;
                    int64_t frames_written = writer.is_open() ? writer.get_frame_count() : 0;
                    double elapsed_secs = static_cast<double>(frame_counter) / fps;
                    uint64_t hours = static_cast<uint64_t>(elapsed_secs / 3600);
                    uint64_t minutes = static_cast<uint64_t>(elapsed_secs / 60) % 60;
                    uint64_t seconds = static_cast<uint64_t>(elapsed_secs) % 60;

                    std::cout << "\racmx2: [" << std::setw(3) << current_percent << "%] "
                              << "Frame " << frame_counter << "/" << static_cast<int>(totalFrames)
                              << " | Written: " << frames_written
                              << " | Time: " << std::setfill('0') << std::setw(2) << hours << ":"
                              << std::setfill('0') << std::setw(2) << minutes << ":"
                              << std::setfill('0') << std::setw(2) << seconds
                              << std::setfill(' ') << "     " << std::flush;
                }
            }

            if (!silent_mode && std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count() >= 3) {
                if (totalFrames <= 0.0) {
                    totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
                }
                std::string timeStr = getTimeString();
                int64_t displayFrames = getFrameCount();
                std::ostringstream stream;
                stream << "ACMX2 - ["
                       << displayFrames << "/"
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
     * - Up/Down: Previous/next shader (or playlist entry if playlist enabled).
     * - Left/Right: Previous/next GPU CUDA filter.
     * - Space: Toggle shader bypass.
     * - P: Toggle playlist mode or pause video.
     * - L: Freeze frame (stop updating texture but keep time advancing).
     * - Z: Take a PNG snapshot.
     * - T: Toggle active time.  Q: Toggle audio time.  Home: Toggle audio delta.
     * - V: Toggle view rotation (3D).  O: Oscillation.  C: Wave.  X: Reset camera.
     * - 3: Toggle 2D/3D mode.  M: Toggle multi-pass.  E: Watermark.
     * - F9: Toggle HUD overlay visibility.
     *
     * Key bindings (SDL_KEYDOWN):
     * - U/I: Manual time step forward/backward.
     * - Insert/Delete: Audio sensitivity +/-.
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
                if (playlist_enabled && !playlist_tree.empty()) {
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
                if (playlist_enabled && !playlist_tree.empty()) {
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
            case SDLK_z:
                if (snapshot_state == 0) {
                    snapshot_state = 1;
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
                float s = get_sense() + 0.5f;
                set_sense(s);
                mx::system_out << "acmx2: Audio sensitivity increased to " << s << "\n";
                fflush(stdout);
                break;
            }
            case SDLK_DELETE: {
                float s = get_sense() - 0.5f;
                if (s < 0.1f) s = 0.1f;
                set_sense(s);
                mx::system_out << "acmx2: Audio sensitivity decreased to " << s << "\n";
                fflush(stdout);
                break;
            }
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
    double totalFrames = 0;
    cv::VideoCapture cap;
    cv::Mat graphic_frame;
    gl::GLSprite sprite;
    GLuint camera_texture = 0;
    GLuint captureFBO = 0;
    GLuint fboTexture = 0;
    GLuint depthBuffer = 0;
    GLuint passFBO[2] = {0, 0};
    GLuint passTexture[2] = {0, 0};
    GLuint crossfadeFBO = 0;
    GLuint crossfadeTexture = 0;
    GLuint crossfadePrevTexture = 0;
    gl::ShaderProgram crossfadeShader;
    float crossfadeAlpha = 1.0f;
    bool crossfadeActive = false;
    float crossfadeDuration = 0.5f;
    std::chrono::steady_clock::time_point crossfadeStartTime;
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
    int gpu_cuda_device = 0;
    bool silent_mode = false;
    bool use_shader_cache_flag = true;
    int last_progress_percent = -1;
    bool enableWatermark = false;

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

                while (writerRunning) {
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
                            std::string name = snap_prefix + "/ACMX2.Snapshot-" + oss.str() + "-" + std::to_string(fd.width) + "x" + std::to_string(fd.height) + "-" + std::to_string(current_offset) + ".png";

                            png::SavePNG_RGBA(name.c_str(),
                                              const_cast<unsigned char *>(fd.pixels.data()),
                                              fd.width, fd.height);

                            mx::system_out << "acmx2: Took snapshot: " << name << "\n";
                            fflush(stdout);
                        });
                    }

                    if (writer.is_open() && (!filename.empty() || !graphic.empty()) && written_frame_counter == 0) {
                        written_frame_counter++;
                        continue;
                    } else if (writer.is_open() && written_frame_counter <= 30 && filename.empty() && graphic.empty()) {
                        written_frame_counter++;
                        continue;
                    }
#ifdef AUDIO_ENABLED
                    if (writerRunning && audio_is_enabled && !audio_record_file.empty() && !is_audio_recording()) {
                        if (!start_audio_recording(audio_record_file)) {
                            mx::system_err << "acmx2: Error could not start audio recording\n";
                        }
                    }
#endif

                    if (writer.is_open() && !fd.isSnapshot) {
                        if (!filename.empty() || !graphic.empty()) {
                            writer.write(fd.pixels.data());
                        } else {
                            writer.write_ts(fd.pixels.data());
                        }
                        written_frame_counter++;
                    }
                }
            } catch (const std::exception &e) {
                mx::system_err << "acmx2: Writer thread exception: " << e.what() << "\n";
                writerRunning = false;
                running = false;
                fflush(stderr);
                fflush(stdout);
            }
        });
    }

    /**
     * @brief Check whether post-recording audio muxing is required.
     *
     * Returns true only when audio is enabled, a recording file
     * was specified, AND an output video file exists.  Used to
     * decide whether to launch the mux thread on shutdown.
     *
     * @return True if ffmpeg audio muxing should run.
     */
    bool needsMux() {
#ifdef AUDIO_ENABLED
        return audio_is_enabled && !audio_record_file.empty() && !ofilename.empty();
#else
        return false;
#endif
    }

    bool needsTransferAudio() {
        return !filename.empty() && !repeat && copy_audio && writer.is_open();
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
        if (is_audio_recording()) {
            stop_audio_recording();
        }
        std::string tmp_out = ofilename + ".tmp.mp4";
        int64_t fc = writer.get_frame_count();
        double video_duration = (fps > 0.0 && fc > 0) ? static_cast<double>(fc) / fps : 0.0;
        std::ostringstream cmd;
        cmd << "ffmpeg -y -i \"" << ofilename << "\" -i \"" << audio_record_file
            << "\" -map 0:v:0 -map 1:a:0"
            << " -c:v copy -c:a aac -b:a 192k";
        if (video_duration > 0.0) {
            cmd << " -t " << std::fixed << std::setprecision(3) << video_duration;
        }
        cmd << " -movflags +faststart \""
            << tmp_out << "\" 2>&1";
        mx::system_out << "acmx2: muxing recorded audio into video...\n";
        fflush(stdout);
        int ret = std::system(cmd.str().c_str());
        if (ret == 0) {
            std::remove(ofilename.c_str());
            std::rename(tmp_out.c_str(), ofilename.c_str());
            mx::system_out << "acmx2: muxed recorded audio from: " << audio_record_file << " to " << ofilename << "\n";
        } else {
            mx::system_err << "acmx2: ffmpeg mux failed (exit code " << ret << ")\n";
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
            if (!filename.empty() && !repeat && copy_audio) {
                transfer_audio(filename, ofilename);
                mx::system_out << "acmx2: copied audio track from: " << filename << " to " << ofilename << "\n";
            }
            runMuxSync();
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
            if (!filename.empty() && repeat == false && copy_audio) {
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
        object->draw(this);
        swap();
        delay();
    }

    /// @brief Placeholder—SDL events are forwarded to ACView::event() by libmx2.
    void event(SDL_Event &e) override {
    }
};

const char *message = R"(
-[ Keyboard controls ]- {
    Up arrow - Previous shader
    Down arrow - Next shader
    Left - Previous GPU filter (if enabled)
    Right - Next GPU filter (if enabled)
    Space - Enable/Disable Processing
    L - Enable/Disable video freeze (Video/Image Modes)
    P - Enable/Disable pause video (Video/Image Modes)
    T - enable/disable time
    U/I - step time if not disabled
    Page Up/Page Down - increase/decrease time speed
    Z - take snapshot
    3 - toggle 2D/3D mode
    M - toggle multi-pass
    F - toggle fullscreen
    Q - toggle reactive time (if AUDIO_ENABLED)
    Insert - increase audio sensitivity
    Delete - decrease audio sensitivity
    Home - toggle audio delta time scaling on/off
    M - toggle multi-shader pass (if --shader-pass set)
    3 - toggle 2D/3D mode (switches between 2D and 3D rendering)
    3D mode controls:
    W,A,S,D - Look around 
    V - Toggle view rotation
    O - Oscillation Toggle
    X - Reset camera distance
    ( +, - ) - increase / decrease camera distance
    B - increase movement speed
    N - decrease movement speed
    C - Toggle Object Wave
    E - Enable/Disable Watermark
}
)";

/// @brief Verify CUDA device availability and print GPU info.
void checkDevices(bool list_only = false) {
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
}

/// @brief Print program version, author, arguments, and keyboard controls.
template <typename T>
void printAbout(Argz<T> &parser) {
    mx::system_out << PROGRAM_NAME << ": " << VERSION_INFO << "\n";
    mx::system_out << "(C) 2026 " << VERSION_AUTHOR << "\n";
    mx::system_out << "https://lostsidedead.biz\n";
    mx::system_out << "Command Line Arguments:\n";
    parser.help(mx::system_out);
    mx::system_out << message;
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
#endif
        .addOptionDouble('N', "fullscreen", "Fullscreen Window (Escape to quit)")
        .addOptionDouble(405, "silent", "Silent mode - process video without window, (video files only)")
        .addOptionDoubleValue(406, "shader-pass", "Shader pass indices (comma-separated, e.g. 0,1,2)")
        .addOptionDoubleValue(407, "build", "Build shader cache for specified library path (compiles shaders and exits)")
        .addOptionDouble(408, "no-cache", "Disable shader caching (always recompile shaders)")
        .addOptionDoubleValue(409, "time-speed", "Constant time_f speed multiplier (default: 1.0)")
        .addOptionDoubleValue(410, "playlist", "Shader playlist text file (one shader name per line, P to toggle)")
        .addOptionDoubleValue(411, "duration", "Recording duration in seconds (float); stop recording and exit after elapsed")
        .addOptionDoubleValue(412, "cross-fade", "Crossfade duration in seconds when switching playlist shaders (default: 0.5)")
#ifdef MIDI_ENABLED
        .addOptionDoubleValue(500, "midi-map", "MIDI config file (.midi_cfg)")
        .addOptionDoubleValue(501, "midi-device", "MIDI input device index")
        .addOptionDouble(502, "list-midi", "List available MIDI input devices")
#endif
    ;

    if (argc == 1) {
        printAbout(parser);
        exit(EXIT_SUCCESS);
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
                printAbout(parser);
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
        args.path = ".";
        mx::system_out << "acmx2: Path name not provided, using current path...\n";
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

                        std::string vert_2d = util.getFilePath("data/vert.glsl");
                        std::string vert_3d = util.getFilePath("data/vertex.glsl");

                        mx::system_out << "acmx2: Building shader cache for: " << lib_path << "\n";
                        mx::system_out << "acmx2: Mode: " << (enable_3d ? "2D+3D" : "2D only") << "\n";
                        mx::system_out << "acmx2: OpenGL Renderer: " << glGetString(GL_RENDERER) << "\n";
                        mx::system_out << "acmx2: OpenGL Version: " << glGetString(GL_VERSION) << "\n";
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
        }

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