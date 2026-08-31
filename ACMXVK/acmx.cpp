/**
 * @file acmx.cpp
 * @brief ACMXVK real-time Vulkan video shader application.
 */

#include <mxvk/mxvk.hpp>
#include <mxvk/mxvk_abstract_model.hpp>
#include <mxvk/mxvk_cv.hpp>
#include <mxvk/mxvk_exception.hpp>
#ifdef MXVK_WITH_FFMPEG_CAPTURE
#include <mxvk/mxvk_ff_capture.hpp>
#endif
#include <mxvk/mxvk_png.hpp>
#include <mxwrite.hpp>

#ifdef ACMXVK_WITH_WEBP
#include <webp/encode.h>
#endif
#ifdef ACMXVK_WITH_TIFF
#include <tiffio.h>
#endif

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#ifdef AUDIO_ENABLED
#include "audio.hpp"
#include "file_audio.hpp"
#endif
#ifdef MIDI_ENABLED
#include "midi.hpp"
#endif
#ifdef ACMXVK_WITH_CUDA
#include "gpu_filters.hpp"
#endif
#ifdef ACMXVK_WITH_DNN
#include "edge_dnn.hpp"
#endif
#include "input_validation.hpp"
#include "interface_control.hpp"
#ifdef ACMXVK_WITH_MXVK_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#ifndef ACMXVK_BUILD_SPRITE_VERTEX_SHADER
#define ACMXVK_BUILD_SPRITE_VERTEX_SHADER "sprite.vert.spv"
#endif

#ifndef ACMXVK_INSTALL_SPRITE_VERTEX_SHADER
#define ACMXVK_INSTALL_SPRITE_VERTEX_SHADER "sprite.vert.spv"
#endif

#ifndef ACMXVK_BUILD_ECHO_CACHE_SHADER
#define ACMXVK_BUILD_ECHO_CACHE_SHADER "echo_cache.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_ECHO_CACHE_SHADER
#define ACMXVK_INSTALL_ECHO_CACHE_SHADER "echo_cache.frag.spv"
#endif

#ifndef ACMXVK_BUILD_FLIP_SHADER
#define ACMXVK_BUILD_FLIP_SHADER "flip.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_FLIP_SHADER
#define ACMXVK_INSTALL_FLIP_SHADER "flip.frag.spv"
#endif

#ifndef ACMXVK_BUILD_PASSTHROUGH_SHADER
#define ACMXVK_BUILD_PASSTHROUGH_SHADER "passthrough.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_PASSTHROUGH_SHADER
#define ACMXVK_INSTALL_PASSTHROUGH_SHADER "passthrough.frag.spv"
#endif

#ifndef ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER
#define ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER "human_composite.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER
#define ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER "human_composite.frag.spv"
#endif

#ifndef ACMXVK_BUILD_MODEL_VERTEX_SHADER
#define ACMXVK_BUILD_MODEL_VERTEX_SHADER "model.vert.spv"
#endif

#ifndef ACMXVK_INSTALL_MODEL_VERTEX_SHADER
#define ACMXVK_INSTALL_MODEL_VERTEX_SHADER "model.vert.spv"
#endif

#ifndef ACMXVK_BUILD_MODEL_FRAGMENT_SHADER
#define ACMXVK_BUILD_MODEL_FRAGMENT_SHADER "model.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER
#define ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER "model.frag.spv"
#endif

#ifndef ACMXVK_BUILD_DEFAULT_MODEL
#define ACMXVK_BUILD_DEFAULT_MODEL "cube.obj"
#endif

#ifndef ACMXVK_INSTALL_DEFAULT_MODEL
#define ACMXVK_INSTALL_DEFAULT_MODEL "cube.obj"
#endif

#ifndef ACMXVK_BUILD_OVERLAY_FONT
#define ACMXVK_BUILD_OVERLAY_FONT "font.ttf"
#endif

#ifndef ACMXVK_INSTALL_OVERLAY_FONT
#define ACMXVK_INSTALL_OVERLAY_FONT "font.ttf"
#endif

#ifndef ACMXVK_BUILD_RESOURCE_DIRECTORY
#define ACMXVK_BUILD_RESOURCE_DIRECTORY "."
#endif

#ifndef ACMXVK_INSTALL_RESOURCE_DIRECTORY
#define ACMXVK_INSTALL_RESOURCE_DIRECTORY "."
#endif

#ifndef ACMXVK_BUILD_CROSSFADE_DIRECTORY
#define ACMXVK_BUILD_CROSSFADE_DIRECTORY "shaders/xfade"
#endif

#ifndef ACMXVK_INSTALL_CROSSFADE_DIRECTORY
#define ACMXVK_INSTALL_CROSSFADE_DIRECTORY "shaders/xfade"
#endif

namespace acmxvk {
    namespace fs = std::filesystem;

    constexpr int MAX_FRAME_DIMENSION = 16384;
    constexpr std::int64_t MAX_FRAME_PIXELS = 67108864;
    constexpr std::array<std::string_view, 35> CROSSFADE_NAMES{
        "xfade_01_linear", "xfade_02_block",
        "xfade_03_wipe", "xfade_04_radial",
        "xfade_05_pixelate", "xfade_06_dissolve",
        "xfade_07_swirl", "xfade_08_glitch",
        "xfade_09_diamond", "xfade_10_burn",
        "xfade_11_fade_black", "xfade_12_fade_white",
        "xfade_13_slide_left", "xfade_14_slide_right",
        "xfade_15_slide_up", "xfade_16_slide_down",
        "xfade_17_diagonal_wipe", "xfade_18_iris_open",
        "xfade_19_iris_close", "xfade_20_checker",
        "xfade_21_blinds_h", "xfade_22_blinds_v",
        "xfade_23_zoom_in", "xfade_24_zoom_out",
        "xfade_25_rotate", "xfade_26_ripple",
        "xfade_27_wave", "xfade_28_chroma",
        "xfade_29_invert", "xfade_30_flash",
        "xfade_31_explode", "xfade_32_mosaic",
        "xfade_33_shutter", "xfade_34_luma",
        "xfade_35_noise"};

    [[nodiscard]] bool dimensions_supported(int width, int height) {
        return width > 0 && height > 0 && width <= MAX_FRAME_DIMENSION &&
               height <= MAX_FRAME_DIMENSION &&
               static_cast<std::int64_t>(width) * height <= MAX_FRAME_PIXELS;
    }

    enum class FrameRotation { None,
                               Clockwise90,
                               Rotate180,
                               Counterclockwise90 };

    struct Options {
        int width = 1280;
        int height = 720;
        int camera_width = 1280;
        int camera_height = 720;
        int camera_device = 0;
        int shader_index = 0;
        int encode_crf = 18;
        int autopilot_frames = 0;
        int autopilot_random_timeout = 0;
        int generate_interval = 0;
        int cache_delay = 1;
        int texture_cache_size = 8;
        int audio_channels = 2;
        int audio_input_device = -1;
        int audio_output_device = -1;
        int audio_buffers = 0;
        int midi_device = -1;
        int gpu_frame_buffer_size = 10;
        int cuda_device = 0;
        double requested_fps = 0.0;
        double duration = 0.0;
        double cross_fade_duration = 0.5;
        double time_speed = 1.0;
        double max_size_mb = 0.0;
        double audio_sensitivity = 1.0;
        double audio_warm_rate = 0.5;
        double audio_pass_through_gain = 1.0;
        double audio_recording_gain = 1.0;
        double human_black_point = 0.35;
        double human_white_point = 0.75;
        bool resolution_specified = false;
        bool use_yuv = false;
        bool maximize_fps = false;
        bool use_source_fps = false;
        bool use_source_audio = false;
        bool fullscreen = false;
        bool repeat = false;
        bool enable_vsync = false;
        bool enable_screenshot = false;
        bool enable_playlist = false;
        bool enable_texture_cache = false;
        bool history_test = false;
        bool enable_3d = false;
        bool normalized_time = false;
        bool flip_output = false;
        bool png_output = false;
        bool encode_realtime = false;
        bool no_drop = false;
        bool copy_audio = false;
        bool mute_output = false;
        bool enable_audio = false;
        bool audio_input_specified = false;
        bool audio_warm_rate_specified = false;
        bool audio_output_specified = false;
        bool audio_pass_through_gain_specified = false;
        bool audio_recording_gain_specified = false;
        bool audio_pass_through = false;
        bool audio_repeat = false;
        bool audio_trunc = false;
        bool list_audio_devices = false;
        bool check_audio = false;
        bool midi_device_specified = false;
        bool midi_monitor = false;
        bool list_midi_devices = false;
        bool check_midi = false;
        bool gpu_buffer_specified = false;
        bool cuda_device_specified = false;
        bool list_gpu_filters = false;
        bool list_cuda_devices = false;
        bool check_cuda = false;
        bool check_dnn = false;
        bool human_background = false;
        bool human_black_specified = false;
        bool human_white_specified = false;
        bool list_encoders = false;
        bool display_filter = false;
        bool disable_counter = false;
        bool build_fix = false;
        bool build_prune = false;
        bool build_force = false;
        bool unbuffered_output = false;
        bool interface_shm = false;
        bool show_help = false;
        FrameRotation frame_rotation = FrameRotation::None;
        std::vector<int> shader_pass_indices;
        std::vector<std::string> shader_pass_files;
        std::vector<std::string> custom_uniform_overrides;
        std::vector<std::string> midi_cc_mappings;
        std::vector<int> gpu_filter_indices;
        std::string input_file;
        std::string graphic_file;
        std::string shader_directory;
        std::string fragment_shader;
        std::string compute_shader;
        std::string shader_file;
        std::string build_manifest;
        std::string build_directory;
        std::string glslc_executable = "glslc";
        std::string model_file;
        std::string playlist_file;
        std::string output_file;
        std::string encode_preset = "medium";
        std::string encode_tune;
        std::string encode_codec = "auto";
        std::string encode_params;
        std::string list_encoder_options;
        std::string audio_file;
        std::string record_audio_file;
        std::string midi_map_file;
        std::string edge_model;
        std::string human_model;
        std::string onnx_configuration;
        std::string snapshot_directory = ".";
        std::string resource_directory;
        std::string watermark_text;
        std::array<std::uint8_t, 3> watermark_color{255U, 0U, 150U};
    };

    [[nodiscard]] std::vector<fs::path>
    resourceDirectories(const Options &options) {
        std::vector<fs::path> directories;
        const auto append = [&](const fs::path &directory) {
            if (directory.empty()) {
                return;
            }
            const fs::path normalized = fs::absolute(directory).lexically_normal();
            if (std::find(directories.begin(), directories.end(), normalized) ==
                directories.end()) {
                directories.push_back(normalized);
            }
        };
        append(options.resource_directory);
        append(ACMXVK_INSTALL_RESOURCE_DIRECTORY);
        append(ACMXVK_BUILD_RESOURCE_DIRECTORY);
        append(fs::current_path());
        return directories;
    }

    [[nodiscard]] fs::path findResource(const Options &options,
                                        const fs::path &relative_path) {
        if (relative_path.empty() || relative_path.is_absolute()) {
            return {};
        }
        const fs::path normalized_relative = relative_path.lexically_normal();
        const std::string relative_text = normalized_relative.generic_string();
        if (relative_text == ".." || relative_text.starts_with("../") ||
            relative_text.find("/../") != std::string::npos) {
            return {};
        }
        for (const fs::path &directory : resourceDirectories(options)) {
            const fs::path candidate =
                (directory / normalized_relative).lexically_normal();
            if (fs::is_regular_file(candidate)) {
                return candidate;
            }
        }
        return {};
    }

    [[nodiscard]] bool hasShaderManifest(const fs::path &directory) {
        return fs::is_regular_file(directory / "library.json") ||
               fs::is_regular_file(directory / "index.txt");
    }

    [[nodiscard]] std::string optionValue(int &index, int argc, char **argv,
                                          std::string_view option) {
        if (++index >= argc) {
            throw std::runtime_error("missing value for " + std::string(option));
        }
        const std::string value(argv[index]);
        input::validate_string(value, input::StringKind::Argument,
                               std::string(option) + " value");
        return value;
    }

    void validateLocator(std::string_view value, std::string_view context,
                         bool allow_empty = false) {
        if (value.empty() && allow_empty) {
            return;
        }
        const input::StringKind kind =
            value.find("://") == std::string_view::npos
                ? input::StringKind::Path
                : input::StringKind::Url;
        input::validate_string(value, kind, context, allow_empty);
    }

    void validateOptionStrings(const Options &options) {
        validateLocator(options.input_file, "--input", true);
        input::validate_string(options.graphic_file, input::StringKind::Path,
                               "--graphic", true);
        input::validate_string(options.audio_file, input::StringKind::Path,
                               "--audio-file", true);
        input::validate_string(options.output_file, input::StringKind::Path,
                               "--output", true);
        input::validate_string(options.record_audio_file,
                               input::StringKind::Path, "--record-audio", true);
        input::validate_string(options.resource_directory,
                               input::StringKind::Path, "--path", true);
        input::validate_string(options.snapshot_directory,
                               input::StringKind::Path, "--prefix");
        input::validate_string(options.shader_directory,
                               input::StringKind::Path, "--shaders", true);
        input::validate_string(options.fragment_shader,
                               input::StringKind::Path, "--fragment", true);
        input::validate_string(options.compute_shader,
                               input::StringKind::Path, "--compute", true);
        input::validate_string(options.shader_file, input::StringKind::Path,
                               "--shader-file", true);
        input::validate_string(options.model_file, input::StringKind::Path,
                               "--model", true);
        input::validate_string(options.playlist_file, input::StringKind::Path,
                               "--playlist", true);
        input::validate_string(options.midi_map_file, input::StringKind::Path,
                               "--midi-map", true);
        input::validate_string(options.edge_model, input::StringKind::Path,
                               "--edge", true);
        input::validate_string(options.human_model, input::StringKind::Path,
                               "--human", true);
        input::validate_string(options.onnx_configuration,
                               input::StringKind::Path, "--onnx", true);

        for (const std::string &path : options.shader_pass_files) {
            input::validate_string(path, input::StringKind::Path,
                                   "--shader-pass-files entry");
        }
        for (const std::string &value : options.custom_uniform_overrides) {
            input::validate_string(value, input::StringKind::StructuredValue,
                                   "--uniform");
        }
        for (const std::string &value : options.midi_cc_mappings) {
            input::validate_string(value, input::StringKind::StructuredValue,
                                   "--midi-cc");
        }

        input::validate_string(options.encode_preset, input::StringKind::Token,
                               "--encode-preset");
        input::validate_string(options.encode_tune, input::StringKind::Token,
                               "--encode-tune", true);
        input::validate_string(options.encode_codec, input::StringKind::Token,
                               "--encode-codec");
        input::validate_string(options.encode_params,
                               input::StringKind::StructuredValue,
                               "--encode-params", true);
        input::validate_string(options.list_encoder_options,
                               input::StringKind::Token,
                               "--list-encoder-options", true);
        input::validate_string(options.watermark_text,
                               input::StringKind::DisplayText,
                               "--use-watermark", true);
    }

    [[nodiscard]] int parseInteger(std::string_view text, std::string_view option) {
        input::validate_string(text, input::StringKind::StructuredValue,
                               option);
        std::size_t parsed = 0;
        int value = 0;
        try {
            value = std::stoi(std::string(text), &parsed);
        } catch (const std::exception &) {
            throw std::runtime_error("invalid integer for " + std::string(option) + ": " +
                                     std::string(text));
        }
        if (parsed != text.size()) {
            throw std::runtime_error("invalid integer for " + std::string(option) + ": " +
                                     std::string(text));
        }
        return value;
    }

    [[nodiscard]] double parseNumber(std::string_view text, std::string_view option) {
        input::validate_string(text, input::StringKind::StructuredValue,
                               option);
        std::size_t parsed = 0;
        double value = 0.0;
        try {
            value = std::stod(std::string(text), &parsed);
        } catch (const std::exception &) {
            throw std::runtime_error("invalid number for " + std::string(option) + ": " +
                                     std::string(text));
        }
        if (parsed != text.size() || !std::isfinite(value)) {
            throw std::runtime_error("invalid number for " + std::string(option) + ": " +
                                     std::string(text));
        }
        return value;
    }

    [[nodiscard]] std::vector<int>
    parseIntegerList(std::string_view text, std::string_view option) {
        input::validate_string(text, input::StringKind::StructuredValue,
                               option);
        if (text.empty()) {
            throw std::runtime_error("empty integer list for " +
                                     std::string(option));
        }
        std::vector<int> values;
        std::size_t start = 0;
        while (start <= text.size()) {
            const std::size_t separator = text.find(',', start);
            const std::size_t end = separator == std::string_view::npos
                                        ? text.size()
                                        : separator;
            if (end == start) {
                throw std::runtime_error("invalid integer list for " +
                                         std::string(option) + ": " +
                                         std::string(text));
            }
            values.push_back(parseInteger(text.substr(start, end - start),
                                          option));
            if (separator == std::string_view::npos) {
                break;
            }
            start = separator + 1;
        }
        return values;
    }

    [[nodiscard]] std::array<std::uint8_t, 3>
    parseColor(std::string_view text, std::string_view option) {
        const std::vector<int> components = parseIntegerList(text, option);
        if (components.size() != 3U) {
            throw std::runtime_error(std::string(option) +
                                     " requires r,g,b");
        }

        std::array<std::uint8_t, 3> color{};
        for (std::size_t index = 0; index < color.size(); ++index) {
            if (components[index] < 0 || components[index] > 255) {
                throw std::runtime_error(std::string(option) +
                                         " components must be between 0 and 255");
            }
            color[index] = static_cast<std::uint8_t>(components[index]);
        }
        return color;
    }

    void parseDimensions(std::string_view text, int &width, int &height,
                         std::string_view option) {
        const std::size_t separator = text.find_first_of("xX");
        if (separator == std::string_view::npos) {
            throw std::runtime_error("invalid dimensions for " + std::string(option) +
                                     "; expected WidthxHeight");
        }

        width = parseInteger(text.substr(0, separator), option);
        height = parseInteger(text.substr(separator + 1), option);
        if (!dimensions_supported(width, height)) {
            throw std::runtime_error(
                "dimensions are outside the supported range for " +
                std::string(option));
        }
    }

    [[nodiscard]] FrameRotation parseFrameRotation(std::string value) {
        input::validate_string(value, input::StringKind::Token, "--rotate");
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        if (value == "clockwise" || value == "cw" || value == "90" || value == "90cw") {
            return FrameRotation::Clockwise90;
        }
        if (value == "180") {
            return FrameRotation::Rotate180;
        }
        if (value == "counterclockwise" || value == "ccw" || value == "90ccw" ||
            value == "270") {
            return FrameRotation::Counterclockwise90;
        }
        throw std::runtime_error(
            "--rotate requires clockwise, 180, or counterclockwise");
    }

    [[nodiscard]] bool isUtilityRequest(const Options &options) {
        return options.show_help || options.list_audio_devices ||
               options.check_audio || options.list_midi_devices ||
               options.check_midi || options.list_gpu_filters ||
               options.list_cuda_devices || options.check_cuda ||
               options.check_dnn ||
               options.list_encoders || !options.list_encoder_options.empty();
    }

    void applyResourceDefaults(Options &options) {
        if (isUtilityRequest(options)) {
            return;
        }

        std::string resource_source;
        if (!options.resource_directory.empty()) {
            resource_source = "--path";
        } else if (const char *environment = std::getenv("ACMXVK_PATH");
                   environment != nullptr && environment[0] != '\0') {
            input::validate_string(environment, input::StringKind::Path,
                                   "ACMXVK_PATH");
            options.resource_directory = environment;
            resource_source = "ACMXVK_PATH";
        } else if (const char *environment = std::getenv("ACMX2_PATH");
                   environment != nullptr && environment[0] != '\0') {
            input::validate_string(environment, input::StringKind::Path,
                                   "ACMX2_PATH");
            const fs::path compatibility_directory =
                fs::absolute(environment).lexically_normal();
            if (fs::is_directory(compatibility_directory)) {
                options.resource_directory = compatibility_directory.string();
                resource_source = "ACMX2_PATH compatibility fallback";
            } else {
                std::cerr << "acmxvk: ignoring unavailable ACMX2_PATH: "
                          << compatibility_directory.string() << '\n';
            }
        }

        if (!options.resource_directory.empty()) {
            const fs::path directory =
                fs::absolute(options.resource_directory).lexically_normal();
            if (!fs::is_directory(directory)) {
                throw std::runtime_error(resource_source +
                                         " is not a readable resource directory: " +
                                         directory.string());
            }
            options.resource_directory = directory.string();
            std::cout << "acmxvk: resource path (" << resource_source
                      << "): " << options.resource_directory << '\n';
        }

        if (!options.shader_directory.empty() ||
            !options.fragment_shader.empty() ||
            !options.compute_shader.empty()) {
            return;
        }

        if (const char *environment = std::getenv("ACMXVK_SHADER_PATH");
            environment != nullptr && environment[0] != '\0') {
            input::validate_string(environment, input::StringKind::Path,
                                   "ACMXVK_SHADER_PATH");
            fs::path directory = fs::absolute(environment).lexically_normal();
            if (fs::is_regular_file(directory) &&
                (directory.filename() == "library.json" ||
                 directory.filename() == "index.txt")) {
                directory = directory.parent_path();
            }
            if (!fs::is_directory(directory) || !hasShaderManifest(directory)) {
                throw std::runtime_error(
                    "ACMXVK_SHADER_PATH does not contain library.json or index.txt: " +
                    directory.string());
            }
            options.shader_directory = directory.string();
            std::cout << "acmxvk: shader library (ACMXVK_SHADER_PATH): "
                      << options.shader_directory << '\n';
            return;
        }

        for (const fs::path &resource_directory :
             resourceDirectories(options)) {
            const fs::path shader_directory = resource_directory / "shaders";
            if (hasShaderManifest(shader_directory)) {
                options.shader_directory = shader_directory.string();
                std::cout << "acmxvk: shader library (resource path): "
                          << options.shader_directory << '\n';
                return;
            }
        }
    }

    [[nodiscard]] Options parseOptions(int argc, char **argv) {
        Options options;
        if (argc < 0 || argv == nullptr || argc > 4096) {
            throw std::runtime_error(
                "command line contains an invalid number of arguments");
        }
        if (argc == 1) {
            options.show_help = true;
            return options;
        }

        const bool library_build_requested = [&] {
            for (int index = 1; index < argc; ++index) {
                const std::string_view argument(argv[index]);
                if (argument == "--build" || argument == "--builddir" ||
                    argument == "--fix" || argument == "--prune" ||
                    argument == "--force" || argument == "--glslc") {
                    return true;
                }
            }
            return false;
        }();
        if (library_build_requested) {
            for (int index = 1; index < argc; ++index) {
                const std::string_view option(argv[index]);
                input::validate_string(option, input::StringKind::Token,
                                       "command-line option");
                if (option == "-h" || option == "-v" ||
                    option == "--help" || option == "--version") {
                    options.show_help = true;
                } else if (option == "--unbuffered") {
                    options.unbuffered_output = true;
                } else if (option == "--build") {
                    if (!options.build_manifest.empty()) {
                        throw std::runtime_error(
                            "--build may only be supplied once");
                    }
                    options.build_manifest =
                        optionValue(index, argc, argv, option);
                } else if (option == "--builddir") {
                    if (!options.build_directory.empty()) {
                        throw std::runtime_error(
                            "--builddir and --fix are mutually exclusive");
                    }
                    options.build_directory =
                        optionValue(index, argc, argv, option);
                } else if (option == "--fix") {
                    if (!options.build_directory.empty()) {
                        throw std::runtime_error(
                            "--builddir and --fix are mutually exclusive");
                    }
                    options.build_directory =
                        optionValue(index, argc, argv, option);
                    options.build_fix = true;
                } else if (option == "--prune") {
                    if (options.build_prune) {
                        throw std::runtime_error(
                            "--prune may only be supplied once");
                    }
                    options.build_prune = true;
                } else if (option == "--force") {
                    if (options.build_force) {
                        throw std::runtime_error(
                            "--force may only be supplied once");
                    }
                    options.build_force = true;
                } else if (option == "--glslc") {
                    options.glslc_executable =
                        optionValue(index, argc, argv, option);
                } else {
                    throw std::runtime_error(
                        "library build mode cannot be combined with option: " +
                        std::string(option));
                }
            }
            input::validate_string(options.build_manifest,
                                   input::StringKind::Path, "--build", true);
            input::validate_string(options.build_directory,
                                   input::StringKind::Path,
                                   options.build_fix ? "--fix" : "--builddir",
                                   true);
            input::validate_string(options.glslc_executable,
                                   input::StringKind::Path, "--glslc");
            if (!options.show_help && options.build_manifest.empty()) {
                throw std::runtime_error(
                    "--builddir/--fix/--prune/--force/--glslc requires "
                    "--build <library.json>");
            }
            if (!options.show_help && options.build_directory.empty()) {
                throw std::runtime_error(
                    "--build requires --builddir or --fix <output-directory>");
            }
            if (!options.show_help && options.build_prune &&
                !options.build_fix) {
                throw std::runtime_error(
                    "--prune requires --fix <output-directory>");
            }
            if (!options.show_help && options.build_force &&
                !options.build_prune) {
                throw std::runtime_error(
                    "--force is only valid with --prune");
            }
            if (!options.show_help && options.build_prune &&
                !options.build_force) {
                throw std::runtime_error(
                    "WARNING: --prune permanently deletes .frag and .comp "
                    "source files that fail compilation; use --force to "
                    "confirm");
            }
            return options;
        }

        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            input::validate_string(option, input::StringKind::Token,
                                   "command-line option");
            if (option == "-h" || option == "-v" || option == "--help" ||
                option == "--version") {
                options.show_help = true;
            } else if (option == "--unbuffered") {
                options.unbuffered_output = true;
            } else if (option == "--interface-shm") {
                options.interface_shm = true;
            } else if (option == "-p" || option == "--path") {
                options.resource_directory =
                    optionValue(index, argc, argv, option);
                if (options.resource_directory.empty()) {
                    throw std::runtime_error(
                        "resource path must not be empty");
                }
            } else if (option == "-i" || option == "--input") {
                options.input_file = optionValue(index, argc, argv, option);
            } else if (option == "-g" || option == "--graphic") {
                options.graphic_file = optionValue(index, argc, argv, option);
            } else if (option == "-o" || option == "--output") {
                options.output_file = optionValue(index, argc, argv, option);
            } else if (option == "-e" || option == "--prefix") {
                options.snapshot_directory =
                    optionValue(index, argc, argv, option);
                if (options.snapshot_directory.empty()) {
                    throw std::runtime_error(
                        "snapshot directory must not be empty");
                }
            } else if (option == "-d" || option == "--device") {
                options.camera_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.camera_device < 0 || options.camera_device > 65535) {
                    throw std::runtime_error(
                        "camera device index must be between 0 and 65535");
                }
            } else if (option == "-c" || option == "--camera-res") {
                parseDimensions(optionValue(index, argc, argv, option),
                                options.camera_width, options.camera_height, option);
            } else if (option == "--use-yuv") {
                options.use_yuv = true;
            } else if (option == "--maximize-fps") {
                options.maximize_fps = true;
            } else if (option == "--use-source-fps") {
                options.use_source_fps = true;
            } else if (option == "--use-source-audio") {
                options.use_source_audio = true;
                options.enable_audio = true;
            } else if (option == "--edge") {
                options.edge_model = optionValue(index, argc, argv, option);
            } else if (option == "--human") {
                options.human_model = optionValue(index, argc, argv, option);
            } else if (option == "--onnx") {
                options.onnx_configuration =
                    optionValue(index, argc, argv, option);
            } else if (option == "--background") {
                options.human_background = true;
            } else if (option == "--black") {
                options.human_black_point =
                    parseNumber(optionValue(index, argc, argv, option), option);
                options.human_black_specified = true;
                if (options.human_black_point < 0.0 ||
                    options.human_black_point > 1.0) {
                    throw std::runtime_error(
                        "--black must be between 0.0 and 1.0");
                }
            } else if (option == "--white") {
                options.human_white_point =
                    parseNumber(optionValue(index, argc, argv, option), option);
                options.human_white_specified = true;
                if (options.human_white_point < 0.0 ||
                    options.human_white_point > 1.0) {
                    throw std::runtime_error(
                        "--white must be between 0.0 and 1.0");
                }
            } else if (option == "--check-dnn") {
                options.check_dnn = true;
            } else if (option == "-r" || option == "--resolution") {
                parseDimensions(optionValue(index, argc, argv, option), options.width,
                                options.height, option);
                options.resolution_specified = true;
            } else if (option == "-s" || option == "--shaders") {
                options.shader_directory = optionValue(index, argc, argv, option);
            } else if (option == "-f" || option == "--fragment") {
                options.fragment_shader = optionValue(index, argc, argv, option);
            } else if (option == "--compute") {
                options.compute_shader = optionValue(index, argc, argv, option);
            } else if (option == "--enable-3d") {
                options.enable_3d = true;
            } else if (option == "--model") {
                options.model_file = optionValue(index, argc, argv, option);
                options.enable_3d = true;
            } else if (option == "-H" || option == "--shader-index") {
                options.shader_index =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.shader_index < 0 ||
                    options.shader_index >=
                        static_cast<int>(input::MAX_SHADER_ENTRIES)) {
                    throw std::runtime_error("shader index is outside the supported range");
                }
            } else if (option == "--shader-file") {
                options.shader_file = optionValue(index, argc, argv, option);
            } else if (option == "--uniform") {
                options.custom_uniform_overrides.push_back(
                    optionValue(index, argc, argv, option));
                if (options.custom_uniform_overrides.size() >
                    mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS) {
                    throw std::runtime_error(
                        "too many --uniform overrides were supplied");
                }
            } else if (option == "--shader-pass") {
                const std::string values = optionValue(index, argc, argv, option);
                std::size_t start = 0;
                while (start <= values.size()) {
                    const std::size_t separator = values.find(',', start);
                    const std::string_view value(
                        values.data() + start,
                        (separator == std::string::npos ? values.size() : separator) - start);
                    if (value.empty()) {
                        throw std::runtime_error(
                            "shader pass list contains an empty entry");
                    }
                    options.shader_pass_indices.push_back(parseInteger(value, option));
                    if (options.shader_pass_indices.back() < 0 ||
                        options.shader_pass_indices.size() >
                            input::MAX_SHADER_ENTRIES) {
                        throw std::runtime_error(
                            "shader pass list is outside the supported range");
                    }
                    if (separator == std::string::npos) {
                        break;
                    }
                    start = separator + 1;
                }
            } else if (option == "--shader-pass-files") {
                const std::string payload = optionValue(index, argc, argv, option);
                std::size_t start = 0;
                while (start < payload.size()) {
                    const std::size_t separator = payload.find(':', start);
                    if (separator == std::string::npos) {
                        throw std::runtime_error("invalid --shader-pass-files payload");
                    }
                    const int length = parseInteger(
                        std::string_view(payload).substr(start, separator - start), option);
                    const std::size_t name_start = separator + 1;
                    if (length < 0 || static_cast<std::size_t>(length) >
                                          payload.size() - name_start) {
                        throw std::runtime_error("invalid --shader-pass-files payload");
                    }
                    options.shader_pass_files.push_back(
                        payload.substr(name_start, static_cast<std::size_t>(length)));
                    if (options.shader_pass_files.size() >
                        input::MAX_SHADER_ENTRIES) {
                        throw std::runtime_error(
                            "shader pass file list contains too many entries");
                    }
                    start = name_start + static_cast<std::size_t>(length);
                }
            } else if (option == "--playlist") {
                options.playlist_file = optionValue(index, argc, argv, option);
            } else if (option == "--enable-playlist") {
                options.enable_playlist = true;
            } else if (option == "--cross-fade") {
                options.cross_fade_duration =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.cross_fade_duration < 0.0 ||
                    options.cross_fade_duration > 60.0) {
                    throw std::runtime_error(
                        "crossfade duration must be between 0 and 60 seconds");
                }
            } else if (option == "--time-speed") {
                options.time_speed =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.time_speed < -1000.0 ||
                    options.time_speed > 1000.0) {
                    throw std::runtime_error(
                        "time speed must be between -1000 and 1000");
                }
            } else if (option == "--normalized") {
                options.normalized_time = true;
            } else if (option == "--autopilot-frames" ||
                       option == "--autopilot-timeout") {
                options.autopilot_frames = parseInteger(
                    optionValue(index, argc, argv, option), option);
                if (options.autopilot_frames < 4 ||
                    options.autopilot_frames > 1000000000) {
                    throw std::runtime_error(
                        "autopilot frame interval must be between 4 and 1000000000");
                }
            } else if (option == "--autopilot-random" ||
                       option == "--autiopilot-random") {
                options.autopilot_random_timeout = parseInteger(
                    optionValue(index, argc, argv, option), option);
                if (options.autopilot_random_timeout < 4 ||
                    options.autopilot_random_timeout > 1000000000) {
                    throw std::runtime_error(
                        "autopilot random interval must be between 4 and 1000000000");
                }
            } else if (option == "-u" || option == "--fps") {
                options.requested_fps =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.requested_fps <= 0.0 ||
                    options.requested_fps > 1000.0) {
                    throw std::runtime_error(
                        "FPS must be between 0 and 1000");
                }
            } else if (option == "-w" || option == "--enable-audio") {
                options.enable_audio = true;
            } else if (option == "-l" || option == "--channels") {
                options.audio_channels =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.audio_channels < 1 ||
                    options.audio_channels > 32) {
                    throw std::runtime_error(
                        "audio channels must be between 1 and 32");
                }
            } else if (option == "-q" || option == "--sense") {
                options.audio_sensitivity =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.audio_sensitivity < 0.1 ||
                    options.audio_sensitivity > 5.0) {
                    throw std::runtime_error(
                        "audio sensitivity must be between 0.1 and 5.0");
                }
            } else if (option == "--audio-warm-rate") {
                options.audio_warm_rate =
                    parseNumber(optionValue(index, argc, argv, option), option);
                options.audio_warm_rate_specified = true;
                if (options.audio_warm_rate < 0.0 ||
                    options.audio_warm_rate > 1000.0) {
                    throw std::runtime_error(
                        "audio warmup rate must be between 0 and 1000");
                }
            } else if (option == "--audio-input") {
                const std::string value = optionValue(index, argc, argv, option);
                options.audio_input_specified = true;
                options.audio_input_device =
                    value == "default" ? -1 : parseInteger(value, option);
                if (options.audio_input_device < -1 ||
                    options.audio_input_device > 65535) {
                    throw std::runtime_error(
                        "audio input must be default or a non-negative device index");
                }
            } else if (option == "--audio-file") {
                options.audio_file = optionValue(index, argc, argv, option);
                options.enable_audio = true;
            } else if (option == "-y" || option == "--pass-through") {
                options.audio_pass_through = true;
                options.enable_audio = true;
            } else if (option == "--audio-output") {
                const std::string value = optionValue(index, argc, argv, option);
                options.audio_output_specified = true;
                options.audio_output_device =
                    value == "default" ? -1 : parseInteger(value, option);
                if (options.audio_output_device < -1 ||
                    options.audio_output_device > 65535) {
                    throw std::runtime_error(
                        "audio output must be default or a non-negative device index");
                }
            } else if (option == "--pass-through-gain") {
                options.audio_pass_through_gain =
                    parseNumber(optionValue(index, argc, argv, option), option);
                options.audio_pass_through_gain_specified = true;
                if (options.audio_pass_through_gain < 0.0 ||
                    options.audio_pass_through_gain > 4.0) {
                    throw std::runtime_error(
                        "pass-through gain must be between 0.0 and 4.0");
                }
            } else if (option == "--record-gain") {
                options.audio_recording_gain =
                    parseNumber(optionValue(index, argc, argv, option), option);
                options.audio_recording_gain_specified = true;
                if (options.audio_recording_gain < 0.0 ||
                    options.audio_recording_gain > 2.0) {
                    throw std::runtime_error(
                        "recording gain must be between 0.0 and 2.0");
                }
            } else if (option == "--record-audio") {
                options.record_audio_file =
                    optionValue(index, argc, argv, option);
                options.enable_audio = true;
            } else if (option == "--audio-repeat") {
                options.audio_repeat = true;
            } else if (option == "--audio-trunc") {
                options.audio_trunc = true;
            } else if (option == "--enable-audio-buffers" ||
                       option == "--audio-buffers") {
                options.audio_buffers = parseInteger(
                    optionValue(index, argc, argv, option), option);
                if (options.audio_buffers < 0 || options.audio_buffers > 64) {
                    throw std::runtime_error(
                        "audio history buffers must be between 0 and 64");
                }
            } else if (option == "--list-devices") {
                options.list_audio_devices = true;
            } else if (option == "--check-audio") {
                options.check_audio = true;
            } else if (option == "--midi-device") {
                options.midi_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
                options.midi_device_specified = true;
                if (options.midi_device < 0 || options.midi_device > 65535) {
                    throw std::runtime_error(
                        "MIDI device index must be between 0 and 65535");
                }
            } else if (option == "--midi-monitor") {
                options.midi_monitor = true;
            } else if (option == "--midi-map") {
                options.midi_map_file = optionValue(index, argc, argv, option);
            } else if (option == "--midi-cc") {
                options.midi_cc_mappings.push_back(
                    optionValue(index, argc, argv, option));
                if (options.midi_cc_mappings.size() >
                    mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS) {
                    throw std::runtime_error(
                        "too many --midi-cc mappings were supplied");
                }
            } else if (option == "--list-midi") {
                options.list_midi_devices = true;
            } else if (option == "--check-midi") {
                options.check_midi = true;
            } else if (option == "--gpu-filter") {
                options.gpu_filter_indices = parseIntegerList(
                    optionValue(index, argc, argv, option), option);
                if (options.gpu_filter_indices.size() > 256U ||
                    std::any_of(options.gpu_filter_indices.begin(),
                                options.gpu_filter_indices.end(), [](int value) {
                                    return value < 0 || value > 65535;
                                })) {
                    throw std::runtime_error(
                        "GPU filter list is outside the supported range");
                }
            } else if (option == "--gpu-buffer") {
                options.gpu_frame_buffer_size =
                    parseInteger(optionValue(index, argc, argv, option), option);
                options.gpu_buffer_specified = true;
                if (options.gpu_frame_buffer_size < 4 ||
                    options.gpu_frame_buffer_size > 32) {
                    throw std::runtime_error(
                        "GPU frame buffer must be between 4 and 32");
                }
            } else if (option == "-m" || option == "--cuda-device") {
                options.cuda_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
                options.cuda_device_specified = true;
                if (options.cuda_device < 0 || options.cuda_device > 1024) {
                    throw std::runtime_error(
                        "CUDA device index must be between 0 and 1024");
                }
            } else if (option == "--list-filters") {
                options.list_gpu_filters = true;
            } else if (option == "--list-cuda-devices") {
                options.list_cuda_devices = true;
            } else if (option == "--check-cuda") {
                options.check_cuda = true;
            } else if (option == "--duration") {
                options.duration = parseNumber(optionValue(index, argc, argv, option), option);
                if (options.duration <= 0.0 || options.duration > 604800.0) {
                    throw std::runtime_error(
                        "duration must be between 0 and 604800 seconds");
                }
            } else if (option == "--max-size") {
                options.max_size_mb =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.max_size_mb <= 0.0 ||
                    options.max_size_mb > 1048576.0) {
                    throw std::runtime_error(
                        "maximum output size must be between 0 and 1048576 MB");
                }
            } else if (option == "--png") {
                options.png_output = true;
            } else if (option == "--generate") {
                options.generate_interval =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.generate_interval <= 0) {
                    throw std::runtime_error("--generate requires a positive frame interval");
                }
            } else if (option == "-b" || option == "--bitrate" ||
                       option == "--encode-crf") {
                options.encode_crf =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.encode_crf < 0 || options.encode_crf > 51) {
                    throw std::runtime_error("encoder CRF must be between 0 and 51");
                }
            } else if (option == "--encode-preset") {
                options.encode_preset = optionValue(index, argc, argv, option);
            } else if (option == "--encode-tune") {
                options.encode_tune = optionValue(index, argc, argv, option);
            } else if (option == "--encode-codec") {
                options.encode_codec = optionValue(index, argc, argv, option);
            } else if (option == "--encode-params") {
                options.encode_params = optionValue(index, argc, argv, option);
            } else if (option == "--list-encoders") {
                options.list_encoders = true;
            } else if (option == "--list-encoder-options") {
                options.list_encoder_options = optionValue(index, argc, argv, option);
            } else if (option == "--encode-realtime") {
                options.encode_realtime = true;
            } else if (option == "--no-drop") {
                options.no_drop = true;
            } else if (option == "--display-filter") {
                options.display_filter = true;
            } else if (option == "--disable-counter") {
                options.disable_counter = true;
            } else if (option == "--use-watermark") {
                options.watermark_text =
                    optionValue(index, argc, argv, option);
                if (options.watermark_text.empty()) {
                    throw std::runtime_error(
                        "--use-watermark requires non-empty text");
                }
            } else if (option == "--use-watermark-color") {
                options.watermark_color = parseColor(
                    optionValue(index, argc, argv, option), option);
            } else if (option == "--copy-audio") {
                options.copy_audio = true;
            } else if (option == "--mute-output") {
                options.mute_output = true;
            } else if (option == "-n" || option == "--fullscreen") {
                options.fullscreen = true;
            } else if (option == "-a" || option == "--repeat") {
                options.repeat = true;
            } else if (option == "--enable-vsync") {
                options.enable_vsync = true;
            } else if (option == "--enable-screenshot") {
                options.enable_screenshot = true;
            } else if (option == "--history-test") {
                options.history_test = true;
                options.enable_texture_cache = true;
            } else if (option == "--texture-cache" ||
                       option == "--texture-cache-array") {
                options.enable_texture_cache = true;
            } else if (option == "--cache-delay") {
                options.cache_delay =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.cache_delay < 0 || options.cache_delay > 1000000) {
                    throw std::runtime_error(
                        "--cache-delay must be between 0 and 1000000");
                }
            } else if (option == "--texture-cache-size") {
                options.texture_cache_size =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.texture_cache_size < 1 || options.texture_cache_size > 64) {
                    throw std::runtime_error(
                        "--texture-cache-size must be between 1 and 64");
                }
            } else if (option == "--flip") {
                options.flip_output = true;
            } else if (option == "--rotate") {
                options.frame_rotation =
                    parseFrameRotation(optionValue(index, argc, argv, option));
            } else {
                throw std::runtime_error("unknown option: " + std::string(option));
            }
        }

        if (options.use_source_audio) {
            if (options.input_file.empty()) {
                throw std::runtime_error(
                    "--use-source-audio requires --input <video>");
            }
            if (!options.use_source_fps) {
                throw std::runtime_error(
                    "--use-source-audio requires --use-source-fps");
            }
            if (!options.audio_file.empty()) {
                throw std::runtime_error(
                    "--use-source-audio cannot be combined with --audio-file");
            }
            options.audio_file = options.input_file;
            options.audio_repeat = options.audio_repeat || options.repeat;
        }

        validateOptionStrings(options);
        applyResourceDefaults(options);

        if (!options.input_file.empty() && !options.graphic_file.empty()) {
            throw std::runtime_error("--input and --graphic cannot be used together");
        }
        const int shader_source_count =
            static_cast<int>(!options.shader_directory.empty()) +
            static_cast<int>(!options.fragment_shader.empty()) +
            static_cast<int>(!options.compute_shader.empty());
        if (shader_source_count > 1) {
            throw std::runtime_error(
                "--shaders, --fragment, and --compute are mutually exclusive");
        }
        if (!options.custom_uniform_overrides.empty() &&
            options.shader_directory.empty()) {
            throw std::runtime_error("--uniform requires --shaders <directory>");
        }
        if (!options.midi_cc_mappings.empty() &&
            options.shader_directory.empty()) {
            throw std::runtime_error("--midi-cc requires --shaders <directory>");
        }
        if (options.gpu_buffer_specified && options.gpu_filter_indices.empty()) {
            throw std::runtime_error("--gpu-buffer requires --gpu-filter <list>");
        }
        if ((!options.shader_pass_indices.empty() || !options.shader_pass_files.empty() ||
             !options.playlist_file.empty() || options.enable_playlist) &&
            options.shader_directory.empty()) {
            throw std::runtime_error(
                "shader passes and playlists require --shaders <directory>");
        }
        if (options.enable_playlist && options.playlist_file.empty()) {
            throw std::runtime_error("--enable-playlist requires --playlist <file>");
        }
        if ((options.autopilot_frames > 0 || options.autopilot_random_timeout > 0) &&
            options.playlist_file.empty()) {
            throw std::runtime_error("autopilot requires --playlist <file>");
        }
        if (!options.output_file.empty() && !options.input_file.empty() &&
            fs::absolute(options.output_file).lexically_normal() ==
                fs::absolute(options.input_file).lexically_normal()) {
            throw std::runtime_error("output file must differ from the input file");
        }
        if (options.duration > 0.0 && options.output_file.empty()) {
            throw std::runtime_error("--duration requires --output <file>");
        }
        if (options.png_output &&
            (options.input_file.empty() || options.output_file.empty())) {
            throw std::runtime_error("--png requires video --input and --output");
        }
        if (options.max_size_mb > 0.0 &&
            (options.output_file.empty() || options.png_output)) {
            throw std::runtime_error("--max-size requires encoded video output");
        }
        if (options.copy_audio &&
            (options.input_file.empty() || options.output_file.empty() ||
             options.png_output || options.repeat)) {
            throw std::runtime_error(
                "--copy-audio requires non-repeating video input and encoded output");
        }
        if (options.copy_audio && !options.audio_file.empty()) {
            throw std::runtime_error(
                "--copy-audio and --audio-file select different audio sources");
        }
        if (!options.record_audio_file.empty() && !options.audio_file.empty()) {
            throw std::runtime_error(
                "--record-audio records live input and cannot be used with --audio-file");
        }
        if (!options.record_audio_file.empty()) {
            const fs::path recording_path =
                fs::absolute(options.record_audio_file).lexically_normal();
            const auto conflicts_with = [&](const std::string &filename) {
                return !filename.empty() &&
                       recording_path == fs::absolute(filename).lexically_normal();
            };
            if (conflicts_with(options.input_file) ||
                conflicts_with(options.graphic_file) ||
                conflicts_with(options.output_file)) {
                throw std::runtime_error(
                    "--record-audio output must differ from media input and video output");
            }
        }
        if (!options.graphic_file.empty() && !options.output_file.empty() &&
            options.duration <= 0.0 &&
            !(options.audio_trunc && !options.audio_file.empty())) {
            throw std::runtime_error(
                "graphic recording requires --duration <seconds> or "
                "--audio-file <media> --audio-trunc");
        }
        if (options.audio_buffers > 0 && !options.enable_audio) {
            throw std::runtime_error(
                "--enable-audio-buffers requires --enable-audio");
        }
        if (options.audio_warm_rate_specified && !options.enable_audio) {
            throw std::runtime_error(
                "--audio-warm-rate requires an enabled audio source");
        }
        if (!options.audio_file.empty() && options.audio_input_specified) {
            throw std::runtime_error(
                "--audio-file and --audio-input cannot be used together");
        }
        if (options.audio_repeat && options.audio_file.empty()) {
            throw std::runtime_error(
                "--audio-repeat requires --audio-file <media>");
        }
        if (options.audio_trunc && options.audio_file.empty()) {
            throw std::runtime_error(
                "--audio-trunc requires --audio-file <media>");
        }
        if (options.audio_output_specified && !options.audio_pass_through) {
            throw std::runtime_error(
                "--audio-output requires --pass-through");
        }
        if (options.audio_pass_through_gain_specified &&
            !options.audio_pass_through) {
            throw std::runtime_error(
                "--pass-through-gain requires --pass-through");
        }
        if (options.audio_recording_gain_specified &&
            (!options.enable_audio || !options.audio_file.empty() ||
             (options.record_audio_file.empty() &&
              (options.output_file.empty() || options.png_output ||
               options.copy_audio || options.mute_output)))) {
            throw std::runtime_error(
                "--record-gain requires live audio recording or encoded output");
        }
        if (options.maximize_fps) {
            if (!options.input_file.empty() || !options.graphic_file.empty()) {
                throw std::runtime_error(
                    "--maximize-fps is available only for camera input");
            }
            if (options.requested_fps <= 0.0) {
                throw std::runtime_error(
                    "--maximize-fps requires a target set with --fps");
            }
        }
        if (options.use_source_fps) {
            if (options.input_file.empty()) {
                throw std::runtime_error(
                    "--use-source-fps requires --input <video>");
            }
            if (options.requested_fps > 0.0) {
                throw std::runtime_error(
                    "--use-source-fps cannot be combined with --fps");
            }
        }
        if ((options.human_background || options.human_black_specified ||
             options.human_white_specified) &&
            options.human_model.empty()) {
            throw std::runtime_error(
                "--background, --black, and --white require --human <model.onnx>");
        }
        if (!options.human_model.empty() &&
            options.human_black_point >= options.human_white_point) {
            throw std::runtime_error(
                "--black must be less than --white for human segmentation");
        }
        if (options.human_background && options.enable_3d) {
            throw std::runtime_error(
                "--background is currently available only in 2D mode");
        }
        return options;
    }

    void printHelp(std::ostream &output) {
        output << "ACMXVK - Vulkan video shader engine (Increment 9Q)\n\n"
               << "Usage:\n"
               << "  acmxvk -i video.mp4 -s shader-directory [options]\n"
               << "  acmxvk -g image.png -f shader.spv [options]\n"
               << "  acmxvk -d 0 -s shader-directory [options]\n\n"
               << "Resources:\n"
               << "  -p, --path <directory>      Assets root containing data/, shaders/,\n"
               << "                              playlists/, and midi-examples/\n"
               << "      ACMXVK_PATH             Default resource root when --path is absent\n"
               << "      ACMXVK_SHADER_PATH      Default SPIR-V library when shader input is absent\n"
               << "                              ACMX2_PATH is accepted as a data fallback\n\n"
               << "Input:\n"
               << "  -i, --input <file>          Read a video file\n"
               << "  -g, --graphic <file>        Read a still image\n"
               << "  -d, --device <index>        Camera device (default 0)\n"
               << "  -c, --camera-res <WxH>      Requested camera dimensions\n"
               << "      --use-yuv               Prefer YUYV camera capture over MJPG\n"
               << "      --maximize-fps          Render at --fps using the latest camera frame\n"
               << "      --use-source-fps        Play video on its reported source clock\n"
               << "      --use-source-audio      Use the video's audio for shader reactivity\n"
               << "  -u, --fps <rate>            Camera/output FPS\n"
               << "                              Video files prefer FFmpeg/NVDEC capture\n\n"
               << "DNN effects (requires WITH_OPENCV_DNN=ON build):\n"
               << "      --edge <model.onnx>     Replace input with a DexiNed edge map\n"
               << "      --human <model.onnx>    Isolate a person with PP-HumanSeg\n"
               << "      --onnx <config.yaml>    Run a YAML-configured image ONNX model\n"
               << "      --background           Apply shaders only behind the person (2D)\n"
               << "      --black <0.0-1.0>      Alpha black point (default 0.35)\n"
               << "      --white <0.0-1.0>      Alpha white point (default 0.75)\n"
               << "      --check-dnn             Report compiled OpenCV DNN support\n"
               << "                              Backend is benchmarked on the first frame\n\n"
               << "Shaders:\n"
               << "      --build <library.json> Compile a source shader library and exit\n"
               << "      --builddir <directory> Output directory required by --build\n"
               << "      --fix <directory>      Continue and omit/remove failed shaders\n"
               << "      --prune                Delete GLSL sources that fail compilation\n"
               << "      --force                Confirm permanent deletion by --prune\n"
               << "      --glslc <executable>   GLSL compiler for --build (default: glslc)\n"
               << "  -s, --shaders <directory>   SPIR-V library with library.json or index.txt\n"
               << "  -f, --fragment <file.spv>   Use one SPIR-V fragment shader\n"
               << "      --compute <file.spv>    Use one SPIR-V compute shader\n"
               << "  -H, --shader-index <index>  Initial library shader index\n"
               << "      --shader-file <name>    Initial library shader filename\n"
               << "      --uniform <name=value>  Override a library.json custom float\n\n"
               << "3D model:\n"
               << "      --enable-3d             Map input frames onto a 3D model\n"
               << "      --model <file>          OBJ, MXMOD, or compressed MXMOD model\n"
               << "                              Defaults to the bundled cube.obj\n\n"
               << "  --shader-pass <indices>     Mixed fragment/compute pre-pass chain\n"
               << "  --shader-pass-files <data>  ACMX2 length-prefixed shader filenames\n"
               << "  --playlist <file>           Shader or named multipass playlist\n\n"
               << "      --cross-fade <seconds>  Shader transition duration (default 0.5)\n\n"
               << "  --enable-playlist           Enable the playlist immediately\n"
               << "  --time-speed <mult>         Scale shader time (default 1.0)\n"
               << "  --normalized                Use fixed frame time outside video mode\n"
               << "  --autopilot-frames <N>      Playlist switch interval (minimum 4)\n"
               << "                              Uses decoded frames for video input\n"
               << "  --autopilot-timeout <N>     Alias for --autopilot-frames\n"
               << "  --autopilot-random <N>      Random playlist interval from 4..N\n\n"
               << "History cache:\n"
               << "      --texture-cache         Enable Vulkan texture history\n"
               << "      --texture-cache-array   Alias using sampler2DArray history\n"
               << "      --texture-cache-size N  History layers, 1-64 (default 8)\n"
               << "      --cache-delay N         Skip N frames between cache writes\n"
               << "      --history-test          Enable history and the built-in echo demo\n\n"
               << "Recording:\n"
               << "  -o, --output <file>         Encode processed output with MXWrite\n"
               << "      --duration <seconds>    Stop after this much output video\n"
               << "      --max-size <MB>         Stop when encoded output exceeds this size\n"
               << "      --png                   Write video output as a PNG sequence\n"
               << "      --generate <N>          Save a PNG every N processed frames\n"
               << "  -e, --prefix <directory>   Directory for Z snapshots (default .)\n"
               << "  -b, --encode-crf <0-51>     Encoder quality (default 18)\n"
               << "      --encode-preset <name>  Encoder speed/quality preset\n"
               << "      --encode-tune <name>    Encoder content/latency tuning\n"
               << "      --encode-codec <name>   auto, software, nvenc, or exact encoder\n"
               << "      --encode-params <text>  Additional FFmpeg encoder options\n"
               << "      --list-encoders         List available video encoders and exit\n"
               << "      --list-encoder-options <name>\n"
               << "                              List one encoder's options and exit\n"
               << "      --encode-realtime       Enable low-latency encoder settings\n"
               << "      --no-drop               Block when the encoder queue is full\n"
               << "      --display-filter        Show active shader/filter details\n"
               << "      --disable-counter       Hide the shader/timer/FPS HUD at startup\n"
               << "      --use-watermark <text>  Show a text watermark in the upper-left\n"
               << "                              and hide the preview HUD by default\n"
               << "      --use-watermark-color <r,g,b>\n"
               << "                              Watermark RGB color (default 255,0,150)\n"
               << "      --copy-audio            Copy input audio into encoded output\n"
               << "      --mute-output           Keep recorded video audio-free\n\n"
               << "Audio (requires AUDIO=ON build):\n"
               << "  -w, --enable-audio          Enable live audio-reactive metrics\n"
               << "  -l, --channels <N>          Capture channels (default 2)\n"
               << "  -q, --sense <0.1-5.0>       Audio sensitivity (default 1.0)\n"
               << "      --audio-warm-rate N     Shader ramp per second (default 0.5; 0 off)\n"
               << "      --audio-input <device>  Input index or default\n"
               << "      --audio-file <media>    Media file or M3U/M3U8 reactivity source\n"
               << "  -y, --pass-through          Play live/file audio through an output device\n"
               << "      --audio-output <device> Output index or default\n"
               << "      --pass-through-gain N   Monitor gain, 0.0-4.0 (default 1.0)\n"
               << "      --record-gain N         Saved/muxed mic gain, 0.0-2.0 (default 1.0)\n"
               << "      --record-audio <wav>    Record live microphone input as PCM16 WAV\n"
               << "      --audio-repeat          Restart file audio at end-of-stream\n"
               << "      --audio-trunc           Stop ACMXVK when file audio finishes\n"
               << "                              Live/file audio is muxed unless --mute-output\n"
               << "      --enable-audio-buffers N\n"
               << "                              FFT history layers at binding 4\n"
               << "      --list-devices          List RtAudio devices and exit\n"
               << "      --check-audio           Report compiled audio support\n"
               << "                              Provides a 256-bin FFT at binding 3\n\n"
               << "MIDI (requires MIDI=ON build):\n"
               << "      --midi-device <index>   Open a MIDI input port (default 0)\n"
               << "      --midi-monitor          Print received MIDI messages\n"
               << "      --midi-map <file>       Load an ACMX2 .midi_cfg mapping\n"
               << "      --midi-cc <map>         Map [channel:]CC to a custom uniform\n"
               << "      --list-midi             List MIDI input ports and exit\n"
               << "      --check-midi            Report compiled MIDI support\n"
               << "                              Paired knobs repeat by distance from 64\n\n"
               << "CUDA and filters:\n"
               << "      --gpu-filter <list>     Comma-separated acidcam-gpu indices\n"
               << "      --gpu-buffer <4-32>     Temporal frame count (default 10)\n"
               << "  -m, --cuda-device <index>   Select NVDEC/filter device (default 0)\n"
               << "      --list-filters          List acidcam-gpu filters and exit\n"
               << "      --list-cuda-devices     List MXVK CUDA devices and exit\n"
               << "      --check-cuda            Report interop and filter support\n"
               << "                              Left/Right selects the active filter\n"
               << "                              NVDEC interop follows the MXVK build\n"
               << "                              Filters require WITH_CUDA=ON\n"
               << "                              Video/camera RGBA and rotation stay on GPU\n\n"
               << "Window:\n"
               << "  -r, --resolution <WxH>      Render/output resolution override\n"
               << "                              Preview fits display without changing output size\n"
               << "  -n, --fullscreen            Start fullscreen\n"
               << "  -a, --repeat                Repeat video input\n"
               << "      --rotate <mode>         clockwise, 180, or counterclockwise\n"
               << "      --flip                  Flip final display/encoded output vertically\n"
               << "      --enable-vsync          Use FIFO presentation\n"
               << "      --enable-screenshot     Enable MXVK F10 screenshots\n\n"
               << "Output:\n"
               << "      --unbuffered           Flush stdout/stderr after each write for GUI capture\n"
               << "      --interface-shm        Accept live shader selection from the ACMX interface\n\n"
               << "Keys: Up/Down shader or playlist node, Shift+Up/Down post-shader,\n"
               << "      P playlist/pause, L freeze, T time, U/I step time,\n"
               << "      Page Up/Down time speed, Q audio time, Home audio delta,\n"
               << "      Insert/Delete audio sensitivity, End FFT sensitivity,\n"
               << "      3 toggle 2D/3D, V 3D view rotation, O 3D oscillation,\n"
               << "      C 3D wave,\n"
               << "      X reset skybox view,\n"
               << "      E watermark, F fullscreen, F9 runtime HUD, K shader lock,\n"
               << "      M multipass,\n"
               << "      J random autopilot, Y sequential autopilot, N random crossfade,\n"
               << "      [/] crossfade effect, Space bypass,\n"
               << "      W/A/S/D 3D look, +/- 3D zoom, Shift+/- 3D scale,\n"
               << "      1/2 zoom sensitivity,\n"
               << "      Z PNG, 4 TIFF (TIFF=ON), 5 WebP (WEBP=ON),\n"
               << "      6 raw RGBA snapshot,\n"
               << "      mouse drag/wheel 3D look/move,\n"
               << "      Escape quit\n";
    }

    [[nodiscard]] std::string cleanEncoderField(std::string value) {
        std::replace_if(value.begin(), value.end(), [](char character) { return character == '\t' || character == '\r' || character == '\n'; }, ' ');
        return value;
    }

    void printEncoders(std::ostream &output) {
        output << "MXWRITE_ENCODERS\t1\n";
        for (const EncoderInfo &encoder : available_video_encoders()) {
            output << "ENCODER\t" << cleanEncoderField(encoder.name) << '\t'
                   << cleanEncoderField(encoder.long_name) << '\t'
                   << cleanEncoderField(encoder.codec_name) << '\t'
                   << (encoder.hardware ? "hardware" : "software") << '\t'
                   << (encoder.experimental ? "experimental" : "stable") << '\t'
                   << cleanEncoderField(encoder.pixel_formats) << '\n';
        }
    }

    [[nodiscard]] bool printEncoderOptions(std::string_view encoder_name,
                                           std::ostream &output,
                                           std::ostream &error_output) {
        const std::string name(encoder_name);
        const std::vector<EncoderOptionInfo> options = video_encoder_options(name);
        if (options.empty() && !avcodec_find_encoder_by_name(name.c_str())) {
            error_output << "acmxvk: encoder not found: " << name << '\n';
            return false;
        }

        output << "MXWRITE_ENCODER_OPTIONS\t1\t" << cleanEncoderField(name) << '\n';
        for (const EncoderOptionInfo &option : options) {
            output << "OPTION\t" << cleanEncoderField(option.name) << '\t'
                   << cleanEncoderField(option.type) << '\t'
                   << cleanEncoderField(option.default_value) << '\t'
                   << cleanEncoderField(option.minimum) << '\t'
                   << cleanEncoderField(option.maximum) << '\t'
                   << cleanEncoderField(option.choices) << '\t'
                   << cleanEncoderField(option.help) << '\n';
        }
        return true;
    }

    [[nodiscard]] std::string trim(std::string text) {
        const auto first = std::find_if_not(text.begin(), text.end(), [](unsigned char value) {
            return std::isspace(value) != 0;
        });
        const auto last = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char value) {
                              return std::isspace(value) != 0;
                          }).base();
        if (first >= last) {
            return {};
        }
        return std::string(first, last);
    }

    struct ShaderManifest {
        struct CustomUniform {
            std::string name;
            std::size_t slot = 0;
            double minimum = 0.0;
            double maximum = 1.0;
            double step = 0.01;
            double value = 0.0;
        };

        fs::path path;
        std::vector<std::string> entries;
        std::vector<CustomUniform> custom_uniforms;
    };

    [[nodiscard]] bool isValidCustomUniformName(const std::string &name) {
        if (name.starts_with("gl_")) {
            return false;
        }
        try {
            input::validate_string(name, input::StringKind::Identifier,
                                   "custom uniform name");
            return true;
        } catch (const std::runtime_error &) {
            return false;
        }
    }

    [[nodiscard]] ShaderManifest loadShaderManifest(const fs::path &directory) {
        ShaderManifest manifest;
        const fs::path json_path = directory / "library.json";
        const fs::path text_path = directory / "index.txt";
        if (fs::is_regular_file(json_path)) {
            manifest.path = json_path;
            input::validate_text_file(json_path, "shader library.json");
            try {
                cv::FileStorage storage(json_path.string(),
                                        cv::FileStorage::READ |
                                            cv::FileStorage::FORMAT_JSON);
                if (!storage.isOpened()) {
                    throw std::runtime_error("unable to open shader manifest: " +
                                             json_path.string());
                }
                const cv::FileNode shader_entries = storage["shaders"];
                if (shader_entries.type() == cv::FileNode::NONE ||
                    !shader_entries.isSeq()) {
                    throw std::runtime_error(json_path.string() +
                                             " must contain a 'shaders' array");
                }
                for (const cv::FileNode &entry : shader_entries) {
                    if (manifest.entries.size() >=
                        input::MAX_SHADER_ENTRIES) {
                        throw std::runtime_error(
                            json_path.string() +
                            " contains too many shader entries");
                    }
                    std::string filename;
                    if (entry.isString()) {
                        entry >> filename;
                    } else if (entry.isMap() && !entry["file"].empty()) {
                        entry["file"] >> filename;
                    } else {
                        throw std::runtime_error(
                            json_path.string() +
                            " contains a shader entry without a file name");
                    }
                    filename = trim(std::move(filename));
                    if (filename.empty()) {
                        throw std::runtime_error(
                            json_path.string() +
                            " contains a shader entry without a file name");
                    }
                    input::validate_string(
                        filename, input::StringKind::Path,
                        json_path.string() + " shader file");
                    manifest.entries.push_back(std::move(filename));
                }

                const cv::FileNode custom_uniforms = storage["custom_uniforms"];
                if (!custom_uniforms.empty()) {
                    if (!custom_uniforms.isMap()) {
                        throw std::runtime_error(
                            json_path.string() +
                            " field 'custom_uniforms' must be an object");
                    }
                    bool has_explicit_slots = false;
                    bool has_implicit_slots = false;
                    std::unordered_set<std::size_t> occupied_slots;
                    for (auto iterator = custom_uniforms.begin();
                         iterator != custom_uniforms.end(); ++iterator) {
                        if (manifest.custom_uniforms.size() >=
                            mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains more than " +
                                std::to_string(mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS) +
                                " custom uniforms");
                        }

                        const cv::FileNode entry = *iterator;
                        ShaderManifest::CustomUniform uniform;
                        uniform.name = entry.name();
                        if (!entry.isMap() ||
                            !isValidCustomUniformName(uniform.name)) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains an invalid custom uniform: " +
                                uniform.name);
                        }
                        uniform.slot = manifest.custom_uniforms.size();
                        if (!entry["slot"].empty()) {
                            int slot = -1;
                            entry["slot"] >> slot;
                            if (slot < 0 ||
                                slot >= static_cast<int>(
                                            mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS)) {
                                throw std::runtime_error(
                                    json_path.string() +
                                    " contains an invalid slot for custom uniform: " +
                                    uniform.name);
                            }
                            uniform.slot = static_cast<std::size_t>(slot);
                            if (!occupied_slots.insert(uniform.slot).second) {
                                throw std::runtime_error(
                                    json_path.string() +
                                    " assigns more than one custom uniform to slot " +
                                    std::to_string(slot));
                            }
                            has_explicit_slots = true;
                        } else {
                            has_implicit_slots = true;
                        }
                        if (!entry["minimum"].empty()) {
                            entry["minimum"] >> uniform.minimum;
                        }
                        if (!entry["maximum"].empty()) {
                            entry["maximum"] >> uniform.maximum;
                        }
                        if (!entry["step"].empty()) {
                            entry["step"] >> uniform.step;
                        }
                        uniform.value = uniform.minimum;
                        if (!entry["value"].empty()) {
                            entry["value"] >> uniform.value;
                        }
                        if (!std::isfinite(uniform.minimum) ||
                            !std::isfinite(uniform.maximum) ||
                            !std::isfinite(uniform.step) ||
                            !std::isfinite(uniform.value) ||
                            uniform.maximum <= uniform.minimum ||
                            uniform.step <= 0.0 ||
                            std::abs(uniform.minimum) >
                                std::numeric_limits<float>::max() ||
                            std::abs(uniform.maximum) >
                                std::numeric_limits<float>::max() ||
                            std::abs(uniform.step) >
                                std::numeric_limits<float>::max() ||
                            std::abs(uniform.value) >
                                std::numeric_limits<float>::max()) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains an invalid range for custom uniform: " +
                                uniform.name);
                        }
                        uniform.value = std::clamp(
                            uniform.value, uniform.minimum, uniform.maximum);
                        manifest.custom_uniforms.push_back(std::move(uniform));
                    }
                    if (has_explicit_slots && has_implicit_slots) {
                        throw std::runtime_error(
                            json_path.string() +
                            " must specify a slot for every custom uniform or none");
                    }
                    if (has_explicit_slots) {
                        std::sort(manifest.custom_uniforms.begin(),
                                  manifest.custom_uniforms.end(),
                                  [](const ShaderManifest::CustomUniform &left,
                                     const ShaderManifest::CustomUniform &right) {
                                      return left.slot < right.slot;
                                  });
                        for (std::size_t slot = 0;
                             slot < manifest.custom_uniforms.size(); ++slot) {
                            if (manifest.custom_uniforms[slot].slot != slot) {
                                throw std::runtime_error(
                                    json_path.string() +
                                    " custom uniform slots must be contiguous from zero");
                            }
                        }
                    }
                }
            } catch (const cv::Exception &error) {
                throw std::runtime_error("unable to parse shader manifest " +
                                         json_path.string() + ": " + error.what());
            }
            return manifest;
        }

        if (!fs::is_regular_file(text_path)) {
            throw std::runtime_error("no library.json or index.txt found in shader library: " +
                                     directory.string());
        }
        manifest.path = text_path;
        input::validate_file_size(text_path, "shader index.txt");
        std::ifstream manifest_input(text_path);
        if (!manifest_input) {
            throw std::runtime_error("unable to open shader manifest: " +
                                     text_path.string());
        }
        std::string line;
        std::size_t line_number = 1;
        while (input::read_bounded_line(manifest_input, line,
                                        "shader index.txt", line_number++)) {
            line = trim(std::move(line));
            if (!line.empty() && line.front() != '#') {
                if (manifest.entries.size() >=
                    input::MAX_SHADER_ENTRIES) {
                    throw std::runtime_error(
                        text_path.string() +
                        " contains too many shader entries");
                }
                input::validate_string(
                    line, input::StringKind::Path,
                    text_path.string() + " shader file");
                manifest.entries.push_back(std::move(line));
            }
        }
        return manifest;
    }

    [[nodiscard]] fs::path resolveShaderManifestEntry(const fs::path &directory,
                                                      std::string entry) {
        std::replace(entry.begin(), entry.end(), '\\', '/');
        const fs::path relative_path(entry);
        if (relative_path.is_absolute()) {
            return {};
        }

        const fs::path normalized = relative_path.lexically_normal();
        const std::string normalized_text = normalized.generic_string();
        if (normalized_text.empty() || normalized_text == "." ||
            normalized_text == ".." || normalized_text.starts_with("../") ||
            normalized_text.find("/../") != std::string::npos ||
            normalized.extension() != ".spv") {
            return {};
        }

        std::error_code error;
        const fs::path root = fs::weakly_canonical(directory, error);
        if (error) {
            return {};
        }
        const fs::path shader = fs::weakly_canonical(root / normalized, error);
        if (error || !fs::is_regular_file(shader)) {
            return {};
        }
        const std::string resolved_relative = shader.lexically_relative(root).generic_string();
        if (resolved_relative.empty() || resolved_relative == ".." ||
            resolved_relative.starts_with("../")) {
            return {};
        }
        return shader;
    }

    [[nodiscard]] fs::path resolveShaderBuildEntry(const fs::path &directory,
                                                   std::string entry) {
        std::replace(entry.begin(), entry.end(), '\\', '/');
        const fs::path relative_path(entry);
        if (relative_path.is_absolute()) {
            return {};
        }

        const fs::path normalized = relative_path.lexically_normal();
        const std::string normalized_text = normalized.generic_string();
        const std::string extension = normalized.extension().string();
        if (normalized_text.empty() || normalized_text == "." ||
            normalized_text == ".." || normalized_text.starts_with("../") ||
            normalized_text.find("/../") != std::string::npos ||
            (extension != ".frag" && extension != ".comp" &&
             extension != ".spv")) {
            return {};
        }

        std::error_code error;
        const fs::path root = fs::weakly_canonical(directory, error);
        if (error) {
            return {};
        }
        const fs::path source = fs::weakly_canonical(root / normalized, error);
        if (error || !fs::is_regular_file(source)) {
            return {};
        }
        const std::string resolved_relative =
            source.lexically_relative(root).generic_string();
        if (resolved_relative.empty() || resolved_relative == ".." ||
            resolved_relative.starts_with("../")) {
            return {};
        }
        return source;
    }

    [[nodiscard]] std::string escapeJson(std::string_view value) {
        std::ostringstream escaped;
        for (const unsigned char character : value) {
            switch (character) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\b':
                escaped << "\\b";
                break;
            case '\f':
                escaped << "\\f";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    escaped << "\\u" << std::hex << std::uppercase
                            << std::setw(4) << std::setfill('0')
                            << static_cast<unsigned int>(character)
                            << std::dec << std::nouppercase;
                } else {
                    escaped << static_cast<char>(character);
                }
                break;
            }
        }
        return escaped.str();
    }

    [[nodiscard]] fs::path temporaryBuildPath(const fs::path &destination) {
        static std::uint64_t sequence = 0;
        for (int attempt = 0; attempt < 100; ++attempt) {
            fs::path temporary = destination;
            temporary += ".acmxvk-tmp-" + std::to_string(::getpid()) + "-" +
                         std::to_string(++sequence);
            if (!fs::exists(temporary)) {
                return temporary;
            }
        }
        throw std::runtime_error(
            "unable to allocate a temporary shader build path for: " +
            destination.string());
    }

    void replaceBuiltFile(const fs::path &temporary,
                          const fs::path &destination) {
        std::error_code error;
        fs::rename(temporary, destination, error);
        if (error) {
            fs::remove(temporary);
            throw std::runtime_error("unable to install built file " +
                                     destination.string() + ": " +
                                     error.message());
        }
    }

    class ShaderCompilationError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    void runGlslc(const std::string &executable, const fs::path &source_root,
                  const fs::path &source, const fs::path &output) {
        std::vector<std::string> arguments{
            executable, "-I", source_root.string(), source.string(), "-o",
            output.string()};
        std::vector<char *> argument_pointers;
        argument_pointers.reserve(arguments.size() + 1U);
        for (std::string &argument : arguments) {
            argument_pointers.push_back(argument.data());
        }
        argument_pointers.push_back(nullptr);

        pid_t process = 0;
        const int spawn_result =
            posix_spawnp(&process, executable.c_str(), nullptr, nullptr,
                         argument_pointers.data(), environ);
        if (spawn_result != 0) {
            throw std::runtime_error("unable to execute glslc '" + executable +
                                     "': " + std::strerror(spawn_result));
        }

        int status = 0;
        while (::waitpid(process, &status, 0) < 0) {
            if (errno != EINTR) {
                throw std::runtime_error("unable to wait for glslc: " +
                                         std::string(std::strerror(errno)));
            }
        }
        if (!WIFEXITED(status)) {
            throw std::runtime_error("glslc terminated by a signal for " +
                                     source.string());
        }
        if (WEXITSTATUS(status) != 0) {
            throw ShaderCompilationError(
                "glslc failed for " + source.string() + " (exit status " +
                std::to_string(WEXITSTATUS(status)) + ")");
        }
    }

    [[nodiscard]] int buildShaderLibrary(const Options &options) {
        const fs::path requested_manifest =
            fs::absolute(options.build_manifest).lexically_normal();
        if (requested_manifest.filename() != "library.json") {
            throw std::runtime_error(
                "--build must name a file called library.json");
        }
        input::validate_text_file(requested_manifest,
                                  "source shader library.json");

        std::error_code error;
        const fs::path source_root =
            fs::weakly_canonical(requested_manifest.parent_path(), error);
        if (error || source_root.empty()) {
            throw std::runtime_error("unable to resolve source shader library: " +
                                     requested_manifest.string());
        }
        fs::create_directories(options.build_directory, error);
        if (error) {
            throw std::runtime_error("unable to create shader build directory: " +
                                     error.message());
        }
        const fs::path output_root =
            fs::weakly_canonical(options.build_directory, error);
        if (error || output_root.empty()) {
            throw std::runtime_error("unable to resolve shader build directory: " +
                                     options.build_directory);
        }
        if (source_root == output_root) {
            throw std::runtime_error(
                "the shader output directory must differ from the source "
                "library directory");
        }

        const ShaderManifest manifest = loadShaderManifest(source_root);
        if (manifest.entries.empty()) {
            throw std::runtime_error(
                "source library.json contains no shader entries");
        }

        std::vector<std::string> output_entries;
        output_entries.reserve(manifest.entries.size());
        std::unordered_set<std::string> unique_outputs;
        std::size_t compiled = 0;
        std::size_t copied = 0;
        std::size_t current = 0;
        std::size_t failed = 0;
        std::size_t pruned = 0;
        std::size_t processed = 0;
        int next_progress = 5;

        const auto report_progress = [&] {
            ++processed;
            const int percentage = static_cast<int>(
                processed * 100U / manifest.entries.size());
            while (next_progress <= 100 && percentage >= next_progress) {
                std::cout << "acmxvk: build progress: " << next_progress
                          << "% (" << processed << '/'
                          << manifest.entries.size() << ")\n"
                          << std::flush;
                next_progress += 5;
            }
        };

        for (const std::string &entry : manifest.entries) {
            fs::path source;
            fs::path destination;
            try {
                source = resolveShaderBuildEntry(source_root, entry);
                if (source.empty()) {
                    throw std::runtime_error(
                        "source library contains an unavailable or unsafe shader: " +
                        entry);
                }

                std::string normalized_entry = entry;
                std::replace(normalized_entry.begin(), normalized_entry.end(),
                             '\\', '/');
                fs::path relative(normalized_entry);
                relative = relative.lexically_normal();
                if (relative.extension() != ".spv") {
                    relative += ".spv";
                }
                const std::string output_entry = relative.generic_string();
                std::string output_key = output_entry;
                std::transform(
                    output_key.begin(), output_key.end(), output_key.begin(),
                    [](unsigned char character) {
                        return static_cast<char>(std::tolower(character));
                    });
                if (!unique_outputs.insert(output_key).second) {
                    throw std::runtime_error(
                        "source library produces a duplicate output path: " +
                        output_entry);
                }

                destination = output_root / relative;
                error.clear();
                fs::create_directories(destination.parent_path(), error);
                if (error) {
                    throw std::runtime_error(
                        "unable to create shader output directory: " +
                        error.message());
                }
                const fs::path destination_parent =
                    fs::weakly_canonical(destination.parent_path(), error);
                const std::string parent_relative =
                    error ? std::string{}
                          : destination_parent.lexically_relative(output_root)
                                .generic_string();
                if (error || parent_relative == ".." ||
                    parent_relative.starts_with("../") ||
                    fs::is_symlink(destination)) {
                    throw std::runtime_error(
                        "shader output resolves outside the output directory: " +
                        output_entry);
                }

                bool needs_build = !fs::is_regular_file(destination);
                if (!needs_build) {
                    needs_build = fs::last_write_time(destination, error) <
                                  fs::last_write_time(source);
                    if (error) {
                        needs_build = true;
                        error.clear();
                    }
                }
                if (!needs_build) {
                    try {
                        input::validate_spirv_file(
                            destination, "built shader module");
                    } catch (const std::runtime_error &) {
                        needs_build = true;
                    }
                }

                if (needs_build) {
                    const fs::path temporary = temporaryBuildPath(destination);
                    const bool copy_source = source.extension() == ".spv";
                    try {
                        if (copy_source) {
                            input::validate_spirv_file(
                                source, "source shader module");
                            fs::copy_file(
                                source, temporary,
                                fs::copy_options::overwrite_existing);
                        } else {
                            input::validate_text_file(source,
                                                      "GLSL shader source");
                            runGlslc(options.glslc_executable, source_root,
                                     source, temporary);
                        }
                        input::validate_spirv_file(temporary,
                                                   "compiled shader module");
                        replaceBuiltFile(temporary, destination);
                    } catch (...) {
                        fs::remove(temporary);
                        throw;
                    }
                    if (copy_source) {
                        ++copied;
                    } else {
                        ++compiled;
                    }
                } else {
                    ++current;
                }
                output_entries.push_back(output_entry);
            } catch (const std::exception &failure) {
                if (!options.build_fix) {
                    throw;
                }
                const bool compilation_failed =
                    dynamic_cast<const ShaderCompilationError *>(&failure) !=
                    nullptr;
                if (!destination.empty()) {
                    std::error_code remove_error;
                    fs::remove(destination, remove_error);
                    if (remove_error) {
                        throw std::runtime_error(
                            "unable to remove failed shader output " +
                            destination.string() + ": " +
                            remove_error.message());
                    }
                }
                if (options.build_prune && compilation_failed &&
                    !source.empty() &&
                    (source.extension() == ".frag" ||
                     source.extension() == ".comp")) {
                    std::error_code remove_error;
                    const bool removed = fs::remove(source, remove_error);
                    if (remove_error || !removed) {
                        throw std::runtime_error(
                            "unable to prune failed shader source " +
                            source.string() +
                            (remove_error ? ": " + remove_error.message()
                                          : ": file was not removed"));
                    }
                    ++pruned;
                    std::cerr << "acmxvk: pruned failed source '"
                              << source.string() << "'\n";
                }
                ++failed;
                std::cerr << "acmxvk: fix omitted '" << entry
                          << "': " << failure.what() << '\n';
            }
            report_progress();
        }

        const fs::path output_manifest = output_root / "library.json";
        if (fs::is_symlink(output_manifest)) {
            throw std::runtime_error(
                "refusing to replace a symbolic-link output library.json");
        }
        const fs::path temporary_manifest =
            temporaryBuildPath(output_manifest);
        {
            std::ofstream output(temporary_manifest,
                                 std::ios::out | std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "unable to create output library.json");
            }
            output << "{\n    \"version\": 1"
                   << ",\n    \"backend\": \"acmxvk\""
                   << ",\n    \"library_type\": \"runtime\"";
            if (!manifest.custom_uniforms.empty()) {
                output << ",\n    \"custom_uniforms\": {\n";
                for (std::size_t index = 0;
                     index < manifest.custom_uniforms.size(); ++index) {
                    const ShaderManifest::CustomUniform &uniform =
                        manifest.custom_uniforms[index];
                    output << "        \"" << escapeJson(uniform.name)
                           << "\": {\n"
                           << std::setprecision(15)
                           << "            \"slot\": " << uniform.slot
                           << ",\n            \"minimum\": " << uniform.minimum
                           << ",\n            \"maximum\": " << uniform.maximum
                           << ",\n            \"step\": " << uniform.step
                           << ",\n            \"value\": " << uniform.value
                           << "\n        }";
                    output << (index + 1U < manifest.custom_uniforms.size()
                                   ? ",\n"
                                   : "\n");
                }
                output << "    }";
            }
            output << ",\n    \"shaders\": [\n";
            for (std::size_t index = 0; index < output_entries.size();
                 ++index) {
                output << "        \"" << escapeJson(output_entries[index])
                       << '"'
                       << (index + 1U < output_entries.size() ? ",\n"
                                                              : "\n");
            }
            output << "    ]\n}\n";
            if (!output) {
                fs::remove(temporary_manifest);
                throw std::runtime_error(
                    "unable to write output library.json");
            }
        }
        try {
            input::validate_text_file(temporary_manifest,
                                      "built shader library.json");
            replaceBuiltFile(temporary_manifest, output_manifest);
        } catch (...) {
            fs::remove(temporary_manifest);
            throw;
        }

        std::cout << "acmxvk: shader library built in " << output_root << '\n'
                  << "acmxvk: " << compiled << " compiled, " << copied
                  << " copied, " << current << " up to date, "
                  << failed << " failed, " << pruned << " pruned, "
                  << output_entries.size()
                  << " included\n";
        return EXIT_SUCCESS;
    }

    [[nodiscard]] cv::Mat loadRgbaImage(const std::string &filename) {
        constexpr std::uintmax_t MAX_IMAGE_FILE_BYTES =
            512U * 1024U * 1024U;
        constexpr std::int64_t MAX_IMAGE_PIXELS = 67108864;
        input::validate_file_size(filename, "graphic input",
                                  MAX_IMAGE_FILE_BYTES);
        const cv::Mat source = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (source.empty()) {
            throw std::runtime_error("unable to load image: " + filename);
        }
        if (source.cols <= 0 || source.rows <= 0 ||
            static_cast<std::int64_t>(source.cols) * source.rows >
                MAX_IMAGE_PIXELS) {
            throw std::runtime_error(
                "graphic input dimensions exceed the supported limit");
        }

        cv::Mat rgba;
        switch (source.channels()) {
        case 1:
            cv::cvtColor(source, rgba, cv::COLOR_GRAY2RGBA);
            break;
        case 3:
            cv::cvtColor(source, rgba, cv::COLOR_BGR2RGBA);
            break;
        case 4:
            cv::cvtColor(source, rgba, cv::COLOR_BGRA2RGBA);
            break;
        default:
            throw std::runtime_error("unsupported image channel count: " +
                                     std::to_string(source.channels()));
        }
        return rgba;
    }

    [[nodiscard]] double probeVideoDuration(const std::string &filename) {
        if (filename.empty() || filename.find("://") != std::string::npos) {
            return 0.0;
        }

        AVFormatContext *format = nullptr;
        if (avformat_open_input(&format, filename.c_str(), nullptr, nullptr) < 0) {
            return 0.0;
        }
        const auto close_format = [&format] {
            if (format != nullptr) {
                avformat_close_input(&format);
            }
        };
        if (avformat_find_stream_info(format, nullptr) < 0) {
            close_format();
            return 0.0;
        }

        double duration = 0.0;
        if (format->duration != AV_NOPTS_VALUE && format->duration > 0) {
            duration = static_cast<double>(format->duration) /
                       static_cast<double>(AV_TIME_BASE);
        } else {
            const int stream_index = av_find_best_stream(
                format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (stream_index >= 0) {
                const AVStream *stream = format->streams[stream_index];
                if (stream->duration != AV_NOPTS_VALUE &&
                    stream->duration > 0 && stream->time_base.num > 0 &&
                    stream->time_base.den > 0) {
                    duration = static_cast<double>(stream->duration) *
                               static_cast<double>(stream->time_base.num) /
                               static_cast<double>(stream->time_base.den);
                }
            }
        }
        close_format();
        return std::isfinite(duration) && duration > 0.0 ? duration : 0.0;
    }

    void rotateFrame(cv::Mat &frame, FrameRotation rotation) {
        switch (rotation) {
        case FrameRotation::None:
            break;
        case FrameRotation::Clockwise90:
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            break;
        case FrameRotation::Rotate180:
            cv::rotate(frame, frame, cv::ROTATE_180);
            break;
        case FrameRotation::Counterclockwise90:
            cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        }
    }

    [[nodiscard]] bool rotationSwapsDimensions(FrameRotation rotation) {
        return rotation == FrameRotation::Clockwise90 ||
               rotation == FrameRotation::Counterclockwise90;
    }

#ifdef ACMXVK_WITH_MXVK_CUDA
    void select_cuda_device(int device_index) {
        const int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (device_count <= 0) {
            throw std::runtime_error("no CUDA-capable devices are available");
        }
        if (device_index < 0 || device_index >= device_count) {
            throw std::runtime_error(
                "CUDA device index must be between 0 and " +
                std::to_string(device_count - 1));
        }
        cv::cuda::setDevice(device_index);
        const cv::cuda::DeviceInfo device(device_index);
        std::cout << "acmxvk: CUDA device " << device_index << ": "
                  << device.name() << '\n';
    }

    void list_cuda_devices(std::ostream &output) {
        const int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (device_count < 0) {
            throw std::runtime_error(
                "OpenCV could not query CUDA devices (error " +
                std::to_string(device_count) + ")");
        }
        output << "acmxvk: found " << device_count << " CUDA device(s)\n";
        for (int index = 0; index < device_count; ++index) {
            const cv::cuda::DeviceInfo device(index);
            output << "  " << index << ": " << device.name() << " ("
                   << (device.totalMemory() / (1024U * 1024U)) << " MiB)\n";
        }
    }
#endif

    struct PlaylistNode {
        std::string name;
        std::vector<fs::path> shaders;
    };

    class LatestCameraFrame {
      public:
        LatestCameraFrame() = default;
        ~LatestCameraFrame() { stop(); }
        LatestCameraFrame(const LatestCameraFrame &) = delete;
        LatestCameraFrame &operator=(const LatestCameraFrame &) = delete;
        LatestCameraFrame(LatestCameraFrame &&) = delete;
        LatestCameraFrame &operator=(LatestCameraFrame &&) = delete;

        void start(mxvk::VK_Capture &source) {
            stop();
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                capture_source = &source;
                stopping = false;
                latest_frame.release();
                published_generation = 0;
                consumed_generation = 0;
            }
            capture_thread = std::thread(&LatestCameraFrame::captureLoop, this);
        }

        void stop() noexcept {
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                stopping = true;
            }
            frame_condition.notify_all();
            if (capture_thread.joinable()) {
                capture_thread.join();
            }
            std::lock_guard<std::mutex> lock(frame_mutex);
            capture_source = nullptr;
            latest_frame.release();
            published_generation = 0;
            consumed_generation = 0;
        }

        [[nodiscard]] bool takeLatest(cv::Mat &frame, bool wait_for_first) {
            std::unique_lock<std::mutex> lock(frame_mutex);
            if (wait_for_first && published_generation == 0 && !stopping) {
                frame_condition.wait_for(lock, std::chrono::seconds(3), [&] {
                    return stopping || published_generation > 0;
                });
            }
            if (stopping || published_generation == consumed_generation ||
                latest_frame.empty()) {
                return false;
            }
            frame = latest_frame;
            consumed_generation = published_generation;
            return true;
        }

      private:
        void captureLoop() noexcept {
            while (true) {
                mxvk::VK_Capture *source = nullptr;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (stopping) {
                        return;
                    }
                    source = capture_source;
                }
                if (source == nullptr) {
                    return;
                }

                cv::Mat captured;
                bool read_frame = false;
                try {
                    read_frame = source->read(captured);
                } catch (const std::exception &error) {
                    std::cerr << "acmxvk: asynchronous camera read failed: "
                              << error.what() << '\n';
                } catch (...) {
                    std::cerr << "acmxvk: asynchronous camera read failed\n";
                }
                if (!read_frame || captured.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (stopping) {
                        return;
                    }
                    latest_frame = std::move(captured);
                    ++published_generation;
                }
                frame_condition.notify_one();
            }
        }

        mxvk::VK_Capture *capture_source = nullptr;
        cv::Mat latest_frame;
        std::thread capture_thread;
        std::mutex frame_mutex;
        std::condition_variable frame_condition;
        std::uint64_t published_generation = 0;
        std::uint64_t consumed_generation = 0;
        bool stopping = true;
    };

    class MainWindow final : public mxvk::VK_Window {
      public:
        explicit MainWindow(Options options)
            : mxvk::VK_Window("ACMXVK", options.width, options.height,
                              options.fullscreen, MXVK_VALIDATION, options.enable_vsync),
              options(std::move(options)) {
            setClearColor(0.0F, 0.0F, 0.0F, 1.0F);
            setEnableScreenshot(this->options.enable_screenshot);
            resolveConfiguredResourcePaths();
            initializeDnn();
            initializeGpuFilters();
            openAudio();
            loadShaders();
            loadShaderPasses();
            initialize_interface_control();
            configureMidiMappings();
            openMidi();
            loadPlaylist();
            resetAutopilotInterval();
            openInput();
            configureRenderResolution();
            initializeSprite();
            initializeOverlayFont();
            start_requested_audio_recording();
            openOutput();
            updateWindowTitle(true);
        }

        ~MainWindow() override {
            cleanup_interface_control();
            latest_camera_frame.stop();
            try {
                flushFrameReadbacks();
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: unable to flush pending frame readbacks: "
                          << error.what() << '\n';
            }
            if (model_initialized && getDevice() != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(getDevice());
                input_model.cleanup(this);
                model_initialized = false;
                std::cout << "acmxvk: released 3D model resources\n";
            }
            const bool should_copy_audio =
                options.copy_audio && !options.mute_output && writer.is_open();
#ifdef AUDIO_ENABLED
            const bool should_mux_file_audio =
                file_audio_source != nullptr && writer.is_open() &&
                !options.output_file.empty() && !options.png_output &&
                !options.mute_output && output_frame_count > 0;
            const bool should_mux_live_audio =
                audio_engine != nullptr && file_audio_source == nullptr &&
                audio_engine->is_recording() && writer.is_open() &&
                !options.output_file.empty() && !options.png_output &&
                !options.copy_audio && !options.mute_output &&
                output_frame_count > 0;
            const bool should_write_live_audio =
                audio_engine != nullptr && audio_engine->is_recording() &&
                !options.record_audio_file.empty();
            audio::AudioRecording live_audio_recording;
            if (audio_engine != nullptr && audio_engine->is_recording()) {
                live_audio_recording = audio_engine->stop_recording();
            }
            if (file_audio_source != nullptr) {
                file_audio_source->stop_output();
            }
#endif
            if (writer.is_open()) {
                writer.close();
                std::cout << "acmxvk: recording closed after " << output_frame_count
                          << " frames\n";
            }
            if (options.png_output) {
                std::cout << "acmxvk: PNG sequence closed after " << png_frame_count
                          << " frames\n";
            }
            if (options.generate_interval > 0) {
                std::cout << "acmxvk: generated " << generated_frame_count
                          << " periodic PNG frames\n";
            }
            if (capture.is_open()) {
                capture.close();
            }
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.is_open()) {
                ffmpeg_capture.close();
            }
#endif
            if (should_copy_audio) {
                transfer_audio(options.input_file, options.output_file);
                std::cout << "acmxvk: copied audio track from " << options.input_file
                          << " to " << options.output_file << '\n';
            }
#ifdef AUDIO_ENABLED
            if (should_mux_file_audio) {
                const double video_duration = writer.get_duration();
                if (!file_audio_source->mux_into_video(options.output_file,
                                                       video_duration)) {
                    std::cerr << "acmxvk: file-audio mux failed; preserving the "
                                 "encoded video without audio\n";
                }
            }
            if (should_write_live_audio) {
                if (live_audio_recording.empty()) {
                    std::cerr << "acmxvk: standalone audio recording was empty; "
                                 "no WAV file was written\n";
                } else if (!audio::write_wav_file(live_audio_recording,
                                                  options.record_audio_file)) {
                    std::cerr << "acmxvk: could not write WAV recording: "
                              << options.record_audio_file << '\n';
                } else {
                    std::cout << "acmxvk: wrote "
                              << live_audio_recording.duration_seconds()
                              << " seconds of microphone audio to "
                              << options.record_audio_file << '\n';
                }
            }
            if (should_mux_live_audio) {
                const double video_duration = writer.get_duration();
                if (live_audio_recording.empty()) {
                    std::cerr << "acmxvk: live audio recording was empty; preserving "
                                 "the encoded video without audio\n";
                } else if (!audio::FileAudioSource::mux_recording_into_video(
                               std::move(live_audio_recording.samples),
                               live_audio_recording.sample_rate,
                               options.output_file, video_duration)) {
                    std::cerr << "acmxvk: live-audio mux failed; preserving the "
                                 "encoded video without audio\n";
                }
            }
#endif
            stopSnapshotWorker();
        }

        void event(SDL_Event &event) override {
            mxvk::VK_Window::event(event);
            if (event.type == SDL_EVENT_KEY_DOWN &&
                event.key.key == SDLK_PAGEUP) {
                adjustTimeSpeed(0.1);
            } else if (event.type == SDL_EVENT_KEY_DOWN &&
                       event.key.key == SDLK_PAGEDOWN) {
                adjustTimeSpeed(-0.1);
            }

            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                switch (event.key.key) {
                case SDLK_UP:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0 || !playlist_enabled) {
                        selectShader(-1);
                    } else {
                        selectPlaylistNode(-1);
                    }
                    break;
                case SDLK_DOWN:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0 || !playlist_enabled) {
                        selectShader(1);
                    } else {
                        selectPlaylistNode(1);
                    }
                    break;
                case SDLK_LEFT:
                    selectGpuFilter(-1);
                    break;
                case SDLK_RIGHT:
                    selectGpuFilter(1);
                    break;
                case SDLK_SPACE:
                    beginCrossfade();
                    effects_enabled = !effects_enabled;
                    applyShaderPipeline();
                    std::cout << "acmxvk: shader effects "
                              << (effects_enabled ? "enabled" : "bypassed") << '\n';
                    break;
                case SDLK_P:
                    if (!playlist.empty()) {
                        beginCrossfade();
                        playlist_enabled = !playlist_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: playlist "
                                  << (playlist_enabled ? "enabled" : "disabled") << '\n';
                        if (playlist_enabled) {
                            logSelectedPlaylistNode("selected");
                        }
                    } else {
                        togglePause();
                    }
                    break;
                case SDLK_L:
                    toggleFreeze();
                    break;
                case SDLK_T:
                    shader_time_active = !shader_time_active;
                    previous_frame = std::chrono::steady_clock::now();
                    std::cout << "acmxvk: shader time "
                              << (shader_time_active ? "enabled" : "disabled") << '\n';
                    break;
                case SDLK_Q:
#ifdef AUDIO_ENABLED
                    if (audioSourceOpen()) {
                        audio_time_active = !audio_time_active;
                        previous_frame = std::chrono::steady_clock::now();
                        std::cout << "acmxvk: audio-reactive shader time "
                                  << (audio_time_active ? "enabled" : "disabled")
                                  << '\n';
                    }
#endif
                    break;
                case SDLK_HOME:
#ifdef AUDIO_ENABLED
                    if (audioSourceOpen()) {
                        audio_delta_time = !audio_delta_time;
                        std::cout << "acmxvk: audio delta-time scaling "
                                  << (audio_delta_time ? "enabled" : "disabled")
                                  << '\n';
                    }
#endif
                    break;
                case SDLK_END:
#ifdef AUDIO_ENABLED
                    if (audioSourceOpen()) {
                        spectrum_scale_by_sensitivity =
                            !spectrum_scale_by_sensitivity;
                        std::cout << "acmxvk: spectrum sensitivity scaling "
                                  << (spectrum_scale_by_sensitivity ? "enabled"
                                                                    : "disabled")
                                  << '\n';
                    }
#endif
                    break;
                case SDLK_U:
                    stepShaderTime(0.05);
                    break;
                case SDLK_I:
                    stepShaderTime(-0.05);
                    break;
                case SDLK_F:
                    toggleFullscreen();
                    break;
                case SDLK_F9:
                    counter_disabled = !counter_disabled;
                    hud_fps_frame_count = 0;
                    hud_fps_last_tick = std::chrono::steady_clock::now();
                    if (!counter_disabled) {
                        initializeOverlayFont();
                    }
                    std::cout << "acmxvk: runtime HUD "
                              << (counter_disabled ? "hidden" : "shown")
                              << " (F9)\n";
                    break;
                case SDLK_E:
                    if (!options.watermark_text.empty()) {
                        watermark_enabled = !watermark_enabled;
                        std::cout << "acmxvk: watermark "
                                  << (watermark_enabled ? "enabled" : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_INSERT:
                    adjustAudioSensitivity(0.1F);
                    break;
                case SDLK_DELETE:
                    adjustAudioSensitivity(-0.1F);
                    break;
                case SDLK_M:
                    if (!configured_passes.empty()) {
                        beginCrossfade();
                        multipass_enabled = !multipass_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: multipass "
                                  << (multipass_enabled ? "enabled" : "disabled") << '\n';
                    }
                    break;
                case SDLK_J:
                    toggleAutopilot(false);
                    break;
                case SDLK_N:
                    autopilot_random_crossfade =
                        !autopilot_random_crossfade;
                    std::cout << "acmxvk: random autopilot crossfade "
                              << (autopilot_random_crossfade ? "enabled"
                                                             : "disabled")
                              << '\n';
                    break;
                case SDLK_K:
                    shader_locked = !shader_locked;
                    std::cout << "acmxvk: shader lock "
                              << (shader_locked ? "enabled" : "disabled")
                              << '\n';
                    break;
                case SDLK_3:
                    if (model_initialized) {
                        model_3d_active = !model_3d_active;
                        model_video_timeline_initialized = false;
                        model_last_render_time =
                            std::chrono::steady_clock::now();
                        applyShaderPipeline();
                        std::cout << "acmxvk: "
                                  << (model_3d_active ? "3D model" : "2D sprite")
                                  << " rendering enabled\n";
                    }
                    break;
                case SDLK_V:
                    if (model_initialized) {
                        model_auto_rotate = !model_auto_rotate;
                        std::cout << "acmxvk: 3D view rotation "
                                  << (model_auto_rotate ? "enabled" : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_C:
                    if (model_initialized) {
                        model_wave_active = !model_wave_active;
                        std::cout << "acmxvk: 3D wave effect "
                                  << (model_wave_active ? "enabled"
                                                        : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_O:
                    if (model_initialized) {
                        model_scale_oscillation_active =
                            !model_scale_oscillation_active;
                        std::cout << "acmxvk: 3D scale oscillation "
                                  << (model_scale_oscillation_active
                                          ? "enabled"
                                          : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_X:
                    if (model_initialized) {
                        model_pitch_degrees = 0.0F;
                        model_yaw_degrees = 270.0F;
                        model_rotation_x_degrees = 0.0F;
                        model_rotation_y_degrees = 0.0F;
                        model_rotation_z_degrees = 0.0F;
                        model_camera_distance = 0.0F;
                        model_scale = 1.0F;
                        model_view_rotation_degrees = 0.0F;
                        std::cout << "acmxvk: model view reset\n";
                    }
                    break;
                case SDLK_LEFTBRACKET:
                    cycleCrossfade(-1);
                    break;
                case SDLK_RIGHTBRACKET:
                    cycleCrossfade(1);
                    break;
                case SDLK_MINUS:
                case SDLK_UNDERSCORE:
                case SDLK_KP_MINUS:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0) {
                        adjustModelScale(-0.05F);
                    }
                    break;
                case SDLK_PLUS:
                case SDLK_EQUALS:
                case SDLK_KP_PLUS:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0) {
                        adjustModelScale(0.05F);
                    }
                    break;
                case SDLK_COMMA:
                    if (model_initialized) {
                        model_rotation_speed =
                            std::max(0.0F, model_rotation_speed - 5.0F);
                        std::cout << "acmxvk: 3D view rotation speed "
                                  << model_rotation_speed << " degrees/second\n";
                    }
                    break;
                case SDLK_PERIOD:
                    if (model_initialized) {
                        model_rotation_speed =
                            std::min(360.0F, model_rotation_speed + 5.0F);
                        std::cout << "acmxvk: 3D view rotation speed "
                                  << model_rotation_speed << " degrees/second\n";
                    }
                    break;
                case SDLK_Y:
                    toggleAutopilot(true);
                    break;
                case SDLK_Z:
                    requestSnapshot(SnapshotFormat::Png);
                    break;
                case SDLK_4:
                    requestSnapshot(SnapshotFormat::Tiff);
                    break;
                case SDLK_5:
                    requestSnapshot(SnapshotFormat::WebP);
                    break;
                case SDLK_6:
                    requestSnapshot(SnapshotFormat::Raw);
                    break;
                default:
                    break;
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                mouse_x = event.motion.x;
                mouse_y = event.motion.y;
                if (model_mouse_dragging && model_initialized) {
                    const int x = static_cast<int>(event.motion.x);
                    const int y = static_cast<int>(event.motion.y);
                    model_yaw_degrees +=
                        static_cast<float>(x - model_last_mouse_x) * 0.35F;
                    model_pitch_degrees = std::clamp(
                        model_pitch_degrees +
                            static_cast<float>(y - model_last_mouse_y) * 0.35F,
                        -89.0F, 89.0F);
                    model_last_mouse_x = x;
                    model_last_mouse_y = y;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = true;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
                model_mouse_dragging = model_initialized;
                model_last_mouse_x = static_cast<int>(event.button.x);
                model_last_mouse_y = static_cast<int>(event.button.y);
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = false;
                model_mouse_dragging = false;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL &&
                       model_initialized &&
                       !model_scale_oscillation_active) {
                const float wheel = event.wheel.y != 0.0F
                                        ? event.wheel.y
                                        : static_cast<float>(
                                              event.wheel.integer_y);
                model_camera_distance = std::clamp(
                    model_camera_distance - wheel * 0.2F, -20.0F, 20.0F);
            }
        }

        void onSwapchainRecreated() override {
            initializeSprite();
            if (model_initialized) {
                input_model.resize(this);
            }
            initializeOverlayFont();
        }

        void onRecordCustomRendering(VkCommandBuffer command_buffer,
                                     std::uint32_t image_index) override {
            if (model_texture_prepass_active) {
                return;
            }
            recordModel(command_buffer, image_index, VK_NULL_HANDLE);
        }

        void onRecordPostProcessingTexture(
            VkCommandBuffer command_buffer, std::uint32_t image_index,
            VkImageView texture_view,
            [[maybe_unused]] VkExtent2D texture_extent) override {
            if (!model_texture_prepass_active) {
                return;
            }
            recordModel(command_buffer, image_index, texture_view);
        }

        void recordModel(VkCommandBuffer command_buffer,
                         std::uint32_t image_index,
                         VkImageView texture_view) {
            if (!model_3d_active || !model_initialized) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            float delta = std::chrono::duration<float>(
                              now - model_last_render_time)
                              .count();
            model_last_render_time = now;
            delta = std::clamp(delta, 0.0F, 0.1F);

            float animation_delta = delta;
            std::uint64_t animation_steps = 1U;
            double video_timeline = 0.0;
            std::uint64_t video_frame_index = 0U;
            if (currentVideoTimeline(video_timeline, &video_frame_index)) {
                if (!model_video_timeline_initialized ||
                    video_frame_index < previous_model_video_frame) {
                    if (model_video_timeline_initialized &&
                        video_frame_index < previous_model_video_frame) {
                        model_wave_phase = 0.0F;
                        model_wave_amplitude_x = 0.0F;
                        model_wave_amplitude_y = 0.0F;
                        model_wave_amplitude_z = 0.0F;
                        model_wave_direction_x = 1.0F;
                        model_wave_direction_y = 1.0F;
                        model_wave_direction_z = 1.0F;
                        model_scale_oscillation_phase = 0.0F;
                        if (model_auto_rotate) {
                            model_view_rotation_degrees = 0.0F;
                        }
                    }
                    animation_delta = 0.0F;
                    animation_steps = 0U;
                    model_video_timeline_initialized = true;
                } else {
                    animation_steps =
                        video_frame_index - previous_model_video_frame;
                    animation_delta = static_cast<float>(
                        static_cast<double>(animation_steps) /
                        video_source_fps);
                }
                previous_model_video_frame = video_frame_index;
            } else {
                model_video_timeline_initialized = false;
            }
            if (model_auto_rotate && !rendering_frozen) {
                model_view_rotation_degrees = std::fmod(
                    model_view_rotation_degrees +
                        model_rotation_speed * animation_delta,
                    360.0F);
            }
            if (model_wave_active) {
                const float wave_step =
                    audio_time_active && audioSourceOpen()
                        ? model_wave_audio_step
                        : 0.05F;
                model_wave_phase = std::fmod(
                    model_wave_phase +
                        wave_step * static_cast<float>(animation_steps),
                    360.0F);

                const auto advance_amplitude = [](float &amplitude,
                                                  float &direction,
                                                  std::uint64_t steps) {
                    constexpr float AMPLITUDE_RANGE = 0.5F;
                    constexpr float AMPLITUDE_PERIOD =
                        AMPLITUDE_RANGE * 2.0F;
                    float phase = direction >= 0.0F
                                      ? amplitude
                                      : AMPLITUDE_PERIOD - amplitude;
                    phase = std::fmod(
                        phase + 0.005F * static_cast<float>(steps),
                        AMPLITUDE_PERIOD);
                    if (phase < AMPLITUDE_RANGE) {
                        amplitude = phase;
                        direction = 1.0F;
                    } else {
                        amplitude = AMPLITUDE_PERIOD - phase;
                        direction = -1.0F;
                    }
                };
                advance_amplitude(model_wave_amplitude_x,
                                  model_wave_direction_x, animation_steps);
                advance_amplitude(model_wave_amplitude_y,
                                  model_wave_direction_y, animation_steps);
                advance_amplitude(model_wave_amplitude_z,
                                  model_wave_direction_z, animation_steps);
            }
            if (model_scale_oscillation_active) {
                model_scale_oscillation_phase = std::fmod(
                    model_scale_oscillation_phase +
                        0.016F * static_cast<float>(animation_steps),
                    2.0F * std::numbers::pi_v<float>);
            }

            const bool *keyboard = SDL_GetKeyboardState(nullptr);
            const bool model_scale_modifier =
                (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
            if (!model_scale_oscillation_active &&
                keyboard[SDL_SCANCODE_1]) {
                model_camera_movement_speed = std::clamp(
                    model_camera_movement_speed + 0.1F * delta * 30.0F,
                    0.01F, 20.0F);
            }
            if (!model_scale_oscillation_active &&
                keyboard[SDL_SCANCODE_2]) {
                model_camera_movement_speed = std::clamp(
                    model_camera_movement_speed - 0.1F * delta * 30.0F,
                    0.01F, 20.0F);
            }
            if (!model_scale_oscillation_active && !model_scale_modifier &&
                (keyboard[SDL_SCANCODE_EQUALS] ||
                 keyboard[SDL_SCANCODE_KP_PLUS])) {
                model_camera_distance = std::clamp(
                    model_camera_distance +
                        model_camera_movement_speed * delta,
                    -20.0F, 20.0F);
            }
            if (!model_scale_oscillation_active && !model_scale_modifier &&
                (keyboard[SDL_SCANCODE_MINUS] ||
                 keyboard[SDL_SCANCODE_KP_MINUS])) {
                model_camera_distance = std::clamp(
                    model_camera_distance -
                        model_camera_movement_speed * delta,
                    -20.0F, 20.0F);
            }
            if (!model_auto_rotate) {
                if (keyboard[SDL_SCANCODE_W]) {
                    model_pitch_degrees +=
                        model_camera_rotation_speed * 0.3F * delta * 30.0F;
                }
                if (keyboard[SDL_SCANCODE_S]) {
                    model_pitch_degrees -=
                        model_camera_rotation_speed * 0.33F * delta * 30.0F;
                }
                model_pitch_degrees =
                    std::fmod(model_pitch_degrees, 360.0F);
                if (model_pitch_degrees < 0.0F) {
                    model_pitch_degrees += 360.0F;
                }
                if (keyboard[SDL_SCANCODE_A]) {
                    model_yaw_degrees -=
                        model_camera_rotation_speed * 0.3F * delta * 30.0F;
                }
                if (keyboard[SDL_SCANCODE_D]) {
                    model_yaw_degrees +=
                        model_camera_rotation_speed * 0.3F * delta * 30.0F;
                }
                model_yaw_degrees = std::fmod(model_yaw_degrees, 360.0F);
                if (model_yaw_degrees < 0.0F) {
                    model_yaw_degrees += 360.0F;
                }
            }

            const VkExtent2D extent = getRenderExtent();
            const float aspect = extent.height > 0U
                                     ? static_cast<float>(extent.width) /
                                           static_cast<float>(extent.height)
                                     : 1.0F;

            mxvk::UniformBufferObject uniforms{};
            uniforms.model = glm::scale(
                glm::mat4(1.0F),
                glm::vec3(input_model.modelRenderScale() * model_scale));
            uniforms.model = glm::rotate(
                uniforms.model, glm::radians(model_rotation_x_degrees),
                glm::vec3(1.0F, 0.0F, 0.0F));
            uniforms.model = glm::rotate(
                uniforms.model, glm::radians(model_rotation_y_degrees),
                glm::vec3(0.0F, 1.0F, 0.0F));
            uniforms.model = glm::rotate(
                uniforms.model, glm::radians(model_rotation_z_degrees),
                glm::vec3(0.0F, 0.0F, 1.0F));
            uniforms.model = glm::translate(
                uniforms.model, input_model.modelCenterOffset());

            glm::vec3 look_direction{};
            glm::vec3 camera_up(0.0F, 1.0F, 0.0F);
            if (model_auto_rotate) {
                const float rotation =
                    glm::radians(model_view_rotation_degrees);
                look_direction = glm::vec3(
                    0.48F * std::sin(rotation),
                    0.48F * std::sin(rotation * 0.7F),
                    0.48F * std::cos(rotation));
            } else {
                const float pitch = glm::radians(model_pitch_degrees);
                const float yaw = glm::radians(model_yaw_degrees);
                look_direction = glm::normalize(glm::vec3(
                                     std::cos(pitch) * std::cos(yaw),
                                     std::sin(pitch),
                                     std::cos(pitch) * std::sin(yaw))) *
                                 0.48F;
                camera_up = glm::vec3(-std::sin(pitch) * std::cos(yaw),
                                      std::cos(pitch),
                                      -std::sin(pitch) * std::sin(yaw));
            }
            const float camera_offset =
                model_scale_oscillation_active
                    ? 0.3F * std::sin(model_scale_oscillation_phase)
                    : model_camera_distance;
            const glm::vec3 camera_position =
                -glm::normalize(look_direction) * camera_offset;
            uniforms.view = glm::lookAt(camera_position,
                                        camera_position + look_direction,
                                        camera_up);
            uniforms.proj = glm::perspective(
                glm::radians(120.0F), aspect, 0.01F, 1000.0F);
            uniforms.proj[1][1] *= -1.0F;
            uniforms.fx =
                model_wave_active
                    ? glm::vec4(model_wave_amplitude_x,
                                model_wave_amplitude_y,
                                model_wave_amplitude_z, model_wave_phase)
                    : glm::vec4(0.0F);

            input_model.updateFragmentUBO(image_index,
                                          model_fragment_uniforms);

            mxvk::ModelFragmentPushConstants fragment_constants{};
            fragment_constants.screenWidth = static_cast<float>(extent.width);
            fragment_constants.screenHeight = static_cast<float>(extent.height);
            fragment_constants.spriteSizeW = static_cast<float>(extent.width);
            fragment_constants.spriteSizeH = static_cast<float>(extent.height);
            fragment_constants.effectsOn = effects_enabled ? 1.0F : 0.0F;
            fragment_constants.params = glm::vec4(
                1.0F, 1.0F, 1.0F, static_cast<float>(shader_time));
            input_model.setFragmentPushConstants(fragment_constants);

            if (texture_view != VK_NULL_HANDLE) {
                input_model.renderWithExternalTexture(
                    command_buffer, image_index, texture_view, uniforms,
                    false);
            } else {
                input_model.renderWithPushConstants(
                    command_buffer, image_index, 0U, uniforms, false);
            }
        }

        void proc() override {
            if (recording_complete) {
                return;
            }

            paceMaximizedRendering();

            pollMidi();
            sync_interface_control();

            source_frame_received = false;
            recording_frame_due = false;
            recording_frame_has_pts = false;
            bool clocked_video_handled = false;

            if (!rendering_frozen && !input_paused) {
                if (source_kind == SourceKind::Graphic) {
                    source_frame_received = true;
                } else if (initial_frame_pending) {
                    initial_frame_pending = false;
                    source_frame_received = true;
                } else {
                    double clock_seconds = 0.0;
                    if (source_kind == SourceKind::Video &&
                        media_timeline_started &&
                        mediaClockSeconds(clock_seconds)) {
                        clocked_video_handled = true;
                        if (!readClockedVideoFrame(clock_seconds)) {
                            return;
                        }
                    } else {
                        const bool read_frame = readTrackedInputFrame();
                        if (!read_frame && !handleCaptureEnd()) {
                            return;
                        }
                        source_frame_received =
                            read_frame || source_kind == SourceKind::Video;
                    }
                }
            }

            startMediaTimelineIfReady();
            const bool render_latest_camera_frame =
                options.maximize_fps && source_kind == SourceKind::Camera &&
                media_timeline_started;
            if ((source_frame_received || render_latest_camera_frame) &&
                !clocked_video_handled) {
                recording_frame_due = true;
                if (source_kind == SourceKind::Video) {
                    recording_frame_has_pts = true;
                    recording_frame_pts = decoded_video_frame_count - 1;
                } else {
                    double clock_seconds = 0.0;
                    if (mediaClockSeconds(clock_seconds)) {
                        const double rate = outputFrameRate();
                        const std::uint64_t target_frame =
                            static_cast<std::uint64_t>(std::floor(
                                std::max(clock_seconds, 0.0) * rate));
                        if (target_frame < next_clock_output_frame) {
                            recording_frame_due = false;
                        } else {
                            recording_frame_has_pts = true;
                            recording_frame_pts = target_frame;
                            next_clock_output_frame = target_frame + 1;
                            if (source_kind == SourceKind::Camera &&
                                writer.is_open() &&
                                !camera_recording_clock_logged) {
                                std::cout
                                    << "acmxvk: camera recording uses real-time "
                                       "PTS; slow frames preserve capture duration\n";
                                camera_recording_clock_logged = true;
                            }
                        }
                    }
                }
            }

            if (!rendering_frozen) {
                updateAutopilot();
            }
            updateCameraHistory();
            const VkExtent2D extent = getRenderExtent();
            const int target_width = extent.width > 0U ? static_cast<int>(extent.width) : options.width;
            const int target_height =
                extent.height > 0U ? static_cast<int>(extent.height) : options.height;

            if (!rendering_frozen) {
                updateShaderUniforms(target_width, target_height);
            }
            if (!model_3d_active || model_texture_prepass_active) {
                frame_sprite->drawSpriteRect(0, 0, target_width,
                                             target_height);
            }
            queueOverlayText();
            updateWindowTitle();
            setFrameReadbackEnabled(
                snapshot_pending ||
                (continuousReadbackEnabled() && recording_frame_due));
        }

      private:
        enum class SourceKind { Camera,
                                Video,
                                Graphic };

        enum class SnapshotFormat { Png,
                                    WebP,
                                    Tiff,
                                    Raw };

        struct SnapshotJob {
            fs::path path;
            std::vector<std::uint8_t> rgba;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            SnapshotFormat format = SnapshotFormat::Png;
        };

        struct ReadbackRequest {
            bool snapshot = false;
            SnapshotFormat snapshot_format = SnapshotFormat::Png;
            bool continuous = false;
            bool frame_due = false;
            bool has_pts = false;
            std::uint64_t pts = 0;
        };

        static constexpr std::size_t SNAPSHOT_QUEUE_CAPACITY = 4;
        static constexpr std::uint32_t COMPATIBILITY_SPECTRUM_BIN_COUNT = 256;

        Options options;
        SourceKind source_kind = SourceKind::Camera;
        mxvk::VK_Capture capture;
        LatestCameraFrame latest_camera_frame;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
        mxvk::VK_FF_Capture ffmpeg_capture;
        std::vector<std::uint8_t> ffmpeg_rgba;
        bool using_ffmpeg_capture = false;
        bool ffmpeg_seek_repeat_logged = false;
#ifdef ACMXVK_WITH_MXVK_CUDA
        cv::cuda::Stream ffmpeg_cuda_stream;
#endif
#endif
        Writer writer;
        mxvk::VKAbstractModel input_model;
        mxvk::VK_Sprite *frame_sprite = nullptr;
        mxvk::VK_Sprite *crossfade_previous_sprite = nullptr;
        mxvk::VK_Sprite *human_overlay_sprite = nullptr;
        cv::Mat graphic_rgba;
        cv::Mat human_overlay_rgba;
        cv::Mat latest_camera_history_rgba;
        std::vector<fs::path> shaders;
        std::vector<fs::path> configured_passes;
        std::vector<PlaylistNode> playlist;
        std::vector<mxvk::VK_Sprite *> post_process_sprites;
        std::vector<ShaderManifest::CustomUniform> custom_uniforms;
        std::vector<float> custom_uniform_values;
        fs::path shader_library_directory;
        fs::path shader_manifest_path;
        fs::path png_output_directory;
        fs::path generate_output_directory;
        int interface_shm_fd = -1;
        ipc::ShaderSelectionData *interface_selection = nullptr;
        sem_t *interface_semaphore = SEM_FAILED;
        std::uint32_t interface_last_sequence = 0;
        std::uint32_t interface_last_audio_file_sequence = 0;
        std::size_t shader_index = 0;
        std::size_t playlist_index = 0;
        std::size_t crossfade_post_process_index =
            std::numeric_limits<std::size_t>::max();
        bool effects_enabled = true;
        bool multipass_enabled = false;
        bool playlist_enabled = false;
        bool shader_locked = false;
        bool model_initialized = false;
        bool model_3d_active = false;
        bool model_texture_prepass_active = false;
        bool model_auto_rotate = false;
        bool model_wave_active = false;
        bool model_scale_oscillation_active = false;
        bool model_mouse_dragging = false;
        bool shader_history_required = false;
        bool shader_spectrum_required = false;
        bool shader_spectrum_history_required = false;
        float mouse_x = 0.0F;
        float mouse_y = 0.0F;
        bool mouse_pressed = false;
        bool history_initialized = false;
        bool initial_frame_pending = false;
        bool async_camera_frame_uploaded = false;
        bool async_camera_initial_wait_completed = false;
        bool render_pacing_started = false;
        bool media_timeline_started = false;
        bool source_frame_received = false;
        bool recording_frame_due = false;
        bool recording_frame_has_pts = false;
        bool media_clock_sync_logged = false;
        bool camera_recording_clock_logged = false;
        bool recording_complete = false;
        bool input_paused = false;
        bool rendering_frozen = false;
        bool source_playback_clock_paused = false;
        bool shader_time_active = true;
        bool audio_time_active = false;
        bool audio_delta_time = false;
        bool spectrum_scale_by_sensitivity = false;
        bool watermark_enabled = !options.watermark_text.empty();
        bool counter_disabled =
            options.disable_counter || !options.watermark_text.empty();
        int overlay_font_size = 18;
        int preview_overlay_font_size = 18;
        bool snapshot_pending = false;
        SnapshotFormat pending_snapshot_format = SnapshotFormat::Png;
        bool autopilot_enabled = false;
        bool autopilot_sequential = false;
        bool autopilot_random_crossfade = false;
        bool crossfade_active = false;
        int recording_width = 0;
        int recording_height = 0;
        int camera_reported_width = 0;
        int camera_reported_height = 0;
        int autopilot_counter = 0;
        int autopilot_interval_frames = 0;
        std::uint64_t previous_autopilot_video_frame = 0;
        int history_delay_counter = 0;
        std::size_t crossfade_shader_index = 0;
        bool camera_history_clock_started = false;
        double recording_fps = 0.0;
        double video_source_fps = 0.0;
        double video_duration_seconds = 0.0;
        double camera_reported_fps = 0.0;
        double camera_delivered_fps = 0.0;
        double camera_last_logged_fps = 0.0;
        double shader_time = 0.0;
        float legacy_alpha = 0.1F;
        float crossfade_alpha = 1.0F;
        float model_pitch_degrees = 0.0F;
        float model_yaw_degrees = 270.0F;
        float model_rotation_x_degrees = 0.0F;
        float model_rotation_y_degrees = 0.0F;
        float model_rotation_z_degrees = 0.0F;
        float model_camera_distance = 0.0F;
        float model_camera_movement_speed = 0.1F;
        float model_camera_rotation_speed = 5.0F;
        float model_scale = 1.0F;
        float model_rotation_speed = 18.0F;
        float model_view_rotation_degrees = 0.0F;
        float model_wave_amplitude_x = 0.0F;
        float model_wave_amplitude_y = 0.0F;
        float model_wave_amplitude_z = 0.0F;
        float model_wave_direction_x = 1.0F;
        float model_wave_direction_y = 1.0F;
        float model_wave_direction_z = 1.0F;
        float model_wave_phase = 0.0F;
        float model_wave_audio_step = 0.0F;
        float model_scale_oscillation_phase = 0.0F;
        fs::path model_effect_shader;
        mxvk::ModelFragmentUniforms model_fragment_uniforms{};
        bool legacy_alpha_increasing = true;
        int model_last_mouse_x = 0;
        int model_last_mouse_y = 0;
        std::chrono::steady_clock::time_point compatibility_clock_start =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point model_last_render_time =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point crossfade_start_time =
            std::chrono::steady_clock::now();
        double crossfade_start_video_timeline = 0.0;
        std::uint64_t output_frame_count = 0;
        std::uint64_t decoded_video_frame_count = 0;
        std::uint64_t video_source_frame_count = 0;
        std::uint64_t recording_frame_pts = 0;
        std::uint64_t next_clock_output_frame = 0;
        std::uint64_t png_frame_count = 0;
        std::uint64_t generated_frame_count = 0;
        std::uint64_t snapshot_count = 0;
        std::uint64_t frame_count = 0;
        std::uint64_t previous_model_video_frame = 0;
        std::uint64_t hud_fps_frame_count = 0;
        std::uint64_t camera_fps_frame_count = 0;
        double hud_display_fps = 0.0;
        std::deque<SnapshotJob> snapshot_jobs;
        std::mutex snapshot_mutex;
        std::condition_variable snapshot_condition;
        std::thread snapshot_worker;
        std::size_t snapshot_jobs_in_flight = 0;
        bool snapshot_worker_stopping = false;
        std::deque<ReadbackRequest> readback_requests;
        std::chrono::steady_clock::time_point hud_session_start{
            std::chrono::steady_clock::now()};
        std::chrono::steady_clock::time_point hud_fps_last_tick{
            hud_session_start};
        std::chrono::steady_clock::time_point camera_fps_last_tick{};
        std::chrono::steady_clock::time_point camera_history_next_update{};
        std::chrono::steady_clock::time_point window_title_last_update{};
        std::chrono::steady_clock::time_point next_render_tick{};
        std::chrono::steady_clock::time_point source_playback_clock_start{};
        std::chrono::steady_clock::time_point source_playback_pause_start{};
        std::chrono::steady_clock::duration source_playback_paused_duration{};
        std::chrono::steady_clock::time_point previous_frame{std::chrono::steady_clock::now()};
        double previous_video_shader_timeline = 0.0;
        bool video_shader_timeline_initialized = false;
        bool video_shader_clock_logged = false;
        bool model_video_timeline_initialized = false;
        bool crossfade_uses_video_timeline = false;
        bool autopilot_video_timeline_initialized = false;
        std::mt19937 autopilot_rng{std::random_device{}()};
#ifdef ACMXVK_WITH_CUDA
        std::unique_ptr<gpu::FilterEngine> gpu_filter_engine;
#endif
#ifdef ACMXVK_WITH_DNN
        std::unique_ptr<dnn::EdgeDetector> edge_detector;
        std::unique_ptr<dnn::HumanSegmenter> human_segmenter;
        std::unique_ptr<dnn::GenericOnnxProcessor> generic_onnx_processor;
#endif
#ifdef ACMXVK_WITH_MXVK_CUDA
        cv::cuda::GpuMat cuda_input_rgba;
        cv::cuda::GpuMat cuda_rotated_rgba;
        cv::cuda::GpuMat cuda_rotation_transpose;
        cv::Mat cuda_input_fallback_rgba;
        cv::Mat cuda_history_fallback_rgba;
        cv::Mat cuda_model_fallback_rgba;
        bool cuda_input_path_logged = false;
        bool cuda_input_fallback_logged = false;
        bool cuda_history_fallback_logged = false;
        bool cuda_model_fallback_logged = false;
#endif
#ifdef MIDI_ENABLED
        struct MidiCcMapping {
            int channel = -1;
            int controller = 0;
            std::size_t uniform_index = 0;
            std::string uniform_name;
        };

        struct MidiKnobState {
            int value = 64;
            int previous_value = 64;
            int direction_action = 0;
            int frame_counter = 0;
            bool active = false;
        };

        std::unique_ptr<midi::MidiInput> midi_input;
        std::vector<midi::MidiMapping> midi_action_mappings;
        std::vector<MidiKnobState> midi_knob_states;
        std::vector<MidiCcMapping> midi_cc_mappings;
        std::array<int, 4> midi_slider_uniform_indices{-1, -1, -1, -1};
        std::uint64_t observed_midi_drops = 0;
#endif
#ifdef AUDIO_ENABLED
        std::unique_ptr<audio::AudioEngine> audio_engine;
        std::unique_ptr<audio::FileAudioSource> file_audio_source;
        float audio_warmup_envelope = 0.0F;
        bool audio_warmup_started = false;
        std::chrono::steady_clock::time_point audio_warmup_last_tick{};

        void resetAudioWarmup() {
            audio_warmup_envelope = 0.0F;
            audio_warmup_started = false;
            if (options.audio_warm_rate <= 0.0) {
                std::cout << "acmxvk: audio shader warmup disabled\n";
            } else {
                std::cout << "acmxvk: audio shader warmup "
                          << options.audio_warm_rate << "/second (~"
                          << 1.0 / options.audio_warm_rate
                          << " seconds to full strength)\n";
            }
        }

        [[nodiscard]] float updateAudioWarmup(
            std::chrono::steady_clock::time_point now) {
            if (options.audio_warm_rate <= 0.0) {
                audio_warmup_envelope = 1.0F;
                return audio_warmup_envelope;
            }
            if (!audio_warmup_started) {
                audio_warmup_started = true;
                audio_warmup_last_tick = now;
                return audio_warmup_envelope;
            }

            const float delta = std::max(
                std::chrono::duration<float>(now - audio_warmup_last_tick).count(),
                0.0F);
            audio_warmup_last_tick = now;
            audio_warmup_envelope = std::min(
                audio_warmup_envelope +
                    delta * static_cast<float>(options.audio_warm_rate),
                1.0F);
            return audio_warmup_envelope;
        }
#endif

        void initializeDnn() {
#ifdef ACMXVK_WITH_DNN
            if (!options.human_model.empty()) {
                human_segmenter =
                    std::make_unique<dnn::HumanSegmenter>(options.human_model);
                std::cout << "acmxvk: PP-HumanSeg enabled: "
                          << options.human_model << " ("
                          << (options.human_background
                                  ? "background-only shader composition"
                                  : "foreground isolation")
                          << ", automatic CPU/CUDA backend selection)\n";
            }
            if (!options.edge_model.empty()) {
                edge_detector =
                    std::make_unique<dnn::EdgeDetector>(options.edge_model);
                std::cout << "acmxvk: DexiNed edge detection enabled: "
                          << options.edge_model
                          << " (automatic CPU/CUDA backend selection)\n";
            }
            if (!options.onnx_configuration.empty()) {
                generic_onnx_processor =
                    std::make_unique<dnn::GenericOnnxProcessor>(
                        options.onnx_configuration);
                std::cout << "acmxvk: generic ONNX processing enabled: "
                          << options.onnx_configuration
                          << " (automatic CPU/CUDA backend selection)\n";
            }
#endif
        }

        void initializeGpuFilters() {
#ifdef ACMXVK_WITH_CUDA
            if (options.gpu_filter_indices.empty()) {
                return;
            }
            gpu_filter_engine = std::make_unique<gpu::FilterEngine>(
                options.gpu_filter_indices, options.gpu_frame_buffer_size);
#endif
        }

        void selectGpuFilter(int direction) {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr &&
                gpu_filter_engine->select_relative_filter(direction) &&
                source_kind == SourceKind::Graphic && !graphic_rgba.empty()) {
                uploadInputFrame(graphic_rgba);
                if (history_initialized) {
                    updateHistoryFrame(graphic_rgba);
                    history_delay_counter = 0;
                }
            }
#else
            static_cast<void>(direction);
#endif
        }

        void openMidi() {
#ifdef MIDI_ENABLED
            if (!options.midi_device_specified && !options.midi_monitor &&
                options.midi_map_file.empty() && midi_cc_mappings.empty()) {
                return;
            }
            midi_input = std::make_unique<midi::MidiInput>();
            const int port = options.midi_device_specified ? options.midi_device : 0;
            if (!midi_input->open(port)) {
                throw std::runtime_error("could not open MIDI input port " +
                                         std::to_string(port));
            }
#endif
        }

        void configureMidiMappings() {
#ifdef MIDI_ENABLED
            if (!options.midi_map_file.empty()) {
                midi_action_mappings =
                    midi::load_mapping_file(options.midi_map_file);
                midi_knob_states.resize(midi_action_mappings.size());
                std::cout << "acmxvk: loaded " << midi_action_mappings.size()
                          << " MIDI mapping(s) from " << options.midi_map_file
                          << '\n';

                for (int slider = 0; slider < 4; ++slider) {
                    const int action = 600 + slider * 2;
                    const bool mapped = std::any_of(
                        midi_action_mappings.begin(),
                        midi_action_mappings.end(),
                        [&](const midi::MidiMapping &mapping) {
                            return mapping.primary_action == action &&
                                   mapping.secondary_action == action + 1;
                        });
                    if (!mapped) {
                        continue;
                    }
                    const std::string name =
                        "slider" + std::to_string(slider + 1);
                    const auto uniform = std::find_if(
                        custom_uniforms.begin(), custom_uniforms.end(),
                        [&](const ShaderManifest::CustomUniform &candidate) {
                            return candidate.name == name;
                        });
                    if (uniform == custom_uniforms.end()) {
                        std::cerr << "acmxvk: MIDI " << name
                                  << " mapping has no matching custom uniform in "
                                     "library.json\n";
                        continue;
                    }
                    midi_slider_uniform_indices[slider] = static_cast<int>(
                        std::distance(custom_uniforms.begin(), uniform));
                    std::cout << "acmxvk: MIDI Slider " << (slider + 1)
                              << " -> " << name << " [" << uniform->minimum
                              << ", " << uniform->maximum << "]\n";
                }

                std::size_t active_mappings = 0;
                for (const midi::MidiMapping &mapping :
                     midi_action_mappings) {
                    if (isMidiMappingSupported(mapping)) {
                        ++active_mappings;
                    } else if (options.midi_monitor) {
                        std::cerr
                            << "acmxvk: MIDI map action unavailable in this build: "
                            << mapping.primary_action << ':'
                            << mapping.secondary_action << '\n';
                    }
                }
                std::cout << "acmxvk: MIDI map has " << active_mappings
                          << " active mapping(s)";
                if (active_mappings != midi_action_mappings.size()) {
                    std::cout << " and "
                              << (midi_action_mappings.size() - active_mappings)
                              << " mapping(s) reserved for unported ACMX2 controls";
                }
                std::cout << '\n';
            }

            for (const std::string &mapping_text : options.midi_cc_mappings) {
                const std::size_t equals = mapping_text.find('=');
                if (equals == std::string::npos || equals == 0 ||
                    equals + 1 >= mapping_text.size() ||
                    mapping_text.find('=', equals + 1) != std::string::npos) {
                    throw std::runtime_error(
                        "--midi-cc requires [channel:]CC=uniform: " +
                        mapping_text);
                }

                const std::string source = trim(mapping_text.substr(0, equals));
                const std::string uniform_name =
                    trim(mapping_text.substr(equals + 1));
                if (!isValidCustomUniformName(uniform_name)) {
                    throw std::runtime_error(
                        "--midi-cc contains an invalid uniform name: " +
                        uniform_name);
                }

                int channel = -1;
                int controller = 0;
                const std::size_t colon = source.find(':');
                if (colon == std::string::npos) {
                    controller = parseInteger(source, "--midi-cc");
                } else {
                    if (colon == 0 || colon + 1 >= source.size() ||
                        source.find(':', colon + 1) != std::string::npos) {
                        throw std::runtime_error(
                            "--midi-cc requires [channel:]CC=uniform: " +
                            mapping_text);
                    }
                    channel = parseInteger(
                        std::string_view(source).substr(0, colon), "--midi-cc");
                    controller = parseInteger(
                        std::string_view(source).substr(colon + 1), "--midi-cc");
                    if (channel < 1 || channel > 16) {
                        throw std::runtime_error(
                            "--midi-cc channel must be between 1 and 16");
                    }
                    --channel;
                }
                if (controller < 0 || controller > 127) {
                    throw std::runtime_error(
                        "--midi-cc controller must be between 0 and 127");
                }

                const auto uniform = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &candidate) {
                        return candidate.name == uniform_name;
                    });
                if (uniform == custom_uniforms.end()) {
                    throw std::runtime_error(
                        "--midi-cc target is not defined in library.json: " +
                        uniform_name);
                }
                const std::size_t uniform_index = static_cast<std::size_t>(
                    std::distance(custom_uniforms.begin(), uniform));
                const auto duplicate = std::find_if(
                    midi_cc_mappings.begin(), midi_cc_mappings.end(),
                    [&](const MidiCcMapping &mapping) {
                        return mapping.uniform_index == uniform_index;
                    });
                if (duplicate != midi_cc_mappings.end()) {
                    throw std::runtime_error(
                        "custom uniform has more than one --midi-cc mapping: " +
                        uniform_name);
                }

                midi_cc_mappings.push_back(
                    {channel, controller, uniform_index, uniform_name});
                std::cout << "acmxvk: MIDI "
                          << (channel < 0
                                  ? std::string("any channel")
                                  : "channel " + std::to_string(channel + 1))
                          << " CC " << controller << " -> " << uniform_name
                          << " [" << uniform->minimum << ", "
                          << uniform->maximum << "]\n";
            }
#endif
        }

#ifdef MIDI_ENABLED
        [[nodiscard]] bool applyMidiCc(const midi::MidiMessage &message) {
            if (message.bytes.size() < 3 ||
                (message.bytes[0] & 0xF0U) != 0xB0U) {
                return false;
            }
            const int channel = message.bytes[0] & 0x0FU;
            const int controller = message.bytes[1] & 0x7FU;
            const int value = message.bytes[2] & 0x7FU;
            bool changed = false;
            for (const MidiCcMapping &mapping : midi_cc_mappings) {
                if (mapping.controller != controller ||
                    (mapping.channel >= 0 && mapping.channel != channel)) {
                    continue;
                }
                const ShaderManifest::CustomUniform &uniform =
                    custom_uniforms[mapping.uniform_index];
                const double normalized = static_cast<double>(value) / 127.0;
                const float mapped = static_cast<float>(
                    uniform.minimum + normalized *
                                          (uniform.maximum - uniform.minimum));
                custom_uniform_values[mapping.uniform_index] = mapped;
                changed = true;
                if (options.midi_monitor) {
                    std::cout << "acmxvk: MIDI CC " << controller << " -> "
                              << mapping.uniform_name << '=' << mapped << '\n';
                }
            }
            return changed;
        }

        [[nodiscard]] SDL_Keycode midiActionKey(int action) const {
            switch (action) {
            case 262:
#ifdef ACMXVK_WITH_CUDA
                return gpu_filter_engine != nullptr ? SDLK_RIGHT : SDLK_UNKNOWN;
#else
                return SDLK_UNKNOWN;
#endif
            case 263:
#ifdef ACMXVK_WITH_CUDA
                return gpu_filter_engine != nullptr ? SDLK_LEFT : SDLK_UNKNOWN;
#else
                return SDLK_UNKNOWN;
#endif
            case 264:
                return SDLK_DOWN;
            case 265:
                return SDLK_UP;
            case 266:
            case 504:
                return SDLK_PAGEUP;
            case 267:
            case 505:
                return SDLK_PAGEDOWN;
            case 268:
#ifdef AUDIO_ENABLED
                return SDLK_HOME;
#else
                return SDLK_UNKNOWN;
#endif
            case 269:
#ifdef AUDIO_ENABLED
                return SDLK_END;
#else
                return SDLK_UNKNOWN;
#endif
            case 260:
                return SDLK_INSERT;
            case 261:
                return SDLK_DELETE;
            case 500:
                return SDLK_U;
            case 501:
                return SDLK_I;
            case 298:
                return SDLK_F9;
            case 32:
                return SDLK_SPACE;
            case 44:
                return options.enable_3d ? SDLK_COMMA : SDLK_UNKNOWN;
            case 46:
                return options.enable_3d ? SDLK_PERIOD : SDLK_UNKNOWN;
            case 51:
                return options.enable_3d ? SDLK_3 : SDLK_UNKNOWN;
            case 52:
                return SDLK_4;
            case 53:
                return SDLK_5;
            case 54:
                return SDLK_6;
            case 67:
                return options.enable_3d ? SDLK_C : SDLK_UNKNOWN;
            case 79:
                return options.enable_3d ? SDLK_O : SDLK_UNKNOWN;
            case 91:
                return options.enable_3d ? SDLK_MINUS : SDLK_UNKNOWN;
            case 93:
                return options.enable_3d ? SDLK_EQUALS : SDLK_UNKNOWN;
            case 69:
                return options.watermark_text.empty() ? SDLK_UNKNOWN : SDLK_E;
            case 74:
                return SDLK_J;
            case 78:
                return SDLK_N;
            case 75:
                return SDLK_K;
            case 76:
                return SDLK_L;
            case 77:
                return SDLK_M;
            case 80:
                return SDLK_P;
            case 81:
#ifdef AUDIO_ENABLED
                return SDLK_Q;
#else
                return SDLK_UNKNOWN;
#endif
            case 70:
                return SDLK_F;
            case 73:
                return SDLK_I;
            case 84:
                return SDLK_T;
            case 85:
                return SDLK_U;
            case 86:
                return options.enable_3d ? SDLK_V : SDLK_UNKNOWN;
            case 88:
                return options.enable_3d ? SDLK_X : SDLK_UNKNOWN;
            case 89:
                return SDLK_Y;
            case 90:
                return SDLK_Z;
            default:
                return SDLK_UNKNOWN;
            }
        }

        [[nodiscard]] bool isMidiSliderMapping(
            const midi::MidiMapping &mapping) const {
            return mapping.primary_action >= 600 &&
                   mapping.primary_action <= 606 &&
                   mapping.primary_action % 2 == 0 &&
                   mapping.secondary_action == mapping.primary_action + 1;
        }

        [[nodiscard]] static bool usesMidiDeltaDirection(
            const midi::MidiMapping &mapping) {
            return mapping.primary_action == 506 ||
                   mapping.primary_action == 508 ||
                   mapping.primary_action == 512;
        }

        [[nodiscard]] bool isMidiModelAction(int action) const {
            return options.enable_3d && action >= 506 && action <= 515;
        }

        [[nodiscard]] bool isMidiMappingSupported(
            const midi::MidiMapping &mapping) const {
            if (isMidiSliderMapping(mapping)) {
                const int slider = (mapping.primary_action - 600) / 2;
                return midi_slider_uniform_indices[slider] >= 0;
            }
            if (mapping.secondary_action == 0) {
                return isMidiModelAction(mapping.primary_action) ||
                       midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
            }
            const bool primary_supported =
                isMidiModelAction(mapping.primary_action) ||
                midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
            const bool secondary_supported =
                isMidiModelAction(mapping.secondary_action) ||
                midiActionKey(mapping.secondary_action) != SDLK_UNKNOWN;
            return primary_supported && secondary_supported;
        }

        [[nodiscard]] std::string_view midiActionName(int action) const {
            switch (action) {
            case 262:
                return "select next CUDA filter";
            case 263:
                return "select previous CUDA filter";
            case 264:
                return "next shader or playlist node";
            case 265:
                return "previous shader or playlist node";
            case 266:
            case 504:
                return "increase shader time speed";
            case 267:
            case 505:
                return "decrease shader time speed";
            case 268:
                return "toggle audio delta-time scaling";
            case 269:
                return "toggle spectrum sensitivity scaling";
            case 260:
                return "increase audio sensitivity";
            case 261:
                return "decrease audio sensitivity";
            case 500:
                return "step shader time forward";
            case 501:
                return "step shader time backward";
            case 506:
                return "rotate model X forward";
            case 507:
                return "rotate model X backward";
            case 508:
                return "rotate model Y forward";
            case 509:
                return "rotate model Y backward";
            case 510:
                return "increase 3D manual rotation speed";
            case 511:
                return "decrease 3D manual rotation speed";
            case 512:
                return "rotate model Z forward";
            case 513:
                return "rotate model Z backward";
            case 514:
                return "increase model scale";
            case 515:
                return "decrease model scale";
            case 298:
                return "toggle runtime HUD";
            case 32:
                return "toggle shader bypass";
            case 44:
                return "decrease 3D view rotation speed";
            case 46:
                return "increase 3D view rotation speed";
            case 51:
                return "toggle 2D/3D rendering";
            case 52:
                return "take TIFF snapshot";
            case 53:
                return "take WebP snapshot";
            case 54:
                return "take raw RGBA snapshot";
            case 67:
                return "toggle 3D wave effect";
            case 79:
                return "toggle 3D scale oscillation";
            case 91:
                return "decrease model scale";
            case 93:
                return "increase model scale";
            case 69:
                return "toggle watermark";
            case 74:
                return "toggle random autopilot";
            case 78:
                return "toggle random autopilot crossfade";
            case 75:
                return "toggle shader lock";
            case 76:
                return "toggle rendering freeze";
            case 77:
                return "toggle multipass";
            case 80:
                return "toggle playlist or input pause";
            case 81:
                return "toggle audio-reactive shader time";
            case 70:
                return "toggle fullscreen";
            case 73:
                return "step shader time backward";
            case 84:
                return "toggle shader time";
            case 85:
                return "step shader time forward";
            case 86:
                return "toggle 3D view rotation";
            case 88:
                return "reset model view";
            case 89:
                return "toggle sequential autopilot";
            case 90:
                return "take screenshot";
            default:
                return "unsupported action";
            }
        }

        void dispatchMidiModelAction(int action) {
            if (!model_initialized || !isMidiModelAction(action)) {
                return;
            }

            const auto rotate = [](float &degrees, float amount) {
                degrees = std::fmod(degrees + amount, 360.0F);
                if (degrees < 0.0F) {
                    degrees += 360.0F;
                }
            };
            switch (action) {
            case 506:
                rotate(model_rotation_x_degrees,
                       model_camera_rotation_speed * 0.3F);
                break;
            case 507:
                rotate(model_rotation_x_degrees,
                       model_camera_rotation_speed * -0.33F);
                break;
            case 508:
                rotate(model_rotation_y_degrees,
                       model_camera_rotation_speed * 0.3F);
                break;
            case 509:
                rotate(model_rotation_y_degrees,
                       model_camera_rotation_speed * -0.3F);
                break;
            case 510:
                model_camera_rotation_speed = std::clamp(
                    model_camera_rotation_speed + 0.5F, 0.5F, 50.0F);
                std::cout << "acmxvk: 3D manual rotation speed "
                          << model_camera_rotation_speed << '\n';
                break;
            case 511:
                model_camera_rotation_speed = std::clamp(
                    model_camera_rotation_speed - 0.5F, 0.5F, 50.0F);
                std::cout << "acmxvk: 3D manual rotation speed "
                          << model_camera_rotation_speed << '\n';
                break;
            case 512:
                rotate(model_rotation_z_degrees,
                       model_camera_rotation_speed * 0.3F);
                break;
            case 513:
                rotate(model_rotation_z_degrees,
                       model_camera_rotation_speed * -0.3F);
                break;
            case 514:
                adjustModelScale(0.05F);
                break;
            case 515:
                adjustModelScale(-0.05F);
                break;
            default:
                break;
            }
        }

        void dispatchMidiAction(int action) {
            if (isMidiModelAction(action)) {
                if (options.midi_monitor) {
                    std::cout << "acmxvk: MIDI action: "
                              << midiActionName(action) << '\n';
                }
                dispatchMidiModelAction(action);
                return;
            }
            const SDL_Keycode key = midiActionKey(action);
            if (key == SDLK_UNKNOWN) {
                return;
            }
            if (options.midi_monitor) {
                std::cout << "acmxvk: MIDI action: " << midiActionName(action)
                          << '\n';
            }
            SDL_Event midi_event{};
            midi_event.type = SDL_EVENT_KEY_DOWN;
            midi_event.key.type = SDL_EVENT_KEY_DOWN;
            midi_event.key.key = key;
            midi_event.key.mod =
                action == 91 || action == 93 ? SDL_KMOD_SHIFT
                                             : SDL_KMOD_NONE;
            midi_event.key.repeat = false;
            event(midi_event);
        }

        [[nodiscard]] bool setMidiUniform(std::size_t uniform_index, int value,
                                          std::string_view label) {
            if (uniform_index >= custom_uniforms.size() ||
                uniform_index >= custom_uniform_values.size()) {
                return false;
            }
            const ShaderManifest::CustomUniform &uniform =
                custom_uniforms[uniform_index];
            const double normalized = static_cast<double>(value) / 127.0;
            const float mapped = static_cast<float>(
                uniform.minimum +
                normalized * (uniform.maximum - uniform.minimum));
            custom_uniform_values[uniform_index] = mapped;
            if (options.midi_monitor) {
                std::cout << "acmxvk: MIDI " << label << " -> "
                          << uniform.name << '=' << mapped << '\n';
            }
            return true;
        }

        [[nodiscard]] bool applyMidiMap(const midi::MidiMessage &message) {
            if (message.bytes.size() < 3) {
                return false;
            }
            bool changed = false;
            for (std::size_t index = 0; index < midi_action_mappings.size();
                 ++index) {
                const midi::MidiMapping &mapping = midi_action_mappings[index];
                if (message.bytes[0] != mapping.status ||
                    message.bytes[1] != mapping.data1) {
                    continue;
                }
                const int value = message.bytes[2] & 0x7FU;
                if (mapping.secondary_action == 0) {
                    if (message.bytes[2] == mapping.data2) {
                        dispatchMidiAction(mapping.primary_action);
                    }
                    continue;
                }

                if (isMidiSliderMapping(mapping)) {
                    const int slider = (mapping.primary_action - 600) / 2;
                    const int uniform_index =
                        midi_slider_uniform_indices[slider];
                    if (uniform_index >= 0) {
                        changed =
                            setMidiUniform(
                                static_cast<std::size_t>(uniform_index), value,
                                "Slider " + std::to_string(slider + 1)) ||
                            changed;
                    }
                    continue;
                }

                MidiKnobState &state = midi_knob_states[index];
                if (usesMidiDeltaDirection(mapping) &&
                    value != state.previous_value) {
                    state.direction_action =
                        value > state.previous_value
                            ? mapping.primary_action
                            : mapping.secondary_action;
                }
                state.previous_value = value;
                state.value = value;
                state.active = value != 64;
                if (!state.active) {
                    state.frame_counter = 0;
                }
            }
            return changed;
        }

        void dispatchMidiKnobs() {
            for (std::size_t index = 0; index < midi_action_mappings.size();
                 ++index) {
                const midi::MidiMapping &mapping = midi_action_mappings[index];
                MidiKnobState &state = midi_knob_states[index];
                if (!state.active || mapping.secondary_action == 0 ||
                    isMidiSliderMapping(mapping) ||
                    !isMidiMappingSupported(mapping)) {
                    continue;
                }

                const int distance = std::abs(state.value - 64);
                const int frame_skip =
                    std::max(1, 17 - (distance * 16 / 63));
                if (++state.frame_counter < frame_skip) {
                    continue;
                }
                state.frame_counter = 0;
                int action = state.value > 64
                                 ? mapping.primary_action
                                 : mapping.secondary_action;
                if (usesMidiDeltaDirection(mapping)) {
                    action = state.direction_action;
                    if (action == 0) {
                        continue;
                    }
                }
                dispatchMidiAction(action);
            }
        }
#endif

        void uploadCustomUniforms() {
            if (frame_sprite != nullptr) {
                frame_sprite->setCustomUniforms(custom_uniform_values);
            }
            for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                sprite->setCustomUniforms(custom_uniform_values);
            }
        }

        void pollMidi() {
#ifdef MIDI_ENABLED
            if (midi_input == nullptr || !midi_input->is_open()) {
                return;
            }
            const std::vector<midi::MidiMessage> messages =
                midi_input->poll_messages();
            bool custom_uniforms_changed = false;
            for (const midi::MidiMessage &message : messages) {
                custom_uniforms_changed =
                    applyMidiCc(message) || custom_uniforms_changed;
                custom_uniforms_changed =
                    applyMidiMap(message) || custom_uniforms_changed;
            }
            dispatchMidiKnobs();
            if (custom_uniforms_changed) {
                uploadCustomUniforms();
            }
            if (options.midi_monitor) {
                for (const midi::MidiMessage &message : messages) {
                    std::ostringstream text;
                    text << "acmxvk: MIDI #" << message.sequence << " +"
                         << std::fixed << std::setprecision(6)
                         << message.delta_seconds << "s [";
                    for (std::size_t index = 0; index < message.bytes.size();
                         ++index) {
                        if (index > 0) {
                            text << ' ';
                        }
                        text << std::hex << std::uppercase << std::setfill('0')
                             << std::setw(2)
                             << static_cast<unsigned int>(message.bytes[index]);
                    }
                    text << ']';
                    std::cout << text.str() << '\n';
                }
            }
            const std::uint64_t dropped = midi_input->dropped_message_count();
            if (dropped != observed_midi_drops) {
                std::cerr << "acmxvk: MIDI queue dropped " << dropped
                          << " message(s) total\n";
                observed_midi_drops = dropped;
            }
#endif
        }

        void openAudio() {
            if (!options.enable_audio) {
                return;
            }
#ifdef AUDIO_ENABLED
            audio_engine = std::make_unique<audio::AudioEngine>();
            audio_engine->set_sensitivity(
                static_cast<float>(options.audio_sensitivity));
            if (!options.audio_file.empty()) {
                file_audio_source = std::make_unique<audio::FileAudioSource>();
                if (!file_audio_source->open(options.audio_file)) {
                    if (options.use_source_audio) {
                        std::cerr
                            << "acmxvk: source video has no decodable audio "
                               "track; continuing with silent audio-reactive "
                               "values";
                        if (options.audio_pass_through) {
                            std::cerr << " and pass-through disabled";
                        }
                        std::cerr << '\n';
                        file_audio_source.reset();
                        return;
                    }
                    throw std::runtime_error("could not decode --audio-file: " +
                                             options.audio_file);
                }
                if (options.use_source_audio) {
                    std::cout << "acmxvk: source video audio drives shader "
                                 "reactivity\n";
                }
                file_audio_source->set_repeat(options.audio_repeat);
                if (options.audio_pass_through &&
                    !file_audio_source->enable_output(
                        options.audio_output_device,
                        static_cast<float>(options.audio_pass_through_gain))) {
                    std::cerr << "acmxvk: file audio output could not be "
                                 "initialized; continuing with silent analysis\n";
                }
                resetAudioWarmup();
                return;
            }
            const audio::AudioStreamConfig config{
                static_cast<unsigned int>(options.audio_channels),
                static_cast<float>(options.audio_sensitivity),
                options.audio_input_device,
                options.audio_output_device,
                options.audio_pass_through,
                static_cast<float>(options.audio_pass_through_gain),
                static_cast<float>(options.audio_recording_gain),
            };
            if (!audio_engine->open(config)) {
                std::cerr << "acmxvk: audio input could not be initialized; "
                             "continuing with zero-valued audio metrics\n";
                audio_engine.reset();
            } else {
                resetAudioWarmup();
            }
#endif
        }

        void start_requested_audio_recording() {
            if (options.record_audio_file.empty()) {
                return;
            }
#ifdef AUDIO_ENABLED
            if (audio_engine == nullptr || file_audio_source != nullptr ||
                !audio_engine->is_open()) {
                throw std::runtime_error(
                    "--record-audio requires an active live audio input");
            }
            if (!options.output_file.empty() && !options.png_output) {
                return;
            }
            if (!audio_engine->is_recording() &&
                !audio_engine->start_recording()) {
                throw std::runtime_error(
                    "could not start standalone microphone recording");
            }
#endif
        }

        void adjustAudioSensitivity(float amount) {
#ifdef AUDIO_ENABLED
            if (audioSourceOpen()) {
                audio_engine->set_sensitivity(audio_engine->sensitivity() + amount);
                options.audio_sensitivity = audio_engine->sensitivity();
                std::cout << "acmxvk: audio sensitivity "
                          << options.audio_sensitivity << '\n';
                return;
            }
#else
            static_cast<void>(amount);
#endif
            std::cout << "acmxvk: audio input is not active\n";
        }

        [[nodiscard]] bool audioSourceOpen() const {
#ifdef AUDIO_ENABLED
            return audio_engine != nullptr &&
                   (audio_engine->is_open() ||
                    (file_audio_source != nullptr && file_audio_source->is_open()));
#else
            return false;
#endif
        }

        void startLiveAudioRecordingIfNeeded() {
#ifdef AUDIO_ENABLED
            if (audio_engine == nullptr || file_audio_source != nullptr ||
                !audio_engine->is_open() || audio_engine->is_recording() ||
                !writer.is_open() || options.png_output ||
                (options.copy_audio && !options.mute_output &&
                 options.record_audio_file.empty())) {
                return;
            }
            if (!audio_engine->start_recording()) {
                std::cerr << "acmxvk: could not start live audio recording; "
                             "continuing with video-only output\n";
            }
#endif
        }

        void startMediaTimelineIfReady() {
            if (media_timeline_started || !source_frame_received) {
                return;
            }
            media_timeline_started = true;
            hud_session_start = std::chrono::steady_clock::now();
            hud_fps_last_tick = hud_session_start;
            hud_fps_frame_count = 0;
            source_playback_clock_start = hud_session_start;
            source_playback_pause_start = {};
            source_playback_paused_duration = {};
            source_playback_clock_paused = false;
#ifdef AUDIO_ENABLED
            resetAudioWarmup();
#endif
            startLiveAudioRecordingIfNeeded();
            std::cout << "acmxvk: media timeline started on first source frame\n";
        }

        void setSourcePlaybackClockPaused(bool paused) {
            if (!options.use_source_fps || source_kind != SourceKind::Video ||
                !media_timeline_started ||
                paused == source_playback_clock_paused) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (paused) {
                source_playback_pause_start = now;
            } else {
                source_playback_paused_duration +=
                    now - source_playback_pause_start;
            }
            source_playback_clock_paused = paused;
        }

        [[nodiscard]] bool mediaClockSeconds(double &seconds) const {
#ifdef AUDIO_ENABLED
            if (file_audio_source != nullptr &&
                file_audio_source->has_output_clock()) {
                seconds = file_audio_source->playback_time();
                return true;
            }
            if ((!options.copy_audio || options.mute_output) && writer.is_open() &&
                audio_engine != nullptr && file_audio_source == nullptr &&
                audio_engine->is_recording()) {
                seconds = audio_engine->recording_time();
                return true;
            }
#endif
            if (source_kind == SourceKind::Camera &&
                media_timeline_started) {
                seconds = hudWallElapsedSeconds();
                return true;
            }
            if (options.use_source_fps && source_kind == SourceKind::Video &&
                media_timeline_started) {
                const auto clock_end = source_playback_clock_paused
                                           ? source_playback_pause_start
                                           : std::chrono::steady_clock::now();
                const auto active_time =
                    clock_end - source_playback_clock_start -
                    source_playback_paused_duration;
                seconds = std::max(
                    0.0, std::chrono::duration<double>(active_time).count());
                return true;
            }
            seconds = 0.0;
            return false;
        }

        void loadShaders() {
            if (!options.fragment_shader.empty() ||
                !options.compute_shader.empty()) {
                const bool compute = !options.compute_shader.empty();
                const fs::path shader = fs::absolute(
                                            compute ? options.compute_shader : options.fragment_shader)
                                            .lexically_normal();
                const std::string label =
                    compute ? "compute shader" : "fragment shader";
                if (shader.extension() != ".spv" ||
                    !fs::is_regular_file(shader)) {
                    throw std::runtime_error(
                        label + " is not a readable .spv file: " +
                        shader.string());
                }
                input::validate_spirv_file(shader, label);
                const mxvk::ShaderModuleInfo module_info =
                    mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
                const mxvk::ShaderStage expected_stage =
                    compute ? mxvk::ShaderStage::Compute
                            : mxvk::ShaderStage::Fragment;
                if (module_info.stage != expected_stage) {
                    throw std::runtime_error(
                        label + " SPIR-V entry point has the wrong shader stage: " +
                        shader.string());
                }
                recordShaderResources(module_info, "shader");
                shaders.push_back(shader);
                return;
            }
            if (options.shader_directory.empty()) {
                return;
            }

            shader_library_directory =
                fs::absolute(options.shader_directory).lexically_normal();
            const ShaderManifest manifest =
                loadShaderManifest(shader_library_directory);
            shader_manifest_path = manifest.path;
            custom_uniforms = manifest.custom_uniforms;
            applyCustomUniformOverrides();
            for (const std::string &entry : manifest.entries) {
                const fs::path shader =
                    resolveShaderManifestEntry(shader_library_directory, entry);
                if (!shader.empty()) {
                    input::validate_spirv_file(shader,
                                               "shader manifest entry");
                    const mxvk::ShaderModuleInfo module_info =
                        mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
                    recordShaderResources(module_info, "shader library");
                    shaders.push_back(shader);
                }
            }
            std::sort(shaders.begin(), shaders.end(), [](const fs::path &left, const fs::path &right) {
                std::string left_text = left.generic_string();
                std::string right_text = right.generic_string();
                std::transform(left_text.begin(), left_text.end(), left_text.begin(),
                               [](unsigned char character) {
                                   return static_cast<char>(std::tolower(character));
                               });
                std::transform(right_text.begin(), right_text.end(), right_text.begin(),
                               [](unsigned char character) {
                                   return static_cast<char>(std::tolower(character));
                               });
                return left_text < right_text;
            });
            if (shaders.empty()) {
                throw std::runtime_error("shader manifest contains no readable SPIR-V files: " +
                                         shader_manifest_path.string());
            }
            std::cout << "acmxvk: loaded " << shaders.size() << " shaders from "
                      << shader_manifest_path.string() << '\n';
            printCustomUniforms();

            if (!options.shader_file.empty()) {
                const auto selected = std::find_if(
                    shaders.begin(), shaders.end(), [&](const fs::path &path) {
                        fs::path requested(options.shader_file);
                        if (requested.extension() != ".spv") {
                            requested.replace_extension(".spv");
                        }
                        return path.filename() == requested.filename() ||
                               path.lexically_relative(shader_library_directory) == requested;
                    });
                if (selected == shaders.end()) {
                    throw std::runtime_error("shader file is not listed in the manifest: " +
                                             options.shader_file);
                }
                shader_index = static_cast<std::size_t>(std::distance(shaders.begin(), selected));
            } else {
                const int count = static_cast<int>(shaders.size());
                const int wrapped_index = ((options.shader_index % count) + count) % count;
                shader_index = static_cast<std::size_t>(wrapped_index);
            }
        }

        void applyCustomUniformOverrides() {
            for (const std::string &override_text :
                 options.custom_uniform_overrides) {
                const std::size_t separator = override_text.find('=');
                if (separator == std::string::npos || separator == 0 ||
                    separator + 1 >= override_text.size()) {
                    throw std::runtime_error(
                        "--uniform requires name=value: " + override_text);
                }
                const std::string name = trim(override_text.substr(0, separator));
                const double value = parseNumber(
                    trim(override_text.substr(separator + 1)), "--uniform");
                const auto match = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &uniform) {
                        return uniform.name == name;
                    });
                if (match == custom_uniforms.end()) {
                    throw std::runtime_error(
                        "custom uniform is not defined in library.json: " + name);
                }
                match->value = std::clamp(value, match->minimum, match->maximum);
            }

            custom_uniform_values.clear();
            custom_uniform_values.reserve(custom_uniforms.size());
            for (const ShaderManifest::CustomUniform &uniform : custom_uniforms) {
                custom_uniform_values.push_back(static_cast<float>(uniform.value));
            }
        }

        void printCustomUniforms() const {
            if (custom_uniforms.empty()) {
                return;
            }
            constexpr std::string_view COMPONENTS = "xyzw";
            std::cout << "acmxvk: custom uniforms (binding 1):\n";
            for (std::size_t index = 0; index < custom_uniforms.size(); ++index) {
                const ShaderManifest::CustomUniform &uniform = custom_uniforms[index];
                std::cout << "  " << uniform.name << '=' << uniform.value
                          << " -> custom_uniforms[" << (index / 4) << "]."
                          << COMPONENTS[index % 4] << '\n';
            }
        }

        [[nodiscard]] std::string currentShader() const {
            return shaders.empty() ? std::string{} : shaders[shader_index].string();
        }

        [[nodiscard]] bool historyCacheEnabled() const {
            return options.enable_texture_cache || shader_history_required;
        }

        void recordShaderResources(const mxvk::ShaderModuleInfo &module_info,
                                   std::string_view source) {
            if (module_info.usesHistoryTexture &&
                !shader_history_required) {
                shader_history_required = true;
                std::cout << "acmxvk: enabled shared history for " << source
                          << " binding 2\n";
            }
            if (module_info.usesSpectrumTexture &&
                !shader_spectrum_required) {
                shader_spectrum_required = true;
                std::cout << "acmxvk: enabled spectrum descriptor for " << source
                          << " binding 3\n";
            }
            if (module_info.usesSpectrumHistoryTexture &&
                !shader_spectrum_history_required) {
                shader_spectrum_history_required = true;
                if (options.audio_buffers == 0) {
                    options.audio_buffers = 8;
                }
                std::cout << "acmxvk: enabled " << options.audio_buffers
                          << " spectrum-history layers for " << source
                          << " binding 4\n";
            }
        }

        [[nodiscard]] std::uint32_t spectrumBinCount() const {
#ifdef AUDIO_ENABLED
            return audio::AudioEngine::spectrum_bin_count();
#else
            return COMPATIBILITY_SPECTRUM_BIN_COUNT;
#endif
        }

        [[nodiscard]] bool spectrumTextureEnabledForShaders() const {
#ifdef AUDIO_ENABLED
            return true;
#else
            return shader_spectrum_required;
#endif
        }

        [[nodiscard]] bool spectrumHistoryEnabledForShaders() const {
            return options.audio_buffers > 0;
        }

        struct InterfaceUniformValue {
            std::string name;
            float value = 0.0F;
        };

        struct InterfaceMultipassState {
            bool enabled = false;
            std::vector<std::string> shader_names;
        };

        struct InterfacePlaybackState {
            bool repeat = false;
            bool normalized_time = false;
        };

        struct InterfaceOverlayState {
            bool display_filter = false;
            bool watermark_enabled = false;
            std::string watermark_text;
            std::array<std::uint8_t, 3> watermark_color{};
        };

        struct InterfaceGpuFilterState {
            bool enabled = false;
            int frame_buffer_size = 8;
            std::vector<int> filter_indices;
        };

        struct InterfaceAudioFileState {
            std::uint32_t request_sequence = 0;
            std::string path;
            int output_device = -1;
            bool pass_through = false;
            bool trunc = false;
            bool repeat = false;
        };

        [[nodiscard]] bool read_interface_selection(
            std::uint32_t &sequence, std::string &selected_name,
            InterfaceMultipassState &multipass,
            std::vector<InterfaceUniformValue> &uniform_values,
            InterfacePlaybackState &playback,
            InterfaceOverlayState &overlay,
            InterfaceGpuFilterState &gpu_filters,
            InterfaceAudioFileState &audio_file) const {
            if (interface_selection == nullptr ||
                interface_semaphore == SEM_FAILED) {
                return false;
            }
            ipc::SemaphoreLock lock(interface_semaphore);
            if (!lock) {
                return false;
            }
            if (interface_selection->magic != ipc::SHADER_SELECTION_MAGIC ||
                interface_selection->version !=
                    ipc::SHADER_SELECTION_VERSION) {
                return false;
            }
            sequence = interface_selection->sequence;
            const auto name_end = std::find(
                std::begin(interface_selection->selected_shader_name),
                std::end(interface_selection->selected_shader_name), '\0');
            selected_name.assign(
                std::begin(interface_selection->selected_shader_name),
                name_end);
            multipass.enabled = interface_selection->shader_pass_enabled != 0;
            const std::uint32_t pass_count = std::min(
                interface_selection->shader_pass_count, ipc::MAX_PASS_COUNT);
            multipass.shader_names.clear();
            multipass.shader_names.reserve(pass_count);
            for (std::uint32_t index = 0; index < pass_count; ++index) {
                const char *name_begin =
                    interface_selection->shader_pass_names[index];
                const char *name_limit = name_begin + ipc::MAX_SHADER_NAME;
                const char *shader_name_end =
                    std::find(name_begin, name_limit, '\0');
                if (shader_name_end != name_begin) {
                    multipass.shader_names.emplace_back(name_begin,
                                                        shader_name_end);
                }
            }
            const std::uint32_t uniform_count = std::min(
                interface_selection->custom_uniform_count,
                ipc::MAX_CUSTOM_UNIFORMS);
            uniform_values.clear();
            uniform_values.reserve(uniform_count);
            for (std::uint32_t index = 0; index < uniform_count; ++index) {
                const char *name_begin =
                    interface_selection->custom_uniform_names[index];
                const char *name_limit = name_begin + ipc::MAX_UNIFORM_NAME;
                const char *uniform_name_end =
                    std::find(name_begin, name_limit, '\0');
                uniform_values.push_back(
                    {std::string(name_begin, uniform_name_end),
                     interface_selection->custom_uniform_values[index]});
            }
            playback.repeat = interface_selection->repeat_enabled != 0;
            playback.normalized_time =
                interface_selection->normalized_time_enabled != 0;
            overlay.display_filter =
                interface_selection->display_filter_enabled != 0;
            overlay.watermark_enabled =
                interface_selection->watermark_enabled != 0;
            const auto watermark_end = std::find(
                std::begin(interface_selection->watermark_text),
                std::end(interface_selection->watermark_text), '\0');
            overlay.watermark_text.assign(
                std::begin(interface_selection->watermark_text), watermark_end);
            overlay.watermark_color = {
                interface_selection->watermark_r,
                interface_selection->watermark_g,
                interface_selection->watermark_b};
            gpu_filters.enabled =
                interface_selection->gpu_filter_enabled != 0;
            gpu_filters.frame_buffer_size =
                static_cast<int>(interface_selection->gpu_buffer_size);
            const std::uint32_t gpu_filter_count = std::min(
                interface_selection->gpu_filter_count,
                ipc::MAX_GPU_FILTER_COUNT);
            gpu_filters.filter_indices.clear();
            gpu_filters.filter_indices.reserve(gpu_filter_count);
            for (std::uint32_t index = 0; index < gpu_filter_count; ++index) {
                const int filter_index =
                    interface_selection->gpu_filter_indices[index];
                if (filter_index >= 0) {
                    gpu_filters.filter_indices.push_back(filter_index);
                }
            }
            audio_file.request_sequence =
                interface_selection->audio_file_sequence;
            const auto audio_path_end = std::find(
                std::begin(interface_selection->audio_file_path),
                std::end(interface_selection->audio_file_path), '\0');
            audio_file.path.assign(
                std::begin(interface_selection->audio_file_path),
                audio_path_end);
            audio_file.output_device =
                interface_selection->audio_output_device;
            audio_file.pass_through =
                interface_selection->audio_pass_through != 0;
            audio_file.trunc = interface_selection->audio_trunc != 0;
            audio_file.repeat = interface_selection->audio_repeat != 0;
            return true;
        }

        void initialize_interface_control() {
            if (!options.interface_shm) {
                return;
            }

            interface_semaphore =
                ::sem_open(ipc::SHADER_SELECTION_SEMAPHORE_NAME, 0);
            if (interface_semaphore == SEM_FAILED) {
                std::cerr
                    << "acmxvk: interface control unavailable: sem_open("
                    << ipc::SHADER_SELECTION_SEMAPHORE_NAME
                    << ") failed: " << std::strerror(errno) << '\n';
                return;
            }

            interface_shm_fd =
                ::shm_open(ipc::SHADER_SELECTION_SHM_NAME, O_RDWR, 0666);
            if (interface_shm_fd < 0) {
                std::cerr << "acmxvk: interface control unavailable: could not "
                             "open shared memory\n";
                cleanup_interface_control();
                return;
            }

            void *mapped = ::mmap(nullptr, sizeof(ipc::ShaderSelectionData),
                                  PROT_READ | PROT_WRITE, MAP_SHARED,
                                  interface_shm_fd, 0);
            if (mapped == MAP_FAILED) {
                std::cerr << "acmxvk: interface control unavailable: could not "
                             "map shared memory\n";
                cleanup_interface_control();
                return;
            }
            interface_selection =
                static_cast<ipc::ShaderSelectionData *>(mapped);

            std::string selected_name;
            InterfaceMultipassState multipass;
            std::vector<InterfaceUniformValue> uniform_values;
            InterfacePlaybackState playback;
            InterfaceOverlayState overlay;
            InterfaceGpuFilterState gpu_filters;
            InterfaceAudioFileState audio_file;
            if (!read_interface_selection(interface_last_sequence,
                                          selected_name, multipass,
                                          uniform_values, playback, overlay,
                                          gpu_filters, audio_file)) {
                std::cerr << "acmxvk: interface control protocol does not match "
                             "this build\n";
                cleanup_interface_control();
                return;
            }
            apply_interface_multipass_state(multipass);
            apply_interface_playback_state(playback, false);
            apply_interface_overlay_state(overlay, false);
            apply_interface_gpu_filter_state(gpu_filters, false);
            interface_last_audio_file_sequence =
                audio_file.request_sequence;
            std::cout << "acmxvk: interface live shader, multipass, playback, "
                         "overlay, GPU-filter, and audio-file control enabled\n";
        }

        void cleanup_interface_control() {
            if (interface_selection != nullptr) {
                ::munmap(interface_selection,
                         sizeof(ipc::ShaderSelectionData));
                interface_selection = nullptr;
            }
            if (interface_shm_fd >= 0) {
                ::close(interface_shm_fd);
                interface_shm_fd = -1;
            }
            if (interface_semaphore != SEM_FAILED) {
                ::sem_close(interface_semaphore);
                interface_semaphore = SEM_FAILED;
            }
        }

        void sync_interface_control() {
            std::uint32_t sequence = 0;
            std::string requested_name;
            InterfaceMultipassState multipass;
            std::vector<InterfaceUniformValue> uniform_values;
            InterfacePlaybackState playback;
            InterfaceOverlayState overlay;
            InterfaceGpuFilterState gpu_filters;
            InterfaceAudioFileState audio_file;
            if (!read_interface_selection(sequence, requested_name,
                                          multipass, uniform_values,
                                          playback, overlay, gpu_filters,
                                          audio_file) ||
                sequence == interface_last_sequence) {
                return;
            }
            interface_last_sequence = sequence;
            apply_interface_shader_selection(requested_name);
            apply_interface_multipass_state(multipass);
            apply_interface_uniform_values(uniform_values);
            apply_interface_playback_state(playback, true);
            apply_interface_overlay_state(overlay, true);
            apply_interface_gpu_filter_state(gpu_filters, true);
            if (audio_file.request_sequence !=
                interface_last_audio_file_sequence) {
                interface_last_audio_file_sequence =
                    audio_file.request_sequence;
                apply_interface_audio_file_state(audio_file);
            }
        }

        void apply_interface_playback_state(
            const InterfacePlaybackState &requested, bool announce) {
            if (options.repeat != requested.repeat) {
                options.repeat = requested.repeat;
                if (announce) {
                    std::cout << "acmxvk: interface video repeat "
                              << (options.repeat ? "enabled" : "disabled")
                              << '\n';
                }
            }
            if (options.normalized_time != requested.normalized_time) {
                options.normalized_time = requested.normalized_time;
                if (announce) {
                    std::cout << "acmxvk: interface normalized time "
                              << (options.normalized_time ? "enabled"
                                                          : "disabled")
                              << '\n';
                }
            }
        }

        void apply_interface_overlay_state(const InterfaceOverlayState &requested,
                                           bool announce) {
            if (options.display_filter != requested.display_filter) {
                options.display_filter = requested.display_filter;
                if (announce) {
                    std::cout << "acmxvk: interface display-filter overlay "
                              << (options.display_filter ? "enabled"
                                                         : "disabled")
                              << '\n';
                }
            }

            try {
                input::validate_string(requested.watermark_text,
                                       input::StringKind::DisplayText,
                                       "interface watermark", true);
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: rejected interface watermark: "
                          << error.what() << '\n';
                return;
            }

            const bool was_enabled =
                watermark_enabled && !options.watermark_text.empty();
            const bool requested_enabled =
                requested.watermark_enabled &&
                !requested.watermark_text.empty();
            const bool changed =
                watermark_enabled != requested_enabled ||
                options.watermark_text != requested.watermark_text ||
                options.watermark_color != requested.watermark_color;
            if (!changed) {
                return;
            }

            options.watermark_text = requested.watermark_text;
            options.watermark_color = requested.watermark_color;
            watermark_enabled = requested_enabled;
            if (!was_enabled && watermark_enabled) {
                counter_disabled = true;
            }
            if (announce) {
                std::cout << "acmxvk: interface watermark "
                          << (watermark_enabled ? "enabled" : "disabled");
                if (watermark_enabled) {
                    std::cout << " (color="
                              << static_cast<int>(options.watermark_color[0])
                              << ','
                              << static_cast<int>(options.watermark_color[1])
                              << ','
                              << static_cast<int>(options.watermark_color[2])
                              << ')';
                }
                std::cout << '\n';
            }
        }

        void apply_interface_gpu_filter_state(
            const InterfaceGpuFilterState &requested, bool announce) {
#ifdef ACMXVK_WITH_CUDA
            const bool requested_enabled =
                requested.enabled && !requested.filter_indices.empty();
            const bool currently_enabled = gpu_filter_engine != nullptr;
            const std::vector<int> effective_indices =
                requested_enabled ? requested.filter_indices
                                  : std::vector<int>{};
            if (requested_enabled == currently_enabled &&
                options.gpu_filter_indices == effective_indices &&
                (!requested_enabled ||
                 options.gpu_frame_buffer_size ==
                     requested.frame_buffer_size)) {
                return;
            }

            if (requested.enabled && requested.filter_indices.empty()) {
                std::cerr << "acmxvk: rejected enabled interface GPU-filter "
                             "state without any filter indices\n";
                return;
            }

            std::unique_ptr<gpu::FilterEngine> replacement;
            if (requested_enabled) {
                try {
                    replacement = std::make_unique<gpu::FilterEngine>(
                        requested.filter_indices,
                        requested.frame_buffer_size);
                } catch (const std::exception &error) {
                    std::cerr
                        << "acmxvk: rejected interface GPU-filter state: "
                        << error.what() << '\n';
                    return;
                }
            }

            gpu_filter_engine = std::move(replacement);
            options.gpu_filter_indices = effective_indices;
            if (requested_enabled) {
                options.gpu_frame_buffer_size = requested.frame_buffer_size;
            }

            if (frame_sprite != nullptr &&
                source_kind == SourceKind::Graphic && !graphic_rgba.empty()) {
                uploadInputFrame(graphic_rgba);
                if (history_initialized) {
                    updateHistoryFrame(graphic_rgba);
                    history_delay_counter = 0;
                }
            }

            if (announce) {
                std::cout << "acmxvk: interface CUDA filter chain "
                          << (requested_enabled ? "enabled" : "disabled");
                if (requested_enabled) {
                    std::cout << " (" << requested.filter_indices.size()
                              << " filters, " << requested.frame_buffer_size
                              << " history frames)";
                }
                std::cout << '\n';
            }
#else
            if (announce && requested.enabled) {
                std::cerr << "acmxvk: ignored interface GPU-filter state: this "
                             "build does not include acidcam-gpu\n";
            }
#endif
        }

        void apply_interface_audio_file_state(
            const InterfaceAudioFileState &requested) {
#ifdef AUDIO_ENABLED
            if (file_audio_source == nullptr || audio_engine == nullptr) {
                std::cerr
                    << "acmxvk: ignored live audio-file change because this "
                       "process was not started in audio-file mode\n";
                return;
            }
            if (requested.path.empty()) {
                std::cerr
                    << "acmxvk: rejected empty interface audio-file request\n";
                return;
            }

            auto replacement = std::make_unique<audio::FileAudioSource>();
            try {
                if (!replacement->open(requested.path)) {
                    std::cerr << "acmxvk: could not switch file audio to: "
                              << requested.path << '\n';
                    return;
                }
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: rejected interface audio-file request: "
                          << error.what() << '\n';
                return;
            }

            replacement->set_repeat(requested.repeat);
            if (requested.pass_through &&
                !replacement->enable_output(
                    requested.output_device,
                    static_cast<float>(options.audio_pass_through_gain))) {
                std::cerr
                    << "acmxvk: live audio-file output could not be opened; "
                       "continuing with visual reactivity only\n";
            }

            file_audio_source->stop_output();
            file_audio_source = std::move(replacement);
            options.audio_file = requested.path;
            options.audio_output_device = requested.output_device;
            options.audio_pass_through = requested.pass_through;
            options.audio_trunc = requested.trunc;
            options.audio_repeat = requested.repeat;
            audio_engine->reset();
            resetAudioWarmup();
            std::cout << "acmxvk: switched file audio to: "
                      << file_audio_source->path() << " (repeat="
                      << (options.audio_repeat ? "on" : "off")
                      << ", trunc=" << (options.audio_trunc ? "on" : "off")
                      << ", pass-through="
                      << (options.audio_pass_through ? "on" : "off")
                      << ")\n";
#else
            static_cast<void>(requested);
            std::cerr << "acmxvk: ignored interface audio-file request: this "
                         "build does not include audio support\n";
#endif
        }

        void apply_interface_multipass_state(
            const InterfaceMultipassState &requested) {
            std::vector<fs::path> requested_passes;
            if (requested.enabled) {
                if (requested.shader_names.empty()) {
                    std::cerr << "acmxvk: rejected enabled interface multipass "
                                 "state without any shader passes\n";
                    return;
                }
                requested_passes.reserve(requested.shader_names.size());
                for (const std::string &name : requested.shader_names) {
                    const fs::path requested_path(name);
                    const bool has_parent_reference = std::any_of(
                        requested_path.begin(), requested_path.end(),
                        [](const fs::path &part) { return part == ".."; });
                    if (requested_path.is_absolute() || has_parent_reference) {
                        std::cerr << "acmxvk: rejected unsafe interface "
                                     "multipass shader name: "
                                  << name << '\n';
                        return;
                    }
                    const fs::path shader = findShader(name);
                    if (shader.empty()) {
                        std::cerr << "acmxvk: interface multipass shader is not "
                                     "in the active library: "
                                  << name << '\n';
                        return;
                    }
                    requested_passes.push_back(shader);
                }
            }

            const bool requested_enabled =
                requested.enabled && !requested_passes.empty();
            if (multipass_enabled == requested_enabled &&
                configured_passes == requested_passes) {
                return;
            }
            if (frame_sprite != nullptr && shader_locked) {
                std::cerr << "acmxvk: interface multipass update ignored while "
                             "shader switching is locked\n";
                return;
            }

            if (frame_sprite != nullptr) {
                beginCrossfade();
            }
            configured_passes = std::move(requested_passes);
            multipass_enabled = requested_enabled;
            if (frame_sprite != nullptr) {
                applyShaderPipeline();
                resetShaderTime();
                autopilot_counter = 0;
            }

            if (multipass_enabled) {
                std::cout << "acmxvk: interface multipass enabled ("
                          << configured_passes.size() << " passes)";
                for (const fs::path &shader : configured_passes) {
                    std::cout << "\n  " << shader.filename().string();
                }
                std::cout << '\n';
            } else {
                std::cout << "acmxvk: interface multipass disabled\n";
            }
        }

        void apply_interface_shader_selection(
            const std::string &requested_name) {
            if (requested_name.empty()) {
                return;
            }

            const fs::path requested(requested_name);
            const bool has_parent_reference =
                std::any_of(requested.begin(), requested.end(),
                            [](const fs::path &part) { return part == ".."; });
            if (requested.is_absolute() || has_parent_reference) {
                std::cerr << "acmxvk: rejected unsafe interface shader name: "
                          << requested_name << '\n';
                return;
            }

            const fs::path shader = findShader(requested_name);
            const auto match = std::find(shaders.begin(), shaders.end(), shader);
            if (shader.empty() || match == shaders.end()) {
                std::cerr << "acmxvk: interface shader is not in the active "
                             "library: "
                          << requested_name << '\n';
                return;
            }

            const std::size_t next_index =
                static_cast<std::size_t>(std::distance(shaders.begin(), match));
            if (next_index == shader_index) {
                return;
            }
            if (shader_locked || frame_sprite == nullptr) {
                std::cerr << "acmxvk: interface shader selection ignored while "
                             "shader switching is locked\n";
                return;
            }

            beginCrossfade();
            shader_index = next_index;
            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            std::cout << "acmxvk: interface selected " << activeShaderRole()
                      << ' ' << (shader_index + 1) << '/' << shaders.size()
                      << ": " << currentShader() << '\n';
        }

        void apply_interface_uniform_values(
            const std::vector<InterfaceUniformValue> &uniform_values) {
            if (uniform_values.empty()) {
                return;
            }

            std::size_t changed_count = 0;
            std::size_t ignored_count = 0;
            for (const InterfaceUniformValue &incoming : uniform_values) {
                if (!isValidCustomUniformName(incoming.name) ||
                    !std::isfinite(incoming.value)) {
                    ++ignored_count;
                    continue;
                }
                const auto match = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &uniform) {
                        return uniform.name == incoming.name;
                    });
                if (match == custom_uniforms.end()) {
                    ++ignored_count;
                    continue;
                }
                const std::size_t index = static_cast<std::size_t>(
                    std::distance(custom_uniforms.begin(), match));
                if (index >= custom_uniform_values.size()) {
                    ++ignored_count;
                    continue;
                }
                const float value = static_cast<float>(std::clamp(
                    static_cast<double>(incoming.value), match->minimum,
                    match->maximum));
                if (custom_uniform_values[index] == value) {
                    continue;
                }
                custom_uniform_values[index] = value;
                ++changed_count;
            }

            if (changed_count > 0) {
                uploadCustomUniforms();
                std::cout << "acmxvk: interface updated " << changed_count
                          << " custom uniform(s)\n";
            }
            if (ignored_count > 0) {
                std::cerr << "acmxvk: interface ignored " << ignored_count
                          << " unknown or invalid custom uniform(s)\n";
            }
        }

        [[nodiscard]] fs::path findShader(std::string name) const {
            name = trim(std::move(name));
            if (name.empty()) {
                return {};
            }

            fs::path requested(name);
            if (requested.extension() != ".spv") {
                requested += ".spv";
            }
            const auto match = std::find_if(shaders.begin(), shaders.end(),
                                            [&](const fs::path &shader) {
                                                return shader.filename() == requested.filename() ||
                                                       (!shader_library_directory.empty() &&
                                                        shader.lexically_relative(
                                                            shader_library_directory) ==
                                                            requested);
                                            });
            return match == shaders.end() ? fs::path{} : *match;
        }

        void loadShaderPasses() {
            for (const int index : options.shader_pass_indices) {
                if (index < 0 || index >= static_cast<int>(shaders.size())) {
                    throw std::runtime_error("shader pass index is out of range: " +
                                             std::to_string(index));
                }
                configured_passes.push_back(shaders[static_cast<std::size_t>(index)]);
            }
            for (const std::string &name : options.shader_pass_files) {
                const fs::path shader = findShader(name);
                if (shader.empty()) {
                    throw std::runtime_error("shader pass file is not listed in the manifest: " +
                                             name);
                }
                configured_passes.push_back(shader);
            }
            multipass_enabled = !configured_passes.empty();
        }

        void loadPlaylist() {
            if (options.playlist_file.empty()) {
                return;
            }

            input::validate_file_size(options.playlist_file,
                                      "shader playlist");
            std::ifstream playlist_input(options.playlist_file);
            if (!playlist_input) {
                throw std::runtime_error("unable to open playlist: " + options.playlist_file);
            }

            PlaylistNode *current_node = nullptr;
            std::vector<fs::path> default_entries;
            std::string line;
            std::size_t line_number = 1;
            std::size_t entry_count = 0;
            while (input::read_bounded_line(
                playlist_input, line, "shader playlist", line_number++)) {
                line = trim(std::move(line));
                if (line.empty() || line.front() == '#') {
                    continue;
                }
                if (line.size() >= 2 && line.front() == '[' && line.back() == ']') {
                    if (playlist.size() >= input::MAX_PLAYLIST_NODES) {
                        throw std::runtime_error(
                            "shader playlist contains too many nodes");
                    }
                    std::string node_name =
                        trim(line.substr(1, line.size() - 2));
                    input::validate_string(node_name,
                                           input::StringKind::DisplayText,
                                           "shader playlist node");
                    playlist.push_back({std::move(node_name), {}});
                    current_node = &playlist.back();
                    continue;
                }
                if (line.front() == '[' || line.back() == ']') {
                    throw std::runtime_error(
                        "malformed shader playlist node at line " +
                        std::to_string(line_number - 1));
                }
                if (++entry_count > input::MAX_PLAYLIST_ENTRIES) {
                    throw std::runtime_error(
                        "shader playlist contains too many entries");
                }
                input::validate_string(line, input::StringKind::Path,
                                       "shader playlist entry");

                const fs::path shader = findShader(line);
                if (shader.empty()) {
                    std::cerr << "acmxvk: playlist shader not found: " << line << '\n';
                    continue;
                }
                if (current_node != nullptr) {
                    current_node->shaders.push_back(shader);
                } else {
                    default_entries.push_back(shader);
                }
            }

            playlist.erase(std::remove_if(playlist.begin(), playlist.end(),
                                          [](const PlaylistNode &node) {
                                              return node.shaders.empty();
                                          }),
                           playlist.end());
            if (!default_entries.empty()) {
                playlist.insert(playlist.begin(), {"Default", std::move(default_entries)});
            }
            if (playlist.empty()) {
                throw std::runtime_error("playlist contains no shaders available in the SPIR-V library");
            }
            playlist_enabled = options.enable_playlist;

            std::size_t shader_count = 0;
            for (const PlaylistNode &node : playlist) {
                shader_count += node.shaders.size();
            }
            std::cout << "acmxvk: playlist loaded " << shader_count << " shaders in "
                      << playlist.size() << " nodes from " << options.playlist_file << '\n';
            logSelectedPlaylistNode("selected");
        }

        [[nodiscard]] std::string spriteVertexShader() const {
            const fs::path resource =
                findResource(options, "shaders/sprite.vert.spv");
            if (!resource.empty()) {
                return resource.string();
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_SPRITE_VERTEX_SHADER)) {
                return ACMXVK_INSTALL_SPRITE_VERTEX_SHADER;
            }
            return ACMXVK_BUILD_SPRITE_VERTEX_SHADER;
        }

        [[nodiscard]] std::string echoCacheShader() const {
            const fs::path resource =
                findResource(options, "shaders/echo_cache.frag.spv");
            if (!resource.empty()) {
                return resource.string();
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_ECHO_CACHE_SHADER)) {
                return ACMXVK_INSTALL_ECHO_CACHE_SHADER;
            }
            return ACMXVK_BUILD_ECHO_CACHE_SHADER;
        }

        [[nodiscard]] fs::path flipShader() const {
            const fs::path resource =
                findResource(options, "shaders/flip.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_FLIP_SHADER)) {
                return ACMXVK_INSTALL_FLIP_SHADER;
            }
            return ACMXVK_BUILD_FLIP_SHADER;
        }

        [[nodiscard]] fs::path passthroughShader() const {
            const fs::path resource =
                findResource(options, "shaders/passthrough.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_PASSTHROUGH_SHADER)) {
                return ACMXVK_INSTALL_PASSTHROUGH_SHADER;
            }
            return ACMXVK_BUILD_PASSTHROUGH_SHADER;
        }

        [[nodiscard]] fs::path humanCompositeShader() const {
            const fs::path resource =
                findResource(options, "shaders/human_composite.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER)) {
                return ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER;
            }
            return ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER;
        }

        [[nodiscard]] fs::path crossfadeShader() const {
            const std::string filename =
                std::string(CROSSFADE_NAMES[crossfade_shader_index]) +
                ".frag.spv";
            const fs::path resource =
                findResource(options, fs::path("shaders/xfade") / filename);
            if (!resource.empty()) {
                return resource;
            }
            const fs::path installed =
                fs::path(ACMXVK_INSTALL_CROSSFADE_DIRECTORY) / filename;
            if (fs::is_regular_file(installed)) {
                return installed;
            }
            const fs::path built =
                fs::path(ACMXVK_BUILD_CROSSFADE_DIRECTORY) / filename;
            if (fs::is_regular_file(built)) {
                return built;
            }
            throw std::runtime_error("crossfade shader was not found: " +
                                     filename);
        }

        [[nodiscard]] fs::path modelVertexShader() const {
            const fs::path resource =
                findResource(options, "shaders/model.vert.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_MODEL_VERTEX_SHADER)) {
                return ACMXVK_INSTALL_MODEL_VERTEX_SHADER;
            }
            return ACMXVK_BUILD_MODEL_VERTEX_SHADER;
        }

        [[nodiscard]] fs::path modelFragmentShader() const {
            const fs::path resource =
                findResource(options, "shaders/model.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER)) {
                return ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER;
            }
            return ACMXVK_BUILD_MODEL_FRAGMENT_SHADER;
        }

        [[nodiscard]] fs::path defaultModel() const {
            const fs::path resource = findResource(options, "models/cube.obj");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_DEFAULT_MODEL)) {
                return ACMXVK_INSTALL_DEFAULT_MODEL;
            }
            return ACMXVK_BUILD_DEFAULT_MODEL;
        }

        [[nodiscard]] fs::path overlayFont() const {
            const fs::path resource = findResource(options, "data/font.ttf");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_OVERLAY_FONT)) {
                return ACMXVK_INSTALL_OVERLAY_FONT;
            }
            return ACMXVK_BUILD_OVERLAY_FONT;
        }

        void resolveConfiguredResourcePaths() {
            const auto resolve = [&](std::string &path,
                                     const fs::path &resource_subdirectory,
                                     std::string_view label) {
                if (path.empty() || fs::is_regular_file(path) ||
                    fs::path(path).is_absolute()) {
                    return;
                }
                fs::path resolved = findResource(options, fs::path(path));
                if (resolved.empty()) {
                    resolved = findResource(
                        options, resource_subdirectory / fs::path(path));
                }
                if (!resolved.empty()) {
                    path = resolved.string();
                    std::cout << "acmxvk: " << label << " (resource path): "
                              << path << '\n';
                }
            };
            resolve(options.playlist_file, "playlists", "playlist");
            resolve(options.midi_map_file, "midi-examples", "MIDI map");
            if (options.enable_3d) {
                if (options.model_file.empty()) {
                    options.model_file = defaultModel().string();
                    std::cout << "acmxvk: 3D model (default): "
                              << options.model_file << '\n';
                } else {
                    resolve(options.model_file, "models", "3D model");
                }

                std::string model_name =
                    fs::path(options.model_file).filename().string();
                std::transform(
                    model_name.begin(), model_name.end(), model_name.begin(),
                    [](unsigned char character) {
                        return static_cast<char>(std::tolower(character));
                    });
                if (!model_name.ends_with(".obj") &&
                    !model_name.ends_with(".mxmod") &&
                    !model_name.ends_with(".mxmod.z")) {
                    throw std::runtime_error(
                        "--model requires an .obj, .mxmod, or .mxmod.z file");
                }
                if (!fs::is_regular_file(options.model_file)) {
                    throw std::runtime_error(
                        "3D model was not found: " + options.model_file);
                }
                constexpr std::uintmax_t MAX_MODEL_BYTES =
                    1024U * 1024U * 1024U;
                input::validate_file_size(options.model_file, "3D model",
                                          MAX_MODEL_BYTES);
            }
        }

        void initializeOverlayFont() {
            if (counter_disabled && !options.display_filter &&
                options.watermark_text.empty() && !options.interface_shm) {
                return;
            }

            const fs::path font = overlayFont();
            if (!fs::is_regular_file(font)) {
                throw std::runtime_error("overlay font was not found: " +
                                         font.string());
            }
            const VkExtent2D preview_extent = getSwapchainExtent();
            const int preview_height = preview_extent.height > 0U
                                           ? static_cast<int>(preview_extent.height)
                                           : options.height;
            constexpr int FONT_HEIGHT_DIVISOR = 60;
            overlay_font_size =
                std::max(12, preview_height / FONT_HEIGHT_DIVISOR);
            preview_overlay_font_size = overlay_font_size;
            setFont(font.string(), overlay_font_size);
            setPreviewFont(font.string(), preview_overlay_font_size);
            std::cout << "acmxvk: window-scaled output/HUD font "
                      << font.string() << " at " << overlay_font_size
                      << " points\n";
        }

        [[nodiscard]] static std::string clipOverlayText(std::string text) {
            constexpr std::size_t MAX_OVERLAY_CHARACTERS = 120;
            return input::truncate_utf8(text, MAX_OVERLAY_CHARACTERS);
        }

        [[nodiscard]] const std::vector<fs::path> *activePasses() const {
            if (playlist_enabled && !playlist.empty()) {
                return &playlist[playlist_index].shaders;
            }
            if (multipass_enabled && !configured_passes.empty()) {
                return &configured_passes;
            }
            return nullptr;
        }

        [[nodiscard]] std::string_view activeShaderRole() const {
            const std::vector<fs::path> *passes = activePasses();
            return passes != nullptr && !passes->empty() ? "Post-shader"
                                                         : "Shader";
        }

        [[nodiscard]] std::string activePassDescription() const {
            const std::vector<fs::path> *passes = activePasses();
            if (passes == nullptr || passes->empty()) {
                return {};
            }

            std::string description = "Multipass: ";
            for (std::size_t index = 0; index < passes->size(); ++index) {
                if (index > 0U) {
                    description += ", ";
                }
                description += (*passes)[index].filename().string();
            }
            return clipOverlayText(std::move(description));
        }

        [[nodiscard]] std::string activePlaylistDescription() const {
            if (!playlist_enabled || playlist.empty()) {
                return {};
            }
            std::ostringstream description;
            description << "Playlist [" << (playlist_index + 1) << '/'
                        << playlist.size() << "]: "
                        << playlist[playlist_index].name;
            return clipOverlayText(description.str());
        }

        [[nodiscard]] static std::string formatHudTime(double seconds_value) {
            const double finite_seconds =
                std::isfinite(seconds_value) ? seconds_value : 0.0;
            const auto elapsed = static_cast<std::uint64_t>(
                std::floor(std::max(0.0, finite_seconds)));
            const std::uint64_t hours = elapsed / 3600U;
            const std::uint64_t minutes = (elapsed / 60U) % 60U;
            const std::uint64_t seconds = elapsed % 60U;
            std::ostringstream text;
            text << std::setfill('0') << std::setw(2) << hours << ':'
                 << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
            return text.str();
        }

        void updateWindowTitle(bool force = false) {
            SDL_Window *window = getSDLWindow();
            if (window == nullptr) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            constexpr auto UPDATE_INTERVAL = std::chrono::milliseconds(500);
            if (!force && window_title_last_update.time_since_epoch().count() != 0 &&
                now - window_title_last_update < UPDATE_INTERVAL) {
                return;
            }
            window_title_last_update = now;

            const bool recording = writer.is_open() || options.png_output;
            double elapsed_seconds = hudWallElapsedSeconds();
            std::uint64_t displayed_frames = frame_count;
            if (recording && recording_fps > 0.0) {
                displayed_frames = output_frame_count;
                elapsed_seconds = writer.is_open()
                                      ? writer.get_duration()
                                      : static_cast<double>(output_frame_count) /
                                            recording_fps;
            } else if (source_kind == SourceKind::Video) {
                displayed_frames = video_source_frame_count;
                elapsed_seconds = hudVideoPositionSeconds();
            }

            std::ostringstream title;
            if (source_kind == SourceKind::Graphic) {
                title << "ACMXVK - Graphics Mode - "
                      << formatHudTime(elapsed_seconds) << " ["
                      << displayed_frames << " frames]";
            } else if (source_kind == SourceKind::Video) {
                const std::uint64_t total_frames =
                    video_duration_seconds > 0.0 && video_source_fps > 0.0
                        ? static_cast<std::uint64_t>(std::llround(
                              video_duration_seconds * video_source_fps))
                        : 0U;
                title << "ACMXVK - [" << video_source_frame_count << '/';
                if (total_frames > 0U) {
                    title << total_frames;
                } else {
                    title << '?';
                }
                title << "] - " << formatHudTime(elapsed_seconds)
                      << " - Video Mode";
            } else {
                title << "ACMXVK - Capture Mode - "
                      << formatHudTime(elapsed_seconds) << " ["
                      << displayed_frames << " frames]";
            }

            if (recording) {
                title << " (Recording)";
                if (writer.is_open()) {
                    constexpr double BYTES_PER_MEGABYTE = 1024.0 * 1024.0;
                    const double file_size_mb =
                        static_cast<double>(writer.get_bytes_written()) /
                        BYTES_PER_MEGABYTE;
                    title << " [File: " << std::fixed << std::setprecision(2)
                          << file_size_mb << " MB]";
                }
            } else {
                title << " (Preview)";
            }

            const std::string text = title.str();
            SDL_SetWindowTitle(window, text.c_str());
        }

        [[nodiscard]] double hudWallElapsedSeconds() const {
            return std::max(
                0.0,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              hud_session_start)
                    .count());
        }

        [[nodiscard]] bool currentVideoTimeline(
            double &timeline,
            std::uint64_t *frame_index = nullptr) const {
            if (source_kind != SourceKind::Video ||
                video_source_frame_count == 0U ||
                !std::isfinite(video_source_fps) || video_source_fps <= 0.0) {
                return false;
            }
            const std::uint64_t index = video_source_frame_count - 1U;
            timeline = static_cast<double>(index) / video_source_fps;
            if (frame_index != nullptr) {
                *frame_index = index;
            }
            return true;
        }

        [[nodiscard]] double hudVideoPositionSeconds() const {
            double position = 0.0;
            if (!currentVideoTimeline(position)) {
                return 0.0;
            }
            if (video_duration_seconds > 0.0) {
                position = std::min(position, video_duration_seconds);
            }
            return std::max(0.0, position);
        }

        [[nodiscard]] std::string hudVideoTimeString() const {
            std::string text = "Video: " +
                               formatHudTime(hudVideoPositionSeconds()) +
                               " / ";
            text += video_duration_seconds > 0.0
                        ? formatHudTime(video_duration_seconds)
                        : "--:--:--";
            return text;
        }

        [[nodiscard]] std::string hudElapsedTimeString() const {
            return "Elapsed: " + formatHudTime(hudWallElapsedSeconds());
        }

        void updateHudFrameRate() {
            ++hud_fps_frame_count;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - hud_fps_last_tick).count();
            if (elapsed < 0.5) {
                return;
            }
            hud_display_fps = static_cast<double>(hud_fps_frame_count) / elapsed;
            hud_fps_frame_count = 0;
            hud_fps_last_tick = now;
        }

        void paceMaximizedRendering() {
            if (!options.maximize_fps || options.requested_fps <= 0.0) {
                return;
            }

            const auto interval = std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / options.requested_fps));
            const auto now = std::chrono::steady_clock::now();
            if (!render_pacing_started) {
                render_pacing_started = true;
                next_render_tick = now;
                return;
            }

            next_render_tick += interval;
            if (next_render_tick > now) {
                std::this_thread::sleep_until(next_render_tick);
                return;
            }

            if (now - next_render_tick > interval * 4) {
                next_render_tick = now;
            }
        }

        void updateCameraFrameRate() {
            if (source_kind != SourceKind::Camera) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (camera_fps_frame_count == 0) {
                camera_fps_frame_count = 1;
                camera_fps_last_tick = now;
                return;
            }

            ++camera_fps_frame_count;
            const double elapsed = std::chrono::duration<double>(
                                       now - camera_fps_last_tick)
                                       .count();
            if (elapsed < 1.0) {
                return;
            }

            camera_delivered_fps =
                static_cast<double>(camera_fps_frame_count - 1) / elapsed;
            camera_fps_frame_count = 1;
            camera_fps_last_tick = now;

            const double log_threshold = std::max(
                5.0, camera_last_logged_fps * 0.2);
            if (camera_last_logged_fps <= 0.0 ||
                std::abs(camera_delivered_fps - camera_last_logged_fps) >=
                    log_threshold) {
                std::ostringstream status;
                status << "acmxvk: camera delivery: " << std::fixed
                       << std::setprecision(1) << camera_delivered_fps
                       << " FPS measured";
                if (camera_reported_fps > 0.0) {
                    status << " (driver reports " << camera_reported_fps
                           << " FPS)";
                }
                std::cout << status.str() << '\n';
                camera_last_logged_fps = camera_delivered_fps;
            }
        }

        void queueRuntimeHud(int &y, int line_height) {
            if (counter_disabled) {
                return;
            }
            updateHudFrameRate();

            const SDL_Color shader_color{0U, 96U, 255U, 255U};
            std::string shader = effects_enabled
                                     ? fs::path(currentShader()).filename().string()
                                     : "bypassed";
            if (shader_locked) {
                shader += " [locked]";
            }
            printPreviewText(clipOverlayText(
                                 std::string(activeShaderRole()) + ": " +
                                 std::move(shader)),
                             10, y, shader_color);
            y += line_height;

            const SDL_Color crossfade_color{255U, 192U, 0U, 255U};
            std::ostringstream crossfade_status;
            crossfade_status << "XFade [" << (crossfade_shader_index + 1)
                             << '/' << CROSSFADE_NAMES.size() << "]: "
                             << CROSSFADE_NAMES[crossfade_shader_index];
            printPreviewText(clipOverlayText(crossfade_status.str()), 10, y,
                             crossfade_color);
            y += line_height;

            const std::string playlist_description =
                activePlaylistDescription();
            if (!playlist_description.empty()) {
                const SDL_Color playlist_color{255U, 0U, 255U, 255U};
                printPreviewText(playlist_description, 10, y,
                                 playlist_color);
                y += line_height;
            }

            const std::vector<fs::path> *passes = activePasses();
            if (passes != nullptr && !passes->empty()) {
                constexpr std::size_t MAX_HUD_PASS_LINES = 8U;
                const std::size_t displayed_passes =
                    std::min(passes->size(), MAX_HUD_PASS_LINES);
                for (std::size_t index = 0; index < displayed_passes;
                     ++index) {
                    std::ostringstream pass;
                    pass << "Pass [" << (index + 1) << '/' << passes->size()
                         << "]: " << (*passes)[index].filename().string();
                    printPreviewText(clipOverlayText(pass.str()), 10, y,
                                     shader_color);
                    y += line_height;
                }
                if (displayed_passes < passes->size()) {
                    const std::string remaining =
                        "Passes: +" +
                        std::to_string(passes->size() - displayed_passes) +
                        " more";
                    printPreviewText(remaining, 10, y, shader_color);
                    y += line_height;
                }
            }

            if (model_initialized) {
                const SDL_Color model_color{0U, 220U, 180U, 255U};
                std::string model_status =
                    model_3d_active ? "Model: " : "Model (2D bypass): ";
                model_status +=
                    fs::path(options.model_file).filename().string();
                if (model_wave_active) {
                    model_status += " [wave]";
                }
                if (model_scale_oscillation_active) {
                    model_status += " [oscillate]";
                }
                printPreviewText(clipOverlayText(std::move(model_status)), 10,
                                 y, model_color);
                y += line_height;
            }

#ifdef ACMXVK_WITH_DNN
            const SDL_Color dnn_color{64U, 220U, 128U, 255U};
            if (human_segmenter != nullptr) {
                printPreviewText(
                    options.human_background
                        ? "DNN: PP-HumanSeg [background]"
                        : "DNN: PP-HumanSeg [foreground]",
                    10, y, dnn_color);
                y += line_height;
            }
            if (edge_detector != nullptr) {
                printPreviewText("DNN: DexiNed edge", 10, y, dnn_color);
                y += line_height;
            }
            if (generic_onnx_processor != nullptr) {
                printPreviewText(
                    clipOverlayText(
                        "DNN: ONNX " +
                        fs::path(options.onnx_configuration)
                            .filename()
                            .string()),
                    10, y, dnn_color);
                y += line_height;
            }
#endif

#ifdef AUDIO_ENABLED
            if (file_audio_source != nullptr && file_audio_source->is_open()) {
                const std::string track = fs::path(
                                              file_audio_source
                                                  ->current_track_path())
                                              .filename()
                                              .string();
                if (!track.empty()) {
                    const SDL_Color track_color{255U, 0U, 255U, 255U};
                    printPreviewText(clipOverlayText("Track: " + track), 10,
                                     y, track_color);
                    y += line_height;
                }
            }
#endif

#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                const SDL_Color gpu_color{255U, 0U, 255U, 255U};
                printPreviewText(
                    clipOverlayText(
                        "GPU: " +
                        gpu_filter_engine->active_filter_description()),
                    10, y, gpu_color);
                y += line_height;
            }
#endif

            if (autopilot_enabled) {
                const int remaining =
                    std::max(0, autopilot_interval_frames - autopilot_counter);
                std::ostringstream status;
                status << "Autopilot "
                       << (autopilot_sequential ? "seq" : "rnd") << ' ';
                if (options.autopilot_random_timeout > 0) {
                    status << "[4-" << options.autopilot_random_timeout
                           << "] cur=" << autopilot_interval_frames;
                } else {
                    status << "every " << autopilot_interval_frames << 'f';
                }
                status << " next=" << remaining << "f";
                if (!playlist.empty()) {
                    status << " idx=" << (playlist_index + 1) << '/'
                           << playlist.size();
                }
                const SDL_Color autopilot_color{0U, 255U, 255U, 255U};
                printPreviewText(clipOverlayText(status.str()), 10, y,
                                 autopilot_color);
                y += line_height;
            }

            const SDL_Color status_color{255U, 255U, 255U, 255U};
            if (source_kind == SourceKind::Video) {
                printPreviewText(hudVideoTimeString(), 10, y, status_color);
                y += line_height;
            }
            printPreviewText(hudElapsedTimeString(), 10, y, status_color);
            y += line_height;
            std::ostringstream fps;
            fps << "Render: " << std::fixed << std::setprecision(1)
                << hud_display_fps << " FPS";
            printPreviewText(fps.str(), 10, y, status_color);
            y += line_height;
            if (source_kind == SourceKind::Camera) {
                std::ostringstream camera_fps;
                camera_fps << "Camera: ";
                if (camera_delivered_fps > 0.0) {
                    camera_fps << std::fixed << std::setprecision(1)
                               << camera_delivered_fps << " FPS measured";
                } else {
                    camera_fps << "measuring...";
                }
                printPreviewText(camera_fps.str(), 10, y, status_color);
                y += line_height;
            }
            const SDL_Color hint_color{128U, 128U, 128U, 255U};
            printPreviewText("F9: Toggle overlay", 10, y, hint_color);
            y += line_height;
        }

        void queueOverlayText() {
            if (counter_disabled && !options.display_filter &&
                (!watermark_enabled || options.watermark_text.empty())) {
                return;
            }

            constexpr int LEFT_MARGIN = 10;
            constexpr int TOP_MARGIN = 10;
            const int line_height = overlay_font_size + 4;
            const int preview_line_height = preview_overlay_font_size + 4;
            int preview_y =
                TOP_MARGIN +
                (!counter_disabled && watermark_enabled &&
                         !options.watermark_text.empty()
                     ? preview_line_height
                     : 0);
            queueRuntimeHud(preview_y, preview_line_height);
            int y = TOP_MARGIN;
            if (options.display_filter) {
                const SDL_Color filter_color{255U, 0U, 255U, 255U};
                std::string shader = effects_enabled
                                         ? fs::path(currentShader()).filename().string()
                                         : "bypassed";
                printText(clipOverlayText(
                              std::string(activeShaderRole()) + ": " +
                              std::move(shader)),
                          LEFT_MARGIN, y, filter_color);
                y += line_height;

                if (playlist_enabled && !playlist.empty()) {
                    printText(clipOverlayText("Playlist: " +
                                              playlist[playlist_index].name),
                              LEFT_MARGIN, y, filter_color);
                    y += line_height;
                }
                const std::string passes = activePassDescription();
                if (!passes.empty()) {
                    printText(passes, LEFT_MARGIN, y, filter_color);
                    y += line_height;
                }
#ifdef ACMXVK_WITH_CUDA
                if (gpu_filter_engine != nullptr) {
                    printText(clipOverlayText(
                                  "GPU: " + gpu_filter_engine
                                                ->active_filter_description()),
                              LEFT_MARGIN, y, filter_color);
                    y += line_height;
                }
#endif
            }

            if (watermark_enabled && !options.watermark_text.empty()) {
                const SDL_Color watermark_color{
                    options.watermark_color[0], options.watermark_color[1],
                    options.watermark_color[2], 255U};
                printText(clipOverlayText(options.watermark_text), LEFT_MARGIN,
                          y, watermark_color);
            }
        }

        [[nodiscard]] static std::string captureFourccName(double value) {
            if (!std::isfinite(value) || value <= 0.0 ||
                value > static_cast<double>(
                            std::numeric_limits<std::uint32_t>::max())) {
                return "unknown";
            }
            const auto fourcc = static_cast<std::uint32_t>(std::llround(value));
            std::string name(4, ' ');
            for (std::size_t index = 0; index < name.size(); ++index) {
                const auto byte = static_cast<unsigned char>(
                    (fourcc >> (index * 8U)) & 0xffU);
                if (!std::isprint(byte)) {
                    return "unknown";
                }
                name[index] = static_cast<char>(byte);
            }
            return name;
        }

        [[nodiscard]] bool dnnHostProcessingEnabled() const {
#ifdef ACMXVK_WITH_DNN
            return edge_detector != nullptr || human_segmenter != nullptr ||
                   generic_onnx_processor != nullptr;
#else
            return false;
#endif
        }

        void applyDnnEffects(cv::Mat &rgba) {
#ifdef ACMXVK_WITH_DNN
            if (human_segmenter != nullptr && !rgba.empty()) {
                cv::Mat bgr;
                cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                const cv::Mat mask = human_segmenter->infer(bgr);
                if (mask.empty()) {
                    throw std::runtime_error(
                        "PP-HumanSeg produced an empty person mask");
                }
                const float black_point =
                    static_cast<float>(options.human_black_point);
                const float white_point =
                    static_cast<float>(options.human_white_point);
                if (options.human_background) {
                    const cv::Mat alpha = dnn::hardenedAlphaMask(
                        bgr, mask, black_point, white_point);
                    cv::cvtColor(bgr, human_overlay_rgba,
                                 cv::COLOR_BGR2RGBA);
                    std::vector<cv::Mat> overlay_channels;
                    cv::split(human_overlay_rgba, overlay_channels);
                    alpha.copyTo(overlay_channels[3]);
                    cv::merge(overlay_channels, human_overlay_rgba);

                    const cv::Mat foreground = dnn::isolateBody(
                        bgr, mask, black_point, white_point);
                    cv::Mat background;
                    cv::subtract(bgr, foreground, background);
                    cv::cvtColor(background, rgba, cv::COLOR_BGR2RGBA);
                } else {
                    const cv::Mat foreground = dnn::isolateBody(
                        bgr, mask, black_point, white_point);
                    cv::cvtColor(foreground, rgba, cv::COLOR_BGR2RGBA);
                }
            }
            if (edge_detector != nullptr && !rgba.empty()) {
                try {
                    cv::Mat bgr;
                    cv::Mat edge;
                    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                    edge_detector->process(bgr, edge);
                    if (edge.empty()) {
                        throw std::runtime_error(
                            "DexiNed produced an empty edge frame");
                    }
                    if (edge.channels() == 1) {
                        cv::cvtColor(edge, rgba, cv::COLOR_GRAY2RGBA);
                    } else {
                        cv::cvtColor(edge, rgba, cv::COLOR_BGR2RGBA);
                    }
                } catch (const std::exception &error) {
                    std::cerr
                        << "acmxvk: edge inference failed; disabling DNN "
                           "effect: "
                        << error.what() << '\n';
                    edge_detector.reset();
                }
            }
            if (generic_onnx_processor != nullptr && !rgba.empty()) {
                try {
                    cv::Mat bgr;
                    cv::Mat processed;
                    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                    generic_onnx_processor->process(bgr, processed);
                    if (processed.empty()) {
                        throw std::runtime_error(
                            "generic ONNX model produced an empty frame");
                    }
                    if (processed.channels() == 1) {
                        cv::cvtColor(processed, rgba,
                                     cv::COLOR_GRAY2RGBA);
                    } else {
                        cv::cvtColor(processed, rgba,
                                     cv::COLOR_BGR2RGBA);
                    }
                } catch (const std::exception &error) {
                    std::cerr
                        << "acmxvk: generic ONNX inference failed; disabling "
                           "model: "
                        << error.what() << '\n';
                    generic_onnx_processor.reset();
                }
            }
#else
            static_cast<void>(rgba);
#endif
        }

        void updateHumanOverlayTexture() {
#ifdef ACMXVK_WITH_DNN
            if (!options.human_background || human_overlay_rgba.empty() ||
                getDevice() == VK_NULL_HANDLE) {
                return;
            }
            if (human_overlay_sprite == nullptr) {
                human_overlay_sprite = createSprite(1, 1);
                human_overlay_sprite->enableHistoryTexture(
                    static_cast<std::uint32_t>(human_overlay_rgba.cols),
                    static_cast<std::uint32_t>(human_overlay_rgba.rows), 1U);
            }
            cv::Mat upload = human_overlay_rgba;
            cv::Mat flipped;
            if (options.flip_output) {
                cv::flip(human_overlay_rgba, flipped, 0);
                upload = flipped;
            }
            human_overlay_sprite->updateHistoryTexture(
                upload.ptr(), upload.cols, upload.rows,
                static_cast<int>(upload.step));
#endif
        }

        void openInput() {
            if (!options.graphic_file.empty()) {
                source_kind = SourceKind::Graphic;
                graphic_rgba = loadRgbaImage(options.graphic_file);
                applyDnnEffects(graphic_rgba);
                rotateFrame(graphic_rgba, options.frame_rotation);
                if (!human_overlay_rgba.empty()) {
                    rotateFrame(human_overlay_rgba, options.frame_rotation);
                }
                return;
            }

            source_kind = options.input_file.empty() ? SourceKind::Camera : SourceKind::Video;
            bool opened = false;
            if (source_kind == SourceKind::Video) {
                opened = openVideoCapture();
            } else {
                opened = capture.open(options.camera_device);
            }
            if (!opened) {
                const std::string source = source_kind == SourceKind::Video
                                               ? options.input_file
                                               : std::to_string(options.camera_device);
                throw std::runtime_error("unable to open capture source: " + source);
            }

            if (source_kind == SourceKind::Video) {
                video_duration_seconds =
                    probeVideoDuration(options.input_file);
                std::ostringstream timeline;
                timeline << "acmxvk: video timeline: " << std::fixed
                         << std::setprecision(3) << video_source_fps
                         << " FPS";
                if (video_duration_seconds > 0.0) {
                    timeline << ", " << video_duration_seconds
                             << " seconds";
                } else {
                    timeline << ", duration unavailable";
                }
                std::cout << timeline.str() << '\n';
                if (options.use_source_fps) {
                    std::cout
                        << "acmxvk: source-FPS playback enabled at "
                        << video_source_fps
                        << " FPS; early frames wait and late frames are skipped\n";
                }
            }

            if (source_kind == SourceKind::Camera) {
                // Match ACMX2's ordering. Some V4L2 drivers renegotiate the
                // frame interval when dimensions or pixel format change.
                capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);
                capture.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
                const int requested_fourcc = options.use_yuv
                                                 ? cv::VideoWriter::fourcc(
                                                       'Y', 'U', 'Y', 'V')
                                                 : cv::VideoWriter::fourcc(
                                                       'M', 'J', 'P', 'G');
                capture.set(cv::CAP_PROP_FOURCC,
                            static_cast<double>(requested_fourcc));
                if (options.requested_fps > 0.0) {
                    capture.set(cv::CAP_PROP_FPS, options.requested_fps);
                }

                camera_reported_width = static_cast<int>(
                    std::lround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
                camera_reported_height = static_cast<int>(
                    std::lround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
                camera_reported_fps = capture.get(cv::CAP_PROP_FPS);
                if (!std::isfinite(camera_reported_fps) ||
                    camera_reported_fps < 0.0) {
                    camera_reported_fps = 0.0;
                }
                const std::string reported_fourcc = captureFourccName(
                    capture.get(cv::CAP_PROP_FOURCC));

                std::cout << "acmxvk: camera opened: "
                          << camera_reported_width << 'x'
                          << camera_reported_height;
                if (camera_reported_fps > 0.0) {
                    std::cout << " at reported " << camera_reported_fps
                              << " FPS";
                } else {
                    std::cout << " at an unreported frame rate";
                }
                std::cout << ", format=" << reported_fourcc << '\n';

                if (camera_reported_width != options.camera_width ||
                    camera_reported_height != options.camera_height) {
                    std::cerr << "acmxvk: camera mode warning: requested "
                              << options.camera_width << 'x'
                              << options.camera_height << " but driver reports "
                              << camera_reported_width << 'x'
                              << camera_reported_height
                              << '\n';
                }
                if (options.requested_fps > 0.0 &&
                    camera_reported_fps > 0.0 &&
                    std::abs(camera_reported_fps - options.requested_fps) >
                        0.05) {
                    std::cerr << "acmxvk: camera mode warning: requested "
                              << options.requested_fps
                              << " FPS but driver reports "
                              << camera_reported_fps << " FPS\n";
                }
                const std::string requested_format =
                    options.use_yuv ? "YUYV" : "MJPG";
                if (reported_fourcc != "unknown" &&
                    reported_fourcc != requested_format) {
                    std::cerr << "acmxvk: camera mode warning: requested "
                              << requested_format << " but driver reports "
                              << reported_fourcc << '\n';
                }
                if (options.maximize_fps) {
                    latest_camera_frame.start(capture);
                    std::cout
                        << "acmxvk: maximize FPS active: asynchronous camera "
                           "capture, Vulkan render target "
                        << options.requested_fps << " FPS\n";
                    if (options.enable_vsync) {
                        std::cout
                            << "acmxvk: maximize FPS note: VSync may cap the "
                               "render rate to the display refresh\n";
                    }
                }
            }
        }

        [[nodiscard]] std::pair<int, int> source_dimensions() {
            int source_width = options.width;
            int source_height = options.height;
            if (source_kind == SourceKind::Graphic) {
                source_width = graphic_rgba.cols;
                source_height = graphic_rgba.rows;
            } else {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                if (using_ffmpeg_capture) {
                    source_width = ffmpeg_capture.width();
                    source_height = ffmpeg_capture.height();
                } else
#endif
                {
                    if (source_kind == SourceKind::Camera &&
                        camera_reported_width > 0 &&
                        camera_reported_height > 0) {
                        source_width = camera_reported_width;
                        source_height = camera_reported_height;
                    } else {
                        source_width = static_cast<int>(
                            std::lround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
                        source_height = static_cast<int>(
                            std::lround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
                    }
                }
                if (source_width <= 0 || source_height <= 0) {
                    source_width = source_kind == SourceKind::Camera
                                       ? options.camera_width
                                       : options.width;
                    source_height = source_kind == SourceKind::Camera
                                        ? options.camera_height
                                        : options.height;
                }
                if (rotationSwapsDimensions(options.frame_rotation)) {
                    std::swap(source_width, source_height);
                }
            }
            return {source_width, source_height};
        }

        void configureRenderResolution() {
            int render_width = options.width;
            int render_height = options.height;
            if (!options.resolution_specified) {
                const auto [source_width, source_height] = source_dimensions();
                if (!dimensions_supported(source_width, source_height)) {
                    throw std::runtime_error(
                        "input source dimensions are outside the supported range");
                }

                render_width = source_width;
                render_height = source_height;
                options.width = render_width;
                options.height = render_height;
                const char *source_name = source_kind == SourceKind::Video
                                              ? "video"
                                          : source_kind == SourceKind::Camera
                                              ? "camera"
                                              : "graphic";
                std::cout << "acmxvk: automatic output resolution: "
                          << render_width << 'x' << render_height << " from "
                          << source_name;
                if (rotationSwapsDimensions(options.frame_rotation)) {
                    std::cout << " after input rotation";
                }
                std::cout << '\n';
            } else {
                std::cout << "acmxvk: requested output resolution: "
                          << render_width << 'x' << render_height << '\n';
            }
            setRenderExtent(static_cast<std::uint32_t>(render_width),
                            static_cast<std::uint32_t>(render_height));

            if (options.fullscreen) {
                std::cout << "acmxvk: fullscreen presentation uses the display "
                             "extent without changing the output resolution\n";
                return;
            }

            SDL_Window *window = getSDLWindow();
            if (window == nullptr) {
                throw std::runtime_error(
                    "unable to configure preview without an SDL window");
            }

            int preview_width = render_width;
            int preview_height = render_height;
            SDL_Rect usable_bounds{};
            SDL_DisplayID display = SDL_GetDisplayForWindow(window);
            if (display == 0) {
                display = SDL_GetPrimaryDisplay();
            }
            if (display != 0 &&
                SDL_GetDisplayUsableBounds(display, &usable_bounds) &&
                usable_bounds.w > 0 && usable_bounds.h > 0) {
                constexpr double PREVIEW_DISPLAY_FRACTION = 0.9;
                const double width_scale =
                    (static_cast<double>(usable_bounds.w) *
                     PREVIEW_DISPLAY_FRACTION) /
                    render_width;
                const double height_scale =
                    (static_cast<double>(usable_bounds.h) *
                     PREVIEW_DISPLAY_FRACTION) /
                    render_height;
                const double preview_scale =
                    std::min({1.0, width_scale, height_scale});
                preview_width = std::max(
                    1, static_cast<int>(std::lround(render_width * preview_scale)));
                preview_height = std::max(
                    1, static_cast<int>(std::lround(render_height * preview_scale)));
            }

            const float render_aspect = static_cast<float>(render_width) /
                                        static_cast<float>(render_height);
            if (!SDL_SetWindowAspectRatio(window, render_aspect,
                                          render_aspect)) {
                std::cerr << "acmxvk: unable to lock preview aspect ratio: "
                          << SDL_GetError() << '\n';
            }
            if (!SDL_SetWindowSize(window, preview_width, preview_height)) {
                throw std::runtime_error(
                    std::string("unable to apply preview resolution: ") +
                    SDL_GetError());
            }
            SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED);
            if (!SDL_SyncWindow(window)) {
                std::cerr << "acmxvk: window resize sync warning: "
                          << SDL_GetError() << '\n';
            }

            int actual_width = 0;
            int actual_height = 0;
            SDL_GetWindowSizeInPixels(window, &actual_width, &actual_height);
            std::cout << "acmxvk: preview resolution: " << actual_width << 'x'
                      << actual_height;
            if (preview_width != render_width ||
                preview_height != render_height) {
                std::cout << " (" << render_width << 'x' << render_height
                          << " output, preview fitted to display)";
            }
            std::cout << '\n';
        }

        [[nodiscard]] double outputFrameRate() {
            if (options.requested_fps > 0.0) {
                return options.requested_fps;
            }
            if (source_kind != SourceKind::Graphic) {
                double source_fps = 0.0;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                if (using_ffmpeg_capture) {
                    source_fps = ffmpeg_capture.fps();
                } else
#endif
                {
                    source_fps = capture.get(cv::CAP_PROP_FPS);
                }
                if (std::isfinite(source_fps) && source_fps > 0.0) {
                    return source_fps;
                }
            }
            return 30.0;
        }

        [[nodiscard]] static fs::path outputFrameDirectory(const std::string &filename,
                                                           std::string_view suffix) {
            const fs::path output_path(filename);
            const fs::path parent = output_path.has_parent_path()
                                        ? output_path.parent_path()
                                        : fs::path(".");
            const std::string name = output_path.filename().empty()
                                         ? std::string("output")
                                         : output_path.filename().string();
            return parent / ("video_file-" + name + "-" + std::string(suffix));
        }

        static void createOutputDirectory(const fs::path &directory) {
            std::error_code error;
            fs::create_directories(directory, error);
            if (error || !fs::is_directory(directory)) {
                throw std::runtime_error("unable to create PNG output directory: " +
                                         directory.string());
            }
        }

        [[nodiscard]] static fs::path framePath(const fs::path &directory,
                                                std::uint64_t index) {
            std::ostringstream filename;
            filename << "frame-" << std::setfill('0') << std::setw(8) << index << ".png";
            return directory / filename.str();
        }

        static void savePng(const fs::path &path, std::uint8_t *rgba, int width,
                            int height) {
            if (!mxvk::SavePNG_RGBA(path.string().c_str(), rgba, width, height)) {
                throw std::runtime_error("unable to write PNG frame: " + path.string());
            }
        }

        static void saveRaw(const fs::path &path,
                            const std::vector<std::uint8_t> &rgba,
                            std::uint32_t width, std::uint32_t height) {
            if (width == 0U || height == 0U) {
                throw std::runtime_error(
                    "invalid image dimensions for raw RGBA snapshot: " +
                    path.string());
            }

            const std::uint64_t byte_count =
                static_cast<std::uint64_t>(width) *
                static_cast<std::uint64_t>(height) * 4U;
            if (byte_count > rgba.size() ||
                byte_count > static_cast<std::uint64_t>(
                                 std::numeric_limits<std::streamsize>::max())) {
                throw std::runtime_error(
                    "invalid pixel buffer for raw RGBA snapshot: " +
                    path.string());
            }

            std::ofstream output(path, std::ios::binary);
            if (!output) {
                throw std::runtime_error("unable to open raw RGBA snapshot: " +
                                         path.string());
            }
            output.write(reinterpret_cast<const char *>(rgba.data()),
                         static_cast<std::streamsize>(byte_count));
            if (!output) {
                throw std::runtime_error("unable to write raw RGBA snapshot: " +
                                         path.string());
            }
        }

#ifdef ACMXVK_WITH_WEBP
        static void saveWebP(const fs::path &path, const std::uint8_t *rgba,
                             int width, int height) {
            if (rgba == nullptr || width <= 0 || height <= 0 ||
                width > std::numeric_limits<int>::max() / 4) {
                throw std::runtime_error(
                    "invalid image dimensions for WebP snapshot: " +
                    path.string());
            }

            std::uint8_t *encoded_pixels = nullptr;
            const std::size_t encoded_size = WebPEncodeLosslessRGBA(
                rgba, width, height, width * 4, &encoded_pixels);
            const std::unique_ptr<std::uint8_t, decltype(&WebPFree)>
                encoded_data(encoded_pixels, &WebPFree);
            if (encoded_size == 0 || encoded_data == nullptr) {
                throw std::runtime_error("unable to encode WebP snapshot: " +
                                         path.string());
            }

            std::ofstream output(path, std::ios::binary);
            if (!output) {
                throw std::runtime_error("unable to open WebP snapshot: " +
                                         path.string());
            }
            output.write(reinterpret_cast<const char *>(encoded_data.get()),
                         static_cast<std::streamsize>(encoded_size));
            if (!output) {
                throw std::runtime_error("unable to write WebP snapshot: " +
                                         path.string());
            }
        }
#endif

#ifdef ACMXVK_WITH_TIFF
        static void saveTiff(const fs::path &path, const std::uint8_t *rgba,
                             int width, int height) {
            if (rgba == nullptr || width <= 0 || height <= 0 ||
                width > std::numeric_limits<int>::max() / 4) {
                throw std::runtime_error(
                    "invalid image dimensions for TIFF snapshot: " +
                    path.string());
            }

            const std::unique_ptr<TIFF, decltype(&TIFFClose)> output(
                TIFFOpen(path.string().c_str(), "w"), &TIFFClose);
            if (output == nullptr) {
                throw std::runtime_error("unable to open TIFF snapshot: " +
                                         path.string());
            }

            const std::uint16_t extra_sample = EXTRASAMPLE_UNASSALPHA;
            const bool configured =
                TIFFSetField(output.get(), TIFFTAG_IMAGEWIDTH,
                             static_cast<std::uint32_t>(width)) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_IMAGELENGTH,
                             static_cast<std::uint32_t>(height)) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_SAMPLESPERPIXEL, 4) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_BITSPERSAMPLE, 8) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_ORIENTATION,
                             ORIENTATION_TOPLEFT) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_PLANARCONFIG,
                             PLANARCONFIG_CONTIG) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_PHOTOMETRIC,
                             PHOTOMETRIC_RGB) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_SAMPLEFORMAT,
                             SAMPLEFORMAT_UINT) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_COMPRESSION,
                             COMPRESSION_LZW) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_ROWSPERSTRIP,
                             TIFFDefaultStripSize(output.get(), 0)) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_EXTRASAMPLES, 1,
                             &extra_sample) != 0 &&
                TIFFSetField(output.get(), TIFFTAG_IMAGEDESCRIPTION,
                             "ACMXVK processed snapshot: 8-bit RGBA TIFF") != 0;
            if (!configured) {
                throw std::runtime_error(
                    "unable to configure TIFF snapshot: " + path.string());
            }

            const std::size_t row_bytes = static_cast<std::size_t>(width) * 4U;
            for (int row = 0; row < height; ++row) {
                auto *row_pixels = const_cast<std::uint8_t *>(
                    rgba + static_cast<std::size_t>(row) * row_bytes);
                if (TIFFWriteScanline(output.get(), row_pixels,
                                      static_cast<std::uint32_t>(row), 0) < 0) {
                    throw std::runtime_error(
                        "unable to write TIFF snapshot: " + path.string());
                }
            }
        }
#endif

        [[nodiscard]] static std::string_view
        snapshotFormatName(SnapshotFormat format) {
            switch (format) {
            case SnapshotFormat::WebP:
                return "WebP";
            case SnapshotFormat::Tiff:
                return "TIFF";
            case SnapshotFormat::Raw:
                return "raw RGBA";
            case SnapshotFormat::Png:
                return "PNG";
            }
            return "snapshot";
        }

        [[nodiscard]] static std::string_view
        snapshotExtension(SnapshotFormat format) {
            switch (format) {
            case SnapshotFormat::WebP:
                return ".webp";
            case SnapshotFormat::Tiff:
                return ".tiff";
            case SnapshotFormat::Raw:
                return ".raw";
            case SnapshotFormat::Png:
                return ".png";
            }
            return ".snapshot";
        }

        void snapshotWorkerLoop() noexcept {
            while (true) {
                SnapshotJob job;
                {
                    std::unique_lock<std::mutex> lock(snapshot_mutex);
                    snapshot_condition.wait(lock, [&] {
                        return snapshot_worker_stopping ||
                               !snapshot_jobs.empty();
                    });
                    if (snapshot_worker_stopping && snapshot_jobs.empty()) {
                        return;
                    }
                    job = std::move(snapshot_jobs.front());
                    snapshot_jobs.pop_front();
                }

                try {
                    if (job.format == SnapshotFormat::Raw) {
                        saveRaw(job.path, job.rgba, job.width, job.height);
                    } else if (job.format == SnapshotFormat::Tiff) {
#ifdef ACMXVK_WITH_TIFF
                        saveTiff(job.path, job.rgba.data(),
                                 static_cast<int>(job.width),
                                 static_cast<int>(job.height));
#else
                        throw std::runtime_error(
                            "TIFF snapshot support is not compiled in");
#endif
                    } else if (job.format == SnapshotFormat::WebP) {
#ifdef ACMXVK_WITH_WEBP
                        saveWebP(job.path, job.rgba.data(),
                                 static_cast<int>(job.width),
                                 static_cast<int>(job.height));
#else
                        throw std::runtime_error(
                            "WebP snapshot support is not compiled in");
#endif
                    } else {
                        savePng(job.path, job.rgba.data(),
                                static_cast<int>(job.width),
                                static_cast<int>(job.height));
                    }
                    std::ostringstream message;
                    message << "acmxvk: took "
                            << snapshotFormatName(job.format) << " snapshot: "
                            << job.path.string() << '\n';
                    std::cout << message.str();
                } catch (const std::exception &error) {
                    std::ostringstream message;
                    message << "acmxvk: snapshot failed: " << error.what()
                            << '\n';
                    std::cerr << message.str();
                } catch (...) {
                    std::cerr << "acmxvk: snapshot failed with an unknown error\n";
                }

                std::lock_guard<std::mutex> lock(snapshot_mutex);
                if (snapshot_jobs_in_flight > 0) {
                    --snapshot_jobs_in_flight;
                }
            }
        }

        [[nodiscard]] bool startSnapshotWorker() {
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            if (snapshot_worker.joinable()) {
                return true;
            }
            snapshot_worker_stopping = false;
            try {
                snapshot_worker =
                    std::thread(&MainWindow::snapshotWorkerLoop, this);
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: could not start snapshot worker: "
                          << error.what() << '\n';
                return false;
            }
            return true;
        }

        void stopSnapshotWorker() noexcept {
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                if (!snapshot_worker.joinable()) {
                    return;
                }
                snapshot_worker_stopping = true;
            }
            snapshot_condition.notify_one();
            snapshot_worker.join();
        }

        [[nodiscard]] bool snapshotQueueFull() {
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            return snapshot_jobs_in_flight >= SNAPSHOT_QUEUE_CAPACITY;
        }

        void enqueueSnapshot(SnapshotJob job) {
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                snapshot_jobs.push_back(std::move(job));
                ++snapshot_jobs_in_flight;
            }
            snapshot_condition.notify_one();
        }

        [[nodiscard]] fs::path snapshotPath(std::uint32_t width,
                                            std::uint32_t height,
                                            SnapshotFormat format) {
            const auto now = std::chrono::system_clock::now();
            const std::time_t now_time =
                std::chrono::system_clock::to_time_t(now);
            std::tm local_time{};
#ifdef _WIN32
            localtime_s(&local_time, &now_time);
#else
            localtime_r(&now_time, &local_time);
#endif
            const fs::path directory(options.snapshot_directory);
            while (true) {
                std::ostringstream filename;
                filename << "ACMXVK.Snapshot-"
                         << std::put_time(&local_time, "%Y.%m.%d-%H.%M.%S")
                         << '-' << width << 'x' << height << '-'
                         << snapshot_count << snapshotExtension(format);
                const fs::path candidate = directory / filename.str();
                if (!fs::exists(candidate)) {
                    return candidate;
                }
                ++snapshot_count;
            }
        }

        void requestSnapshot(SnapshotFormat format) {
#ifndef ACMXVK_WITH_TIFF
            if (format == SnapshotFormat::Tiff) {
                std::cerr << "acmxvk: TIFF snapshots require a build configured "
                             "with -DTIFF=ON\n";
                return;
            }
#endif
#ifndef ACMXVK_WITH_WEBP
            if (format == SnapshotFormat::WebP) {
                std::cerr << "acmxvk: WebP snapshots require a build configured "
                             "with -DWEBP=ON\n";
                return;
            }
#endif
            if (snapshot_pending) {
                return;
            }
            if (snapshotQueueFull()) {
                std::cerr << "acmxvk: snapshot queue is full; request ignored\n";
                return;
            }
            std::error_code error;
            const fs::path directory(options.snapshot_directory);
            fs::create_directories(directory, error);
            if (error || !fs::is_directory(directory)) {
                std::cerr << "acmxvk: unable to create snapshot directory: "
                          << directory.string() << '\n';
                return;
            }
            if (!startSnapshotWorker()) {
                return;
            }
            snapshot_pending = true;
            pending_snapshot_format = format;
            setFrameReadbackEnabled(true);
            std::cout << "acmxvk: " << snapshotFormatName(format)
                      << " snapshot requested\n";
        }

        [[nodiscard]] bool continuousReadbackEnabled() const {
            return writer.is_open() || options.png_output ||
                   options.generate_interval > 0;
        }

        void openOutput() {
            if (options.output_file.empty() && options.generate_interval <= 0) {
                return;
            }

            const VkExtent2D extent = getRenderExtent();
            if (options.resolution_specified) {
                recording_width = extent.width > 0U
                                      ? static_cast<int>(extent.width)
                                      : options.width;
                recording_height = extent.height > 0U
                                       ? static_cast<int>(extent.height)
                                       : options.height;
            } else {
                recording_width = options.width;
                recording_height = options.height;
            }
            recording_fps = outputFrameRate();

            if (options.png_output) {
                png_output_directory = outputFrameDirectory(options.output_file, "png");
                createOutputDirectory(png_output_directory);
                std::cout << "acmxvk: writing PNG sequence to "
                          << png_output_directory.string() << '\n';
            }

            if (options.generate_interval > 0) {
                if (!options.output_file.empty()) {
                    generate_output_directory =
                        outputFrameDirectory(options.output_file, "generate");
                } else if (!options.input_file.empty()) {
                    generate_output_directory =
                        outputFrameDirectory(options.input_file, "generate");
                } else {
                    generate_output_directory = "camera-generate";
                }
                createOutputDirectory(generate_output_directory);
                std::cout << "acmxvk: saving every " << options.generate_interval
                          << "th frame to " << generate_output_directory.string() << '\n';
            }

            if (!options.output_file.empty() && !options.png_output) {
                EncodeOptions encode_options;
                encode_options.preset = options.encode_preset;
                encode_options.tune = options.encode_tune;
                encode_options.crf = options.encode_crf;
                encode_options.codec = options.encode_codec;
                encode_options.ffmpeg_options = options.encode_params;
                encode_options.realtime = options.encode_realtime;
                encode_options.block_when_full = options.no_drop;

                if (!writer.open(options.output_file, recording_width, recording_height,
                                 static_cast<float>(recording_fps), encode_options)) {
                    throw std::runtime_error("unable to open output video: " +
                                             options.output_file);
                }
                writer.set_block_when_full(options.no_drop);
                std::cout << "acmxvk: recording " << recording_width << 'x'
                          << recording_height << " at " << recording_fps << " FPS to "
                          << options.output_file
                          << (options.no_drop ? " (no-drop)\n" : "\n");
                if (options.mute_output) {
                    std::cout
                        << "acmxvk: recorded video audio disabled (--mute-output); "
                           "reactivity and pass-through remain active\n";
                }
            }

            setFrameReadbackEnabled(true);
        }

        void onFrameReadbackScheduled() override {
            ReadbackRequest request;
            request.snapshot = snapshot_pending;
            request.snapshot_format = pending_snapshot_format;
            request.continuous = continuousReadbackEnabled();
            request.frame_due = recording_frame_due;
            request.has_pts = recording_frame_has_pts;
            request.pts = recording_frame_pts;
            readback_requests.push_back(request);

            if (snapshot_pending) {
                snapshot_pending = false;
                if (!request.continuous) {
                    setFrameReadbackEnabled(false);
                }
            }
        }

        void onFrameReadback(std::vector<std::uint8_t> &rgba, uint32_t width,
                             uint32_t height) override {
            if (readback_requests.empty()) {
                std::cerr << "acmxvk: received frame readback without queued metadata\n";
                return;
            }
            const ReadbackRequest request = readback_requests.front();
            readback_requests.pop_front();

            if (request.snapshot) {
                const fs::path path =
                    snapshotPath(width, height, request.snapshot_format);
                SnapshotJob job;
                job.path = path;
                job.width = width;
                job.height = height;
                job.format = request.snapshot_format;
                if (request.continuous) {
                    job.rgba = rgba;
                } else {
                    job.rgba = std::move(rgba);
                }
                enqueueSnapshot(std::move(job));
                ++snapshot_count;
                std::cout << "acmxvk: queued "
                          << snapshotFormatName(request.snapshot_format)
                          << " snapshot: " << path.string() << '\n';
            }

            if (!request.continuous || recording_complete ||
                !request.frame_due) {
                return;
            }

            std::uint8_t *output_pixels = rgba.data();
            cv::Mat resized;
            if (static_cast<int>(width) != recording_width ||
                static_cast<int>(height) != recording_height) {
                const cv::Mat source(static_cast<int>(height), static_cast<int>(width),
                                     CV_8UC4, rgba.data());
                cv::resize(source, resized, cv::Size(recording_width, recording_height),
                           0.0, 0.0, cv::INTER_LINEAR);
                output_pixels = resized.ptr();
            }

            if (writer.is_open()) {
                if (request.has_pts) {
                    writer.write_at_pts(
                        output_pixels,
                        static_cast<std::int64_t>(request.pts));
                } else {
                    writer.write(output_pixels);
                }
            }
            if (options.png_output) {
                savePng(framePath(png_output_directory, png_frame_count), output_pixels,
                        recording_width, recording_height);
                ++png_frame_count;
            }
            if (options.generate_interval > 0 &&
                (request.has_pts ? request.pts : output_frame_count) %
                        static_cast<std::uint64_t>(options.generate_interval) ==
                    0) {
                savePng(framePath(generate_output_directory, generated_frame_count),
                        output_pixels, recording_width, recording_height);
                ++generated_frame_count;
            }
            ++output_frame_count;

            if (options.duration > 0.0) {
                double output_duration = 0.0;
                if (request.has_pts) {
                    output_duration =
                        static_cast<double>(request.pts + 1) / recording_fps;
                } else if (writer.is_open()) {
                    output_duration = writer.get_duration();
                } else {
                    output_duration =
                        static_cast<double>(output_frame_count) / recording_fps;
                }
                if (output_duration >= options.duration) {
                    recording_complete = true;
                    exit();
                }
            }

            if (options.max_size_mb > 0.0 && writer.is_open()) {
                const double maximum_bytes = options.max_size_mb * 1024.0 * 1024.0;
                if (static_cast<double>(writer.get_bytes_written()) >=
                    maximum_bytes) {
                    std::cout << "acmxvk: maximum output size reached ("
                              << options.max_size_mb << " MB)\n";
                    recording_complete = true;
                    exit();
                }
            }
        }

        void initializeModel() {
            if (!options.enable_3d || model_initialized) {
                return;
            }

            try {
                input_model.enableExtendedFragmentUniforms();
                input_model.load(this, options.model_file, "", "", 1.0F);
                input_model.setShaders(
                    this, modelVertexShader().string(),
                    modelFragmentShader().string());
                model_effect_shader = modelFragmentShader();
                input_model.setBackfaceCulling(false);
                model_initialized = true;
                model_3d_active = true;
                model_last_render_time = std::chrono::steady_clock::now();
                std::cout << "acmxvk: loaded 3D model: "
                          << options.model_file << " ("
                          << input_model.model().vertices().size()
                          << " vertices, "
                          << input_model.model().indexCount()
                          << " indices; skybox camera centered; view rotation "
                          << (model_auto_rotate ? "enabled" : "disabled")
                          << ")\n";
            } catch (...) {
                if (getDevice() != VK_NULL_HANDLE) {
                    vkDeviceWaitIdle(getDevice());
                    input_model.cleanup(this);
                }
                throw;
            }
        }

        void initializeSprite() {
            if (!ensureRenderResources()) {
                throw std::runtime_error("MXVK failed to initialize render resources");
            }

            const auto [source_width, source_height] = source_dimensions();

            if (frame_sprite == nullptr) {
                frame_sprite = createSprite(source_width, source_height);
            }
            frame_sprite->enableExtendedUBO();
            frame_sprite->setCustomUniforms(custom_uniform_values);
            if (spectrumTextureEnabledForShaders()) {
                frame_sprite->enableSpectrumTexture(spectrumBinCount());
            }
            if (spectrumHistoryEnabledForShaders()) {
                frame_sprite->enableSpectrumHistoryTexture(
                    spectrumBinCount(),
                    static_cast<std::uint32_t>(options.audio_buffers));
            }
            if (historyCacheEnabled()) {
                frame_sprite->enableHistoryTexture(source_width, source_height,
                                                   static_cast<uint32_t>(
                                                       options.texture_cache_size));
            }
            frame_sprite->createEmptySprite(
                source_width, source_height, spriteVertexShader(),
                options.history_test ? echoCacheShader() : std::string{});

            if (options.human_background &&
                human_overlay_sprite == nullptr) {
                human_overlay_sprite = createSprite(1, 1);
                human_overlay_sprite->enableHistoryTexture(
                    static_cast<std::uint32_t>(source_width),
                    static_cast<std::uint32_t>(source_height), 1U);
                const cv::Mat transparent(source_height, source_width,
                                          CV_8UC4, cv::Scalar::all(0));
                human_overlay_sprite->updateHistoryTexture(
                    transparent.ptr(), transparent.cols, transparent.rows,
                    static_cast<int>(transparent.step));
            }

            initializeModel();

            if (source_kind == SourceKind::Graphic) {
                initial_frame_pending = false;
                uploadInputFrame(graphic_rgba);
                updateHumanOverlayTexture();
                initializeHistory(graphic_rgba);
            } else if (!readTrackedInputFrame()) {
                std::cerr << "acmxvk: capture did not provide an initial frame\n";
            } else {
                initial_frame_pending = true;
            }

            applyShaderPipeline();
            if (!currentShader().empty()) {
                std::cout << "acmxvk: " << activeShaderRole() << ' '
                          << (shader_index + 1) << '/' << shaders.size()
                          << ": " << currentShader() << '\n';
            }
        }

        void resetShaderTime() {
            previous_frame = std::chrono::steady_clock::now();
            previous_video_shader_timeline = 0.0;
            video_shader_timeline_initialized = false;
            shader_time = 0.0;
            frame_count = 0;
        }

        void beginCrossfade() {
            if (options.cross_fade_duration <= 0.0 || frame_count == 0 ||
                getDevice() == VK_NULL_HANDLE) {
                crossfade_active = false;
                crossfade_alpha = 1.0F;
                crossfade_uses_video_timeline = false;
                return;
            }

            try {
                std::vector<std::uint8_t> captured;
                std::uint32_t captured_width = 0;
                std::uint32_t captured_height = 0;
                captureSnapshotPixels(captured, captured_width,
                                      captured_height);
                const VkExtent2D extent = getRenderExtent();
                if (captured.empty() || captured_width == 0U ||
                    captured_height == 0U || extent.width == 0U ||
                    extent.height == 0U) {
                    throw std::runtime_error(
                        "the previous rendered frame is unavailable");
                }

                cv::Mat captured_rgba(static_cast<int>(captured_height),
                                      static_cast<int>(captured_width),
                                      CV_8UC4, captured.data());
                cv::Mat previous_rgba;
                if (captured_width == extent.width &&
                    captured_height == extent.height) {
                    previous_rgba = captured_rgba;
                } else {
                    const double captured_aspect =
                        static_cast<double>(captured_width) / captured_height;
                    const double target_aspect =
                        static_cast<double>(extent.width) / extent.height;
                    cv::Rect crop(0, 0, static_cast<int>(captured_width),
                                  static_cast<int>(captured_height));
                    if (captured_aspect > target_aspect) {
                        crop.width = std::max(
                            1, static_cast<int>(std::lround(
                                   captured_height * target_aspect)));
                        crop.x =
                            (static_cast<int>(captured_width) - crop.width) / 2;
                    } else if (captured_aspect < target_aspect) {
                        crop.height = std::max(
                            1, static_cast<int>(std::lround(
                                   captured_width / target_aspect)));
                        crop.y = (static_cast<int>(captured_height) -
                                  crop.height) /
                                 2;
                    }
                    cv::resize(captured_rgba(crop), previous_rgba,
                               cv::Size(static_cast<int>(extent.width),
                                        static_cast<int>(extent.height)),
                               0.0, 0.0, cv::INTER_LINEAR);
                }

                if (crossfade_previous_sprite == nullptr) {
                    crossfade_previous_sprite = createSprite(1, 1);
                }
                crossfade_previous_sprite->enableHistoryTexture(
                    extent.width, extent.height, 1U);
                crossfade_previous_sprite->updateHistoryTexture(
                    previous_rgba.ptr(), static_cast<int>(extent.width),
                    static_cast<int>(extent.height),
                    static_cast<int>(previous_rgba.step));
                crossfade_alpha = 0.0F;
                crossfade_active = true;
                crossfade_start_time = std::chrono::steady_clock::now();
                crossfade_uses_video_timeline = currentVideoTimeline(
                    crossfade_start_video_timeline);
            } catch (const std::exception &error) {
                crossfade_active = false;
                crossfade_alpha = 1.0F;
                crossfade_uses_video_timeline = false;
                std::cerr << "acmxvk: crossfade snapshot unavailable: "
                          << error.what() << "; switching immediately\n";
            }
        }

        void updateCrossfade(const std::chrono::steady_clock::time_point now) {
            if (!crossfade_active) {
                return;
            }
            double elapsed = 0.0;
            double video_timeline = 0.0;
            if (crossfade_uses_video_timeline &&
                currentVideoTimeline(video_timeline)) {
                if (video_timeline < crossfade_start_video_timeline) {
                    crossfade_start_video_timeline = video_timeline;
                }
                elapsed = video_timeline - crossfade_start_video_timeline;
            } else {
                elapsed = std::chrono::duration<double>(
                              now - crossfade_start_time)
                              .count();
            }
            crossfade_alpha = static_cast<float>(std::clamp(
                elapsed / options.cross_fade_duration, 0.0, 1.0));
            if (crossfade_alpha >= 1.0F) {
                crossfade_active = false;
                crossfade_uses_video_timeline = false;
                applyShaderPipeline();
            }
        }

        void cycleCrossfade(int direction) {
            const auto count =
                static_cast<std::ptrdiff_t>(CROSSFADE_NAMES.size());
            auto index =
                static_cast<std::ptrdiff_t>(crossfade_shader_index) + direction;
            index = (index % count + count) % count;
            crossfade_shader_index = static_cast<std::size_t>(index);
            std::cout << "acmxvk: crossfade shader: "
                      << CROSSFADE_NAMES[crossfade_shader_index] << " ("
                      << (crossfade_shader_index + 1) << '/'
                      << CROSSFADE_NAMES.size() << ")\n";
        }

        void adjustModelScale(float amount) {
            if (!model_initialized || model_scale_oscillation_active) {
                return;
            }
            model_scale = std::clamp(model_scale + amount, 0.05F, 20.0F);
            std::cout << "acmxvk: model scale " << model_scale << '\n';
        }

        void maybeRandomizeCrossfade() {
            if (!autopilot_random_crossfade || CROSSFADE_NAMES.empty()) {
                return;
            }
            std::uniform_int_distribution<std::size_t> distribution(
                0, CROSSFADE_NAMES.size() - 1);
            std::size_t next = distribution(autopilot_rng);
            if (CROSSFADE_NAMES.size() > 1 &&
                next == crossfade_shader_index) {
                next = (next + 1) % CROSSFADE_NAMES.size();
            }
            crossfade_shader_index = next;
        }

        void togglePause() {
            if (source_kind == SourceKind::Camera) {
                std::cout << "acmxvk: pause is available for video and graphic input\n";
                return;
            }
            input_paused = !input_paused;
            setSourcePlaybackClockPaused(input_paused || rendering_frozen);
            std::cout << "acmxvk: input pause "
                      << (input_paused ? "enabled" : "disabled") << '\n';
        }

        void toggleFreeze() {
            if (source_kind == SourceKind::Camera) {
                std::cout << "acmxvk: freeze is available for video and graphic input\n";
                return;
            }
            rendering_frozen = !rendering_frozen;
            setSourcePlaybackClockPaused(input_paused || rendering_frozen);
            previous_frame = std::chrono::steady_clock::now();
            std::cout << "acmxvk: rendering freeze "
                      << (rendering_frozen ? "enabled" : "disabled") << '\n';
        }

        void stepShaderTime(double amount) {
            shader_time += amount;
            std::cout << "acmxvk: shader time stepped to " << shader_time << '\n';
        }

        void adjustTimeSpeed(double amount) {
            options.time_speed += amount;
            if (std::abs(options.time_speed) < 0.01) {
                options.time_speed = 0.0;
            }
            std::cout << "acmxvk: shader time speed " << options.time_speed << '\n';
        }

        void toggleFullscreen() {
            SDL_Window *window = getSDLWindow();
            if (window == nullptr) {
                return;
            }
            const bool fullscreen =
                (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
            if (!SDL_SetWindowFullscreen(window, !fullscreen)) {
                std::cerr << "acmxvk: unable to toggle fullscreen: "
                          << SDL_GetError() << '\n';
                return;
            }
            std::cout << "acmxvk: fullscreen "
                      << (!fullscreen ? "enabled" : "disabled") << '\n';
        }

        void resetAutopilotInterval() {
            if (options.autopilot_random_timeout > 0) {
                std::uniform_int_distribution<int> distribution(
                    4, std::max(4, options.autopilot_random_timeout));
                autopilot_interval_frames = distribution(autopilot_rng);
            } else {
                autopilot_interval_frames = options.autopilot_frames;
            }
        }

        void logSelectedPlaylistNode(std::string_view action) const {
            if (playlist.empty()) {
                return;
            }
            std::cout << "acmxvk: " << action << " playlist node "
                      << (playlist_index + 1) << '/' << playlist.size() << ": "
                      << playlist[playlist_index].name << " ("
                      << playlist[playlist_index].shaders.size()
                      << " passes)\n";
        }

        [[nodiscard]] std::uint64_t autopilotFrameAdvance() {
            double video_timeline = 0.0;
            std::uint64_t video_frame_index = 0U;
            if (!currentVideoTimeline(video_timeline, &video_frame_index)) {
                autopilot_video_timeline_initialized = false;
                return 1U;
            }

            if (!autopilot_video_timeline_initialized ||
                video_frame_index < previous_autopilot_video_frame) {
                previous_autopilot_video_frame = video_frame_index;
                autopilot_video_timeline_initialized = true;
                return 1U;
            }

            const std::uint64_t advance =
                video_frame_index - previous_autopilot_video_frame;
            previous_autopilot_video_frame = video_frame_index;
            return advance;
        }

        void toggleAutopilot(bool sequential) {
            if (!playlist_enabled) {
                std::cout << "acmxvk: "
                          << (sequential ? "sequential autopilot" : "autopilot")
                          << " requires playlist mode (press P first)\n";
                return;
            }
            if (playlist.empty()) {
                std::cout << "acmxvk: autopilot has no playlist entries\n";
                return;
            }

            if (autopilot_enabled && autopilot_sequential == sequential) {
                autopilot_enabled = false;
                autopilot_sequential = false;
                std::cout << "acmxvk: autopilot disabled\n";
                return;
            }

            autopilot_enabled = true;
            autopilot_sequential = sequential;
            autopilot_counter = 0;
            autopilot_video_timeline_initialized = false;
            if (options.autopilot_random_timeout <= 0 && options.autopilot_frames <= 0) {
                options.autopilot_frames = 300;
            }
            resetAutopilotInterval();
            std::cout << "acmxvk: " << (sequential ? "sequential " : "random ")
                      << "autopilot enabled (";
            if (options.autopilot_random_timeout > 0) {
                std::cout << "random interval 4-" << options.autopilot_random_timeout
                          << ", current " << autopilot_interval_frames;
            } else {
                std::cout << "every " << autopilot_interval_frames << " frames";
            }
            std::cout << ")\n";
        }

        void updateAutopilot() {
            const std::uint64_t frame_advance = autopilotFrameAdvance();
            if (shader_locked || !autopilot_enabled || !playlist_enabled ||
                playlist.empty() || autopilot_interval_frames <= 0) {
                return;
            }
            const std::uint64_t remaining = static_cast<std::uint64_t>(
                std::max(0, autopilot_interval_frames - autopilot_counter));
            if (frame_advance < remaining) {
                autopilot_counter += static_cast<int>(frame_advance);
                return;
            }
            autopilot_counter = 0;

            maybeRandomizeCrossfade();
            beginCrossfade();
            if (autopilot_sequential && options.autopilot_random_timeout <= 0) {
                playlist_index = (playlist_index + 1) % playlist.size();
            } else {
                std::uniform_int_distribution<std::size_t> distribution(0,
                                                                        playlist.size() - 1);
                std::size_t next = distribution(autopilot_rng);
                if (playlist.size() > 1 && next == playlist_index) {
                    next = (next + 1) % playlist.size();
                }
                playlist_index = next;
            }

            applyShaderPipeline();
            resetShaderTime();
            if (options.autopilot_random_timeout > 0) {
                resetAutopilotInterval();
            }
            logSelectedPlaylistNode("autopilot selected");
        }

        void selectShader(int direction) {
            if (shader_locked || shaders.size() < 2 || frame_sprite == nullptr) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(shaders.size());
            beginCrossfade();
            auto index = static_cast<std::ptrdiff_t>(shader_index) + direction;
            index = (index % count + count) % count;
            shader_index = static_cast<std::size_t>(index);

            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            std::cout << "acmxvk: " << activeShaderRole() << ' '
                      << (shader_index + 1) << '/' << shaders.size() << ": "
                      << currentShader() << '\n';
        }

        void selectPlaylistNode(int direction) {
            if (shader_locked || playlist.empty()) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(playlist.size());
            beginCrossfade();
            auto index = static_cast<std::ptrdiff_t>(playlist_index) + direction;
            index = (index % count + count) % count;
            playlist_index = static_cast<std::size_t>(index);
            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            logSelectedPlaylistNode("selected");
        }

        [[nodiscard]] std::vector<fs::path> activeShaderPipeline() const {
            std::vector<fs::path> pipeline;
            if (effects_enabled) {
                if (playlist_enabled && !playlist.empty()) {
                    pipeline = playlist[playlist_index].shaders;
                } else if (multipass_enabled) {
                    pipeline = configured_passes;
                }
                if (!currentShader().empty()) {
                    pipeline.emplace_back(currentShader());
                }
            }
            if (options.flip_output) {
                pipeline.emplace_back(flipShader());
            }
            if (crossfade_active) {
                pipeline.emplace_back(crossfadeShader());
            }
            if (pipeline.empty()) {
                pipeline.emplace_back(passthroughShader());
            }
            if (options.human_background) {
                pipeline.emplace_back(humanCompositeShader());
            }
            return pipeline;
        }

        [[nodiscard]] fs::path directModelFragmentShader() const {
            if (!model_3d_active || !model_initialized || !effects_enabled ||
                playlist_enabled || multipass_enabled ||
                currentShader().empty()) {
                return {};
            }

            const fs::path shader(currentShader());
            const mxvk::ShaderModuleInfo module_info =
                mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
            if (module_info.stage != mxvk::ShaderStage::Fragment ||
                module_info.usesHistoryTexture ||
                module_info.usesSpectrumTexture ||
                module_info.usesSpectrumHistoryTexture) {
                return {};
            }
            return shader;
        }

        void applyShaderPipeline() {
            if (getDevice() == VK_NULL_HANDLE) {
                return;
            }
            vkDeviceWaitIdle(getDevice());
            detachPostProcessingShader();
            post_process_sprites.clear();
            frame_sprite->setEffectsEnabled(effects_enabled);

            const fs::path direct_model_shader = directModelFragmentShader();
            model_texture_prepass_active =
                model_3d_active && direct_model_shader.empty();
            setPostProcessingTextureConsumerEnabled(
                model_texture_prepass_active);
            if (model_initialized) {
                const fs::path desired_model_shader =
                    direct_model_shader.empty() ? modelFragmentShader()
                                                : direct_model_shader;
                if (desired_model_shader != model_effect_shader) {
                    input_model.setShaders(this, modelVertexShader().string(),
                                           desired_model_shader.string());
                    model_effect_shader = desired_model_shader;
                }
            }

            std::vector<fs::path> pipeline = activeShaderPipeline();
            if (!direct_model_shader.empty()) {
                const auto selected = std::find(
                    pipeline.begin(), pipeline.end(), direct_model_shader);
                if (selected != pipeline.end()) {
                    pipeline.erase(selected);
                }
                if (pipeline.empty()) {
                    pipeline.emplace_back(passthroughShader());
                }
                std::cout << "acmxvk: 3D texture effect: "
                          << direct_model_shader.filename().string()
                          << " [fragment, evaluated on model UVs]\n";
            } else if (model_3d_active && effects_enabled &&
                       !currentShader().empty()) {
                std::cout << "acmxvk: 3D texture prepass: fragment/compute "
                             "chain output mapped onto model UVs\n";
            }
            if (pipeline.empty()) {
                return;
            }

            std::vector<PostProcessingEffect> effects;
            effects.reserve(pipeline.size());
            crossfade_post_process_index =
                std::numeric_limits<std::size_t>::max();
            for (std::size_t index = 0; index < pipeline.size(); ++index) {
                const fs::path &shader = pipeline[index];
                PostProcessingEffect effect{
                    shader.string(), {1.0F, 1.0F, 1.0F, 0.0F}, false};
                if (crossfade_active && shader == crossfadeShader()) {
                    crossfade_post_process_index = index;
                    effect.historySource = crossfade_previous_sprite;
                    effect.params[0] = crossfade_alpha;
                } else if (options.human_background &&
                           shader == humanCompositeShader()) {
                    effect.historySource = human_overlay_sprite;
                } else if (historyCacheEnabled()) {
                    effect.historySource = frame_sprite;
                }
                if (spectrumTextureEnabledForShaders()) {
                    effect.spectrumBinCount = spectrumBinCount();
                }
                if (spectrumHistoryEnabledForShaders()) {
                    effect.spectrumHistoryLayerCount =
                        static_cast<std::uint32_t>(options.audio_buffers);
                }
                effects.push_back(effect);
            }
            post_process_sprites = attachPostProcessingShaders(effects);
            for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                sprite->enableExtendedUBO();
                sprite->setCustomUniforms(custom_uniform_values);
                if (spectrumTextureEnabledForShaders()) {
                    sprite->enableSpectrumTexture(spectrumBinCount());
                }
                if (spectrumHistoryEnabledForShaders()) {
                    sprite->enableSpectrumHistoryTexture(
                        spectrumBinCount(),
                        static_cast<std::uint32_t>(options.audio_buffers));
                }
            }

            std::cout << "acmxvk: Vulkan shader pipeline (" << pipeline.size() << " passes):\n";
            for (std::size_t index = 0; index < pipeline.size(); ++index) {
                const bool compute =
                    index < post_process_effect_stages.size() &&
                    post_process_effect_stages[index] ==
                        mxvk::ShaderStage::Compute;
                std::cout << "  " << (index + 1) << ": "
                          << pipeline[index].filename().string() << " ["
                          << (compute ? "compute" : "fragment") << "]\n";
            }
        }

        [[nodiscard]] bool readTrackedInputFrame() {
            if (!readInputFrame()) {
                return false;
            }
            if (source_kind == SourceKind::Video) {
                ++decoded_video_frame_count;
                ++video_source_frame_count;
            } else if (source_kind == SourceKind::Camera) {
                updateCameraFrameRate();
            }
            return true;
        }

        [[nodiscard]] bool skipInputFrame() {
            if (source_kind != SourceKind::Video) {
                return false;
            }
            bool skipped = false;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (using_ffmpeg_capture) {
                skipped = ffmpeg_capture.skip();
            } else
#endif
            {
                skipped = capture.grab();
            }
            if (skipped) {
                ++decoded_video_frame_count;
                ++video_source_frame_count;
            }
            return skipped;
        }

        [[nodiscard]] bool handleCaptureEnd(bool discard = false) {
            if (source_kind == SourceKind::Camera) {
                return true;
            }
            if (!options.repeat) {
                setFrameReadbackEnabled(false);
                exit();
                return false;
            }

#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (using_ffmpeg_capture && ffmpeg_capture.seek_start()) {
                video_source_frame_count = 0;
                const bool restarted =
                    discard ? skipInputFrame() : readTrackedInputFrame();
                if (restarted) {
                    if (!ffmpeg_seek_repeat_logged) {
                        std::cout
                            << "acmxvk: video repeat: in-place FFmpeg seek; "
                            << (ffmpeg_capture.using_hardware_decode()
                                    ? "NVDEC decoder and CUDA device preserved\n"
                                    : "software decoder preserved\n");
                        ffmpeg_seek_repeat_logged = true;
                    }
                    return true;
                }
                std::cerr << "acmxvk: in-place FFmpeg repeat did not produce a "
                             "frame; reopening the input\n";
            }
#endif
            closeVideoCapture();
            if (!openVideoCapture() ||
                !(discard ? skipInputFrame() : readTrackedInputFrame())) {
                throw std::runtime_error("unable to restart video input: " + options.input_file);
            }
            return true;
        }

        [[nodiscard]] bool readClockedVideoFrame(double clock_seconds) {
            const double rate = outputFrameRate();
            if (!std::isfinite(rate) || rate <= 0.0) {
                return readTrackedInputFrame();
            }

            std::uint64_t target_frame = static_cast<std::uint64_t>(
                std::floor(std::max(clock_seconds, 0.0) * rate));
            if (target_frame < decoded_video_frame_count) {
                const double next_frame_time =
                    static_cast<double>(decoded_video_frame_count) / rate;
                const double wait_seconds = next_frame_time - clock_seconds;
                if (wait_seconds > 0.0) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(wait_seconds));
                }
                double updated_clock = 0.0;
                if (mediaClockSeconds(updated_clock)) {
                    target_frame = static_cast<std::uint64_t>(
                        std::floor(std::max(updated_clock, 0.0) * rate));
                }
            }
            if (target_frame < decoded_video_frame_count) {
                return true;
            }

            const std::uint64_t frames_to_advance =
                target_frame - decoded_video_frame_count + 1;
            for (std::uint64_t frame = 0; frame < frames_to_advance; ++frame) {
                const bool discard = frame + 1 < frames_to_advance;
                bool advanced = discard ? skipInputFrame()
                                        : readTrackedInputFrame();
                if (!advanced) {
                    advanced = handleCaptureEnd(discard);
                }
                if (!advanced) {
                    return false;
                }
            }

            source_frame_received = true;
            recording_frame_due = true;
            recording_frame_has_pts = true;
            recording_frame_pts = decoded_video_frame_count - 1;
            if (!media_clock_sync_logged) {
                std::cout << "acmxvk: media-clock synchronization active; "
                             "late video frames will be skipped and encoded "
                             "with timeline PTS\n";
                media_clock_sync_logged = true;
            }
            return true;
        }

        void closeVideoCapture() {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.is_open()) {
                ffmpeg_capture.close();
            }
            using_ffmpeg_capture = false;
#endif
            if (capture.is_open()) {
                capture.close();
            }
        }

        [[nodiscard]] bool openVideoCapture() {
            video_source_frame_count = 0;
            video_source_fps = 0.0;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.open(options.input_file, options.cuda_device)) {
                using_ffmpeg_capture = true;
                video_source_fps = ffmpeg_capture.fps();
                std::cout << "acmxvk: video capture: FFmpeg ";
                if (ffmpeg_capture.using_hardware_decode()) {
                    std::cout << "with CUDA/NVDEC";
                    if (ffmpeg_capture.hardware_decode_device() >= 0) {
                        std::cout << " on device "
                                  << ffmpeg_capture.hardware_decode_device();
                    }
                    std::cout << '\n';
                } else {
                    std::cout << "software decode\n";
                }
                return true;
            }
#endif
            const bool opened = capture.open(options.input_file);
            if (opened) {
                video_source_fps = capture.get(cv::CAP_PROP_FPS);
                std::cout << "acmxvk: video capture: OpenCV fallback\n";
            }
            if (!std::isfinite(video_source_fps) || video_source_fps <= 0.0) {
                video_source_fps = 30.0;
            }
            return opened;
        }

        [[nodiscard]] bool readHostRgba(cv::Mat &rgba) {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (using_ffmpeg_capture) {
                int width = 0;
                int height = 0;
                int pitch = 0;
                if (!ffmpeg_capture.readRgba(ffmpeg_rgba, width, height, pitch,
                                             false) ||
                    ffmpeg_rgba.empty() || width <= 0 || height <= 0 ||
                    pitch < width * 4) {
                    return false;
                }
                rgba = cv::Mat(height, width, CV_8UC4, ffmpeg_rgba.data(),
                               static_cast<std::size_t>(pitch));
                return true;
            }
#endif
            return capture.readRgba(rgba, false);
        }

        void initializeHistory(const cv::Mat &rgba) {
            if (!historyCacheEnabled() || history_initialized) {
                return;
            }
            for (uint32_t layer = 0; layer < frame_sprite->getHistoryLayerCount(); ++layer) {
                updateHistoryFrame(rgba);
            }
            history_initialized = true;
            history_delay_counter = 0;
            camera_history_clock_started = false;
            std::cout << "acmxvk: initialized " << frame_sprite->getHistoryLayerCount()
                      << " Vulkan history-cache layers (delay " << options.cache_delay
                      << ")\n";
        }

        void updateHistoryFrame(const cv::Mat &rgba) {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                updateFilteredCudaHistoryFrame();
                return;
            }
#endif
            frame_sprite->updateHistoryTexture(
                rgba.ptr(), rgba.cols, rgba.rows,
                static_cast<int>(rgba.step));
        }

        void updateCameraHistory() {
            if (source_kind != SourceKind::Camera || rendering_frozen ||
                input_paused || !history_initialized) {
                return;
            }

            const double rate = outputFrameRate();
            if (!std::isfinite(rate) || rate <= 0.0) {
                return;
            }
            const auto interval = std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(std::chrono::duration<double>(
                static_cast<double>(options.cache_delay + 1) / rate));
            const auto now = std::chrono::steady_clock::now();
            if (!camera_history_clock_started) {
                camera_history_next_update = now + interval;
                camera_history_clock_started = true;
                return;
            }
            if (now < camera_history_next_update) {
                return;
            }

            bool history_updated = false;
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                updateFilteredCudaHistoryFrame();
                history_updated = true;
            }
#endif
            if (!history_updated && !latest_camera_history_rgba.empty()) {
                updateHistoryFrame(latest_camera_history_rgba);
                history_updated = true;
            }
            if (!history_updated) {
                return;
            }

            camera_history_next_update += interval;
            if (camera_history_next_update <= now) {
                camera_history_next_update = now + interval;
            }
        }

#ifdef ACMXVK_WITH_MXVK_CUDA
        void updateModelTextureCuda(const cv::cuda::GpuMat &rgba,
                                    cv::cuda::Stream &source_stream) {
            if (!model_initialized) {
                return;
            }
            if (input_model.updatePrimaryTextureCuda(rgba, source_stream)) {
                return;
            }

            rgba.download(cuda_model_fallback_rgba, source_stream);
            source_stream.waitForCompletion();
            if (!cuda_model_fallback_logged) {
                std::cerr << "acmxvk: direct CUDA model-texture upload "
                             "unavailable; using host staging\n";
                cuda_model_fallback_logged = true;
            }
            if (!input_model.updatePrimaryTexture(
                    cuda_model_fallback_rgba.ptr(),
                    cuda_model_fallback_rgba.cols,
                    cuda_model_fallback_rgba.rows,
                    static_cast<int>(cuda_model_fallback_rgba.step))) {
                throw std::runtime_error(
                    "MXVK could not update the 3D model texture");
            }
        }

        void updateCudaHistoryFrame(const cv::cuda::GpuMat &rgba,
                                    cv::cuda::Stream &source_stream) {
            if (frame_sprite->updateHistoryTextureCuda(rgba, source_stream)) {
                return;
            }

            rgba.download(cuda_history_fallback_rgba, source_stream);
            source_stream.waitForCompletion();
            if (!cuda_history_fallback_logged) {
                std::cerr << "acmxvk: direct CUDA history upload unavailable; "
                             "using a host-staging fallback\n";
                cuda_history_fallback_logged = true;
            }
            frame_sprite->updateHistoryTexture(
                cuda_history_fallback_rgba.ptr(),
                cuda_history_fallback_rgba.cols,
                cuda_history_fallback_rgba.rows,
                static_cast<int>(cuda_history_fallback_rgba.step));
        }

#ifdef ACMXVK_WITH_CUDA
        void updateFilteredCudaHistoryFrame() {
            updateCudaHistoryFrame(gpu_filter_engine->output(),
                                   gpu_filter_engine->stream());
        }
#endif

        void initializeCudaHistory(const cv::cuda::GpuMat &rgba,
                                   cv::cuda::Stream &source_stream,
                                   bool filtered) {
            if (!historyCacheEnabled() || history_initialized) {
                return;
            }
            for (uint32_t layer = 0;
                 layer < frame_sprite->getHistoryLayerCount(); ++layer) {
                updateCudaHistoryFrame(rgba, source_stream);
            }
            history_initialized = true;
            history_delay_counter = 0;
            camera_history_clock_started = false;
            std::cout << "acmxvk: initialized "
                      << frame_sprite->getHistoryLayerCount()
                      << (filtered ? " filtered" : " NVDEC")
                      << " Vulkan history-cache layers (delay "
                      << options.cache_delay << ")\n";
        }

#ifdef ACMXVK_WITH_CUDA
        void uploadInputFrame(const cv::cuda::GpuMat &rgba,
                              cv::cuda::Stream &source_stream) {
            if (!gpu_filter_engine->process(rgba, source_stream)) {
                throw std::runtime_error(
                    "acidcam-gpu rejected the CUDA RGBA input frame");
            }
            if (!frame_sprite->updateTextureCuda(
                    gpu_filter_engine->output(),
                    gpu_filter_engine->stream())) {
                throw std::runtime_error(
                    "MXVK could not upload the CUDA-filtered frame");
            }
            updateModelTextureCuda(gpu_filter_engine->output(),
                                   gpu_filter_engine->stream());
        }
#endif

        [[nodiscard]] const cv::cuda::GpuMat &
        rotateCudaFrame(const cv::cuda::GpuMat &rgba,
                        cv::cuda::Stream &source_stream) {
            switch (options.frame_rotation) {
            case FrameRotation::None:
                return rgba;
            case FrameRotation::Clockwise90:
                cv::cuda::transpose(rgba, cuda_rotation_transpose, source_stream);
                cv::cuda::flip(cuda_rotation_transpose, cuda_rotated_rgba, 1,
                               source_stream);
                break;
            case FrameRotation::Rotate180:
                cv::cuda::flip(rgba, cuda_rotated_rgba, -1, source_stream);
                break;
            case FrameRotation::Counterclockwise90:
                cv::cuda::transpose(rgba, cuda_rotation_transpose, source_stream);
                cv::cuda::flip(cuda_rotation_transpose, cuda_rotated_rgba, 0,
                               source_stream);
                break;
            }
            return cuda_rotated_rgba;
        }
#endif

        void uploadInputFrame(const cv::Mat &rgba) {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                if (!gpu_filter_engine->process(rgba)) {
                    throw std::runtime_error(
                        "acidcam-gpu rejected the RGBA input frame");
                }
                if (!frame_sprite->updateTextureCuda(
                        gpu_filter_engine->output(),
                        gpu_filter_engine->stream())) {
                    throw std::runtime_error(
                        "MXVK could not upload the CUDA-filtered frame");
                }
                updateModelTextureCuda(gpu_filter_engine->output(),
                                       gpu_filter_engine->stream());
                return;
            }
#endif
            frame_sprite->updateTexture(rgba.ptr(), rgba.cols, rgba.rows,
                                        static_cast<int>(rgba.step));
            if (model_initialized &&
                !input_model.updatePrimaryTexture(
                    rgba.ptr(), rgba.cols, rgba.rows,
                    static_cast<int>(rgba.step))) {
                throw std::runtime_error(
                    "MXVK could not update the 3D model texture");
            }
        }

        [[nodiscard]] bool readLatestCameraFrame() {
            cv::Mat bgr;
            const bool wait_for_first = !async_camera_frame_uploaded &&
                                        !async_camera_initial_wait_completed;
            async_camera_initial_wait_completed = true;
            if (!latest_camera_frame.takeLatest(bgr, wait_for_first)) {
                return false;
            }

            cv::Mat rgba;
            cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
            applyDnnEffects(rgba);
            rotateFrame(rgba, options.frame_rotation);
            if (!human_overlay_rgba.empty()) {
                rotateFrame(human_overlay_rgba, options.frame_rotation);
            }
            uploadInputFrame(rgba);
            updateHumanOverlayTexture();
            latest_camera_history_rgba = rgba;
            async_camera_frame_uploaded = true;

#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                initializeCudaHistory(gpu_filter_engine->output(),
                                      gpu_filter_engine->stream(), true);
                return true;
            }
#endif

            initializeHistory(rgba);
            return true;
        }

        [[nodiscard]] bool readInputFrame() {
            if (source_kind == SourceKind::Camera && options.maximize_fps) {
                return readLatestCameraFrame();
            }
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr && !dnnHostProcessingEnabled()) {
                cv::cuda::Stream *capture_stream = nullptr;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                if (using_ffmpeg_capture) {
                    if (!ffmpeg_capture.readGpuRgba(cuda_input_rgba,
                                                    ffmpeg_cuda_stream, false)) {
                        return false;
                    }
                    capture_stream = &ffmpeg_cuda_stream;
                } else
#endif
                {
                    if (!capture.readGpuRgba(cuda_input_rgba, false)) {
                        return false;
                    }
                    capture_stream = &capture.cudaStream();
                }
                const cv::cuda::GpuMat &filter_input =
                    rotateCudaFrame(cuda_input_rgba, *capture_stream);
                uploadInputFrame(filter_input, *capture_stream);
                if (!cuda_input_path_logged) {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                    if (using_ffmpeg_capture) {
                        std::cout << "acmxvk: CUDA input path active: FFmpeg "
                                  << (ffmpeg_capture.using_hardware_decode()
                                          ? "NVDEC -> CUDA RGBA -> "
                                          : "software decode -> CUDA upload -> ");
                    } else
#endif
                    {
                        std::cout
                            << "acmxvk: CUDA input path active: MXVK capture -> ";
                    }
                    if (options.frame_rotation != FrameRotation::None) {
                        std::cout << "CUDA rotation -> ";
                    }
                    std::cout
                        << "acidcam-gpu temporal buffer -> Vulkan texture\n";
                    cuda_input_path_logged = true;
                }
                const bool history_was_initialized = history_initialized;
                initializeCudaHistory(gpu_filter_engine->output(),
                                      gpu_filter_engine->stream(), true);
                if (source_kind != SourceKind::Camera &&
                    history_was_initialized &&
                    ++history_delay_counter > options.cache_delay) {
                    updateFilteredCudaHistoryFrame();
                    history_delay_counter = 0;
                }
                return true;
            }
#endif
#ifdef ACMXVK_WITH_MXVK_CUDA
#if defined(MXVK_WITH_FFMPEG_CAPTURE)
            if (using_ffmpeg_capture &&
                ffmpeg_capture.using_hardware_decode() &&
                !dnnHostProcessingEnabled()) {
                if (!ffmpeg_capture.readGpuRgba(cuda_input_rgba,
                                                ffmpeg_cuda_stream, false)) {
                    return false;
                }
                const cv::cuda::GpuMat &render_input =
                    rotateCudaFrame(cuda_input_rgba, ffmpeg_cuda_stream);
                if (!frame_sprite->updateTextureCuda(render_input,
                                                     ffmpeg_cuda_stream)) {
                    render_input.download(cuda_input_fallback_rgba,
                                          ffmpeg_cuda_stream);
                    ffmpeg_cuda_stream.waitForCompletion();
                    if (!cuda_input_fallback_logged) {
                        std::cerr
                            << "acmxvk: direct NVDEC/Vulkan upload unavailable; "
                               "using host staging\n";
                        cuda_input_fallback_logged = true;
                    }
                    frame_sprite->updateTexture(
                        cuda_input_fallback_rgba.ptr(),
                        cuda_input_fallback_rgba.cols,
                        cuda_input_fallback_rgba.rows,
                        static_cast<int>(cuda_input_fallback_rgba.step));
                }
                updateModelTextureCuda(render_input, ffmpeg_cuda_stream);
                if (!cuda_input_path_logged) {
                    std::cout << "acmxvk: CUDA input path active: FFmpeg "
                                 "NVDEC -> CUDA RGBA -> ";
                    if (options.frame_rotation != FrameRotation::None) {
                        std::cout << "CUDA rotation -> ";
                    }
                    std::cout << "Vulkan texture";
                    if (cuda_input_fallback_logged) {
                        std::cout << " (host-staging fallback)";
                    }
                    std::cout << '\n';
                    cuda_input_path_logged = true;
                }
                const bool history_was_initialized = history_initialized;
                initializeCudaHistory(render_input, ffmpeg_cuda_stream, false);
                if (history_was_initialized &&
                    ++history_delay_counter > options.cache_delay) {
                    updateCudaHistoryFrame(render_input, ffmpeg_cuda_stream);
                    history_delay_counter = 0;
                }
                return true;
            }
#endif
#endif

            bool requires_host_frame = dnnHostProcessingEnabled() ||
                                       historyCacheEnabled() ||
                                       options.frame_rotation != FrameRotation::None ||
                                       model_initialized;
            if (!requires_host_frame
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                && !using_ffmpeg_capture
#endif
            ) {
                return capture.readToSprite(*frame_sprite, false);
            }

            cv::Mat rgba;
            if (!readHostRgba(rgba)) {
                return false;
            }
            applyDnnEffects(rgba);
            rotateFrame(rgba, options.frame_rotation);
            if (!human_overlay_rgba.empty()) {
                rotateFrame(human_overlay_rgba, options.frame_rotation);
            }
            uploadInputFrame(rgba);
            updateHumanOverlayTexture();
            if (source_kind == SourceKind::Camera) {
                latest_camera_history_rgba = rgba;
            }
            const bool history_was_initialized = history_initialized;
            initializeHistory(rgba);
            if (source_kind != SourceKind::Camera && history_was_initialized &&
                ++history_delay_counter > options.cache_delay) {
                updateHistoryFrame(rgba);
                history_delay_counter = 0;
            }
            return true;
        }

        void updateShaderUniforms(int width, int height) {
            const auto now = std::chrono::steady_clock::now();
            updateCrossfade(now);
            const float wall_delta =
                std::chrono::duration<float>(now - previous_frame).count();
            previous_frame = now;
            ++frame_count;

            double video_timeline = 0.0;
            const bool video_timeline_available =
                currentVideoTimeline(video_timeline);
            float delta = wall_delta;
            if (video_timeline_available) {
                if (!video_shader_clock_logged) {
                    std::cout
                        << "acmxvk: shader clock: decoded video timeline; "
                           "effects are independent of processing speed\n";
                    video_shader_clock_logged = true;
                }
                if (!video_shader_timeline_initialized ||
                    video_timeline < previous_video_shader_timeline) {
                    if (video_shader_timeline_initialized &&
                        video_timeline < previous_video_shader_timeline) {
                        shader_time = 0.0;
                        frame_count = 1;
                    }
                    delta = 0.0F;
                    video_shader_timeline_initialized = true;
                } else {
                    delta = static_cast<float>(
                        video_timeline - previous_video_shader_timeline);
                }
                previous_video_shader_timeline = video_timeline;
            } else if (options.normalized_time) {
                delta = static_cast<float>(1.0 / outputFrameRate());
            }
            const float frame_rate =
                video_timeline_available
                    ? static_cast<float>(video_source_fps)
                    : (delta > 0.0F ? 1.0F / delta : 0.0F);
            float raw_audio_amplitude = 0.0F;
            float audio_sensitivity = 1.0F;
            float audio_amplitude = 0.0F;
            float audio_frequency = 0.0F;
            float audio_peak = 0.0F;
            float audio_rms = 0.0F;
            float audio_smooth = 0.0F;
            float audio_low = 0.0F;
            float audio_mid = 0.0F;
            float audio_high = 0.0F;
            float audio_sample_rate = 44100.0F;
#ifdef AUDIO_ENABLED
            std::vector<float> spectrum_values;
            if (file_audio_source != nullptr && audio_engine != nullptr &&
                media_timeline_started &&
                (file_audio_source->has_output_clock() ||
                 source_frame_received)) {
                double source_audio_time = 0.0;
                if (options.use_source_audio &&
                    !file_audio_source->has_output_clock() &&
                    mediaClockSeconds(source_audio_time)) {
                    file_audio_source->process_at_time(
                        source_audio_time, outputFrameRate(), *audio_engine);
                } else {
                    file_audio_source->process_frame(outputFrameRate(),
                                                     *audio_engine);
                }
                if (options.audio_trunc && !file_audio_source->is_active()) {
                    std::cout << "acmxvk: audio source finished, stopping "
                                 "(--audio-trunc)\n";
                    exit();
                }
            }
            if (audioSourceOpen()) {
                const audio::AudioMetrics metrics = audio_engine->metrics();
                const float warmup = updateAudioWarmup(now);
                raw_audio_amplitude = metrics.amplitude;
                audio_sensitivity = audio_engine->sensitivity();
                const float delta_scale = audio_delta_time ? delta : 1.0F;
                const float sense = audio_sensitivity * 4.0F * warmup;
                audio_amplitude = raw_audio_amplitude * audio_sensitivity *
                                  static_cast<float>(options.time_speed) *
                                  delta_scale * warmup;
                audio_frequency = metrics.frequency;
                audio_peak = std::sqrt(std::max(metrics.peak, 0.0F)) * sense;
                audio_rms = std::sqrt(std::max(metrics.rms, 0.0F)) * sense;
                audio_smooth = std::sqrt(std::max(metrics.smooth, 0.0F)) * sense;
                audio_low = std::sqrt(std::max(metrics.low, 0.0F)) * sense;
                audio_mid = std::sqrt(std::max(metrics.mid, 0.0F)) * sense;
                audio_high = std::sqrt(std::max(metrics.high, 0.0F)) * sense;
                audio_sample_rate = static_cast<float>(audio_engine->sample_rate());
                spectrum_values = audio_engine->spectrum();
                const float spectrum_scale =
                    warmup *
                    (spectrum_scale_by_sensitivity ? audio_sensitivity : 1.0F);
                for (float &value : spectrum_values) {
                    value *= spectrum_scale;
                }
            }
#endif
            if (audio_time_active) {
                const float delta_scale = audio_delta_time ? delta : 1.0F;
                shader_time += static_cast<double>(raw_audio_amplitude) *
                               static_cast<double>(audio_sensitivity) *
                               options.time_speed *
                               static_cast<double>(delta_scale);
            } else if (shader_time_active) {
                shader_time += static_cast<double>(delta) * options.time_speed;
            }
            if (!std::isfinite(shader_time)) {
                shader_time = 0.0;
            }
            model_wave_audio_step =
                audio_amplitude * raw_audio_amplitude;
            if (video_timeline_available) {
                const std::uint64_t source_frame =
                    video_source_frame_count - 1U;
                if (source_frame <= 58U) {
                    legacy_alpha =
                        0.2F + 0.1F * static_cast<float>(source_frame);
                } else {
                    const std::uint64_t phase = (source_frame - 59U) % 100U;
                    legacy_alpha =
                        phase < 50U
                            ? 5.9F - 0.1F * static_cast<float>(phase)
                            : 1.1F +
                                  0.1F * static_cast<float>(phase - 50U);
                }
            } else if (legacy_alpha_increasing) {
                legacy_alpha += 0.1F;
                if (legacy_alpha >= 6.0F) {
                    legacy_alpha = 6.0F;
                    legacy_alpha_increasing = false;
                }
            } else {
                legacy_alpha -= 0.1F;
                if (legacy_alpha <= 1.0F) {
                    legacy_alpha = 1.0F;
                    legacy_alpha_increasing = true;
                }
            }
            const float elapsed = static_cast<float>(shader_time);
            const float compatibility_time = video_timeline_available
                                                 ? static_cast<float>(
                                                       video_timeline)
                                                 : std::chrono::duration<float>(
                                                       now - compatibility_clock_start)
                                                       .count();
            const float shader_frame =
                video_timeline_available
                    ? static_cast<float>(video_source_frame_count - 1U)
                    : static_cast<float>(frame_count);
            frame_sprite->setShaderParams(1.0F, 1.0F, 1.0F, elapsed);
            frame_sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
            frame_sprite->setUniform0(legacy_alpha, compatibility_time,
                                      static_cast<float>(width),
                                      static_cast<float>(height));
            frame_sprite->setUniform1(delta, audio_amplitude, audio_frequency,
                                      frame_rate);
            frame_sprite->setUniform2(shader_frame, elapsed,
                                      audio_sample_rate, audio_peak);
            frame_sprite->setUniform3(static_cast<float>(frame_sprite->getHistoryHead()),
                                      static_cast<float>(frame_sprite->getHistoryLayerCount()),
                                      audio_rms, audio_smooth);
            frame_sprite->setAudioBands(audio_low, audio_mid, audio_high);

            model_fragment_uniforms = {};
            model_fragment_uniforms.mouse = glm::vec4(
                mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F, 0.0F);
            model_fragment_uniforms.u0 =
                glm::vec4(legacy_alpha, compatibility_time,
                          static_cast<float>(width),
                          static_cast<float>(height));
            model_fragment_uniforms.u1 =
                glm::vec4(delta, audio_amplitude, audio_frequency,
                          frame_rate);
            model_fragment_uniforms.u2 = glm::vec4(
                shader_frame, elapsed, audio_sample_rate, audio_peak);
            model_fragment_uniforms.u3 = glm::vec4(
                static_cast<float>(frame_sprite->getHistoryHead()),
                static_cast<float>(frame_sprite->getHistoryLayerCount()),
                audio_rms, audio_smooth);
            for (std::size_t index = 0;
                 index < custom_uniform_values.size() && index < 64U;
                 ++index) {
                model_fragment_uniforms.custom_uniforms[index / 4U]
                                                       [index % 4U] =
                    custom_uniform_values[index];
            }
            model_fragment_uniforms.audio_bands =
                glm::vec4(audio_low, audio_mid, audio_high, 0.0F);

            for (std::size_t index = 0; index < post_process_sprites.size(); ++index) {
                mxvk::VK_Sprite *sprite = post_process_sprites[index];
                if (crossfade_active &&
                    index == crossfade_post_process_index) {
                    setPostProcessingShaderParams(index, crossfade_alpha, 0.0F,
                                                  0.0F, 0.0F);
                } else {
                    setPostProcessingShaderParams(index, 1.0F, 1.0F, 1.0F,
                                                  elapsed);
                }
                sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
                sprite->setUniform0(legacy_alpha, compatibility_time,
                                    static_cast<float>(width),
                                    static_cast<float>(height));
                sprite->setUniform1(delta, audio_amplitude, audio_frequency,
                                    frame_rate);
                sprite->setUniform2(shader_frame, elapsed,
                                    audio_sample_rate, audio_peak);
                sprite->setUniform3(
                    static_cast<float>(frame_sprite->getHistoryHead()),
                    static_cast<float>(frame_sprite->getHistoryLayerCount()),
                    audio_rms, audio_smooth);
                sprite->setAudioBands(audio_low, audio_mid, audio_high);
            }
#ifdef AUDIO_ENABLED
            if (!spectrum_values.empty()) {
                frame_sprite->updateSpectrumTexture(
                    spectrum_values.data(),
                    static_cast<std::uint32_t>(spectrum_values.size()));
                if (options.audio_buffers > 0) {
                    frame_sprite->updateSpectrumHistoryTexture(
                        spectrum_values.data(),
                        static_cast<std::uint32_t>(spectrum_values.size()));
                }
                for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                    sprite->updateSpectrumTexture(
                        spectrum_values.data(),
                        static_cast<std::uint32_t>(spectrum_values.size()));
                    if (options.audio_buffers > 0) {
                        sprite->updateSpectrumHistoryTexture(
                            spectrum_values.data(),
                            static_cast<std::uint32_t>(spectrum_values.size()));
                    }
                }
            }
#endif
        }
    };
} // namespace acmxvk

int main(int argc, char **argv) {
    try {
        for (int index = 1; index < argc; ++index) {
            if (argv[index] != nullptr &&
                std::string_view(argv[index]) == "--unbuffered") {
                std::cout << std::unitbuf;
                std::cerr << std::unitbuf;
                break;
            }
        }
        acmxvk::Options options = acmxvk::parseOptions(argc, argv);
        if (options.show_help) {
            acmxvk::printHelp(std::cout);
            return EXIT_SUCCESS;
        }
        if (!options.build_manifest.empty()) {
            return acmxvk::buildShaderLibrary(options);
        }
        if (options.check_audio) {
#ifdef AUDIO_ENABLED
            std::cout << "AUDIO: enabled\n";
#else
            std::cout << "AUDIO: disabled\n";
#endif
            return EXIT_SUCCESS;
        }
        if (options.check_midi) {
#ifdef MIDI_ENABLED
            std::cout << "MIDI: enabled\n";
#else
            std::cout << "MIDI: disabled\n";
#endif
            return EXIT_SUCCESS;
        }
        if (options.check_cuda) {
#ifdef ACMXVK_WITH_MXVK_CUDA
            std::cout << "MXVK CUDA interop: enabled\n";
#else
            std::cout << "MXVK CUDA interop: disabled\n";
#endif
#ifdef ACMXVK_WITH_CUDA
            std::cout << "acidcam-gpu filters: enabled\n";
#else
            std::cout << "acidcam-gpu filters: disabled\n";
#endif
            return EXIT_SUCCESS;
        }
        if (options.check_dnn) {
#ifdef ACMXVK_WITH_DNN
            std::cout << "OpenCV DNN effects: enabled\n";
#else
            std::cout << "OpenCV DNN effects: disabled\n";
#endif
            return EXIT_SUCCESS;
        }
        if (options.list_audio_devices) {
#ifdef AUDIO_ENABLED
            acmxvk::audio::AudioEngine::list_devices();
            return EXIT_SUCCESS;
#else
            throw std::runtime_error(
                "--list-devices requires an ACMXVK build configured with -DAUDIO=ON");
#endif
        }
        if (options.list_midi_devices) {
#ifdef MIDI_ENABLED
            acmxvk::midi::MidiInput::list_ports(std::cout);
            return EXIT_SUCCESS;
#else
            throw std::runtime_error(
                "--list-midi requires an ACMXVK build configured with -DMIDI=ON");
#endif
        }
        if (options.list_gpu_filters) {
#ifdef ACMXVK_WITH_CUDA
            acmxvk::gpu::FilterEngine::list_filters(std::cout);
            return EXIT_SUCCESS;
#else
            throw std::runtime_error(
                "--list-filters requires an ACMXVK build configured with "
                "-DWITH_CUDA=ON");
#endif
        }
        if (options.list_cuda_devices) {
#ifdef ACMXVK_WITH_MXVK_CUDA
            acmxvk::list_cuda_devices(std::cout);
            return EXIT_SUCCESS;
#else
            throw std::runtime_error(
                "--list-cuda-devices requires a CUDA-enabled MXVK installation");
#endif
        }
#ifndef AUDIO_ENABLED
        if (options.enable_audio) {
            throw std::runtime_error(
                "--enable-audio requires an ACMXVK build configured with -DAUDIO=ON");
        }
#endif
#ifndef MIDI_ENABLED
        if (options.midi_device_specified || options.midi_monitor ||
            !options.midi_map_file.empty() || !options.midi_cc_mappings.empty()) {
            throw std::runtime_error(
                "MIDI input requires an ACMXVK build configured with -DMIDI=ON");
        }
#endif
#ifndef ACMXVK_WITH_CUDA
        if (!options.gpu_filter_indices.empty()) {
            throw std::runtime_error(
                "CUDA filters require an ACMXVK build configured with "
                "-DWITH_CUDA=ON");
        }
#endif
#ifndef ACMXVK_WITH_MXVK_CUDA
        if (options.cuda_device_specified) {
            throw std::runtime_error(
                "--cuda-device requires a CUDA-enabled MXVK installation");
        }
#endif
#ifndef ACMXVK_WITH_DNN
        if (!options.edge_model.empty() || !options.human_model.empty() ||
            !options.onnx_configuration.empty() ||
            options.human_background || options.human_black_specified ||
            options.human_white_specified) {
            throw std::runtime_error(
                "DNN effects require an ACMXVK build configured with "
                "-DWITH_OPENCV_DNN=ON");
        }
#endif
        if (options.list_encoders) {
            acmxvk::printEncoders(std::cout);
            return EXIT_SUCCESS;
        }
        if (!options.list_encoder_options.empty()) {
            return acmxvk::printEncoderOptions(options.list_encoder_options, std::cout,
                                               std::cerr)
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE;
        }

#ifdef ACMXVK_WITH_CUDA
        if (!options.gpu_filter_indices.empty()) {
            acmxvk::gpu::FilterEngine::validate_filter_indices(
                options.gpu_filter_indices);
        }
#endif
#ifdef ACMXVK_WITH_MXVK_CUDA
        if (!options.gpu_filter_indices.empty() || options.cuda_device_specified) {
            acmxvk::select_cuda_device(options.cuda_device);
        }
#endif

        if (!options.graphic_file.empty() && !options.resolution_specified) {
            const cv::Mat image = cv::imread(options.graphic_file, cv::IMREAD_UNCHANGED);
            if (!image.empty()) {
                options.width = image.cols;
                options.height = image.rows;
                if (acmxvk::rotationSwapsDimensions(options.frame_rotation)) {
                    std::swap(options.width, options.height);
                }
            }
        }

        acmxvk::MainWindow main_window(std::move(options));
        main_window.loop();
    } catch (const mxvk::Exception &error) {
        std::cerr << "acmxvk: MXVK exception: " << error.text() << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "acmxvk: exception: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
