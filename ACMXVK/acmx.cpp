/**
 * @file acmx.cpp
 * @brief ACMXVK real-time Vulkan video shader application.
 */

#include <mxvk/mxvk.hpp>
#include <mxvk/mxvk_cv.hpp>
#include <mxvk/mxvk_exception.hpp>
#ifdef MXVK_WITH_FFMPEG_CAPTURE
#include <mxvk/mxvk_ff_capture.hpp>
#endif
#include <mxvk/mxvk_png.hpp>
#include <mxwrite.hpp>

#ifdef AUDIO_ENABLED
#include "audio.hpp"
#include "file_audio.hpp"
#endif
#ifdef MIDI_ENABLED
#include "midi.hpp"
#endif
#ifdef ACMXVK_WITH_CUDA
#include "gpu_filters.hpp"
#include <opencv2/cudaarithm.hpp>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

namespace acmxvk {
    namespace fs = std::filesystem;

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
        double time_speed = 1.0;
        double max_size_mb = 0.0;
        double audio_sensitivity = 1.0;
        double audio_warm_rate = 0.5;
        double audio_pass_through_gain = 1.0;
        double audio_recording_gain = 1.0;
        bool resolution_specified = false;
        bool fullscreen = false;
        bool repeat = false;
        bool enable_vsync = false;
        bool enable_screenshot = false;
        bool enable_playlist = false;
        bool enable_texture_cache = false;
        bool normalized_time = false;
        bool flip_output = false;
        bool png_output = false;
        bool encode_realtime = false;
        bool no_drop = false;
        bool copy_audio = false;
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
        bool list_encoders = false;
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
        std::string shader_file;
        std::string playlist_file;
        std::string output_file;
        std::string encode_preset = "medium";
        std::string encode_tune;
        std::string encode_codec = "auto";
        std::string encode_params;
        std::string list_encoder_options;
        std::string audio_file;
        std::string midi_map_file;
    };

    [[nodiscard]] std::string optionValue(int &index, int argc, char **argv,
                                          std::string_view option) {
        if (++index >= argc) {
            throw std::runtime_error("missing value for " + std::string(option));
        }
        return argv[index];
    }

    [[nodiscard]] int parseInteger(std::string_view text, std::string_view option) {
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

    void parseDimensions(std::string_view text, int &width, int &height,
                         std::string_view option) {
        const std::size_t separator = text.find_first_of("xX");
        if (separator == std::string_view::npos) {
            throw std::runtime_error("invalid dimensions for " + std::string(option) +
                                     "; expected WidthxHeight");
        }

        width = parseInteger(text.substr(0, separator), option);
        height = parseInteger(text.substr(separator + 1), option);
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("dimensions must be positive for " + std::string(option));
        }
    }

    [[nodiscard]] FrameRotation parseFrameRotation(std::string value) {
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

    [[nodiscard]] Options parseOptions(int argc, char **argv) {
        Options options;
        if (argc == 1) {
            options.show_help = true;
            return options;
        }

        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "-h" || option == "-v" || option == "--help" ||
                option == "--version") {
                options.show_help = true;
            } else if (option == "-i" || option == "--input") {
                options.input_file = optionValue(index, argc, argv, option);
            } else if (option == "-g" || option == "--graphic") {
                options.graphic_file = optionValue(index, argc, argv, option);
            } else if (option == "-o" || option == "--output") {
                options.output_file = optionValue(index, argc, argv, option);
            } else if (option == "-d" || option == "--device") {
                options.camera_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
            } else if (option == "-c" || option == "--camera-res") {
                parseDimensions(optionValue(index, argc, argv, option),
                                options.camera_width, options.camera_height, option);
            } else if (option == "-r" || option == "--resolution") {
                parseDimensions(optionValue(index, argc, argv, option), options.width,
                                options.height, option);
                options.resolution_specified = true;
            } else if (option == "-s" || option == "--shaders") {
                options.shader_directory = optionValue(index, argc, argv, option);
            } else if (option == "-f" || option == "--fragment") {
                options.fragment_shader = optionValue(index, argc, argv, option);
            } else if (option == "-H" || option == "--shader-index") {
                options.shader_index =
                    parseInteger(optionValue(index, argc, argv, option), option);
            } else if (option == "--shader-file") {
                options.shader_file = optionValue(index, argc, argv, option);
            } else if (option == "--uniform") {
                options.custom_uniform_overrides.push_back(
                    optionValue(index, argc, argv, option));
            } else if (option == "--shader-pass") {
                const std::string values = optionValue(index, argc, argv, option);
                std::size_t start = 0;
                while (start <= values.size()) {
                    const std::size_t separator = values.find(',', start);
                    const std::string_view value(
                        values.data() + start,
                        (separator == std::string::npos ? values.size() : separator) - start);
                    if (!value.empty()) {
                        options.shader_pass_indices.push_back(parseInteger(value, option));
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
                    start = name_start + static_cast<std::size_t>(length);
                }
            } else if (option == "--playlist") {
                options.playlist_file = optionValue(index, argc, argv, option);
            } else if (option == "--enable-playlist") {
                options.enable_playlist = true;
            } else if (option == "--time-speed") {
                options.time_speed =
                    parseNumber(optionValue(index, argc, argv, option), option);
            } else if (option == "--normalized") {
                options.normalized_time = true;
            } else if (option == "--autopilot-frames" ||
                       option == "--autopilot-timeout") {
                options.autopilot_frames =
                    std::max(4, parseInteger(optionValue(index, argc, argv, option), option));
            } else if (option == "--autopilot-random" ||
                       option == "--autiopilot-random") {
                options.autopilot_random_timeout =
                    std::max(4, parseInteger(optionValue(index, argc, argv, option), option));
            } else if (option == "-u" || option == "--fps") {
                options.requested_fps =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.requested_fps <= 0.0) {
                    throw std::runtime_error("FPS must be positive");
                }
            } else if (option == "-w" || option == "--enable-audio") {
                options.enable_audio = true;
            } else if (option == "-l" || option == "--channels") {
                options.audio_channels =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.audio_channels < 1) {
                    throw std::runtime_error("audio channels must be positive");
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
                if (options.audio_warm_rate < 0.0) {
                    throw std::runtime_error(
                        "audio warmup rate must be non-negative");
                }
            } else if (option == "--audio-input") {
                const std::string value = optionValue(index, argc, argv, option);
                options.audio_input_specified = true;
                options.audio_input_device =
                    value == "default" ? -1 : parseInteger(value, option);
                if (options.audio_input_device < -1) {
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
                if (options.audio_output_device < -1) {
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
            } else if (option == "--audio-repeat") {
                options.audio_repeat = true;
            } else if (option == "--audio-trunc") {
                options.audio_trunc = true;
            } else if (option == "--enable-audio-buffers" ||
                       option == "--audio-buffers") {
                options.audio_buffers = std::max(
                    parseInteger(optionValue(index, argc, argv, option), option), 0);
            } else if (option == "--list-devices") {
                options.list_audio_devices = true;
            } else if (option == "--check-audio") {
                options.check_audio = true;
            } else if (option == "--midi-device") {
                options.midi_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
                options.midi_device_specified = true;
                if (options.midi_device < 0) {
                    throw std::runtime_error(
                        "MIDI device index must be non-negative");
                }
            } else if (option == "--midi-monitor") {
                options.midi_monitor = true;
            } else if (option == "--midi-map") {
                options.midi_map_file = optionValue(index, argc, argv, option);
            } else if (option == "--midi-cc") {
                options.midi_cc_mappings.push_back(
                    optionValue(index, argc, argv, option));
            } else if (option == "--list-midi") {
                options.list_midi_devices = true;
            } else if (option == "--check-midi") {
                options.check_midi = true;
            } else if (option == "--gpu-filter") {
                options.gpu_filter_indices = parseIntegerList(
                    optionValue(index, argc, argv, option), option);
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
                if (options.cuda_device < 0) {
                    throw std::runtime_error(
                        "CUDA device index must be non-negative");
                }
            } else if (option == "--list-filters") {
                options.list_gpu_filters = true;
            } else if (option == "--list-cuda-devices") {
                options.list_cuda_devices = true;
            } else if (option == "--check-cuda") {
                options.check_cuda = true;
            } else if (option == "--duration") {
                options.duration = parseNumber(optionValue(index, argc, argv, option), option);
                if (options.duration <= 0.0) {
                    throw std::runtime_error("duration must be positive");
                }
            } else if (option == "--max-size") {
                options.max_size_mb =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.max_size_mb <= 0.0) {
                    throw std::runtime_error("maximum output size must be positive");
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
            } else if (option == "--copy-audio") {
                options.copy_audio = true;
            } else if (option == "-n" || option == "--fullscreen") {
                options.fullscreen = true;
            } else if (option == "-a" || option == "--repeat") {
                options.repeat = true;
            } else if (option == "--enable-vsync") {
                options.enable_vsync = true;
            } else if (option == "--enable-screenshot") {
                options.enable_screenshot = true;
            } else if (option == "--history-test" || option == "--texture-cache" ||
                       option == "--texture-cache-array") {
                options.enable_texture_cache = true;
            } else if (option == "--cache-delay") {
                options.cache_delay =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.cache_delay < 0) {
                    throw std::runtime_error("--cache-delay must be zero or greater");
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

        if (!options.input_file.empty() && !options.graphic_file.empty()) {
            throw std::runtime_error("--input and --graphic cannot be used together");
        }
        if (!options.shader_directory.empty() && !options.fragment_shader.empty()) {
            throw std::runtime_error("--shaders and --fragment cannot be used together");
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
             options.output_file.empty() || options.png_output ||
             options.copy_audio)) {
            throw std::runtime_error(
                "--record-gain requires live --enable-audio and encoded output");
        }
        return options;
    }

    void printHelp(std::ostream &output) {
        output << "ACMXVK - Vulkan video shader engine (Increment 7G)\n\n"
               << "Usage:\n"
               << "  acmxvk -i video.mp4 -s shader-directory [options]\n"
               << "  acmxvk -g image.png -f shader.spv [options]\n"
               << "  acmxvk -d 0 -s shader-directory [options]\n\n"
               << "Input:\n"
               << "  -i, --input <file>          Read a video file\n"
               << "  -g, --graphic <file>        Read a still image\n"
               << "  -d, --device <index>        Camera device (default 0)\n"
               << "  -c, --camera-res <WxH>      Requested camera dimensions\n"
               << "  -u, --fps <rate>            Camera/output FPS\n"
               << "                              Video files prefer FFmpeg/NVDEC capture\n\n"
               << "Shaders:\n"
               << "  -s, --shaders <directory>   SPIR-V library with library.json or index.txt\n"
               << "  -f, --fragment <file.spv>   Use one SPIR-V fragment shader\n"
               << "  -H, --shader-index <index>  Initial library shader index\n"
               << "      --shader-file <name>    Initial library shader filename\n"
               << "      --uniform <name=value>  Override a library.json custom float\n\n"
               << "  --shader-pass <indices>     Comma-separated pre-shader pass chain\n"
               << "  --shader-pass-files <data>  ACMX2 length-prefixed shader filenames\n"
               << "  --playlist <file>           Shader or named multipass playlist\n\n"
               << "  --enable-playlist           Enable the playlist immediately\n"
               << "  --time-speed <mult>         Scale shader time (default 1.0)\n"
               << "  --normalized                Use fixed output-frame shader time\n"
               << "  --autopilot-frames <N>      Playlist switch interval (minimum 4)\n"
               << "  --autopilot-timeout <N>     Alias for --autopilot-frames\n"
               << "  --autopilot-random <N>      Random playlist interval from 4..N\n\n"
               << "History cache:\n"
               << "      --texture-cache         Enable Vulkan texture history\n"
               << "      --texture-cache-array   Alias using sampler2DArray history\n"
               << "      --texture-cache-size N  History layers, 1-64 (default 8)\n"
               << "      --cache-delay N         Skip N frames between cache writes\n"
               << "      --history-test          Compatibility alias for --texture-cache\n\n"
               << "Recording:\n"
               << "  -o, --output <file>         Encode processed output with MXWrite\n"
               << "      --duration <seconds>    Stop after this much output video\n"
               << "      --max-size <MB>         Stop when encoded output exceeds this size\n"
               << "      --png                   Write video output as a PNG sequence\n"
               << "      --generate <N>          Save a PNG every N processed frames\n"
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
               << "      --copy-audio            Copy input audio into encoded output\n\n"
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
               << "      --record-gain N         Muxed mic gain, 0.0-2.0 (default 1.0)\n"
               << "      --audio-repeat          Restart file audio at end-of-stream\n"
               << "      --audio-trunc           Stop ACMXVK when file audio finishes\n"
               << "                              Live/file audio is muxed into encoded output\n"
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
               << "CUDA filters (requires WITH_CUDA=ON build):\n"
               << "      --gpu-filter <list>     Comma-separated acidcam-gpu indices\n"
               << "      --gpu-buffer <4-32>     Temporal frame count (default 10)\n"
               << "  -m, --cuda-device <index>   Select CUDA device (default 0)\n"
               << "      --list-filters          List acidcam-gpu filters and exit\n"
               << "      --list-cuda-devices     List CUDA devices and exit\n"
               << "      --check-cuda            Report compiled CUDA-filter support\n"
               << "                              Left/Right selects the active filter\n"
               << "                              NVDEC remains resident without a filter\n"
               << "                              Video/camera RGBA and rotation stay on GPU\n\n"
               << "Window:\n"
               << "  -r, --resolution <WxH>      Window resolution\n"
               << "  -n, --fullscreen            Start fullscreen\n"
               << "  -a, --repeat                Repeat video input\n"
               << "      --rotate <mode>         clockwise, 180, or counterclockwise\n"
               << "      --flip                  Flip final display/encoded output vertically\n"
               << "      --enable-vsync          Use FIFO presentation\n"
               << "      --enable-screenshot     Enable MXVK F10 screenshots\n\n"
               << "Keys: Up/Down shader or playlist node, Shift+Up/Down final shader,\n"
               << "      P playlist/pause, L freeze, T time, U/I step time,\n"
               << "      Page Up/Down time speed, Insert/Delete audio sensitivity,\n"
               << "      F fullscreen, M multipass,\n"
               << "      J random autopilot, Y sequential autopilot, Space bypass,\n"
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
        constexpr std::size_t MAX_CUSTOM_UNIFORM_NAME_LENGTH = 64;
        if (name.empty() || name.size() >= MAX_CUSTOM_UNIFORM_NAME_LENGTH ||
            name.starts_with("gl_") ||
            !(std::isalpha(static_cast<unsigned char>(name.front())) ||
              name.front() == '_')) {
            return false;
        }
        return std::all_of(name.begin() + 1, name.end(), [](unsigned char character) {
            return std::isalnum(character) != 0 || character == '_';
        });
    }

    [[nodiscard]] ShaderManifest loadShaderManifest(const fs::path &directory) {
        ShaderManifest manifest;
        const fs::path json_path = directory / "library.json";
        const fs::path text_path = directory / "index.txt";
        if (fs::is_regular_file(json_path)) {
            manifest.path = json_path;
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
                    manifest.entries.push_back(std::move(filename));
                }

                const cv::FileNode custom_uniforms = storage["custom_uniforms"];
                if (!custom_uniforms.empty()) {
                    if (!custom_uniforms.isMap()) {
                        throw std::runtime_error(
                            json_path.string() +
                            " field 'custom_uniforms' must be an object");
                    }
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
                            uniform.step <= 0.0) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains an invalid range for custom uniform: " +
                                uniform.name);
                        }
                        uniform.value = std::clamp(
                            uniform.value, uniform.minimum, uniform.maximum);
                        manifest.custom_uniforms.push_back(std::move(uniform));
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
        std::ifstream input(text_path);
        if (!input) {
            throw std::runtime_error("unable to open shader manifest: " +
                                     text_path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            line = trim(std::move(line));
            if (!line.empty() && line.front() != '#') {
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

    [[nodiscard]] cv::Mat loadRgbaImage(const std::string &filename) {
        const cv::Mat source = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (source.empty()) {
            throw std::runtime_error("unable to load image: " + filename);
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

    struct PlaylistNode {
        std::string name;
        std::vector<fs::path> shaders;
    };

    class MainWindow final : public mxvk::VK_Window {
      public:
        explicit MainWindow(Options options)
            : mxvk::VK_Window("ACMXVK", options.width, options.height,
                              options.fullscreen, MXVK_VALIDATION, options.enable_vsync),
              options(std::move(options)) {
            setClearColor(0.0F, 0.0F, 0.0F, 1.0F);
            setEnableScreenshot(this->options.enable_screenshot);
            initializeGpuFilters();
            openAudio();
            loadShaders();
            configureMidiMappings();
            openMidi();
            loadShaderPasses();
            loadPlaylist();
            resetAutopilotInterval();
            openInput();
            initializeSprite();
            openOutput();
        }

        ~MainWindow() override {
            const bool should_copy_audio = options.copy_audio && writer.is_open();
#ifdef AUDIO_ENABLED
            const bool should_mux_file_audio =
                file_audio_source != nullptr && writer.is_open() &&
                !options.output_file.empty() && !options.png_output &&
                output_frame_count > 0;
            const bool should_mux_live_audio =
                audio_engine != nullptr && file_audio_source == nullptr &&
                audio_engine->is_recording() && writer.is_open() &&
                !options.output_file.empty() && !options.png_output &&
                output_frame_count > 0;
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
                    effects_enabled = !effects_enabled;
                    applyShaderPipeline();
                    std::cout << "acmxvk: shader effects "
                              << (effects_enabled ? "enabled" : "bypassed") << '\n';
                    break;
                case SDLK_P:
                    if (!playlist.empty()) {
                        playlist_enabled = !playlist_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: playlist "
                                  << (playlist_enabled ? "enabled" : "disabled") << '\n';
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
                case SDLK_U:
                    stepShaderTime(0.05);
                    break;
                case SDLK_I:
                    stepShaderTime(-0.05);
                    break;
                case SDLK_F:
                    toggleFullscreen();
                    break;
                case SDLK_INSERT:
                    adjustAudioSensitivity(0.1F);
                    break;
                case SDLK_DELETE:
                    adjustAudioSensitivity(-0.1F);
                    break;
                case SDLK_M:
                    if (!configured_passes.empty()) {
                        multipass_enabled = !multipass_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: multipass "
                                  << (multipass_enabled ? "enabled" : "disabled") << '\n';
                    }
                    break;
                case SDLK_J:
                    toggleAutopilot(false);
                    break;
                case SDLK_Y:
                    toggleAutopilot(true);
                    break;
                default:
                    break;
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                mouse_x = event.motion.x;
                mouse_y = event.motion.y;
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = true;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = false;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
            }
        }

        void onSwapchainRecreated() override {
            initializeSprite();
        }

        void proc() override {
            if (recording_complete) {
                return;
            }

            pollMidi();

            if (!rendering_frozen && !input_paused &&
                source_kind != SourceKind::Graphic) {
                if (initial_frame_pending) {
                    initial_frame_pending = false;
                } else if (!readInputFrame()) {
                    if (!handleCaptureEnd()) {
                        return;
                    }
                }
            }

            if (!rendering_frozen) {
                updateAutopilot();
            }
            const VkExtent2D extent = getSwapchainExtent();
            const int target_width = extent.width > 0U ? static_cast<int>(extent.width) : options.width;
            const int target_height =
                extent.height > 0U ? static_cast<int>(extent.height) : options.height;

            if (!rendering_frozen) {
                updateShaderUniforms(target_width, target_height);
            }
            frame_sprite->drawSpriteRect(0, 0, target_width, target_height);
        }

      private:
        enum class SourceKind { Camera,
                                Video,
                                Graphic };

        Options options;
        SourceKind source_kind = SourceKind::Camera;
        mxvk::VK_Capture capture;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
        mxvk::VK_FF_Capture ffmpeg_capture;
        std::vector<std::uint8_t> ffmpeg_rgba;
        bool using_ffmpeg_capture = false;
#ifdef ACMXVK_WITH_CUDA
        cv::cuda::Stream ffmpeg_cuda_stream;
#endif
#endif
        Writer writer;
        mxvk::VK_Sprite *frame_sprite = nullptr;
        cv::Mat graphic_rgba;
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
        std::size_t shader_index = 0;
        std::size_t playlist_index = 0;
        bool effects_enabled = true;
        bool multipass_enabled = false;
        bool playlist_enabled = false;
        float mouse_x = 0.0F;
        float mouse_y = 0.0F;
        bool mouse_pressed = false;
        bool history_initialized = false;
        bool initial_frame_pending = false;
        bool recording_complete = false;
        bool input_paused = false;
        bool rendering_frozen = false;
        bool shader_time_active = true;
        bool autopilot_enabled = false;
        bool autopilot_sequential = false;
        int recording_width = 0;
        int recording_height = 0;
        int autopilot_counter = 0;
        int autopilot_interval_frames = 0;
        int history_delay_counter = 0;
        double recording_fps = 0.0;
        double shader_time = 0.0;
        std::uint64_t output_frame_count = 0;
        std::uint64_t png_frame_count = 0;
        std::uint64_t generated_frame_count = 0;
        std::uint64_t frame_count = 0;
        std::chrono::steady_clock::time_point previous_frame{std::chrono::steady_clock::now()};
        std::mt19937 autopilot_rng{std::random_device{}()};
#ifdef ACMXVK_WITH_CUDA
        std::unique_ptr<gpu::FilterEngine> gpu_filter_engine;
        cv::cuda::GpuMat cuda_input_rgba;
        cv::cuda::GpuMat cuda_rotated_rgba;
        cv::cuda::GpuMat cuda_rotation_transpose;
        cv::Mat cuda_input_fallback_rgba;
        cv::Mat cuda_history_fallback_rgba;
        bool cuda_input_path_logged = false;
        bool cuda_input_fallback_logged = false;
        bool cuda_history_fallback_logged = false;
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
                            << "acmxvk: MIDI map action unavailable in 6C: "
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
            case 260:
                return SDLK_INSERT;
            case 261:
                return SDLK_DELETE;
            case 500:
                return SDLK_U;
            case 501:
                return SDLK_I;
            case 32:
                return SDLK_SPACE;
            case 74:
            case 78:
                return SDLK_J;
            case 76:
                return SDLK_L;
            case 77:
                return SDLK_M;
            case 80:
                return SDLK_P;
            case 84:
                return SDLK_T;
            case 89:
                return SDLK_Y;
            case 90:
                return SDLK_F10;
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

        [[nodiscard]] bool isMidiMappingSupported(
            const midi::MidiMapping &mapping) const {
            if (isMidiSliderMapping(mapping)) {
                const int slider = (mapping.primary_action - 600) / 2;
                return midi_slider_uniform_indices[slider] >= 0;
            }
            if (mapping.secondary_action == 0) {
                return midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
            }
            return midiActionKey(mapping.primary_action) != SDLK_UNKNOWN &&
                   midiActionKey(mapping.secondary_action) != SDLK_UNKNOWN;
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
            case 260:
                return "increase audio sensitivity";
            case 261:
                return "decrease audio sensitivity";
            case 500:
                return "step shader time forward";
            case 501:
                return "step shader time backward";
            case 32:
                return "toggle shader bypass";
            case 74:
            case 78:
                return "toggle random autopilot";
            case 76:
                return "toggle rendering freeze";
            case 77:
                return "toggle multipass";
            case 80:
                return "toggle playlist or input pause";
            case 84:
                return "toggle shader time";
            case 89:
                return "toggle sequential autopilot";
            case 90:
                return "take screenshot";
            default:
                return "unsupported action";
            }
        }

        void dispatchMidiAction(int action) {
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
                dispatchMidiAction(state.value > 64
                                       ? mapping.primary_action
                                       : mapping.secondary_action);
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
                    throw std::runtime_error("could not decode --audio-file: " +
                                             options.audio_file);
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

        void loadShaders() {
            if (!options.fragment_shader.empty()) {
                const fs::path fragment = fs::absolute(options.fragment_shader).lexically_normal();
                if (fragment.extension() != ".spv" || !fs::is_regular_file(fragment)) {
                    throw std::runtime_error("fragment shader is not a readable .spv file: " +
                                             fragment.string());
                }
                shaders.push_back(fragment);
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

        [[nodiscard]] fs::path findShader(std::string name) const {
            name = trim(std::move(name));
            if (name.empty()) {
                return {};
            }

            fs::path requested(name);
            if (requested.extension() != ".spv") {
                requested.replace_extension(".spv");
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

            std::ifstream input(options.playlist_file);
            if (!input) {
                throw std::runtime_error("unable to open playlist: " + options.playlist_file);
            }

            PlaylistNode *current_node = nullptr;
            std::vector<fs::path> default_entries;
            std::string line;
            while (std::getline(input, line)) {
                line = trim(std::move(line));
                if (line.empty() || line.front() == '#') {
                    continue;
                }
                if (line.size() >= 2 && line.front() == '[' && line.back() == ']') {
                    playlist.push_back({line.substr(1, line.size() - 2), {}});
                    current_node = &playlist.back();
                    continue;
                }

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
        }

        [[nodiscard]] std::string spriteVertexShader() const {
            if (fs::is_regular_file(ACMXVK_INSTALL_SPRITE_VERTEX_SHADER)) {
                return ACMXVK_INSTALL_SPRITE_VERTEX_SHADER;
            }
            return ACMXVK_BUILD_SPRITE_VERTEX_SHADER;
        }

        [[nodiscard]] std::string echoCacheShader() const {
            if (fs::is_regular_file(ACMXVK_INSTALL_ECHO_CACHE_SHADER)) {
                return ACMXVK_INSTALL_ECHO_CACHE_SHADER;
            }
            return ACMXVK_BUILD_ECHO_CACHE_SHADER;
        }

        [[nodiscard]] fs::path flipShader() const {
            if (fs::is_regular_file(ACMXVK_INSTALL_FLIP_SHADER)) {
                return ACMXVK_INSTALL_FLIP_SHADER;
            }
            return ACMXVK_BUILD_FLIP_SHADER;
        }

        void openInput() {
            if (!options.graphic_file.empty()) {
                source_kind = SourceKind::Graphic;
                graphic_rgba = loadRgbaImage(options.graphic_file);
                rotateFrame(graphic_rgba, options.frame_rotation);
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

            if (source_kind == SourceKind::Camera) {
                capture.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
                if (options.requested_fps > 0.0) {
                    capture.set(cv::CAP_PROP_FPS, options.requested_fps);
                }
            }
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

        void openOutput() {
            if (options.output_file.empty() && options.generate_interval <= 0) {
                return;
            }

            const VkExtent2D extent = getSwapchainExtent();
            recording_width = extent.width > 0U ? static_cast<int>(extent.width) : options.width;
            recording_height =
                extent.height > 0U ? static_cast<int>(extent.height) : options.height;
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
            }

            setFrameReadbackEnabled(true);
        }

        void onFrameReadback(std::vector<std::uint8_t> &rgba, uint32_t width,
                             uint32_t height) override {
            if ((!writer.is_open() && !options.png_output &&
                 options.generate_interval <= 0) ||
                recording_complete) {
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
#ifdef AUDIO_ENABLED
                if (output_frame_count == 0 && audio_engine != nullptr &&
                    file_audio_source == nullptr && !options.copy_audio &&
                    !audio_engine->start_recording()) {
                    std::cerr << "acmxvk: could not start live audio recording; "
                                 "continuing with video-only output\n";
                }
#endif
                writer.write(output_pixels);
            }
            if (options.png_output) {
                savePng(framePath(png_output_directory, png_frame_count), output_pixels,
                        recording_width, recording_height);
                ++png_frame_count;
            }
            if (options.generate_interval > 0 &&
                output_frame_count %
                        static_cast<std::uint64_t>(options.generate_interval) ==
                    0) {
                savePng(framePath(generate_output_directory, generated_frame_count),
                        output_pixels, recording_width, recording_height);
                ++generated_frame_count;
            }
            ++output_frame_count;

            if (options.duration > 0.0) {
                const auto maximum_frames = static_cast<std::uint64_t>(
                    std::ceil(options.duration * recording_fps));
                if (output_frame_count >= maximum_frames) {
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

        void initializeSprite() {
            if (!ensureRenderResources()) {
                throw std::runtime_error("MXVK failed to initialize render resources");
            }

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
                    source_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
                    source_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
                }
                if (source_width <= 0 || source_height <= 0) {
                    source_width = options.camera_width;
                    source_height = options.camera_height;
                }
                if (rotationSwapsDimensions(options.frame_rotation)) {
                    std::swap(source_width, source_height);
                }
            }

            if (frame_sprite == nullptr) {
                frame_sprite = createSprite(source_width, source_height);
            }
            frame_sprite->enableExtendedUBO();
            frame_sprite->setCustomUniforms(custom_uniform_values);
#ifdef AUDIO_ENABLED
            frame_sprite->enableSpectrumTexture(audio::AudioEngine::spectrum_bin_count());
            if (options.audio_buffers > 0) {
                frame_sprite->enableSpectrumHistoryTexture(
                    audio::AudioEngine::spectrum_bin_count(),
                    static_cast<std::uint32_t>(options.audio_buffers));
            }
#endif
            if (options.enable_texture_cache) {
                frame_sprite->enableHistoryTexture(source_width, source_height,
                                                   static_cast<uint32_t>(
                                                       options.texture_cache_size));
            }
            frame_sprite->createEmptySprite(source_width, source_height, spriteVertexShader(),
                                            options.enable_texture_cache ? echoCacheShader()
                                                                         : std::string{});

            if (source_kind == SourceKind::Graphic) {
                initial_frame_pending = false;
                uploadInputFrame(graphic_rgba);
                initializeHistory(graphic_rgba);
            } else if (!readInputFrame()) {
                std::cerr << "acmxvk: capture did not provide an initial frame\n";
            } else {
                initial_frame_pending = true;
            }

            applyShaderPipeline();
            if (!currentShader().empty()) {
                std::cout << "acmxvk: shader " << (shader_index + 1) << '/' << shaders.size()
                          << ": " << currentShader() << '\n';
            }
        }

        void resetShaderTime() {
            previous_frame = std::chrono::steady_clock::now();
            shader_time = 0.0;
            frame_count = 0;
        }

        void togglePause() {
            if (source_kind == SourceKind::Camera) {
                std::cout << "acmxvk: pause is available for video and graphic input\n";
                return;
            }
            input_paused = !input_paused;
            std::cout << "acmxvk: input pause "
                      << (input_paused ? "enabled" : "disabled") << '\n';
        }

        void toggleFreeze() {
            if (source_kind == SourceKind::Camera) {
                std::cout << "acmxvk: freeze is available for video and graphic input\n";
                return;
            }
            rendering_frozen = !rendering_frozen;
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
            if (!autopilot_enabled || !playlist_enabled || playlist.empty() ||
                autopilot_interval_frames <= 0) {
                return;
            }
            if (++autopilot_counter < autopilot_interval_frames) {
                return;
            }
            autopilot_counter = 0;

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
            std::cout << "acmxvk: autopilot -> " << playlist[playlist_index].name << " ("
                      << (playlist_index + 1) << '/' << playlist.size() << ")\n";
        }

        void selectShader(int direction) {
            if (shaders.size() < 2 || frame_sprite == nullptr) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(shaders.size());
            auto index = static_cast<std::ptrdiff_t>(shader_index) + direction;
            index = (index % count + count) % count;
            shader_index = static_cast<std::size_t>(index);

            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            std::cout << "acmxvk: shader " << (shader_index + 1) << '/' << shaders.size()
                      << ": " << currentShader() << '\n';
        }

        void selectPlaylistNode(int direction) {
            if (playlist.empty()) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(playlist.size());
            auto index = static_cast<std::ptrdiff_t>(playlist_index) + direction;
            index = (index % count + count) % count;
            playlist_index = static_cast<std::size_t>(index);
            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            std::cout << "acmxvk: playlist node " << (playlist_index + 1) << '/'
                      << playlist.size() << ": " << playlist[playlist_index].name << " ("
                      << playlist[playlist_index].shaders.size() << " passes)\n";
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
            return pipeline;
        }

        void applyShaderPipeline() {
            if (getDevice() == VK_NULL_HANDLE) {
                return;
            }
            vkDeviceWaitIdle(getDevice());
            detachPostProcessingShader();
            post_process_sprites.clear();
            frame_sprite->setEffectsEnabled(effects_enabled);

            const std::vector<fs::path> pipeline = activeShaderPipeline();
            if (pipeline.empty()) {
                return;
            }

            std::vector<PostProcessingEffect> effects;
            effects.reserve(pipeline.size());
            for (const fs::path &shader : pipeline) {
                PostProcessingEffect effect{
                    shader.string(), {1.0F, 1.0F, 1.0F, 0.0F}, false};
#ifdef AUDIO_ENABLED
                effect.spectrumBinCount = audio::AudioEngine::spectrum_bin_count();
                effect.spectrumHistoryLayerCount =
                    static_cast<std::uint32_t>(options.audio_buffers);
#endif
                effects.push_back(effect);
            }
            post_process_sprites = attachPostProcessingShaders(effects);
            for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                sprite->enableExtendedUBO();
                sprite->setCustomUniforms(custom_uniform_values);
#ifdef AUDIO_ENABLED
                sprite->enableSpectrumTexture(audio::AudioEngine::spectrum_bin_count());
                if (options.audio_buffers > 0) {
                    sprite->enableSpectrumHistoryTexture(
                        audio::AudioEngine::spectrum_bin_count(),
                        static_cast<std::uint32_t>(options.audio_buffers));
                }
#endif
            }

            std::cout << "acmxvk: Vulkan shader pipeline (" << pipeline.size() << " passes):\n";
            for (std::size_t index = 0; index < pipeline.size(); ++index) {
                std::cout << "  " << (index + 1) << ": " << pipeline[index].filename().string()
                          << '\n';
            }
        }

        [[nodiscard]] bool handleCaptureEnd() {
            if (source_kind == SourceKind::Camera) {
                return true;
            }
            if (!options.repeat) {
                setFrameReadbackEnabled(false);
                exit();
                return false;
            }

            closeVideoCapture();
            if (!openVideoCapture() || !readInputFrame()) {
                throw std::runtime_error("unable to restart video input: " + options.input_file);
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
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.open(options.input_file)) {
                using_ffmpeg_capture = true;
                std::cout << "acmxvk: video capture: FFmpeg "
                          << (ffmpeg_capture.using_hardware_decode()
                                  ? "with CUDA/NVDEC\n"
                                  : "software decode\n");
                return true;
            }
#endif
            const bool opened = capture.open(options.input_file);
            if (opened) {
                std::cout << "acmxvk: video capture: OpenCV fallback\n";
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
            if (!options.enable_texture_cache || history_initialized) {
                return;
            }
            for (uint32_t layer = 0; layer < frame_sprite->getHistoryLayerCount(); ++layer) {
                updateHistoryFrame(rgba);
            }
            history_initialized = true;
            history_delay_counter = 0;
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

#ifdef ACMXVK_WITH_CUDA
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

        void updateFilteredCudaHistoryFrame() {
            updateCudaHistoryFrame(gpu_filter_engine->output(),
                                   gpu_filter_engine->stream());
        }

        void initializeCudaHistory(const cv::cuda::GpuMat &rgba,
                                   cv::cuda::Stream &source_stream,
                                   bool filtered) {
            if (!options.enable_texture_cache || history_initialized) {
                return;
            }
            for (uint32_t layer = 0;
                 layer < frame_sprite->getHistoryLayerCount(); ++layer) {
                updateCudaHistoryFrame(rgba, source_stream);
            }
            history_initialized = true;
            history_delay_counter = 0;
            std::cout << "acmxvk: initialized "
                      << frame_sprite->getHistoryLayerCount()
                      << (filtered ? " filtered" : " NVDEC")
                      << " Vulkan history-cache layers (delay "
                      << options.cache_delay << ")\n";
        }

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
        }

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
                return;
            }
#endif
            frame_sprite->updateTexture(rgba.ptr(), rgba.cols, rgba.rows,
                                        static_cast<int>(rgba.step));
        }

        [[nodiscard]] bool readInputFrame() {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
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
                if (history_was_initialized &&
                    ++history_delay_counter > options.cache_delay) {
                    updateFilteredCudaHistoryFrame();
                    history_delay_counter = 0;
                }
                return true;
            }
#if defined(MXVK_WITH_FFMPEG_CAPTURE)
            if (using_ffmpeg_capture &&
                ffmpeg_capture.using_hardware_decode()) {
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

            bool requires_host_frame = options.enable_texture_cache ||
                                       options.frame_rotation != FrameRotation::None;
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
            rotateFrame(rgba, options.frame_rotation);
            uploadInputFrame(rgba);
            const bool history_was_initialized = history_initialized;
            initializeHistory(rgba);
            if (history_was_initialized &&
                ++history_delay_counter > options.cache_delay) {
                updateHistoryFrame(rgba);
                history_delay_counter = 0;
            }
            return true;
        }

        void updateShaderUniforms(int width, int height) {
            const auto now = std::chrono::steady_clock::now();
            const float wall_delta =
                std::chrono::duration<float>(now - previous_frame).count();
            previous_frame = now;
            ++frame_count;

            const float delta = options.normalized_time
                                    ? static_cast<float>(1.0 / outputFrameRate())
                                    : wall_delta;
            if (shader_time_active) {
                shader_time += static_cast<double>(delta) * options.time_speed;
            }
            const float elapsed = static_cast<float>(shader_time);
            const float frame_rate = delta > 0.0F ? 1.0F / delta : 0.0F;
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
            if (file_audio_source != nullptr && audio_engine != nullptr) {
                file_audio_source->process_frame(outputFrameRate(), *audio_engine);
                if (options.audio_trunc && !file_audio_source->is_active()) {
                    std::cout << "acmxvk: audio source finished, stopping "
                                 "(--audio-trunc)\n";
                    exit();
                }
            }
            if (audioSourceOpen()) {
                const audio::AudioMetrics metrics = audio_engine->metrics();
                const float warmup = updateAudioWarmup(now);
                const float sense =
                    audio_engine->sensitivity() * 4.0F * warmup;
                audio_amplitude = metrics.amplitude * warmup;
                audio_frequency = metrics.frequency;
                audio_peak = std::sqrt(std::max(metrics.peak, 0.0F)) * sense;
                audio_rms = std::sqrt(std::max(metrics.rms, 0.0F)) * sense;
                audio_smooth = std::sqrt(std::max(metrics.smooth, 0.0F)) * sense;
                audio_low = std::sqrt(std::max(metrics.low, 0.0F)) * sense;
                audio_mid = std::sqrt(std::max(metrics.mid, 0.0F)) * sense;
                audio_high = std::sqrt(std::max(metrics.high, 0.0F)) * sense;
                audio_sample_rate = static_cast<float>(audio_engine->sample_rate());
                spectrum_values = audio_engine->spectrum();
                for (float &value : spectrum_values) {
                    value *= warmup;
                }
            }
#endif
            frame_sprite->setShaderParams(1.0F, 1.0F, 1.0F, elapsed);
            frame_sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
            frame_sprite->setUniform0(1.0F, 1.0F, static_cast<float>(width),
                                      static_cast<float>(height));
            frame_sprite->setUniform1(delta, audio_amplitude, audio_frequency,
                                      frame_rate);
            frame_sprite->setUniform2(static_cast<float>(frame_count), elapsed,
                                      audio_sample_rate, audio_peak);
            frame_sprite->setUniform3(static_cast<float>(frame_sprite->getHistoryHead()),
                                      static_cast<float>(frame_sprite->getHistoryLayerCount()),
                                      audio_rms, audio_smooth);
            frame_sprite->setAudioBands(audio_low, audio_mid, audio_high);

            for (std::size_t index = 0; index < post_process_sprites.size(); ++index) {
                mxvk::VK_Sprite *sprite = post_process_sprites[index];
                setPostProcessingShaderParams(index, 1.0F, 1.0F, 1.0F, elapsed);
                sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
                sprite->setUniform0(1.0F, 1.0F, static_cast<float>(width),
                                    static_cast<float>(height));
                sprite->setUniform1(delta, audio_amplitude, audio_frequency,
                                    frame_rate);
                sprite->setUniform2(static_cast<float>(frame_count), elapsed,
                                    audio_sample_rate, audio_peak);
                sprite->setUniform3(0.0F, 0.0F, audio_rms, audio_smooth);
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
        acmxvk::Options options = acmxvk::parseOptions(argc, argv);
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
#ifdef ACMXVK_WITH_CUDA
            std::cout << "CUDA filters: enabled\n";
#else
            std::cout << "CUDA filters: disabled\n";
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
#ifdef ACMXVK_WITH_CUDA
            acmxvk::gpu::FilterEngine::list_devices(std::cout);
            return EXIT_SUCCESS;
#else
            throw std::runtime_error(
                "--list-cuda-devices requires an ACMXVK build configured with "
                "-DWITH_CUDA=ON");
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
        if (!options.gpu_filter_indices.empty() ||
            options.cuda_device_specified) {
            throw std::runtime_error(
                "CUDA filters require an ACMXVK build configured with "
                "-DWITH_CUDA=ON");
        }
#endif
        if (options.show_help) {
            acmxvk::printHelp(std::cout);
            return EXIT_SUCCESS;
        }
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
        if (!options.gpu_filter_indices.empty() || options.cuda_device_specified) {
            acmxvk::gpu::FilterEngine::select_device(options.cuda_device);
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
