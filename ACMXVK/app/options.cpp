#include "options.hpp"

#include "../input_validation.hpp"
#include "resource_paths.hpp"
#include <mxvk/mxvk.hpp>
#include <mxwrite.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace acmxvk {
    namespace {
        constexpr int MAX_FRAME_DIMENSION = 16384;
        constexpr std::int64_t MAX_FRAME_PIXELS = 67108864;
    } // namespace

    [[nodiscard]] bool dimensions_supported(int width, int height) {
        return width > 0 && height > 0 && width <= MAX_FRAME_DIMENSION &&
               height <= MAX_FRAME_DIMENSION &&
               static_cast<std::int64_t>(width) * height <= MAX_FRAME_PIXELS;
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
        input::validate_string(options.probe_hdr_file,
                               input::StringKind::Path, "--probe-hdr", true);
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
               options.list_camera_devices ||
               options.check_audio || options.list_midi_devices ||
               options.check_midi || options.list_gpu_filters ||
               options.list_cuda_devices || options.check_cuda ||
               options.check_dnn ||
               !options.probe_hdr_file.empty() ||
               options.enumerate_camera_device >= 0 ||
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
             resource_directories(options)) {
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
            } else if (option == "--silent" || option == "--headless") {
                options.headless = true;
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
            } else if (option == "--enumerate-device" ||
                       option == "--probe-camera") {
                options.enumerate_camera_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
                if (options.enumerate_camera_device < 0 ||
                    options.enumerate_camera_device > 65535) {
                    throw std::runtime_error(
                        "camera probe device index must be between 0 and 65535");
                }
            } else if (option == "--list-camera-devices") {
                options.list_camera_devices = true;
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
            } else if (option == "--probe-hdr") {
                options.probe_hdr_file =
                    optionValue(index, argc, argv, option);
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
        if (options.headless && !isUtilityRequest(options)) {
            if (options.input_file.empty() && options.graphic_file.empty()) {
                throw std::runtime_error(
                    "--headless/--silent requires --input <video> or "
                    "--graphic <image>; camera input is not supported");
            }
            if (options.output_file.empty()) {
                throw std::runtime_error(
                    "--headless/--silent requires --output <file>");
            }
            if (!options.graphic_file.empty() && options.duration <= 0.0) {
                throw std::runtime_error(
                    "headless graphic processing requires --duration "
                    "<seconds>");
            }
            if (options.repeat && options.duration <= 0.0) {
                throw std::runtime_error(
                    "--headless/--silent with --repeat requires --duration "
                    "<seconds>");
            }
            if (options.fullscreen || options.enable_vsync ||
                options.enable_screenshot) {
                throw std::runtime_error(
                    "--headless/--silent cannot be combined with "
                    "--fullscreen, --enable-vsync, or --enable-screenshot");
            }
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
        output << "ACMXVK - Vulkan video shader engine (Increment 9Z / HDR 5)\n\n"
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
               << "      --enumerate-device N   List native camera modes and exit\n"
               << "      --probe-camera N       Alias for --enumerate-device\n"
               << "      --list-camera-devices  List native camera indices and names\n"
               << "      --use-yuv               Prefer YUYV camera capture over MJPG\n"
               << "      --maximize-fps          Render at --fps using the latest camera frame\n"
               << "      --use-source-fps        Play video on its reported source clock\n"
               << "      --use-source-audio      Use the video's audio for shader reactivity\n"
               << "      --probe-hdr <video>     Print HDR/color metadata and exit\n"
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
               << "Headless processing:\n"
               << "      --headless              Surface-free terminal/batch rendering\n"
               << "      --silent                Alias for --headless\n"
               << "                              Requires video/image input and --output\n"
               << "                              Image input and --repeat require --duration\n\n"
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
} // namespace acmxvk
