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
#include "app/media_helpers.hpp"
#include "app/media_utils.hpp"
#include "app/options.hpp"
#include "app/shader_library.hpp"
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
// These sections form one ordered implementation unit. Their order preserves
// the open MainWindow class definition across section files.
// clang-format off
#include "app/window_lifecycle.ipp"
#include "app/window_state.ipp"
#include "app/window_audio_midi.ipp"
#include "app/window_shaders.ipp"
#include "app/window_overlay.ipp"
#include "app/window_io.ipp"
#include "app/window_rendering.ipp"
    // clang-format on
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
