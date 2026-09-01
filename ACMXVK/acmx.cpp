/**
 * @file acmx.cpp
 * @brief ACMXVK real-time Vulkan video shader application.
 */

#include <mxvk/mxvk_exception.hpp>

#ifdef AUDIO_ENABLED
#include "audio.hpp"
#endif
#ifdef MIDI_ENABLED
#include "midi.hpp"
#endif
#ifdef ACMXVK_WITH_CUDA
#include "gpu_filters.hpp"
#endif
#include "app/camera_probe.hpp"
#include "app/media_utils.hpp"
#include "app/options.hpp"
#include "app/shader_library.hpp"
#include "main_window.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string_view>
#include <utility>

int main(int argc, char **argv) {
    try {
        for (int index = 1; index < argc; ++index) {
            if (argv[index] != nullptr &&
                (std::string_view(argv[index]) == "--unbuffered" ||
                 std::string_view(argv[index]) == "--silent" ||
                 std::string_view(argv[index]) == "--headless")) {
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
        if (options.enumerate_camera_device >= 0) {
            return acmxvk::probeCameraDevice(options.enumerate_camera_device,
                                             std::cout, std::cerr)
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE;
        }
        if (options.list_camera_devices) {
            return acmxvk::listCameraDevices(std::cout, std::cerr)
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE;
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
        if (!options.probe_hdr_file.empty()) {
            const acmxvk::VideoHdrInfo info =
                acmxvk::probeVideoHdrInfo(options.probe_hdr_file);
            if (!info.valid) {
                std::cerr << "acmxvk: unable to probe HDR metadata: "
                          << options.probe_hdr_file << '\n';
                return EXIT_FAILURE;
            }
            std::cout << "acmxvk: HDR probe: " << options.probe_hdr_file
                      << '\n';
            acmxvk::printVideoHdrInfo(info, std::cout);
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

        if (options.headless &&
            std::signal(SIGINT, acmxvk::request_headless_shutdown) ==
                SIG_ERR) {
            throw std::runtime_error(
                "unable to install Ctrl+C handler for headless mode");
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
