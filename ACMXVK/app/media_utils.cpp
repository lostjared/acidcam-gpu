#include "media_utils.hpp"

#include "../input_validation.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef ACMXVK_WITH_MXVK_CUDA
#include <opencv2/core/cuda.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace acmxvk {
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
    // Frame transforms, CUDA device helpers, and asynchronous camera capture.
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
} // namespace acmxvk
