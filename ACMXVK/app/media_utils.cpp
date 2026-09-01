#include "media_utils.hpp"

#include "../input_validation.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
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
#include <string_view>

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

    [[nodiscard]] VideoHdrInfo probeVideoHdrInfo(
        const std::string &filename) {
        VideoHdrInfo info;
        if (filename.empty() || filename.find("://") != std::string::npos) {
            return info;
        }

        AVFormatContext *format = nullptr;
        if (avformat_open_input(&format, filename.c_str(), nullptr, nullptr) < 0) {
            return info;
        }
        const auto close_format = [&format] {
            if (format != nullptr) {
                avformat_close_input(&format);
            }
        };
        if (avformat_find_stream_info(format, nullptr) < 0) {
            close_format();
            return info;
        }

        const int stream_index = av_find_best_stream(
            format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index < 0) {
            close_format();
            return info;
        }

        const AVStream *stream = format->streams[stream_index];
        const AVCodecParameters *parameters = stream->codecpar;
        info.valid = true;
        info.color_primaries = parameters->color_primaries;
        info.color_transfer = parameters->color_trc;
        info.color_space = parameters->color_space;
        info.color_range = parameters->color_range;
        info.bit_depth = parameters->bits_per_raw_sample;
        if (info.bit_depth <= 0 && parameters->format != AV_PIX_FMT_NONE) {
            const auto pixel_format =
                static_cast<AVPixelFormat>(parameters->format);
            const AVPixFmtDescriptor *descriptor =
                av_pix_fmt_desc_get(pixel_format);
            if (descriptor != nullptr && descriptor->nb_components > 0) {
                info.bit_depth = descriptor->comp[0].depth;
            }
        }

        const bool bt2020_primaries =
            parameters->color_primaries == AVCOL_PRI_BT2020;
        const bool hdr_transfer =
            parameters->color_trc == AVCOL_TRC_SMPTE2084 ||
            parameters->color_trc == AVCOL_TRC_ARIB_STD_B67 ||
            parameters->color_trc == AVCOL_TRC_BT2020_10 ||
            parameters->color_trc == AVCOL_TRC_BT2020_12;
        const bool bt2020_space =
            parameters->color_space == AVCOL_SPC_BT2020_NCL ||
            parameters->color_space == AVCOL_SPC_BT2020_CL;
        info.hdr = info.bit_depth >= 10 &&
                   (bt2020_primaries || hdr_transfer || bt2020_space);

        const auto copy_side_data = [&info](const AVPacketSideData *side_data,
                                            int side_data_count) {
            for (int index = 0; index < side_data_count; ++index) {
                const AVPacketSideData &entry = side_data[index];
                if (entry.type == AV_PKT_DATA_MASTERING_DISPLAY_METADATA &&
                    entry.size == sizeof(AVMasteringDisplayMetadata)) {
                    info.mastering_display.assign(entry.data,
                                                  entry.data + entry.size);
                } else if (entry.type ==
                               AV_PKT_DATA_CONTENT_LIGHT_LEVEL &&
                           entry.size == sizeof(AVContentLightMetadata)) {
                    info.content_light.assign(entry.data,
                                              entry.data + entry.size);
                }
            }
        };
#if LIBAVCODEC_VERSION_MAJOR >= 60
        copy_side_data(parameters->coded_side_data,
                       parameters->nb_coded_side_data);
#else
        copy_side_data(stream->side_data, stream->nb_side_data);
#endif

        if (info.mastering_display.empty() || info.content_light.empty()) {
            const AVCodec *decoder =
                avcodec_find_decoder(parameters->codec_id);
            AVCodecContext *decoder_context =
                decoder != nullptr ? avcodec_alloc_context3(decoder)
                                   : nullptr;
            AVPacket *packet = av_packet_alloc();
            AVFrame *frame = av_frame_alloc();
            const auto copy_frame_side_data = [&info](const AVFrame *source) {
                if (info.mastering_display.empty()) {
                    const AVFrameSideData *side_data = av_frame_get_side_data(
                        source, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
                    if (side_data != nullptr &&
                        side_data->size ==
                            sizeof(AVMasteringDisplayMetadata)) {
                        info.mastering_display.assign(
                            side_data->data,
                            side_data->data + side_data->size);
                    }
                }
                if (info.content_light.empty()) {
                    const AVFrameSideData *side_data = av_frame_get_side_data(
                        source, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
                    if (side_data != nullptr &&
                        side_data->size == sizeof(AVContentLightMetadata)) {
                        info.content_light.assign(
                            side_data->data,
                            side_data->data + side_data->size);
                    }
                }
            };
            if (decoder_context != nullptr && packet != nullptr &&
                frame != nullptr &&
                avcodec_parameters_to_context(decoder_context, parameters) >=
                    0 &&
                avcodec_open2(decoder_context, decoder, nullptr) >= 0) {
                constexpr int MAX_METADATA_PACKETS = 256;
                bool decoded_frame = false;
                for (int packet_count = 0;
                     packet_count < MAX_METADATA_PACKETS && !decoded_frame &&
                     av_read_frame(format, packet) >= 0;
                     ++packet_count) {
                    if (packet->stream_index == stream_index &&
                        avcodec_send_packet(decoder_context, packet) >= 0) {
                        while (avcodec_receive_frame(decoder_context, frame) >=
                               0) {
                            copy_frame_side_data(frame);
                            decoded_frame = true;
                            av_frame_unref(frame);
                        }
                    }
                    av_packet_unref(packet);
                }
                if (!decoded_frame) {
                    avcodec_send_packet(decoder_context, nullptr);
                    while (avcodec_receive_frame(decoder_context, frame) >=
                           0) {
                        copy_frame_side_data(frame);
                        decoded_frame = true;
                        av_frame_unref(frame);
                    }
                }
            }
            av_frame_free(&frame);
            av_packet_free(&packet);
            avcodec_free_context(&decoder_context);
        }

        close_format();
        return info;
    }

    void printVideoHdrInfo(const VideoHdrInfo &info, std::ostream &output) {
        const auto label = [](const char *name) -> std::string_view {
            return name != nullptr ? std::string_view(name)
                                   : std::string_view("unknown");
        };
        output << "HDR: " << (info.hdr ? "yes" : "no") << '\n'
               << "Bit depth: " << info.bit_depth << '\n'
               << "Color primaries: "
               << label(av_color_primaries_name(
                      static_cast<AVColorPrimaries>(info.color_primaries)))
               << " (" << info.color_primaries << ")\n"
               << "Transfer: "
               << label(av_color_transfer_name(
                      static_cast<AVColorTransferCharacteristic>(
                          info.color_transfer)))
               << " (" << info.color_transfer << ")\n"
               << "Matrix: "
               << label(av_color_space_name(
                      static_cast<AVColorSpace>(info.color_space)))
               << " (" << info.color_space << ")\n"
               << "Range: "
               << label(av_color_range_name(
                      static_cast<AVColorRange>(info.color_range)))
               << " (" << info.color_range << ")\n"
               << "Mastering-display metadata: "
               << info.mastering_display.size() << " bytes\n"
               << "Content-light metadata: " << info.content_light.size()
               << " bytes\n";
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
