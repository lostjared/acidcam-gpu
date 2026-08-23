#include "file_audio.hpp"

#include "audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libswresample/swresample.h>
}

namespace acmxvk::audio {
    namespace {

        constexpr unsigned int FILE_SAMPLE_RATE = 44100;

        [[nodiscard]] std::string ffmpegError(int error) {
            char message[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(error, message, sizeof(message));
            return message;
        }

    } // namespace

    class FileAudioSource::Impl {
      public:
        bool open(const std::string &requested_path) {
            close();
            const std::filesystem::path source =
                std::filesystem::absolute(requested_path).lexically_normal();
            if (!std::filesystem::is_regular_file(source)) {
                std::cerr << "acmxvk: audio file is not readable: "
                          << source.string() << '\n';
                return false;
            }

            av_log_set_level(AV_LOG_ERROR);
            AVFormatContext *format = nullptr;
            AVCodecContext *codec = nullptr;
            SwrContext *resampler = nullptr;
            AVPacket *packet = nullptr;
            AVFrame *frame = nullptr;

            auto release = [&]() {
                av_frame_free(&frame);
                av_packet_free(&packet);
                swr_free(&resampler);
                avcodec_free_context(&codec);
                avformat_close_input(&format);
            };

            int result = avformat_open_input(&format, source.c_str(), nullptr, nullptr);
            if (result < 0) {
                std::cerr << "acmxvk: could not open audio file: "
                          << ffmpegError(result) << '\n';
                release();
                return false;
            }
            result = avformat_find_stream_info(format, nullptr);
            if (result < 0) {
                std::cerr << "acmxvk: could not read audio stream information: "
                          << ffmpegError(result) << '\n';
                release();
                return false;
            }

            const AVCodec *decoder = nullptr;
            const int stream_index = av_find_best_stream(
                format, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
            if (stream_index < 0 || decoder == nullptr) {
                std::cerr << "acmxvk: media file contains no decodable audio stream\n";
                release();
                return false;
            }

            codec = avcodec_alloc_context3(decoder);
            if (codec == nullptr) {
                std::cerr << "acmxvk: could not allocate the audio decoder\n";
                release();
                return false;
            }
            result = avcodec_parameters_to_context(
                codec, format->streams[stream_index]->codecpar);
            if (result < 0 || (result = avcodec_open2(codec, decoder, nullptr)) < 0) {
                std::cerr << "acmxvk: could not initialize the audio decoder: "
                          << ffmpegError(result) << '\n';
                release();
                return false;
            }
            if (codec->sample_rate <= 0 || codec->ch_layout.nb_channels == 0) {
                std::cerr << "acmxvk: audio stream has an invalid sample format\n";
                release();
                return false;
            }

            AVChannelLayout output_layout = AV_CHANNEL_LAYOUT_MONO;
            result = swr_alloc_set_opts2(
                &resampler, &output_layout, AV_SAMPLE_FMT_FLT,
                static_cast<int>(FILE_SAMPLE_RATE), &codec->ch_layout,
                codec->sample_fmt, codec->sample_rate, 0, nullptr);
            av_channel_layout_uninit(&output_layout);
            if (result < 0 || resampler == nullptr ||
                (result = swr_init(resampler)) < 0) {
                std::cerr << "acmxvk: could not initialize audio resampling: "
                          << ffmpegError(result) << '\n';
                release();
                return false;
            }

            packet = av_packet_alloc();
            frame = av_frame_alloc();
            if (packet == nullptr || frame == nullptr) {
                std::cerr << "acmxvk: could not allocate FFmpeg audio frames\n";
                release();
                return false;
            }

            auto append_frame = [&]() -> bool {
                const int capacity = static_cast<int>(av_rescale_rnd(
                    swr_get_delay(resampler, codec->sample_rate) + frame->nb_samples,
                    FILE_SAMPLE_RATE, codec->sample_rate, AV_ROUND_UP));
                if (capacity <= 0) {
                    return true;
                }
                std::vector<float> converted(static_cast<std::size_t>(capacity));
                std::uint8_t *output[] = {
                    reinterpret_cast<std::uint8_t *>(converted.data())};
                const int count = swr_convert(
                    resampler, output, capacity,
                    const_cast<const std::uint8_t **>(frame->extended_data),
                    frame->nb_samples);
                if (count < 0) {
                    std::cerr << "acmxvk: audio resampling failed: "
                              << ffmpegError(count) << '\n';
                    return false;
                }
                samples.insert(samples.end(), converted.begin(),
                               converted.begin() + count);
                return true;
            };

            auto drain_decoder = [&]() -> bool {
                while (true) {
                    const int receive = avcodec_receive_frame(codec, frame);
                    if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) {
                        return true;
                    }
                    if (receive < 0) {
                        std::cerr << "acmxvk: audio decoding failed: "
                                  << ffmpegError(receive) << '\n';
                        return false;
                    }
                    if (!append_frame()) {
                        return false;
                    }
                    av_frame_unref(frame);
                }
            };

            bool decoded = true;
            while ((result = av_read_frame(format, packet)) >= 0) {
                if (packet->stream_index == stream_index) {
                    result = avcodec_send_packet(codec, packet);
                    if (result < 0 || !drain_decoder()) {
                        if (result < 0) {
                            std::cerr << "acmxvk: could not submit audio packet: "
                                      << ffmpegError(result) << '\n';
                        }
                        decoded = false;
                    }
                }
                av_packet_unref(packet);
                if (!decoded) {
                    break;
                }
            }
            if (decoded) {
                result = avcodec_send_packet(codec, nullptr);
                decoded = (result >= 0 || result == AVERROR_EOF) && drain_decoder();
            }

            while (decoded) {
                const int capacity = static_cast<int>(av_rescale_rnd(
                    swr_get_delay(resampler, codec->sample_rate), FILE_SAMPLE_RATE,
                    codec->sample_rate, AV_ROUND_UP));
                if (capacity <= 0) {
                    break;
                }
                std::vector<float> converted(static_cast<std::size_t>(capacity));
                std::uint8_t *output[] = {
                    reinterpret_cast<std::uint8_t *>(converted.data())};
                const int count =
                    swr_convert(resampler, output, capacity, nullptr, 0);
                if (count <= 0) {
                    decoded = count == 0;
                    break;
                }
                samples.insert(samples.end(), converted.begin(),
                               converted.begin() + count);
            }

            release();
            if (!decoded || samples.empty()) {
                std::cerr << "acmxvk: audio file produced no usable samples\n";
                close();
                return false;
            }

            source_path = source.string();
            playback_position = 0.0;
            active = true;
            std::cout << "acmxvk: decoded audio file " << source_path << " ("
                      << duration_seconds() << " seconds, " << samples.size()
                      << " mono samples at " << FILE_SAMPLE_RATE << " Hz)\n";
            return true;
        }

        void close() {
            samples.clear();
            samples.shrink_to_fit();
            source_path.clear();
            playback_position = 0.0;
            active = false;
        }

        [[nodiscard]] double duration_seconds() const {
            return static_cast<double>(samples.size()) /
                   static_cast<double>(FILE_SAMPLE_RATE);
        }

        bool process_frame(double frames_per_second, AudioEngine &engine) {
            if (samples.empty() || !active) {
                engine.reset();
                return false;
            }
            const double rate =
                std::isfinite(frames_per_second) && frames_per_second > 0.0
                    ? frames_per_second
                    : 60.0;
            const double next_position =
                std::min(playback_position +
                             static_cast<double>(FILE_SAMPLE_RATE) / rate,
                         static_cast<double>(samples.size()));
            const std::size_t first = static_cast<std::size_t>(playback_position);
            const std::size_t last = std::min(
                std::max(first + 1, static_cast<std::size_t>(next_position)),
                samples.size());
            engine.process_samples(samples.data() + first,
                                   static_cast<unsigned int>(last - first), 1,
                                   FILE_SAMPLE_RATE);
            playback_position = next_position;
            if (playback_position >= static_cast<double>(samples.size())) {
                active = false;
                std::cout << "acmxvk: audio file reached end of stream\n";
            }
            return true;
        }

        std::vector<float> samples;
        std::string source_path;
        double playback_position = 0.0;
        bool active = false;
    };

    FileAudioSource::FileAudioSource() : impl(std::make_unique<Impl>()) {}
    FileAudioSource::~FileAudioSource() = default;

    bool FileAudioSource::open(const std::string &path) {
        return impl->open(path);
    }

    void FileAudioSource::close() {
        impl->close();
    }

    bool FileAudioSource::is_open() const {
        return !impl->samples.empty();
    }

    bool FileAudioSource::is_active() const {
        return impl->active;
    }

    double FileAudioSource::duration_seconds() const {
        return impl->duration_seconds();
    }

    const std::string &FileAudioSource::path() const {
        return impl->source_path;
    }

    bool FileAudioSource::process_frame(double frames_per_second,
                                        AudioEngine &engine) {
        return impl->process_frame(frames_per_second, engine);
    }

} // namespace acmxvk::audio
