#include "file_audio.hpp"

#include "audio.hpp"
#include "input_validation.hpp"

#include <rtaudio/RtAudio.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
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

        [[nodiscard]] bool resampleMonoRecording(std::vector<float> &samples,
                                                 unsigned int source_rate) {
            if (source_rate == FILE_SAMPLE_RATE) {
                return true;
            }
            if (source_rate == 0 || samples.empty() ||
                samples.size() > static_cast<std::size_t>(
                                     std::numeric_limits<int>::max())) {
                std::cerr << "acmxvk: live audio recording has an invalid sample "
                             "rate or sample count\n";
                return false;
            }

            const std::int64_t expected_count = av_rescale_rnd(
                static_cast<std::int64_t>(samples.size()), FILE_SAMPLE_RATE,
                source_rate, AV_ROUND_UP);
            if (expected_count <= 0 ||
                expected_count > std::numeric_limits<int>::max()) {
                std::cerr << "acmxvk: resampled live audio is too large\n";
                return false;
            }

            SwrContext *resampler = nullptr;
            AVChannelLayout input_layout = AV_CHANNEL_LAYOUT_MONO;
            AVChannelLayout output_layout = AV_CHANNEL_LAYOUT_MONO;
            int result = swr_alloc_set_opts2(
                &resampler, &output_layout, AV_SAMPLE_FMT_FLT, FILE_SAMPLE_RATE,
                &input_layout, AV_SAMPLE_FMT_FLT, static_cast<int>(source_rate), 0,
                nullptr);
            av_channel_layout_uninit(&output_layout);
            av_channel_layout_uninit(&input_layout);
            if (result < 0 || resampler == nullptr ||
                (result = swr_init(resampler)) < 0) {
                std::cerr << "acmxvk: could not initialize live audio resampler";
                if (result < 0) {
                    std::cerr << ": " << ffmpegError(result);
                }
                std::cerr << '\n';
                swr_free(&resampler);
                return false;
            }

            std::vector<float> converted_samples(
                static_cast<std::size_t>(expected_count));
            const std::uint8_t *input_data[] = {
                reinterpret_cast<const std::uint8_t *>(samples.data())};
            std::uint8_t *output_data[] = {
                reinterpret_cast<std::uint8_t *>(converted_samples.data())};
            int converted = swr_convert(
                resampler, output_data, static_cast<int>(expected_count), input_data,
                static_cast<int>(samples.size()));
            if (converted >= 0 && converted < expected_count) {
                std::uint8_t *flush_data[] = {reinterpret_cast<std::uint8_t *>(
                    converted_samples.data() + converted)};
                const int flushed = swr_convert(
                    resampler, flush_data,
                    static_cast<int>(expected_count) - converted, nullptr, 0);
                if (flushed < 0) {
                    converted = flushed;
                } else {
                    converted += flushed;
                }
            }
            swr_free(&resampler);
            if (converted < 0) {
                std::cerr << "acmxvk: could not resample live audio: "
                          << ffmpegError(converted) << '\n';
                return false;
            }

            converted_samples.resize(static_cast<std::size_t>(converted));
            samples = std::move(converted_samples);
            std::cout << "acmxvk: resampled live audio from " << source_rate
                      << " Hz to " << FILE_SAMPLE_RATE << " Hz\n";
            return !samples.empty();
        }

        [[nodiscard]] std::string trimPlaylistLine(std::string line) {
            constexpr std::string_view WHITESPACE = " \t\r\n";
            const std::size_t first = line.find_first_not_of(WHITESPACE);
            if (first == std::string::npos) {
                return {};
            }
            const std::size_t last = line.find_last_not_of(WHITESPACE);
            return line.substr(first, last - first + 1);
        }

        [[nodiscard]] bool isM3uPath(const std::filesystem::path &path) {
            std::string extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char value) {
                               return static_cast<char>(std::tolower(value));
                           });
            return extension == ".m3u" || extension == ".m3u8";
        }

        [[nodiscard]] bool isUrl(std::string_view path) {
            return path.find("://") != std::string_view::npos;
        }

        [[nodiscard]] std::vector<std::string>
        readM3uPlaylist(const std::filesystem::path &playlist) {
            input::validate_file_size(playlist, "M3U playlist");
            std::ifstream playlist_input(playlist);
            if (!playlist_input) {
                std::cerr << "acmxvk: could not open M3U playlist: "
                          << playlist.string() << '\n';
                return {};
            }

            std::vector<std::string> paths;
            std::string line;
            std::size_t line_number = 1;
            while (input::read_bounded_line(playlist_input, line,
                                            "M3U playlist", line_number++)) {
                line = trimPlaylistLine(std::move(line));
                if (line.empty() || line.front() == '#') {
                    continue;
                }
                if (paths.size() >= input::MAX_AUDIO_PLAYLIST_ENTRIES) {
                    throw std::runtime_error(
                        "M3U playlist contains too many entries");
                }
                if (isUrl(line)) {
                    input::validate_string(line, input::StringKind::Url,
                                           "M3U URL");
                    paths.push_back(std::move(line));
                    continue;
                }

                input::validate_string(line, input::StringKind::Path,
                                       "M3U path");

                std::filesystem::path track(line);
                if (!track.is_absolute()) {
                    track = playlist.parent_path() / track;
                }
                paths.push_back(track.lexically_normal().string());
            }
            return paths;
        }

        class FileAudioOutput {
          public:
#ifdef __linux__
            FileAudioOutput() : stream(RtAudio::LINUX_PULSE) {}
#else
            FileAudioOutput() = default;
#endif

            ~FileAudioOutput() {
                close();
            }

            bool open(const float *source, std::size_t sample_count,
                      int requested_device, float requested_gain) {
                close();
                if (source == nullptr || sample_count == 0) {
                    return false;
                }

                try {
                    const std::vector<unsigned int> device_ids =
                        stream.getDeviceIds();
                    if (device_ids.empty()) {
                        std::cerr << "acmxvk: no audio output devices found\n";
                        return false;
                    }

                    const unsigned int device =
                        requested_device >= 0
                            ? static_cast<unsigned int>(requested_device)
                            : stream.getDefaultOutputDevice();
                    if (std::find(device_ids.begin(), device_ids.end(), device) ==
                        device_ids.end()) {
                        std::cerr << "acmxvk: audio output device " << device
                                  << " was not found\n";
                        return false;
                    }
                    const RtAudio::DeviceInfo info = stream.getDeviceInfo(device);
                    input::validate_string(info.name,
                                           input::StringKind::DisplayText,
                                           "audio output device name");
                    if (info.outputChannels == 0) {
                        std::cerr << "acmxvk: audio device " << device
                                  << " has no output channels\n";
                        return false;
                    }

                    output_channels = std::min(2U, info.outputChannels);
                    output_sample_rate = choose_sample_rate(info.sampleRates);
                    source_samples = source;
                    source_sample_count = sample_count;
                    source_position = 0.0;
                    total_source_position = 0.0;
                    gain = std::clamp(requested_gain, 0.0F, 4.0F);
                    playback_position.store(0, std::memory_order_relaxed);
                    total_playback_position.store(0, std::memory_order_relaxed);
                    completed_loops.store(0, std::memory_order_relaxed);
                    finished.store(false, std::memory_order_relaxed);

                    RtAudio::StreamParameters output_parameters;
                    output_parameters.deviceId = device;
                    output_parameters.nChannels = output_channels;
                    output_parameters.firstChannel = 0;

                    unsigned int buffer_frames = 512;
                    stream.openStream(&output_parameters, nullptr, RTAUDIO_FLOAT32,
                                      output_sample_rate, &buffer_frames,
                                      &FileAudioOutput::audio_callback, this);
                    configured = true;
                    std::cout << "acmxvk: file audio output " << device << ": "
                              << info.name << " (" << output_sample_rate << " Hz, "
                              << output_channels << " channel"
                              << (output_channels == 1 ? "" : "s") << ", gain "
                              << gain << ")\n";
                    return true;
                } catch (const std::exception &error) {
                    std::cerr << "acmxvk: audio output error: " << error.what()
                              << '\n';
                    close();
                    return false;
                }
            }

            bool start() {
                if (!configured || started.load(std::memory_order_acquire)) {
                    return configured;
                }
                try {
                    active.store(true, std::memory_order_release);
                    finished.store(false, std::memory_order_release);
                    stream.startStream();
                    started.store(true, std::memory_order_release);
                    std::cout << "acmxvk: file audio playback started\n";
                    return true;
                } catch (const std::exception &error) {
                    active.store(false, std::memory_order_release);
                    std::cerr << "acmxvk: could not start audio output: "
                              << error.what() << '\n';
                    return false;
                }
            }

            void close() {
                active.store(false, std::memory_order_release);
                if (stream.isStreamOpen()) {
                    try {
                        if (stream.isStreamRunning()) {
                            stream.stopStream();
                        }
                        stream.closeStream();
                    } catch (const std::exception &error) {
                        std::cerr << "acmxvk: error closing audio output: "
                                  << error.what() << '\n';
                    }
                }
                configured = false;
                started.store(false, std::memory_order_release);
                finished.store(false, std::memory_order_release);
                source_samples = nullptr;
                source_sample_count = 0;
                source_position = 0.0;
                total_source_position = 0.0;
                playback_position.store(0, std::memory_order_relaxed);
                total_playback_position.store(0, std::memory_order_relaxed);
                completed_loops.store(0, std::memory_order_relaxed);
            }

            void set_repeat(bool enabled) {
                repeat.store(enabled, std::memory_order_release);
            }

            [[nodiscard]] bool is_configured() const {
                return configured;
            }

            [[nodiscard]] bool is_started() const {
                return started.load(std::memory_order_acquire);
            }

            [[nodiscard]] bool is_finished() const {
                return finished.load(std::memory_order_acquire);
            }

            [[nodiscard]] std::size_t position() const {
                return playback_position.load(std::memory_order_acquire);
            }

            [[nodiscard]] std::uint64_t total_position() const {
                return total_playback_position.load(std::memory_order_acquire);
            }

            [[nodiscard]] std::uint64_t loop_count() const {
                return completed_loops.load(std::memory_order_acquire);
            }

          private:
            [[nodiscard]] static unsigned int
            choose_sample_rate(const std::vector<unsigned int> &rates) {
                if (rates.empty() ||
                    std::find(rates.begin(), rates.end(), FILE_SAMPLE_RATE) !=
                        rates.end()) {
                    return FILE_SAMPLE_RATE;
                }
                constexpr unsigned int FALLBACK_SAMPLE_RATE = 48000;
                if (std::find(rates.begin(), rates.end(), FALLBACK_SAMPLE_RATE) !=
                    rates.end()) {
                    return FALLBACK_SAMPLE_RATE;
                }
                return rates.front();
            }

            static int audio_callback(void *output_buffer, void *,
                                      unsigned int frame_count, double,
                                      RtAudioStreamStatus, void *user_data) {
                return static_cast<FileAudioOutput *>(user_data)
                    ->write_samples(static_cast<float *>(output_buffer), frame_count);
            }

            int write_samples(float *output, unsigned int frame_count) {
                if (output == nullptr) {
                    return 0;
                }

                const double source_step =
                    static_cast<double>(FILE_SAMPLE_RATE) /
                    static_cast<double>(output_sample_rate);
                for (unsigned int frame = 0; frame < frame_count; ++frame) {
                    float sample = 0.0F;
                    if (active.load(std::memory_order_relaxed) &&
                        source_position >=
                            static_cast<double>(source_sample_count)) {
                        if (repeat.load(std::memory_order_relaxed)) {
                            source_position = std::fmod(
                                source_position,
                                static_cast<double>(source_sample_count));
                            completed_loops.fetch_add(1,
                                                      std::memory_order_release);
                        } else {
                            active.store(false, std::memory_order_release);
                            finished.store(true, std::memory_order_release);
                            source_position =
                                static_cast<double>(source_sample_count);
                        }
                    }

                    const std::size_t index =
                        static_cast<std::size_t>(source_position);
                    if (active.load(std::memory_order_relaxed) &&
                        index < source_sample_count) {
                        const std::size_t next_index =
                            repeat.load(std::memory_order_relaxed)
                                ? (index + 1) % source_sample_count
                                : std::min(index + 1, source_sample_count - 1);
                        const float fraction = static_cast<float>(
                            source_position - static_cast<double>(index));
                        sample = std::clamp(
                            (source_samples[index] +
                             (source_samples[next_index] - source_samples[index]) *
                                 fraction) *
                                gain,
                            -1.0F, 1.0F);
                        source_position += source_step;
                        total_source_position += source_step;
                    }

                    for (unsigned int channel = 0; channel < output_channels;
                         ++channel) {
                        output[frame * output_channels + channel] = sample;
                    }
                }

                playback_position.store(
                    std::min(static_cast<std::size_t>(source_position),
                             source_sample_count),
                    std::memory_order_release);
                total_playback_position.store(
                    static_cast<std::uint64_t>(total_source_position),
                    std::memory_order_release);
                return 0;
            }

            RtAudio stream;
            const float *source_samples = nullptr;
            std::size_t source_sample_count = 0;
            double source_position = 0.0;
            double total_source_position = 0.0;
            unsigned int output_channels = 0;
            unsigned int output_sample_rate = FILE_SAMPLE_RATE;
            float gain = 1.0F;
            std::atomic<std::size_t> playback_position{0};
            std::atomic<std::uint64_t> total_playback_position{0};
            std::atomic<std::uint64_t> completed_loops{0};
            std::atomic<bool> active{false};
            std::atomic<bool> started{false};
            std::atomic<bool> finished{false};
            std::atomic<bool> repeat{false};
            bool configured = false;
        };

    } // namespace

    class FileAudioSource::Impl {
      public:
        ~Impl() {
            close();
        }

        bool open(const std::string &requested_path) {
            close();
            input::validate_string(requested_path, input::StringKind::Path,
                                   "audio file path");
            const std::filesystem::path source =
                std::filesystem::absolute(requested_path).lexically_normal();
            if (!std::filesystem::is_regular_file(source)) {
                std::cerr << "acmxvk: audio file is not readable: "
                          << source.string() << '\n';
                return false;
            }

            const bool playlist = isM3uPath(source);
            const std::vector<std::string> requested_tracks =
                playlist ? readM3uPlaylist(source)
                         : std::vector<std::string>{source.string()};
            if (requested_tracks.empty()) {
                std::cerr << "acmxvk: M3U playlist contains no tracks: "
                          << source.string() << '\n';
                return false;
            }

            av_log_set_level(AV_LOG_ERROR);
            for (const std::string &track : requested_tracks) {
                if (decode_track(track)) {
                    track_paths.push_back(track);
                    track_end_positions.push_back(samples.size());
                } else if (playlist) {
                    std::cerr << "acmxvk: skipping unusable playlist track: "
                              << track << '\n';
                }
            }
            if (track_paths.empty()) {
                close();
                return false;
            }

            source_path = source.string();
            playlist_source = playlist;
            playback_position = 0.0;
            current_track_index = 0;
            active = true;
            restart_pending = false;
            if (playlist) {
                std::cout << "acmxvk: loaded M3U playlist with "
                          << track_paths.size() << " track(s), "
                          << duration_seconds() << " seconds total: "
                          << source_path << '\n';
                report_current_track();
            }
            return true;
        }

        bool decode_track(const std::string &requested_path) {
            input::validate_string(
                requested_path,
                isUrl(requested_path) ? input::StringKind::Url
                                      : input::StringKind::Path,
                "audio track");
            const std::string source =
                isUrl(requested_path)
                    ? requested_path
                    : std::filesystem::absolute(requested_path)
                          .lexically_normal()
                          .string();
            if (!isUrl(source) &&
                !std::filesystem::is_regular_file(std::filesystem::path(source))) {
                std::cerr << "acmxvk: audio file is not readable: " << source
                          << '\n';
                return false;
            }
            const std::size_t initial_sample_count = samples.size();

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
            const std::size_t decoded_sample_count =
                samples.size() - initial_sample_count;
            if (!decoded || decoded_sample_count == 0) {
                samples.resize(initial_sample_count);
                std::cerr << "acmxvk: audio file produced no usable samples: "
                          << source << '\n';
                return false;
            }

            std::cout << "acmxvk: decoded audio track " << source << " ("
                      << static_cast<double>(decoded_sample_count) /
                             static_cast<double>(FILE_SAMPLE_RATE)
                      << " seconds, " << decoded_sample_count
                      << " mono samples at " << FILE_SAMPLE_RATE << " Hz)\n";
            return true;
        }

        void close() {
            output.reset();
            samples.clear();
            samples.shrink_to_fit();
            source_path.clear();
            track_paths.clear();
            track_end_positions.clear();
            playback_position = 0.0;
            current_track_index = 0;
            active = false;
            repeat = false;
            restart_pending = false;
            playlist_source = false;
            observed_output_loops = 0;
        }

        void set_repeat(bool enabled) {
            repeat = enabled;
            if (output != nullptr) {
                output->set_repeat(enabled);
            }
        }

        bool enable_output(int device, float gain) {
            if (samples.empty()) {
                return false;
            }
            auto requested_output = std::make_unique<FileAudioOutput>();
            requested_output->set_repeat(repeat);
            if (!requested_output->open(samples.data(), samples.size(), device,
                                        gain)) {
                return false;
            }
            output = std::move(requested_output);
            observed_output_loops = 0;
            return true;
        }

        void stop_output() {
            output.reset();
        }

        [[nodiscard]] bool has_output_clock() const {
            return active && output != nullptr && output->is_configured();
        }

        [[nodiscard]] double playback_time() const {
            if (!has_output_clock()) {
                return 0.0;
            }
            return static_cast<double>(output->total_position()) /
                   static_cast<double>(FILE_SAMPLE_RATE);
        }

        bool mux_into_video(const std::string &requested_video_path,
                            double video_duration) {
            output.reset();
            if (samples.empty() || !std::isfinite(video_duration) ||
                video_duration <= 0.0) {
                std::cerr << "acmxvk: cannot mux "
                          << (live_recording_source ? "live audio input"
                                                    : "file audio")
                          << " without samples and a positive video duration\n";
                return false;
            }

            const std::filesystem::path video_path =
                std::filesystem::absolute(requested_video_path).lexically_normal();
            if (!std::filesystem::is_regular_file(video_path)) {
                std::cerr << "acmxvk: encoded video is not readable for audio mux: "
                          << video_path.string() << '\n';
                return false;
            }

            const double source_duration = duration_seconds();
            const double mux_duration =
                repeat ? video_duration : std::min(video_duration, source_duration);
            const std::int64_t target_sample_count =
                static_cast<std::int64_t>(std::floor(
                    mux_duration * static_cast<double>(FILE_SAMPLE_RATE)));
            if (target_sample_count <= 0) {
                std::cerr << "acmxvk: file audio mux duration is empty\n";
                return false;
            }

            const auto unique_value = std::chrono::steady_clock::now()
                                          .time_since_epoch()
                                          .count();
            const std::filesystem::path temporary_path =
                video_path.parent_path() /
                (video_path.stem().string() + ".acmxvk-mux-" +
                 std::to_string(unique_value) + video_path.extension().string());

            AVFormatContext *input_context = nullptr;
            AVFormatContext *output_context = nullptr;
            AVCodecContext *audio_encoder = nullptr;
            SwrContext *resampler = nullptr;
            AVFrame *audio_frame = nullptr;
            AVPacket *input_packet = nullptr;
            AVPacket *audio_packet = nullptr;

            auto cleanup = [&]() {
                av_packet_free(&audio_packet);
                av_packet_free(&input_packet);
                av_frame_free(&audio_frame);
                swr_free(&resampler);
                avcodec_free_context(&audio_encoder);
                avformat_close_input(&input_context);
                if (output_context != nullptr) {
                    if ((output_context->oformat->flags & AVFMT_NOFILE) == 0) {
                        avio_closep(&output_context->pb);
                    }
                    avformat_free_context(output_context);
                    output_context = nullptr;
                }
            };

            auto fail = [&](std::string_view message, int error) {
                std::cerr << "acmxvk: " << message;
                if (error < 0) {
                    std::cerr << ": " << ffmpegError(error);
                }
                std::cerr << '\n';
                cleanup();
                std::error_code remove_error;
                std::filesystem::remove(temporary_path, remove_error);
                return false;
            };

            int result = avformat_open_input(&input_context, video_path.c_str(),
                                             nullptr, nullptr);
            if (result < 0) {
                return fail("could not open encoded video for audio mux", result);
            }
            result = avformat_find_stream_info(input_context, nullptr);
            if (result < 0) {
                return fail("could not read encoded video stream information",
                            result);
            }
            const int input_video_index = av_find_best_stream(
                input_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (input_video_index < 0) {
                return fail("encoded output contains no video stream",
                            input_video_index);
            }

            result = avformat_alloc_output_context2(
                &output_context, nullptr, nullptr, temporary_path.c_str());
            if (result < 0 || output_context == nullptr) {
                return fail("could not create audio-mux output container", result);
            }

            AVStream *input_video = input_context->streams[input_video_index];
            AVStream *output_video = avformat_new_stream(output_context, nullptr);
            if (output_video == nullptr) {
                return fail("could not create remuxed video stream", AVERROR(ENOMEM));
            }
            result = avcodec_parameters_copy(output_video->codecpar,
                                             input_video->codecpar);
            if (result < 0) {
                return fail("could not copy encoded video parameters", result);
            }
            output_video->codecpar->codec_tag = 0;
            output_video->time_base = input_video->time_base;
            output_video->avg_frame_rate = input_video->avg_frame_rate;

            const AVCodec *aac_encoder = avcodec_find_encoder(AV_CODEC_ID_AAC);
            if (aac_encoder == nullptr) {
                return fail("linked FFmpeg has no AAC encoder", AVERROR_ENCODER_NOT_FOUND);
            }
            AVStream *output_audio =
                avformat_new_stream(output_context, aac_encoder);
            if (output_audio == nullptr) {
                return fail("could not create encoded audio stream", AVERROR(ENOMEM));
            }
            audio_encoder = avcodec_alloc_context3(aac_encoder);
            if (audio_encoder == nullptr) {
                return fail("could not allocate AAC encoder", AVERROR(ENOMEM));
            }
            audio_encoder->bit_rate = 192000;
            audio_encoder->sample_fmt = AV_SAMPLE_FMT_FLTP;
            audio_encoder->sample_rate = static_cast<int>(FILE_SAMPLE_RATE);
            audio_encoder->time_base =
                AVRational{1, static_cast<int>(FILE_SAMPLE_RATE)};
            av_channel_layout_default(&audio_encoder->ch_layout, 1);
            if ((output_context->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
                audio_encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }
            result = avcodec_open2(audio_encoder, aac_encoder, nullptr);
            if (result < 0) {
                return fail("could not initialize AAC encoder", result);
            }
            result = avcodec_parameters_from_context(output_audio->codecpar,
                                                     audio_encoder);
            if (result < 0) {
                return fail("could not export AAC stream parameters", result);
            }
            output_audio->codecpar->codec_tag = 0;
            output_audio->time_base = audio_encoder->time_base;

            AVChannelLayout input_layout = AV_CHANNEL_LAYOUT_MONO;
            result = swr_alloc_set_opts2(
                &resampler, &audio_encoder->ch_layout, audio_encoder->sample_fmt,
                audio_encoder->sample_rate, &input_layout, AV_SAMPLE_FMT_FLT,
                static_cast<int>(FILE_SAMPLE_RATE), 0, nullptr);
            av_channel_layout_uninit(&input_layout);
            if (result < 0 || resampler == nullptr ||
                (result = swr_init(resampler)) < 0) {
                return fail("could not initialize audio mux resampler", result);
            }

            const int audio_frame_capacity =
                audio_encoder->frame_size > 0 ? audio_encoder->frame_size : 1024;
            audio_frame = av_frame_alloc();
            input_packet = av_packet_alloc();
            audio_packet = av_packet_alloc();
            if (audio_frame == nullptr || input_packet == nullptr ||
                audio_packet == nullptr) {
                return fail("could not allocate audio mux frames", AVERROR(ENOMEM));
            }
            audio_frame->format = audio_encoder->sample_fmt;
            audio_frame->sample_rate = audio_encoder->sample_rate;
            audio_frame->nb_samples = audio_frame_capacity;
            result = av_channel_layout_copy(&audio_frame->ch_layout,
                                            &audio_encoder->ch_layout);
            if (result < 0 || (result = av_frame_get_buffer(audio_frame, 0)) < 0) {
                return fail("could not allocate AAC sample buffer", result);
            }

            if ((output_context->oformat->flags & AVFMT_NOFILE) == 0) {
                result = avio_open(&output_context->pb, temporary_path.c_str(),
                                   AVIO_FLAG_WRITE);
                if (result < 0) {
                    return fail("could not open temporary mux output", result);
                }
            }
            result = avformat_write_header(output_context, nullptr);
            if (result < 0) {
                return fail("could not write audio-mux container header", result);
            }

            std::int64_t source_position = 0;
            std::int64_t encoded_position = 0;
            std::vector<float> input_samples(
                static_cast<std::size_t>(audio_frame_capacity));

            auto drain_audio_packets = [&]() {
                while (true) {
                    const int receive =
                        avcodec_receive_packet(audio_encoder, audio_packet);
                    if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) {
                        return true;
                    }
                    if (receive < 0) {
                        result = receive;
                        return false;
                    }
                    audio_packet->stream_index = output_audio->index;
                    av_packet_rescale_ts(audio_packet, audio_encoder->time_base,
                                         output_audio->time_base);
                    const int write =
                        av_interleaved_write_frame(output_context, audio_packet);
                    av_packet_unref(audio_packet);
                    if (write < 0) {
                        result = write;
                        return false;
                    }
                }
            };

            auto encode_audio_frame = [&]() {
                const std::int64_t remaining =
                    target_sample_count - source_position;
                if (remaining <= 0) {
                    return true;
                }
                const int source_count = static_cast<int>(
                    std::min<std::int64_t>(remaining, audio_frame_capacity));
                int submitted_count = source_count;
                if (source_count < audio_frame_capacity &&
                    (aac_encoder->capabilities &
                     AV_CODEC_CAP_SMALL_LAST_FRAME) == 0) {
                    submitted_count = audio_frame_capacity;
                }
                for (int index = 0; index < submitted_count; ++index) {
                    if (index >= source_count) {
                        input_samples[static_cast<std::size_t>(index)] = 0.0F;
                        continue;
                    }
                    const std::size_t sample_index = repeat
                                                         ? static_cast<std::size_t>(source_position + index) %
                                                               samples.size()
                                                         : static_cast<std::size_t>(source_position + index);
                    input_samples[static_cast<std::size_t>(index)] =
                        samples[sample_index];
                }

                audio_frame->nb_samples = submitted_count;
                result = av_frame_make_writable(audio_frame);
                if (result < 0) {
                    return false;
                }
                const std::uint8_t *input_data[] = {
                    reinterpret_cast<const std::uint8_t *>(input_samples.data())};
                const int converted = swr_convert(
                    resampler, audio_frame->data, submitted_count, input_data,
                    submitted_count);
                if (converted < 0) {
                    result = converted;
                    return false;
                }
                audio_frame->nb_samples = converted;
                audio_frame->pts = encoded_position;
                result = avcodec_send_frame(audio_encoder, audio_frame);
                if (result < 0 || !drain_audio_packets()) {
                    return false;
                }
                source_position += source_count;
                encoded_position += converted;
                return true;
            };

            bool video_complete = false;
            while ((result = av_read_frame(input_context, input_packet)) >= 0) {
                if (input_packet->stream_index != input_video_index) {
                    av_packet_unref(input_packet);
                    continue;
                }
                const std::int64_t timestamp =
                    input_packet->pts != AV_NOPTS_VALUE ? input_packet->pts
                                                        : input_packet->dts;
                const double packet_time =
                    timestamp == AV_NOPTS_VALUE
                        ? 0.0
                        : static_cast<double>(timestamp) *
                              av_q2d(input_video->time_base);
                if (timestamp != AV_NOPTS_VALUE && packet_time > mux_duration) {
                    av_packet_unref(input_packet);
                    video_complete = true;
                    break;
                }

                const std::int64_t audio_target = std::min<std::int64_t>(
                    target_sample_count,
                    static_cast<std::int64_t>(std::ceil(
                        (packet_time +
                         static_cast<double>(audio_frame_capacity) /
                             static_cast<double>(FILE_SAMPLE_RATE)) *
                        static_cast<double>(FILE_SAMPLE_RATE))));
                while (source_position < audio_target) {
                    if (!encode_audio_frame()) {
                        return fail("could not encode AAC samples", result);
                    }
                }

                av_packet_rescale_ts(input_packet, input_video->time_base,
                                     output_video->time_base);
                input_packet->stream_index = output_video->index;
                input_packet->pos = -1;
                result = av_interleaved_write_frame(output_context, input_packet);
                av_packet_unref(input_packet);
                if (result < 0) {
                    return fail("could not remux encoded video packet", result);
                }
            }
            if (!video_complete && result != AVERROR_EOF) {
                return fail("could not finish reading encoded video", result);
            }

            while (source_position < target_sample_count) {
                if (!encode_audio_frame()) {
                    return fail("could not finish AAC encoding", result);
                }
            }
            result = avcodec_send_frame(audio_encoder, nullptr);
            if (result < 0 || !drain_audio_packets()) {
                return fail("could not flush AAC encoder", result);
            }
            result = av_write_trailer(output_context);
            if (result < 0) {
                return fail("could not finalize audio-mux container", result);
            }
            if ((output_context->oformat->flags & AVFMT_NOFILE) == 0) {
                result = avio_closep(&output_context->pb);
                if (result < 0) {
                    return fail("could not flush temporary mux output", result);
                }
            }

            cleanup();
            if (std::rename(temporary_path.c_str(), video_path.c_str()) != 0) {
                std::cerr << "acmxvk: could not atomically replace encoded video "
                             "with muxed output\n";
                std::error_code remove_error;
                std::filesystem::remove(temporary_path, remove_error);
                return false;
            }

            std::cout << "acmxvk: muxed "
                      << (live_recording_source
                              ? "live audio input"
                              : (track_paths.size() > 1 ? "audio playlist"
                                                        : "audio file"))
                      << " into " << video_path.string() << " (" << mux_duration
                      << " seconds" << (repeat ? ", repeated" : "") << ")\n";
            return true;
        }

        [[nodiscard]] double duration_seconds() const {
            return static_cast<double>(samples.size()) /
                   static_cast<double>(FILE_SAMPLE_RATE);
        }

        [[nodiscard]] const std::string &current_track_path() const {
            static const std::string EMPTY_PATH;
            if (!active || track_paths.empty() ||
                current_track_index >= track_paths.size()) {
                return EMPTY_PATH;
            }
            return track_paths[current_track_index];
        }

        void report_current_track() const {
            if (!playlist_source || track_paths.empty()) {
                return;
            }
            std::cout << "acmxvk: audio playlist track "
                      << (current_track_index + 1) << '/' << track_paths.size()
                      << ": " << current_track_path() << '\n';
        }

        void update_current_track(double position) {
            while (current_track_index + 1 < track_end_positions.size() &&
                   position >= static_cast<double>(
                                   track_end_positions[current_track_index])) {
                ++current_track_index;
                report_current_track();
            }
        }

        bool process_output_frame(double frames_per_second, AudioEngine &engine) {
            if (output->is_finished()) {
                active = false;
                engine.reset();
                std::cout << "acmxvk: audio "
                          << (playlist_source ? "playlist" : "file")
                          << " reached end of output stream\n";
                return false;
            }

            const std::uint64_t output_loops = output->loop_count();
            if (output_loops != observed_output_loops) {
                observed_output_loops = output_loops;
                current_track_index = 0;
                engine.reset();
                std::cout << "acmxvk: audio "
                          << (playlist_source ? "playlist" : "file")
                          << " reached end of output stream; "
                             "restarting (--audio-repeat)\n";
                report_current_track();
            }

            playback_position = static_cast<double>(output->position());
            update_current_track(playback_position);
            const double samples_per_frame =
                static_cast<double>(FILE_SAMPLE_RATE) / frames_per_second;
            const std::size_t first = std::min(
                static_cast<std::size_t>(playback_position), samples.size());
            const std::size_t last = std::min(
                std::max(first + 1,
                         static_cast<std::size_t>(playback_position +
                                                  samples_per_frame)),
                samples.size());
            if (first < last) {
                engine.process_samples(samples.data() + first,
                                       static_cast<unsigned int>(last - first), 1,
                                       FILE_SAMPLE_RATE);
            }

            if (!output->is_started() && !output->start()) {
                std::cerr << "acmxvk: continuing with silent file-audio "
                             "analysis\n";
                output.reset();
                playback_position = static_cast<double>(last);
            }
            return true;
        }

        bool process_frame(double frames_per_second, AudioEngine &engine) {
            if (samples.empty() || !active) {
                engine.reset();
                return false;
            }
            if (restart_pending) {
                playback_position = 0.0;
                current_track_index = 0;
                restart_pending = false;
                engine.reset();
                report_current_track();
            }
            const double rate =
                std::isfinite(frames_per_second) && frames_per_second > 0.0
                    ? frames_per_second
                    : 60.0;
            if (has_output_clock()) {
                return process_output_frame(rate, engine);
            }
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
            update_current_track(playback_position);
            if (playback_position >= static_cast<double>(samples.size())) {
                if (repeat) {
                    restart_pending = true;
                    std::cout << "acmxvk: audio "
                              << (playlist_source ? "playlist" : "file")
                              << " reached end of stream; "
                                 "restarting (--audio-repeat)\n";
                } else {
                    active = false;
                    std::cout << "acmxvk: audio "
                              << (playlist_source ? "playlist" : "file")
                              << " reached end of stream\n";
                }
            }
            return true;
        }

        std::vector<float> samples;
        std::string source_path;
        std::vector<std::string> track_paths;
        std::vector<std::size_t> track_end_positions;
        std::unique_ptr<FileAudioOutput> output;
        double playback_position = 0.0;
        std::size_t current_track_index = 0;
        std::uint64_t observed_output_loops = 0;
        bool active = false;
        bool repeat = false;
        bool restart_pending = false;
        bool playlist_source = false;
        bool live_recording_source = false;
    };

    FileAudioSource::FileAudioSource() : impl(std::make_unique<Impl>()) {}
    FileAudioSource::~FileAudioSource() = default;

    bool FileAudioSource::open(const std::string &path) {
        return impl->open(path);
    }

    void FileAudioSource::close() {
        impl->close();
    }

    void FileAudioSource::set_repeat(bool enabled) {
        impl->set_repeat(enabled);
    }

    bool FileAudioSource::enable_output(int device, float gain) {
        return impl->enable_output(device, gain);
    }

    void FileAudioSource::stop_output() {
        impl->stop_output();
    }

    bool FileAudioSource::has_output_clock() const {
        return impl->has_output_clock();
    }

    double FileAudioSource::playback_time() const {
        return impl->playback_time();
    }

    bool FileAudioSource::mux_into_video(const std::string &video_path,
                                         double video_duration) {
        return impl->mux_into_video(video_path, video_duration);
    }

    bool FileAudioSource::mux_recording_into_video(
        std::vector<float> samples, unsigned int sample_rate,
        const std::string &video_path, double video_duration) {
        if (!resampleMonoRecording(samples, sample_rate)) {
            return false;
        }

        FileAudioSource source;
        source.impl->samples = std::move(samples);
        source.impl->track_paths.emplace_back("live audio input");
        source.impl->live_recording_source = true;
        return source.impl->mux_into_video(video_path, video_duration);
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

    std::size_t FileAudioSource::track_count() const {
        return impl->track_paths.size();
    }

    const std::string &FileAudioSource::current_track_path() const {
        return impl->current_track_path();
    }

    bool FileAudioSource::process_frame(double frames_per_second,
                                        AudioEngine &engine) {
        return impl->process_frame(frames_per_second, engine);
    }

} // namespace acmxvk::audio
