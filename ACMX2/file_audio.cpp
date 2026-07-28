/**
 * @file file_audio.cpp
 * @brief FFmpeg-based audio file decoder for audio-reactive shaders.
 *
 * Decodes an entire audio file upfront into a mono float buffer at
 * 44 100 Hz using FFmpeg's libavformat / libavcodec / libswresample.
 * Each video frame, file_audio_process_frame() advances through the buffer
 * and feeds the shared AudioAnalyzer used by the shader uniform pipeline.
 */

#include "file_audio.hpp"
#include "audio.hpp"

#include <RtAudio.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

static AVFormatContext *fmtCtx = nullptr;
static AVCodecContext *codecCtx = nullptr;
static SwrContext *swrCtx = nullptr;
static int audioStreamIndex = -1;
static std::vector<float> decodedSamples; // all decoded mono float samples at 44100 Hz
static size_t playbackPos = 0;
static double framePlaybackPos = 0.0;
static std::atomic<bool> fileAudioActive{false};

namespace {

    constexpr unsigned int FILE_AUDIO_SAMPLE_RATE = 44100;

    class FileAudioOutput {
      public:
#ifdef __linux__
        FileAudioOutput() : audio(RtAudio::LINUX_PULSE) {}
#else
        FileAudioOutput() = default;
#endif

        ~FileAudioOutput() { close(); }

        bool open(const float *samples, std::size_t sample_count, int output_device) {
            close();
            if (samples == nullptr || sample_count == 0)
                return false;

            try {
                const std::vector<unsigned int> device_ids = audio.getDeviceIds();
                if (device_ids.empty()) {
                    std::cerr << "acmx2: file_audio: No audio output devices found\n";
                    return false;
                }

                const unsigned int device =
                    output_device >= 0
                        ? static_cast<unsigned int>(output_device)
                        : audio.getDefaultOutputDevice();
                const RtAudio::DeviceInfo info = audio.getDeviceInfo(device);
                if (info.outputChannels == 0) {
                    std::cerr << "acmx2: file_audio: Selected device has no output channels\n";
                    return false;
                }

                output_channels = std::min(2U, info.outputChannels);
                output_sample_rate = choose_sample_rate(info.sampleRates);
                source_samples = samples;
                source_sample_count = sample_count;
                source_position = 0.0;
                playback_position.store(0, std::memory_order_relaxed);

                RtAudio::StreamParameters output_parameters;
                output_parameters.deviceId = device;
                output_parameters.nChannels = output_channels;
                output_parameters.firstChannel = 0;

                unsigned int buffer_frames = 512;
                audio.openStream(&output_parameters, nullptr, RTAUDIO_FLOAT32,
                                 output_sample_rate, &buffer_frames,
                                 &FileAudioOutput::audio_callback, this);
                configured = true;
                std::cout << "acmx2: file_audio: Playback configured on device "
                          << device << ": " << info.name << " (" << output_channels
                          << " ch, " << output_sample_rate << " Hz)\n";
                return true;
            } catch (const std::exception &error) {
                std::cerr << "acmx2: file_audio: Could not open output stream: "
                          << error.what() << "\n";
                close();
                return false;
            }
        }

        bool start() {
            if (!configured || started.load(std::memory_order_acquire))
                return configured;
            try {
                active.store(true, std::memory_order_release);
                started.store(true, std::memory_order_release);
                audio.startStream();
                std::cout << "acmx2: file_audio: Playback started\n";
                return true;
            } catch (const std::exception &error) {
                active.store(false, std::memory_order_release);
                started.store(false, std::memory_order_release);
                std::cerr << "acmx2: file_audio: Could not start output stream: "
                          << error.what() << "\n";
                return false;
            }
        }

        void close() {
            active.store(false, std::memory_order_release);
            if (audio.isStreamOpen()) {
                try {
                    if (audio.isStreamRunning())
                        audio.stopStream();
                    audio.closeStream();
                } catch (const std::exception &error) {
                    std::cerr << "acmx2: file_audio: Error closing output stream: "
                              << error.what() << "\n";
                }
            }
            configured = false;
            started.store(false, std::memory_order_release);
            source_samples = nullptr;
            source_sample_count = 0;
            source_position = 0.0;
            playback_position.store(0, std::memory_order_relaxed);
        }

        bool is_configured() const { return configured; }

        bool is_started() const {
            return started.load(std::memory_order_acquire);
        }

        std::size_t position() const {
            return playback_position.load(std::memory_order_acquire);
        }

      private:
        static unsigned int choose_sample_rate(const std::vector<unsigned int> &rates) {
            if (rates.empty() ||
                std::find(rates.begin(), rates.end(), FILE_AUDIO_SAMPLE_RATE) != rates.end())
                return FILE_AUDIO_SAMPLE_RATE;
            if (std::find(rates.begin(), rates.end(), 48000U) != rates.end())
                return 48000;
            return rates.front();
        }

        static int audio_callback(void *output_buffer, void *, unsigned int frame_count,
                                  double, RtAudioStreamStatus, void *user_data) {
            return static_cast<FileAudioOutput *>(user_data)
                ->write_samples(static_cast<float *>(output_buffer), frame_count);
        }

        int write_samples(float *output, unsigned int frame_count) {
            if (output == nullptr)
                return 0;

            const double source_step =
                static_cast<double>(FILE_AUDIO_SAMPLE_RATE) /
                static_cast<double>(output_sample_rate);
            for (unsigned int frame = 0; frame < frame_count; ++frame) {
                float sample = 0.0f;
                const std::size_t index = static_cast<std::size_t>(source_position);
                if (active.load(std::memory_order_relaxed) &&
                    index < source_sample_count) {
                    const std::size_t next_index =
                        std::min(index + 1, source_sample_count - 1);
                    const float fraction =
                        static_cast<float>(source_position - static_cast<double>(index));
                    sample = source_samples[index] +
                             (source_samples[next_index] - source_samples[index]) * fraction;
                    source_position += source_step;
                } else {
                    active.store(false, std::memory_order_release);
                    source_position = static_cast<double>(source_sample_count);
                }

                for (unsigned int channel = 0; channel < output_channels; ++channel)
                    output[frame * output_channels + channel] = sample;
            }

            playback_position.store(
                std::min(static_cast<std::size_t>(source_position), source_sample_count),
                std::memory_order_release);
            return 0;
        }

        RtAudio audio;
        const float *source_samples = nullptr;
        std::size_t source_sample_count = 0;
        double source_position = 0.0;
        unsigned int output_channels = 0;
        unsigned int output_sample_rate = FILE_AUDIO_SAMPLE_RATE;
        std::atomic<std::size_t> playback_position{0};
        std::atomic<bool> started{false};
        std::atomic<bool> active{false};
        bool configured = false;
    };

    std::unique_ptr<FileAudioOutput> fileAudioOutput;

} // namespace

/**
 * @brief Decode every audio packet from the open format context.
 *
 * Reads packets from @c fmtCtx, sends them through the codec, and
 * resamples the output to mono float 44 100 Hz via @c swrCtx.
 * Decoded samples are appended to the module-level @c decodedSamples
 * vector.  The resampler is flushed at the end to capture trailing
 * samples.
 *
 * @return @c true if decoding completed without fatal errors.
 */
static bool decodeAllSamples() {
    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!pkt || !frame) {
        if (pkt)
            av_packet_free(&pkt);
        if (frame)
            av_frame_free(&frame);
        return false;
    }

    while (av_read_frame(fmtCtx, pkt) >= 0) {
        if (pkt->stream_index == audioStreamIndex) {
            int ret = avcodec_send_packet(codecCtx, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(codecCtx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                if (ret < 0) {
                    av_packet_unref(pkt);
                    av_frame_free(&frame);
                    av_packet_free(&pkt);
                    return false;
                }

                // Resample to mono float 44100 Hz
                int outSamples = swr_get_out_samples(swrCtx, frame->nb_samples);
                std::vector<float> buf(outSamples);
                uint8_t *outBuf = reinterpret_cast<uint8_t *>(buf.data());
                int converted = swr_convert(swrCtx, &outBuf, outSamples,
                                            const_cast<const uint8_t **>(frame->extended_data),
                                            frame->nb_samples);
                if (converted > 0) {
                    decodedSamples.insert(decodedSamples.end(), buf.begin(), buf.begin() + converted);
                }
            }
        }
        av_packet_unref(pkt);
    }

    // Flush the resampler
    int flushed = swr_convert(swrCtx, nullptr, 0, nullptr, 0);
    if (flushed > 0) {
        std::vector<float> buf(flushed);
        uint8_t *outBuf = reinterpret_cast<uint8_t *>(buf.data());
        flushed = swr_convert(swrCtx, &outBuf, flushed, nullptr, 0);
        if (flushed > 0)
            decodedSamples.insert(decodedSamples.end(), buf.begin(), buf.begin() + flushed);
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
    return true;
}

/// @brief Open and fully decode an audio file to mono float PCM at 44.1 kHz.
bool file_audio_open(const std::string &filepath) {
    file_audio_close();

    av_log_set_level(AV_LOG_ERROR);

    if (avformat_open_input(&fmtCtx, filepath.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "acmx2: file_audio: Cannot open: " << filepath << "\n";
        return false;
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        std::cerr << "acmx2: file_audio: Cannot find stream info\n";
        avformat_close_input(&fmtCtx);
        return false;
    }

    audioStreamIndex = -1;
    for (unsigned i = 0; i < fmtCtx->nb_streams; ++i) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = static_cast<int>(i);
            break;
        }
    }
    if (audioStreamIndex < 0) {
        std::cerr << "acmx2: file_audio: No audio stream found in: " << filepath << "\n";
        avformat_close_input(&fmtCtx);
        return false;
    }

    const AVCodec *codec = avcodec_find_decoder(fmtCtx->streams[audioStreamIndex]->codecpar->codec_id);
    if (!codec) {
        std::cerr << "acmx2: file_audio: Unsupported audio codec\n";
        avformat_close_input(&fmtCtx);
        return false;
    }
    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, fmtCtx->streams[audioStreamIndex]->codecpar);
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        std::cerr << "acmx2: file_audio: Cannot open codec\n";
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        return false;
    }

    // Set up resampler: input format → mono float 44100 Hz
    AVChannelLayout outLayout = AV_CHANNEL_LAYOUT_MONO;
    int ret = swr_alloc_set_opts2(&swrCtx,
                                  &outLayout, AV_SAMPLE_FMT_FLT, 44100,
                                  &codecCtx->ch_layout, codecCtx->sample_fmt, codecCtx->sample_rate,
                                  0, nullptr);
    if (ret < 0 || swr_init(swrCtx) < 0) {
        std::cerr << "acmx2: file_audio: Cannot init resampler\n";
        avcodec_free_context(&codecCtx);
        avformat_close_input(&fmtCtx);
        return false;
    }

    decodedSamples.clear();
    decodedSamples.reserve(44100 * 300); // reserve ~5 minutes

    if (!decodeAllSamples()) {
        std::cerr << "acmx2: file_audio: Decode failed\n";
        file_audio_close();
        return false;
    }

    playbackPos = 0;
    framePlaybackPos = 0.0;
    fileAudioActive = true;

    std::cout << "acmx2: file_audio: Loaded " << decodedSamples.size()
              << " samples (" << (decodedSamples.size() / 44100.0) << "s) from: " << filepath << "\n";

    // Clean up decoder resources — samples are fully buffered
    swr_free(&swrCtx);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&fmtCtx);

    return true;
}

bool file_audio_enable_output(int output_device) {
    if (decodedSamples.empty())
        return false;

    fileAudioOutput = std::make_unique<FileAudioOutput>();
    if (!fileAudioOutput->open(decodedSamples.data(), decodedSamples.size(),
                               output_device)) {
        fileAudioOutput.reset();
        return false;
    }
    return true;
}

bool file_audio_has_output_clock() {
    return fileAudioActive.load(std::memory_order_acquire) &&
           fileAudioOutput != nullptr && fileAudioOutput->is_configured();
}

double file_audio_playback_time() {
    if (!file_audio_has_output_clock())
        return 0.0;
    return static_cast<double>(fileAudioOutput->position()) /
           static_cast<double>(FILE_AUDIO_SAMPLE_RATE);
}

/// @brief Advance one video-frame worth of samples and update audio analysis.
void file_audio_process_frame(double video_fps, acmx2::audio::AudioAnalyzer &analyzer) {
    if (!fileAudioActive || decodedSamples.empty())
        return;

    const bool output_playback =
        fileAudioOutput != nullptr && fileAudioOutput->is_configured();
    if (output_playback && fileAudioOutput->is_started())
        playbackPos = fileAudioOutput->position();

    if (playbackPos >= decodedSamples.size()) {
        fileAudioActive = false;
        return;
    }

    const double samples_per_frame =
        video_fps > 0.0
            ? static_cast<double>(FILE_AUDIO_SAMPLE_RATE) / video_fps
            : 512.0;
    size_t next_playback_pos = playbackPos;
    if (output_playback) {
        next_playback_pos += std::max<size_t>(
            1, static_cast<size_t>(std::floor(samples_per_frame)));
    } else {
        framePlaybackPos += samples_per_frame;
        next_playback_pos = std::max(
            playbackPos + 1,
            static_cast<size_t>(std::floor(framePlaybackPos)));
    }
    next_playback_pos = std::min(next_playback_pos, decodedSamples.size());

    unsigned int available =
        static_cast<unsigned int>(next_playback_pos - playbackPos);
    const float *samples = decodedSamples.data() + playbackPos;

    analyzer.process_samples(samples, available, 1);

    if (output_playback) {
        if (!fileAudioOutput->is_started() && !fileAudioOutput->start()) {
            fileAudioOutput.reset();
            playbackPos = next_playback_pos;
            framePlaybackPos = static_cast<double>(playbackPos);
        }
    } else {
        playbackPos = next_playback_pos;
    }
}

/// @brief Return true while decoded file-audio samples remain.
bool file_audio_is_active() {
    return fileAudioActive.load(std::memory_order_relaxed);
}

/// @brief Stop file-audio playback and release decoder/sample resources.
void file_audio_close() {
    fileAudioActive = false;
    fileAudioOutput.reset();
    if (swrCtx)
        swr_free(&swrCtx);
    if (codecCtx)
        avcodec_free_context(&codecCtx);
    if (fmtCtx)
        avformat_close_input(&fmtCtx);
    decodedSamples.clear();
    decodedSamples.shrink_to_fit();
    playbackPos = 0;
    framePlaybackPos = 0.0;
}
