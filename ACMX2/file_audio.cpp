#include "file_audio.hpp"
#include "audio.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

// Shared globals defined in audio.cpp
extern float gAmplitude;
extern float gFrequency;
extern float gPeak;
extern float gRMS;
extern float gSmooth;
extern float gLow;
extern float gMid;
extern float gHigh;
extern float amp_sense;
extern unsigned int input_channels;
extern unsigned int gSampleRate;

static AVFormatContext *fmtCtx = nullptr;
static AVCodecContext *codecCtx = nullptr;
static SwrContext *swrCtx = nullptr;
static int audioStreamIndex = -1;
static std::vector<float> decodedSamples;  // all decoded mono float samples at 44100 Hz
static size_t playbackPos = 0;
static std::atomic<bool> fileAudioActive{false};

// Single-pole filter states for 3-band energy (matching audio.cpp)
static float lpState = 0.0f;
static float mpState = 0.0f;

static bool decodeAllSamples() {
    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!pkt || !frame) {
        if (pkt) av_packet_free(&pkt);
        if (frame) av_frame_free(&frame);
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
    lpState = 0.0f;
    mpState = 0.0f;
    fileAudioActive = true;

    std::cout << "acmx2: file_audio: Loaded " << decodedSamples.size()
              << " samples (" << (decodedSamples.size() / 44100.0) << "s) from: " << filepath << "\n";

    // Clean up decoder resources — samples are fully buffered
    swr_free(&swrCtx);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&fmtCtx);

    return true;
}

void file_audio_process_frame(double video_fps) {
    if (!fileAudioActive || decodedSamples.empty())
        return;

    unsigned int samplesPerFrame = static_cast<unsigned int>(44100.0 / video_fps);
    if (samplesPerFrame == 0)
        samplesPerFrame = 512;

    if (playbackPos >= decodedSamples.size()) {
        fileAudioActive = false;
        return;
    }

    unsigned int available = static_cast<unsigned int>(
        std::min(static_cast<size_t>(samplesPerFrame), decodedSamples.size() - playbackPos));
    const float *samples = decodedSamples.data() + playbackPos;

    // Compute amplitude (average absolute value)
    float sum = 0.0f;
    for (unsigned int i = 0; i < available; ++i)
        sum += std::abs(samples[i]);
    gAmplitude = sum / available;

    // Peak and RMS
    float peak = 0.0f;
    float sumSq = 0.0f;
    for (unsigned int i = 0; i < available; ++i) {
        float s = std::abs(samples[i]);
        if (s > peak) peak = s;
        sumSq += s * s;
    }
    gPeak = peak;
    gRMS = std::sqrt(sumSq / available);

    // Exponentially smoothed amplitude
    constexpr float SMOOTH_ALPHA = 0.15f;
    gSmooth += SMOOTH_ALPHA * (gAmplitude - gSmooth);

    // 3-band energy (Low/Mid/High) — same filters as audio.cpp
    float lpCoeff = 1.0f - std::exp(-2.0f * 3.14159f * 300.0f / 44100.0f);
    float mpCoeff = 1.0f - std::exp(-2.0f * 3.14159f * 3000.0f / 44100.0f);
    float lowSum = 0.0f, midSum = 0.0f, highSum = 0.0f;
    for (unsigned int i = 0; i < available; ++i) {
        float s = samples[i];
        lpState += lpCoeff * (s - lpState);
        mpState += mpCoeff * (s - mpState);
        float lo = lpState;
        float mid = mpState - lpState;
        float hi = s - mpState;
        lowSum += lo * lo;
        midSum += mid * mid;
        highSum += hi * hi;
    }
    gLow = std::sqrt(lowSum / available);
    gMid = std::sqrt(midSum / available);
    gHigh = std::sqrt(highSum / available);

    // Dominant frequency via zero-crossing rate
    unsigned int crossings = 0;
    for (unsigned int i = 1; i < available; ++i) {
        float prev = samples[i - 1];
        float curr = samples[i];
        if ((prev >= 0.0f && curr < 0.0f) || (prev < 0.0f && curr >= 0.0f))
            ++crossings;
    }
    gFrequency = (static_cast<float>(crossings) * 44100.0f) / (2.0f * available);

    // Push to FFT buffer (mono, 1 channel)
    push_audio_buffer(samples, available, 1);

    playbackPos += available;
}

bool file_audio_is_active() {
    return fileAudioActive.load(std::memory_order_relaxed);
}

void file_audio_close() {
    fileAudioActive = false;
    if (swrCtx) swr_free(&swrCtx);
    if (codecCtx) avcodec_free_context(&codecCtx);
    if (fmtCtx) avformat_close_input(&fmtCtx);
    decodedSamples.clear();
    decodedSamples.shrink_to_fit();
    playbackPos = 0;
    lpState = 0.0f;
    mpState = 0.0f;
}
