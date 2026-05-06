#include "audio.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

float gAmplitude = 0.0f;
float gFrequency = 0.0f;
float gPeak = 0.0f;
float gRMS = 0.0f;
float gSmooth = 0.0f;
float gLow = 0.0f;
float gMid = 0.0f;
float gHigh = 0.0f;
float amp_sense = 1.0f;
unsigned int input_channels = 2;
unsigned int output_channels = 0;
bool output_buffer = false;
unsigned int gSampleRate = 44100;

static std::ofstream gRecordFile;
static std::atomic<bool> gRecording{false};
static uint32_t gRecordDataSize = 0;
static std::atomic<uint32_t> gRecordFadeIn{0};
static std::atomic<float> gRecordGain{1.0f};

// Lock-free SPSC ring buffer for audio recording
static constexpr size_t RING_CAPACITY = 1 << 20; // ~1M int16 samples
static int16_t gRingBuffer[RING_CAPACITY];
static std::atomic<size_t> gRingHead{0};  // written by callback
static std::atomic<size_t> gRingTail{0};  // read by disk thread
static std::thread gDiskThread;
static std::atomic<bool> gDiskRunning{false};

int audioCallback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                  double streamTime, RtAudioStreamStatus status, void *userData) {

    float *in = static_cast<float *>(inputBuffer);
    float *out = static_cast<float *>(outputBuffer);
    float sum = 0.0f;

    if (status || in == nullptr) {
        if (out && output_channels > 0)
            std::fill_n(out, nBufferFrames * output_channels, 0.0f);
        return 0;
    }

    for (unsigned int i = 0; i < nBufferFrames; ++i) {
        for (unsigned int ch = 0; ch < input_channels; ++ch) {
            unsigned int inIndex = i * input_channels + ch;
            sum += std::abs(in[inIndex]);
        }
        if (out && output_channels > 0) {
            for (unsigned int ch = 0; ch < output_channels; ++ch) {
                unsigned int outIndex = i * output_channels + ch;
                if (output_buffer) {
                    unsigned int inCh = ch < input_channels ? ch : 0;
                    out[outIndex] = in[i * input_channels + inCh];
                } else {
                    out[outIndex] = 0.0f;
                }
            }
        }
    }

    gAmplitude = sum / (nBufferFrames * input_channels);

    // Peak absolute sample value
    float peak = 0.0f;
    float sumSq = 0.0f;
    for (unsigned int i = 0; i < nBufferFrames; ++i) {
        float s = std::abs(in[i * input_channels]);
        if (s > peak) peak = s;
        sumSq += s * s;
    }
    gPeak = peak;
    gRMS = std::sqrt(sumSq / nBufferFrames);

    // Exponentially smoothed amplitude (low-pass on amplitude)
    constexpr float SMOOTH_ALPHA = 0.15f;
    gSmooth += SMOOTH_ALPHA * (gAmplitude - gSmooth);

    // 3-band energy via simple single-pole filters on channel 0
    // Low-pass  cutoff ~300 Hz, Mid-pass ~300-3000 Hz, High-pass ~3000 Hz
    static float lpState = 0.0f;
    static float mpState = 0.0f;
    float lpCoeff = 1.0f - std::exp(-2.0f * 3.14159f * 300.0f / gSampleRate);
    float mpCoeff = 1.0f - std::exp(-2.0f * 3.14159f * 3000.0f / gSampleRate);
    float lowSum = 0.0f, midSum = 0.0f, highSum = 0.0f;
    for (unsigned int i = 0; i < nBufferFrames; ++i) {
        float s = in[i * input_channels];
        lpState += lpCoeff * (s - lpState);
        mpState += mpCoeff * (s - mpState);
        float lo = lpState;
        float mid = mpState - lpState;
        float hi = s - mpState;
        lowSum += lo * lo;
        midSum += mid * mid;
        highSum += hi * hi;
    }
    gLow = std::sqrt(lowSum / nBufferFrames);
    gMid = std::sqrt(midSum / nBufferFrames);
    gHigh = std::sqrt(highSum / nBufferFrames);

    // Estimate dominant frequency via zero-crossing rate
    unsigned int crossings = 0;
    for (unsigned int i = 1; i < nBufferFrames; ++i) {
        float prev = in[(i - 1) * input_channels];
        float curr = in[i * input_channels];
        if ((prev >= 0.0f && curr < 0.0f) || (prev < 0.0f && curr >= 0.0f)) {
            ++crossings;
        }
    }
    gFrequency = (static_cast<float>(crossings) * gSampleRate) / (2.0f * nBufferFrames);

    if (gRecording.load(std::memory_order_relaxed)) {
        unsigned int totalSamples = nBufferFrames * input_channels;
        float gain = gRecordGain.load(std::memory_order_relaxed);
        uint32_t fadePos = gRecordFadeIn.load(std::memory_order_relaxed);
        constexpr uint32_t FADE_SAMPLES = 2048;
        size_t head = gRingHead.load(std::memory_order_relaxed);
        size_t tail = gRingTail.load(std::memory_order_acquire);
        for (unsigned int i = 0; i < totalSamples; ++i) {
            // Drop samples if ring buffer is full
            if (((head + 1) & (RING_CAPACITY - 1)) == (tail & (RING_CAPACITY - 1)))
                break;
            float sample = std::clamp(in[i] * gain, -1.0f, 1.0f);
            if (fadePos < FADE_SAMPLES) {
                sample *= static_cast<float>(fadePos) / static_cast<float>(FADE_SAMPLES);
                ++fadePos;
            }
            gRingBuffer[head & (RING_CAPACITY - 1)] = static_cast<int16_t>(sample * 32767.0f);
            ++head;
        }
        gRingHead.store(head, std::memory_order_release);
        gRecordFadeIn.store(fadePos, std::memory_order_relaxed);
    }

    // Snapshot the raw PCM input for FFT analysis on the render thread.
    push_audio_buffer(in, nBufferFrames, input_channels);

    return 0;
}

float get_amp() { return gAmplitude; }
float get_sense() { return amp_sense; }
void set_sense(float s) { amp_sense = std::clamp(s, 0.1f, 5.0f); }
float get_freq() { return gFrequency; }
float get_amp_peak() { return gPeak; }
float get_amp_rms() { return gRMS; }
float get_amp_smooth() { return gSmooth; }
float get_amp_low() { return gLow; }
float get_amp_mid() { return gMid; }
float get_amp_high() { return gHigh; }

RtAudio audio(RtAudio::LINUX_PULSE);

void set_output(bool o) {
    output_buffer = o;
}

static void writeWavHeader(std::ofstream &file, uint32_t dataSize, uint32_t sampleRate, uint16_t channels) {
    uint16_t bitsPerSample = 16;
    uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
    uint16_t blockAlign = channels * bitsPerSample / 8;
    uint32_t chunkSize = 36 + dataSize;

    file.seekp(0);
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char *>(&chunkSize), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    uint32_t subchunk1Size = 16;
    file.write(reinterpret_cast<const char *>(&subchunk1Size), 4);
    uint16_t audioFormat = 1; // PCM
    file.write(reinterpret_cast<const char *>(&audioFormat), 2);
    file.write(reinterpret_cast<const char *>(&channels), 2);
    file.write(reinterpret_cast<const char *>(&sampleRate), 4);
    file.write(reinterpret_cast<const char *>(&byteRate), 4);
    file.write(reinterpret_cast<const char *>(&blockAlign), 2);
    file.write(reinterpret_cast<const char *>(&bitsPerSample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char *>(&dataSize), 4);
}

static void diskWriterFunc() {
    constexpr size_t BATCH = 4096;
    int16_t buf[BATCH];
    while (gDiskRunning.load(std::memory_order_relaxed) || gRingTail.load(std::memory_order_relaxed) != gRingHead.load(std::memory_order_acquire)) {
        size_t tail = gRingTail.load(std::memory_order_relaxed);
        size_t head = gRingHead.load(std::memory_order_acquire);
        if (tail == head) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        size_t avail = head - tail;
        size_t n = std::min(avail, BATCH);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = gRingBuffer[(tail + i) & (RING_CAPACITY - 1)];
        }
        gRingTail.store(tail + n, std::memory_order_release);
        if (gRecordFile.is_open()) {
            gRecordFile.write(reinterpret_cast<const char *>(buf), n * sizeof(int16_t));
            gRecordDataSize += static_cast<uint32_t>(n * sizeof(int16_t));
        }
    }
}

bool start_audio_recording(const std::string &filepath) {
    if (gRecording.load())
        return false;
    gRecordFile.open(filepath, std::ios::binary | std::ios::trunc);
    if (!gRecordFile.is_open()) {
        std::cerr << "acmx2: Failed to open audio recording file: " << filepath << "\n";
        return false;
    }
    gRecordDataSize = 0;
    gRecordFadeIn.store(0, std::memory_order_relaxed);
    gRingHead.store(0, std::memory_order_relaxed);
    gRingTail.store(0, std::memory_order_relaxed);
    writeWavHeader(gRecordFile, 0, gSampleRate, static_cast<uint16_t>(input_channels));
    gDiskRunning.store(true, std::memory_order_relaxed);
    gRecording.store(true, std::memory_order_release);
    gDiskThread = std::thread(diskWriterFunc);
    std::cout << "acmx2: Audio recording started: " << filepath << "\n";
    return true;
}

void stop_audio_recording() {
    gRecording.store(false, std::memory_order_release);
    gDiskRunning.store(false, std::memory_order_relaxed);
    if (gDiskThread.joinable()) {
        gDiskThread.join();
    }
    if (gRecordFile.is_open()) {
        writeWavHeader(gRecordFile, gRecordDataSize, gSampleRate, static_cast<uint16_t>(input_channels));
        gRecordFile.close();
        std::cout << "acmx2: Audio recording stopped (" << gRecordDataSize << " bytes written)\n";
    }
}

bool is_audio_recording() {
    return gRecording.load(std::memory_order_relaxed);
}

double get_audio_recorded_duration_seconds() {
    const uint32_t bytes = gRecordDataSize;
    const uint32_t bps = gSampleRate * input_channels * static_cast<uint32_t>(sizeof(int16_t));
    if (bps == 0) return 0.0;
    return static_cast<double>(bytes) / static_cast<double>(bps);
}

void set_record_gain(float gain) {
    gRecordGain.store(std::clamp(gain, 0.0f, 2.0f), std::memory_order_relaxed);
}

float get_record_gain() {
    return gRecordGain.load(std::memory_order_relaxed);
}

void list_audio_devices() {
    std::vector<unsigned int> ids = audio.getDeviceIds();
    std::cout << "acmx2: Found " << ids.size() << " audio device(s):\n";

    for (auto id : ids) {
        RtAudio::DeviceInfo info = audio.getDeviceInfo(id);
        std::cout << "  Device " << id << ": " << info.name;
        if (info.isDefaultInput)
            std::cout << " [DEFAULT INPUT]";
        if (info.isDefaultOutput)
            std::cout << " [DEFAULT OUTPUT]";
        std::cout << "\n";
        std::cout << "    Input channels: " << info.inputChannels << "\n";
        std::cout << "    Output channels: " << info.outputChannels << "\n";
        std::cout << "    Sample rates: ";
        for (auto rate : info.sampleRates) {
            std::cout << rate << " ";
        }
        std::cout << "\n";
    }
}

int init_audio(unsigned int channels, float sense, int inputDeviceId, int outputDeviceId) {
    input_channels = channels;
    amp_sense = sense;

    std::vector<unsigned int> ids = audio.getDeviceIds();
    if (ids.empty()) {
        std::cerr << "acmx2: No audio devices found!" << std::endl;
        return 1;
    } else {
        std::cout << "acmx2: Audio device found...\n";
    }

    unsigned int sampleRate = 44100;
    unsigned int bufferFrames = 512;

    RtAudio::StreamParameters inputParams;

    unsigned int inputDevice;
    if (inputDeviceId >= 0) {
        inputDevice = static_cast<unsigned int>(inputDeviceId);
        std::cout << "acmx2: Using specified input device: " << inputDevice << "\n";
    } else {
        // Use getDefaultInputDevice() which returns a proper device ID in
        // RtAudio 6.x.  Fall back to scanning if it returns 0 (no default).
        inputDevice = audio.getDefaultInputDevice();
        if (inputDevice == 0) {
            for (auto id : ids) {
                RtAudio::DeviceInfo di = audio.getDeviceInfo(id);
                if (di.isDefaultInput && di.inputChannels > 0) {
                    inputDevice = id;
                    break;
                }
            }
        }
        if (inputDevice == 0) {
            // Last resort: pick the first device with input channels
            for (auto id : ids) {
                RtAudio::DeviceInfo di = audio.getDeviceInfo(id);
                if (di.inputChannels > 0) {
                    inputDevice = id;
                    break;
                }
            }
        }
        std::cout << "acmx2: Using default input device: " << inputDevice << "\n";
    }

    RtAudio::DeviceInfo inInfo = audio.getDeviceInfo(inputDevice);
    if (inInfo.inputChannels == 0) {
        std::cout << "acmx2: Input device has no input channels...\n";
        return 1;
    }

    std::cout << "acmx2: Selected input device " << inputDevice << ": " << inInfo.name << "\n";
    std::cout << "acmx2:   Input channels: " << inInfo.inputChannels << "\n";
    if (inInfo.isDefaultInput)
        std::cout << "acmx2:   [DEFAULT INPUT]\n";

    input_channels = std::min(channels, inInfo.inputChannels);

    inputParams.deviceId = inputDevice;
    inputParams.nChannels = input_channels;
    inputParams.firstChannel = 0;

    // Set up output stream for pass-through if enabled
    RtAudio::StreamParameters outputParams;
    RtAudio::StreamParameters *outParamsPtr = nullptr;

    if (output_buffer) {
        unsigned int outputDevice;
        if (outputDeviceId >= 0) {
            outputDevice = static_cast<unsigned int>(outputDeviceId);
            std::cout << "acmx2: Using specified output device: " << outputDevice << "\n";
        } else {
            outputDevice = audio.getDefaultOutputDevice();
            std::cout << "acmx2: Using default output device: " << outputDevice << "\n";
        }

        RtAudio::DeviceInfo outInfo = audio.getDeviceInfo(outputDevice);
        if (outInfo.outputChannels > 0) {
            output_channels = std::min(static_cast<unsigned int>(2), outInfo.outputChannels);
            outputParams.deviceId = outputDevice;
            outputParams.nChannels = output_channels;
            outputParams.firstChannel = 0;
            outParamsPtr = &outputParams;
            std::cout << "acmx2: Audio pass-through enabled on device " << outputDevice
                      << ": " << outInfo.name << " (" << output_channels << " ch)\n";
        } else {
            std::cerr << "acmx2: Output device has no output channels, pass-through disabled.\n";
            output_buffer = false;
            output_channels = 0;
        }
    } else {
        output_channels = 0;
    }

    std::vector<unsigned int> sampleRates = inInfo.sampleRates;
    if (!sampleRates.empty()) {
        if (std::find(sampleRates.begin(), sampleRates.end(), sampleRate) == sampleRates.end()) {
            sampleRate = 48000;
            if (std::find(sampleRates.begin(), sampleRates.end(), sampleRate) == sampleRates.end()) {
                sampleRate = sampleRates[0];
            }
        }
    }

    try {
        gSampleRate = sampleRate;
        audio.openStream(outParamsPtr, &inputParams, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &audioCallback);
        audio.startStream();
        if (audio.isStreamOpen())
            std::cout << "acmx2: Audio input stream opened (rate=" << sampleRate
                      << " Hz, channels=" << input_channels
                      << ", sensitivity=" << amp_sense << ")\n";
    } catch (std::exception &e) {
        std::cerr << "acmx2: Standard exception: " << e.what() << std::endl;
        if (audio.isStreamOpen())
            audio.closeStream();
        return 1;
    } catch (...) {
        std::cerr << "acmx2: Unknown error occurred!" << std::endl;
        if (audio.isStreamOpen())
            audio.closeStream();
        return 1;
    }

    return 0;
}

void close_audio() {
    if (audio.isStreamOpen()) {
        audio.closeStream();
        std::cout << "acmx2: Audio stream closed.\n";
    }
}

// ---------------------------------------------------------------------------
// FFT spectrum analysis
// ---------------------------------------------------------------------------

/**
 * @brief Double-buffered PCM snapshot for lock-free audio→render transfer.
 *
 * Two buffers (`buf[0]` and `buf[1]`) alternate roles as *front* (render
 * reads) and *back* (audio callback writes).  The atomic `which` index
 * tells the render thread which buffer holds the most recent complete
 * snapshot.  Because only one thread writes each buffer at a time, no
 * mutex is required.
 */
static float gFftBuffer[2][FFT_SIZE] = {};

/// Index (0 or 1) of the buffer currently readable by the render thread.
static std::atomic<int> gFftWhich{0};

/// Internal storage for the last computed magnitude spectrum.
static std::vector<float> gFftMagnitudes(FFT_SIZE / 2, 0.0f);

void push_audio_buffer(const float *samples, unsigned int count, unsigned int channels) {
    int back = 1 - gFftWhich.load(std::memory_order_acquire);
    unsigned int n = std::min(static_cast<unsigned int>(FFT_SIZE), count);
    for (unsigned int i = 0; i < n; ++i)
        gFftBuffer[back][i] = samples[i * channels];
    for (unsigned int i = n; i < FFT_SIZE; ++i)
        gFftBuffer[back][i] = 0.0f;
    gFftWhich.store(back, std::memory_order_release);
}

/**
 * @brief In-place iterative radix-2 Cooley–Tukey FFT.
 *
 * This is the classic decimation-in-time butterfly algorithm.  It
 * operates on interleaved real/imaginary pairs stored in a flat array
 * of length `2 * n` (where `n` is a power of two).
 *
 * ### Algorithm outline
 * 1. **Bit-reversal permutation** — reorder the input so that each
 *    butterfly stage reads its operands from adjacent memory.
 * 2. **Butterfly passes** — for each stage \f$s = 1 \ldots \log_2 n\f$,
 *    combine pairs of sub-transforms using the twiddle factor
 *    \f$W_N^k = e^{-2\pi i\,k / N}\f$.
 *
 * @param data  Flat array of `[re0, im0, re1, im1, …]` — length `2*n`.
 * @param n     Number of complex samples.  **Must** be a power of two.
 */
static void fft_radix2(float *data, int n) {
    // --- Bit-reversal permutation ---
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[2 * i], data[2 * j]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
    }
    // --- Butterfly stages ---
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * 3.14159265358979f / static_cast<float>(len);
        float wRe = std::cos(angle);
        float wIm = std::sin(angle);
        for (int i = 0; i < n; i += len) {
            float curRe = 1.0f, curIm = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int even = i + j;
                int odd = i + j + len / 2;
                float tRe = curRe * data[2 * odd] - curIm * data[2 * odd + 1];
                float tIm = curRe * data[2 * odd + 1] + curIm * data[2 * odd];
                data[2 * odd] = data[2 * even] - tRe;
                data[2 * odd + 1] = data[2 * even + 1] - tIm;
                data[2 * even] += tRe;
                data[2 * even + 1] += tIm;
                float newRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = newRe;
            }
        }
    }
}

void compute_audio_fft() {
    int front = gFftWhich.load(std::memory_order_acquire);

    // Pack into interleaved complex array: [re0, im0, re1, im1, …]
    float complex[FFT_SIZE * 2];
    for (int i = 0; i < FFT_SIZE; ++i) {
        // Apply Hann window to reduce spectral leakage:
        //   w(n) = 0.5 * (1 - cos(2π n / (N-1)))
        float hann = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979f * i / (FFT_SIZE - 1)));
        complex[2 * i] = gFftBuffer[front][i] * hann;  // real part
        complex[2 * i + 1] = 0.0f;                      // imaginary part
    }

    fft_radix2(complex, FFT_SIZE);

    // Compute magnitude of each positive-frequency bin
    float inv = 2.0f / static_cast<float>(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE / 2; ++i) {
        float re = complex[2 * i];
        float im = complex[2 * i + 1];
        gFftMagnitudes[i] = std::sqrt(re * re + im * im) * inv;
    }
}

const std::vector<float> &get_fft_magnitudes() {
    return gFftMagnitudes;
}

int get_fft_bin_count() {
    return FFT_SIZE / 2;
}
