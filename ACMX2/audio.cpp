#include "audio.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

float gAmplitude = 0.0f;
float gFrequency = 0.0f;
float gPeak = 0.0f;
float gRMS = 0.0f;
float gSmooth = 0.0f;
float gLow = 0.0f;
float gMid = 0.0f;
float gHigh = 0.0f;
float amp_sense = 25.0f;
unsigned int input_channels = 2;
unsigned int output_channels = 0;
bool output_buffer = false;
unsigned int gSampleRate = 44100;

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
                out[outIndex] = 0.0f;
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

    return 0;
}

float get_amp() { return gAmplitude; }
float get_sense() { return amp_sense; }
void set_sense(float s) { amp_sense = s; }
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
    (void)outputDeviceId;

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
    output_channels = 0;

    inputParams.deviceId = inputDevice;
    inputParams.nChannels = input_channels;
    inputParams.firstChannel = 0;

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
        audio.openStream(nullptr, &inputParams, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &audioCallback);
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
