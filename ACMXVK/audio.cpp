#include "audio.hpp"

#include <rtaudio/RtAudio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <vector>

namespace acmxvk::audio {
    namespace {

        constexpr float PI = 3.14159265358979F;

        void fft_radix2(float *data, int size) {
            for (int index = 1, reversed = 0; index < size; ++index) {
                int bit = size >> 1;
                while ((reversed & bit) != 0) {
                    reversed ^= bit;
                    bit >>= 1;
                }
                reversed ^= bit;
                if (index < reversed) {
                    std::swap(data[2 * index], data[2 * reversed]);
                    std::swap(data[2 * index + 1], data[2 * reversed + 1]);
                }
            }

            for (int length = 2; length <= size; length <<= 1) {
                const float angle = -2.0F * PI / static_cast<float>(length);
                const float rotation_real = std::cos(angle);
                const float rotation_imaginary = std::sin(angle);
                for (int start = 0; start < size; start += length) {
                    float current_real = 1.0F;
                    float current_imaginary = 0.0F;
                    for (int offset = 0; offset < length / 2; ++offset) {
                        const int even = start + offset;
                        const int odd = even + length / 2;
                        const float odd_real =
                            current_real * data[2 * odd] -
                            current_imaginary * data[2 * odd + 1];
                        const float odd_imaginary =
                            current_real * data[2 * odd + 1] +
                            current_imaginary * data[2 * odd];

                        data[2 * odd] = data[2 * even] - odd_real;
                        data[2 * odd + 1] = data[2 * even + 1] - odd_imaginary;
                        data[2 * even] += odd_real;
                        data[2 * even + 1] += odd_imaginary;

                        const float next_real =
                            current_real * rotation_real -
                            current_imaginary * rotation_imaginary;
                        current_imaginary =
                            current_real * rotation_imaginary +
                            current_imaginary * rotation_real;
                        current_real = next_real;
                    }
                }
            }
        }

        [[nodiscard]] RtAudio makeRtAudio() {
#ifdef __linux__
            return RtAudio(RtAudio::LINUX_PULSE);
#else
            return RtAudio();
#endif
        }

    } // namespace

    class AudioEngine::Impl {
      public:
        Impl() : stream(makeRtAudio()) {}
        ~Impl() { close(); }

        bool open(const AudioStreamConfig &config) {
            close();

            try {
                const std::vector<unsigned int> device_ids = stream.getDeviceIds();
                if (device_ids.empty()) {
                    std::cerr << "acmxvk: no audio devices found\n";
                    return false;
                }

                unsigned int input_device = 0;
                if (config.input_device >= 0) {
                    input_device = static_cast<unsigned int>(config.input_device);
                } else {
                    input_device = stream.getDefaultInputDevice();
                }

                const auto selected = std::find(device_ids.begin(), device_ids.end(),
                                                input_device);
                if (selected == device_ids.end()) {
                    std::cerr << "acmxvk: audio input device " << input_device
                              << " was not found\n";
                    return false;
                }

                const RtAudio::DeviceInfo input_info =
                    stream.getDeviceInfo(input_device);
                if (input_info.inputChannels == 0) {
                    std::cerr << "acmxvk: audio device " << input_device
                              << " has no input channels\n";
                    return false;
                }

                input_channels = std::min(std::max(config.channels, 1U),
                                          input_info.inputChannels);
                sensitivity_value.store(std::clamp(config.sensitivity, 0.1F, 5.0F),
                                        std::memory_order_relaxed);
                resetMetrics();

                unsigned int selected_rate = 44100;
                if (!input_info.sampleRates.empty() &&
                    std::find(input_info.sampleRates.begin(), input_info.sampleRates.end(),
                              selected_rate) == input_info.sampleRates.end()) {
                    selected_rate = 48000;
                    if (std::find(input_info.sampleRates.begin(),
                                  input_info.sampleRates.end(), selected_rate) ==
                        input_info.sampleRates.end()) {
                        selected_rate = input_info.sampleRates.front();
                    }
                }
                sample_rate.store(selected_rate, std::memory_order_relaxed);

                RtAudio::StreamParameters input_parameters;
                input_parameters.deviceId = input_device;
                input_parameters.nChannels = input_channels;
                input_parameters.firstChannel = 0;

                unsigned int buffer_frames = 512;
                stream.openStream(nullptr, &input_parameters, RTAUDIO_FLOAT32,
                                  selected_rate, &buffer_frames, &Impl::audioCallback,
                                  this);
                stream.startStream();

                std::cout << "acmxvk: audio input " << input_device << ": "
                          << input_info.name << " (" << selected_rate << " Hz, "
                          << input_channels << " channel"
                          << (input_channels == 1 ? "" : "s") << ")\n";
                return stream.isStreamOpen();
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: audio error: " << error.what() << '\n';
                close();
                return false;
            }
        }

        void close() {
            static_cast<void>(stopRecording());
            if (!stream.isStreamOpen()) {
                return;
            }
            try {
                if (stream.isStreamRunning()) {
                    stream.stopStream();
                }
                stream.closeStream();
                std::cout << "acmxvk: audio input closed\n";
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: error closing audio input: " << error.what()
                          << '\n';
            }
        }

        [[nodiscard]] AudioMetrics metrics() const {
            return {
                amplitude.load(std::memory_order_relaxed),
                frequency.load(std::memory_order_relaxed),
                peak.load(std::memory_order_relaxed),
                rms.load(std::memory_order_relaxed),
                smooth.load(std::memory_order_relaxed),
                low.load(std::memory_order_relaxed),
                mid.load(std::memory_order_relaxed),
                high.load(std::memory_order_relaxed),
            };
        }

        [[nodiscard]] std::vector<float> spectrum() const {
            const int front = spectrum_front.load(std::memory_order_acquire);
            std::array<float, AudioEngine::FFT_SIZE * 2> complex{};
            for (std::size_t index = 0; index < AudioEngine::FFT_SIZE; ++index) {
                const float hann =
                    0.5F *
                    (1.0F -
                     std::cos(2.0F * PI * static_cast<float>(index) /
                              static_cast<float>(AudioEngine::FFT_SIZE - 1)));
                complex[2 * index] =
                    spectrum_samples[front][index].load(std::memory_order_relaxed) *
                    hann;
            }

            fft_radix2(complex.data(), static_cast<int>(AudioEngine::FFT_SIZE));

            std::vector<float> magnitudes(AudioEngine::spectrum_bin_count());
            constexpr float INVERSE = 2.0F / static_cast<float>(AudioEngine::FFT_SIZE);
            for (std::size_t index = 0; index < magnitudes.size(); ++index) {
                const float real = complex[2 * index];
                const float imaginary = complex[2 * index + 1];
                magnitudes[index] =
                    std::sqrt(real * real + imaginary * imaginary) * INVERSE;
            }
            return magnitudes;
        }

        void resetMetrics() {
            amplitude.store(0.0F, std::memory_order_relaxed);
            frequency.store(0.0F, std::memory_order_relaxed);
            peak.store(0.0F, std::memory_order_relaxed);
            rms.store(0.0F, std::memory_order_relaxed);
            smooth.store(0.0F, std::memory_order_relaxed);
            low.store(0.0F, std::memory_order_relaxed);
            mid.store(0.0F, std::memory_order_relaxed);
            high.store(0.0F, std::memory_order_relaxed);
            smooth_value = 0.0F;
            low_pass_state = 0.0F;
            mid_pass_state = 0.0F;
            for (auto &buffer : spectrum_samples) {
                for (std::atomic<float> &sample : buffer) {
                    sample.store(0.0F, std::memory_order_relaxed);
                }
            }
            spectrum_front.store(0, std::memory_order_release);
        }

        static int audioCallback(void *, void *input_buffer,
                                 unsigned int frame_count, double,
                                 RtAudioStreamStatus status, void *user_data) {
            return static_cast<Impl *>(user_data)
                ->processSamples(static_cast<const float *>(input_buffer), frame_count,
                                 status);
        }

        int processSamples(const float *samples, unsigned int frame_count,
                           RtAudioStreamStatus status) {
            static_cast<void>(status);
            if (samples == nullptr || frame_count == 0 || input_channels == 0) {
                return 0;
            }

            float amplitude_sum = 0.0F;
            float peak_value = 0.0F;
            float square_sum = 0.0F;
            float low_sum = 0.0F;
            float mid_sum = 0.0F;
            float high_sum = 0.0F;
            unsigned int crossings = 0;
            float previous = samples[0];
            const int spectrum_back =
                1 - spectrum_front.load(std::memory_order_acquire);
            const float rate = static_cast<float>(
                std::max(sample_rate.load(std::memory_order_relaxed), 1U));
            const float low_coefficient = 1.0F - std::exp(-2.0F * PI * 300.0F / rate);
            const float mid_coefficient =
                1.0F - std::exp(-2.0F * PI * 3000.0F / rate);

            std::unique_lock<std::mutex> recording_lock;
            bool capture_recording = recording.load(std::memory_order_acquire);
            if (capture_recording) {
                recording_lock = std::unique_lock<std::mutex>(recording_mutex);
                capture_recording =
                    recording.load(std::memory_order_acquire);
            }

            for (unsigned int frame = 0; frame < frame_count; ++frame) {
                float mono = 0.0F;
                for (unsigned int channel = 0; channel < input_channels; ++channel) {
                    const float sample = samples[frame * input_channels + channel];
                    amplitude_sum += std::abs(sample);
                    mono += sample;
                }
                mono /= static_cast<float>(input_channels);
                if (capture_recording) {
                    recorded_samples.push_back(std::clamp(mono, -1.0F, 1.0F));
                }
                low_pass_state += low_coefficient * (mono - low_pass_state);
                mid_pass_state += mid_coefficient * (mono - mid_pass_state);
                const float low_sample = low_pass_state;
                const float mid_sample = mid_pass_state - low_pass_state;
                const float high_sample = mono - mid_pass_state;
                low_sum += low_sample * low_sample;
                mid_sum += mid_sample * mid_sample;
                high_sum += high_sample * high_sample;
                if (frame < AudioEngine::FFT_SIZE) {
                    spectrum_samples[spectrum_back][frame].store(
                        mono, std::memory_order_relaxed);
                }
                peak_value = std::max(peak_value, std::abs(mono));
                square_sum += mono * mono;
                if (frame > 0 && ((previous >= 0.0F && mono < 0.0F) ||
                                  (previous < 0.0F && mono >= 0.0F))) {
                    ++crossings;
                }
                previous = mono;
            }
            for (std::size_t frame =
                     std::min<std::size_t>(frame_count, AudioEngine::FFT_SIZE);
                 frame < AudioEngine::FFT_SIZE; ++frame) {
                spectrum_samples[spectrum_back][frame].store(
                    0.0F, std::memory_order_relaxed);
            }
            spectrum_front.store(spectrum_back, std::memory_order_release);

            const float amplitude_value =
                amplitude_sum /
                static_cast<float>(frame_count * input_channels);
            constexpr float SMOOTH_ALPHA = 0.15F;
            smooth_value += SMOOTH_ALPHA * (amplitude_value - smooth_value);

            amplitude.store(amplitude_value, std::memory_order_relaxed);
            peak.store(peak_value, std::memory_order_relaxed);
            rms.store(std::sqrt(square_sum / static_cast<float>(frame_count)),
                      std::memory_order_relaxed);
            smooth.store(smooth_value, std::memory_order_relaxed);
            low.store(std::sqrt(low_sum / static_cast<float>(frame_count)),
                      std::memory_order_relaxed);
            mid.store(std::sqrt(mid_sum / static_cast<float>(frame_count)),
                      std::memory_order_relaxed);
            high.store(std::sqrt(high_sum / static_cast<float>(frame_count)),
                       std::memory_order_relaxed);
            frequency.store(
                static_cast<float>(crossings) *
                    rate /
                    (2.0F * static_cast<float>(frame_count)),
                std::memory_order_relaxed);
            return 0;
        }

        bool startRecording() {
            if (!stream.isStreamOpen() ||
                recording.load(std::memory_order_acquire)) {
                return false;
            }

            std::lock_guard<std::mutex> lock(recording_mutex);
            recorded_samples.clear();
            recording_sample_rate =
                std::max(sample_rate.load(std::memory_order_relaxed), 1U);
            constexpr unsigned int RESERVE_SECONDS = 60;
            recorded_samples.reserve(static_cast<std::size_t>(recording_sample_rate) *
                                     RESERVE_SECONDS);
            recording.store(true, std::memory_order_release);
            std::cout << "acmxvk: live audio recording started ("
                      << recording_sample_rate << " Hz, mono)\n";
            return true;
        }

        [[nodiscard]] AudioRecording stopRecording() {
            recording.store(false, std::memory_order_release);
            std::lock_guard<std::mutex> lock(recording_mutex);

            AudioRecording result;
            result.samples = std::move(recorded_samples);
            result.sample_rate = recording_sample_rate;
            if (!result.empty()) {
                std::cout << "acmxvk: live audio recording stopped ("
                          << result.duration_seconds() << " seconds)\n";
            }
            recorded_samples.clear();
            return result;
        }

        RtAudio stream;
        std::atomic<float> amplitude{0.0F};
        std::atomic<float> frequency{0.0F};
        std::atomic<float> peak{0.0F};
        std::atomic<float> rms{0.0F};
        std::atomic<float> smooth{0.0F};
        std::atomic<float> low{0.0F};
        std::atomic<float> mid{0.0F};
        std::atomic<float> high{0.0F};
        std::atomic<float> sensitivity_value{1.0F};
        std::atomic<unsigned int> sample_rate{44100};
        std::atomic<bool> recording{false};
        std::array<std::array<std::atomic<float>, AudioEngine::FFT_SIZE>, 2>
            spectrum_samples{};
        std::atomic<int> spectrum_front{0};
        std::mutex recording_mutex;
        std::vector<float> recorded_samples;
        unsigned int recording_sample_rate = 44100;
        unsigned int input_channels = 0;
        float smooth_value = 0.0F;
        float low_pass_state = 0.0F;
        float mid_pass_state = 0.0F;
    };

    AudioEngine::AudioEngine() : impl(std::make_unique<Impl>()) {}
    AudioEngine::~AudioEngine() = default;

    bool AudioEngine::open(const AudioStreamConfig &config) {
        return impl->open(config);
    }

    void AudioEngine::close() {
        impl->close();
    }

    bool AudioEngine::is_open() const {
        return impl->stream.isStreamOpen();
    }

    AudioMetrics AudioEngine::metrics() const {
        return impl->metrics();
    }

    unsigned int AudioEngine::sample_rate() const {
        return impl->sample_rate.load(std::memory_order_relaxed);
    }

    void AudioEngine::set_sensitivity(float sensitivity) {
        impl->sensitivity_value.store(std::clamp(sensitivity, 0.1F, 5.0F),
                                      std::memory_order_relaxed);
    }

    float AudioEngine::sensitivity() const {
        return impl->sensitivity_value.load(std::memory_order_relaxed);
    }

    std::vector<float> AudioEngine::spectrum() const {
        return impl->spectrum();
    }

    void AudioEngine::process_samples(const float *samples,
                                      unsigned int frame_count,
                                      unsigned int channels,
                                      unsigned int sample_rate) {
        impl->input_channels = channels;
        impl->sample_rate.store(std::max(sample_rate, 1U),
                                std::memory_order_relaxed);
        impl->processSamples(samples, frame_count, 0);
    }

    bool AudioEngine::start_recording() {
        return impl->startRecording();
    }

    AudioRecording AudioEngine::stop_recording() {
        return impl->stopRecording();
    }

    bool AudioEngine::is_recording() const {
        return impl->recording.load(std::memory_order_acquire);
    }

    void AudioEngine::reset() {
        impl->resetMetrics();
    }

    void AudioEngine::list_devices() {
        try {
            RtAudio stream = makeRtAudio();
            const std::vector<unsigned int> device_ids = stream.getDeviceIds();
            std::cout << "acmxvk: found " << device_ids.size()
                      << " audio device(s)\n";
            for (const unsigned int id : device_ids) {
                const RtAudio::DeviceInfo info = stream.getDeviceInfo(id);
                std::cout << "  Device " << id << ": " << info.name;
                if (info.isDefaultInput) {
                    std::cout << " [DEFAULT INPUT]";
                }
                if (info.isDefaultOutput) {
                    std::cout << " [DEFAULT OUTPUT]";
                }
                std::cout << "\n    Input channels: " << info.inputChannels
                          << "\n    Output channels: " << info.outputChannels
                          << "\n";
            }
        } catch (const std::exception &error) {
            throw std::runtime_error(std::string("could not enumerate audio devices: ") +
                                     error.what());
        }
    }

} // namespace acmxvk::audio
