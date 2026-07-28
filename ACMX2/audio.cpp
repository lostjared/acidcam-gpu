#include "audio.hpp"

#include <RtAudio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace acmx2::audio {
    namespace {

        constexpr float PI = 3.14159265358979f;
        constexpr std::size_t RING_CAPACITY = 1U << 20;
        constexpr std::size_t RING_MASK = RING_CAPACITY - 1;
        constexpr std::size_t WRITE_BATCH_SIZE = 4096;
        constexpr std::uint32_t RECORD_FADE_SAMPLES = 2048;

        void fft_radix2(float *data, int size) {
            for (int i = 1, j = 0; i < size; ++i) {
                int bit = size >> 1;
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

            for (int length = 2; length <= size; length <<= 1) {
                const float angle = -2.0f * PI / static_cast<float>(length);
                const float rotation_real = std::cos(angle);
                const float rotation_imaginary = std::sin(angle);
                for (int i = 0; i < size; i += length) {
                    float current_real = 1.0f;
                    float current_imaginary = 0.0f;
                    for (int j = 0; j < length / 2; ++j) {
                        const int even = i + j;
                        const int odd = even + length / 2;
                        const float odd_real =
                            current_real * data[2 * odd] - current_imaginary * data[2 * odd + 1];
                        const float odd_imaginary =
                            current_real * data[2 * odd + 1] + current_imaginary * data[2 * odd];

                        data[2 * odd] = data[2 * even] - odd_real;
                        data[2 * odd + 1] = data[2 * even + 1] - odd_imaginary;
                        data[2 * even] += odd_real;
                        data[2 * even + 1] += odd_imaginary;

                        const float next_real =
                            current_real * rotation_real - current_imaginary * rotation_imaginary;
                        current_imaginary =
                            current_real * rotation_imaginary + current_imaginary * rotation_real;
                        current_real = next_real;
                    }
                }
            }
        }

        void write_wav_header(std::ofstream &file, std::uint32_t data_size,
                              std::uint32_t sample_rate, std::uint16_t channels) {
            constexpr std::uint16_t bits_per_sample = 16;
            const std::uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
            const std::uint16_t block_align = channels * bits_per_sample / 8;
            const std::uint32_t chunk_size = 36 + data_size;
            constexpr std::uint32_t format_chunk_size = 16;
            constexpr std::uint16_t pcm_format = 1;

            file.seekp(0);
            file.write("RIFF", 4);
            file.write(reinterpret_cast<const char *>(&chunk_size), 4);
            file.write("WAVE", 4);
            file.write("fmt ", 4);
            file.write(reinterpret_cast<const char *>(&format_chunk_size), 4);
            file.write(reinterpret_cast<const char *>(&pcm_format), 2);
            file.write(reinterpret_cast<const char *>(&channels), 2);
            file.write(reinterpret_cast<const char *>(&sample_rate), 4);
            file.write(reinterpret_cast<const char *>(&byte_rate), 4);
            file.write(reinterpret_cast<const char *>(&block_align), 2);
            file.write(reinterpret_cast<const char *>(&bits_per_sample), 2);
            file.write("data", 4);
            file.write(reinterpret_cast<const char *>(&data_size), 4);
        }

        RtAudio make_rt_audio() {
#ifdef __linux__
            return RtAudio(RtAudio::LINUX_PULSE);
#else
            return RtAudio();
#endif
        }

    } // namespace

    class AudioAnalyzer::Impl {
      public:
        void process_samples(const float *samples, unsigned int frame_count, unsigned int channels) {
            if (samples == nullptr || frame_count == 0 || channels == 0)
                return;

            float amplitude_sum = 0.0f;
            float peak_value = 0.0f;
            float square_sum = 0.0f;
            for (unsigned int frame = 0; frame < frame_count; ++frame) {
                for (unsigned int channel = 0; channel < channels; ++channel)
                    amplitude_sum += std::abs(samples[frame * channels + channel]);

                const float sample = std::abs(samples[frame * channels]);
                peak_value = std::max(peak_value, sample);
                square_sum += sample * sample;
            }

            const float amplitude_value =
                amplitude_sum / static_cast<float>(frame_count * channels);
            amplitude.store(amplitude_value, std::memory_order_relaxed);
            peak.store(peak_value, std::memory_order_relaxed);
            rms.store(std::sqrt(square_sum / static_cast<float>(frame_count)),
                      std::memory_order_relaxed);

            constexpr float smooth_alpha = 0.15f;
            smooth_value += smooth_alpha * (amplitude_value - smooth_value);
            smooth.store(smooth_value, std::memory_order_relaxed);

            const float rate = static_cast<float>(sample_rate.load(std::memory_order_relaxed));
            const float low_coefficient = 1.0f - std::exp(-2.0f * PI * 300.0f / rate);
            const float mid_coefficient = 1.0f - std::exp(-2.0f * PI * 3000.0f / rate);
            float low_sum = 0.0f;
            float mid_sum = 0.0f;
            float high_sum = 0.0f;
            for (unsigned int frame = 0; frame < frame_count; ++frame) {
                const float sample = samples[frame * channels];
                low_pass_state += low_coefficient * (sample - low_pass_state);
                mid_pass_state += mid_coefficient * (sample - mid_pass_state);
                const float low_sample = low_pass_state;
                const float mid_sample = mid_pass_state - low_pass_state;
                const float high_sample = sample - mid_pass_state;
                low_sum += low_sample * low_sample;
                mid_sum += mid_sample * mid_sample;
                high_sum += high_sample * high_sample;
            }
            low.store(std::sqrt(low_sum / static_cast<float>(frame_count)),
                      std::memory_order_relaxed);
            mid.store(std::sqrt(mid_sum / static_cast<float>(frame_count)),
                      std::memory_order_relaxed);
            high.store(std::sqrt(high_sum / static_cast<float>(frame_count)),
                       std::memory_order_relaxed);

            unsigned int crossings = 0;
            for (unsigned int frame = 1; frame < frame_count; ++frame) {
                const float previous = samples[(frame - 1) * channels];
                const float current = samples[frame * channels];
                if ((previous >= 0.0f && current < 0.0f) ||
                    (previous < 0.0f && current >= 0.0f))
                    ++crossings;
            }
            frequency.store(
                static_cast<float>(crossings) * rate / (2.0f * static_cast<float>(frame_count)),
                std::memory_order_relaxed);

            const int back = 1 - spectrum_front.load(std::memory_order_acquire);
            const unsigned int copied_frames =
                std::min(static_cast<unsigned int>(FFT_SIZE), frame_count);
            for (unsigned int i = 0; i < copied_frames; ++i)
                spectrum_samples[back][i].store(samples[i * channels], std::memory_order_relaxed);
            for (unsigned int i = copied_frames; i < FFT_SIZE; ++i)
                spectrum_samples[back][i].store(0.0f, std::memory_order_relaxed);
            spectrum_front.store(back, std::memory_order_release);
        }

        std::atomic<float> amplitude{0.0f};
        std::atomic<float> frequency{0.0f};
        std::atomic<float> peak{0.0f};
        std::atomic<float> rms{0.0f};
        std::atomic<float> smooth{0.0f};
        std::atomic<float> low{0.0f};
        std::atomic<float> mid{0.0f};
        std::atomic<float> high{0.0f};
        std::atomic<float> sensitivity{1.0f};
        std::atomic<unsigned int> sample_rate{44100};

        float smooth_value = 0.0f;
        float low_pass_state = 0.0f;
        float mid_pass_state = 0.0f;

        std::array<std::array<std::atomic<float>, FFT_SIZE>, 2> spectrum_samples{};
        std::atomic<int> spectrum_front{0};
        std::vector<float> spectrum_magnitudes =
            std::vector<float>(FFT_SIZE / 2, 0.0f);
    };

    AudioAnalyzer::AudioAnalyzer() : impl(std::make_unique<Impl>()) {}
    AudioAnalyzer::~AudioAnalyzer() = default;

    void AudioAnalyzer::process_samples(const float *samples, unsigned int frame_count,
                                        unsigned int channels) {
        impl->process_samples(samples, frame_count, channels);
    }

    void AudioAnalyzer::reset() {
        const float current_sensitivity = sensitivity();
        const unsigned int current_sample_rate = sample_rate();
        impl = std::make_unique<Impl>();
        set_sensitivity(current_sensitivity);
        set_sample_rate(current_sample_rate);
    }

    void AudioAnalyzer::set_sample_rate(unsigned int sample_rate) {
        impl->sample_rate.store(std::max(sample_rate, 1U), std::memory_order_relaxed);
    }

    unsigned int AudioAnalyzer::sample_rate() const {
        return impl->sample_rate.load(std::memory_order_relaxed);
    }

    void AudioAnalyzer::set_sensitivity(float sensitivity) {
        impl->sensitivity.store(std::clamp(sensitivity, 0.1f, 5.0f),
                                std::memory_order_relaxed);
    }

    float AudioAnalyzer::sensitivity() const {
        return impl->sensitivity.load(std::memory_order_relaxed);
    }

    AudioMetrics AudioAnalyzer::metrics() const {
        return {
            impl->amplitude.load(std::memory_order_relaxed),
            impl->frequency.load(std::memory_order_relaxed),
            impl->peak.load(std::memory_order_relaxed),
            impl->rms.load(std::memory_order_relaxed),
            impl->smooth.load(std::memory_order_relaxed),
            impl->low.load(std::memory_order_relaxed),
            impl->mid.load(std::memory_order_relaxed),
            impl->high.load(std::memory_order_relaxed),
        };
    }

    void AudioAnalyzer::compute_spectrum() {
        const int front = impl->spectrum_front.load(std::memory_order_acquire);
        std::array<float, FFT_SIZE * 2> complex{};
        for (std::size_t i = 0; i < FFT_SIZE; ++i) {
            const float hann =
                0.5f * (1.0f - std::cos(2.0f * PI * static_cast<float>(i) /
                                        static_cast<float>(FFT_SIZE - 1)));
            complex[2 * i] =
                impl->spectrum_samples[front][i].load(std::memory_order_relaxed) * hann;
        }

        fft_radix2(complex.data(), static_cast<int>(FFT_SIZE));

        constexpr float inverse = 2.0f / static_cast<float>(FFT_SIZE);
        for (std::size_t i = 0; i < FFT_SIZE / 2; ++i) {
            const float real = complex[2 * i];
            const float imaginary = complex[2 * i + 1];
            impl->spectrum_magnitudes[i] =
                std::sqrt(real * real + imaginary * imaginary) * inverse;
        }
    }

    const std::vector<float> &AudioAnalyzer::spectrum() const {
        return impl->spectrum_magnitudes;
    }

    class AudioRecorder::Impl {
      public:
        ~Impl() { stop(); }

        bool start(const std::string &filepath, unsigned int new_sample_rate,
                   unsigned int new_channels) {
            if (recording.load(std::memory_order_acquire) || new_channels == 0)
                return false;

            file.open(filepath, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                std::cerr << "acmx2: Failed to open audio recording file: " << filepath << "\n";
                return false;
            }

            sample_rate = new_sample_rate;
            channels = new_channels;
            data_size.store(0, std::memory_order_relaxed);
            fade_position.store(0, std::memory_order_relaxed);
            ring_head.store(0, std::memory_order_relaxed);
            ring_tail.store(0, std::memory_order_relaxed);
            write_wav_header(file, 0, sample_rate, static_cast<std::uint16_t>(channels));
            disk_running.store(true, std::memory_order_relaxed);
            recording.store(true, std::memory_order_release);
            disk_thread = std::thread(&Impl::write_to_disk, this);
            std::cout << "acmx2: Audio recording started: " << filepath << "\n";
            return true;
        }

        void stop() {
            recording.store(false, std::memory_order_release);
            while (active_captures.load(std::memory_order_acquire) != 0)
                std::this_thread::yield();

            disk_running.store(false, std::memory_order_release);
            if (disk_thread.joinable())
                disk_thread.join();

            if (file.is_open()) {
                const std::uint64_t bytes = data_size.load(std::memory_order_relaxed);
                const std::uint32_t wav_bytes = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(bytes, std::numeric_limits<std::uint32_t>::max()));
                write_wav_header(file, wav_bytes, sample_rate,
                                 static_cast<std::uint16_t>(channels));
                file.close();
                std::cout << "acmx2: Audio recording stopped (" << bytes
                          << " bytes written)\n";
            }
        }

        void capture(const float *samples, unsigned int frame_count,
                     unsigned int input_channels) {
            if (samples == nullptr || !recording.load(std::memory_order_acquire))
                return;

            active_captures.fetch_add(1, std::memory_order_acq_rel);
            if (!recording.load(std::memory_order_acquire)) {
                active_captures.fetch_sub(1, std::memory_order_release);
                return;
            }

            const unsigned int total_samples = frame_count * input_channels;
            const float current_gain = gain.load(std::memory_order_relaxed);
            std::uint32_t fade = fade_position.load(std::memory_order_relaxed);
            std::size_t head = ring_head.load(std::memory_order_relaxed);
            const std::size_t tail = ring_tail.load(std::memory_order_acquire);
            for (unsigned int i = 0; i < total_samples; ++i) {
                if (((head + 1) & RING_MASK) == (tail & RING_MASK))
                    break;

                float sample = std::clamp(samples[i] * current_gain, -1.0f, 1.0f);
                if (fade < RECORD_FADE_SAMPLES) {
                    sample *= static_cast<float>(fade) /
                              static_cast<float>(RECORD_FADE_SAMPLES);
                    ++fade;
                }
                ring[head & RING_MASK] =
                    static_cast<std::int16_t>(sample * 32767.0f);
                ++head;
            }
            ring_head.store(head, std::memory_order_release);
            fade_position.store(fade, std::memory_order_relaxed);
            active_captures.fetch_sub(1, std::memory_order_release);
        }

        void write_to_disk() {
            std::array<std::int16_t, WRITE_BATCH_SIZE> batch{};
            while (disk_running.load(std::memory_order_acquire) ||
                   ring_tail.load(std::memory_order_relaxed) !=
                       ring_head.load(std::memory_order_acquire)) {
                const std::size_t tail = ring_tail.load(std::memory_order_relaxed);
                const std::size_t head = ring_head.load(std::memory_order_acquire);
                if (tail == head) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                const std::size_t count =
                    std::min<std::size_t>(head - tail, WRITE_BATCH_SIZE);
                for (std::size_t i = 0; i < count; ++i)
                    batch[i] = ring[(tail + i) & RING_MASK];
                ring_tail.store(tail + count, std::memory_order_release);

                if (file.is_open()) {
                    const auto bytes = count * sizeof(std::int16_t);
                    file.write(reinterpret_cast<const char *>(batch.data()),
                               static_cast<std::streamsize>(bytes));
                    data_size.fetch_add(bytes, std::memory_order_relaxed);
                }
            }
        }

        std::ofstream file;
        std::atomic<bool> recording{false};
        std::atomic<bool> disk_running{false};
        std::atomic<unsigned int> active_captures{0};
        std::atomic<std::uint64_t> data_size{0};
        std::atomic<std::uint32_t> fade_position{0};
        std::atomic<float> gain{1.0f};
        unsigned int sample_rate = 44100;
        unsigned int channels = 2;

        std::array<std::int16_t, RING_CAPACITY> ring{};
        std::atomic<std::size_t> ring_head{0};
        std::atomic<std::size_t> ring_tail{0};
        std::thread disk_thread;
    };

    AudioRecorder::AudioRecorder() : impl(std::make_unique<Impl>()) {}
    AudioRecorder::~AudioRecorder() = default;

    bool AudioRecorder::start(const std::string &filepath, unsigned int sample_rate,
                              unsigned int channels) {
        return impl->start(filepath, sample_rate, channels);
    }

    void AudioRecorder::stop() {
        impl->stop();
    }

    bool AudioRecorder::is_recording() const {
        return impl->recording.load(std::memory_order_relaxed);
    }

    void AudioRecorder::capture(const float *samples, unsigned int frame_count,
                                unsigned int channels) {
        impl->capture(samples, frame_count, channels);
    }

    void AudioRecorder::set_gain(float gain) {
        impl->gain.store(std::clamp(gain, 0.0f, 2.0f), std::memory_order_relaxed);
    }

    float AudioRecorder::gain() const {
        return impl->gain.load(std::memory_order_relaxed);
    }

    double AudioRecorder::duration_seconds() const {
        const std::uint64_t bytes = impl->data_size.load(std::memory_order_relaxed);
        const std::uint64_t bytes_per_second =
            static_cast<std::uint64_t>(impl->sample_rate) * impl->channels *
            sizeof(std::int16_t);
        if (bytes_per_second == 0)
            return 0.0;
        return static_cast<double>(bytes) / static_cast<double>(bytes_per_second);
    }

    class AudioEngine::Impl {
      public:
        Impl() : stream(make_rt_audio()) {}

        ~Impl() { close(); }

        bool open(const AudioStreamConfig &config) {
            close();

            const std::vector<unsigned int> device_ids = stream.getDeviceIds();
            if (device_ids.empty()) {
                std::cerr << "acmx2: No audio devices found!\n";
                return false;
            }
            std::cout << "acmx2: Audio device found...\n";

            unsigned int input_device = 0;
            if (config.input_device >= 0) {
                input_device = static_cast<unsigned int>(config.input_device);
                std::cout << "acmx2: Using specified input device: " << input_device << "\n";
            } else {
                input_device = stream.getDefaultInputDevice();
                if (input_device == 0) {
                    for (const unsigned int id : device_ids) {
                        const RtAudio::DeviceInfo info = stream.getDeviceInfo(id);
                        if (info.isDefaultInput && info.inputChannels > 0) {
                            input_device = id;
                            break;
                        }
                    }
                }
                if (input_device == 0) {
                    for (const unsigned int id : device_ids) {
                        const RtAudio::DeviceInfo info = stream.getDeviceInfo(id);
                        if (info.inputChannels > 0) {
                            input_device = id;
                            break;
                        }
                    }
                }
                std::cout << "acmx2: Using default input device: " << input_device << "\n";
            }

            const RtAudio::DeviceInfo input_info = stream.getDeviceInfo(input_device);
            if (input_info.inputChannels == 0) {
                std::cerr << "acmx2: Input device has no input channels.\n";
                return false;
            }

            std::cout << "acmx2: Selected input device " << input_device << ": "
                      << input_info.name << "\n";
            std::cout << "acmx2:   Input channels: " << input_info.inputChannels << "\n";
            if (input_info.isDefaultInput)
                std::cout << "acmx2:   [DEFAULT INPUT]\n";

            input_channels = std::min(std::max(config.channels, 1U),
                                      input_info.inputChannels);
            pass_through = config.pass_through;
            analyzer.reset();
            analyzer.set_sensitivity(config.sensitivity);

            RtAudio::StreamParameters input_parameters;
            input_parameters.deviceId = input_device;
            input_parameters.nChannels = input_channels;
            input_parameters.firstChannel = 0;

            RtAudio::StreamParameters output_parameters;
            RtAudio::StreamParameters *output_parameters_ptr = nullptr;
            output_channels = 0;
            if (pass_through) {
                const unsigned int output_device =
                    config.output_device >= 0
                        ? static_cast<unsigned int>(config.output_device)
                        : stream.getDefaultOutputDevice();
                std::cout << "acmx2: Using "
                          << (config.output_device >= 0 ? "specified" : "default")
                          << " output device: " << output_device << "\n";

                const RtAudio::DeviceInfo output_info = stream.getDeviceInfo(output_device);
                if (output_info.outputChannels > 0) {
                    output_channels = std::min(2U, output_info.outputChannels);
                    output_parameters.deviceId = output_device;
                    output_parameters.nChannels = output_channels;
                    output_parameters.firstChannel = 0;
                    output_parameters_ptr = &output_parameters;
                    std::cout << "acmx2: Audio pass-through enabled on device "
                              << output_device << ": " << output_info.name << " ("
                              << output_channels << " ch)\n";
                } else {
                    std::cerr << "acmx2: Output device has no output channels, "
                                 "pass-through disabled.\n";
                    pass_through = false;
                }
            }

            unsigned int sample_rate = 44100;
            if (!input_info.sampleRates.empty() &&
                std::find(input_info.sampleRates.begin(), input_info.sampleRates.end(),
                          sample_rate) == input_info.sampleRates.end()) {
                sample_rate = 48000;
                if (std::find(input_info.sampleRates.begin(), input_info.sampleRates.end(),
                              sample_rate) == input_info.sampleRates.end())
                    sample_rate = input_info.sampleRates.front();
            }
            analyzer.set_sample_rate(sample_rate);

            unsigned int buffer_frames = static_cast<unsigned int>(FFT_SIZE);
            try {
                stream.openStream(output_parameters_ptr, &input_parameters, RTAUDIO_FLOAT32,
                                  sample_rate, &buffer_frames, &Impl::audio_callback, this);
                stream.startStream();
            } catch (const std::exception &error) {
                std::cerr << "acmx2: Audio error: " << error.what() << "\n";
                close();
                return false;
            } catch (...) {
                std::cerr << "acmx2: Unknown audio error occurred.\n";
                close();
                return false;
            }

            std::cout << "acmx2: Audio input stream opened (rate=" << sample_rate
                      << " Hz, channels=" << input_channels
                      << ", sensitivity=" << analyzer.sensitivity() << ")\n";
            return stream.isStreamOpen();
        }

        void close() {
            recorder.stop();
            if (!stream.isStreamOpen())
                return;
            try {
                if (stream.isStreamRunning())
                    stream.stopStream();
                stream.closeStream();
                std::cout << "acmx2: Audio stream closed.\n";
            } catch (const std::exception &error) {
                std::cerr << "acmx2: Error closing audio stream: " << error.what() << "\n";
            }
        }

        static int audio_callback(void *output_buffer, void *input_buffer,
                                  unsigned int frame_count, double,
                                  RtAudioStreamStatus status, void *user_data) {
            return static_cast<Impl *>(user_data)
                ->process_callback(output_buffer, input_buffer, frame_count, status);
        }

        int process_callback(void *output_buffer, void *input_buffer,
                             unsigned int frame_count, RtAudioStreamStatus status) {
            auto *input = static_cast<float *>(input_buffer);
            auto *output = static_cast<float *>(output_buffer);
            if (status || input == nullptr) {
                if (output != nullptr && output_channels > 0)
                    std::fill_n(output, frame_count * output_channels, 0.0f);
                return 0;
            }

            if (output != nullptr && output_channels > 0) {
                for (unsigned int frame = 0; frame < frame_count; ++frame) {
                    for (unsigned int channel = 0; channel < output_channels; ++channel) {
                        const unsigned int input_channel =
                            channel < input_channels ? channel : 0;
                        output[frame * output_channels + channel] =
                            pass_through ? input[frame * input_channels + input_channel]
                                         : 0.0f;
                    }
                }
            }

            analyzer.process_samples(input, frame_count, input_channels);
            recorder.capture(input, frame_count, input_channels);
            return 0;
        }

        RtAudio stream;
        AudioAnalyzer analyzer;
        AudioRecorder recorder;
        unsigned int input_channels = 2;
        unsigned int output_channels = 0;
        bool pass_through = false;
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

    unsigned int AudioEngine::input_channels() const {
        return impl->input_channels;
    }

    AudioAnalyzer &AudioEngine::analyzer() {
        return impl->analyzer;
    }

    const AudioAnalyzer &AudioEngine::analyzer() const {
        return impl->analyzer;
    }

    AudioRecorder &AudioEngine::recorder() {
        return impl->recorder;
    }

    const AudioRecorder &AudioEngine::recorder() const {
        return impl->recorder;
    }

    void AudioEngine::list_devices() {
        RtAudio stream = make_rt_audio();
        const std::vector<unsigned int> device_ids = stream.getDeviceIds();
        std::cout << "acmx2: Found " << device_ids.size() << " audio device(s):\n";
        for (const unsigned int id : device_ids) {
            const RtAudio::DeviceInfo info = stream.getDeviceInfo(id);
            std::cout << "  Device " << id << ": " << info.name;
            if (info.isDefaultInput)
                std::cout << " [DEFAULT INPUT]";
            if (info.isDefaultOutput)
                std::cout << " [DEFAULT OUTPUT]";
            std::cout << "\n";
            std::cout << "    Input channels: " << info.inputChannels << "\n";
            std::cout << "    Output channels: " << info.outputChannels << "\n";
            std::cout << "    Sample rates: ";
            for (const unsigned int rate : info.sampleRates)
                std::cout << rate << " ";
            std::cout << "\n";
        }
    }

} // namespace acmx2::audio
