#ifndef ACMXVK_AUDIO_HPP
#define ACMXVK_AUDIO_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace acmxvk::audio {

    struct AudioMetrics {
        float amplitude = 0.0F;
        float frequency = 0.0F;
        float peak = 0.0F;
        float rms = 0.0F;
        float smooth = 0.0F;
    };

    struct AudioStreamConfig {
        unsigned int channels = 2;
        float sensitivity = 1.0F;
        int input_device = -1;
    };

    class AudioEngine {
      public:
        static constexpr std::size_t FFT_SIZE = 512;

        AudioEngine();
        ~AudioEngine();

        AudioEngine(const AudioEngine &) = delete;
        AudioEngine &operator=(const AudioEngine &) = delete;

        bool open(const AudioStreamConfig &config);
        void close();
        [[nodiscard]] bool is_open() const;

        [[nodiscard]] AudioMetrics metrics() const;
        [[nodiscard]] unsigned int sample_rate() const;
        void set_sensitivity(float sensitivity);
        [[nodiscard]] float sensitivity() const;
        [[nodiscard]] std::vector<float> spectrum() const;
        static constexpr std::uint32_t spectrum_bin_count() {
            return static_cast<std::uint32_t>(FFT_SIZE / 2);
        }

        static void list_devices();

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace acmxvk::audio

#endif
