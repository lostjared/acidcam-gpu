#ifndef ACMX2_AUDIO_HPP
#define ACMX2_AUDIO_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace acmx2::audio {

    inline constexpr std::size_t FFT_SIZE = 512;

    struct AudioMetrics {
        float amplitude = 0.0f;
        float frequency = 0.0f;
        float peak = 0.0f;
        float rms = 0.0f;
        float smooth = 0.0f;
        float low = 0.0f;
        float mid = 0.0f;
        float high = 0.0f;
    };

    /**
     * Owns the audio-reactive analysis state shared by live and file audio.
     *
     * process_samples() is safe to call from the audio callback while the render
     * thread reads metrics and computes the spectrum.
     */
    class AudioAnalyzer {
      public:
        AudioAnalyzer();
        ~AudioAnalyzer();

        AudioAnalyzer(const AudioAnalyzer &) = delete;
        AudioAnalyzer &operator=(const AudioAnalyzer &) = delete;

        void process_samples(const float *samples, unsigned int frame_count, unsigned int channels);
        void reset();

        void set_sample_rate(unsigned int sample_rate);
        unsigned int sample_rate() const;
        void set_sensitivity(float sensitivity);
        float sensitivity() const;

        AudioMetrics metrics() const;

        void compute_spectrum();
        const std::vector<float> &spectrum() const;
        static constexpr int spectrum_bin_count() { return static_cast<int>(FFT_SIZE / 2); }

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

    /**
     * Asynchronously records interleaved float PCM samples to a 16-bit WAV file.
     */
    class AudioRecorder {
      public:
        AudioRecorder();
        ~AudioRecorder();

        AudioRecorder(const AudioRecorder &) = delete;
        AudioRecorder &operator=(const AudioRecorder &) = delete;

        bool start(const std::string &filepath, unsigned int sample_rate, unsigned int channels);
        void stop();
        bool is_recording() const;

        void capture(const float *samples, unsigned int frame_count, unsigned int channels);

        void set_gain(float gain);
        float gain() const;
        double duration_seconds() const;

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

    struct AudioStreamConfig {
        unsigned int channels = 2;
        float sensitivity = 1.0f;
        int input_device = -1;
        int output_device = -1;
        bool pass_through = false;
    };

    /**
     * RAII owner for the RtAudio stream, analysis pipeline, and WAV recorder.
     */
    class AudioEngine {
      public:
        AudioEngine();
        ~AudioEngine();

        AudioEngine(const AudioEngine &) = delete;
        AudioEngine &operator=(const AudioEngine &) = delete;

        bool open(const AudioStreamConfig &config);
        void close();
        bool is_open() const;
        unsigned int input_channels() const;

        AudioAnalyzer &analyzer();
        const AudioAnalyzer &analyzer() const;
        AudioRecorder &recorder();
        const AudioRecorder &recorder() const;

        static void list_devices();

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace acmx2::audio

#endif
