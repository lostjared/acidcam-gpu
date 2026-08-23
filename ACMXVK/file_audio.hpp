#ifndef ACMXVK_FILE_AUDIO_HPP
#define ACMXVK_FILE_AUDIO_HPP

#include <cstddef>
#include <memory>
#include <string>

namespace acmxvk::audio {

    class AudioEngine;

    class FileAudioSource {
      public:
        FileAudioSource();
        ~FileAudioSource();

        FileAudioSource(const FileAudioSource &) = delete;
        FileAudioSource &operator=(const FileAudioSource &) = delete;

        bool open(const std::string &path);
        void close();
        void set_repeat(bool enabled);
        bool enable_output(int device = -1);
        [[nodiscard]] bool has_output_clock() const;
        [[nodiscard]] bool is_open() const;
        [[nodiscard]] bool is_active() const;
        [[nodiscard]] double duration_seconds() const;
        [[nodiscard]] const std::string &path() const;
        [[nodiscard]] std::size_t track_count() const;
        [[nodiscard]] const std::string &current_track_path() const;

        bool process_frame(double frames_per_second, AudioEngine &engine);

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace acmxvk::audio

#endif
