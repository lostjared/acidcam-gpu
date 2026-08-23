#ifndef ACMXVK_FILE_AUDIO_HPP
#define ACMXVK_FILE_AUDIO_HPP

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
        [[nodiscard]] bool is_open() const;
        [[nodiscard]] bool is_active() const;
        [[nodiscard]] double duration_seconds() const;
        [[nodiscard]] const std::string &path() const;

        bool process_frame(double frames_per_second, AudioEngine &engine);

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace acmxvk::audio

#endif
