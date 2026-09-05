#ifndef ACMXVK_APP_INTERFACE_CLIENT_HPP
#define ACMXVK_APP_INTERFACE_CLIENT_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace acmxvk {
    struct InterfaceUniformValue {
        std::string name;
        float value = 0.0F;
    };

    struct InterfaceMultipassState {
        bool enabled = false;
        std::vector<std::string> shader_names;
    };

    struct InterfacePlaybackState {
        bool repeat = false;
        bool normalized_time = false;
    };

    struct InterfaceOverlayState {
        bool display_filter = false;
        bool watermark_enabled = false;
        std::string watermark_text;
        std::array<std::uint8_t, 3> watermark_color{};
    };

    struct InterfaceGpuFilterState {
        bool enabled = false;
        int frame_buffer_size = 8;
        std::vector<int> filter_indices;
    };

    struct InterfaceAudioFileState {
        std::uint32_t request_sequence = 0;
        std::string path;
        int output_device = -1;
        bool pass_through = false;
        bool trunc = false;
        bool repeat = false;
    };

    struct InterfaceReloadState {
        std::uint32_t request_sequence = 0;
        std::int32_t shader_index = -1;
        std::string path;
    };

    struct InterfaceState {
        std::uint32_t sequence = 0;
        std::string selected_shader_name;
        InterfaceMultipassState multipass;
        std::vector<InterfaceUniformValue> uniform_values;
        InterfacePlaybackState playback;
        InterfaceOverlayState overlay;
        InterfaceGpuFilterState gpu_filters;
        InterfaceAudioFileState audio_file;
        InterfaceReloadState reload;
    };

    class InterfaceClient {
      public:
        InterfaceClient();
        ~InterfaceClient();
        InterfaceClient(const InterfaceClient &) = delete;
        InterfaceClient &operator=(const InterfaceClient &) = delete;
        InterfaceClient(InterfaceClient &&) = delete;
        InterfaceClient &operator=(InterfaceClient &&) = delete;

        [[nodiscard]] bool open();
        void close() noexcept;
        [[nodiscard]] bool is_open() const noexcept;
        [[nodiscard]] bool read(InterfaceState &state) const;

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
} // namespace acmxvk

#endif
