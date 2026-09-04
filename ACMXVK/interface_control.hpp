#ifndef ACMXVK_INTERFACE_CONTROL_HPP
#define ACMXVK_INTERFACE_CONTROL_HPP

#include <cstdint>
#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <semaphore.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace acmxvk::ipc {
    // This layout intentionally mirrors ACMX2/shader_selection_shm.hpp version
    // 10. Keeping a local protocol declaration lets ACMXVK remain buildable as
    // a standalone source tree while sharing the Qt launcher's control block.
    inline constexpr const char *SHADER_SELECTION_SHM_NAME =
        "/acmx2_shader_selection";
    inline constexpr const char *SHADER_SELECTION_SEMAPHORE_NAME =
        "/acmx2_shm_v10";
#ifdef _WIN32
    inline constexpr const wchar_t *SHADER_SELECTION_MAPPING_NAME_WINDOWS =
        L"Local\\ACMX2ShaderSelectionV10";
    inline constexpr const wchar_t *SHADER_SELECTION_MUTEX_NAME_WINDOWS =
        L"Local\\ACMX2ShaderSelectionMutexV10";
#endif
    inline constexpr std::uint32_t SHADER_SELECTION_MAGIC = 0x41434D58;
    inline constexpr std::uint32_t SHADER_SELECTION_VERSION = 10;
    inline constexpr std::uint32_t MAX_PASS_COUNT = 64;
    inline constexpr std::uint32_t MAX_GPU_FILTER_COUNT = 64;
    inline constexpr std::uint32_t MAX_WATERMARK_TEXT = 256;
    inline constexpr std::uint32_t MAX_RELOAD_PATH = 1024;
    inline constexpr std::uint32_t MAX_CUSTOM_UNIFORMS = 64;
    inline constexpr std::uint32_t MAX_UNIFORM_NAME = 64;
    inline constexpr std::uint32_t MAX_AUDIO_FILE_PATH = 4096;
    inline constexpr std::uint32_t MAX_SHADER_NAME = 1024;

    struct ShaderSelectionData {
        std::uint32_t magic = SHADER_SELECTION_MAGIC;
        std::uint32_t version = SHADER_SELECTION_VERSION;
        std::int32_t selected_index = -1;
        std::uint32_t shader_pass_count = 0;
        std::uint8_t shader_pass_enabled = 0;
        std::uint8_t repeat_enabled = 0;
        std::uint8_t display_filter_enabled = 0;
        std::uint8_t watermark_enabled = 0;
        std::uint8_t normalized_time_enabled = 0;
        std::uint8_t reserved_flags[3] = {0, 0, 0};
        std::int32_t shader_pass_indices[MAX_PASS_COUNT] = {};
        char shader_pass_names[MAX_PASS_COUNT][MAX_SHADER_NAME] = {};
        std::uint32_t gpu_filter_count = 0;
        std::uint8_t gpu_filter_enabled = 0;
        std::uint8_t gpu_buffer_size = 8;
        std::uint8_t watermark_r = 255;
        std::uint8_t watermark_g = 0;
        std::uint8_t watermark_b = 150;
        std::uint8_t reserved[3] = {0, 0, 0};
        std::int32_t gpu_filter_indices[MAX_GPU_FILTER_COUNT] = {};
        char watermark_text[MAX_WATERMARK_TEXT] = {};
        std::int32_t reload_shader_index = -1;
        char reload_shader_path[MAX_RELOAD_PATH] = {};
        std::uint32_t reload_sequence = 0;
        std::uint32_t custom_uniform_count = 0;
        char custom_uniform_names[MAX_CUSTOM_UNIFORMS][MAX_UNIFORM_NAME] = {};
        float custom_uniform_values[MAX_CUSTOM_UNIFORMS] = {};
        char audio_file_path[MAX_AUDIO_FILE_PATH] = {};
        std::int32_t audio_output_device = -1;
        std::uint8_t audio_pass_through = 0;
        std::uint8_t audio_trunc = 0;
        std::uint8_t audio_repeat = 0;
        std::uint8_t audio_reserved = 0;
        std::uint32_t audio_file_sequence = 0;
        char selected_shader_name[MAX_SHADER_NAME] = {};
        std::uint32_t sequence = 0;
    };

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    class InterfaceLock {
      public:
#if defined(__linux__) || defined(__APPLE__)
        explicit InterfaceLock(sem_t *value) : semaphore(value) {
            if (semaphore == nullptr || semaphore == SEM_FAILED)
                return;
            while (::sem_wait(semaphore) != 0) {
                if (errno != EINTR)
                    return;
            }
            locked = true;
        }
#else
        explicit InterfaceLock(HANDLE value) : mutex(value) {
            if (mutex == nullptr)
                return;
            const DWORD result = ::WaitForSingleObject(mutex, INFINITE);
            locked = result == WAIT_OBJECT_0 || result == WAIT_ABANDONED;
        }
#endif

        ~InterfaceLock() {
            if (!locked)
                return;
#if defined(__linux__) || defined(__APPLE__)
            ::sem_post(semaphore);
#else
            ::ReleaseMutex(mutex);
#endif
        }

        InterfaceLock(const InterfaceLock &) = delete;
        InterfaceLock &operator=(const InterfaceLock &) = delete;

        explicit operator bool() const { return locked; }

      private:
#if defined(__linux__) || defined(__APPLE__)
        sem_t *semaphore = nullptr;
#else
        HANDLE mutex = nullptr;
#endif
        bool locked = false;
    };
#endif
} // namespace acmxvk::ipc

#endif
