#ifndef ACMX2_SHADER_SELECTION_SHM_HPP
#define ACMX2_SHADER_SELECTION_SHM_HPP

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

namespace acmx2::ipc {

    inline constexpr const char *kShaderSelectionShmName = "/acmx2_shader_selection";
    inline constexpr const char *kShaderSelectionSemaphoreName =
        "/acmx2_shm_v10";
#ifdef _WIN32
    inline constexpr const wchar_t *kShaderSelectionMappingNameWindows =
        L"Local\\ACMX2ShaderSelectionV10";
    inline constexpr const wchar_t *kShaderSelectionMutexNameWindows =
        L"Local\\ACMX2ShaderSelectionMutexV10";
#endif
    inline constexpr std::uint32_t kShaderSelectionMagic = 0x41434D58; // 'ACMX'
    inline constexpr std::uint32_t kShaderSelectionVersion = 10;
    inline constexpr std::uint32_t kShaderSelectionMaxPassCount = 64;
    inline constexpr std::uint32_t kShaderSelectionMaxGpuFilterCount = 64;
    inline constexpr std::uint32_t kShaderSelectionMaxWatermarkText = 256;
    inline constexpr std::uint32_t kShaderSelectionMaxReloadPath = 1024;
    inline constexpr std::uint32_t kShaderSelectionMaxCustomUniforms = 64;
    inline constexpr std::uint32_t kShaderSelectionMaxUniformName = 64;
    inline constexpr std::uint32_t kShaderSelectionMaxAudioFilePath = 4096;
    inline constexpr std::uint32_t kShaderSelectionMaxShaderName = 1024;

    struct ShaderSelectionShmData {
        std::uint32_t magic = kShaderSelectionMagic;
        std::uint32_t version = kShaderSelectionVersion;
        std::int32_t selected_index = -1;
        std::uint32_t shader_pass_count = 0;
        std::uint8_t shader_pass_enabled = 0;
        std::uint8_t repeat_enabled = 0;
        std::uint8_t display_filter_enabled = 0;
        std::uint8_t watermark_enabled = 0;
        std::uint8_t normalized_time_enabled = 0;
        std::uint8_t reserved_flags[3] = {0, 0, 0};
        std::int32_t shader_pass_indices[kShaderSelectionMaxPassCount] = {};
        char shader_pass_names[kShaderSelectionMaxPassCount]
                              [kShaderSelectionMaxShaderName] = {};
        std::uint32_t gpu_filter_count = 0;
        std::uint8_t gpu_filter_enabled = 0;
        std::uint8_t gpu_buffer_size = 8;
        std::uint8_t watermark_r = 255;
        std::uint8_t watermark_g = 0;
        std::uint8_t watermark_b = 150;
        std::uint8_t reserved[3] = {0, 0, 0};
        std::int32_t gpu_filter_indices[kShaderSelectionMaxGpuFilterCount] = {};
        char watermark_text[kShaderSelectionMaxWatermarkText] = {};
        std::int32_t reload_shader_index = -1;
        char reload_shader_path[kShaderSelectionMaxReloadPath] = {};
        std::uint32_t reload_sequence = 0;
        std::uint32_t custom_uniform_count = 0;
        char custom_uniform_names[kShaderSelectionMaxCustomUniforms]
                                 [kShaderSelectionMaxUniformName] = {};
        float custom_uniform_values[kShaderSelectionMaxCustomUniforms] = {};
        char audio_file_path[kShaderSelectionMaxAudioFilePath] = {};
        std::int32_t audio_output_device = -1;
        std::uint8_t audio_pass_through = 0;
        std::uint8_t audio_trunc = 0;
        std::uint8_t audio_repeat = 0;
        std::uint8_t audio_reserved = 0;
        std::uint32_t audio_file_sequence = 0;
        char selected_shader_name[kShaderSelectionMaxShaderName] = {};
        std::uint32_t sequence = 0;
    };

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    class ShaderSelectionLock {
      public:
#if defined(__linux__) || defined(__APPLE__)
        explicit ShaderSelectionLock(sem_t *semaphoreValue)
            : semaphore(semaphoreValue) {
            if (semaphore == nullptr || semaphore == SEM_FAILED)
                return;
            while (::sem_wait(semaphore) != 0) {
                if (errno != EINTR)
                    return;
            }
            locked = true;
        }
#else
        explicit ShaderSelectionLock(HANDLE mutexValue) : mutex(mutexValue) {
            if (mutex == nullptr)
                return;
            const DWORD result = ::WaitForSingleObject(mutex, INFINITE);
            locked = result == WAIT_OBJECT_0 || result == WAIT_ABANDONED;
        }
#endif

        ~ShaderSelectionLock() {
            if (!locked)
                return;
#if defined(__linux__) || defined(__APPLE__)
            ::sem_post(semaphore);
#else
            ::ReleaseMutex(mutex);
#endif
        }

        ShaderSelectionLock(const ShaderSelectionLock &) = delete;
        ShaderSelectionLock &operator=(const ShaderSelectionLock &) = delete;

        explicit operator bool() const {
            return locked;
        }

      private:
#if defined(__linux__) || defined(__APPLE__)
        sem_t *semaphore = nullptr;
#else
        HANDLE mutex = nullptr;
#endif
        bool locked = false;
    };
#endif

} // namespace acmx2::ipc

#endif
