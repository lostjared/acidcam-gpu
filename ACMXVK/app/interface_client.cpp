#include "interface_client.hpp"

#include "../interface_control.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <iterator>
#include <semaphore.h>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace acmxvk {
    struct InterfaceClient::Impl {
        int shm_fd = -1;
        ipc::ShaderSelectionData *selection = nullptr;
        sem_t *semaphore = SEM_FAILED;
    };

    InterfaceClient::InterfaceClient() : impl(std::make_unique<Impl>()) {}

    InterfaceClient::~InterfaceClient() {
        close();
    }

    bool InterfaceClient::open() {
        close();

        impl->semaphore =
            ::sem_open(ipc::SHADER_SELECTION_SEMAPHORE_NAME, 0);
        if (impl->semaphore == SEM_FAILED) {
            std::cerr << "acmxvk: interface control unavailable: sem_open("
                      << ipc::SHADER_SELECTION_SEMAPHORE_NAME
                      << ") failed: " << std::strerror(errno) << '\n';
            return false;
        }

        impl->shm_fd =
            ::shm_open(ipc::SHADER_SELECTION_SHM_NAME, O_RDWR, 0666);
        if (impl->shm_fd < 0) {
            const int open_error = errno;
            std::cerr << "acmxvk: interface control unavailable: shm_open("
                      << ipc::SHADER_SELECTION_SHM_NAME
                      << ") failed: " << std::strerror(open_error) << '\n';
            close();
            return false;
        }

        constexpr std::size_t SHARED_MEMORY_SIZE =
            sizeof(ipc::ShaderSelectionData);
        struct stat shm_stat{};
        if (::fstat(impl->shm_fd, &shm_stat) != 0) {
            const int stat_error = errno;
            std::cerr << "acmxvk: interface control unavailable: fstat("
                      << ipc::SHADER_SELECTION_SHM_NAME
                      << ") failed: " << std::strerror(stat_error) << '\n';
            close();
            return false;
        }
        if (shm_stat.st_size != static_cast<off_t>(SHARED_MEMORY_SIZE)) {
            std::cerr << "acmxvk: interface control unavailable: "
                      << ipc::SHADER_SELECTION_SHM_NAME << " has size "
                      << shm_stat.st_size << " bytes; expected "
                      << SHARED_MEMORY_SIZE << '\n';
            close();
            return false;
        }

        void *mapped = ::mmap(nullptr, SHARED_MEMORY_SIZE,
                              PROT_READ | PROT_WRITE, MAP_SHARED,
                              impl->shm_fd, 0);
        if (mapped == MAP_FAILED) {
            const int map_error = errno;
            std::cerr << "acmxvk: interface control unavailable: mmap("
                      << ipc::SHADER_SELECTION_SHM_NAME << ", "
                      << SHARED_MEMORY_SIZE
                      << ") failed: " << std::strerror(map_error) << '\n';
            close();
            return false;
        }
        impl->selection = static_cast<ipc::ShaderSelectionData *>(mapped);
        return true;
    }

    void InterfaceClient::close() noexcept {
        if (impl->selection != nullptr) {
            ::munmap(impl->selection, sizeof(ipc::ShaderSelectionData));
            impl->selection = nullptr;
        }
        if (impl->shm_fd >= 0) {
            ::close(impl->shm_fd);
            impl->shm_fd = -1;
        }
        if (impl->semaphore != SEM_FAILED) {
            ::sem_close(impl->semaphore);
            impl->semaphore = SEM_FAILED;
        }
    }

    bool InterfaceClient::read(InterfaceState &state) const {
        if (impl->selection == nullptr || impl->semaphore == SEM_FAILED) {
            return false;
        }
        ipc::SemaphoreLock lock(impl->semaphore);
        if (!lock) {
            return false;
        }
        if (impl->selection->magic != ipc::SHADER_SELECTION_MAGIC ||
            impl->selection->version != ipc::SHADER_SELECTION_VERSION) {
            return false;
        }

        InterfaceState next;
        next.sequence = impl->selection->sequence;
        const auto name_end = std::find(
            std::begin(impl->selection->selected_shader_name),
            std::end(impl->selection->selected_shader_name), '\0');
        next.selected_shader_name.assign(
            std::begin(impl->selection->selected_shader_name), name_end);

        next.multipass.enabled = impl->selection->shader_pass_enabled != 0;
        const std::uint32_t pass_count = std::min(
            impl->selection->shader_pass_count, ipc::MAX_PASS_COUNT);
        next.multipass.shader_names.reserve(pass_count);
        for (std::uint32_t index = 0; index < pass_count; ++index) {
            const char *name_begin = impl->selection->shader_pass_names[index];
            const char *name_limit = name_begin + ipc::MAX_SHADER_NAME;
            const char *shader_name_end =
                std::find(name_begin, name_limit, '\0');
            if (shader_name_end != name_begin) {
                next.multipass.shader_names.emplace_back(name_begin,
                                                         shader_name_end);
            }
        }

        const std::uint32_t uniform_count = std::min(
            impl->selection->custom_uniform_count, ipc::MAX_CUSTOM_UNIFORMS);
        next.uniform_values.reserve(uniform_count);
        for (std::uint32_t index = 0; index < uniform_count; ++index) {
            const char *name_begin =
                impl->selection->custom_uniform_names[index];
            const char *name_limit = name_begin + ipc::MAX_UNIFORM_NAME;
            const char *uniform_name_end =
                std::find(name_begin, name_limit, '\0');
            next.uniform_values.push_back(
                {std::string(name_begin, uniform_name_end),
                 impl->selection->custom_uniform_values[index]});
        }

        next.playback.repeat = impl->selection->repeat_enabled != 0;
        next.playback.normalized_time =
            impl->selection->normalized_time_enabled != 0;
        next.overlay.display_filter =
            impl->selection->display_filter_enabled != 0;
        next.overlay.watermark_enabled =
            impl->selection->watermark_enabled != 0;
        const auto watermark_end = std::find(
            std::begin(impl->selection->watermark_text),
            std::end(impl->selection->watermark_text), '\0');
        next.overlay.watermark_text.assign(
            std::begin(impl->selection->watermark_text), watermark_end);
        next.overlay.watermark_color = {impl->selection->watermark_r,
                                        impl->selection->watermark_g,
                                        impl->selection->watermark_b};

        next.gpu_filters.enabled =
            impl->selection->gpu_filter_enabled != 0;
        next.gpu_filters.frame_buffer_size =
            static_cast<int>(impl->selection->gpu_buffer_size);
        const std::uint32_t gpu_filter_count = std::min(
            impl->selection->gpu_filter_count, ipc::MAX_GPU_FILTER_COUNT);
        next.gpu_filters.filter_indices.reserve(gpu_filter_count);
        for (std::uint32_t index = 0; index < gpu_filter_count; ++index) {
            const int filter_index =
                impl->selection->gpu_filter_indices[index];
            if (filter_index >= 0) {
                next.gpu_filters.filter_indices.push_back(filter_index);
            }
        }

        next.audio_file.request_sequence =
            impl->selection->audio_file_sequence;
        const auto audio_path_end = std::find(
            std::begin(impl->selection->audio_file_path),
            std::end(impl->selection->audio_file_path), '\0');
        next.audio_file.path.assign(
            std::begin(impl->selection->audio_file_path), audio_path_end);
        next.audio_file.output_device =
            impl->selection->audio_output_device;
        next.audio_file.pass_through =
            impl->selection->audio_pass_through != 0;
        next.audio_file.trunc = impl->selection->audio_trunc != 0;
        next.audio_file.repeat = impl->selection->audio_repeat != 0;

        next.reload.request_sequence = impl->selection->reload_sequence;
        const auto reload_path_end = std::find(
            std::begin(impl->selection->reload_shader_path),
            std::end(impl->selection->reload_shader_path), '\0');
        next.reload.path.assign(
            std::begin(impl->selection->reload_shader_path), reload_path_end);

        state = std::move(next);
        return true;
    }
} // namespace acmxvk
