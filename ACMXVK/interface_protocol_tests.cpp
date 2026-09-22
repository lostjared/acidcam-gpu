#include "interface_control.hpp"

#include "../ACMX2/shader_selection_shm.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string_view>

static_assert(acmxvk::ipc::SHADER_SELECTION_VERSION == acmx2::ipc::kShaderSelectionVersion);
static_assert(sizeof(acmxvk::ipc::ShaderSelectionData) == sizeof(acmx2::ipc::ShaderSelectionShmData));
static_assert(offsetof(acmxvk::ipc::ShaderSelectionData, effect_pack_manifest_path) == offsetof(acmx2::ipc::ShaderSelectionShmData, effect_pack_manifest_path));
static_assert(offsetof(acmxvk::ipc::ShaderSelectionData, effect_pack_sequence) == offsetof(acmx2::ipc::ShaderSelectionShmData, effect_pack_sequence));
static_assert(offsetof(acmxvk::ipc::ShaderSelectionData, selected_shader_name) == offsetof(acmx2::ipc::ShaderSelectionShmData, selected_shader_name));
static_assert(offsetof(acmxvk::ipc::ShaderSelectionData, sequence) == offsetof(acmx2::ipc::ShaderSelectionShmData, sequence));

int main() {
    if (std::string_view(acmxvk::ipc::SHADER_SELECTION_SHM_NAME) != acmx2::ipc::kShaderSelectionShmName || std::string_view(acmxvk::ipc::SHADER_SELECTION_SEMAPHORE_NAME) != acmx2::ipc::kShaderSelectionSemaphoreName) {
        std::cerr << "interface protocol shared-memory names do not match\n";
        return 1;
    }
#ifdef __linux__
    sem_t unavailable;
    if (::sem_init(&unavailable, 0, 0) != 0) {
        std::cerr << "could not initialize test semaphore\n";
        return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    const bool interface_locked = static_cast<bool>(acmx2::ipc::ShaderSelectionLock(&unavailable));
    const bool engine_locked = static_cast<bool>(acmxvk::ipc::InterfaceLock(&unavailable));
    const auto elapsed = std::chrono::steady_clock::now() - start;
    ::sem_destroy(&unavailable);
    if (interface_locked || engine_locked || elapsed > std::chrono::seconds(3)) {
        std::cerr << "interface lock did not time out promptly\n";
        return 1;
    }
#endif
    return 0;
}
