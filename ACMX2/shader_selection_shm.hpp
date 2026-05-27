#ifndef ACMX2_SHADER_SELECTION_SHM_HPP
#define ACMX2_SHADER_SELECTION_SHM_HPP

#include <cstdint>

namespace acmx2::ipc {

inline constexpr const char *kShaderSelectionShmName = "/acmx2_shader_selection";
inline constexpr std::uint32_t kShaderSelectionMagic = 0x41434D58; // 'ACMX'
inline constexpr std::uint32_t kShaderSelectionVersion = 3;
inline constexpr std::uint32_t kShaderSelectionMaxPassCount = 64;

struct ShaderSelectionShmData {
    std::uint32_t magic = kShaderSelectionMagic;
    std::uint32_t version = kShaderSelectionVersion;
    std::int32_t selected_index = -1;
    std::uint32_t shader_pass_count = 0;
    std::uint8_t shader_pass_enabled = 0;
    std::uint8_t repeat_enabled = 0;
    std::uint8_t reserved[2] = {0, 0};
    std::int32_t shader_pass_indices[kShaderSelectionMaxPassCount] = {};
    std::uint32_t sequence = 0;
};

} // namespace acmx2::ipc

#endif