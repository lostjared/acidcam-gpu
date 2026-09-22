#ifndef ACMXVK_APP_EFFECT_PACK_BUILD_HPP
#define ACMXVK_APP_EFFECT_PACK_BUILD_HPP

#include "effect_pack.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace acmxvk {

    struct EffectPackBuildOptions {
        std::string glslc_executable = "glslc";
        std::size_t parallel_jobs = 1;
        bool force = false;
        std::function<void(std::size_t, std::size_t)> progress;
    };

    struct EffectPackBuildResult {
        std::filesystem::path build_root;
        std::vector<std::filesystem::path> compiled_passes;
        std::size_t compiled = 0;
        std::size_t copied = 0;
        std::size_t current = 0;
        std::size_t removed_temporary_files = 0;
    };

    [[nodiscard]] EffectPackBuildResult build_effect_pack(const EffectPack &pack, const EffectPackBuildOptions &options = {});
    [[nodiscard]] EffectPackBuildResult load_effect_pack_cache(const EffectPack &pack);

} // namespace acmxvk

#endif
