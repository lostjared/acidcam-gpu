#ifndef ACMXVK_APP_RESOURCE_PATHS_HPP
#define ACMXVK_APP_RESOURCE_PATHS_HPP

#include "options.hpp"

#include <cstddef>

namespace acmxvk {
    [[nodiscard]] std::vector<fs::path>
    resource_directories(const Options &options);
    [[nodiscard]] fs::path find_resource(const Options &options,
                                         const fs::path &relative_path);
    [[nodiscard]] fs::path
    sprite_vertex_shader_path(const Options &options);
    [[nodiscard]] fs::path echo_cache_shader_path(const Options &options);
    [[nodiscard]] fs::path flip_shader_path(const Options &options);
    [[nodiscard]] fs::path passthrough_shader_path(const Options &options);
    [[nodiscard]] fs::path
    human_composite_shader_path(const Options &options);
    [[nodiscard]] fs::path crossfade_shader_path(const Options &options,
                                                 std::size_t shader_index);
    [[nodiscard]] fs::path model_vertex_shader_path(const Options &options);
    [[nodiscard]] fs::path model_fragment_shader_path(const Options &options);
    [[nodiscard]] fs::path default_model_path(const Options &options);
    [[nodiscard]] fs::path overlay_font_path(const Options &options);
} // namespace acmxvk

#endif
