#ifndef ACMXVK_APP_PLAYLIST_HPP
#define ACMXVK_APP_PLAYLIST_HPP

#include "options.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace acmxvk {
    struct PlaylistNode {
        std::string name;
        std::vector<fs::path> shaders;
    };

    [[nodiscard]] fs::path
    find_shader_path(const std::vector<fs::path> &available_shaders,
                     const fs::path &library_directory, std::string name);
    [[nodiscard]] std::vector<PlaylistNode>
    load_playlist(const fs::path &playlist_path,
                  const std::vector<fs::path> &available_shaders,
                  const fs::path &library_directory,
                  std::ostream &warning_output);
    [[nodiscard]] std::size_t
    playlist_shader_count(const std::vector<PlaylistNode> &playlist) noexcept;
} // namespace acmxvk

#endif
