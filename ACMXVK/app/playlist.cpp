#include "playlist.hpp"

#include "../input_validation.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace acmxvk {
    namespace {
        [[nodiscard]] std::string trim_text(std::string text) {
            const auto first = std::find_if_not(
                text.begin(), text.end(),
                [](unsigned char character) { return std::isspace(character); });
            const auto last = std::find_if_not(
                                  text.rbegin(), text.rend(),
                                  [](unsigned char character) { return std::isspace(character); })
                                  .base();
            if (first >= last) {
                return {};
            }
            return std::string(first, last);
        }
    } // namespace

    fs::path find_shader_path(
        const std::vector<fs::path> &available_shaders,
        const fs::path &library_directory, std::string name) {
        name = trim_text(std::move(name));
        if (name.empty()) {
            return {};
        }

        fs::path requested(name);
        if (requested.extension() != ".spv") {
            requested += ".spv";
        }
        const auto match = std::find_if(
            available_shaders.begin(), available_shaders.end(),
            [&](const fs::path &shader) {
                return shader.filename() == requested.filename() ||
                       (!library_directory.empty() &&
                        shader.lexically_relative(library_directory) ==
                            requested);
            });
        return match == available_shaders.end() ? fs::path{} : *match;
    }

    std::vector<PlaylistNode>
    load_playlist(const fs::path &playlist_path,
                  const std::vector<fs::path> &available_shaders,
                  const fs::path &library_directory,
                  std::ostream &warning_output) {
        input::validate_file_size(playlist_path, "shader playlist");
        std::ifstream playlist_input(playlist_path);
        if (!playlist_input) {
            throw std::runtime_error("unable to open playlist: " +
                                     playlist_path.string());
        }

        std::vector<PlaylistNode> playlist;
        PlaylistNode *current_node = nullptr;
        std::vector<fs::path> default_entries;
        std::string line;
        std::size_t line_number = 1;
        std::size_t entry_count = 0;
        while (input::read_bounded_line(playlist_input, line,
                                        "shader playlist", line_number++)) {
            line = trim_text(std::move(line));
            if (line.empty() || line.front() == '#') {
                continue;
            }
            if (line.size() >= 2 && line.front() == '[' &&
                line.back() == ']') {
                if (playlist.size() >= input::MAX_PLAYLIST_NODES) {
                    throw std::runtime_error(
                        "shader playlist contains too many nodes");
                }
                std::string node_name =
                    trim_text(line.substr(1, line.size() - 2));
                input::validate_string(node_name,
                                       input::StringKind::DisplayText,
                                       "shader playlist node");
                playlist.push_back({std::move(node_name), {}});
                current_node = &playlist.back();
                continue;
            }
            if (line.front() == '[' || line.back() == ']') {
                throw std::runtime_error(
                    "malformed shader playlist node at line " +
                    std::to_string(line_number - 1));
            }
            if (++entry_count > input::MAX_PLAYLIST_ENTRIES) {
                throw std::runtime_error(
                    "shader playlist contains too many entries");
            }
            input::validate_string(line, input::StringKind::Path,
                                   "shader playlist entry");

            const fs::path shader = find_shader_path(
                available_shaders, library_directory, line);
            if (shader.empty()) {
                warning_output << "acmxvk: playlist shader not found: " << line
                               << '\n';
                continue;
            }
            if (current_node != nullptr) {
                current_node->shaders.push_back(shader);
            } else {
                default_entries.push_back(shader);
            }
        }

        playlist.erase(std::remove_if(playlist.begin(), playlist.end(),
                                      [](const PlaylistNode &node) {
                                          return node.shaders.empty();
                                      }),
                       playlist.end());
        if (!default_entries.empty()) {
            playlist.insert(playlist.begin(),
                            {"Default", std::move(default_entries)});
        }
        if (playlist.empty()) {
            throw std::runtime_error(
                "playlist contains no shaders available in the SPIR-V library");
        }
        return playlist;
    }

    std::size_t
    playlist_shader_count(const std::vector<PlaylistNode> &playlist) noexcept {
        std::size_t count = 0;
        for (const PlaylistNode &node : playlist) {
            count += node.shaders.size();
        }
        return count;
    }
} // namespace acmxvk
