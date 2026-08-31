#ifndef ACMXVK_APP_OUTPUT_PATHS_HPP
#define ACMXVK_APP_OUTPUT_PATHS_HPP

#include "options.hpp"
#include "snapshot_writer.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

namespace acmxvk {
    [[nodiscard]] fs::path
    output_frame_directory(const std::string &filename,
                           std::string_view suffix);
    void create_output_directory(const fs::path &directory);
    [[nodiscard]] fs::path frame_path(const fs::path &directory,
                                      std::uint64_t index);
    [[nodiscard]] std::string_view
    snapshot_extension(SnapshotFormat format) noexcept;
    [[nodiscard]] fs::path snapshot_path(
        const fs::path &directory, std::uint32_t width, std::uint32_t height,
        std::uint64_t &counter, SnapshotFormat format,
        std::chrono::system_clock::time_point timestamp =
            std::chrono::system_clock::now());
} // namespace acmxvk

#endif
