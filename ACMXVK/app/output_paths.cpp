#include "output_paths.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace acmxvk {
    fs::path output_frame_directory(const std::string &filename,
                                    std::string_view suffix) {
        const fs::path output_path(filename);
        const fs::path parent = output_path.has_parent_path()
                                    ? output_path.parent_path()
                                    : fs::path(".");
        const std::string name = output_path.filename().empty()
                                     ? std::string("output")
                                     : output_path.filename().string();
        return parent /
               ("video_file-" + name + "-" + std::string(suffix));
    }

    void create_output_directory(const fs::path &directory) {
        std::error_code error;
        fs::create_directories(directory, error);
        if (error || !fs::is_directory(directory)) {
            throw std::runtime_error(
                "unable to create PNG output directory: " +
                directory.string());
        }
    }

    fs::path frame_path(const fs::path &directory, std::uint64_t index) {
        std::ostringstream filename;
        filename << "frame-" << std::setfill('0') << std::setw(8) << index
                 << ".png";
        return directory / filename.str();
    }

    std::string_view snapshot_extension(SnapshotFormat format) noexcept {
        switch (format) {
        case SnapshotFormat::WebP:
            return ".webp";
        case SnapshotFormat::Tiff:
            return ".tiff";
        case SnapshotFormat::Raw:
            return ".raw";
        case SnapshotFormat::Png:
            return ".png";
        }
        return ".snapshot";
    }

    fs::path snapshot_path(
        const fs::path &directory, std::uint32_t width, std::uint32_t height,
        std::uint64_t &counter, SnapshotFormat format,
        std::chrono::system_clock::time_point timestamp) {
        const std::time_t timestamp_time =
            std::chrono::system_clock::to_time_t(timestamp);
        std::tm local_time{};
#ifdef _WIN32
        localtime_s(&local_time, &timestamp_time);
#else
        localtime_r(&timestamp_time, &local_time);
#endif
        while (true) {
            std::ostringstream filename;
            filename << "ACMXVK.Snapshot-"
                     << std::put_time(&local_time, "%Y.%m.%d-%H.%M.%S") << '-'
                     << width << 'x' << height << '-' << counter
                     << snapshot_extension(format);
            const fs::path candidate = directory / filename.str();
            if (!fs::exists(candidate)) {
                return candidate;
            }
            ++counter;
        }
    }
} // namespace acmxvk
