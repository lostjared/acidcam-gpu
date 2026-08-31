#include "app/output_paths.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {
    namespace fs = std::filesystem;

    class TemporaryDirectory {
      public:
        TemporaryDirectory() {
            const auto suffix = std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count();
            path = fs::temp_directory_path() /
                   ("acmxvk-output-paths-" + std::to_string(suffix));
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code error;
            fs::remove_all(path, error);
        }

        TemporaryDirectory(const TemporaryDirectory &) = delete;
        TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

        fs::path path;
    };

    void expect(bool condition, const std::string &message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void expect_suffix(const fs::path &path, std::string_view suffix,
                       const std::string &message) {
        expect(path.filename().string().ends_with(suffix), message);
    }
} // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        expect(acmxvk::output_frame_directory("movie.mp4", "png") ==
                   fs::path(".") / "video_file-movie.mp4-png",
               "relative output directory is incorrect");
        expect(acmxvk::output_frame_directory(
                   (temporary.path / "movie.mp4").string(), "generate") ==
                   temporary.path / "video_file-movie.mp4-generate",
               "parent output directory was not retained");

        const fs::path output_directory = temporary.path / "nested/frames";
        acmxvk::create_output_directory(output_directory);
        expect(fs::is_directory(output_directory),
               "output directory was not created");
        expect(acmxvk::frame_path(output_directory, 42) ==
                   output_directory / "frame-00000042.png",
               "numbered PNG frame path is incorrect");

        expect(acmxvk::snapshot_extension(acmxvk::SnapshotFormat::Png) ==
                   ".png",
               "PNG snapshot extension is incorrect");
        expect(acmxvk::snapshot_extension(acmxvk::SnapshotFormat::WebP) ==
                   ".webp",
               "WebP snapshot extension is incorrect");
        expect(acmxvk::snapshot_extension(acmxvk::SnapshotFormat::Tiff) ==
                   ".tiff",
               "TIFF snapshot extension is incorrect");
        expect(acmxvk::snapshot_extension(acmxvk::SnapshotFormat::Raw) ==
                   ".raw",
               "raw snapshot extension is incorrect");

        std::uint64_t counter = 7;
        const auto timestamp = std::chrono::system_clock::from_time_t(0);
        const fs::path first = acmxvk::snapshot_path(
            temporary.path, 1920, 1080, counter,
            acmxvk::SnapshotFormat::Png, timestamp);
        expect_suffix(first, "-1920x1080-7.png",
                      "snapshot dimensions or counter are incorrect");
        std::ofstream(first) << "collision";
        const fs::path second = acmxvk::snapshot_path(
            temporary.path, 1920, 1080, counter,
            acmxvk::SnapshotFormat::Png, timestamp);
        expect(counter == 8, "snapshot collision did not advance the counter");
        expect_suffix(second, "-1920x1080-8.png",
                      "collision-safe snapshot path is incorrect");

        const fs::path regular_file = temporary.path / "not-a-directory";
        std::ofstream(regular_file) << "file";
        bool rejected_file = false;
        try {
            acmxvk::create_output_directory(regular_file);
        } catch (const std::runtime_error &) {
            rejected_file = true;
        }
        expect(rejected_file, "regular file was accepted as an output directory");
    } catch (const std::exception &error) {
        std::cerr << "output path test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
