#include "app/playlist.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
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
                   ("acmxvk-playlist-" + std::to_string(suffix));
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

    void write_text(const fs::path &path, const std::string &text) {
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not create playlist test file");
        }
        output << text;
    }

    void expect(bool condition, const std::string &message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }
} // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        const fs::path library = temporary.path / "library";
        const std::vector<fs::path> shaders{
            library / "orphan.frag.spv", library / "one.frag.spv",
            library / "compute/two.comp.spv"};

        expect(acmxvk::find_shader_path(shaders, library, " one.frag ") ==
                   shaders[1],
               "fragment shader name was not resolved");
        expect(acmxvk::find_shader_path(shaders, library,
                                        "compute/two.comp") == shaders[2],
               "relative compute shader path was not resolved");

        const fs::path playlist_path = temporary.path / "test.playlist.txt";
        write_text(playlist_path,
                   "orphan.frag\n"
                   "[Warm]\n"
                   "one.frag\n"
                   "missing.frag\n"
                   "[Compute]\n"
                   "compute/two.comp\n"
                   "[Empty]\n"
                   "missing-again.frag\n");
        std::ostringstream warnings;
        const std::vector<acmxvk::PlaylistNode> playlist =
            acmxvk::load_playlist(playlist_path, shaders, library, warnings);
        expect(playlist.size() == 3, "empty playlist node was not removed");
        expect(playlist[0].name == "Default" &&
                   playlist[0].shaders ==
                       std::vector<fs::path>{shaders[0]},
               "default playlist entries were not retained");
        expect(playlist[1].name == "Warm" &&
                   playlist[1].shaders ==
                       std::vector<fs::path>{shaders[1]},
               "named fragment node was parsed incorrectly");
        expect(playlist[2].name == "Compute" &&
                   playlist[2].shaders ==
                       std::vector<fs::path>{shaders[2]},
               "named compute node was parsed incorrectly");
        expect(acmxvk::playlist_shader_count(playlist) == 3,
               "playlist shader count is incorrect");
        expect(warnings.str().find("missing.frag") != std::string::npos,
               "missing shader warning was not emitted");

        const fs::path malformed_path = temporary.path / "malformed.txt";
        write_text(malformed_path, "[Broken\none.frag\n");
        bool rejected_malformed = false;
        try {
            static_cast<void>(acmxvk::load_playlist(
                malformed_path, shaders, library, warnings));
        } catch (const std::runtime_error &) {
            rejected_malformed = true;
        }
        expect(rejected_malformed, "malformed playlist node was accepted");
    } catch (const std::exception &error) {
        std::cerr << "playlist test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
