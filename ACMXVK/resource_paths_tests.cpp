#include "app/resource_paths.hpp"

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
                   ("acmxvk-resource-paths-" + std::to_string(suffix));
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

    void create_file(const fs::path &path) {
        fs::create_directories(path.parent_path());
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not create test file: " +
                                     path.string());
        }
        output << "test";
    }

    void expect_equal(const fs::path &actual, const fs::path &expected,
                      const std::string &label) {
        if (actual != expected) {
            throw std::runtime_error(label + ": expected " +
                                     expected.string() + ", received " +
                                     actual.string());
        }
    }
} // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        const fs::path flip = temporary.path / "shaders/flip.frag.spv";
        const fs::path crossfade =
            temporary.path / "shaders/xfade/xfade_01_linear.frag.spv";
        const fs::path model = temporary.path / "models/cube.obj";
        const fs::path font = temporary.path / "data/font.ttf";
        create_file(flip);
        create_file(crossfade);
        create_file(model);
        create_file(font);

        acmxvk::Options options;
        options.resource_directory = temporary.path.string();
        expect_equal(acmxvk::flip_shader_path(options), flip,
                     "user resource shader");
        expect_equal(acmxvk::crossfade_shader_path(options, 0), crossfade,
                     "user resource crossfade");
        expect_equal(acmxvk::default_model_path(options), model,
                     "user resource model");
        expect_equal(acmxvk::overlay_font_path(options), font,
                     "user resource font");

        if (!acmxvk::find_resource(options, "../outside").empty()) {
            throw std::runtime_error("parent traversal was accepted");
        }
        if (!acmxvk::find_resource(options, flip).empty()) {
            throw std::runtime_error("absolute resource path was accepted");
        }

        bool rejected_index = false;
        try {
            static_cast<void>(acmxvk::crossfade_shader_path(
                options, acmxvk::CROSSFADE_NAMES.size()));
        } catch (const std::out_of_range &) {
            rejected_index = true;
        }
        if (!rejected_index) {
            throw std::runtime_error("invalid crossfade index was accepted");
        }
    } catch (const std::exception &error) {
        std::cerr << "resource path test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
