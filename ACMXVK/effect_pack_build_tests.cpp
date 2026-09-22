#include "app/effect_pack.hpp"
#include "app/effect_pack_build.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

    namespace fs = std::filesystem;

    class TemporaryDirectory {
      public:
        TemporaryDirectory() {
            const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
            path = fs::temp_directory_path() / ("acmxvk-effect-pack-build-" + std::to_string(suffix));
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

    void copy_directory(const fs::path &source, const fs::path &destination) {
        fs::create_directories(destination);
        fs::copy(source, destination, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    }

    void write_text(const fs::path &path, const std::string &text) {
        fs::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("unable to write effect-pack test file");
        }
        output << text;
    }

} // namespace

int main() {
    try {
        const fs::path fixtures = fs::path(ACMXVK_EFFECT_PACK_FIXTURE_DIRECTORY);
        TemporaryDirectory temporary;
        const fs::path pack_root = temporary.path / "build-three";
        copy_directory(fixtures / "build-three", pack_root);

        const acmxvk::EffectPack pack = acmxvk::load_effect_pack(pack_root / "effect.json");
        const fs::path build_root = pack_root / ".acmxvk-build";
        write_text(build_root / "old.frag.spv.acmxvk-tmp-1", "temporary");
        write_text(build_root / ".editor-preview/preview.frag.spv", "temporary");

        acmxvk::EffectPackBuildOptions options;
        options.glslc_executable = ACMXVK_TEST_GLSLC_EXECUTABLE;
        options.parallel_jobs = 2;
        const acmxvk::EffectPackBuildResult first = acmxvk::build_effect_pack(pack, options);
        expect(first.compiled == 3 && first.current == 0, "initial effect-pack build did not compile all three passes");
        expect(first.removed_temporary_files == 2, "stale effect-pack temporary files were not removed");
        expect(fs::is_regular_file(first.build_root / "effect-cache.json"), "effect-pack cache manifest was not written");

        const acmxvk::EffectPackBuildResult second = acmxvk::build_effect_pack(pack, options);
        expect(second.compiled == 0 && second.current == 3, "unchanged effect-pack build was not incremental");
        const acmxvk::EffectPackBuildResult loaded = acmxvk::load_effect_pack_cache(pack);
        expect(loaded.current == 3 && loaded.compiled_passes == second.compiled_passes, "validated effect-pack cache did not preserve pass order");

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        write_text(pack_root / "common.glsl",
                   "vec3 pack_tint(vec3 color) {\n"
                   "    return color * vec3(1.0, 0.9, 0.95);\n"
                   "}\n");
        bool rejected_stale_cache = false;
        try {
            static_cast<void>(acmxvk::load_effect_pack_cache(pack));
        } catch (const std::exception &error) {
            rejected_stale_cache = std::string(error.what()).find("missing or stale") != std::string::npos;
        }
        expect(rejected_stale_cache, "stale effect-pack cache was accepted for activation");
        const acmxvk::EffectPackBuildResult include_update = acmxvk::build_effect_pack(pack, options);
        expect(include_update.compiled == 1 && include_update.current == 2, "changing an include did not rebuild exactly its dependent pass");

        const fs::path repeated_root = temporary.path / "repeated-pass";
        copy_directory(fixtures / "build-three", repeated_root);
        write_text(repeated_root / "effect.json", R"({"format":"acmxvk-effect-pack","version":1,"id":"test.repeated","name":"Repeated","passes":["shaders/first.frag","shaders/first.frag","shaders/warp.comp","shaders/final.frag"],"requires":{"history":true,"original_frame":true}})");
        const acmxvk::EffectPack repeated = acmxvk::load_effect_pack(repeated_root / "effect.json");
        std::atomic<std::size_t> reported_total{0};
        options.progress = [&reported_total](std::size_t, std::size_t total) { reported_total = total; };
        const acmxvk::EffectPackBuildResult repeated_build = acmxvk::build_effect_pack(repeated, options);
        expect(repeated_build.compiled_passes.size() == 4 && repeated_build.compiled_passes[0] == repeated_build.compiled_passes[1], "repeated shader passes did not preserve their order");
        expect(reported_total == 3, "repeated shader pass was compiled more than once");
        options.progress = {};

        acmxvk::EffectPack invalid_requirements = pack;
        invalid_requirements.requirements.history = false;
        bool rejected_requirements = false;
        try {
            static_cast<void>(acmxvk::build_effect_pack(invalid_requirements, options));
        } catch (const std::exception &error) {
            rejected_requirements = std::string(error.what()).find("requires.history") != std::string::npos;
        }
        expect(rejected_requirements, "undeclared shader history requirement was accepted");

        const acmxvk::EffectPackCatalog catalog = acmxvk::discover_effect_packs({fixtures});
        expect(catalog.packs.size() == 3, "effect-pack discovery did not retain the three valid fixtures");
        expect(!catalog.diagnostics.empty(), "effect-pack discovery did not report invalid fixtures or the missing icon");

        const fs::path duplicates = temporary.path / "duplicates";
        const std::string duplicate_manifest = R"({"format":"acmxvk-effect-pack","version":1,"id":"test.duplicate","name":"Duplicate","passes":["shader.frag"]})";
        write_text(duplicates / "one/effect.json", duplicate_manifest);
        write_text(duplicates / "two/effect.json", duplicate_manifest);
        const acmxvk::EffectPackCatalog duplicate_catalog = acmxvk::discover_effect_packs({duplicates});
        expect(duplicate_catalog.packs.size() == 1, "duplicate effect-pack IDs were not collapsed");
        expect(duplicate_catalog.diagnostics.size() == 1 && duplicate_catalog.diagnostics.front().message.find("duplicate effect-pack ID") != std::string::npos, "duplicate effect-pack ID diagnostic is missing");
    } catch (const std::exception &error) {
        std::cerr << "effect-pack build test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
