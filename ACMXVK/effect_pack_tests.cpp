#include "app/effect_pack.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

    namespace fs = std::filesystem;

    class TemporaryDirectory {
      public:
        TemporaryDirectory() {
            const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
            path = fs::temp_directory_path() / ("acmxvk-effect-pack-" + std::to_string(suffix));
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

    void write_text(const fs::path &path, const std::string &text) {
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not create effect-pack test file");
        }
        output << text;
    }

    void expect_rejected(const fs::path &path, const std::string &field) {
        try {
            static_cast<void>(acmxvk::load_effect_pack(path));
        } catch (const std::exception &error) {
            expect(std::string(error.what()).find(field) != std::string::npos, "error did not identify " + field + ": " + error.what());
            return;
        }
        throw std::runtime_error("invalid effect pack was accepted: " + path.string());
    }

} // namespace

int main() {
    try {
        const fs::path fixture_root = fs::path(ACMXVK_EFFECT_PACK_FIXTURE_DIRECTORY);
        const acmxvk::EffectPack minimal = acmxvk::load_effect_pack(fixture_root / "minimal/effect.json");
        expect(minimal.version == 1, "minimal fixture version is incorrect");
        expect(minimal.id == "org.acmxvk.example.minimum", "minimal fixture ID is incorrect");
        expect(minimal.passes.size() == 1 && minimal.passes.front().filename() == "passthrough.frag", "minimal fixture pass is incorrect");
        expect(minimal.controls.empty(), "minimal fixture unexpectedly contains controls");

        const acmxvk::EffectPack complete = acmxvk::load_effect_pack(fixture_root / "complete/effect.json");
        expect(complete.icon.has_value(), "complete fixture icon was not parsed");
        expect(complete.passes.size() == 3, "complete fixture pass count is incorrect");
        expect(complete.requirements.history && complete.requirements.original_frame, "complete fixture requirements are incorrect");
        expect(complete.controls.size() == 2 && complete.controls[0].default_value == 6.0, "complete fixture controls are incorrect");
        expect(complete.audio_mappings.size() == 1, "complete fixture audio mapping is missing");
        expect(complete.midi_mappings.size() == 1 && complete.midi_mappings[0].controller == 74, "complete fixture MIDI mapping is incorrect");
        expect(complete.deep_dream.has_value() && complete.deep_dream->layer == "relu4_2", "complete fixture Deep Dream settings are incorrect");

        expect_rejected(fixture_root / "invalid/malformed.json", "root");
        expect_rejected(fixture_root / "invalid/traversal.json", "passes[0]");
        expect_rejected(fixture_root / "invalid/absolute.json", "icon");
        expect_rejected(fixture_root / "invalid/windows-drive.json", "passes[0]");
        expect_rejected(fixture_root / "invalid/duplicate-control.json", "controls[1].id");
        expect_rejected(fixture_root / "invalid/unknown-stable-diffusion.json", "stable_diffusion");

        TemporaryDirectory temporary;
        const fs::path manifest = temporary.path / "effect.json";
        write_text(manifest, R"({"format":"acmxvk-effect-pack","version":1,"id":"test.range","name":"Range","passes":["test.frag"],"controls":[{"id":"amount","label":"Amount","uniform":"amount","minimum":1,"maximum":0,"step":0.1,"default":0.5}]})");
        expect_rejected(manifest, "controls[0].maximum");

        std::string excessive = R"({"format":"acmxvk-effect-pack","version":1,"id":"test.passes","name":"Passes","passes":[)";
        for (std::size_t index = 0; index <= acmxvk::MAX_EFFECT_PACK_PASSES; ++index) {
            if (index != 0) {
                excessive += ',';
            }
            excessive += "\"shader-" + std::to_string(index) + ".frag\"";
        }
        excessive += "]}";
        write_text(manifest, excessive);
        expect_rejected(manifest, "passes");
    } catch (const std::exception &error) {
        std::cerr << "effect-pack test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
