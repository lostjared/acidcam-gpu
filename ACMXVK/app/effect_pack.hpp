#ifndef ACMXVK_APP_EFFECT_PACK_HPP
#define ACMXVK_APP_EFFECT_PACK_HPP

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace acmxvk {

    constexpr int EFFECT_PACK_FORMAT_VERSION = 1;
    constexpr std::size_t MAX_EFFECT_PACK_PASSES = 64;
    constexpr std::size_t MAX_EFFECT_PACK_CONTROLS = 64;
    constexpr std::size_t MAX_EFFECT_PACK_AUDIO_MAPPINGS = 64;
    constexpr std::size_t MAX_EFFECT_PACK_MIDI_MAPPINGS = 64;
    constexpr std::size_t MAX_DISCOVERED_EFFECT_PACKS = 4096;

    struct EffectPackRequirements {
        bool history = false;
        bool spectrum = false;
        bool spectrum_history = false;
        bool original_frame = false;
    };

    struct EffectPackControl {
        std::string id;
        std::string label;
        std::string uniform;
        double minimum = 0.0;
        double maximum = 1.0;
        double step = 0.01;
        double default_value = 0.0;
    };

    struct EffectPackAudioMapping {
        std::string source;
        std::string uniform;
        double minimum = 0.0;
        double maximum = 1.0;
    };

    struct EffectPackMidiMapping {
        std::string input;
        std::string uniform;
        double minimum = 0.0;
        double maximum = 1.0;
    };

    struct EffectPackDeepDream {
        bool enabled = false;
        std::string model;
        std::string layer;
        int channel = -1;
        int iterations = 1;
        double strength = 0.05;
        double feedback = 0.9;
        double zoom = 1.01;
        double rotation = 0.1;
        int working_size = 512;
        bool fp16 = false;
        int octaves = 1;
        double octave_scale = 1.4;
        int jitter = 0;
        int smoothing = 0;
        bool gpu_filter_before_dream = false;
    };

    struct EffectPack {
        int version = EFFECT_PACK_FORMAT_VERSION;
        std::string id;
        std::string name;
        std::string description;
        std::filesystem::path root;
        std::filesystem::path manifest;
        std::optional<std::filesystem::path> icon;
        std::vector<std::filesystem::path> passes;
        EffectPackRequirements requirements;
        std::vector<EffectPackControl> controls;
        std::vector<EffectPackAudioMapping> audio_mappings;
        std::vector<EffectPackMidiMapping> midi_mappings;
        std::optional<EffectPackDeepDream> deep_dream;
    };

    enum class EffectPackDiagnosticSeverity { Warning, Error };

    struct EffectPackDiagnostic {
        EffectPackDiagnosticSeverity severity = EffectPackDiagnosticSeverity::Warning;
        std::filesystem::path path;
        std::string message;
    };

    struct EffectPackCatalog {
        std::vector<EffectPack> packs;
        std::vector<EffectPackDiagnostic> diagnostics;
    };

    [[nodiscard]] EffectPack load_effect_pack(const std::filesystem::path &manifest_path);
    [[nodiscard]] EffectPackCatalog discover_effect_packs(const std::vector<std::filesystem::path> &roots);

} // namespace acmxvk

#endif
