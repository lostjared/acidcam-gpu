#include "effect_pack.hpp"

#include "../input_validation.hpp"

#include <json/json.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace acmxvk {
    namespace {

        namespace fs = std::filesystem;

        constexpr std::string_view FORMAT_NAME = "acmxvk-effect-pack";

        [[noreturn]] void fail(std::string_view field, std::string_view message) { throw std::runtime_error("effect.json " + std::string(field) + " " + std::string(message)); }

        void require_object(const Json::Value &value, std::string_view field) {
            if (!value.isObject()) {
                fail(field, "must be an object");
            }
        }

        void reject_unknown_fields(const Json::Value &value, std::string_view field, std::initializer_list<std::string_view> allowed) {
            std::set<std::string_view> names(allowed);
            for (const std::string &name : value.getMemberNames()) {
                if (!names.contains(name)) {
                    fail(std::string(field) + "." + name, "is not supported");
                }
            }
        }

        const Json::Value &required(const Json::Value &object, std::string_view name, std::string_view field) {
            const std::string key(name);
            if (!object.isMember(key)) {
                fail(std::string(field) + "." + key, "is required");
            }
            return object[key];
        }

        std::string read_string(const Json::Value &value, input::StringKind kind, std::string_view field, bool allow_empty = false) {
            if (!value.isString()) {
                fail(field, "must be a string");
            }
            const std::string text = value.asString();
            try {
                input::validate_string(text, kind, field, allow_empty);
            } catch (const std::exception &error) {
                fail(field, error.what());
            }
            return text;
        }

        bool read_bool(const Json::Value &value, std::string_view field) {
            if (!value.isBool()) {
                fail(field, "must be a boolean");
            }
            return value.asBool();
        }

        int read_integer(const Json::Value &value, std::string_view field, int minimum, int maximum) {
            if (!value.isInt()) {
                fail(field, "must be an integer");
            }
            const int result = value.asInt();
            if (result < minimum || result > maximum) {
                fail(field, "must be between " + std::to_string(minimum) + " and " + std::to_string(maximum));
            }
            return result;
        }

        double read_number(const Json::Value &value, std::string_view field) {
            if (!value.isNumeric()) {
                fail(field, "must be a number");
            }
            const double result = value.asDouble();
            if (!std::isfinite(result)) {
                fail(field, "must be finite");
            }
            return result;
        }

        double read_number_in_range(const Json::Value &value, std::string_view field, double minimum, double maximum) {
            const double result = read_number(value, field);
            if (result < minimum || result > maximum) {
                fail(field, "is outside the supported range");
            }
            return result;
        }

        fs::path read_relative_path(const Json::Value &value, std::string_view field) {
            std::string text = read_string(value, input::StringKind::Path, field);
            if (text.find('\\') != std::string::npos) {
                fail(field, "must use portable forward slashes");
            }
            if (text.find_first_of("<>:\"|?*") != std::string::npos) {
                fail(field, "contains a character that is not portable across supported platforms");
            }
            std::size_t component_start = 0;
            while (component_start <= text.size()) {
                const std::size_t component_end = text.find('/', component_start);
                const std::string component = text.substr(component_start, component_end - component_start);
                if (component.empty()) {
                    fail(field, "contains an empty path component");
                }
                if (component.back() == '.' || component.back() == ' ') {
                    fail(field, "contains a path component ending in a dot or space");
                }
                std::string device_name = component.substr(0, component.find('.'));
                std::transform(device_name.begin(), device_name.end(), device_name.begin(), [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
                const bool numbered_device = device_name.size() == 4 && (device_name.starts_with("COM") || device_name.starts_with("LPT")) && device_name[3] >= '1' && device_name[3] <= '9';
                if (device_name == "CON" || device_name == "PRN" || device_name == "AUX" || device_name == "NUL" || numbered_device) {
                    fail(field, "contains a reserved platform path component");
                }
                if (component_end == std::string::npos) {
                    break;
                }
                component_start = component_end + 1;
            }
            const std::u8string utf8_path(reinterpret_cast<const char8_t *>(text.data()), reinterpret_cast<const char8_t *>(text.data() + text.size()));
            const fs::path path(utf8_path);
            if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
                fail(field, "must be relative to the effect-pack directory");
            }
            for (const fs::path &component : path) {
                if (component == "..") {
                    fail(field, "must not contain parent traversal");
                }
            }
            const fs::path normalized = path.lexically_normal();
            if (normalized.empty() || normalized == ".") {
                fail(field, "must name a file");
            }
            return normalized;
        }

        void validate_range(double minimum, double maximum, double step, double default_value, std::string_view field) {
            if (!(minimum < maximum)) {
                fail(std::string(field) + ".maximum", "must be greater than minimum");
            }
            if (!(step > 0.0) || step > maximum - minimum) {
                fail(std::string(field) + ".step", "must be positive and no larger than the range");
            }
            if (default_value < minimum || default_value > maximum) {
                fail(std::string(field) + ".default", "must be within the control range");
            }
        }

        EffectPackRequirements parse_requirements(const Json::Value &value) {
            constexpr std::string_view FIELD = "requires";
            require_object(value, FIELD);
            reject_unknown_fields(value, FIELD, {"history", "spectrum", "spectrum_history", "original_frame"});
            EffectPackRequirements requirements;
            if (value.isMember("history")) {
                requirements.history = read_bool(value["history"], "requires.history");
            }
            if (value.isMember("spectrum")) {
                requirements.spectrum = read_bool(value["spectrum"], "requires.spectrum");
            }
            if (value.isMember("spectrum_history")) {
                requirements.spectrum_history = read_bool(value["spectrum_history"], "requires.spectrum_history");
            }
            if (value.isMember("original_frame")) {
                requirements.original_frame = read_bool(value["original_frame"], "requires.original_frame");
            }
            return requirements;
        }

        std::vector<EffectPackControl> parse_controls(const Json::Value &value) {
            if (!value.isArray()) {
                fail("controls", "must be an array");
            }
            if (value.size() > MAX_EFFECT_PACK_CONTROLS) {
                fail("controls", "contains more than 64 entries");
            }
            std::vector<EffectPackControl> controls;
            std::set<std::string> ids;
            std::set<std::string> uniforms;
            controls.reserve(value.size());
            for (Json::ArrayIndex index = 0; index < value.size(); ++index) {
                const Json::Value &item = value[index];
                const std::string field = "controls[" + std::to_string(index) + "]";
                require_object(item, field);
                reject_unknown_fields(item, field, {"id", "label", "uniform", "minimum", "maximum", "step", "default"});
                EffectPackControl control;
                control.id = read_string(required(item, "id", field), input::StringKind::Token, field + ".id");
                control.label = read_string(required(item, "label", field), input::StringKind::DisplayText, field + ".label");
                control.uniform = read_string(required(item, "uniform", field), input::StringKind::Identifier, field + ".uniform");
                control.minimum = read_number(required(item, "minimum", field), field + ".minimum");
                control.maximum = read_number(required(item, "maximum", field), field + ".maximum");
                control.step = read_number(required(item, "step", field), field + ".step");
                control.default_value = read_number(required(item, "default", field), field + ".default");
                validate_range(control.minimum, control.maximum, control.step, control.default_value, field);
                if (!ids.insert(control.id).second) {
                    fail(field + ".id", "duplicates another control ID");
                }
                if (!uniforms.insert(control.uniform).second) {
                    fail(field + ".uniform", "duplicates another control uniform");
                }
                controls.push_back(std::move(control));
            }
            return controls;
        }

        std::vector<EffectPackAudioMapping> parse_audio_mappings(const Json::Value &value) {
            if (!value.isArray()) {
                fail("audio_mappings", "must be an array");
            }
            if (value.size() > MAX_EFFECT_PACK_AUDIO_MAPPINGS) {
                fail("audio_mappings", "contains more than 64 entries");
            }
            std::vector<EffectPackAudioMapping> mappings;
            mappings.reserve(value.size());
            for (Json::ArrayIndex index = 0; index < value.size(); ++index) {
                const Json::Value &item = value[index];
                const std::string field = "audio_mappings[" + std::to_string(index) + "]";
                require_object(item, field);
                reject_unknown_fields(item, field, {"source", "uniform", "minimum", "maximum"});
                EffectPackAudioMapping mapping;
                mapping.source = read_string(required(item, "source", field), input::StringKind::Token, field + ".source");
                mapping.uniform = read_string(required(item, "uniform", field), input::StringKind::Identifier, field + ".uniform");
                mapping.minimum = read_number(required(item, "minimum", field), field + ".minimum");
                mapping.maximum = read_number(required(item, "maximum", field), field + ".maximum");
                if (!(mapping.minimum < mapping.maximum)) {
                    fail(field + ".maximum", "must be greater than minimum");
                }
                mappings.push_back(std::move(mapping));
            }
            return mappings;
        }

        std::vector<EffectPackMidiMapping> parse_midi_mappings(const Json::Value &value) {
            if (!value.isArray()) {
                fail("midi_mappings", "must be an array");
            }
            if (value.size() > MAX_EFFECT_PACK_MIDI_MAPPINGS) {
                fail("midi_mappings", "contains more than 64 entries");
            }
            std::vector<EffectPackMidiMapping> mappings;
            mappings.reserve(value.size());
            for (Json::ArrayIndex index = 0; index < value.size(); ++index) {
                const Json::Value &item = value[index];
                const std::string field = "midi_mappings[" + std::to_string(index) + "]";
                require_object(item, field);
                reject_unknown_fields(item, field, {"uniform", "channel", "controller", "minimum", "maximum"});
                EffectPackMidiMapping mapping;
                mapping.uniform = read_string(required(item, "uniform", field), input::StringKind::Identifier, field + ".uniform");
                mapping.channel = read_integer(required(item, "channel", field), field + ".channel", 1, 16);
                mapping.controller = read_integer(required(item, "controller", field), field + ".controller", 0, 127);
                mapping.minimum = read_number(required(item, "minimum", field), field + ".minimum");
                mapping.maximum = read_number(required(item, "maximum", field), field + ".maximum");
                if (!(mapping.minimum < mapping.maximum)) {
                    fail(field + ".maximum", "must be greater than minimum");
                }
                mappings.push_back(std::move(mapping));
            }
            return mappings;
        }

        EffectPackDeepDream parse_deep_dream(const Json::Value &value) {
            constexpr std::string_view FIELD = "deep_dream";
            require_object(value, FIELD);
            reject_unknown_fields(value, FIELD, {"enabled", "model", "layer", "channel", "iterations", "strength", "feedback", "zoom", "rotation", "working_size", "fp16", "octaves", "octave_scale", "jitter", "smoothing", "gpu_filter_before_dream"});
            EffectPackDeepDream dream;
            if (value.isMember("enabled")) {
                dream.enabled = read_bool(value["enabled"], "deep_dream.enabled");
            }
            if (value.isMember("model")) {
                dream.model = read_string(value["model"], input::StringKind::Token, "deep_dream.model");
            }
            if (value.isMember("layer")) {
                dream.layer = read_string(value["layer"], input::StringKind::Token, "deep_dream.layer");
            }
            if (value.isMember("channel")) {
                dream.channel = read_integer(value["channel"], "deep_dream.channel", -1, 65535);
            }
            if (value.isMember("iterations")) {
                dream.iterations = read_integer(value["iterations"], "deep_dream.iterations", 1, 100);
            }
            if (value.isMember("strength")) {
                dream.strength = read_number_in_range(value["strength"], "deep_dream.strength", 0.000001, 10.0);
            }
            if (value.isMember("feedback")) {
                dream.feedback = read_number_in_range(value["feedback"], "deep_dream.feedback", 0.0, 0.99);
            }
            if (value.isMember("zoom")) {
                dream.zoom = read_number_in_range(value["zoom"], "deep_dream.zoom", 0.9, 1.1);
            }
            if (value.isMember("rotation")) {
                dream.rotation = read_number_in_range(value["rotation"], "deep_dream.rotation", -5.0, 5.0);
            }
            if (value.isMember("working_size")) {
                dream.working_size = read_integer(value["working_size"], "deep_dream.working_size", 0, 4096);
                if (dream.working_size != 0 && dream.working_size < 64) {
                    fail("deep_dream.working_size", "must be 0 or between 64 and 4096");
                }
            }
            if (value.isMember("fp16")) {
                dream.fp16 = read_bool(value["fp16"], "deep_dream.fp16");
            }
            if (value.isMember("octaves")) {
                dream.octaves = read_integer(value["octaves"], "deep_dream.octaves", 1, 8);
            }
            if (value.isMember("octave_scale")) {
                dream.octave_scale = read_number_in_range(value["octave_scale"], "deep_dream.octave_scale", 1.1, 3.0);
            }
            if (value.isMember("jitter")) {
                dream.jitter = read_integer(value["jitter"], "deep_dream.jitter", 0, 64);
            }
            if (value.isMember("smoothing")) {
                dream.smoothing = read_integer(value["smoothing"], "deep_dream.smoothing", 0, 16);
            }
            if (value.isMember("gpu_filter_before_dream")) {
                dream.gpu_filter_before_dream = read_bool(value["gpu_filter_before_dream"], "deep_dream.gpu_filter_before_dream");
            }
            if (dream.enabled && (dream.model.empty() || dream.layer.empty())) {
                fail(FIELD, "requires model and layer when enabled");
            }
            return dream;
        }

    } // namespace

    EffectPack load_effect_pack(const fs::path &manifest_path) {
        input::validate_text_file(manifest_path, "effect-pack manifest");
        std::ifstream input_file(manifest_path, std::ios::binary);
        std::ostringstream buffer;
        buffer << input_file.rdbuf();
        const std::string source = buffer.str();

        Json::CharReaderBuilder builder;
        builder["allowComments"] = false;
        builder["allowTrailingCommas"] = false;
        builder["rejectDupKeys"] = true;
        builder["strictRoot"] = true;
        Json::Value root;
        std::string errors;
        const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (!reader->parse(source.data(), source.data() + source.size(), &root, &errors)) {
            fail("root", "contains invalid JSON: " + errors);
        }
        require_object(root, "root");
        reject_unknown_fields(root, "root", {"format", "version", "id", "name", "description", "icon", "passes", "requires", "controls", "audio_mappings", "midi_mappings", "deep_dream"});

        const std::string format = read_string(required(root, "format", "root"), input::StringKind::Token, "format");
        if (format != FORMAT_NAME) {
            fail("format", "must be acmxvk-effect-pack");
        }

        EffectPack pack;
        pack.version = read_integer(required(root, "version", "root"), "version", EFFECT_PACK_FORMAT_VERSION, EFFECT_PACK_FORMAT_VERSION);
        pack.id = read_string(required(root, "id", "root"), input::StringKind::Token, "id");
        pack.name = read_string(required(root, "name", "root"), input::StringKind::DisplayText, "name");
        if (root.isMember("description")) {
            pack.description = read_string(root["description"], input::StringKind::DisplayText, "description", true);
        }
        pack.manifest = fs::absolute(manifest_path).lexically_normal();
        pack.root = pack.manifest.parent_path();
        if (root.isMember("icon")) {
            pack.icon = pack.root / read_relative_path(root["icon"], "icon");
        }

        const Json::Value &passes = required(root, "passes", "root");
        if (!passes.isArray() || passes.empty()) {
            fail("passes", "must be a non-empty array");
        }
        if (passes.size() > MAX_EFFECT_PACK_PASSES) {
            fail("passes", "contains more than 64 entries");
        }
        for (Json::ArrayIndex index = 0; index < passes.size(); ++index) {
            const std::string field = "passes[" + std::to_string(index) + "]";
            const fs::path relative = read_relative_path(passes[index], field);
            pack.passes.push_back(pack.root / relative);
        }
        if (root.isMember("requires")) {
            pack.requirements = parse_requirements(root["requires"]);
        }
        if (root.isMember("controls")) {
            pack.controls = parse_controls(root["controls"]);
        }
        if (root.isMember("audio_mappings")) {
            pack.audio_mappings = parse_audio_mappings(root["audio_mappings"]);
        }
        if (root.isMember("midi_mappings")) {
            pack.midi_mappings = parse_midi_mappings(root["midi_mappings"]);
        }
        if (root.isMember("deep_dream")) {
            pack.deep_dream = parse_deep_dream(root["deep_dream"]);
        }
        return pack;
    }

    EffectPackCatalog discover_effect_packs(const std::vector<fs::path> &roots) {
        EffectPackCatalog catalog;
        std::vector<fs::path> manifests;
        std::unordered_set<std::string> seen_manifests;
        for (const fs::path &requested_root : roots) {
            std::error_code error;
            const fs::path root = fs::weakly_canonical(requested_root, error);
            if (error || !fs::is_directory(root) || fs::is_symlink(root)) {
                catalog.diagnostics.push_back({EffectPackDiagnosticSeverity::Warning, requested_root, "effect-pack search root is unavailable"});
                continue;
            }
            if (fs::is_regular_file(root / "effect.json")) {
                const std::string key = (root / "effect.json").generic_string();
                if (seen_manifests.insert(key).second) {
                    manifests.push_back(root / "effect.json");
                }
            }
            for (fs::recursive_directory_iterator iterator(root, fs::directory_options::skip_permission_denied, error), end; iterator != end; iterator.increment(error)) {
                if (error) {
                    error.clear();
                    continue;
                }
                if (iterator.depth() >= 8) {
                    iterator.disable_recursion_pending();
                }
                const fs::directory_entry &entry = *iterator;
                if (entry.is_symlink(error)) {
                    if (entry.is_directory(error)) {
                        iterator.disable_recursion_pending();
                    }
                    error.clear();
                    continue;
                }
                if (entry.is_directory(error) && (entry.path().filename() == ".acmxvk-build" || entry.path().filename() == ".editor-preview")) {
                    iterator.disable_recursion_pending();
                    continue;
                }
                if (!entry.is_regular_file(error) || entry.path().filename() != "effect.json") {
                    error.clear();
                    continue;
                }
                const fs::path manifest = fs::weakly_canonical(entry.path(), error);
                if (!error && seen_manifests.insert(manifest.generic_string()).second) {
                    manifests.push_back(manifest);
                }
                error.clear();
                if (manifests.size() > MAX_DISCOVERED_EFFECT_PACKS) {
                    catalog.diagnostics.push_back({EffectPackDiagnosticSeverity::Error, root, "effect-pack search exceeded the 4096-pack limit"});
                    manifests.resize(MAX_DISCOVERED_EFFECT_PACKS);
                    break;
                }
            }
        }

        std::sort(manifests.begin(), manifests.end());
        std::unordered_map<std::string, fs::path> pack_ids;
        for (const fs::path &manifest : manifests) {
            try {
                EffectPack pack = load_effect_pack(manifest);
                const auto [existing, inserted] = pack_ids.emplace(pack.id, pack.manifest);
                if (!inserted) {
                    catalog.diagnostics.push_back({EffectPackDiagnosticSeverity::Warning, pack.manifest, "duplicate effect-pack ID '" + pack.id + "'; first declared by " + existing->second.string()});
                    continue;
                }
                if (pack.icon.has_value() && !fs::is_regular_file(*pack.icon)) {
                    catalog.diagnostics.push_back({EffectPackDiagnosticSeverity::Warning, *pack.icon, "effect-pack icon is missing"});
                }
                catalog.packs.push_back(std::move(pack));
            } catch (const std::exception &error) {
                catalog.diagnostics.push_back({EffectPackDiagnosticSeverity::Error, manifest, error.what()});
            }
        }
        std::sort(catalog.packs.begin(), catalog.packs.end(), [](const EffectPack &left, const EffectPack &right) { return left.id < right.id; });
        return catalog;
    }

} // namespace acmxvk
