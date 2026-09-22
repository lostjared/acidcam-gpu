#include "effect_pack_build.hpp"

#include "../input_validation.hpp"
#include "shader_compiler.hpp"

#include <mxvk/mxvk.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace acmxvk {
    namespace {

        namespace fs = std::filesystem;

        [[nodiscard]] bool path_is_below(const fs::path &path, const fs::path &root) {
            const std::string relative = path.lexically_relative(root).generic_string();
            return !relative.empty() && relative != ".." && !relative.starts_with("../");
        }

        [[nodiscard]] fs::path canonical_pack_file(const fs::path &path, const fs::path &root, std::string_view context) {
            std::error_code error;
            const fs::path canonical = fs::weakly_canonical(path, error);
            if (error || !fs::is_regular_file(canonical) || !path_is_below(canonical, root)) {
                throw std::runtime_error(std::string(context) + " is unavailable or resolves outside the effect-pack directory: " + path.string());
            }
            return canonical;
        }

        void collect_dependencies(const fs::path &source, const fs::path &root, std::unordered_set<std::string> &visited, std::vector<fs::path> &dependencies, std::size_t depth) {
            if (depth > 32U || dependencies.size() >= 256U) {
                throw std::runtime_error("shader include graph exceeds the supported limits: " + source.string());
            }
            const fs::path canonical = canonical_pack_file(source, root, "shader include");
            if (!visited.insert(canonical.generic_string()).second) {
                return;
            }
            dependencies.push_back(canonical);
            if (canonical.extension() == ".spv") {
                return;
            }

            input::validate_text_file(canonical, "effect-pack shader source");
            std::ifstream input_file(canonical);
            std::string line;
            std::size_t line_number = 1;
            while (input::read_bounded_line(input_file, line, "effect-pack shader source", line_number++)) {
                const std::size_t directive = line.find_first_not_of(" \t");
                if (directive == std::string::npos || line.compare(directive, 8U, "#include") != 0) {
                    continue;
                }
                const std::size_t opening = line.find_first_of("\"<", directive + 8U);
                if (opening == std::string::npos) {
                    throw std::runtime_error("malformed shader include in " + canonical.string() + " line " + std::to_string(line_number - 1U));
                }
                const char closing_character = line[opening] == '"' ? '"' : '>';
                const std::size_t closing = line.find(closing_character, opening + 1U);
                if (closing == std::string::npos || closing == opening + 1U) {
                    throw std::runtime_error("malformed shader include in " + canonical.string() + " line " + std::to_string(line_number - 1U));
                }
                const fs::path include_name(line.substr(opening + 1U, closing - opening - 1U));
                if (include_name.is_absolute()) {
                    throw std::runtime_error("absolute shader include is not allowed: " + include_name.string());
                }
                fs::path include_path = line[opening] == '"' ? canonical.parent_path() / include_name : root / include_name;
                std::error_code error;
                if (!fs::is_regular_file(include_path, error) && line[opening] == '"') {
                    include_path = root / include_name;
                }
                collect_dependencies(include_path, root, visited, dependencies, depth + 1U);
            }
        }

        [[nodiscard]] bool dependencies_newer_than(const fs::path &source, const fs::path &manifest, const fs::path &destination, const fs::path &root) {
            if (!fs::is_regular_file(destination)) {
                return true;
            }
            std::error_code error;
            const fs::file_time_type output_time = fs::last_write_time(destination, error);
            if (error || fs::last_write_time(manifest, error) > output_time || error) {
                return true;
            }
            std::unordered_set<std::string> visited;
            std::vector<fs::path> dependencies;
            collect_dependencies(source, root, visited, dependencies, 0);
            for (const fs::path &dependency : dependencies) {
                if (fs::last_write_time(dependency, error) > output_time || error) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool uses_descriptor_binding(const fs::path &shader, std::uint32_t requested_binding) {
            const std::vector<char> bytes = mxvk::load_spv(shader.string());
            if (bytes.size() < 5U * sizeof(std::uint32_t) || bytes.size() % sizeof(std::uint32_t) != 0U) {
                return false;
            }
            std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
            std::memcpy(words.data(), bytes.data(), bytes.size());
            for (std::size_t offset = 5U; offset < words.size();) {
                const std::uint32_t instruction = words[offset];
                const std::uint16_t word_count = static_cast<std::uint16_t>(instruction >> 16U);
                const std::uint16_t opcode = static_cast<std::uint16_t>(instruction & 0xFFFFU);
                if (word_count == 0U || offset + word_count > words.size()) {
                    return false;
                }
                constexpr std::uint16_t OP_DECORATE = 71U;
                constexpr std::uint32_t DECORATION_BINDING = 33U;
                if (opcode == OP_DECORATE && word_count >= 4U && words[offset + 2U] == DECORATION_BINDING && words[offset + 3U] == requested_binding) {
                    return true;
                }
                offset += word_count;
            }
            return false;
        }

        void validate_resources(const EffectPack &pack, bool history, bool spectrum, bool spectrum_history, bool original_frame) {
            if (history && !pack.requirements.history) {
                throw std::runtime_error("effect pack uses history binding 2 but requires.history is false");
            }
            if (spectrum && !pack.requirements.spectrum) {
                throw std::runtime_error("effect pack uses spectrum binding 3 but requires.spectrum is false");
            }
            if (spectrum_history && !pack.requirements.spectrum_history) {
                throw std::runtime_error("effect pack uses spectrum-history binding 4 but requires.spectrum_history is false");
            }
            if (original_frame && !pack.requirements.original_frame) {
                throw std::runtime_error("effect pack uses original-frame binding 6 but requires.original_frame is false");
            }
        }

        void write_cache_manifest(const EffectPack &pack, const EffectPackBuildResult &result) {
            const fs::path destination = result.build_root / "effect-cache.json";
            fs::path temporary = destination;
            temporary += ".acmxvk-tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            {
                std::ofstream output(temporary, std::ios::out | std::ios::trunc);
                if (!output) {
                    throw std::runtime_error("unable to create effect-pack cache manifest");
                }
                output << "{\n    \"format\": \"acmxvk-effect-cache\",\n    \"version\": 1,\n    \"pack_id\": \"" << pack.id << "\",\n    \"passes\": [\n";
                for (std::size_t index = 0; index < result.compiled_passes.size(); ++index) {
                    output << "        \"" << result.compiled_passes[index].lexically_relative(result.build_root).generic_string() << '"' << (index + 1U < result.compiled_passes.size() ? ",\n" : "\n");
                }
                output << "    ]\n}\n";
            }
            std::error_code error;
#ifdef _WIN32
            fs::remove(destination, error);
            error.clear();
#endif
            fs::rename(temporary, destination, error);
            if (error) {
                fs::remove(temporary, error);
                throw std::runtime_error("unable to install effect-pack cache manifest: " + error.message());
            }
        }

    } // namespace

    EffectPackBuildResult build_effect_pack(const EffectPack &pack, const EffectPackBuildOptions &options) {
        if (options.parallel_jobs == 0U || options.parallel_jobs > MAX_EFFECT_PACK_PASSES) {
            throw std::runtime_error("effect-pack parallel job count must be between 1 and 64");
        }
        std::error_code error;
        const fs::path root = fs::weakly_canonical(pack.root, error);
        if (error || !fs::is_directory(root)) {
            throw std::runtime_error("effect-pack directory is unavailable: " + pack.root.string());
        }
        const fs::path manifest = canonical_pack_file(pack.manifest, root, "effect-pack manifest");
        EffectPackBuildResult result;
        result.build_root = root / ".acmxvk-build";
        if (fs::is_symlink(result.build_root)) {
            throw std::runtime_error("effect-pack build directory must not be a symbolic link");
        }
        fs::create_directories(result.build_root, error);
        if (error) {
            throw std::runtime_error("unable to create effect-pack build directory: " + error.message());
        }
        result.build_root = fs::weakly_canonical(result.build_root, error);
        if (error || !path_is_below(result.build_root, root)) {
            throw std::runtime_error("effect-pack build directory resolves outside the pack");
        }
        result.removed_temporary_files = remove_shader_build_temporary_files(result.build_root);

        struct PreparedPass {
            fs::path source;
            fs::path destination;
            mxvk::ShaderStage expected_stage = mxvk::ShaderStage::Unknown;
        };
        std::vector<PreparedPass> prepared;
        prepared.reserve(pack.passes.size());
        for (const fs::path &requested_source : pack.passes) {
            const fs::path source = canonical_pack_file(requested_source, root, "effect-pack shader pass");
            const fs::path relative = source.lexically_relative(root);
            const std::string extension = source.extension().string();
            mxvk::ShaderStage expected_stage = mxvk::ShaderStage::Unknown;
            if (extension == ".frag") {
                expected_stage = mxvk::ShaderStage::Fragment;
            } else if (extension == ".comp") {
                expected_stage = mxvk::ShaderStage::Compute;
            } else if (extension != ".spv") {
                throw std::runtime_error("effect-pack shader pass must use .frag, .comp, or .spv: " + source.string());
            }
            fs::path output_relative = relative;
            if (extension != ".spv") {
                output_relative += ".spv";
            }
            const fs::path destination = result.build_root / output_relative;
            fs::create_directories(destination.parent_path(), error);
            if (error) {
                throw std::runtime_error("unable to create effect-pack shader output directory: " + error.message());
            }
            const fs::path destination_parent = fs::weakly_canonical(destination.parent_path(), error);
            if (error || !path_is_below(destination_parent, result.build_root) || fs::is_symlink(destination)) {
                throw std::runtime_error("effect-pack shader output resolves outside the build directory: " + destination.string());
            }
            prepared.push_back({source, destination, expected_stage});
        }

        std::vector<ShaderBuildStatus> statuses(prepared.size(), ShaderBuildStatus::Current);
        std::unordered_map<std::string, std::size_t> first_pass_by_source;
        std::vector<std::size_t> work_indices;
        std::vector<std::size_t> source_indices(prepared.size());
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            const auto [entry, inserted] = first_pass_by_source.emplace(prepared[index].source.generic_string(), index);
            source_indices[index] = entry->second;
            if (inserted) {
                work_indices.push_back(index);
            }
        }
        std::atomic<std::size_t> next_pass{0};
        std::atomic<std::size_t> completed_passes{0};
        std::mutex failure_mutex;
        std::exception_ptr first_failure;
        const auto worker = [&] {
            while (true) {
                const std::size_t index = next_pass.fetch_add(1U);
                if (index >= work_indices.size()) {
                    return;
                }
                try {
                    const std::size_t pass_index = work_indices[index];
                    const PreparedPass &pass = prepared[pass_index];
                    const bool stale = options.force || dependencies_newer_than(pass.source, manifest, pass.destination, root);
                    statuses[pass_index] = build_shader_file(options.glslc_executable, root, pass.source, pass.destination, stale);
                } catch (...) {
                    const std::lock_guard lock(failure_mutex);
                    if (!first_failure) {
                        first_failure = std::current_exception();
                    }
                }
                const std::size_t completed = completed_passes.fetch_add(1U) + 1U;
                if (options.progress) {
                    options.progress(completed, work_indices.size());
                }
            }
        };
        const std::size_t worker_count = std::min(options.parallel_jobs, work_indices.size());
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t index = 0; index < worker_count; ++index) {
            workers.emplace_back(worker);
        }
        for (std::thread &thread : workers) {
            thread.join();
        }
        if (first_failure) {
            std::rethrow_exception(first_failure);
        }
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            statuses[index] = statuses[source_indices[index]];
        }

        bool uses_history = false;
        bool uses_spectrum = false;
        bool uses_spectrum_history = false;
        bool uses_original_frame = false;
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            const PreparedPass &pass = prepared[index];
            input::validate_spirv_file(pass.destination, "effect-pack compiled shader");
            const mxvk::ShaderModuleInfo info = mxvk::inspect_spirv(mxvk::load_spv(pass.destination.string()));
            if (info.stage != mxvk::ShaderStage::Fragment && info.stage != mxvk::ShaderStage::Compute) {
                throw std::runtime_error("effect-pack pass is not a fragment or compute shader: " + pass.source.string());
            }
            if (pass.expected_stage != mxvk::ShaderStage::Unknown && info.stage != pass.expected_stage) {
                throw std::runtime_error("effect-pack compiled shader stage does not match its source extension: " + pass.source.string());
            }
            uses_history = uses_history || info.usesHistoryTexture;
            uses_spectrum = uses_spectrum || info.usesSpectrumTexture;
            uses_spectrum_history = uses_spectrum_history || info.usesSpectrumHistoryTexture;
            uses_original_frame = uses_original_frame || uses_descriptor_binding(pass.destination, 6U);
            result.compiled_passes.push_back(pass.destination);
            if (statuses[index] == ShaderBuildStatus::Compiled) {
                ++result.compiled;
            } else if (statuses[index] == ShaderBuildStatus::Copied) {
                ++result.copied;
            } else {
                ++result.current;
            }
        }
        validate_resources(pack, uses_history, uses_spectrum, uses_spectrum_history, uses_original_frame);
        write_cache_manifest(pack, result);
        return result;
    }

    EffectPackBuildResult load_effect_pack_cache(const EffectPack &pack) {
        std::error_code error;
        const fs::path root = fs::weakly_canonical(pack.root, error);
        if (error || !fs::is_directory(root)) {
            throw std::runtime_error("effect-pack directory is unavailable: " + pack.root.string());
        }
        const fs::path manifest = canonical_pack_file(pack.manifest, root, "effect-pack manifest");
        EffectPackBuildResult result;
        result.build_root = fs::weakly_canonical(root / ".acmxvk-build", error);
        if (error || !fs::is_directory(result.build_root) || fs::is_symlink(root / ".acmxvk-build") || !path_is_below(result.build_root, root)) {
            throw std::runtime_error("effect-pack compiled cache is unavailable");
        }

        bool uses_history = false;
        bool uses_spectrum = false;
        bool uses_spectrum_history = false;
        bool uses_original_frame = false;
        for (const fs::path &requested_source : pack.passes) {
            const fs::path source = canonical_pack_file(requested_source, root, "effect-pack shader pass");
            const std::string extension = source.extension().string();
            mxvk::ShaderStage expected_stage = mxvk::ShaderStage::Unknown;
            if (extension == ".frag") {
                expected_stage = mxvk::ShaderStage::Fragment;
            } else if (extension == ".comp") {
                expected_stage = mxvk::ShaderStage::Compute;
            } else if (extension != ".spv") {
                throw std::runtime_error("effect-pack shader pass must use .frag, .comp, or .spv: " + source.string());
            }
            fs::path output_relative = source.lexically_relative(root);
            if (extension != ".spv") {
                output_relative += ".spv";
            }
            const fs::path requested_output = result.build_root / output_relative;
            const fs::path output = fs::weakly_canonical(requested_output, error);
            if (error || !path_is_below(output, result.build_root) || fs::is_symlink(requested_output)) {
                throw std::runtime_error("effect-pack cached shader escapes its cache directory: " + requested_output.string());
            }
            if (dependencies_newer_than(source, manifest, output, root)) {
                throw std::runtime_error("effect-pack compiled cache is missing or stale for: " + source.string());
            }
            input::validate_spirv_file(output, "effect-pack cached shader");
            const mxvk::ShaderModuleInfo info = mxvk::inspect_spirv(mxvk::load_spv(output.string()));
            if (info.stage != mxvk::ShaderStage::Fragment && info.stage != mxvk::ShaderStage::Compute) {
                throw std::runtime_error("effect-pack cached pass is not a fragment or compute shader: " + source.string());
            }
            if (expected_stage != mxvk::ShaderStage::Unknown && info.stage != expected_stage) {
                throw std::runtime_error("effect-pack cached shader stage does not match its source extension: " + source.string());
            }
            uses_history = uses_history || info.usesHistoryTexture;
            uses_spectrum = uses_spectrum || info.usesSpectrumTexture;
            uses_spectrum_history = uses_spectrum_history || info.usesSpectrumHistoryTexture;
            uses_original_frame = uses_original_frame || uses_descriptor_binding(output, 6U);
            result.compiled_passes.push_back(output);
            ++result.current;
        }
        validate_resources(pack, uses_history, uses_spectrum, uses_spectrum_history, uses_original_frame);
        return result;
    }

} // namespace acmxvk
