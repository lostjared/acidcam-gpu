#include "effect_pack_build.hpp"

#include "../input_validation.hpp"
#include "../version_info.hpp"
#include "shader_compiler.hpp"

#include <json/json.h>
#include <mxvk/mxvk.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
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
        constexpr int CACHE_FORMAT_VERSION = 2;
        constexpr std::string_view SHADER_ABI = "acmxvk-effect-abi-1";
        constexpr std::string_view VULKAN_TARGET = "vulkan1.0";

        struct CacheFingerprints {
            std::string source_hash;
            std::string include_hash;
            std::vector<std::string> pass_hashes;
        };

        void hash_bytes(std::uint64_t &hash, std::string_view bytes) {
            for (const unsigned char byte : bytes) {
                hash = (hash ^ byte) * 1099511628211ULL;
            }
            hash = (hash ^ 0U) * 1099511628211ULL;
        }

        std::string hash_hex(std::uint64_t hash) {
            std::ostringstream output;
            output << std::hex << std::setfill('0') << std::setw(16) << hash;
            return output.str();
        }

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

        [[nodiscard]] CacheFingerprints fingerprint_sources(const EffectPack &pack, const fs::path &root) {
            std::set<std::string> source_paths;
            std::set<std::string> include_paths;
            std::unordered_set<std::string> visited;
            std::vector<fs::path> dependencies;
            std::vector<std::set<std::string>> per_pass_paths;
            for (const fs::path &requested : pack.passes) {
                const fs::path source = canonical_pack_file(requested, root, "effect-pack shader pass");
                source_paths.insert(source.lexically_relative(root).generic_string());
                std::unordered_set<std::string> pass_visited;
                std::vector<fs::path> pass_dependencies;
                collect_dependencies(source, root, pass_visited, pass_dependencies, 0);
                std::set<std::string> paths;
                for (const fs::path &dependency : pass_dependencies) {
                    paths.insert(dependency.lexically_relative(root).generic_string());
                }
                per_pass_paths.push_back(std::move(paths));
                collect_dependencies(source, root, visited, dependencies, 0);
            }
            for (const fs::path &dependency : dependencies) {
                const std::string relative = dependency.lexically_relative(root).generic_string();
                if (!source_paths.contains(relative)) {
                    include_paths.insert(relative);
                }
            }
            const auto fingerprint = [&root](const std::set<std::string> &paths) {
                std::uint64_t hash = 14695981039346656037ULL;
                for (const std::string &relative : paths) {
                    if (fs::path(relative).extension() == ".spv") {
                        input::validate_spirv_file(root / relative, "effect-pack cache dependency");
                    } else {
                        input::validate_text_file(root / relative, "effect-pack cache dependency");
                    }
                    std::ifstream file(root / relative, std::ios::binary);
                    if (!file) {
                        throw std::runtime_error("cannot hash effect-pack dependency: " + relative);
                    }
                    const std::string bytes(std::istreambuf_iterator<char>(file), {});
                    hash_bytes(hash, relative);
                    hash_bytes(hash, bytes);
                }
                return hash_hex(hash);
            };
            CacheFingerprints result{fingerprint(source_paths), fingerprint(include_paths), {}};
            for (const auto &paths : per_pass_paths) {
                result.pass_hashes.push_back(fingerprint(paths));
            }
            return result;
        }

        [[nodiscard]] Json::Value read_cache_metadata(const fs::path &build_root) {
            const fs::path path = build_root / "effect-cache.json";
            if (!fs::is_regular_file(path)) {
                return {};
            }
            input::validate_text_file(path, "effect-pack cache manifest");
            std::ifstream file(path);
            Json::Value cache;
            Json::CharReaderBuilder reader;
            std::string errors;
            if (!Json::parseFromStream(reader, file, &cache, &errors) || !cache.isObject()) {
                return {};
            }
            return cache;
        }

        [[nodiscard]] bool cache_metadata_compatible(const EffectPack &pack, const Json::Value &cache) { return cache.isObject() && cache["format"].isString() && cache["version"].isInt() && cache["pack_id"].isString() && cache["shader_abi"].isString() && cache["vulkan_target"].isString() && cache["format"].asString() == "acmxvk-effect-cache" && cache["version"].asInt() == CACHE_FORMAT_VERSION && cache["pack_id"].asString() == pack.id && cache["shader_abi"].asString() == SHADER_ABI && cache["vulkan_target"].asString() == VULKAN_TARGET; }

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

        void write_cache_manifest(const EffectPack &pack, const EffectPackBuildResult &result, const CacheFingerprints &fingerprints, const std::string &compiler) {
            const fs::path destination = result.build_root / "effect-cache.json";
            fs::path temporary = destination;
            temporary += ".acmxvk-tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            Json::Value cache(Json::objectValue);
            cache["format"] = "acmxvk-effect-cache";
            cache["version"] = CACHE_FORMAT_VERSION;
            cache["pack_id"] = pack.id;
            cache["shader_abi"] = std::string(SHADER_ABI);
            cache["vulkan_target"] = std::string(VULKAN_TARGET);
            cache["source_hash"] = fingerprints.source_hash;
            cache["include_hash"] = fingerprints.include_hash;
            cache["compiler"] = fs::path(compiler).filename().string();
            cache["compiled_by_acmxvk"] = ACMXVK_VERSION_INFO;
            for (const fs::path &pass : result.compiled_passes) {
                cache["passes"].append(pass.lexically_relative(result.build_root).generic_string());
            }
            for (const std::string &hash : fingerprints.pass_hashes) {
                cache["pass_hashes"].append(hash);
            }
            {
                std::ofstream output(temporary, std::ios::out | std::ios::trunc);
                if (!output) {
                    throw std::runtime_error("unable to create effect-pack cache manifest");
                }
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "    ";
                output << Json::writeString(writer, cache);
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

        const CacheFingerprints fingerprints = fingerprint_sources(pack, root);
        const Json::Value previous_cache = read_cache_metadata(result.build_root);
        const bool cache_compatible = cache_metadata_compatible(pack, previous_cache);

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
                    const bool pass_compatible = cache_compatible && previous_cache["pass_hashes"].isArray() && previous_cache["pass_hashes"].size() == prepared.size() && previous_cache["pass_hashes"][static_cast<Json::ArrayIndex>(pass_index)].isString() && previous_cache["pass_hashes"][static_cast<Json::ArrayIndex>(pass_index)].asString() == fingerprints.pass_hashes[pass_index];
                    const bool stale = options.force || !pass_compatible || dependencies_newer_than(pass.source, manifest, pass.destination, root);
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
        write_cache_manifest(pack, result, fingerprints, options.glslc_executable);
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
        const CacheFingerprints fingerprints = fingerprint_sources(pack, root);
        const Json::Value cache_metadata = read_cache_metadata(result.build_root);
        if (!cache_metadata_compatible(pack, cache_metadata) || !cache_metadata["source_hash"].isString() || !cache_metadata["include_hash"].isString() || cache_metadata["source_hash"].asString() != fingerprints.source_hash || cache_metadata["include_hash"].asString() != fingerprints.include_hash || !cache_metadata["pass_hashes"].isArray() || cache_metadata["pass_hashes"].size() != pack.passes.size()) {
            throw std::runtime_error("effect-pack compiled cache is missing or incompatible with its sources or shader ABI");
        }
        for (std::size_t index = 0; index < fingerprints.pass_hashes.size(); ++index) {
            if (!cache_metadata["pass_hashes"][static_cast<Json::ArrayIndex>(index)].isString() || cache_metadata["pass_hashes"][static_cast<Json::ArrayIndex>(index)].asString() != fingerprints.pass_hashes[index]) {
                throw std::runtime_error("effect-pack compiled cache is incompatible with shader source or includes");
            }
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
