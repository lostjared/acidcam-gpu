#include "shader_library.hpp"

#include "../input_validation.hpp"

#include <mxvk/mxvk.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <stdexcept>
#include <unordered_set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
extern char **environ;
#endif

namespace acmxvk {
    [[nodiscard]] bool isValidCustomUniformName(const std::string &name) {
        if (name.starts_with("gl_")) {
            return false;
        }
        try {
            input::validate_string(name, input::StringKind::Identifier,
                                   "custom uniform name");
            return true;
        } catch (const std::runtime_error &) {
            return false;
        }
    }

    [[nodiscard]] ShaderManifest loadShaderManifest(const fs::path &directory) {
        ShaderManifest manifest;
        const fs::path json_path = directory / "library.json";
        const fs::path text_path = directory / "index.txt";
        if (fs::is_regular_file(json_path)) {
            manifest.path = json_path;
            input::validate_text_file(json_path, "shader library.json");
            try {
                cv::FileStorage storage(json_path.string(),
                                        cv::FileStorage::READ |
                                            cv::FileStorage::FORMAT_JSON);
                if (!storage.isOpened()) {
                    throw std::runtime_error("unable to open shader manifest: " +
                                             json_path.string());
                }
                const cv::FileNode shader_entries = storage["shaders"];
                if (shader_entries.type() == cv::FileNode::NONE ||
                    !shader_entries.isSeq()) {
                    throw std::runtime_error(json_path.string() +
                                             " must contain a 'shaders' array");
                }
                for (const cv::FileNode &entry : shader_entries) {
                    if (manifest.entries.size() >=
                        input::MAX_SHADER_ENTRIES) {
                        throw std::runtime_error(
                            json_path.string() +
                            " contains too many shader entries");
                    }
                    std::string filename;
                    if (entry.isString()) {
                        entry >> filename;
                    } else if (entry.isMap() && !entry["file"].empty()) {
                        entry["file"] >> filename;
                    } else {
                        throw std::runtime_error(
                            json_path.string() +
                            " contains a shader entry without a file name");
                    }
                    filename = trim(std::move(filename));
                    if (filename.empty()) {
                        throw std::runtime_error(
                            json_path.string() +
                            " contains a shader entry without a file name");
                    }
                    input::validate_string(
                        filename, input::StringKind::Path,
                        json_path.string() + " shader file");
                    manifest.entries.push_back(std::move(filename));
                }

                const cv::FileNode custom_uniforms = storage["custom_uniforms"];
                if (!custom_uniforms.empty()) {
                    if (!custom_uniforms.isMap()) {
                        throw std::runtime_error(
                            json_path.string() +
                            " field 'custom_uniforms' must be an object");
                    }
                    bool has_explicit_slots = false;
                    bool has_implicit_slots = false;
                    std::unordered_set<std::size_t> occupied_slots;
                    for (auto iterator = custom_uniforms.begin();
                         iterator != custom_uniforms.end(); ++iterator) {
                        if (manifest.custom_uniforms.size() >=
                            mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains more than " +
                                std::to_string(mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS) +
                                " custom uniforms");
                        }

                        const cv::FileNode entry = *iterator;
                        ShaderManifest::CustomUniform uniform;
                        uniform.name = entry.name();
                        if (!entry.isMap() ||
                            !isValidCustomUniformName(uniform.name)) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains an invalid custom uniform: " +
                                uniform.name);
                        }
                        uniform.slot = manifest.custom_uniforms.size();
                        if (!entry["slot"].empty()) {
                            int slot = -1;
                            entry["slot"] >> slot;
                            if (slot < 0 ||
                                slot >= static_cast<int>(
                                            mxvk::VK_Sprite::MAX_CUSTOM_UNIFORMS)) {
                                throw std::runtime_error(
                                    json_path.string() +
                                    " contains an invalid slot for custom uniform: " +
                                    uniform.name);
                            }
                            uniform.slot = static_cast<std::size_t>(slot);
                            if (!occupied_slots.insert(uniform.slot).second) {
                                throw std::runtime_error(
                                    json_path.string() +
                                    " assigns more than one custom uniform to slot " +
                                    std::to_string(slot));
                            }
                            has_explicit_slots = true;
                        } else {
                            has_implicit_slots = true;
                        }
                        if (!entry["minimum"].empty()) {
                            entry["minimum"] >> uniform.minimum;
                        }
                        if (!entry["maximum"].empty()) {
                            entry["maximum"] >> uniform.maximum;
                        }
                        if (!entry["step"].empty()) {
                            entry["step"] >> uniform.step;
                        }
                        uniform.value = uniform.minimum;
                        if (!entry["value"].empty()) {
                            entry["value"] >> uniform.value;
                        }
                        if (!std::isfinite(uniform.minimum) ||
                            !std::isfinite(uniform.maximum) ||
                            !std::isfinite(uniform.step) ||
                            !std::isfinite(uniform.value) ||
                            uniform.maximum <= uniform.minimum ||
                            uniform.step <= 0.0 ||
                            std::abs(uniform.minimum) >
                                std::numeric_limits<float>::max() ||
                            std::abs(uniform.maximum) >
                                std::numeric_limits<float>::max() ||
                            std::abs(uniform.step) >
                                std::numeric_limits<float>::max() ||
                            std::abs(uniform.value) >
                                std::numeric_limits<float>::max()) {
                            throw std::runtime_error(
                                json_path.string() +
                                " contains an invalid range for custom uniform: " +
                                uniform.name);
                        }
                        uniform.value = std::clamp(
                            uniform.value, uniform.minimum, uniform.maximum);
                        manifest.custom_uniforms.push_back(std::move(uniform));
                    }
                    if (has_explicit_slots && has_implicit_slots) {
                        throw std::runtime_error(
                            json_path.string() +
                            " must specify a slot for every custom uniform or none");
                    }
                    if (has_explicit_slots) {
                        std::sort(manifest.custom_uniforms.begin(),
                                  manifest.custom_uniforms.end(),
                                  [](const ShaderManifest::CustomUniform &left,
                                     const ShaderManifest::CustomUniform &right) {
                                      return left.slot < right.slot;
                                  });
                        for (std::size_t slot = 0;
                             slot < manifest.custom_uniforms.size(); ++slot) {
                            if (manifest.custom_uniforms[slot].slot != slot) {
                                throw std::runtime_error(
                                    json_path.string() +
                                    " custom uniform slots must be contiguous from zero");
                            }
                        }
                    }
                }
            } catch (const cv::Exception &error) {
                throw std::runtime_error("unable to parse shader manifest " +
                                         json_path.string() + ": " + error.what());
            }
            return manifest;
        }

        if (!fs::is_regular_file(text_path)) {
            throw std::runtime_error("no library.json or index.txt found in shader library: " +
                                     directory.string());
        }
        manifest.path = text_path;
        input::validate_file_size(text_path, "shader index.txt");
        std::ifstream manifest_input(text_path);
        if (!manifest_input) {
            throw std::runtime_error("unable to open shader manifest: " +
                                     text_path.string());
        }
        std::string line;
        std::size_t line_number = 1;
        while (input::read_bounded_line(manifest_input, line,
                                        "shader index.txt", line_number++)) {
            line = trim(std::move(line));
            if (!line.empty() && line.front() != '#') {
                if (manifest.entries.size() >=
                    input::MAX_SHADER_ENTRIES) {
                    throw std::runtime_error(
                        text_path.string() +
                        " contains too many shader entries");
                }
                input::validate_string(
                    line, input::StringKind::Path,
                    text_path.string() + " shader file");
                manifest.entries.push_back(std::move(line));
            }
        }
        return manifest;
    }

    [[nodiscard]] fs::path resolveShaderManifestEntry(const fs::path &directory,
                                                      std::string entry) {
        std::replace(entry.begin(), entry.end(), '\\', '/');
        const fs::path relative_path(entry);
        if (relative_path.is_absolute()) {
            return {};
        }

        const fs::path normalized = relative_path.lexically_normal();
        const std::string normalized_text = normalized.generic_string();
        if (normalized_text.empty() || normalized_text == "." ||
            normalized_text == ".." || normalized_text.starts_with("../") ||
            normalized_text.find("/../") != std::string::npos ||
            normalized.extension() != ".spv") {
            return {};
        }

        std::error_code error;
        const fs::path root = fs::weakly_canonical(directory, error);
        if (error) {
            return {};
        }
        const fs::path shader = fs::weakly_canonical(root / normalized, error);
        if (error || !fs::is_regular_file(shader)) {
            return {};
        }
        const std::string resolved_relative = shader.lexically_relative(root).generic_string();
        if (resolved_relative.empty() || resolved_relative == ".." ||
            resolved_relative.starts_with("../")) {
            return {};
        }
        return shader;
    }

    [[nodiscard]] fs::path resolveShaderBuildEntry(const fs::path &directory,
                                                   std::string entry) {
        std::replace(entry.begin(), entry.end(), '\\', '/');
        const fs::path relative_path(entry);
        if (relative_path.is_absolute()) {
            return {};
        }

        const fs::path normalized = relative_path.lexically_normal();
        const std::string normalized_text = normalized.generic_string();
        const std::string extension = normalized.extension().string();
        if (normalized_text.empty() || normalized_text == "." ||
            normalized_text == ".." || normalized_text.starts_with("../") ||
            normalized_text.find("/../") != std::string::npos ||
            (extension != ".frag" && extension != ".comp" &&
             extension != ".spv")) {
            return {};
        }

        std::error_code error;
        const fs::path root = fs::weakly_canonical(directory, error);
        if (error) {
            return {};
        }
        const fs::path source = fs::weakly_canonical(root / normalized, error);
        if (error || !fs::is_regular_file(source)) {
            return {};
        }
        const std::string resolved_relative =
            source.lexically_relative(root).generic_string();
        if (resolved_relative.empty() || resolved_relative == ".." ||
            resolved_relative.starts_with("../")) {
            return {};
        }
        return source;
    }

    [[nodiscard]] std::string escapeJson(std::string_view value) {
        std::ostringstream escaped;
        for (const unsigned char character : value) {
            switch (character) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\b':
                escaped << "\\b";
                break;
            case '\f':
                escaped << "\\f";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    escaped << "\\u" << std::hex << std::uppercase
                            << std::setw(4) << std::setfill('0')
                            << static_cast<unsigned int>(character)
                            << std::dec << std::nouppercase;
                } else {
                    escaped << static_cast<char>(character);
                }
                break;
            }
        }
        return escaped.str();
    }

    [[nodiscard]] fs::path temporaryBuildPath(const fs::path &destination) {
        static std::uint64_t sequence = 0;
        for (int attempt = 0; attempt < 100; ++attempt) {
            fs::path temporary = destination;
            temporary += ".acmxvk-tmp-" +
                         std::to_string(std::chrono::steady_clock::now()
                                            .time_since_epoch()
                                            .count()) +
                         "-" +
                         std::to_string(++sequence);
            if (!fs::exists(temporary)) {
                return temporary;
            }
        }
        throw std::runtime_error(
            "unable to allocate a temporary shader build path for: " +
            destination.string());
    }

    void replaceBuiltFile(const fs::path &temporary,
                          const fs::path &destination) {
        std::error_code error;
        fs::rename(temporary, destination, error);
        if (error) {
            fs::remove(temporary);
            throw std::runtime_error("unable to install built file " +
                                     destination.string() + ": " +
                                     error.message());
        }
    }

    class ShaderCompilationError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

#ifdef _WIN32
    [[nodiscard]] std::wstring utf8_to_wide(const std::string &value) {
        if (value.empty()) {
            return {};
        }
        const int length = MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
            static_cast<int>(value.size()), nullptr, 0);
        if (length <= 0) {
            throw std::runtime_error("invalid UTF-8 in Windows command argument");
        }
        std::wstring result(static_cast<std::size_t>(length), L'\0');
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                static_cast<int>(value.size()), result.data(),
                                length) != length) {
            throw std::runtime_error("unable to convert Windows command argument");
        }
        return result;
    }

    [[nodiscard]] std::wstring
    quote_windows_argument(const std::wstring &value) {
        std::wstring quoted{L"\""};
        std::size_t backslash_count = 0;
        for (const wchar_t character : value) {
            if (character == L'\\') {
                ++backslash_count;
                continue;
            }
            if (character == L'"') {
                quoted.append(backslash_count * 2U + 1U, L'\\');
                quoted += character;
                backslash_count = 0;
                continue;
            }
            quoted.append(backslash_count, L'\\');
            backslash_count = 0;
            quoted += character;
        }
        quoted.append(backslash_count * 2U, L'\\');
        quoted += L'"';
        return quoted;
    }

    [[nodiscard]] DWORD run_windows_process(
        const std::vector<std::wstring> &arguments) {
        std::wstring command_line;
        for (const std::wstring &argument : arguments) {
            if (!command_line.empty()) {
                command_line += L' ';
            }
            command_line += quote_windows_argument(argument);
        }

        STARTUPINFOW startup_info{};
        startup_info.cb = sizeof(startup_info);
        PROCESS_INFORMATION process_info{};
        if (CreateProcessW(nullptr, command_line.data(), nullptr, nullptr,
                           TRUE, 0, nullptr, nullptr, &startup_info,
                           &process_info) == FALSE) {
            const DWORD process_error = GetLastError();
            throw std::runtime_error(
                "unable to execute glslc (Windows error " +
                std::to_string(process_error) + ")");
        }

        CloseHandle(process_info.hThread);
        const DWORD wait_result =
            WaitForSingleObject(process_info.hProcess, INFINITE);
        DWORD exit_code = 1;
        if (wait_result != WAIT_OBJECT_0 ||
            GetExitCodeProcess(process_info.hProcess, &exit_code) == FALSE) {
            const DWORD process_error = GetLastError();
            CloseHandle(process_info.hProcess);
            throw std::runtime_error(
                "unable to wait for glslc (Windows error " +
                std::to_string(process_error) + ")");
        }
        CloseHandle(process_info.hProcess);
        return exit_code;
    }
#endif

    void runGlslc(const std::string &executable, const fs::path &source_root,
                  const fs::path &source, const fs::path &output) {
#if defined(__linux__) || defined(__APPLE__)
        std::vector<std::string> arguments{
            executable, "-I", source_root.string(), source.string(), "-o",
            output.string()};
        std::vector<char *> argument_pointers;
        argument_pointers.reserve(arguments.size() + 1U);
        for (std::string &argument : arguments) {
            argument_pointers.push_back(argument.data());
        }
        argument_pointers.push_back(nullptr);

        pid_t process = 0;
        const int spawn_result =
            posix_spawnp(&process, executable.c_str(), nullptr, nullptr,
                         argument_pointers.data(), environ);
        if (spawn_result != 0) {
            throw std::runtime_error("unable to execute glslc '" + executable +
                                     "': " + std::strerror(spawn_result));
        }

        int status = 0;
        while (::waitpid(process, &status, 0) < 0) {
            if (errno != EINTR) {
                throw std::runtime_error("unable to wait for glslc: " +
                                         std::string(std::strerror(errno)));
            }
        }
        if (!WIFEXITED(status)) {
            throw std::runtime_error("glslc terminated by a signal for " +
                                     source.string());
        }
        if (WEXITSTATUS(status) != 0) {
            throw ShaderCompilationError(
                "glslc failed for " + source.string() + " (exit status " +
                std::to_string(WEXITSTATUS(status)) + ")");
        }
#elif defined(_WIN32)
        const DWORD result = run_windows_process(
            {utf8_to_wide(executable), L"-I", source_root.wstring(),
             source.wstring(), L"-o", output.wstring()});
        if (result != 0U) {
            throw ShaderCompilationError(
                "glslc failed for " + source.string() + " (exit status " +
                std::to_string(result) + ")");
        }
#else
#error Unsupported platform
#endif
    }

    [[nodiscard]] int buildShaderLibrary(const Options &options) {
        const fs::path requested_manifest =
            fs::absolute(options.build_manifest).lexically_normal();
        if (requested_manifest.filename() != "library.json") {
            throw std::runtime_error(
                "--build must name a file called library.json");
        }
        input::validate_text_file(requested_manifest,
                                  "source shader library.json");

        std::error_code error;
        const fs::path source_root =
            fs::weakly_canonical(requested_manifest.parent_path(), error);
        if (error || source_root.empty()) {
            throw std::runtime_error("unable to resolve source shader library: " +
                                     requested_manifest.string());
        }
        fs::create_directories(options.build_directory, error);
        if (error) {
            throw std::runtime_error("unable to create shader build directory: " +
                                     error.message());
        }
        const fs::path output_root =
            fs::weakly_canonical(options.build_directory, error);
        if (error || output_root.empty()) {
            throw std::runtime_error("unable to resolve shader build directory: " +
                                     options.build_directory);
        }
        if (source_root == output_root) {
            throw std::runtime_error(
                "the shader output directory must differ from the source "
                "library directory");
        }

        const ShaderManifest manifest = loadShaderManifest(source_root);
        if (manifest.entries.empty()) {
            throw std::runtime_error(
                "source library.json contains no shader entries");
        }

        std::vector<std::string> output_entries;
        output_entries.reserve(manifest.entries.size());
        std::unordered_set<std::string> unique_outputs;
        std::size_t compiled = 0;
        std::size_t copied = 0;
        std::size_t current = 0;
        std::size_t failed = 0;
        std::size_t pruned = 0;
        std::size_t processed = 0;
        int next_progress = 5;

        const auto report_progress = [&] {
            ++processed;
            const int percentage = static_cast<int>(
                processed * 100U / manifest.entries.size());
            while (next_progress <= 100 && percentage >= next_progress) {
                std::cout << "acmxvk: build progress: " << next_progress
                          << "% (" << processed << '/'
                          << manifest.entries.size() << ")\n"
                          << std::flush;
                next_progress += 5;
            }
        };

        for (const std::string &entry : manifest.entries) {
            fs::path source;
            fs::path destination;
            try {
                source = resolveShaderBuildEntry(source_root, entry);
                if (source.empty()) {
                    throw std::runtime_error(
                        "source library contains an unavailable or unsafe shader: " +
                        entry);
                }

                std::string normalized_entry = entry;
                std::replace(normalized_entry.begin(), normalized_entry.end(),
                             '\\', '/');
                fs::path relative(normalized_entry);
                relative = relative.lexically_normal();
                if (relative.extension() != ".spv") {
                    relative += ".spv";
                }
                const std::string output_entry = relative.generic_string();
                std::string output_key = output_entry;
                std::transform(
                    output_key.begin(), output_key.end(), output_key.begin(),
                    [](unsigned char character) {
                        return static_cast<char>(std::tolower(character));
                    });
                if (!unique_outputs.insert(output_key).second) {
                    throw std::runtime_error(
                        "source library produces a duplicate output path: " +
                        output_entry);
                }

                destination = output_root / relative;
                error.clear();
                fs::create_directories(destination.parent_path(), error);
                if (error) {
                    throw std::runtime_error(
                        "unable to create shader output directory: " +
                        error.message());
                }
                const fs::path destination_parent =
                    fs::weakly_canonical(destination.parent_path(), error);
                const std::string parent_relative =
                    error ? std::string{}
                          : destination_parent.lexically_relative(output_root)
                                .generic_string();
                if (error || parent_relative == ".." ||
                    parent_relative.starts_with("../") ||
                    fs::is_symlink(destination)) {
                    throw std::runtime_error(
                        "shader output resolves outside the output directory: " +
                        output_entry);
                }

                bool needs_build = !fs::is_regular_file(destination);
                if (!needs_build) {
                    needs_build = fs::last_write_time(destination, error) <
                                  fs::last_write_time(source);
                    if (error) {
                        needs_build = true;
                        error.clear();
                    }
                }
                if (!needs_build) {
                    try {
                        input::validate_spirv_file(
                            destination, "built shader module");
                    } catch (const std::runtime_error &) {
                        needs_build = true;
                    }
                }

                if (needs_build) {
                    const fs::path temporary = temporaryBuildPath(destination);
                    const bool copy_source = source.extension() == ".spv";
                    try {
                        if (copy_source) {
                            input::validate_spirv_file(
                                source, "source shader module");
                            fs::copy_file(
                                source, temporary,
                                fs::copy_options::overwrite_existing);
                        } else {
                            input::validate_text_file(source,
                                                      "GLSL shader source");
                            runGlslc(options.glslc_executable, source_root,
                                     source, temporary);
                        }
                        input::validate_spirv_file(temporary,
                                                   "compiled shader module");
                        replaceBuiltFile(temporary, destination);
                    } catch (...) {
                        fs::remove(temporary);
                        throw;
                    }
                    if (copy_source) {
                        ++copied;
                    } else {
                        ++compiled;
                    }
                } else {
                    ++current;
                }
                output_entries.push_back(output_entry);
            } catch (const std::exception &failure) {
                if (!options.build_fix) {
                    throw;
                }
                const bool compilation_failed =
                    dynamic_cast<const ShaderCompilationError *>(&failure) !=
                    nullptr;
                if (!destination.empty()) {
                    std::error_code remove_error;
                    fs::remove(destination, remove_error);
                    if (remove_error) {
                        throw std::runtime_error(
                            "unable to remove failed shader output " +
                            destination.string() + ": " +
                            remove_error.message());
                    }
                }
                if (options.build_prune && compilation_failed &&
                    !source.empty() &&
                    (source.extension() == ".frag" ||
                     source.extension() == ".comp")) {
                    std::error_code remove_error;
                    const bool removed = fs::remove(source, remove_error);
                    if (remove_error || !removed) {
                        throw std::runtime_error(
                            "unable to prune failed shader source " +
                            source.string() +
                            (remove_error ? ": " + remove_error.message()
                                          : ": file was not removed"));
                    }
                    ++pruned;
                    std::cerr << "acmxvk: pruned failed source '"
                              << source.string() << "'\n";
                }
                ++failed;
                std::cerr << "acmxvk: fix omitted '" << entry
                          << "': " << failure.what() << '\n';
            }
            report_progress();
        }

        const fs::path output_manifest = output_root / "library.json";
        if (fs::is_symlink(output_manifest)) {
            throw std::runtime_error(
                "refusing to replace a symbolic-link output library.json");
        }
        const fs::path temporary_manifest =
            temporaryBuildPath(output_manifest);
        {
            std::ofstream output(temporary_manifest,
                                 std::ios::out | std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "unable to create output library.json");
            }
            output << "{\n    \"version\": 1"
                   << ",\n    \"backend\": \"acmxvk\""
                   << ",\n    \"library_type\": \"runtime\"";
            if (!manifest.custom_uniforms.empty()) {
                output << ",\n    \"custom_uniforms\": {\n";
                for (std::size_t index = 0;
                     index < manifest.custom_uniforms.size(); ++index) {
                    const ShaderManifest::CustomUniform &uniform =
                        manifest.custom_uniforms[index];
                    output << "        \"" << escapeJson(uniform.name)
                           << "\": {\n"
                           << std::setprecision(15)
                           << "            \"slot\": " << uniform.slot
                           << ",\n            \"minimum\": " << uniform.minimum
                           << ",\n            \"maximum\": " << uniform.maximum
                           << ",\n            \"step\": " << uniform.step
                           << ",\n            \"value\": " << uniform.value
                           << "\n        }";
                    output << (index + 1U < manifest.custom_uniforms.size()
                                   ? ",\n"
                                   : "\n");
                }
                output << "    }";
            }
            output << ",\n    \"shaders\": [\n";
            for (std::size_t index = 0; index < output_entries.size();
                 ++index) {
                output << "        \"" << escapeJson(output_entries[index])
                       << '"'
                       << (index + 1U < output_entries.size() ? ",\n"
                                                              : "\n");
            }
            output << "    ]\n}\n";
            if (!output) {
                fs::remove(temporary_manifest);
                throw std::runtime_error(
                    "unable to write output library.json");
            }
        }
        try {
            input::validate_text_file(temporary_manifest,
                                      "built shader library.json");
            replaceBuiltFile(temporary_manifest, output_manifest);
        } catch (...) {
            fs::remove(temporary_manifest);
            throw;
        }

        std::cout << "acmxvk: shader library built in " << output_root << '\n'
                  << "acmxvk: " << compiled << " compiled, " << copied
                  << " copied, " << current << " up to date, "
                  << failed << " failed, " << pruned << " pruned, "
                  << output_entries.size()
                  << " included\n";
        return EXIT_SUCCESS;
    }
} // namespace acmxvk
