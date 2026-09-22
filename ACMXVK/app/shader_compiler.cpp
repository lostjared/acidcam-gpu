#include "shader_compiler.hpp"

#include "../input_validation.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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
    namespace {

        namespace fs = std::filesystem;

        [[nodiscard]] fs::path temporary_build_path(const fs::path &destination) {
            static std::atomic<std::uint64_t> sequence{0};
            for (int attempt = 0; attempt < 100; ++attempt) {
                fs::path temporary = destination;
                temporary += ".acmxvk-tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence.fetch_add(1U) + 1U);
                if (!fs::exists(temporary)) {
                    return temporary;
                }
            }
            throw std::runtime_error("unable to allocate a temporary shader build path for: " + destination.string());
        }

        void remove_temporary_file(const fs::path &path) noexcept {
#ifdef _WIN32
            constexpr int REMOVE_ATTEMPTS = 40;
            for (int attempt = 0; attempt < REMOVE_ATTEMPTS; ++attempt) {
                if (DeleteFileW(path.c_str()) != FALSE) {
                    return;
                }
                const DWORD remove_error = GetLastError();
                if (remove_error == ERROR_FILE_NOT_FOUND || remove_error == ERROR_PATH_NOT_FOUND) {
                    return;
                }
                if (remove_error != ERROR_SHARING_VIOLATION && remove_error != ERROR_LOCK_VIOLATION && remove_error != ERROR_ACCESS_DENIED) {
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
#else
            std::error_code error;
            fs::remove(path, error);
#endif
        }

        void replace_built_file(const fs::path &temporary, const fs::path &destination) {
#ifdef _WIN32
            constexpr int REPLACE_ATTEMPTS = 80;
            DWORD replace_error = ERROR_SUCCESS;
            for (int attempt = 0; attempt < REPLACE_ATTEMPTS; ++attempt) {
                if (MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE) {
                    return;
                }
                replace_error = GetLastError();
                if (replace_error != ERROR_SHARING_VIOLATION && replace_error != ERROR_LOCK_VIOLATION && replace_error != ERROR_ACCESS_DENIED) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            remove_temporary_file(temporary);
            const std::error_code error(static_cast<int>(replace_error), std::system_category());
            throw std::runtime_error("unable to install built file " + destination.string() + ": " + error.message());
#else
            std::error_code error;
            fs::rename(temporary, destination, error);
            if (error) {
                remove_temporary_file(temporary);
                throw std::runtime_error("unable to install built file " + destination.string() + ": " + error.message());
            }
#endif
        }

#ifdef _WIN32
        [[nodiscard]] std::wstring utf8_to_wide(const std::string &value) {
            if (value.empty()) {
                return {};
            }
            const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
            if (length <= 0) {
                throw std::runtime_error("invalid UTF-8 in Windows command argument");
            }
            std::wstring result(static_cast<std::size_t>(length), L'\0');
            if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), result.data(), length) != length) {
                throw std::runtime_error("unable to convert Windows command argument");
            }
            return result;
        }

        [[nodiscard]] std::wstring quote_windows_argument(const std::wstring &value) {
            std::wstring quoted{L"\""};
            std::size_t backslash_count = 0;
            for (const wchar_t character : value) {
                if (character == L'\\') {
                    ++backslash_count;
                    continue;
                }
                if (character == L'\"') {
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
            quoted += L'\"';
            return quoted;
        }

        [[nodiscard]] DWORD run_windows_process(const std::vector<std::wstring> &arguments) {
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
            if (CreateProcessW(nullptr, command_line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup_info, &process_info) == FALSE) {
                const DWORD process_error = GetLastError();
                throw std::runtime_error("unable to execute glslc (Windows error " + std::to_string(process_error) + ")");
            }

            CloseHandle(process_info.hThread);
            const DWORD wait_result = WaitForSingleObject(process_info.hProcess, INFINITE);
            DWORD exit_code = 1;
            if (wait_result != WAIT_OBJECT_0 || GetExitCodeProcess(process_info.hProcess, &exit_code) == FALSE) {
                const DWORD process_error = GetLastError();
                CloseHandle(process_info.hProcess);
                throw std::runtime_error("unable to wait for glslc (Windows error " + std::to_string(process_error) + ")");
            }
            CloseHandle(process_info.hProcess);
            return exit_code;
        }
#endif

        void run_glslc(const std::string &executable, const fs::path &source_root, const fs::path &source, const fs::path &output) {
#if defined(__linux__) || defined(__APPLE__)
            std::vector<std::string> arguments{executable, "-I", source_root.string(), source.string(), "-o", output.string()};
            std::vector<char *> argument_pointers;
            argument_pointers.reserve(arguments.size() + 1U);
            for (std::string &argument : arguments) {
                argument_pointers.push_back(argument.data());
            }
            argument_pointers.push_back(nullptr);

            pid_t process = 0;
            const int spawn_result = posix_spawnp(&process, executable.c_str(), nullptr, nullptr, argument_pointers.data(), environ);
            if (spawn_result != 0) {
                throw std::runtime_error("unable to execute glslc '" + executable + "': " + std::strerror(spawn_result));
            }

            int status = 0;
            while (::waitpid(process, &status, 0) < 0) {
                if (errno != EINTR) {
                    throw std::runtime_error("unable to wait for glslc: " + std::string(std::strerror(errno)));
                }
            }
            if (!WIFEXITED(status)) {
                throw std::runtime_error("glslc terminated by a signal for " + source.string());
            }
            if (WEXITSTATUS(status) != 0) {
                throw ShaderCompilationError("glslc failed for " + source.string() + " (exit status " + std::to_string(WEXITSTATUS(status)) + ")");
            }
#elif defined(_WIN32)
            const DWORD result = run_windows_process({utf8_to_wide(executable), L"-I", source_root.wstring(), source.wstring(), L"-o", output.wstring()});
            if (result != 0U) {
                throw ShaderCompilationError("glslc failed for " + source.string() + " (exit status " + std::to_string(result) + ")");
            }
#else
#error Unsupported platform
#endif
        }

    } // namespace

    ShaderBuildStatus build_shader_file(const std::string &glslc_executable, const fs::path &source_root, const fs::path &source, const fs::path &destination, bool force) {
        std::error_code error;
        bool needs_build = force || !fs::is_regular_file(destination);
        if (!needs_build) {
            needs_build = fs::last_write_time(destination, error) < fs::last_write_time(source);
            if (error) {
                needs_build = true;
                error.clear();
            }
        }
        if (!needs_build) {
            try {
                input::validate_spirv_file(destination, "built shader module");
            } catch (const std::runtime_error &) {
                needs_build = true;
            }
        }
        if (!needs_build) {
            return ShaderBuildStatus::Current;
        }

        fs::create_directories(destination.parent_path(), error);
        if (error) {
            throw std::runtime_error("unable to create shader output directory: " + error.message());
        }
        if (fs::is_symlink(destination)) {
            throw std::runtime_error("refusing to replace symbolic-link shader output: " + destination.string());
        }

        const fs::path temporary = temporary_build_path(destination);
        const bool copy_source = source.extension() == ".spv";
        try {
            if (copy_source) {
                input::validate_spirv_file(source, "source shader module");
                fs::copy_file(source, temporary, fs::copy_options::overwrite_existing);
            } else {
                input::validate_text_file(source, "GLSL shader source");
                run_glslc(glslc_executable, source_root, source, temporary);
            }
            input::validate_spirv_file(temporary, "compiled shader module");
            replace_built_file(temporary, destination);
        } catch (...) {
            remove_temporary_file(temporary);
            throw;
        }
        return copy_source ? ShaderBuildStatus::Copied : ShaderBuildStatus::Compiled;
    }

    std::size_t remove_shader_build_temporary_files(const fs::path &build_root) {
        if (!fs::is_directory(build_root) || fs::is_symlink(build_root)) {
            return 0;
        }
        std::size_t removed = 0;
        std::error_code error;
        for (fs::recursive_directory_iterator iterator(build_root, fs::directory_options::skip_permission_denied, error), end; iterator != end; iterator.increment(error)) {
            if (error) {
                error.clear();
                continue;
            }
            const fs::directory_entry &entry = *iterator;
            const std::string name = entry.path().filename().string();
            if (entry.is_directory(error) && name == ".editor-preview") {
                iterator.disable_recursion_pending();
                const std::uintmax_t count = fs::remove_all(entry.path(), error);
                if (!error && count > 0U) {
                    ++removed;
                }
                error.clear();
                continue;
            }
            if (!entry.is_regular_file(error)) {
                error.clear();
                continue;
            }
            if (name.find(".acmxvk-tmp-") == std::string::npos && name.find(".live-tmp-") == std::string::npos && name.find(".preview-tmp-") == std::string::npos) {
                continue;
            }
            if (fs::remove(entry.path(), error)) {
                ++removed;
            }
            error.clear();
        }
        return removed;
    }

} // namespace acmxvk
