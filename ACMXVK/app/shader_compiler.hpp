#ifndef ACMXVK_APP_SHADER_COMPILER_HPP
#define ACMXVK_APP_SHADER_COMPILER_HPP

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace acmxvk {

    enum class ShaderBuildStatus { Compiled, Copied, Current };

    class ShaderCompilationError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    [[nodiscard]] ShaderBuildStatus build_shader_file(const std::string &glslc_executable, const std::filesystem::path &source_root, const std::filesystem::path &source, const std::filesystem::path &destination, bool force);
    [[nodiscard]] std::size_t remove_shader_build_temporary_files(const std::filesystem::path &build_root);

} // namespace acmxvk

#endif
