#ifndef ACMXVK_APP_SHADER_LIBRARY_HPP
#define ACMXVK_APP_SHADER_LIBRARY_HPP

#include "options.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace acmxvk {
    struct ShaderManifest {
        struct CustomUniform {
            std::string name;
            std::size_t slot = 0;
            double minimum = 0.0;
            double maximum = 1.0;
            double step = 0.01;
            double value = 0.0;
        };

        fs::path path;
        std::vector<std::string> entries;
        std::vector<CustomUniform> custom_uniforms;
    };

    [[nodiscard]] bool isValidCustomUniformName(const std::string &name);
    [[nodiscard]] ShaderManifest loadShaderManifest(const fs::path &directory);
    [[nodiscard]] fs::path resolveShaderManifestEntry(const fs::path &directory,
                                                      std::string entry);
    [[nodiscard]] int buildShaderLibrary(const Options &options);

} // namespace acmxvk

#endif
