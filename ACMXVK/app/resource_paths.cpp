#include "resource_paths.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#endif

#ifndef ACMXVK_BUILD_RESOURCE_DIRECTORY
#define ACMXVK_BUILD_RESOURCE_DIRECTORY "."
#endif

#ifndef ACMXVK_INSTALL_RESOURCE_DIRECTORY
#define ACMXVK_INSTALL_RESOURCE_DIRECTORY "."
#endif

#ifndef ACMXVK_BUILD_SPRITE_VERTEX_SHADER
#define ACMXVK_BUILD_SPRITE_VERTEX_SHADER "sprite.vert.spv"
#endif

#ifndef ACMXVK_INSTALL_SPRITE_VERTEX_SHADER
#define ACMXVK_INSTALL_SPRITE_VERTEX_SHADER "sprite.vert.spv"
#endif

#ifndef ACMXVK_BUILD_ECHO_CACHE_SHADER
#define ACMXVK_BUILD_ECHO_CACHE_SHADER "echo_cache.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_ECHO_CACHE_SHADER
#define ACMXVK_INSTALL_ECHO_CACHE_SHADER "echo_cache.frag.spv"
#endif

#ifndef ACMXVK_BUILD_FLIP_SHADER
#define ACMXVK_BUILD_FLIP_SHADER "flip.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_FLIP_SHADER
#define ACMXVK_INSTALL_FLIP_SHADER "flip.frag.spv"
#endif

#ifndef ACMXVK_BUILD_PASSTHROUGH_SHADER
#define ACMXVK_BUILD_PASSTHROUGH_SHADER "passthrough.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_PASSTHROUGH_SHADER
#define ACMXVK_INSTALL_PASSTHROUGH_SHADER "passthrough.frag.spv"
#endif

#ifndef ACMXVK_BUILD_HDR_TRANSFER_DIRECTORY
#define ACMXVK_BUILD_HDR_TRANSFER_DIRECTORY "."
#endif

#ifndef ACMXVK_INSTALL_HDR_TRANSFER_DIRECTORY
#define ACMXVK_INSTALL_HDR_TRANSFER_DIRECTORY "."
#endif

#ifndef ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER
#define ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER "human_composite.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER
#define ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER "human_composite.frag.spv"
#endif

#ifndef ACMXVK_BUILD_MODEL_VERTEX_SHADER
#define ACMXVK_BUILD_MODEL_VERTEX_SHADER "model.vert.spv"
#endif

#ifndef ACMXVK_INSTALL_MODEL_VERTEX_SHADER
#define ACMXVK_INSTALL_MODEL_VERTEX_SHADER "model.vert.spv"
#endif

#ifndef ACMXVK_BUILD_MODEL_FRAGMENT_SHADER
#define ACMXVK_BUILD_MODEL_FRAGMENT_SHADER "model.frag.spv"
#endif

#ifndef ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER
#define ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER "model.frag.spv"
#endif

#ifndef ACMXVK_BUILD_DEFAULT_MODEL
#define ACMXVK_BUILD_DEFAULT_MODEL "cube.obj"
#endif

#ifndef ACMXVK_INSTALL_DEFAULT_MODEL
#define ACMXVK_INSTALL_DEFAULT_MODEL "cube.obj"
#endif

#ifndef ACMXVK_BUILD_OVERLAY_FONT
#define ACMXVK_BUILD_OVERLAY_FONT "font.ttf"
#endif

#ifndef ACMXVK_INSTALL_OVERLAY_FONT
#define ACMXVK_INSTALL_OVERLAY_FONT "font.ttf"
#endif

#ifndef ACMXVK_BUILD_CROSSFADE_DIRECTORY
#define ACMXVK_BUILD_CROSSFADE_DIRECTORY "shaders/xfade"
#endif

#ifndef ACMXVK_INSTALL_CROSSFADE_DIRECTORY
#define ACMXVK_INSTALL_CROSSFADE_DIRECTORY "shaders/xfade"
#endif

namespace acmxvk {
    namespace {
#ifdef _WIN32
        [[nodiscard]] fs::path executable_resource_directory() {
            std::vector<wchar_t> buffer(MAX_PATH);
            for (;;) {
                const DWORD length = GetModuleFileNameW(
                    nullptr, buffer.data(),
                    static_cast<DWORD>(buffer.size()));
                if (length == 0U) {
                    return {};
                }
                if (length < buffer.size() - 1U) {
                    const std::wstring executable_name(buffer.data(), length);
                    const fs::path executable(executable_name);
                    return executable.parent_path().parent_path() / "share" /
                           "acmxvk";
                }
                if (buffer.size() >= 32768U) {
                    return {};
                }
                buffer.resize(buffer.size() * 2U);
            }
        }
#endif

        [[nodiscard]] fs::path
        resolve_resource(const Options &options, const fs::path &relative_path,
                         const fs::path &installed_path,
                         const fs::path &built_path) {
            const fs::path resource = find_resource(options, relative_path);
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(installed_path)) {
                return installed_path;
            }
            return built_path;
        }
    } // namespace

    std::vector<fs::path> resource_directories(const Options &options) {
        std::vector<fs::path> directories;
        const auto append = [&](const fs::path &directory) {
            if (directory.empty()) {
                return;
            }
            const fs::path normalized =
                fs::absolute(directory).lexically_normal();
            if (std::find(directories.begin(), directories.end(), normalized) ==
                directories.end()) {
                directories.push_back(normalized);
            }
        };
        append(options.resource_directory);
        append(ACMXVK_INSTALL_RESOURCE_DIRECTORY);
        append(ACMXVK_BUILD_RESOURCE_DIRECTORY);
#ifdef _WIN32
        append(executable_resource_directory());
#endif
        append(fs::current_path());
        return directories;
    }

    fs::path find_resource(const Options &options,
                           const fs::path &relative_path) {
        if (relative_path.empty() || relative_path.is_absolute()) {
            return {};
        }
        const fs::path normalized_relative = relative_path.lexically_normal();
        const std::string relative_text = normalized_relative.generic_string();
        if (relative_text == ".." || relative_text.starts_with("../") ||
            relative_text.find("/../") != std::string::npos) {
            return {};
        }
        for (const fs::path &directory : resource_directories(options)) {
            const fs::path candidate =
                (directory / normalized_relative).lexically_normal();
            if (fs::is_regular_file(candidate)) {
                return candidate;
            }
        }
        return {};
    }

    fs::path sprite_vertex_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/sprite.vert.spv",
                                ACMXVK_INSTALL_SPRITE_VERTEX_SHADER,
                                ACMXVK_BUILD_SPRITE_VERTEX_SHADER);
    }

    fs::path echo_cache_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/echo_cache.frag.spv",
                                ACMXVK_INSTALL_ECHO_CACHE_SHADER,
                                ACMXVK_BUILD_ECHO_CACHE_SHADER);
    }

    fs::path flip_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/flip.frag.spv",
                                ACMXVK_INSTALL_FLIP_SHADER,
                                ACMXVK_BUILD_FLIP_SHADER);
    }

    fs::path passthrough_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/passthrough.frag.spv",
                                ACMXVK_INSTALL_PASSTHROUGH_SHADER,
                                ACMXVK_BUILD_PASSTHROUGH_SHADER);
    }

    fs::path hdr_transfer_shader_path(const Options &options, bool hlg,
                                      bool encode) {
        const std::string filename =
            std::string("hdr_") + (hlg ? "hlg_" : "pq_") +
            (encode ? "encode.frag.spv" : "decode.frag.spv");
        return resolve_resource(
            options, fs::path("shaders") / filename,
            fs::path(ACMXVK_INSTALL_HDR_TRANSFER_DIRECTORY) / filename,
            fs::path(ACMXVK_BUILD_HDR_TRANSFER_DIRECTORY) / filename);
    }

    fs::path hdr_preview_shader_path(const Options &options, bool hlg) {
        const std::string filename =
            std::string("hdr_preview_") + (hlg ? "hlg" : "pq") +
            ".frag.spv";
        return resolve_resource(
            options, fs::path("shaders") / filename,
            fs::path(ACMXVK_INSTALL_HDR_TRANSFER_DIRECTORY) / filename,
            fs::path(ACMXVK_BUILD_HDR_TRANSFER_DIRECTORY) / filename);
    }

    fs::path human_composite_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/human_composite.frag.spv",
                                ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER,
                                ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER);
    }

    fs::path crossfade_shader_path(const Options &options,
                                   std::size_t shader_index) {
        if (shader_index >= CROSSFADE_NAMES.size()) {
            throw std::out_of_range("crossfade shader index is out of range");
        }
        const std::string filename =
            std::string(CROSSFADE_NAMES[shader_index]) + ".frag.spv";
        const fs::path resource =
            find_resource(options, fs::path("shaders/xfade") / filename);
        if (!resource.empty()) {
            return resource;
        }
        const fs::path installed =
            fs::path(ACMXVK_INSTALL_CROSSFADE_DIRECTORY) / filename;
        if (fs::is_regular_file(installed)) {
            return installed;
        }
        const fs::path built =
            fs::path(ACMXVK_BUILD_CROSSFADE_DIRECTORY) / filename;
        if (fs::is_regular_file(built)) {
            return built;
        }
        throw std::runtime_error("crossfade shader was not found: " +
                                 filename);
    }

    fs::path model_vertex_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/model.vert.spv",
                                ACMXVK_INSTALL_MODEL_VERTEX_SHADER,
                                ACMXVK_BUILD_MODEL_VERTEX_SHADER);
    }

    fs::path model_fragment_shader_path(const Options &options) {
        return resolve_resource(options, "shaders/model.frag.spv",
                                ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER,
                                ACMXVK_BUILD_MODEL_FRAGMENT_SHADER);
    }

    fs::path default_model_path(const Options &options) {
        return resolve_resource(options, "models/cube.obj",
                                ACMXVK_INSTALL_DEFAULT_MODEL,
                                ACMXVK_BUILD_DEFAULT_MODEL);
    }

    fs::path overlay_font_path(const Options &options) {
        return resolve_resource(options, "data/font.ttf",
                                ACMXVK_INSTALL_OVERLAY_FONT,
                                ACMXVK_BUILD_OVERLAY_FONT);
    }
} // namespace acmxvk
