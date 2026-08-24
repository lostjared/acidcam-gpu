#ifndef ACMXVK_INPUT_VALIDATION_HPP
#define ACMXVK_INPUT_VALIDATION_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <string>
#include <string_view>

namespace acmxvk::input {

    enum class StringKind {
        Argument,
        ConfigurationLine,
        Path,
        DisplayText,
        Identifier,
        Token,
        StructuredValue,
        Url,
    };

    constexpr std::size_t MAX_ARGUMENT_BYTES = 8192;
    constexpr std::size_t MAX_PATH_BYTES = 4096;
    constexpr std::size_t MAX_DISPLAY_TEXT_BYTES = 1024;
    constexpr std::size_t MAX_IDENTIFIER_BYTES = 64;
    constexpr std::size_t MAX_TOKEN_BYTES = 128;
    constexpr std::size_t MAX_STRUCTURED_VALUE_BYTES = 4096;
    constexpr std::size_t MAX_CONFIGURATION_LINE_BYTES = 4096;
    constexpr std::uintmax_t MAX_CONFIGURATION_FILE_BYTES = 4U * 1024U * 1024U;
    constexpr std::size_t MAX_SHADER_ENTRIES = 16384;
    constexpr std::size_t MAX_PLAYLIST_NODES = 1024;
    constexpr std::size_t MAX_PLAYLIST_ENTRIES = 16384;
    constexpr std::size_t MAX_MIDI_MAPPINGS = 4096;
    constexpr std::size_t MAX_AUDIO_PLAYLIST_ENTRIES = 4096;

    void validate_string(std::string_view value, StringKind kind,
                         std::string_view context, bool allow_empty = false);

    void validate_file_size(
        const std::filesystem::path &path, std::string_view context,
        std::uintmax_t maximum_bytes = MAX_CONFIGURATION_FILE_BYTES);

    void validate_text_file(
        const std::filesystem::path &path, std::string_view context,
        std::uintmax_t maximum_bytes = MAX_CONFIGURATION_FILE_BYTES,
        std::size_t maximum_line_bytes = MAX_CONFIGURATION_FILE_BYTES);

    void validate_spirv_file(const std::filesystem::path &path,
                             std::string_view context);

    [[nodiscard]] std::string truncate_utf8(std::string_view value,
                                            std::size_t maximum_bytes,
                                            std::string_view suffix = "...");

    [[nodiscard]] bool read_bounded_line(
        std::istream &input, std::string &line, std::string_view context,
        std::size_t line_number,
        std::size_t maximum_bytes = MAX_CONFIGURATION_LINE_BYTES);

} // namespace acmxvk::input

#endif
