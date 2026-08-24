#include "input_validation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace acmxvk::input {
    namespace {

        [[nodiscard]] std::size_t maximumLength(StringKind kind) {
            switch (kind) {
            case StringKind::Argument:
                return MAX_ARGUMENT_BYTES;
            case StringKind::ConfigurationLine:
                return MAX_CONFIGURATION_LINE_BYTES;
            case StringKind::Path:
            case StringKind::Url:
                return MAX_PATH_BYTES;
            case StringKind::DisplayText:
                return MAX_DISPLAY_TEXT_BYTES;
            case StringKind::Identifier:
                return MAX_IDENTIFIER_BYTES;
            case StringKind::Token:
                return MAX_TOKEN_BYTES;
            case StringKind::StructuredValue:
                return MAX_STRUCTURED_VALUE_BYTES;
            }
            return 0;
        }

        [[nodiscard]] bool isUnicodeNoncharacter(std::uint32_t codepoint) {
            return (codepoint >= 0xFDD0U && codepoint <= 0xFDEFU) ||
                   (codepoint & 0xFFFFU) == 0xFFFEU ||
                   (codepoint & 0xFFFFU) == 0xFFFFU;
        }

        [[nodiscard]] bool isUnsafeUnicodeControl(std::uint32_t codepoint) {
            return codepoint == 0x00ADU || codepoint == 0x061CU ||
                   (codepoint >= 0x200BU && codepoint <= 0x200FU) ||
                   codepoint == 0x2028U || codepoint == 0x2029U ||
                   (codepoint >= 0x202AU && codepoint <= 0x202EU) ||
                   (codepoint >= 0x2060U && codepoint <= 0x206FU) ||
                   codepoint == 0x3164U || codepoint == 0xFEFFU ||
                   isUnicodeNoncharacter(codepoint);
        }

        [[nodiscard]] std::uint32_t decodeCodepoint(std::string_view value,
                                                    std::size_t &offset,
                                                    std::string_view context) {
            const auto fail = [&]() -> std::uint32_t {
                throw std::runtime_error(std::string(context) +
                                         " contains malformed UTF-8");
            };

            const auto first = static_cast<unsigned char>(value[offset++]);
            if (first < 0x80U) {
                return first;
            }

            std::size_t continuation_count = 0;
            std::uint32_t codepoint = 0;
            std::uint32_t minimum = 0;
            if (first >= 0xC2U && first <= 0xDFU) {
                continuation_count = 1;
                codepoint = first & 0x1FU;
                minimum = 0x80U;
            } else if (first >= 0xE0U && first <= 0xEFU) {
                continuation_count = 2;
                codepoint = first & 0x0FU;
                minimum = 0x800U;
            } else if (first >= 0xF0U && first <= 0xF4U) {
                continuation_count = 3;
                codepoint = first & 0x07U;
                minimum = 0x10000U;
            } else {
                return fail();
            }

            if (continuation_count > value.size() - offset) {
                return fail();
            }
            for (std::size_t index = 0; index < continuation_count; ++index) {
                const auto next = static_cast<unsigned char>(value[offset++]);
                if ((next & 0xC0U) != 0x80U) {
                    return fail();
                }
                codepoint = (codepoint << 6U) | (next & 0x3FU);
            }
            if (codepoint < minimum || codepoint > 0x10FFFFU ||
                (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
                return fail();
            }
            return codepoint;
        }

        [[nodiscard]] bool isTokenCharacter(unsigned char character) {
            return std::isalnum(character) != 0 || character == '_' ||
                   character == '-' || character == '.' || character == '+';
        }

        [[nodiscard]] bool isStructuredCharacter(unsigned char character) {
            constexpr std::string_view PUNCTUATION = "_.,:=+\\-/@%{}[]() ";
            return character == '\t' || std::isalnum(character) != 0 ||
                   PUNCTUATION.find(static_cast<char>(character)) !=
                       std::string_view::npos;
        }

        void validateUrl(std::string_view value, std::string_view context) {
            const std::size_t separator = value.find("://");
            if (separator == std::string_view::npos || separator == 0 ||
                separator + 3 >= value.size()) {
                throw std::runtime_error(std::string(context) +
                                         " is not a valid URL");
            }
            const std::string_view scheme = value.substr(0, separator);
            if (std::isalpha(static_cast<unsigned char>(scheme.front())) == 0 ||
                !std::all_of(scheme.begin() + 1, scheme.end(), [](char value) {
                    const auto character = static_cast<unsigned char>(value);
                    return std::isalnum(character) != 0 || character == '+' ||
                           character == '-' || character == '.';
                })) {
                throw std::runtime_error(std::string(context) +
                                         " has an invalid URL scheme");
            }
            constexpr std::array<std::string_view, 5> ALLOWED_SCHEMES{
                "http", "https", "file", "rtsp", "rtmp"};
            std::string lowered(scheme);
            std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            if (std::find(ALLOWED_SCHEMES.begin(), ALLOWED_SCHEMES.end(),
                          lowered) == ALLOWED_SCHEMES.end()) {
                throw std::runtime_error(std::string(context) +
                                         " uses an unsupported URL scheme");
            }
            const std::string_view payload = value.substr(separator + 3);
            if (std::any_of(payload.begin(), payload.end(), [](char value) {
                    const auto character = static_cast<unsigned char>(value);
                    return character == ' ' || character == '\\' ||
                           character == '<' || character == '>' ||
                           character == '"' || character == '{' ||
                           character == '}';
                })) {
                throw std::runtime_error(std::string(context) +
                                         " contains an invalid URL character");
            }
            if (lowered != "file") {
                const std::size_t authority_end = payload.find_first_of("/?#");
                if (authority_end == 0) {
                    throw std::runtime_error(std::string(context) +
                                             " is missing a URL host");
                }
            }
            for (std::size_t index = 0; index < payload.size(); ++index) {
                if (payload[index] != '%') {
                    continue;
                }
                if (index + 2 >= payload.size() ||
                    std::isxdigit(static_cast<unsigned char>(
                        payload[index + 1])) == 0 ||
                    std::isxdigit(static_cast<unsigned char>(
                        payload[index + 2])) == 0) {
                    throw std::runtime_error(std::string(context) +
                                             " contains an invalid URL escape");
                }
                index += 2;
            }
        }

    } // namespace

    void validate_string(std::string_view value, StringKind kind,
                         std::string_view context, bool allow_empty) {
        if (value.empty()) {
            if (allow_empty) {
                return;
            }
            throw std::runtime_error(std::string(context) + " must not be empty");
        }
        if (value.size() > maximumLength(kind)) {
            throw std::runtime_error(std::string(context) + " is too long");
        }

        std::size_t offset = 0;
        while (offset < value.size()) {
            const std::uint32_t codepoint =
                decodeCodepoint(value, offset, context);
            const bool allowed_tab =
                (kind == StringKind::ConfigurationLine ||
                 kind == StringKind::StructuredValue) &&
                codepoint == '\t';
            if ((!allowed_tab && codepoint < 0x20U) || codepoint == 0x7FU ||
                (codepoint >= 0x80U && codepoint < 0xA0U) ||
                isUnsafeUnicodeControl(codepoint)) {
                throw std::runtime_error(std::string(context) +
                                         " contains a disallowed control character");
            }
            if ((kind == StringKind::Identifier || kind == StringKind::Token ||
                 kind == StringKind::StructuredValue) &&
                codepoint > 0x7FU) {
                throw std::runtime_error(std::string(context) +
                                         " must contain ASCII characters only");
            }
        }

        if (kind == StringKind::Identifier) {
            const auto first = static_cast<unsigned char>(value.front());
            if ((std::isalpha(first) == 0 && first != '_') ||
                !std::all_of(value.begin() + 1, value.end(), [](char value) {
                    const auto character = static_cast<unsigned char>(value);
                    return std::isalnum(character) != 0 || character == '_';
                })) {
                throw std::runtime_error(std::string(context) +
                                         " contains an invalid identifier");
            }
        } else if (kind == StringKind::Token) {
            if (!std::all_of(value.begin(), value.end(), [](char value) {
                    return isTokenCharacter(
                        static_cast<unsigned char>(value));
                })) {
                throw std::runtime_error(std::string(context) +
                                         " contains a disallowed token character");
            }
        } else if (kind == StringKind::StructuredValue) {
            if (!std::all_of(value.begin(), value.end(), [](char value) {
                    return isStructuredCharacter(
                        static_cast<unsigned char>(value));
                })) {
                throw std::runtime_error(
                    std::string(context) +
                    " contains a disallowed structured-value character");
            }
        } else if (kind == StringKind::Url) {
            validateUrl(value, context);
        }
    }

    void validate_file_size(const std::filesystem::path &path,
                            std::string_view context,
                            std::uintmax_t maximum_bytes) {
        std::error_code error;
        const std::uintmax_t size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error("unable to inspect " +
                                     std::string(context));
        }
        if (size > maximum_bytes) {
            throw std::runtime_error(std::string(context) +
                                     " exceeds the allowed file size");
        }
    }

    void validate_spirv_file(const std::filesystem::path &path,
                             std::string_view context) {
        constexpr std::uintmax_t MAX_SHADER_BYTES = 64U * 1024U * 1024U;
        std::error_code error;
        const std::uintmax_t size = std::filesystem::file_size(path, error);
        if (error || size < 20U || size > MAX_SHADER_BYTES || size % 4U != 0U) {
            throw std::runtime_error(std::string(context) +
                                     " has an invalid SPIR-V file size");
        }

        std::ifstream shader(path, std::ios::binary);
        std::array<unsigned char, 4> magic{};
        if (!shader.read(reinterpret_cast<char *>(magic.data()),
                         static_cast<std::streamsize>(magic.size()))) {
            throw std::runtime_error("unable to read " + std::string(context));
        }
        const std::uint32_t word = static_cast<std::uint32_t>(magic[0]) |
                                   (static_cast<std::uint32_t>(magic[1]) << 8U) |
                                   (static_cast<std::uint32_t>(magic[2]) << 16U) |
                                   (static_cast<std::uint32_t>(magic[3]) << 24U);
        if (word != 0x07230203U) {
            throw std::runtime_error(std::string(context) +
                                     " does not contain SPIR-V bytecode");
        }
    }

    void validate_text_file(const std::filesystem::path &path,
                            std::string_view context,
                            std::uintmax_t maximum_bytes,
                            std::size_t maximum_line_bytes) {
        validate_file_size(path, context, maximum_bytes);
        std::ifstream text_file(path, std::ios::binary);
        if (!text_file) {
            throw std::runtime_error("unable to open " + std::string(context));
        }
        std::string line;
        std::size_t line_number = 1;
        while (read_bounded_line(text_file, line, context, line_number++,
                                 maximum_line_bytes)) {
        }
    }

    std::string truncate_utf8(std::string_view value, std::size_t maximum_bytes,
                              std::string_view suffix) {
        if (value.size() <= maximum_bytes) {
            return std::string(value);
        }
        if (suffix.size() >= maximum_bytes) {
            return std::string(suffix.substr(0, maximum_bytes));
        }

        std::size_t end = maximum_bytes - suffix.size();
        while (end > 0 && end < value.size() &&
               (static_cast<unsigned char>(value[end]) & 0xC0U) == 0x80U) {
            --end;
        }
        std::string result(value.substr(0, end));
        result.append(suffix);
        return result;
    }

    bool read_bounded_line(std::istream &input, std::string &line,
                           std::string_view context, std::size_t line_number,
                           std::size_t maximum_bytes) {
        line.clear();
        char character = 0;
        bool read_anything = false;
        while (input.get(character)) {
            read_anything = true;
            if (character == '\n') {
                break;
            }
            if (line.size() >= maximum_bytes) {
                throw std::runtime_error(std::string(context) + " line " +
                                         std::to_string(line_number) +
                                         " exceeds the allowed length");
            }
            line.push_back(character);
        }
        if (!read_anything) {
            return false;
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line_number == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEFU &&
            static_cast<unsigned char>(line[1]) == 0xBBU &&
            static_cast<unsigned char>(line[2]) == 0xBFU) {
            line.erase(0, 3);
        }
        validate_string(line, StringKind::ConfigurationLine, context, true);
        return true;
    }

} // namespace acmxvk::input
