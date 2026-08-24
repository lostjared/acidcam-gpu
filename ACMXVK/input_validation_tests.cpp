#include "input_validation.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

    void expectAccepted(std::string_view value, acmxvk::input::StringKind kind) {
        acmxvk::input::validate_string(value, kind, "test value");
    }

    void expectRejected(const std::function<void()> &operation) {
        try {
            operation();
        } catch (const std::runtime_error &) {
            return;
        }
        throw std::runtime_error("malformed input was unexpectedly accepted");
    }

} // namespace

int main() {
    using acmxvk::input::StringKind;
    using acmxvk::input::validate_string;

    expectAccepted("video files/input 01.mp4", StringKind::Path);
    expectAccepted("Watermark \xF0\x9F\x8C\x88", StringKind::DisplayText);
    expectAccepted("custom_uniform_1", StringKind::Identifier);
    expectAccepted("h264_nvenc", StringKind::Token);
    expectAccepted("preset=p7:rc=vbr_hq", StringKind::StructuredValue);
    expectAccepted("https://example.com/audio.ogg", StringKind::Url);

    expectRejected([] {
        validate_string(std::string("bad\0path", 8), StringKind::Path,
                        "test path");
    });
    expectRejected([] {
        validate_string("bad\npath", StringKind::Path, "test path");
    });
    expectRejected([] {
        validate_string("\xC0\xAF", StringKind::Path, "test path");
    });
    expectRejected([] {
        validate_string("bad-name", StringKind::Identifier, "test identifier");
    });
    expectRejected([] {
        validate_string("value;command", StringKind::StructuredValue,
                        "test value");
    });
    expectRejected([] {
        validate_string("javascript://example", StringKind::Url, "test URL");
    });
    expectRejected([] {
        validate_string("https://example.com/bad path", StringKind::Url,
                        "test URL");
    });
    expectRejected([] {
        validate_string("hidden\xE2\x80\xAEtext", StringKind::DisplayText,
                        "test text");
    });
    if (acmxvk::input::truncate_utf8("abc\xF0\x9F\x8C\x88xyz", 8) !=
        "abc...") {
        throw std::runtime_error("UTF-8 truncation split a codepoint");
    }

    std::istringstream valid_lines("\xEF\xBB\xBF"
                                   "first\r\nsecond\tvalue\n");
    std::string line;
    if (!acmxvk::input::read_bounded_line(valid_lines, line, "test input", 1,
                                          16) ||
        line != "first") {
        throw std::runtime_error("bounded line reader rejected a valid line");
    }
    if (!acmxvk::input::read_bounded_line(valid_lines, line, "test input", 2,
                                          16) ||
        line != "second\tvalue") {
        throw std::runtime_error(
            "bounded line reader rejected valid configuration whitespace");
    }
    expectRejected([] {
        std::istringstream long_line("123456789\n");
        std::string value;
        static_cast<void>(acmxvk::input::read_bounded_line(
            long_line, value, "test input", 1, 8));
    });

    std::cout << "ACMXVK input validation tests passed\n";
    return 0;
}
