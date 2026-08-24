#include "midi.hpp"

#include "input_validation.hpp"

#include <rtmidi/RtMidi.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace acmxvk::midi {
    std::vector<MidiMapping> load_mapping_file(const std::string &filename) {
        input::validate_string(filename, input::StringKind::Path,
                               "MIDI map path");
        input::validate_file_size(filename, "MIDI map file");
        std::ifstream mapping_input(filename);
        if (!mapping_input) {
            throw std::runtime_error("could not open MIDI map file: " + filename);
        }

        std::vector<MidiMapping> mappings;
        std::string line;
        std::size_t line_number = 0;
        while (input::read_bounded_line(mapping_input, line, "MIDI map file",
                                        line_number + 1)) {
            ++line_number;
            const std::size_t first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos || line[first] == '#') {
                continue;
            }
            if (mappings.size() >= input::MAX_MIDI_MAPPINGS) {
                throw std::runtime_error(
                    "MIDI map file contains too many mappings");
            }
            input::validate_string(line, input::StringKind::StructuredValue,
                                   "MIDI map entry");

            std::istringstream stream(line);
            std::string action_pair;
            char open_brace = 0;
            char close_brace = 0;
            int status = 0;
            int data1 = 0;
            int data2 = 0;
            if (!(stream >> action_pair >> open_brace >> status >> data1 >> data2 >>
                  close_brace) ||
                open_brace != '{' || close_brace != '}') {
                throw std::runtime_error(
                    "invalid MIDI map entry at " + filename + ':' +
                    std::to_string(line_number));
            }
            stream >> std::ws;
            if (!stream.eof()) {
                throw std::runtime_error(
                    "unexpected text in MIDI map at " + filename + ':' +
                    std::to_string(line_number));
            }

            const std::size_t colon = action_pair.find(':');
            if (colon == std::string::npos || colon == 0 ||
                colon + 1 >= action_pair.size() ||
                action_pair.find(':', colon + 1) != std::string::npos) {
                throw std::runtime_error(
                    "invalid MIDI action pair at " + filename + ':' +
                    std::to_string(line_number));
            }

            std::size_t primary_parsed = 0;
            std::size_t secondary_parsed = 0;
            int primary = 0;
            int secondary = 0;
            try {
                primary = std::stoi(action_pair.substr(0, colon),
                                    &primary_parsed);
                secondary = std::stoi(action_pair.substr(colon + 1),
                                      &secondary_parsed);
            } catch (const std::exception &) {
                throw std::runtime_error(
                    "invalid MIDI action code at " + filename + ':' +
                    std::to_string(line_number));
            }
            if (primary_parsed != colon ||
                secondary_parsed != action_pair.size() - colon - 1 ||
                primary <= 0 || primary > 65535 || secondary < 0 ||
                secondary > 65535 || status < 0 || status > 255 ||
                data1 < 0 || data1 > 255 || data2 < 0 || data2 > 255) {
                throw std::runtime_error(
                    "MIDI map value outside its valid range at " + filename +
                    ':' + std::to_string(line_number));
            }

            mappings.push_back(
                {primary, secondary, static_cast<unsigned char>(status),
                 static_cast<unsigned char>(data1),
                 static_cast<unsigned char>(data2)});
        }
        return mappings;
    }

    class MidiInput::Impl {
      public:
        static constexpr std::size_t MAX_PENDING_MESSAGES = 256;
        static constexpr std::size_t MAX_MESSAGE_BYTES = 1024;

        ~Impl() { close(); }

        bool open(int requested_port) {
            close();
            if (requested_port < 0) {
                std::cerr << "acmxvk: MIDI port must be non-negative\n";
                return false;
            }

            try {
                input = std::make_unique<RtMidiIn>();
                const unsigned int port_count = input->getPortCount();
                const auto port = static_cast<unsigned int>(requested_port);
                if (port_count == 0) {
                    std::cerr << "acmxvk: no MIDI input ports found\n";
                    input.reset();
                    return false;
                }
                if (port >= port_count) {
                    std::cerr << "acmxvk: MIDI input port " << port
                              << " is outside the available range 0.."
                              << (port_count - 1) << '\n';
                    input.reset();
                    return false;
                }

                port_name = input->getPortName(port);
                acmxvk::input::validate_string(
                    port_name, acmxvk::input::StringKind::DisplayText,
                    "MIDI port name");
                input->ignoreTypes(false, false, false);
                input->setCallback(&Impl::messageCallback, this);
                input->openPort(port, "ACMXVK MIDI Input");
                open_port = requested_port;
                std::cout << "acmxvk: MIDI input " << open_port << ": "
                          << port_name << '\n';
                return input->isPortOpen();
            } catch (const RtMidiError &error) {
                std::cerr << "acmxvk: MIDI input error: " << error.getMessage()
                          << '\n';
                close();
                return false;
            }
        }

        void close() {
            if (input != nullptr) {
                try {
                    input->cancelCallback();
                    if (input->isPortOpen()) {
                        input->closePort();
                    }
                } catch (const RtMidiError &error) {
                    std::cerr << "acmxvk: error closing MIDI input: "
                              << error.getMessage() << '\n';
                }
                input.reset();
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                pending_messages.clear();
                message_sequence = 0;
                dropped_messages = 0;
            }
            open_port = -1;
            port_name.clear();
        }

        [[nodiscard]] bool is_open() const {
            return input != nullptr && input->isPortOpen();
        }

        [[nodiscard]] std::vector<MidiMessage> pollMessages() {
            std::lock_guard<std::mutex> lock(queue_mutex);
            std::vector<MidiMessage> messages;
            messages.reserve(pending_messages.size());
            while (!pending_messages.empty()) {
                messages.push_back(std::move(pending_messages.front()));
                pending_messages.pop_front();
            }
            return messages;
        }

        [[nodiscard]] std::uint64_t droppedMessageCount() const {
            std::lock_guard<std::mutex> lock(queue_mutex);
            return dropped_messages;
        }

        static void messageCallback(double delta_seconds,
                                    std::vector<unsigned char> *bytes,
                                    void *user_data) {
            if (user_data == nullptr || bytes == nullptr || bytes->empty()) {
                return;
            }
            static_cast<Impl *>(user_data)->enqueue(delta_seconds, *bytes);
        }

        void enqueue(double delta_seconds,
                     const std::vector<unsigned char> &bytes) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (bytes.size() > MAX_MESSAGE_BYTES) {
                ++dropped_messages;
                return;
            }
            if (pending_messages.size() >= MAX_PENDING_MESSAGES) {
                pending_messages.pop_front();
                ++dropped_messages;
            }
            pending_messages.push_back(
                MidiMessage{std::isfinite(delta_seconds)
                                ? std::max(delta_seconds, 0.0)
                                : 0.0,
                            bytes, ++message_sequence});
        }

        std::unique_ptr<RtMidiIn> input;
        mutable std::mutex queue_mutex;
        std::deque<MidiMessage> pending_messages;
        std::string port_name;
        std::uint64_t message_sequence = 0;
        std::uint64_t dropped_messages = 0;
        int open_port = -1;
    };

    MidiInput::MidiInput() : impl(std::make_unique<Impl>()) {}
    MidiInput::~MidiInput() = default;

    bool MidiInput::open(int port) {
        return impl->open(port);
    }

    void MidiInput::close() {
        impl->close();
    }

    bool MidiInput::is_open() const {
        return impl->is_open();
    }

    std::vector<MidiMessage> MidiInput::poll_messages() {
        return impl->pollMessages();
    }

    std::uint64_t MidiInput::dropped_message_count() const {
        return impl->droppedMessageCount();
    }

    void MidiInput::list_ports(std::ostream &output) {
        try {
            RtMidiIn input;
            const unsigned int port_count = input.getPortCount();
            output << "acmxvk: found " << port_count << " MIDI input port(s)\n";
            for (unsigned int port = 0; port < port_count; ++port) {
                const std::string name = input.getPortName(port);
                acmxvk::input::validate_string(
                    name, acmxvk::input::StringKind::DisplayText,
                    "MIDI port name");
                output << "  " << port << ": " << name << '\n';
            }
        } catch (const RtMidiError &error) {
            throw std::runtime_error(std::string("could not enumerate MIDI inputs: ") +
                                     error.getMessage());
        }
    }

} // namespace acmxvk::midi
