#include "midi.hpp"

#include <rtmidi/RtMidi.h>

#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace acmxvk::midi {
    class MidiInput::Impl {
      public:
        static constexpr std::size_t MAX_PENDING_MESSAGES = 256;

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
            if (pending_messages.size() >= MAX_PENDING_MESSAGES) {
                pending_messages.pop_front();
                ++dropped_messages;
            }
            pending_messages.push_back(
                MidiMessage{delta_seconds, bytes, ++message_sequence});
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
                output << "  " << port << ": " << input.getPortName(port) << '\n';
            }
        } catch (const RtMidiError &error) {
            throw std::runtime_error(std::string("could not enumerate MIDI inputs: ") +
                                     error.getMessage());
        }
    }

} // namespace acmxvk::midi
