#ifndef ACMXVK_MIDI_HPP
#define ACMXVK_MIDI_HPP

#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

namespace acmxvk::midi {

    struct MidiMessage {
        double delta_seconds = 0.0;
        std::vector<unsigned char> bytes;
        std::uint64_t sequence = 0;
    };

    class MidiInput {
      public:
        MidiInput();
        ~MidiInput();

        MidiInput(const MidiInput &) = delete;
        MidiInput &operator=(const MidiInput &) = delete;

        bool open(int port = 0);
        void close();
        [[nodiscard]] bool is_open() const;
        [[nodiscard]] std::vector<MidiMessage> poll_messages();
        [[nodiscard]] std::uint64_t dropped_message_count() const;

        static void list_ports(std::ostream &output);

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace acmxvk::midi

#endif
