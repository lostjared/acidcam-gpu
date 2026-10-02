#include <cstdlib>
#include <iostream>
#include <mxwrite.hpp>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <source_video_with_audio> <destination_video>\n";
        return EXIT_FAILURE;
    }
    transfer_audio(argv[1], argv[2]);
    return 0;
}
