#ifndef ACMXVK_APP_MEDIA_UTILS_HPP
#define ACMXVK_APP_MEDIA_UTILS_HPP

#include "options.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace acmxvk {

    struct VideoHdrInfo {
        bool valid = false;
        bool hdr = false;
        int bit_depth = 0;
        int color_primaries = 0;
        int color_transfer = 0;
        int color_space = 0;
        int color_range = 0;
        std::vector<std::uint8_t> mastering_display;
        std::vector<std::uint8_t> content_light;
    };

    [[nodiscard]] cv::Mat loadRgbaImage(const std::string &filename);
    [[nodiscard]] double probeVideoDuration(const std::string &filename);
    [[nodiscard]] VideoHdrInfo probeVideoHdrInfo(const std::string &filename);
    void printVideoHdrInfo(const VideoHdrInfo &info, std::ostream &output);
    void rotateFrame(cv::Mat &frame, FrameRotation rotation);
    [[nodiscard]] bool rotationSwapsDimensions(FrameRotation rotation);
#ifdef ACMXVK_WITH_MXVK_CUDA
    void select_cuda_device(int device_index);
    void list_cuda_devices(std::ostream &output);
#endif

} // namespace acmxvk

#endif
