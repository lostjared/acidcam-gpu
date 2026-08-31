#ifndef ACMXVK_APP_MEDIA_UTILS_HPP
#define ACMXVK_APP_MEDIA_UTILS_HPP

#include "options.hpp"

#include <opencv2/core.hpp>

#include <iosfwd>
#include <string>

namespace acmxvk {

    [[nodiscard]] cv::Mat loadRgbaImage(const std::string &filename);
    [[nodiscard]] double probeVideoDuration(const std::string &filename);
    void rotateFrame(cv::Mat &frame, FrameRotation rotation);
    [[nodiscard]] bool rotationSwapsDimensions(FrameRotation rotation);
#ifdef ACMXVK_WITH_MXVK_CUDA
    void select_cuda_device(int device_index);
    void list_cuda_devices(std::ostream &output);
#endif

} // namespace acmxvk

#endif
