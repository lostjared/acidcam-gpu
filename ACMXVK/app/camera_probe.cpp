#include "camera_probe.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace acmxvk {
    namespace fs = std::filesystem;
    namespace {
#ifdef __linux__
        [[nodiscard]] std::string sanitizeDeviceText(std::string text) {
            for (char &character : text) {
                const unsigned char value =
                    static_cast<unsigned char>(character);
                if (value < 0x20U || value == 0x7FU) {
                    character = ' ';
                }
            }
            return text;
        }

        template <std::size_t Size>
        [[nodiscard]] std::string v4l2Text(const __u8 (&value)[Size]) {
            const char *text = reinterpret_cast<const char *>(value);
            return sanitizeDeviceText(
                std::string(text, ::strnlen(text, Size)));
        }

        [[nodiscard]] int cameraIoctl(int descriptor, unsigned long request,
                                      void *argument) {
            int result = -1;
            do {
                result = ioctl(descriptor, request, argument);
            } while (result < 0 && errno == EINTR);
            return result;
        }

        void appendFrameRate(std::vector<double> &frame_rates,
                             double frame_rate) {
            if (!std::isfinite(frame_rate) || frame_rate <= 0.0) {
                return;
            }
            const auto existing = std::find_if(
                frame_rates.begin(), frame_rates.end(),
                [frame_rate](double value) {
                    return std::abs(value - frame_rate) < 0.05;
                });
            if (existing == frame_rates.end()) {
                frame_rates.push_back(frame_rate);
            }
        }

        [[nodiscard]] double frameRate(const v4l2_fract &interval) {
            if (interval.numerator == 0U) {
                return 0.0;
            }
            return static_cast<double>(interval.denominator) /
                   static_cast<double>(interval.numerator);
        }

        [[nodiscard]] std::string fourccName(std::uint32_t fourcc) {
            std::array<char, 5> name{
                static_cast<char>(fourcc & 0xFFU),
                static_cast<char>((fourcc >> 8U) & 0xFFU),
                static_cast<char>((fourcc >> 16U) & 0xFFU),
                static_cast<char>((fourcc >> 24U) & 0xFFU), '\0'};
            for (std::size_t index = 0; index < 4U; ++index) {
                const unsigned char character =
                    static_cast<unsigned char>(name[index]);
                if (character < 0x20U || character > 0x7EU) {
                    name[index] = '?';
                }
            }
            return name.data();
        }

        [[nodiscard]] bool enumerateCaptureType(
            int descriptor, v4l2_buf_type capture_type, bool loopback_device,
            double current_fps, std::ostream &output) {
            bool found_format = false;
            v4l2_fmtdesc format{};
            format.type = capture_type;
            for (format.index = 0U;
                 cameraIoctl(descriptor, VIDIOC_ENUM_FMT, &format) == 0;
                 ++format.index) {
                found_format = true;
                output << "\n  Format: " << fourccName(format.pixelformat)
                       << " (" << v4l2Text(format.description) << ")\n";

                v4l2_frmsizeenum frame_size{};
                frame_size.pixel_format = format.pixelformat;
                for (frame_size.index = 0U;
                     cameraIoctl(descriptor, VIDIOC_ENUM_FRAMESIZES,
                                 &frame_size) == 0;
                     ++frame_size.index) {
                    if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                        std::vector<double> frame_rates;
                        v4l2_frmivalenum interval{};
                        interval.pixel_format = format.pixelformat;
                        interval.width = frame_size.discrete.width;
                        interval.height = frame_size.discrete.height;
                        for (interval.index = 0U;
                             cameraIoctl(descriptor,
                                         VIDIOC_ENUM_FRAMEINTERVALS,
                                         &interval) == 0;
                             ++interval.index) {
                            if (interval.type ==
                                V4L2_FRMIVAL_TYPE_DISCRETE) {
                                appendFrameRate(frame_rates,
                                                frameRate(interval.discrete));
                            } else if (interval.type ==
                                           V4L2_FRMIVAL_TYPE_STEPWISE ||
                                       interval.type ==
                                           V4L2_FRMIVAL_TYPE_CONTINUOUS) {
                                appendFrameRate(
                                    frame_rates,
                                    frameRate(interval.stepwise.min));
                                appendFrameRate(
                                    frame_rates,
                                    frameRate(interval.stepwise.max));
                                break;
                            }
                        }

                        if (frame_rates.empty()) {
                            appendFrameRate(frame_rates, current_fps);
                        }
                        if (loopback_device) {
                            constexpr std::array<double, 9>
                                LOOPBACK_FRAME_RATES{24.0, 25.0, 30.0,
                                                     50.0, 60.0, 90.0,
                                                     120.0, 144.0, 240.0};
                            for (double frame_rate : LOOPBACK_FRAME_RATES) {
                                appendFrameRate(frame_rates, frame_rate);
                            }
                        }
                        std::sort(frame_rates.begin(), frame_rates.end(),
                                  std::greater<double>());

                        output << "    " << frame_size.discrete.width << 'x'
                               << frame_size.discrete.height;
                        bool first = true;
                        for (double frame_rate : frame_rates) {
                            output << (first ? " @ " : ", ") << std::fixed
                                   << std::setprecision(1) << frame_rate
                                   << " fps";
                            first = false;
                        }
                        output << '\n';
                    } else if (frame_size.type ==
                                   V4L2_FRMSIZE_TYPE_STEPWISE ||
                               frame_size.type ==
                                   V4L2_FRMSIZE_TYPE_CONTINUOUS) {
                        output << "    " << frame_size.stepwise.min_width
                               << 'x' << frame_size.stepwise.min_height
                               << " to " << frame_size.stepwise.max_width
                               << 'x' << frame_size.stepwise.max_height
                               << " (step " << frame_size.stepwise.step_width
                               << 'x' << frame_size.stepwise.step_height
                               << ")\n";
                        break;
                    }
                }
            }
            return found_format;
        }

        [[nodiscard]] bool primaryVideoNode(int device_index) {
            std::ifstream index_file(
                "/sys/class/video4linux/video" +
                std::to_string(device_index) + "/index");
            int stream_index = 0;
            return !(index_file >> stream_index) || stream_index == 0;
        }
#endif
    } // namespace

    [[nodiscard]] bool listCameraDevices(std::ostream &output,
                                         std::ostream &error) {
#ifdef __linux__
        std::vector<int> device_indices;
        std::error_code directory_error;
        for (const fs::directory_entry &entry :
             fs::directory_iterator("/dev", directory_error)) {
            const std::string name = entry.path().filename().string();
            if (!name.starts_with("video") || name.size() <= 5U ||
                !std::all_of(name.begin() + 5, name.end(), [](char character) {
                    return character >= '0' && character <= '9';
                })) {
                continue;
            }
            try {
                const int index = std::stoi(name.substr(5));
                if (index >= 0 && primaryVideoNode(index)) {
                    device_indices.push_back(index);
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        if (directory_error) {
            error << "acmxvk: cannot enumerate /dev video devices: "
                  << directory_error.message() << '\n';
            return false;
        }
        std::sort(device_indices.begin(), device_indices.end());
        device_indices.erase(
            std::unique(device_indices.begin(), device_indices.end()),
            device_indices.end());

        bool found_device = false;
        for (int device_index : device_indices) {
            const std::string path =
                "/dev/video" + std::to_string(device_index);
            int descriptor =
                open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
            if (descriptor < 0) {
                continue;
            }
            v4l2_capability capability{};
            const bool queried =
                cameraIoctl(descriptor, VIDIOC_QUERYCAP, &capability) == 0;
            close(descriptor);
            if (!queried) {
                continue;
            }
            const std::uint32_t capabilities =
                (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
                    ? capability.device_caps
                    : capability.capabilities;
            if ((capabilities & (V4L2_CAP_VIDEO_CAPTURE |
                                 V4L2_CAP_VIDEO_CAPTURE_MPLANE)) == 0U) {
                continue;
            }
            output << device_index << '\t' << v4l2Text(capability.card)
                   << '\n';
            found_device = true;
        }
        if (!found_device) {
            error << "acmxvk: no V4L2 capture devices found\n";
        }
        return found_device;
#else
        static_cast<void>(output);
        error << "acmxvk: --list-camera-devices is supported on Linux and macOS\n";
        return false;
#endif
    }

    [[nodiscard]] bool probeCameraDevice(int device_index,
                                         std::ostream &output,
                                         std::ostream &error) {
#ifdef __linux__
        const std::string device_path =
            "/dev/video" + std::to_string(device_index);
        int descriptor =
            open(device_path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (descriptor < 0) {
            descriptor =
                open(device_path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        }
        if (descriptor < 0) {
            error << "acmxvk: cannot open " << device_path << ": "
                  << std::strerror(errno) << '\n';
            return false;
        }

        struct DescriptorGuard {
            int value;
            ~DescriptorGuard() {
                if (value >= 0) {
                    close(value);
                }
            }
        } guard{descriptor};

        v4l2_capability capability{};
        if (cameraIoctl(descriptor, VIDIOC_QUERYCAP, &capability) != 0) {
            error << "acmxvk: cannot query " << device_path << ": "
                  << std::strerror(errno) << '\n';
            return false;
        }

        const std::uint32_t device_capabilities =
            (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
                ? capability.device_caps
                : capability.capabilities;
        const std::string driver = v4l2Text(capability.driver);
        const bool loopback_device =
            driver.find("v4l2loopback") != std::string::npos ||
            driver.find("v4l2 loopback") != std::string::npos;

        output << "Device " << device_index << ": " << device_path << '\n'
               << "  Driver : " << driver << '\n'
               << "  Card   : " << v4l2Text(capability.card) << '\n'
               << "  Bus    : " << v4l2Text(capability.bus_info) << '\n';

        double current_fps = 0.0;
        v4l2_streamparm stream_parameters{};
        stream_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (cameraIoctl(descriptor, VIDIOC_G_PARM, &stream_parameters) == 0) {
            current_fps =
                frameRate(stream_parameters.parm.capture.timeperframe);
        }

        bool found_format = false;
        if ((device_capabilities & V4L2_CAP_VIDEO_CAPTURE) != 0U) {
            found_format = enumerateCaptureType(
                               descriptor, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                               loopback_device, current_fps, output) ||
                           found_format;
        }
        if ((device_capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0U) {
            found_format = enumerateCaptureType(
                               descriptor, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                               loopback_device, current_fps, output) ||
                           found_format;
        }
        if (!found_format) {
            error << "acmxvk: " << device_path
                  << " exposes no capture formats\n";
        }
        return found_format;
#else
        static_cast<void>(device_index);
        static_cast<void>(output);
        error << "acmxvk: --enumerate-device is supported on Linux and macOS\n";
        return false;
#endif
    }
} // namespace acmxvk
