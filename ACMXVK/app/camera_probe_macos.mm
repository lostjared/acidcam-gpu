#include "camera_probe.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace acmxvk {
    namespace {
        struct CameraMode {
            std::string format;
            int width = 0;
            int height = 0;
            std::vector<double> frame_rates;
        };

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

        [[nodiscard]] std::string stringValue(NSString *value) {
            if (value == nil) {
                return {};
            }
            const char *text = [value UTF8String];
            std::string result =
                text == nullptr ? std::string{} : std::string(text);
            for (char &character : result) {
                const unsigned char byte =
                    static_cast<unsigned char>(character);
                if (byte < 0x20U || byte == 0x7FU) {
                    character = ' ';
                }
            }
            return result;
        }

        [[nodiscard]] std::string fourccName(OSType subtype) {
            char name[5]{static_cast<char>((subtype >> 24U) & 0xFFU),
                         static_cast<char>((subtype >> 16U) & 0xFFU),
                         static_cast<char>((subtype >> 8U) & 0xFFU),
                         static_cast<char>(subtype & 0xFFU), '\0'};
            bool printable = true;
            for (int index = 0; index < 4; ++index) {
                const unsigned char character =
                    static_cast<unsigned char>(name[index]);
                printable = printable && character >= 0x20U &&
                            character <= 0x7EU;
            }
            if (printable) {
                return name;
            }
            std::ostringstream text;
            text << "0x" << std::hex << std::uppercase << subtype;
            return text.str();
        }
    } // namespace

    [[nodiscard]] bool listCameraDevices(std::ostream &output,
                                         std::ostream &error) {
        @autoreleasepool {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            NSArray<AVCaptureDevice *> *devices =
                [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
            for (NSUInteger index = 0; index < [devices count]; ++index) {
                AVCaptureDevice *device = [devices objectAtIndex:index];
                output << index << '\t'
                       << stringValue([device localizedName]) << '\n';
            }
            if ([devices count] == 0U) {
                error << "acmxvk: no AVFoundation video devices found\n";
                return false;
            }
            return true;
        }
    }

    [[nodiscard]] bool probeCameraDevice(int device_index,
                                         std::ostream &output,
                                         std::ostream &error) {
        @autoreleasepool {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            NSArray<AVCaptureDevice *> *devices =
                [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
            if (device_index < 0 ||
                device_index >= static_cast<int>([devices count])) {
                error << "acmxvk: camera device index " << device_index
                      << " is unavailable; AVFoundation found "
                      << [devices count] << " video device(s)\n";
                return false;
            }

            AVCaptureDevice *device =
                [devices objectAtIndex:static_cast<NSUInteger>(device_index)];
            output << "Device " << device_index << ": AVFoundation\n"
                   << "  Card   : " << stringValue([device localizedName])
                   << '\n'
                   << "  ID     : " << stringValue([device uniqueID]) << '\n';

            using ModeKey = std::tuple<std::string, int, int>;
            std::map<ModeKey, CameraMode> modes;
            for (AVCaptureDeviceFormat *device_format in [device formats]) {
                CMFormatDescriptionRef description =
                    [device_format formatDescription];
                const CMVideoDimensions dimensions =
                    CMVideoFormatDescriptionGetDimensions(description);
                const std::string format = fourccName(
                    CMFormatDescriptionGetMediaSubType(description));
                const ModeKey key{format, dimensions.width,
                                  dimensions.height};
                CameraMode &mode = modes[key];
                mode.format = format;
                mode.width = dimensions.width;
                mode.height = dimensions.height;

                for (AVFrameRateRange *range in
                     [device_format videoSupportedFrameRateRanges]) {
                    appendFrameRate(mode.frame_rates, [range maxFrameRate]);
                    appendFrameRate(mode.frame_rates, [range minFrameRate]);
                }
            }

            if (modes.empty()) {
                error << "acmxvk: AVFoundation camera " << device_index
                      << " exposes no capture formats\n";
                return false;
            }

            std::string current_format;
            for (auto &[key, mode] : modes) {
                static_cast<void>(key);
                if (mode.format != current_format) {
                    current_format = mode.format;
                    output << "\n  Format: " << current_format
                           << " (AVFoundation)\n";
                }
                std::sort(mode.frame_rates.begin(), mode.frame_rates.end(),
                          std::greater<double>());
                output << "    " << mode.width << 'x' << mode.height;
                bool first = true;
                for (double frame_rate : mode.frame_rates) {
                    output << (first ? " @ " : ", ") << std::fixed
                           << std::setprecision(1) << frame_rate << " fps";
                    first = false;
                }
                output << '\n';
            }
            return true;
        }
    }
} // namespace acmxvk
