#ifndef ACMXVK_APP_CAMERA_PROBE_HPP
#define ACMXVK_APP_CAMERA_PROBE_HPP

#include <iosfwd>

namespace acmxvk {
    /** Print native camera device indices and names as tab-separated records. */
    [[nodiscard]] bool listCameraDevices(std::ostream &output,
                                         std::ostream &error);

    /**
     * Print the native capture formats, resolutions, and frame rates exposed by
     * a camera device.
     *
     * Output intentionally matches ACMX2's --enumerate-device format so the Qt
     * interface can parse either backend with the same code.
     */
    [[nodiscard]] bool probeCameraDevice(int device_index,
                                         std::ostream &output,
                                         std::ostream &error);
} // namespace acmxvk

#endif
