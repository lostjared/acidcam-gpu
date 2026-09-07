#ifndef ACMXVK_DEEP_DREAM_HPP
#define ACMXVK_DEEP_DREAM_HPP

#include <iosfwd>
#include <string_view>

namespace acmxvk::dream {

    [[nodiscard]] bool probe(int cuda_device, std::string_view model_file,
                             std::string_view layer, int iterations,
                             float strength, float feedback, float zoom,
                             float rotation_degrees, int max_dimension,
                             bool use_half, int target_channel,
                             int octaves, float octave_scale,
                             int jitter, int smoothing,
                             std::ostream &output,
                             std::ostream &error);

} // namespace acmxvk::dream

#endif
