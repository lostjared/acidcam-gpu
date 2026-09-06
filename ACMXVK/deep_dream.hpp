#ifndef ACMXVK_DEEP_DREAM_HPP
#define ACMXVK_DEEP_DREAM_HPP

#include <iosfwd>
#include <string_view>

namespace acmxvk::dream {

    [[nodiscard]] bool probe(int cuda_device, std::string_view model_file,
                             std::string_view layer, std::ostream &output,
                             std::ostream &error);

} // namespace acmxvk::dream

#endif
