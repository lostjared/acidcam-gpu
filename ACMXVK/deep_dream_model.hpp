#ifndef ACMXVK_DEEP_DREAM_MODEL_HPP
#define ACMXVK_DEEP_DREAM_MODEL_HPP

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace acmxvk::dream {

    struct LayerMetadata {
        std::string name;
        std::int64_t source_index = 0;
    };

    struct ModelMetadata {
        std::string architecture;
        std::string default_layer;
        std::vector<LayerMetadata> layers;
        std::vector<double> input_mean;
        std::vector<double> input_std;
        std::int64_t input_channels = 0;
        std::int64_t minimum_input_size = 0;
    };

    class Model {
      public:
        Model(Model &&) noexcept;
        Model &operator=(Model &&) noexcept;
        ~Model();

        Model(const Model &) = delete;
        Model &operator=(const Model &) = delete;

        [[nodiscard]] static Model load(std::string_view filename,
                                        int cuda_device,
                                        std::string_view layer = {});
        [[nodiscard]] const ModelMetadata &metadata() const;
        [[nodiscard]] std::size_t selected_layer() const;
        void print(std::ostream &output) const;

      private:
        struct Impl;
        explicit Model(std::unique_ptr<Impl> implementation);

        std::unique_ptr<Impl> implementation;
    };

} // namespace acmxvk::dream

#endif
