#ifndef ACMXVK_DEEP_DREAM_MODEL_HPP
#define ACMXVK_DEEP_DREAM_MODEL_HPP

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/mat.hpp>

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

    struct GradientAscentOptions {
        int iterations = 1;
        float step_size = 0.05F;
        float feedback = 0.9F;
        float zoom = 1.01F;
        float rotation_degrees = 0.1F;
        int max_dimension = 512;
        int target_channel = -1;
        int octaves = 1;
        float octave_scale = 1.4F;
    };

    struct GradientAscentResult {
        float activation_loss = 0.0F;
        float mean_gradient = 0.0F;
        float mean_pixel_change = 0.0F;
        int processed_width = 0;
        int processed_height = 0;
        int processed_octaves = 0;
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
                                        std::string_view layer = {},
                                        bool use_half = false);
        [[nodiscard]] const ModelMetadata &metadata() const;
        [[nodiscard]] std::size_t selected_layer() const;
        [[nodiscard]] std::size_t selected_channels() const;
        [[nodiscard]] GradientAscentResult apply_gradient_ascent(
            cv::Mat &rgba, const GradientAscentOptions &options = {});
        void print(std::ostream &output) const;

      private:
        struct Impl;
        explicit Model(std::unique_ptr<Impl> implementation);

        std::unique_ptr<Impl> implementation;
    };

} // namespace acmxvk::dream

#endif
