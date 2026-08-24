#ifndef ACMXVK_GPU_FILTERS_HPP
#define ACMXVK_GPU_FILTERS_HPP

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace acmxvk::gpu {

    class FilterEngine {
      public:
        FilterEngine(std::vector<int> filter_indices, int frame_buffer_size);
        ~FilterEngine();

        FilterEngine(const FilterEngine &) = delete;
        FilterEngine &operator=(const FilterEngine &) = delete;

        [[nodiscard]] bool process(const cv::Mat &rgba);
        [[nodiscard]] bool process(const cv::cuda::GpuMat &rgba,
                                   cv::cuda::Stream &source_stream);
        [[nodiscard]] bool select_relative_filter(int direction);
        [[nodiscard]] std::string active_filter_description() const;
        [[nodiscard]] const cv::cuda::GpuMat &output() const;
        [[nodiscard]] cv::cuda::Stream &stream();

        static void select_device(int device_index);
        static void validate_filter_indices(
            const std::vector<int> &filter_indices);
        static void list_devices(std::ostream &output);
        static void list_filters(std::ostream &output);

      private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };

} // namespace acmxvk::gpu

#endif
