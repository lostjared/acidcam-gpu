#ifndef ACMXVK_EDGE_DNN_HPP
#define ACMXVK_EDGE_DNN_HPP

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace acmxvk::dnn {

    class GenericOnnxProcessor {
      public:
        explicit GenericOnnxProcessor(const std::string &configuration_path);
        ~GenericOnnxProcessor();

        GenericOnnxProcessor(const GenericOnnxProcessor &) = delete;
        GenericOnnxProcessor &operator=(const GenericOnnxProcessor &) = delete;

        void process(const cv::Mat &image, cv::Mat &result);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    class EdgeDetector {
      public:
        explicit EdgeDetector(const std::string &model_path);
        ~EdgeDetector();

        EdgeDetector(const EdgeDetector &) = delete;
        EdgeDetector &operator=(const EdgeDetector &) = delete;

        void process(const cv::Mat &image, cv::Mat &result);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    class HumanSegmenter {
      public:
        explicit HumanSegmenter(const std::string &model_path);
        ~HumanSegmenter();

        HumanSegmenter(const HumanSegmenter &) = delete;
        HumanSegmenter &operator=(const HumanSegmenter &) = delete;

        [[nodiscard]] cv::Mat infer(const cv::Mat &image);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    [[nodiscard]] cv::Mat hardenedAlphaMask(const cv::Mat &image,
                                            const cv::Mat &mask,
                                            float black_point,
                                            float white_point);
    [[nodiscard]] cv::Mat isolateBody(const cv::Mat &image,
                                      const cv::Mat &mask,
                                      float black_point,
                                      float white_point);

} // namespace acmxvk::dnn

#endif
