#ifndef ACMX2_DNN_HPP
#define ACMX2_DNN_HPP

#include <memory>
#include <string>
#include <string_view>

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace ac_dnn {

    /// Generic ONNX image-to-image wrapper configured by a YAML file.
    class OnnxWrapper {
      public:
        explicit OnnxWrapper(std::string_view yamlPath);
        ~OnnxWrapper();

        OnnxWrapper(const OnnxWrapper &) = delete;
        OnnxWrapper &operator=(const OnnxWrapper &) = delete;

        void proc(const cv::Mat &image, cv::Mat &output);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    /// DexiNed edge detector.
    class Dexined {
      public:
        explicit Dexined(const std::string &modelPath);
        ~Dexined();

        Dexined(const Dexined &) = delete;
        Dexined &operator=(const Dexined &) = delete;

        void processFrame(const cv::Mat &image, cv::Mat &result);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    /// PP-HumanSeg foreground segmentation model.
    class PPHS {
      public:
        // Passing -1/-1 enables the one-time automatic CPU/CUDA benchmark.
        explicit PPHS(const std::string &modelPath,
                      int backendId = -1, int targetId = -1);
        ~PPHS();

        PPHS(const PPHS &) = delete;
        PPHS &operator=(const PPHS &) = delete;

        cv::Mat preprocess(const cv::Mat &image);
        cv::Mat infer(const cv::Mat &image);
        cv::Mat postprocess(const cv::Mat &outputBlob);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    cv::Mat isolateBody(const cv::Mat &image, const cv::Mat &mask,
                        float blackPoint = 0.35f, float whitePoint = 0.75f);
    cv::Mat hardenedAlphaMask(const cv::Mat &image, const cv::Mat &mask,
                              float blackPoint = 0.35f, float whitePoint = 0.75f);

} // namespace ac_dnn

#endif // ACMX2_DNN_HPP
