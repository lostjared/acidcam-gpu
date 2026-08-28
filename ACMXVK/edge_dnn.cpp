#include "edge_dnn.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/core/version.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace acmxvk::dnn {
    namespace {

        struct BackendState {
            bool selected = false;
        };

        struct TimedOutput {
            cv::Mat output;
            double milliseconds = std::numeric_limits<double>::infinity();
        };

        [[nodiscard]] bool backendAvailable(cv::dnn::Backend backend,
                                            cv::dnn::Target target) {
            try {
                const std::vector<cv::dnn::Target> available =
                    cv::dnn::getAvailableTargets(backend);
                return std::find(available.begin(), available.end(), target) !=
                       available.end();
            } catch (const cv::Exception &) {
                return false;
            }
        }

        void setCpuBackend(cv::dnn::Net &net) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        void setCudaBackend(cv::dnn::Net &net, bool fp16) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(fp16 ? cv::dnn::DNN_TARGET_CUDA_FP16
                                         : cv::dnn::DNN_TARGET_CUDA);
        }

        [[nodiscard]] cv::Mat runForward(cv::dnn::Net &net,
                                         const cv::Mat &blob,
                                         const cv::String &input_name,
                                         const cv::String &output_name) {
            if (input_name.empty()) {
                net.setInput(blob);
            } else {
                net.setInput(blob, input_name);
            }
            return output_name.empty() ? net.forward()
                                       : net.forward(output_name);
        }

        [[nodiscard]] TimedOutput benchmarkBackend(
            cv::dnn::Net &net, const cv::Mat &blob,
            const cv::String &input_name,
            const cv::String &output_name) {
            static_cast<void>(runForward(net, blob, input_name, output_name));

            TimedOutput measured;
            constexpr int TIMED_RUNS = 2;
            const auto start = std::chrono::steady_clock::now();
            for (int run = 0; run < TIMED_RUNS; ++run) {
                measured.output =
                    runForward(net, blob, input_name, output_name);
            }
            measured.milliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count() /
                TIMED_RUNS;
            return measured;
        }

        [[nodiscard]] cv::Mat selectBackendAndForward(
            cv::dnn::Net &net, BackendState &state, const cv::Mat &blob,
            const cv::String &input_name,
            const cv::String &output_name) {
            if (state.selected) {
                return runForward(net, blob, input_name, output_name);
            }
            state.selected = true;

            const bool fp16_available = backendAvailable(
                cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16);
            const bool fp32_available = backendAvailable(
                cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);
            if (!fp16_available && !fp32_available) {
                setCpuBackend(net);
                std::cout << "acmxvk: DNN backend: CPU (CUDA unavailable)\n";
                return runForward(net, blob, input_name, output_name);
            }

            setCpuBackend(net);
            TimedOutput cpu =
                benchmarkBackend(net, blob, input_name, output_name);
            const bool use_fp16 = fp16_available;
            try {
                setCudaBackend(net, use_fp16);
                TimedOutput cuda =
                    benchmarkBackend(net, blob, input_name, output_name);
                if (cuda.milliseconds < cpu.milliseconds) {
                    std::cout << "acmxvk: DNN backend: CUDA "
                              << (use_fp16 ? "FP16" : "FP32") << " ("
                              << cuda.milliseconds << " ms vs CPU "
                              << cpu.milliseconds << " ms)\n";
                    return cuda.output;
                }
                setCpuBackend(net);
                std::cout << "acmxvk: DNN backend: CPU (" << cpu.milliseconds
                          << " ms vs CUDA " << cuda.milliseconds << " ms)\n";
                return cpu.output;
            } catch (const cv::Exception &error) {
                setCpuBackend(net);
                std::cerr << "acmxvk: CUDA DNN benchmark failed; using CPU ("
                          << cpu.milliseconds << " ms): " << error.what()
                          << '\n';
                return cpu.output;
            }
        }

        [[nodiscard]] cv::String lastOutputName(const cv::dnn::Net &net) {
            const std::vector<cv::String> names =
                net.getUnconnectedOutLayersNames();
            return names.empty() ? cv::String() : names.back();
        }

        [[nodiscard]] cv::Mat spatialPlane(const cv::Mat &output) {
            if (output.dims == 4 && output.size[0] == 1 &&
                output.size[1] == 1 && output.type() == CV_32F) {
                return cv::Mat(output.size[2], output.size[3], CV_32F,
                               const_cast<float *>(output.ptr<float>(0, 0)));
            }
            if (output.dims == 2 && output.type() == CV_32F) {
                return output;
            }
            return {};
        }

        [[nodiscard]] cv::Mat buildHardenedFloatAlpha(
            const cv::Mat &image, const cv::Mat &mask, float black_point,
            float white_point) {
            constexpr int MAXIMUM_WORK_DIMENSION = 512;
            const int image_max_dimension = std::max(image.cols, image.rows);
            const double work_scale =
                image_max_dimension > MAXIMUM_WORK_DIMENSION
                    ? static_cast<double>(MAXIMUM_WORK_DIMENSION) /
                          image_max_dimension
                    : 1.0;
            const cv::Size work_size(
                std::max(1, cvRound(image.cols * work_scale)),
                std::max(1, cvRound(image.rows * work_scale)));

            cv::Mat soft;
            if (mask.channels() == 1) {
                mask.convertTo(soft, CV_32F,
                               mask.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
            } else {
                cv::Mat gray;
                cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
                gray.convertTo(soft, CV_32F,
                               gray.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
            }
            if (soft.size() != work_size) {
                cv::resize(soft, soft, work_size, 0, 0, cv::INTER_LINEAR);
            }
            cv::threshold(soft, soft, 1.0, 1.0, cv::THRESH_TRUNC);
            cv::threshold(soft, soft, 0.0, 0.0, cv::THRESH_TOZERO);

            cv::Mat binary;
            cv::threshold(soft, binary, 0.5F, 255.0F, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8U);
            const auto scaled_kernel_size = [work_scale](int full_size) {
                int size = std::max(1, cvRound(full_size * work_scale));
                if ((size & 1) == 0) {
                    ++size;
                }
                return size;
            };
            const int open_size = scaled_kernel_size(3);
            const int close_size = scaled_kernel_size(7);
            const int erode_size = scaled_kernel_size(3);
            if (open_size > 1) {
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(open_size, open_size));
                cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
            }
            if (close_size > 1) {
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(close_size, close_size));
                cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
            }

            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            const int label_count = cv::connectedComponentsWithStats(
                binary, labels, stats, centroids, 8, CV_32S);
            if (label_count > 1) {
                int best_label = -1;
                int best_area = 0;
                for (int label = 1; label < label_count; ++label) {
                    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                    if (area > best_area) {
                        best_area = area;
                        best_label = label;
                    }
                }
                const int minimum_area =
                    (binary.cols * binary.rows) / 200;
                if (best_label > 0 && best_area >= minimum_area) {
                    cv::compare(labels, best_label, binary, cv::CMP_EQ);
                }
            }
            if (erode_size > 1) {
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(erode_size, erode_size));
                cv::erode(binary, binary, kernel);
            }

            cv::Mat silhouette;
            binary.convertTo(silhouette, CV_32F, 1.0 / 255.0);
            cv::multiply(soft, silhouette, soft);
            cv::GaussianBlur(soft, soft, cv::Size(), 1.2 * work_scale);

            const float range =
                std::max(white_point - black_point, 1.0e-6F);
            soft.convertTo(soft, CV_32F, 1.0F / range,
                           -black_point / range);
            cv::threshold(soft, soft, 1.0, 1.0, cv::THRESH_TRUNC);
            cv::threshold(soft, soft, 0.0, 0.0, cv::THRESH_TOZERO);
            cv::pow(soft, 1.6, soft);
            if (soft.size() != image.size()) {
                cv::resize(soft, soft, image.size(), 0, 0, cv::INTER_LINEAR);
            }
            return soft;
        }

    } // namespace

    struct EdgeDetector::Impl {
        cv::dnn::Net net;
        BackendState backend;
        cv::String output_name;
        cv::Mat blob;
        cv::Mat work;
        cv::Mat edge;

        explicit Impl(const std::string &model_path) {
            if (!std::filesystem::is_regular_file(model_path)) {
                throw std::runtime_error("edge model is not a regular file: " +
                                         model_path);
            }
#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
            net = cv::dnn::readNetFromONNX(model_path,
                                           cv::dnn::ENGINE_CLASSIC);
#else
            net = cv::dnn::readNetFromONNX(model_path);
#endif
            if (net.empty()) {
                throw std::runtime_error("DexiNed ONNX model is empty");
            }
            output_name = lastOutputName(net);
        }

        void process(const cv::Mat &image, cv::Mat &result) {
            if (image.empty()) {
                result.release();
                return;
            }

            cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(512, 512),
                                   cv::Scalar(103.5, 116.2, 123.6), false,
                                   false, CV_32F);
            const cv::Mat raw =
                selectBackendAndForward(net, backend, blob, {}, output_name);
            const cv::Mat plane = spatialPlane(raw);
            if (plane.empty()) {
                throw std::runtime_error(
                    "DexiNed output does not contain a float edge plane");
            }

            cv::exp(-plane, work);
            cv::add(work, cv::Scalar::all(1.0), work);
            cv::divide(1.0, work, work);
            cv::normalize(work, edge, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::resize(edge, result, image.size(), 0, 0, cv::INTER_LINEAR);
        }
    };

    EdgeDetector::EdgeDetector(const std::string &model_path)
        : impl(std::make_unique<Impl>(model_path)) {}

    EdgeDetector::~EdgeDetector() = default;

    void EdgeDetector::process(const cv::Mat &image, cv::Mat &result) {
        impl->process(image, result);
    }

    struct HumanSegmenter::Impl {
        cv::dnn::Net net;
        BackendState backend;
        const cv::Size input_size{192, 192};
        const cv::String input_name{"x"};
        const cv::String output_name{"save_infer_model/scale_0.tmp_1"};
        cv::Size source_size;
        cv::Mat blob;
        cv::Mat logit;
        cv::Mat probability;
        cv::Mat resized_mask;
        cv::Mat previous_mask;

        explicit Impl(const std::string &model_path) {
            if (!std::filesystem::is_regular_file(model_path)) {
                throw std::runtime_error(
                    "human segmentation model is not a regular file: " +
                    model_path);
            }
#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
            net = cv::dnn::readNetFromONNX(model_path,
                                           cv::dnn::ENGINE_CLASSIC);
#else
            net = cv::dnn::readNetFromONNX(model_path);
#endif
            if (net.empty()) {
                throw std::runtime_error("PP-HumanSeg ONNX model is empty");
            }
        }

        [[nodiscard]] cv::Mat infer(const cv::Mat &image) {
            if (image.empty()) {
                return {};
            }
            source_size = image.size();
            cv::dnn::blobFromImage(image, blob, 1.0 / 127.5, input_size,
                                   cv::Scalar(127.5, 127.5, 127.5), false,
                                   false, CV_32F);
            const cv::Mat output = selectBackendAndForward(
                net, backend, blob, input_name, output_name);
            if (output.dims != 4 || output.size[0] != 1 ||
                output.size[1] < 2 || output.type() != CV_32F) {
                throw std::runtime_error(
                    "PP-HumanSeg output is not a two-channel float mask");
            }

            const int height = output.size[2];
            const int width = output.size[3];
            const cv::Mat background(
                height, width, CV_32F,
                const_cast<float *>(output.ptr<float>(0, 0)));
            const cv::Mat foreground(
                height, width, CV_32F,
                const_cast<float *>(output.ptr<float>(0, 1)));
            cv::subtract(foreground, background, logit);
            cv::exp(-logit, probability);
            cv::add(probability, cv::Scalar::all(1.0), probability);
            cv::divide(1.0, probability, probability);
            cv::resize(probability, resized_mask, source_size, 0, 0,
                       cv::INTER_CUBIC);
            if (previous_mask.empty() ||
                previous_mask.size() != resized_mask.size()) {
                resized_mask.copyTo(previous_mask);
            } else {
                cv::addWeighted(resized_mask, 0.6, previous_mask, 0.4, 0.0,
                                previous_mask);
            }
            return previous_mask;
        }
    };

    HumanSegmenter::HumanSegmenter(const std::string &model_path)
        : impl(std::make_unique<Impl>(model_path)) {}

    HumanSegmenter::~HumanSegmenter() = default;

    cv::Mat HumanSegmenter::infer(const cv::Mat &image) {
        return impl->infer(image);
    }

    cv::Mat hardenedAlphaMask(const cv::Mat &image, const cv::Mat &mask,
                              float black_point, float white_point) {
        if (image.empty() || mask.empty()) {
            return {};
        }
        cv::Mat alpha;
        buildHardenedFloatAlpha(image, mask, black_point, white_point)
            .convertTo(alpha, CV_8U, 255.0);
        return alpha;
    }

    cv::Mat isolateBody(const cv::Mat &image, const cv::Mat &mask,
                        float black_point, float white_point) {
        if (image.empty() || mask.empty()) {
            return image.clone();
        }
        const cv::Mat alpha =
            hardenedAlphaMask(image, mask, black_point, white_point);
        cv::Mat alpha_bgr;
        cv::cvtColor(alpha, alpha_bgr, cv::COLOR_GRAY2BGR);
        cv::Mat output;
        cv::multiply(image, alpha_bgr, output, 1.0 / 255.0, CV_8UC3);
        return output;
    }

} // namespace acmxvk::dnn
