#include "edge_dnn.hpp"

#include "input_validation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <opencv2/core/persistence.hpp>
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

        void validateMapKeys(const cv::FileNode &node,
                             const std::set<std::string> &allowed,
                             std::string_view context) {
            if (!node.isMap()) {
                throw std::runtime_error(std::string(context) +
                                         " must be a YAML mapping");
            }
            for (auto iterator = node.begin(); iterator != node.end();
                 ++iterator) {
                const std::string name = (*iterator).name();
                if (!allowed.contains(name)) {
                    throw std::runtime_error(std::string(context) +
                                             " contains unsupported field '" +
                                             name + "'");
                }
            }
        }

        [[nodiscard]] std::string readString(const cv::FileNode &node,
                                             std::string_view context,
                                             bool allow_empty = false) {
            if (node.empty() || !node.isString()) {
                throw std::runtime_error(std::string(context) +
                                         " must be a string");
            }
            std::string value;
            node >> value;
            input::validate_string(value, input::StringKind::Path, context,
                                   allow_empty);
            return value;
        }

        [[nodiscard]] double readNumber(const cv::FileNode &node,
                                        double default_value,
                                        std::string_view context) {
            if (node.empty()) {
                return default_value;
            }
            if (!node.isInt() && !node.isReal()) {
                throw std::runtime_error(std::string(context) +
                                         " must be numeric");
            }
            const double value = node.real();
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(context) +
                                         " must be finite");
            }
            return value;
        }

        [[nodiscard]] int readInteger(const cv::FileNode &node,
                                      int default_value,
                                      std::string_view context) {
            const double value = readNumber(node, default_value, context);
            if (std::trunc(value) != value ||
                value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max()) {
                throw std::runtime_error(std::string(context) +
                                         " must be an integer");
            }
            return static_cast<int>(value);
        }

        [[nodiscard]] bool readBoolean(const cv::FileNode &node,
                                       bool default_value,
                                       std::string_view context) {
            if (node.empty()) {
                return default_value;
            }
            if (node.isInt() || node.isReal()) {
                const double value = node.real();
                if (value == 0.0) {
                    return false;
                }
                if (value == 1.0) {
                    return true;
                }
            } else if (node.isString()) {
                std::string value;
                node >> value;
                std::transform(value.begin(), value.end(), value.begin(),
                               [](unsigned char character) {
                                   return static_cast<char>(
                                       std::tolower(character));
                               });
                if (value == "true") {
                    return true;
                }
                if (value == "false") {
                    return false;
                }
            }
            throw std::runtime_error(std::string(context) +
                                     " must be true or false");
        }

        [[nodiscard]] std::string readYamlText(
            const std::filesystem::path &path) {
            input::validate_text_file(path, "ONNX YAML configuration");
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error(
                    "unable to open ONNX YAML configuration: " +
                    path.string());
            }
            std::string text((std::istreambuf_iterator<char>(stream)),
                             std::istreambuf_iterator<char>());
            if (!text.starts_with("%YAML")) {
                const std::size_t first_content =
                    text.find_first_not_of(" \t\r\n");
                const bool has_document_marker =
                    first_content != std::string::npos &&
                    text.compare(first_content, 3, "---") == 0;
                text.insert(0, has_document_marker ? "%YAML:1.0\n"
                                                   : "%YAML:1.0\n---\n");
            }
            return text;
        }

        void validateTensorName(std::string_view name) {
            if (name.empty()) {
                return;
            }
            if (name.size() > 256U ||
                !std::all_of(name.begin(), name.end(), [](char value) {
                    const auto character =
                        static_cast<unsigned char>(value);
                    return std::isalnum(character) != 0 || value == '_' ||
                           value == ':' || value == '/' || value == '.' ||
                           value == '-';
                })) {
                throw std::runtime_error(
                    "model.input contains unsupported characters");
            }
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

    struct GenericOnnxProcessor::Impl {
        cv::dnn::Net net;
        BackendState backend;
        cv::Size input_size{224, 224};
        cv::Size active_input_size;
        double scale = 1.0 / 255.0;
        cv::Scalar mean{0.0, 0.0, 0.0};
        bool swap_rb = true;
        bool dynamic_shape = false;
        int shape_alignment = 4;
        bool bilateral_smoothing = false;
        int bilateral_diameter = 5;
        double bilateral_sigma_color = 5.0;
        double bilateral_sigma_space = 5.0;
        bool multiple_outputs = false;
        cv::String input_name;
        cv::String output_name;
        cv::Mat blob;
        cv::Mat work;
        cv::Mat converted;
        cv::Mat smoothed;

        explicit Impl(const std::string &configuration_path) {
            const std::filesystem::path yaml_path(configuration_path);
            if (!std::filesystem::is_regular_file(yaml_path)) {
                throw std::runtime_error(
                    "ONNX configuration is not a regular file: " +
                    configuration_path);
            }

            const std::string yaml_text = readYamlText(yaml_path);
            cv::FileStorage storage(yaml_text,
                                    cv::FileStorage::READ |
                                        cv::FileStorage::MEMORY |
                                        cv::FileStorage::FORMAT_YAML);
            if (!storage.isOpened()) {
                throw std::runtime_error(
                    "unable to parse ONNX YAML configuration: " +
                    configuration_path);
            }
            const cv::FileNode root = storage.root();
            validateMapKeys(root, {"model", "preprocessing", "postprocessing"},
                            "ONNX YAML root");

            const cv::FileNode model = root["model"];
            validateMapKeys(model, {"path", "input"}, "model");
            const std::string configured_model_path =
                readString(model["path"], "model.path");
            std::string configured_input_name;
            if (!model["input"].empty()) {
                if (!model["input"].isString()) {
                    throw std::runtime_error("model.input must be a string");
                }
                model["input"] >> configured_input_name;
                validateTensorName(configured_input_name);
            }
            input_name = configured_input_name;

            std::filesystem::path model_path(configured_model_path);
            if (model_path.is_relative()) {
                model_path = yaml_path.parent_path() / model_path;
            }
            model_path = model_path.lexically_normal();
            if (!std::filesystem::is_regular_file(model_path)) {
                throw std::runtime_error(
                    "ONNX model referenced by YAML is not a regular file: " +
                    model_path.string());
            }

            const cv::FileNode preprocessing = root["preprocessing"];
            if (!preprocessing.empty()) {
                validateMapKeys(preprocessing,
                                {"width", "height", "scale", "swap_rb",
                                 "mean", "dynamic", "alignment"},
                                "preprocessing");
                input_size.width = readInteger(preprocessing["width"], 224,
                                               "preprocessing.width");
                input_size.height = readInteger(preprocessing["height"], 224,
                                                "preprocessing.height");
                scale = readNumber(preprocessing["scale"], 1.0 / 255.0,
                                   "preprocessing.scale");
                swap_rb = readBoolean(preprocessing["swap_rb"], true,
                                      "preprocessing.swap_rb");
                dynamic_shape = readBoolean(preprocessing["dynamic"], false,
                                            "preprocessing.dynamic");
                shape_alignment = readInteger(preprocessing["alignment"], 4,
                                              "preprocessing.alignment");
                const cv::FileNode mean_node = preprocessing["mean"];
                if (!mean_node.empty()) {
                    if (!mean_node.isSeq() || mean_node.size() != 3U) {
                        throw std::runtime_error(
                            "preprocessing.mean must contain three numbers");
                    }
                    for (int index = 0; index < 3; ++index) {
                        mean[index] = readNumber(
                            mean_node[index], 0.0, "preprocessing.mean");
                    }
                }
            }
            if (input_size.width < 0 || input_size.width > 16384 ||
                input_size.height < 0 || input_size.height > 16384 ||
                (!dynamic_shape &&
                 (input_size.width == 0 || input_size.height == 0))) {
                throw std::runtime_error(
                    "preprocessing dimensions are outside the supported range");
            }
            if (std::abs(scale) > 1.0e6 ||
                std::any_of(&mean[0], &mean[0] + 3, [](double value) {
                    return !std::isfinite(value) || std::abs(value) > 1.0e6;
                })) {
                throw std::runtime_error(
                    "preprocessing scale or mean is outside the supported range");
            }
            if (shape_alignment < 1 || shape_alignment > 1024) {
                throw std::runtime_error(
                    "preprocessing.alignment must be between 1 and 1024");
            }

            bilateral_smoothing =
                dynamic_shape && input_size.width > 0 &&
                input_size.height > 0 && input_size.width <= 256 &&
                input_size.height <= 256;
            const cv::FileNode postprocessing = root["postprocessing"];
            if (!postprocessing.empty()) {
                validateMapKeys(postprocessing, {"bilateral"},
                                "postprocessing");
                const cv::FileNode bilateral = postprocessing["bilateral"];
                if (!bilateral.empty() && bilateral.isMap()) {
                    validateMapKeys(
                        bilateral,
                        {"enabled", "diameter", "sigma_color", "sigma_space"},
                        "postprocessing.bilateral");
                    bilateral_smoothing = readBoolean(
                        bilateral["enabled"], bilateral_smoothing,
                        "postprocessing.bilateral.enabled");
                    bilateral_diameter = readInteger(
                        bilateral["diameter"], bilateral_diameter,
                        "postprocessing.bilateral.diameter");
                    bilateral_sigma_color = readNumber(
                        bilateral["sigma_color"], bilateral_sigma_color,
                        "postprocessing.bilateral.sigma_color");
                    bilateral_sigma_space = readNumber(
                        bilateral["sigma_space"], bilateral_sigma_space,
                        "postprocessing.bilateral.sigma_space");
                } else if (!bilateral.empty()) {
                    bilateral_smoothing = readBoolean(
                        bilateral, bilateral_smoothing,
                        "postprocessing.bilateral");
                }
            }
            if (bilateral_diameter < 1 || bilateral_diameter > 255 ||
                bilateral_sigma_color < 0.0 ||
                bilateral_sigma_color > 1.0e6 ||
                bilateral_sigma_space < 0.0 ||
                bilateral_sigma_space > 1.0e6) {
                throw std::runtime_error(
                    "bilateral smoothing values are outside the supported range");
            }
            if ((bilateral_diameter & 1) == 0) {
                ++bilateral_diameter;
            }

#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
            net = cv::dnn::readNetFromONNX(model_path.string(),
                                           cv::dnn::ENGINE_CLASSIC);
#else
            net = cv::dnn::readNetFromONNX(model_path.string());
#endif
            if (net.empty()) {
                throw std::runtime_error("generic ONNX model is empty");
            }
            const std::vector<cv::String> names =
                net.getUnconnectedOutLayersNames();
            multiple_outputs = names.size() > 1U;
            output_name = names.empty() ? cv::String() : names.back();
        }

        [[nodiscard]] cv::Size resolveInputSize(
            const cv::Size &source_size) const {
            if (!dynamic_shape) {
                return input_size;
            }
            int width = input_size.width;
            int height = input_size.height;
            if (width <= 0 && height <= 0) {
                width = source_size.width;
                height = source_size.height;
            } else if (width <= 0) {
                width = cvRound(height *
                                static_cast<double>(source_size.width) /
                                std::max(source_size.height, 1));
            } else if (height <= 0) {
                height = cvRound(width *
                                 static_cast<double>(source_size.height) /
                                 std::max(source_size.width, 1));
            }
            const auto align_dimension = [this](int value) {
                value = std::max(value, shape_alignment);
                return std::max(
                    shape_alignment,
                    cvRound(static_cast<double>(value) / shape_alignment) *
                        shape_alignment);
            };
            return {align_dimension(width), align_dimension(height)};
        }

        [[nodiscard]] const cv::Mat &smoothOutput(const cv::Mat &source) {
            if (!bilateral_smoothing) {
                return source;
            }
            cv::bilateralFilter(source, smoothed, bilateral_diameter,
                                bilateral_sigma_color,
                                bilateral_sigma_space);
            return smoothed;
        }

        void process(const cv::Mat &image, cv::Mat &result) {
            if (image.empty()) {
                result.release();
                return;
            }
            const cv::Size frame_input_size = resolveInputSize(image.size());
            if (frame_input_size != active_input_size) {
                if (!active_input_size.empty()) {
                    backend.selected = false;
                }
                active_input_size = frame_input_size;
            }
            cv::dnn::blobFromImage(image, blob, scale, frame_input_size, mean,
                                   swap_rb, false, CV_32F);
            const cv::Mat raw = selectBackendAndForward(
                net, backend, blob, input_name, output_name);

            if (multiple_outputs) {
                const cv::Mat plane = spatialPlane(raw);
                if (plane.empty()) {
                    throw std::runtime_error(
                        "generic ONNX multi-output result has no float plane");
                }
                cv::exp(-plane, work);
                cv::add(work, cv::Scalar::all(1.0), work);
                cv::divide(1.0, work, work);
                cv::normalize(work, converted, 0, 255, cv::NORM_MINMAX,
                              CV_8U);
                const cv::Mat &display = smoothOutput(converted);
                cv::resize(display, converted, image.size(), 0, 0,
                           cv::INTER_LINEAR);
                cv::cvtColor(converted, result, cv::COLOR_GRAY2BGR);
                return;
            }
            if (raw.dims != 4 || raw.size[0] != 1 ||
                raw.type() != CV_32F) {
                throw std::runtime_error(
                    "generic ONNX output must be a 1xCxHxW float tensor");
            }
            const int channels = raw.size[1];
            const int height = raw.size[2];
            const int width = raw.size[3];
            if (channels == 1) {
                const cv::Mat plane(
                    height, width, CV_32F,
                    const_cast<float *>(raw.ptr<float>(0, 0)));
                cv::normalize(plane, converted, 0, 255, cv::NORM_MINMAX,
                              CV_8U);
                const cv::Mat &display = smoothOutput(converted);
                cv::resize(display, converted, image.size(), 0, 0,
                           cv::INTER_LINEAR);
                cv::cvtColor(converted, result, cv::COLOR_GRAY2BGR);
                return;
            }
            if (channels < 3) {
                throw std::runtime_error(
                    "generic ONNX output has fewer than three color channels");
            }
            std::vector<cv::Mat> planes;
            planes.reserve(3);
            for (int channel = 0; channel < 3; ++channel) {
                planes.emplace_back(
                    height, width, CV_32F,
                    const_cast<float *>(raw.ptr<float>(0, channel)));
            }
            cv::merge(planes, work);
            cv::normalize(work, converted, 0, 255, cv::NORM_MINMAX, CV_8U);
            const cv::Mat &display = smoothOutput(converted);
            cv::resize(display, converted, image.size(), 0, 0,
                       cv::INTER_LINEAR);
            cv::cvtColor(converted, result, cv::COLOR_RGB2BGR);
        }
    };

    GenericOnnxProcessor::GenericOnnxProcessor(
        const std::string &configuration_path)
        : impl(std::make_unique<Impl>(configuration_path)) {}

    GenericOnnxProcessor::~GenericOnnxProcessor() = default;

    void GenericOnnxProcessor::process(const cv::Mat &image,
                                       cv::Mat &result) {
        impl->process(image, result);
    }

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
