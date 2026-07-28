#include "dnn.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#ifdef ACMX2_WITH_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
#include <yaml-cpp/yaml.h>

namespace ac_dnn {
    namespace {

        enum class BackendMode {
            Auto,
            Cpu,
            Cuda,
            CudaFp16,
            Explicit
        };

        struct BackendState {
            BackendMode mode = BackendMode::Auto;
            int explicitBackend = -1;
            int explicitTarget = -1;
            bool selected = false;
            bool usesCuda = false;
        };

        BackendMode backendModeFromEnvironment() {
            const char *value = std::getenv("ACMX2_DNN_BACKEND");
            if (value == nullptr)
                return BackendMode::Auto;

            std::string mode(value);
            std::transform(mode.begin(), mode.end(), mode.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (mode == "cpu")
                return BackendMode::Cpu;
            if (mode == "cuda")
                return BackendMode::Cuda;
            if (mode == "cuda_fp16" || mode == "cuda-fp16" || mode == "fp16")
                return BackendMode::CudaFp16;
            if (mode != "auto")
                std::cerr << "acmx2: Unknown ACMX2_DNN_BACKEND='" << value
                          << "'; using automatic selection\n";
            return BackendMode::Auto;
        }

        bool backendAvailable(cv::dnn::Backend backend, cv::dnn::Target target) {
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

        cv::Mat runForward(cv::dnn::Net &net, const cv::Mat &blob,
                           const cv::String &inputName, const cv::String &outputName) {
            if (inputName.empty())
                net.setInput(blob);
            else
                net.setInput(blob, inputName);
            return outputName.empty() ? net.forward() : net.forward(outputName);
        }

        struct TimedOutput {
            cv::Mat output;
            double milliseconds = std::numeric_limits<double>::infinity();
        };

        TimedOutput benchmarkBackend(cv::dnn::Net &net, const cv::Mat &blob,
                                     const cv::String &inputName,
                                     const cv::String &outputName) {
            // The first forward builds/fuses the backend graph and is not
            // representative of steady-state video processing.
            runForward(net, blob, inputName, outputName);

            TimedOutput measured;
            constexpr int timedRuns = 2;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < timedRuns; ++i)
                measured.output = runForward(net, blob, inputName, outputName);
            measured.milliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count() /
                timedRuns;
            return measured;
        }

        cv::Mat selectBackendAndForward(cv::dnn::Net &net, BackendState &state,
                                        const cv::Mat &blob,
                                        const cv::String &inputName,
                                        const cv::String &outputName) {
            if (state.selected)
                return runForward(net, blob, inputName, outputName);

            state.selected = true;
            if (state.mode == BackendMode::Explicit) {
                net.setPreferableBackend(state.explicitBackend);
                net.setPreferableTarget(state.explicitTarget);
                state.usesCuda =
                    state.explicitBackend == cv::dnn::DNN_BACKEND_CUDA;
                std::cout << "[ACMX2] DNN backend: explicitly configured ("
                          << state.explicitBackend << '/' << state.explicitTarget << ")\n";
                return runForward(net, blob, inputName, outputName);
            }

            const BackendMode requested = state.mode == BackendMode::Auto
                                              ? backendModeFromEnvironment()
                                              : state.mode;
            if (requested == BackendMode::Cpu) {
                setCpuBackend(net);
                state.usesCuda = false;
                std::cout << "[ACMX2] DNN backend: CPU (forced by ACMX2_DNN_BACKEND)\n";
                return runForward(net, blob, inputName, outputName);
            }

            const bool requestFp16 = requested == BackendMode::CudaFp16;
            if (requested == BackendMode::Cuda || requested == BackendMode::CudaFp16) {
                try {
                    setCudaBackend(net, requestFp16);
                    state.usesCuda = true;
                    std::cout << "[ACMX2] DNN backend: CUDA "
                              << (requestFp16 ? "FP16" : "FP32")
                              << " (forced by ACMX2_DNN_BACKEND)\n";
                    return runForward(net, blob, inputName, outputName);
                } catch (const cv::Exception &error) {
                    std::cerr << "acmx2: Requested CUDA DNN backend failed; falling back to CPU: "
                              << error.what() << '\n';
                    setCpuBackend(net);
                    state.usesCuda = false;
                    return runForward(net, blob, inputName, outputName);
                }
            }

            const bool fp16Available =
                backendAvailable(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16);
            const bool fp32Available =
                backendAvailable(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);
            if (!fp16Available && !fp32Available) {
                setCpuBackend(net);
                state.usesCuda = false;
                std::cout << "[ACMX2] DNN backend: CPU (CUDA unavailable)\n";
                return runForward(net, blob, inputName, outputName);
            }

            setCpuBackend(net);
            state.usesCuda = false;
            TimedOutput cpu = benchmarkBackend(net, blob, inputName, outputName);
            const bool useFp16 = fp16Available;
            try {
                setCudaBackend(net, useFp16);
                state.usesCuda = true;
                TimedOutput cuda = benchmarkBackend(net, blob, inputName, outputName);
                if (cuda.milliseconds < cpu.milliseconds) {
                    std::cout << "[ACMX2] DNN backend: CUDA "
                              << (useFp16 ? "FP16" : "FP32") << " ("
                              << cuda.milliseconds << " ms vs CPU "
                              << cpu.milliseconds << " ms)\n";
                    return cuda.output;
                }
                setCpuBackend(net);
                state.usesCuda = false;
                std::cout << "[ACMX2] DNN backend: CPU (" << cpu.milliseconds
                          << " ms vs CUDA " << cuda.milliseconds << " ms)\n";
                return cpu.output;
            } catch (const cv::Exception &error) {
                setCpuBackend(net);
                state.usesCuda = false;
                std::cerr << "acmx2: CUDA DNN benchmark failed; using CPU ("
                          << cpu.milliseconds << " ms): " << error.what() << '\n';
                return cpu.output;
            }
        }

        cv::String lastOutputName(const cv::dnn::Net &net) {
            const std::vector<cv::String> names = net.getUnconnectedOutLayersNames();
            return names.empty() ? cv::String() : names.back();
        }

        cv::Mat spatialPlane(const cv::Mat &output) {
            if (output.dims == 4 && output.size[0] == 1 && output.size[1] == 1)
                return cv::Mat(output.size[2], output.size[3], CV_32F,
                               const_cast<float *>(output.ptr<float>(0, 0)));
            if (output.dims == 2 && output.type() == CV_32F)
                return output;
            return cv::Mat();
        }

    } // namespace

    struct OnnxWrapper::Impl {
        cv::dnn::Net net;
        BackendState backend;
        bool loaded = false;
        bool inferenceFailed = false;
        bool multipleOutputs = false;
        cv::Size inputSize{224, 224};
        cv::Size activeInputSize;
        double scale = 1.0 / 255.0;
        cv::Scalar mean{0.0, 0.0, 0.0};
        bool swapRb = true;
        bool dynamicShape = false;
        int shapeAlignment = 4;
        bool bilateralSmoothing = false;
        int bilateralDiameter = 5;
        double bilateralSigmaColor = 5.0;
        double bilateralSigmaSpace = 5.0;
        bool cudaSmoothingFailed = false;
        cv::String inputName;
        cv::String outputName;
        cv::Mat blob;
        cv::Mat work;
        cv::Mat converted;
        cv::Mat smoothed;
#ifdef ACMX2_WITH_CUDA
        cv::cuda::GpuMat gpuConverted;
        cv::cuda::GpuMat gpuSmoothed;
#endif

        explicit Impl(const std::string &yamlPath) {
            if (!std::filesystem::exists(yamlPath)) {
                std::cerr << "acmx2: YAML config not found: " << yamlPath << '\n';
                return;
            }

            try {
                const YAML::Node cfg = YAML::LoadFile(yamlPath);
                const std::filesystem::path baseDir =
                    std::filesystem::path(yamlPath).parent_path();
                const std::filesystem::path modelPath =
                    baseDir / cfg["model"]["path"].as<std::string>();
                inputName = cfg["model"]["input"].as<std::string>("");
                if (!std::filesystem::exists(modelPath)) {
                    std::cerr << "acmx2: ONNX model not found: " << modelPath << '\n';
                    return;
                }

#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
                net = cv::dnn::readNetFromONNX(modelPath.string(), cv::dnn::ENGINE_CLASSIC);
#else
                net = cv::dnn::readNetFromONNX(modelPath.string());
#endif
                if (net.empty())
                    return;

                if (cfg["preprocessing"]) {
                    const YAML::Node &pre = cfg["preprocessing"];
                    inputSize.width = pre["width"].as<int>(224);
                    inputSize.height = pre["height"].as<int>(224);
                    scale = pre["scale"].as<double>(1.0 / 255.0);
                    swapRb = pre["swap_rb"].as<bool>(true);
                    dynamicShape = pre["dynamic"].as<bool>(false);
                    shapeAlignment =
                        std::max(1, pre["alignment"].as<int>(4));
                    if (pre["mean"]) {
                        const auto values = pre["mean"].as<std::vector<double>>();
                        if (values.size() >= 3)
                            mean = cv::Scalar(values[0], values[1], values[2]);
                    }
                }
                // The 256x256 dynamic configs enable edge-preserving smoothing
                // by default. A YAML postprocessing block can override it.
                bilateralSmoothing =
                    dynamicShape && inputSize.width > 0 && inputSize.height > 0 &&
                    inputSize.width <= 256 && inputSize.height <= 256;
                if (cfg["postprocessing"] && cfg["postprocessing"]["bilateral"]) {
                    const YAML::Node bilateral =
                        cfg["postprocessing"]["bilateral"];
                    if (bilateral.IsScalar()) {
                        bilateralSmoothing =
                            bilateral.as<bool>(bilateralSmoothing);
                    } else {
                        bilateralSmoothing =
                            bilateral["enabled"].as<bool>(bilateralSmoothing);
                        bilateralDiameter =
                            bilateral["diameter"].as<int>(bilateralDiameter);
                        bilateralSigmaColor =
                            bilateral["sigma_color"].as<double>(
                                bilateralSigmaColor);
                        bilateralSigmaSpace =
                            bilateral["sigma_space"].as<double>(
                                bilateralSigmaSpace);
                    }
                }
                bilateralDiameter = std::max(1, bilateralDiameter);
                if ((bilateralDiameter & 1) == 0)
                    ++bilateralDiameter;

                const std::vector<cv::String> names = net.getUnconnectedOutLayersNames();
                multipleOutputs = names.size() > 1;
                outputName = names.empty() ? cv::String() : names.back();
                loaded = true;
            } catch (const YAML::Exception &error) {
                std::cerr << "acmx2: YAML parse error: " << error.what() << '\n';
            } catch (const cv::Exception &error) {
                std::cerr << "acmx2: Failed to load ONNX model: " << error.what() << '\n';
            }
        }

        cv::Size resolveInputSize(const cv::Size &sourceSize) const {
            if (!dynamicShape)
                return inputSize;

            int width = inputSize.width;
            int height = inputSize.height;
            if (width <= 0 && height <= 0) {
                width = sourceSize.width;
                height = sourceSize.height;
            } else if (width <= 0) {
                width = cvRound(height * static_cast<double>(sourceSize.width) /
                                std::max(sourceSize.height, 1));
            } else if (height <= 0) {
                height = cvRound(width * static_cast<double>(sourceSize.height) /
                                 std::max(sourceSize.width, 1));
            }

            const auto alignDimension = [this](int value) {
                value = std::max(value, shapeAlignment);
                return std::max(shapeAlignment,
                                cvRound(static_cast<double>(value) / shapeAlignment) *
                                    shapeAlignment);
            };
            return {alignDimension(width), alignDimension(height)};
        }

        const cv::Mat &smoothLowResolutionOutput(const cv::Mat &source) {
            if (!bilateralSmoothing)
                return source;
#ifdef ACMX2_WITH_CUDA
            if (backend.usesCuda && !cudaSmoothingFailed) {
                try {
                    gpuConverted.upload(source);
                    cv::cuda::bilateralFilter(
                        gpuConverted, gpuSmoothed, bilateralDiameter,
                        static_cast<float>(bilateralSigmaColor),
                        static_cast<float>(bilateralSigmaSpace));
                    gpuSmoothed.download(smoothed);
                    return smoothed;
                } catch (const cv::Exception &error) {
                    cudaSmoothingFailed = true;
                    std::cerr
                        << "acmx2: CUDA bilateral smoothing failed; using CPU: "
                        << error.what() << '\n';
                }
            }
#endif
            cv::bilateralFilter(source, smoothed, bilateralDiameter,
                                bilateralSigmaColor, bilateralSigmaSpace);
            return smoothed;
        }

        void process(const cv::Mat &image, cv::Mat &output) {
            if (!loaded || inferenceFailed || image.empty())
                return;

            try {
                const cv::Size frameInputSize = resolveInputSize(image.size());
                if (frameInputSize != activeInputSize) {
                    // A backend graph is specialized for its input dimensions.
                    // Re-run selection if a dynamic source changes shape.
                    if (!activeInputSize.empty())
                        backend.selected = false;
                    activeInputSize = frameInputSize;
                }
                cv::dnn::blobFromImage(image, blob, scale, frameInputSize, mean,
                                       swapRb, false, CV_32F);
                const cv::Mat raw =
                    selectBackendAndForward(net, backend, blob, inputName, outputName);

                if (multipleOutputs) {
                    const cv::Mat plane = spatialPlane(raw);
                    if (plane.empty())
                        return;

                    cv::exp(-plane, work);
                    cv::add(work, cv::Scalar::all(1.0), work);
                    cv::divide(1.0, work, work);
                    cv::normalize(work, converted, 0, 255, cv::NORM_MINMAX, CV_8U);
                    const cv::Mat &display = smoothLowResolutionOutput(converted);
                    cv::resize(display, converted, image.size(), 0, 0, cv::INTER_LINEAR);
                    cv::cvtColor(converted, output, cv::COLOR_GRAY2BGR);
                    return;
                }

                if (raw.dims != 4 || raw.size[0] != 1 || raw.type() != CV_32F)
                    return;
                const int channels = raw.size[1];
                const int height = raw.size[2];
                const int width = raw.size[3];
                if (channels == 1) {
                    const cv::Mat plane(height, width, CV_32F,
                                        const_cast<float *>(raw.ptr<float>(0, 0)));
                    cv::normalize(plane, converted, 0, 255, cv::NORM_MINMAX, CV_8U);
                    const cv::Mat &display = smoothLowResolutionOutput(converted);
                    cv::resize(display, converted, image.size(), 0, 0, cv::INTER_LINEAR);
                    cv::cvtColor(converted, output, cv::COLOR_GRAY2BGR);
                    return;
                }
                if (channels < 3)
                    return;

                std::vector<cv::Mat> planes;
                planes.reserve(3);
                for (int channel = 0; channel < 3; ++channel) {
                    planes.emplace_back(height, width, CV_32F,
                                        const_cast<float *>(raw.ptr<float>(0, channel)));
                }
                cv::merge(planes, work);
                cv::normalize(work, converted, 0, 255, cv::NORM_MINMAX, CV_8U);
                const cv::Mat &display = smoothLowResolutionOutput(converted);
                cv::resize(display, converted, image.size(), 0, 0, cv::INTER_LINEAR);
                cv::cvtColor(converted, output, cv::COLOR_RGB2BGR);
            } catch (const cv::Exception &error) {
                inferenceFailed = true;
                std::cerr << "acmx2: OnnxWrapper inference failed (model disabled): "
                          << error.what() << '\n';
            }
        }
    };

    OnnxWrapper::OnnxWrapper(std::string_view yamlPath)
        : impl(std::make_unique<Impl>(std::string(yamlPath))) {}

    OnnxWrapper::~OnnxWrapper() = default;

    void OnnxWrapper::proc(const cv::Mat &image, cv::Mat &output) {
        impl->process(image, output);
    }

    struct Dexined::Impl {
        cv::dnn::Net net;
        BackendState backend;
        cv::String outputName;
        cv::Mat blob;
        cv::Mat work;
        cv::Mat edge;

        explicit Impl(const std::string &modelPath) {
#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
            net = cv::dnn::readNetFromONNX(modelPath, cv::dnn::ENGINE_CLASSIC);
#else
            net = cv::dnn::readNetFromONNX(modelPath);
#endif
            if (net.empty())
                throw std::runtime_error("DexiNed ONNX model is empty");
            outputName = lastOutputName(net);
        }

        void process(const cv::Mat &image, cv::Mat &result) {
            if (image.empty())
                return;

            cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(512, 512),
                                   cv::Scalar(103.5, 116.2, 123.6),
                                   false, false, CV_32F);
            const cv::Mat raw =
                selectBackendAndForward(net, backend, blob, {}, outputName);
            const cv::Mat plane = spatialPlane(raw);
            if (plane.empty())
                return;

            // Only the fused (last) DexiNed output is displayed. The former path
            // transferred and processed every side output and even computed an
            // unused average.
            cv::exp(-plane, work);
            cv::add(work, cv::Scalar::all(1.0), work);
            cv::divide(1.0, work, work);
            cv::normalize(work, edge, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::resize(edge, result, image.size(), 0, 0, cv::INTER_LINEAR);
        }
    };

    Dexined::Dexined(const std::string &modelPath)
        : impl(std::make_unique<Impl>(modelPath)) {}

    Dexined::~Dexined() = default;

    void Dexined::processFrame(const cv::Mat &image, cv::Mat &result) {
        impl->process(image, result);
    }

    struct PPHS::Impl {
        cv::dnn::Net model;
        BackendState backend;
        cv::Size modelInputSize{192, 192};
        cv::Size currentSize;
        const cv::String inputName{"x"};
        const cv::String outputName{"save_infer_model/scale_0.tmp_1"};
        cv::Mat blob;
        cv::Mat logit;
        cv::Mat probability;
        cv::Mat resizedMask;
        cv::Mat previousMask;

        Impl(const std::string &modelPath, int backendId, int targetId) {
#if defined(CV_VERSION_MAJOR) && (CV_VERSION_MAJOR >= 5)
            model = cv::dnn::readNetFromONNX(modelPath, cv::dnn::ENGINE_CLASSIC);
#else
            model = cv::dnn::readNet(modelPath);
#endif
            if (model.empty())
                throw std::runtime_error("PP-HumanSeg model is empty");
            if (backendId >= 0 && targetId >= 0) {
                backend.mode = BackendMode::Explicit;
                backend.explicitBackend = backendId;
                backend.explicitTarget = targetId;
            }
        }

        cv::Mat preprocessImage(const cv::Mat &image) {
            currentSize = image.size();
            // Equivalent to resize -> x/255 -> (x-0.5)/0.5 -> blob, but fused
            // into blobFromImage so there are no GPU round trips or temporary
            // HWC float images.
            cv::dnn::blobFromImage(image, blob, 1.0 / 127.5, modelInputSize,
                                   cv::Scalar(127.5, 127.5, 127.5),
                                   false, false, CV_32F);
            return blob;
        }

        cv::Mat postprocessOutput(const cv::Mat &output) {
            if (output.dims != 4 || output.size[0] != 1 ||
                output.size[1] < 2 || output.type() != CV_32F)
                return {};

            const int height = output.size[2];
            const int width = output.size[3];
            const cv::Mat background(height, width, CV_32F,
                                     const_cast<float *>(output.ptr<float>(0, 0)));
            const cv::Mat foreground(height, width, CV_32F,
                                     const_cast<float *>(output.ptr<float>(0, 1)));

            // Two-class softmax foreground probability is sigmoid(fg - bg).
            // This replaces two exponentials, an add, and a divide.
            cv::subtract(foreground, background, logit);
            cv::exp(-logit, probability);
            cv::add(probability, cv::Scalar::all(1.0), probability);
            cv::divide(1.0, probability, probability);
            cv::resize(probability, resizedMask, currentSize, 0, 0, cv::INTER_CUBIC);

            if (previousMask.empty() || previousMask.size() != resizedMask.size())
                resizedMask.copyTo(previousMask);
            else
                cv::addWeighted(resizedMask, 0.6, previousMask, 0.4, 0.0,
                                previousMask);
            return previousMask;
        }
    };

    PPHS::PPHS(const std::string &modelPath, int backendId, int targetId)
        : impl(std::make_unique<Impl>(modelPath, backendId, targetId)) {}

    PPHS::~PPHS() = default;

    cv::Mat PPHS::preprocess(const cv::Mat &image) {
        return impl->preprocessImage(image);
    }

    cv::Mat PPHS::infer(const cv::Mat &image) {
        if (image.empty())
            return {};
        const cv::Mat input = impl->preprocessImage(image);
        const cv::Mat output =
            selectBackendAndForward(impl->model, impl->backend, input,
                                    impl->inputName, impl->outputName);
        return impl->postprocessOutput(output);
    }

    cv::Mat PPHS::postprocess(const cv::Mat &outputBlob) {
        return impl->postprocessOutput(outputBlob);
    }

    namespace {

        cv::Mat buildHardenedFloatAlpha(const cv::Mat &image, const cv::Mat &mask,
                                        float blackPoint, float whitePoint) {
            constexpr int maximumWorkDimension = 512;
            const int imageMaxDimension = std::max(image.cols, image.rows);
            const double workScale =
                imageMaxDimension > maximumWorkDimension
                    ? static_cast<double>(maximumWorkDimension) / imageMaxDimension
                    : 1.0;
            const cv::Size workSize(
                std::max(1, cvRound(image.cols * workScale)),
                std::max(1, cvRound(image.rows * workScale)));

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
            if (soft.size() != workSize)
                cv::resize(soft, soft, workSize, 0, 0, cv::INTER_LINEAR);
            cv::threshold(soft, soft, 1.0, 1.0, cv::THRESH_TRUNC);
            cv::threshold(soft, soft, 0.0, 0.0, cv::THRESH_TOZERO);

            cv::Mat binary;
            cv::threshold(soft, binary, 0.5f, 255.0f, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8U);
            const auto scaledKernelSize = [workScale](int fullSize) {
                int size = std::max(1, cvRound(fullSize * workScale));
                if ((size & 1) == 0)
                    ++size;
                return size;
            };
            const int openSize = scaledKernelSize(3);
            const int closeSize = scaledKernelSize(7);
            const int erodeSize = scaledKernelSize(3);
            if (openSize > 1) {
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(openSize, openSize));
                cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
            }
            if (closeSize > 1) {
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(closeSize, closeSize));
                cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
            }

            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            const int labelCount = cv::connectedComponentsWithStats(
                binary, labels, stats, centroids, 8, CV_32S);
            if (labelCount > 1) {
                int bestLabel = -1;
                int bestArea = 0;
                for (int label = 1; label < labelCount; ++label) {
                    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                    if (area > bestArea) {
                        bestArea = area;
                        bestLabel = label;
                    }
                }
                const int minimumArea = (binary.cols * binary.rows) / 200;
                if (bestLabel > 0 && bestArea >= minimumArea)
                    cv::compare(labels, bestLabel, binary, cv::CMP_EQ);
            }
            if (erodeSize > 1) {
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(erodeSize, erodeSize));
                cv::erode(binary, binary, kernel);
            }

            cv::Mat silhouette;
            binary.convertTo(silhouette, CV_32F, 1.0 / 255.0);
            cv::multiply(soft, silhouette, soft);
            cv::GaussianBlur(soft, soft, cv::Size(), 1.2 * workScale);

            const float range = std::max(whitePoint - blackPoint, 1.0e-6f);
            soft.convertTo(soft, CV_32F, 1.0f / range, -blackPoint / range);
            cv::threshold(soft, soft, 1.0, 1.0, cv::THRESH_TRUNC);
            cv::threshold(soft, soft, 0.0, 0.0, cv::THRESH_TOZERO);
            cv::pow(soft, 1.6, soft);
            if (soft.size() != image.size())
                cv::resize(soft, soft, image.size(), 0, 0, cv::INTER_LINEAR);
            return soft;
        }

    } // namespace

    cv::Mat hardenedAlphaMask(const cv::Mat &image, const cv::Mat &mask,
                              float blackPoint, float whitePoint) {
        if (image.empty() || mask.empty())
            return {};
        cv::Mat alpha;
        buildHardenedFloatAlpha(image, mask, blackPoint, whitePoint)
            .convertTo(alpha, CV_8U, 255.0);
        return alpha;
    }

    cv::Mat isolateBody(const cv::Mat &image, const cv::Mat &mask,
                        float blackPoint, float whitePoint) {
        if (image.empty() || mask.empty())
            return image.clone();

        cv::Mat alpha;
        buildHardenedFloatAlpha(image, mask, blackPoint, whitePoint)
            .convertTo(alpha, CV_8U, 255.0);
        cv::Mat alphaBgr;
        cv::cvtColor(alpha, alphaBgr, cv::COLOR_GRAY2BGR);
        cv::Mat output;
        cv::multiply(image, alphaBgr, output, 1.0 / 255.0, CV_8UC3);
        return output;
    }

} // namespace ac_dnn
