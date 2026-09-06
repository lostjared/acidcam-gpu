#include "deep_dream_model.hpp"

#include "input_validation.hpp"

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace acmxvk::dream {
    namespace {
        constexpr std::uintmax_t MAX_MODEL_BYTES = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        constexpr std::size_t MAX_LAYERS = 128;
        constexpr std::int64_t MAX_SOURCE_LAYER_INDEX = 4096;
        constexpr std::int64_t MAX_TEST_INPUT_SIZE = 256;
        constexpr int MAX_GRADIENT_ASCENT_ITERATIONS = 100;
        constexpr float MAX_GRADIENT_ASCENT_STEP = 10.0F;
        constexpr float GRADIENT_EPSILON = 1.0e-8F;

        [[nodiscard]] c10::IValue require_attribute(
            const torch::jit::Module &module, const std::string &name) {
            if (!module.hasattr(name)) {
                throw std::runtime_error("Deep Dream model is missing metadata attribute '" +
                                         name + "'");
            }
            return module.attr(name);
        }

        [[nodiscard]] std::string string_attribute(
            const torch::jit::Module &module, const std::string &name) {
            const c10::IValue value = require_attribute(module, name);
            if (!value.isString()) {
                throw std::runtime_error("Deep Dream metadata attribute '" + name +
                                         "' must be a string");
            }
            return value.toStringRef();
        }

        [[nodiscard]] std::int64_t integer_attribute(
            const torch::jit::Module &module, const std::string &name) {
            const c10::IValue value = require_attribute(module, name);
            if (!value.isInt()) {
                throw std::runtime_error("Deep Dream metadata attribute '" + name +
                                         "' must be an integer");
            }
            return value.toInt();
        }

        [[nodiscard]] std::vector<std::string> string_list_attribute(
            const torch::jit::Module &module, const std::string &name) {
            const c10::IValue value = require_attribute(module, name);
            if (!value.isList()) {
                throw std::runtime_error("Deep Dream metadata attribute '" + name +
                                         "' must be a string list");
            }
            std::vector<std::string> result;
            result.reserve(value.toListRef().size());
            for (const c10::IValue &entry : value.toListRef()) {
                if (!entry.isString()) {
                    throw std::runtime_error("Deep Dream metadata attribute '" +
                                             name + "' must contain strings");
                }
                result.push_back(entry.toStringRef());
            }
            return result;
        }

        [[nodiscard]] std::vector<std::int64_t> integer_list_attribute(
            const torch::jit::Module &module, const std::string &name) {
            const c10::IValue value = require_attribute(module, name);
            if (!value.isIntList()) {
                throw std::runtime_error("Deep Dream metadata attribute '" + name +
                                         "' must be an integer list");
            }
            return value.toIntVector();
        }

        [[nodiscard]] std::vector<double> double_list_attribute(
            const torch::jit::Module &module, const std::string &name) {
            const c10::IValue value = require_attribute(module, name);
            if (!value.isDoubleList()) {
                throw std::runtime_error("Deep Dream metadata attribute '" + name +
                                         "' must be a float list");
            }
            return value.toDoubleVector();
        }

        void validate_normalization(const std::vector<double> &values,
                                    bool standard_deviation) {
            if (values.size() != 3) {
                throw std::runtime_error(
                    "Deep Dream input normalization must contain three channels");
            }
            for (double value : values) {
                if (!std::isfinite(value) || value < 0.0 || value > 10.0 ||
                    (standard_deviation && value <= 0.0)) {
                    throw std::runtime_error(
                        "Deep Dream input normalization contains an invalid value");
                }
            }
        }

        [[nodiscard]] ModelMetadata read_metadata(
            const torch::jit::Module &module) {
            if (string_attribute(module, "acmxvk_deep_dream_format") !=
                "acmxvk-deep-dream") {
                throw std::runtime_error(
                    "TorchScript file is not an ACMXVK Deep Dream model");
            }
            if (integer_attribute(module, "acmxvk_deep_dream_version") != 1) {
                throw std::runtime_error(
                    "Deep Dream model metadata version is unsupported");
            }

            ModelMetadata metadata;
            metadata.architecture = string_attribute(module, "architecture");
            metadata.default_layer = string_attribute(module, "default_layer");
            metadata.input_channels =
                integer_attribute(module, "input_channels");
            metadata.minimum_input_size =
                integer_attribute(module, "minimum_input_size");
            metadata.input_mean = double_list_attribute(module, "input_mean");
            metadata.input_std = double_list_attribute(module, "input_std");
            const std::vector<std::string> names =
                string_list_attribute(module, "layer_names");
            const std::vector<std::int64_t> source_indices =
                integer_list_attribute(module, "layer_source_indices");

            input::validate_string(metadata.architecture,
                                   input::StringKind::Token,
                                   "Deep Dream model architecture");
            input::validate_string(metadata.default_layer,
                                   input::StringKind::Identifier,
                                   "Deep Dream default layer");
            if (metadata.input_channels != 3 || metadata.minimum_input_size < 32 ||
                metadata.minimum_input_size > MAX_TEST_INPUT_SIZE) {
                throw std::runtime_error(
                    "Deep Dream model has unsupported input dimensions");
            }
            validate_normalization(metadata.input_mean, false);
            validate_normalization(metadata.input_std, true);
            if (names.empty() || names.size() > MAX_LAYERS ||
                names.size() != source_indices.size()) {
                throw std::runtime_error(
                    "Deep Dream model has an invalid feature-layer table");
            }

            std::unordered_set<std::string> unique_names;
            std::int64_t previous_index = -1;
            for (std::size_t index = 0; index < names.size(); ++index) {
                input::validate_string(names[index], input::StringKind::Identifier,
                                       "Deep Dream feature-layer name");
                if (!unique_names.insert(names[index]).second ||
                    source_indices[index] <= previous_index ||
                    source_indices[index] > MAX_SOURCE_LAYER_INDEX) {
                    throw std::runtime_error(
                        "Deep Dream model has an invalid feature-layer table");
                }
                metadata.layers.push_back(
                    LayerMetadata{names[index], source_indices[index]});
                previous_index = source_indices[index];
            }
            if (!unique_names.contains(metadata.default_layer)) {
                throw std::runtime_error(
                    "Deep Dream default layer is absent from the layer table");
            }
            return metadata;
        }

        [[nodiscard]] std::size_t resolve_layer(
            const ModelMetadata &metadata, std::string_view selector) {
            if (selector.empty()) {
                selector = metadata.default_layer;
            }
            std::size_t numeric_layer = 0;
            const char *begin = selector.data();
            const char *end = begin + selector.size();
            const auto [position, error] =
                std::from_chars(begin, end, numeric_layer);
            if (error == std::errc{} && position == end) {
                if (numeric_layer >= metadata.layers.size()) {
                    throw std::runtime_error(
                        "Deep Dream layer index is outside the model's range");
                }
                return numeric_layer;
            }
            const auto match = std::find_if(
                metadata.layers.begin(), metadata.layers.end(),
                [selector](const LayerMetadata &layer) {
                    return layer.name == selector;
                });
            if (match == metadata.layers.end()) {
                throw std::runtime_error("Deep Dream layer is not present in model: " +
                                         std::string(selector));
            }
            return static_cast<std::size_t>(
                std::distance(metadata.layers.begin(), match));
        }

        [[nodiscard]] std::vector<torch::Tensor> feature_outputs(
            const c10::IValue &value) {
            if (!value.isTensorList()) {
                throw std::runtime_error(
                    "Deep Dream model forward method must return List[Tensor]");
            }
            return value.toTensorVector();
        }

        void validate_gradient_options(const GradientAscentOptions &options) {
            if (options.iterations < 1 ||
                options.iterations > MAX_GRADIENT_ASCENT_ITERATIONS) {
                throw std::runtime_error(
                    "Deep Dream iterations must be between 1 and 100");
            }
            if (!std::isfinite(options.step_size) || options.step_size <= 0.0F ||
                options.step_size > MAX_GRADIENT_ASCENT_STEP) {
                throw std::runtime_error(
                    "Deep Dream step size must be greater than 0 and no more than 10");
            }
        }

        [[nodiscard]] torch::Tensor normalization_tensor(
            const std::vector<double> &values, const torch::Device &device) {
            return torch::tensor(
                       values,
                       torch::TensorOptions().dtype(torch::kFloat32).device(device))
                .view({1, 3, 1, 1});
        }
    } // namespace

    struct Model::Impl {
        torch::jit::Module module;
        ModelMetadata metadata;
        std::filesystem::path filename;
        std::vector<std::vector<std::int64_t>> output_shapes;
        std::size_t selected_layer = 0;
        int cuda_device = 0;
    };

    Model::Model(std::unique_ptr<Impl> implementation)
        : implementation(std::move(implementation)) {}

    Model::Model(Model &&) noexcept = default;
    Model &Model::operator=(Model &&) noexcept = default;
    Model::~Model() = default;

    [[nodiscard]] Model Model::load(std::string_view filename, int cuda_device,
                                    std::string_view layer) {
        input::validate_string(filename, input::StringKind::Path,
                               "Deep Dream model path");
        const std::filesystem::path model_path =
            std::filesystem::absolute(filename).lexically_normal();
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("Deep Dream model is not a regular file: " +
                                     model_path.string());
        }
        input::validate_file_size(model_path, "Deep Dream TorchScript model",
                                  MAX_MODEL_BYTES);

        const c10::DeviceIndex device_count = torch::cuda::device_count();
        if (!torch::cuda::is_available() || device_count == 0) {
            throw std::runtime_error(
                "Deep Dream model loading requires an available CUDA device");
        }
        if (cuda_device < 0 || cuda_device >= device_count) {
            throw std::runtime_error("Deep Dream CUDA device index is outside "
                                     "the available range");
        }

        const torch::Device device(torch::kCUDA, cuda_device);
        torch::jit::Module module = torch::jit::load(model_path.string(), device);
        module.eval();
        for (torch::Tensor parameter : module.parameters()) {
            parameter.set_requires_grad(false);
        }

        auto implementation = std::make_unique<Impl>();
        implementation->metadata = read_metadata(module);
        implementation->selected_layer =
            resolve_layer(implementation->metadata, layer);
        implementation->filename = model_path;
        implementation->cuda_device = cuda_device;
        implementation->module = std::move(module);

        const std::int64_t input_size =
            std::max<std::int64_t>(64, implementation->metadata.minimum_input_size);
        torch::NoGradGuard no_grad;
        const torch::Tensor input = torch::zeros(
            {1, implementation->metadata.input_channels, input_size, input_size},
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        const std::vector<torch::Tensor> outputs = feature_outputs(
            implementation->module.forward({input}));
        if (outputs.size() != implementation->metadata.layers.size()) {
            throw std::runtime_error(
                "Deep Dream model output count does not match its metadata");
        }
        implementation->output_shapes.reserve(outputs.size());
        for (const torch::Tensor &output : outputs) {
            if (!output.defined() || output.dim() != 4 || output.size(0) != 1 ||
                !output.is_floating_point() || !output.device().is_cuda() ||
                output.get_device() != cuda_device || output.size(2) <= 0 ||
                output.size(3) <= 0) {
                throw std::runtime_error(
                    "Deep Dream model returned an invalid feature tensor");
            }
            implementation->output_shapes.push_back(output.sizes().vec());
        }
        torch::cuda::synchronize(cuda_device);
        return Model(std::move(implementation));
    }

    [[nodiscard]] const ModelMetadata &Model::metadata() const {
        return implementation->metadata;
    }

    [[nodiscard]] std::size_t Model::selected_layer() const {
        return implementation->selected_layer;
    }

    [[nodiscard]] GradientAscentResult Model::apply_gradient_ascent(
        cv::Mat &rgba, const GradientAscentOptions &options) {
        validate_gradient_options(options);
        if (rgba.empty() || rgba.type() != CV_8UC4 || rgba.cols < 1 ||
            rgba.rows < 1) {
            throw std::runtime_error(
                "Deep Dream input must be a non-empty RGBA8 image");
        }

        cv::Mat rgb;
        cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
        cv::Mat rgb_float;
        rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

        const torch::Device device(torch::kCUDA,
                                   implementation->cuda_device);
        const torch::Tensor input =
            torch::from_blob(rgb_float.data, {rgb_float.rows, rgb_float.cols, 3},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .permute({2, 0, 1})
                .unsqueeze(0)
                .to(device)
                .contiguous();
        const torch::Tensor mean = normalization_tensor(
            implementation->metadata.input_mean, device);
        const torch::Tensor standard_deviation = normalization_tensor(
            implementation->metadata.input_std, device);
        torch::Tensor dream_input =
            ((input - mean) / standard_deviation).detach();
        dream_input.requires_grad_(true);
        const torch::Tensor minimum = (torch::zeros_like(mean) - mean) /
                                      standard_deviation;
        const torch::Tensor maximum = (torch::ones_like(mean) - mean) /
                                      standard_deviation;

        GradientAscentResult result;
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            const std::vector<torch::Tensor> outputs = feature_outputs(
                implementation->module.forward({dream_input}));
            if (outputs.size() != implementation->metadata.layers.size()) {
                throw std::runtime_error(
                    "Deep Dream model output count changed during gradient ascent");
            }
            const torch::Tensor activation =
                outputs[implementation->selected_layer];
            const torch::Tensor loss = activation.square().mean();
            if (!torch::isfinite(loss).item<bool>()) {
                throw std::runtime_error(
                    "Deep Dream activation loss is not finite");
            }
            loss.backward();

            torch::Tensor gradient = dream_input.grad();
            if (!gradient.defined() || !torch::isfinite(gradient).all().item<bool>()) {
                throw std::runtime_error(
                    "Deep Dream produced an invalid input gradient");
            }
            const torch::Tensor mean_gradient = gradient.abs().mean();
            const float gradient_value = mean_gradient.item<float>();
            if (!std::isfinite(gradient_value) ||
                gradient_value <= GRADIENT_EPSILON) {
                throw std::runtime_error(
                    "Deep Dream selected layer produced no usable input gradient");
            }

            {
                torch::NoGradGuard no_grad;
                dream_input.add_(gradient *
                                 (options.step_size /
                                  (mean_gradient + GRADIENT_EPSILON)));
                dream_input.copy_(torch::maximum(
                    torch::minimum(dream_input, maximum), minimum));
            }
            dream_input.grad().zero_();
            result.activation_loss = loss.item<float>();
            result.mean_gradient = gradient_value;
        }

        torch::Tensor output =
            (dream_input.detach() * standard_deviation + mean)
                .clamp(0.0F, 1.0F);
        result.mean_pixel_change =
            (output - input).abs().mean().item<float>();
        output = output.squeeze(0)
                     .permute({1, 2, 0})
                     .mul(255.0F)
                     .round()
                     .to(torch::kUInt8)
                     .to(torch::kCPU)
                     .contiguous();
        torch::cuda::synchronize(implementation->cuda_device);

        cv::Mat dreamed_rgb(rgb.rows, rgb.cols, CV_8UC3,
                            output.data_ptr<std::uint8_t>());
        cv::Mat dreamed_rgba;
        cv::cvtColor(dreamed_rgb, dreamed_rgba, cv::COLOR_RGB2RGBA);
        std::vector<cv::Mat> original_channels;
        std::vector<cv::Mat> dreamed_channels;
        cv::split(rgba, original_channels);
        cv::split(dreamed_rgba, dreamed_channels);
        dreamed_channels[3] = original_channels[3];
        cv::merge(dreamed_channels, rgba);
        return result;
    }

    void Model::print(std::ostream &output) const {
        output << "Deep Dream model: " << implementation->filename.string()
               << '\n'
               << "Deep Dream architecture: "
               << implementation->metadata.architecture << '\n'
               << "Deep Dream CUDA device: " << implementation->cuda_device
               << '\n'
               << "Deep Dream feature layers: "
               << implementation->metadata.layers.size() << '\n';
        for (std::size_t index = 0;
             index < implementation->metadata.layers.size(); ++index) {
            const LayerMetadata &layer = implementation->metadata.layers[index];
            output << "  " << index << ": " << layer.name << " (source "
                   << layer.source_index << ')';
            if (index == implementation->selected_layer) {
                output << " [selected]";
            }
            output << '\n';
        }
        const std::vector<std::int64_t> &shape =
            implementation->output_shapes[implementation->selected_layer];
        output << "Deep Dream selected activation: ";
        for (std::size_t index = 0; index < shape.size(); ++index) {
            if (index != 0) {
                output << 'x';
            }
            output << shape[index];
        }
        output << '\n';
    }

} // namespace acmxvk::dream
