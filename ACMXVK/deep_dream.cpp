#include "deep_dream.hpp"
#include "deep_dream_model.hpp"

#include <torch/cuda.h>
#include <torch/headeronly/version.h>
#include <torch/torch.h>

#include <exception>
#include <iomanip>
#include <ostream>

namespace acmxvk::dream {
    namespace {
        [[nodiscard]] bool autograd_smoke_test(const torch::Device &device) {
            torch::Tensor input = torch::ones(
                {1, 1, 2, 2},
                torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device)
                    .requires_grad(true));
            const torch::Tensor loss = input.square().mean();
            loss.backward();
            const torch::Tensor expected = torch::full_like(input, 0.5F);
            return input.grad().defined() &&
                   torch::allclose(input.grad(), expected);
        }
    } // namespace

    [[nodiscard]] bool probe(int cuda_device, std::string_view model_file,
                             std::string_view layer, int iterations,
                             float strength, float feedback, float zoom,
                             float rotation_degrees, int max_dimension,
                             bool use_half, int target_channel,
                             int octaves, float octave_scale,
                             int jitter, int smoothing,
                             std::ostream &output,
                             std::ostream &error) {
        output << "Deep Dream: enabled\n"
               << "LibTorch version: " << TORCH_VERSION << '\n';

        try {
            if (!autograd_smoke_test(torch::Device(torch::kCPU))) {
                error << "Deep Dream LibTorch CPU autograd smoke test failed\n";
                return false;
            }
            output << "LibTorch CPU autograd: ready\n";

            const c10::DeviceIndex device_count = torch::cuda::device_count();
            output << "LibTorch CUDA devices: "
                   << static_cast<int>(device_count) << '\n';
            if (!torch::cuda::is_available() || device_count == 0) {
                output << "LibTorch CUDA autograd: unavailable "
                          "(no CUDA device visible)\n";
                if (!model_file.empty()) {
                    error << "Deep Dream model inspection requires an "
                             "available CUDA device\n";
                    return false;
                }
                return true;
            }
            if (cuda_device < 0 || cuda_device >= device_count) {
                error << "Deep Dream CUDA device index " << cuda_device
                      << " is outside the available range 0-"
                      << (device_count - 1) << '\n';
                return false;
            }

            const torch::Device device(torch::kCUDA, cuda_device);
            if (!autograd_smoke_test(device)) {
                error << "Deep Dream LibTorch CUDA autograd smoke test failed "
                         "on device "
                      << cuda_device << '\n';
                return false;
            }
            torch::cuda::synchronize(cuda_device);
            output << "LibTorch CUDA autograd: ready on device " << cuda_device
                   << '\n'
                   << "LibTorch cuDNN: "
                   << (torch::cuda::cudnn_is_available() ? "ready" : "unavailable")
                   << '\n';

            if (!model_file.empty()) {
                Model model =
                    Model::load(model_file, cuda_device, layer, use_half);
                model.print(output);
                output << "Deep Dream target channel: ";
                if (target_channel < 0) {
                    output << "all\n";
                } else {
                    output << target_channel << '\n';
                }
                cv::Mat test_image(64, 64, CV_8UC4);
                for (int y = 0; y < test_image.rows; ++y) {
                    for (int x = 0; x < test_image.cols; ++x) {
                        test_image.at<cv::Vec4b>(y, x) = cv::Vec4b{
                            static_cast<std::uint8_t>((x * 255) /
                                                      (test_image.cols - 1)),
                            static_cast<std::uint8_t>((y * 255) /
                                                      (test_image.rows - 1)),
                            static_cast<std::uint8_t>(((x + y) * 255) /
                                                      (test_image.cols +
                                                       test_image.rows - 2)),
                            255U};
                    }
                }
                GradientAscentResult result = model.apply_gradient_ascent(
                    test_image,
                    GradientAscentOptions{iterations, strength, feedback, zoom,
                                          rotation_degrees, max_dimension,
                                          target_channel, octaves,
                                          octave_scale, jitter, smoothing});
                output << std::fixed << std::setprecision(6)
                       << "Deep Dream gradient ascent: ready"
                       << " (loss=" << result.activation_loss
                       << ", mean gradient=" << result.mean_gradient
                       << ", mean pixel change="
                       << result.mean_pixel_change << ", working size="
                       << result.processed_width << 'x'
                       << result.processed_height << ", octaves="
                       << result.processed_octaves << ", jitter=" << jitter
                       << ", smoothing=" << smoothing << ")\n";
                if (feedback > 0.0F) {
                    result = model.apply_gradient_ascent(
                        test_image,
                        GradientAscentOptions{iterations, strength, feedback,
                                              zoom, rotation_degrees,
                                              max_dimension, target_channel,
                                              octaves, octave_scale, jitter,
                                              smoothing});
                    output << "Deep Dream temporal feedback: ready"
                           << " (blend=" << feedback << ", zoom=" << zoom
                           << ", rotation=" << rotation_degrees
                           << ", next-frame change="
                           << result.mean_pixel_change << ")\n";
                }
            }
        } catch (const std::exception &exception) {
            error << "Deep Dream LibTorch probe failed: " << exception.what()
                  << '\n';
            return false;
        }
        return true;
    }

} // namespace acmxvk::dream
