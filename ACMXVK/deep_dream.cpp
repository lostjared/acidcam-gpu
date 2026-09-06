#include "deep_dream.hpp"
#include "deep_dream_model.hpp"

#include <torch/cuda.h>
#include <torch/headeronly/version.h>
#include <torch/torch.h>

#include <exception>
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
                             std::string_view layer, std::ostream &output,
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
                Model model = Model::load(model_file, cuda_device, layer);
                model.print(output);
            }
        } catch (const std::exception &exception) {
            error << "Deep Dream LibTorch probe failed: " << exception.what()
                  << '\n';
            return false;
        }
        return true;
    }

} // namespace acmxvk::dream
