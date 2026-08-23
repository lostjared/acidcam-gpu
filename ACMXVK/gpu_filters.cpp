#include "gpu_filters.hpp"

#include <ac-gpu/ac-gpu.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace acmxvk::gpu {
    namespace {
        void check_cuda(cudaError_t result, const std::string &operation) {
            if (result != cudaSuccess) {
                throw std::runtime_error(operation + ": " +
                                         cudaGetErrorString(result));
            }
        }

        int validate_frame_buffer_size(int frame_buffer_size) {
            if (frame_buffer_size < 4 || frame_buffer_size > 32) {
                throw std::runtime_error(
                    "GPU frame buffer must be between 4 and 32");
            }
            return frame_buffer_size;
        }
    } // namespace

    class FilterEngine::Impl {
      public:
        Impl(std::vector<int> filter_indices, int frame_buffer_size)
            : frame_buffer(std::make_unique<ac_gpu::DynamicFrameBuffer>(
                  validate_frame_buffer_size(frame_buffer_size))) {
            FilterEngine::validate_filter_indices(filter_indices);
            for (int index : filter_indices) {
                filters.push_back({index, ac_gpu::filters[index].name});
            }

            check_cuda(cudaMalloc(&device_frame_pointers,
                                  static_cast<std::size_t>(frame_buffer_size) *
                                      sizeof(unsigned char *)),
                       "could not allocate CUDA frame-pointer list");
            std::cout << "acmxvk: CUDA filter chain (" << filters.size()
                      << " filters, " << frame_buffer_size
                      << " history frames):\n";
            for (const ac_gpu::Filter &filter : filters) {
                std::cout << "  " << filter.index << ": " << filter.name
                          << '\n';
            }
        }

        ~Impl() {
            if (device_filter_list != nullptr) {
                cudaFree(device_filter_list);
            }
            if (device_frame_pointers != nullptr) {
                cudaFree(device_frame_pointers);
            }
        }

        [[nodiscard]] bool process(const cv::Mat &rgba) {
            if (rgba.empty() || rgba.type() != CV_8UC4) {
                return false;
            }
            frame_buffer->update(rgba);
            if (working_buffer.empty() || working_buffer.cols != rgba.cols ||
                working_buffer.rows != rgba.rows) {
                working_buffer.create(rgba.rows, rgba.cols, CV_8UC4);
            }

            update_parameters();
            check_cuda(
                cudaMemcpy(device_frame_pointers,
                           frame_buffer->getDeviceFramePointers(),
                           static_cast<std::size_t>(frame_buffer->arraySize) *
                               sizeof(unsigned char *),
                           cudaMemcpyHostToDevice),
                "could not upload CUDA frame-pointer list");
            check_cuda(
                cudaMemcpy2D(
                    working_buffer.ptr<unsigned char>(), working_buffer.step,
                    frame_buffer->deviceFrames.back().ptr<unsigned char>(),
                    frame_buffer->framePitch,
                    static_cast<std::size_t>(frame_buffer->w) * 4U,
                    static_cast<std::size_t>(frame_buffer->h),
                    cudaMemcpyDeviceToDevice),
                "could not prepare CUDA filter frame");

            launch_filter(filters.data(), filters.size(),
                          working_buffer.ptr<unsigned char>(),
                          device_frame_pointers, frame_buffer->arraySize,
                          working_buffer.cols, working_buffer.rows,
                          working_buffer.step, alpha, false, square_size,
                          frame_index, frame_direction, &device_filter_list,
                          filters_changed);
            filters_changed = false;
            return true;
        }

        void update_parameters() {
            alpha += alpha_direction > 0 ? 0.01F : -0.01F;
            if (alpha >= 3.0F) {
                alpha = 3.0F;
                alpha_direction = -1;
            } else if (alpha <= 1.0F) {
                alpha = 1.0F;
                alpha_direction = 1;
            }

            frame_index += frame_direction;
            const int last = frame_buffer->arraySize - 1;
            if (frame_index >= last) {
                frame_index = last;
                frame_direction = -1;
            } else if (frame_index <= 0) {
                frame_index = 0;
                frame_direction = 1;
            }
        }

        std::vector<ac_gpu::Filter> filters;
        std::unique_ptr<ac_gpu::DynamicFrameBuffer> frame_buffer;
        cv::cuda::GpuMat working_buffer;
        cv::cuda::Stream upload_stream;
        unsigned char **device_frame_pointers = nullptr;
        ac_gpu::GPUFilter *device_filter_list = nullptr;
        bool filters_changed = true;
        float alpha = 1.0F;
        int alpha_direction = 1;
        int square_size = 8;
        int frame_index = 0;
        int frame_direction = 1;
    };

    FilterEngine::FilterEngine(std::vector<int> filter_indices,
                               int frame_buffer_size)
        : impl(std::make_unique<Impl>(std::move(filter_indices),
                                      frame_buffer_size)) {}

    FilterEngine::~FilterEngine() = default;

    bool FilterEngine::process(const cv::Mat &rgba) {
        return impl->process(rgba);
    }

    const cv::cuda::GpuMat &FilterEngine::output() const {
        return impl->working_buffer;
    }

    cv::cuda::Stream &FilterEngine::stream() {
        return impl->upload_stream;
    }

    void FilterEngine::select_device(int device_index) {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count),
                   "could not enumerate CUDA devices");
        if (device_index < 0 || device_index >= device_count) {
            throw std::runtime_error(
                "CUDA device index must be between 0 and " +
                std::to_string(std::max(device_count - 1, 0)));
        }
        check_cuda(cudaSetDevice(device_index), "could not select CUDA device");
        const cv::cuda::DeviceInfo device(device_index);
        std::cout << "acmxvk: CUDA device " << device_index << ": "
                  << device.name() << '\n';
    }

    void FilterEngine::validate_filter_indices(
        const std::vector<int> &filter_indices) {
        if (filter_indices.empty()) {
            throw std::runtime_error("CUDA filter list cannot be empty");
        }
        for (int index : filter_indices) {
            if (index < 0 || index >= ac_gpu::AC_FILTER_MAX) {
                throw std::runtime_error(
                    "CUDA filter index must be between 0 and " +
                    std::to_string(ac_gpu::AC_FILTER_MAX - 1) + ": " +
                    std::to_string(index));
            }
        }
    }

    void FilterEngine::list_devices(std::ostream &output_stream) {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count),
                   "could not enumerate CUDA devices");
        output_stream << "acmxvk: found " << device_count
                      << " CUDA device(s)\n";
        for (int index = 0; index < device_count; ++index) {
            const cv::cuda::DeviceInfo device(index);
            output_stream << "  " << index << ": " << device.name() << " ("
                          << (device.totalMemory() / (1024U * 1024U))
                          << " MiB)\n";
        }
    }

    void FilterEngine::list_filters(std::ostream &output_stream) {
        output_stream << "acmxvk: found " << ac_gpu::AC_FILTER_MAX
                      << " CUDA filter(s)\n";
        for (int index = 0; index < ac_gpu::AC_FILTER_MAX; ++index) {
            output_stream << "  " << index << ": "
                          << ac_gpu::filters[index].name << '\n';
        }
    }

} // namespace acmxvk::gpu
