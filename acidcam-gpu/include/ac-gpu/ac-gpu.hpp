#ifndef __AC_GPU_HPP__
#define __AC_GPU_HPP__

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace ac_gpu {

    inline const int AC_FILTER_MAX = 11;

    struct Filter {
        int index;
        std::string name;
    };

    class ACException {
    public:
        ACException(const std::string &text) : txt{text} {}
        std::string why() const { return txt; }
    private:
        std::string txt;
    };
    
    class DynamicFrameBuffer {
    public:
        int arraySize;
        int w, h;
        size_t framePitch;
        size_t frameSize;
        std::vector<unsigned char*> deviceFrames;
        int completedFrames;

        DynamicFrameBuffer(int size) : arraySize(size), w(0), h(0), framePitch(0), frameSize(0), completedFrames(0) {
            deviceFrames.assign(size, nullptr); 
        }

        ~DynamicFrameBuffer() { release(); }

        void update(cv::Mat &inputFrame) {
            bool resized = checkResize(inputFrame.cols, inputFrame.rows);

            cv::Mat rgba;
            if (inputFrame.channels() == 3) {
                cv::cvtColor(inputFrame, rgba, cv::COLOR_BGR2RGBA);
            } else {
                rgba = inputFrame;
            }

            if (resized) {
                fillAll(rgba);
                completedFrames = 0;
            } else {
                shiftAndAdd(rgba);
                if (completedFrames < arraySize) ++completedFrames;
            }
        }

        void release() {
            for (size_t i = 0; i < deviceFrames.size(); ++i) {
                if (deviceFrames[i] != nullptr) {
                    cudaFree(deviceFrames[i]);
                    deviceFrames[i] = nullptr; 
                }
            }
        }

    private:
        bool checkResize(int newW, int newH) {
            if (newW != w || newH != h) {
                w = newW;
                h = newH;
                release(); 
                for (int i = 0; i < arraySize; ++i) {
                    cudaError_t err = cudaMallocPitch(&deviceFrames[i], &framePitch, w * 4, h);
                    if (err != cudaSuccess) {
                        deviceFrames[i] = nullptr; 
                    }
                }
                frameSize = framePitch * h;
                return true;
            }
            return false;
        }

        void fillAll(cv::Mat &rgba) {
            for (int i = 0; i < arraySize; ++i) {
                if (deviceFrames[i])
                    cudaMemcpy2D(deviceFrames[i], framePitch, rgba.data, w * 4, w * 4, h, cudaMemcpyHostToDevice);
            }
        }

        void shiftAndAdd(cv::Mat &rgba) {
            unsigned char* oldestFramePtr = deviceFrames[0];
            for (int i = 0; i < arraySize - 1; ++i) {
                deviceFrames[i] = deviceFrames[i + 1];
            }
            deviceFrames[arraySize - 1] = oldestFramePtr;
            cudaMemcpy2D(deviceFrames[arraySize - 1], framePitch, 
                        rgba.data, w * 4, 
                        w * 4, h, 
                        cudaMemcpyHostToDevice);
        }
    };

    extern Filter filters[];
}

extern "C" {
    void launch_filter(ac_gpu::Filter *f_host, size_t c, unsigned char* data, unsigned char** allFrames,
                            int numFrames, int width, int height, size_t step,
                            float alpha, bool isNegative, int square_size,
                            int start_index, int start_dir, 
                            ac_gpu::Filter** d_list_ptr, bool& changed);
}

#endif