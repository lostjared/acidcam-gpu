#ifndef __AC_GPU_HPP__
#define __AC_GPU_HPP__

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
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

    inline const int AC_FILTER_MAX = 736;
    
    
    struct GPUFilter {
        int index;
    };
    
    struct Filter {
        int index;
        std::string name;
        GPUFilter toGPU() const { return {index}; }
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
        std::vector<cv::cuda::GpuMat> deviceFrames; 
        std::vector<unsigned char*> rawPointers;    
        int completedFrames;
        cv::cuda::GpuMat d_uploadBuffer;

        DynamicFrameBuffer(int size) : arraySize(size), w(0), h(0), framePitch(0), completedFrames(0) {
            deviceFrames.resize(size);
            rawPointers.resize(size, nullptr);
        }

        void update(const cv::Mat& inputFrame) {
            d_uploadBuffer.upload(inputFrame);
            if (d_uploadBuffer.cols != w || d_uploadBuffer.rows != h) {
                w = d_uploadBuffer.cols;
                h = d_uploadBuffer.rows;
                
                for (int i = 0; i < arraySize; ++i) {
                    deviceFrames[i].create(h, w, CV_8UC4);
                }
                framePitch = deviceFrames[0].step;
                completedFrames = 0;
            }
            
            std::rotate(deviceFrames.begin(), deviceFrames.begin() + 1, deviceFrames.end());
            
            if (d_uploadBuffer.channels() == 3) {
                cv::cuda::cvtColor(d_uploadBuffer, deviceFrames.back(), cv::COLOR_BGR2RGBA);
            } else {
                d_uploadBuffer.copyTo(deviceFrames.back());
            }

            for(int i=0; i<arraySize; ++i) {
                rawPointers[i] = deviceFrames[i].data;
            }

            if (completedFrames < arraySize) ++completedFrames;
        }
        
        unsigned char** getDeviceFramePointers() {
            return rawPointers.data();
        }
    };
    extern Filter filters[];
} 

extern "C" {
    void launch_filter(ac_gpu::Filter *f_host, size_t c, unsigned char* data, unsigned char** allFrames,
                            int numFrames, int width, int height, size_t step,
                            float alpha, bool isNegative, int square_size,
                            int start_index, int start_dir, 
                            ac_gpu::GPUFilter** d_list_ptr, bool& changed);
}

#endif