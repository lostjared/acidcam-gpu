#ifndef ACMX2_DNN_HPP
#define ACMX2_DNN_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <map>
#include <vector>
#include <string>
#include <iostream>

namespace ac_dnn {
    using namespace std;
    using namespace cv;
    using namespace dnn;

  class OnnxWrapper {
  private:
        cv::dnn::Net net;
        bool is_loaded = false;
        bool inference_failed = false;
        bool use_cuda_ = false;
        cv::Size input_size = {224, 224};
        double scale = 1.0 / 255.0;
        cv::Scalar mean = {0.0, 0.0, 0.0};
        bool swap_rb = true;

        void loadFromYaml(const std::string &yaml_path) {
            if (!std::filesystem::exists(yaml_path)) {
                std::cerr << "acmx2: YAML config not found: " << yaml_path << '\n';
                return;
            }
            try {
                YAML::Node cfg = YAML::LoadFile(yaml_path);
                std::string model_rel_path = cfg["model"]["path"].as<std::string>();
                std::filesystem::path base_dir = std::filesystem::path(yaml_path).parent_path();
                std::filesystem::path full_path = base_dir / model_rel_path;
                if (!std::filesystem::exists(full_path)) {
                    std::cerr << "acmx2: ONNX model not found: " << full_path << '\n';
                    return;
                }
                std::string model_path_str = full_path.string();
                net = cv::dnn::readNetFromONNX(model_path_str);
                if (net.empty()) return;
                if (cfg["preprocessing"]) {
                    const YAML::Node &pre = cfg["preprocessing"];
                    input_size.width  = pre["width"].as<int>(224);
                    input_size.height = pre["height"].as<int>(224);
                    scale    = pre["scale"].as<double>(1.0 / 255.0);
                    swap_rb  = pre["swap_rb"].as<bool>(true);
                    if (pre["mean"]) {
                        auto v = pre["mean"].as<std::vector<double>>();
                        if (v.size() >= 3)
                            mean = cv::Scalar(v[0], v[1], v[2]);
                    }
                }
                //    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                //    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                optimizeNeuralNet();
                is_loaded = true;
            } catch (const YAML::Exception &e) {
                std::cerr << "acmx2: YAML parse error: " << e.what() << '\n';
            } catch (const cv::Exception &e) {
                std::cerr << "acmx2: Failed to load ONNX model: " << e.what() << '\n';
            }
        }

        void optimizeNeuralNet() {
           auto available_backends = cv::dnn::getAvailableBackends();
           auto is_supported = [&available_backends](cv::dnn::Backend backend, cv::dnn::Target target) {
                return std::find(available_backends.begin(), available_backends.end(), 
                                std::make_pair(backend, target)) != available_backends.end();
            };

            if (is_supported(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16)) {
                std::cout << "[ACMX2] Inference hardware: CUDA (FP16)\n";
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                use_cuda_ = true;
                return;
            }
            if (is_supported(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA)) {
                std::cout << "[ACMX2] Inference hardware: CUDA (FP32)\n";
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                use_cuda_ = true;
                return;
            }

            if (is_supported(cv::dnn::DNN_BACKEND_VKCOM, cv::dnn::DNN_TARGET_VULKAN)) {
                std::cout << "[ACMX2] Inference hardware: Vulkan\n";
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
                return;
            }

            if (is_supported(cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL_FP16)) {
                std::cout << "[ACMX2] Inference hardware: OpenCL (FP16)\n";
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
                return;
            }
            if (is_supported(cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL)) {
                std::cout << "[ACMX2] Inference hardware: OpenCL (FP32)\n";
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                return;
            }
            std::cout << "[ACMX2] Inference hardware: CPU (No GPU acceleration found)\n";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

  public:
        /// Construct from a YAML config file that specifies model path and preprocessing.
        explicit OnnxWrapper(std::string_view yaml_path) {
            loadFromYaml(std::string(yaml_path));
        }

        void proc(const cv::Mat &image, cv::Mat &output) {
            if (!is_loaded || inference_failed) return;
            if (image.empty()) return;
            try {
                const int orig_h = image.rows;
                const int orig_w = image.cols;

                cv::Mat blob;
                if (use_cuda_) {
                    cv::cuda::GpuMat g_frame, g_resized;
                    g_frame.upload(image);
                    cv::cuda::resize(g_frame, g_resized, input_size);
                    if (swap_rb)
                        cv::cuda::cvtColor(g_resized, g_resized, cv::COLOR_BGR2RGB);
                    cv::Mat resized_cpu;
                    g_resized.download(resized_cpu);
                    blob = cv::dnn::blobFromImage(resized_cpu, scale, {}, mean, false, false);
                } else {
                    blob = cv::dnn::blobFromImage(image, scale, input_size, mean, swap_rb, false);
                }
                net.setInput(blob);

                std::vector<cv::Mat> outputs;
                net.forward(outputs);
                if (outputs.empty()) return;

                if (outputs.size() > 1) {
                    if (use_cuda_) {
                        std::vector<cv::cuda::GpuMat> g_preds;
                        g_preds.reserve(outputs.size());
                        for (const cv::Mat &p : outputs) {
                            cv::Mat processed;
                            if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1)
                                processed = p.reshape(0, {p.size[2], p.size[3]});
                            else
                                processed = p.clone();
                            cv::cuda::GpuMat g_proc, g_img;
                            g_proc.upload(processed);
                            cv::cuda::multiply(g_proc, cv::Scalar(-1.0), g_proc);
                            cv::cuda::exp(g_proc, g_proc);
                            cv::cuda::add(g_proc, cv::Scalar(1.0), g_proc);
                            cv::cuda::divide(1.0, g_proc, g_proc);
                            cv::cuda::normalize(g_proc, g_img, 0, 255, cv::NORM_MINMAX, CV_8U);
                            cv::cuda::resize(g_img, g_img, cv::Size(orig_w, orig_h));
                            g_preds.push_back(std::move(g_img));
                        }
                        cv::cuda::GpuMat g_out;
                        cv::cuda::cvtColor(g_preds.back(), g_out, cv::COLOR_GRAY2BGR);
                        g_out.download(output);
                    } else {
                        std::vector<cv::Mat> preds;
                        preds.reserve(outputs.size());
                        for (const cv::Mat &p : outputs) {
                            cv::Mat processed;
                            if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1)
                                processed = p.reshape(0, {p.size[2], p.size[3]});
                            else
                                processed = p.clone();
                            cv::exp(-processed, processed);
                            processed = 1.0 / (1.0 + processed);
                            cv::Mat img;
                            cv::normalize(processed, img, 0, 255, cv::NORM_MINMAX, CV_8U);
                            cv::resize(img, img, cv::Size(orig_w, orig_h));
                            preds.push_back(img);
                        }
                        cv::cvtColor(preds.back(), output, cv::COLOR_GRAY2BGR);
                    }
                } else {
                    // Single-output path: normalize spatial blob and convert to BGR.
                    const cv::Mat &raw = outputs[0];
                    if (raw.dims != 4) return;
                    const int c = raw.size[1];
                    const int h = raw.size[2];
                    const int w = raw.size[3];
                    if (use_cuda_) {
                        if (c == 1) {
                            cv::Mat img(h, w, CV_32F, const_cast<float *>(raw.ptr<float>(0, 0)));
                            cv::cuda::GpuMat g_img, g_norm, g_bgr, g_out;
                            g_img.upload(img);
                            cv::cuda::normalize(g_img, g_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
                            cv::cuda::cvtColor(g_norm, g_bgr, cv::COLOR_GRAY2BGR);
                            cv::cuda::resize(g_bgr, g_out, cv::Size(orig_w, orig_h));
                            g_out.download(output);
                        } else {
                            const int use_c = std::min(c, 3);
                            std::vector<cv::cuda::GpuMat> g_chs(use_c);
                            for (int i = 0; i < use_c; ++i) {
                                cv::Mat ch(h, w, CV_32F, const_cast<float *>(raw.ptr<float>(0, i)));
                                g_chs[i].upload(ch);
                            }
                            double g_min = std::numeric_limits<double>::max();
                            double g_max = std::numeric_limits<double>::lowest();
                            for (int i = 0; i < use_c; ++i) {
                                double ch_min, ch_max;
                                cv::cuda::minMaxLoc(g_chs[i], &ch_min, &ch_max, nullptr, nullptr);
                                g_min = std::min(g_min, ch_min);
                                g_max = std::max(g_max, ch_max);
                            }
                            const double range = std::max(g_max - g_min, 1e-9);
                            const double alpha = 255.0 / range;
                            const double beta  = -g_min * alpha;
                            std::vector<cv::cuda::GpuMat> g_chs_u8(use_c);
                            for (int i = 0; i < use_c; ++i)
                                g_chs[i].convertTo(g_chs_u8[i], CV_8U, alpha, beta);
                            cv::cuda::GpuMat g_merged, g_bgr, g_out;
                            cv::cuda::merge(g_chs_u8, g_merged);
                            cv::cuda::cvtColor(g_merged, g_bgr,
                                use_c == 3 ? cv::COLOR_RGB2BGR : cv::COLOR_GRAY2BGR);
                            cv::cuda::resize(g_bgr, g_out, cv::Size(orig_w, orig_h));
                            g_out.download(output);
                        }
                    } else {
                        cv::Mat result;
                        if (c == 1) {
                            cv::Mat img(h, w, CV_32F, const_cast<float *>(raw.ptr<float>(0, 0)));
                            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
                            img.convertTo(result, CV_8U);
                            cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
                        } else {
                            const int use_c = std::min(c, 3);
                            std::vector<cv::Mat> chs(use_c);
                            for (int i = 0; i < use_c; ++i)
                                chs[i] = cv::Mat(h, w, CV_32F, const_cast<float *>(raw.ptr<float>(0, i))).clone();
                            cv::Mat merged;
                            cv::merge(chs, merged);
                            cv::normalize(merged, merged, 0, 255, cv::NORM_MINMAX);
                            merged.convertTo(result, CV_8UC3);
                            cv::cvtColor(result, result, use_c == 3 ? cv::COLOR_RGB2BGR : cv::COLOR_GRAY2BGR);
                        }
                        cv::resize(result, output, cv::Size(orig_w, orig_h));
                    }
                }
            } catch (const cv::Exception &e) {
                inference_failed = true;
                std::cerr << "acmx2: OnnxWrapper inference failed (model disabled): " << e.what() << '\n';
            }
        }
    };

    class Dexined {
    public:
        Dexined(const string& modelPath) {
            loadModel(modelPath);
        }

        void processFrame(const Mat& image, Mat& result) {
            Mat blob = blobFromImage(image, 1.0, Size(512, 512), Scalar(103.5, 116.2, 123.6), false, false, CV_32F);
            net.setInput(blob);
            applyDexined(image, result);
        }

    private:
        Net net;

        void loadModel(const string modelPath) {
            net = readNetFromONNX(modelPath);
            net.setPreferableBackend(DNN_BACKEND_CUDA);
            net.setPreferableTarget(DNN_TARGET_CUDA);
        }

        static void sigmoid(Mat& input) {
            exp(-input, input);          // e^-input
            input = 1.0 / (1.0 + input); // 1 / (1 + e^-input)
        }

        static pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width) {
            std::vector<cv::cuda::GpuMat> g_preds;
            g_preds.reserve(output.size());
            for (const Mat &p : output) {
                cv::Mat processed;
                if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1)
                    processed = p.reshape(0, {p.size[2], p.size[3]});
                else
                    processed = p.clone();
                cv::cuda::GpuMat g_proc, g_img;
                g_proc.upload(processed);
                cv::cuda::multiply(g_proc, cv::Scalar(-1.0), g_proc);
                cv::cuda::exp(g_proc, g_proc);
                cv::cuda::add(g_proc, cv::Scalar(1.0), g_proc);
                cv::cuda::divide(1.0, g_proc, g_proc);
                cv::cuda::normalize(g_proc, g_img, 0, 255, NORM_MINMAX, CV_8U);
                cv::cuda::resize(g_img, g_img, Size(width, height));
                g_preds.push_back(std::move(g_img));
            }
            cv::cuda::GpuMat g_fuse = g_preds.back();
            cv::cuda::GpuMat g_ave(height, width, CV_32F);
            g_ave.setTo(0.0f);
            for (cv::cuda::GpuMat &g_pred : g_preds) {
                cv::cuda::GpuMat g_temp;
                g_pred.convertTo(g_temp, CV_32F);
                cv::cuda::add(g_ave, g_temp, g_ave);
            }
            cv::cuda::multiply(g_ave, cv::Scalar(1.0 / static_cast<double>(g_preds.size())), g_ave);
            cv::cuda::GpuMat g_ave_u8;
            g_ave.convertTo(g_ave_u8, CV_8U);
            Mat fuse, ave;
            g_fuse.download(fuse);
            g_ave_u8.download(ave);
            return {fuse, ave};
        }

        void applyDexined(const Mat& image, Mat& result) {
            int originalWidth = image.cols;
            int originalHeight = image.rows;
            vector<Mat> outputs;
            net.forward(outputs);
            pair<Mat, Mat> res = postProcess(outputs, originalHeight, originalWidth);
            result = res.first; 
        }
    };

    class PPHS
    {
    private:
        Net model;
        string modelPath;
        Scalar imageMean = Scalar(0.5, 0.5, 0.5);
        Scalar imageStd = Scalar(0.5, 0.5, 0.5);
        Size modelInputSize = Size(192, 192);
        Size currentSize;
        const String inputNames = "x";
        const String outputNames = "save_infer_model/scale_0.tmp_1";
        int backend_id [[maybe_unused]];
        int target_id [[maybe_unused]];
        cv::cuda::GpuMat prevMask;

    public:
        PPHS(const string& modelPath,
             int backend_id = DNN_BACKEND_CUDA,
             int target_id = DNN_TARGET_CUDA)
            : modelPath(modelPath), backend_id(backend_id), target_id(target_id)
        {
            this->model = readNet(modelPath);
            this->model.setPreferableBackend(backend_id);
            this->model.setPreferableTarget(target_id);
        }

        Mat preprocess(const Mat image)
        {
            this->currentSize = image.size();
            cv::cuda::GpuMat g_image, g_resized, g_float;
            g_image.upload(image);
            cv::cuda::resize(g_image, g_resized, this->modelInputSize);
            g_resized.convertTo(g_float, CV_32F, 1.0 / 255.0);
            Mat preprocessed;
            g_float.download(preprocessed);
            preprocessed -= imageMean;
            preprocessed /= imageStd;
            return blobFromImage(preprocessed);
        }

        Mat infer(const Mat image)
        {
            Mat inputBlob = preprocess(image);
            this->model.setInput(inputBlob, this->inputNames);
            Mat outputBlob = this->model.forward(this->outputNames);
            return postprocess(outputBlob);
        }

        Mat postprocess(Mat image)
        {
            int H = image.size[2];
            int W = image.size[3];

            Mat bg_cpu(H, W, CV_32F, image.ptr<float>(0, 0));
            Mat fg_cpu(H, W, CV_32F, image.ptr<float>(0, 1));

            cv::cuda::GpuMat g_bg, g_fg, g_bg_exp, g_fg_exp, g_sum, g_fg_prob, g_mask;
            g_bg.upload(bg_cpu);
            g_fg.upload(fg_cpu);
            cv::cuda::exp(g_bg, g_bg_exp);
            cv::cuda::exp(g_fg, g_fg_exp);
            cv::cuda::add(g_bg_exp, g_fg_exp, g_sum);
            cv::cuda::divide(g_fg_exp, g_sum, g_fg_prob);
            cv::cuda::resize(g_fg_prob, g_mask, this->currentSize, 0, 0, INTER_CUBIC);
            if (this->prevMask.empty()) {
                this->prevMask = g_mask.clone();
            }
            cv::cuda::addWeighted(g_mask, 0.6, this->prevMask, 0.4, 0.0, g_mask);
            this->prevMask = g_mask.clone();
            Mat result;
            g_mask.download(result);
            return result;
        }
    };
    Mat isolateBody(const Mat& image, const Mat& mask,
                    float blackPoint = 0.35f, float whitePoint = 0.75f);
    Mat hardenedAlphaMask(const Mat& image, const Mat& mask,
                         float blackPoint = 0.35f, float whitePoint = 0.75f);


}

#endif // ACMX2_DNN_HPP
