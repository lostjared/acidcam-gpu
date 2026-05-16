#ifndef ACMX2_DNN_HPP
#define ACMX2_DNN_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
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
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                is_loaded = true;
            } catch (const YAML::Exception &e) {
                std::cerr << "acmx2: YAML parse error: " << e.what() << '\n';
            } catch (const cv::Exception &e) {
                std::cerr << "acmx2: Failed to load ONNX model: " << e.what() << '\n';
            }
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
                cv::Mat blob = cv::dnn::blobFromImage(image, scale, input_size, mean, swap_rb, false);
                net.setInput(blob);

                std::vector<cv::Mat> outputs;
                net.forward(outputs);
                if (outputs.empty()) return;

                const int orig_h = image.rows;
                const int orig_w = image.cols;

                if (outputs.size() > 1) {
                    // Multi-output path (e.g. Dexined): sigmoid + normalize each head, use fused last.
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
                    cv::Mat result = preds.back();
                    cv::cvtColor(result, output, cv::COLOR_GRAY2BGR);
                } else {
                    // Single-output path: normalize spatial blob and convert to BGR.
                    const cv::Mat &raw = outputs[0];
                    if (raw.dims != 4) return;
                    const int c = raw.size[1];
                    const int h = raw.size[2];
                    const int w = raw.size[3];
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
            vector<Mat> preds;
            preds.reserve(output.size());
            for (const Mat &p : output) {
                Mat img;
                Mat processed;
                if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1) {
                    processed = p.reshape(0, {p.size[2], p.size[3]});
                } else {
                    processed = p.clone();
                }
                sigmoid(processed);
                normalize(processed, img, 0, 255, NORM_MINMAX, CV_8U);
                resize(img, img, Size(width, height));
                preds.push_back(img);
            }
            Mat fuse = preds.back();
            Mat ave = Mat::zeros(height, width, CV_32F);
            for (Mat &pred : preds) {
                Mat temp;
                pred.convertTo(temp, CV_32F);
                ave += temp;
            }
            ave /= static_cast<float>(preds.size());
            ave.convertTo(ave, CV_8U);
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
        Mat prevMask;

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
            Mat preprocessed = Mat::zeros(this->modelInputSize, image.type());
            resize(image, preprocessed, this->modelInputSize);

            preprocessed.convertTo(preprocessed, CV_32F, 1.0 / 255.0);
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

            Mat bg(H, W, CV_32F, image.ptr<float>(0, 0));
            Mat fg(H, W, CV_32F, image.ptr<float>(0, 1));

            Mat bg_exp, fg_exp, sum_exp, fg_prob;
            exp(bg, bg_exp);
            exp(fg, fg_exp);
            add(bg_exp, fg_exp, sum_exp);
            divide(fg_exp, sum_exp, fg_prob);
            Mat fullSizeMask;
            resize(fg_prob, fullSizeMask, this->currentSize, 0, 0, INTER_CUBIC);
            if (this->prevMask.empty()) {
                this->prevMask = fullSizeMask.clone();
            }
            addWeighted(fullSizeMask, 0.6, this->prevMask, 0.4, 0.0, fullSizeMask);
            this->prevMask = fullSizeMask.clone();
            return fullSizeMask;
        }
    };
    Mat isolateBody(const Mat& image, const Mat& mask,
                    float blackPoint = 0.35f, float whitePoint = 0.75f);
    Mat hardenedAlphaMask(const Mat& image, const Mat& mask,
                         float blackPoint = 0.35f, float whitePoint = 0.75f);


}

#endif // ACMX2_DNN_HPP
