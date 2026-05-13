#ifndef ACMX2_DNN_HPP
#define ACMX2_DNN_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <map>
#include <vector>
#include <string>
#include <iostream>

namespace ac_dnn {
    using namespace std;
    using namespace cv;
    using namespace dnn;

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
