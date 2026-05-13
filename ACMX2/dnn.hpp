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
