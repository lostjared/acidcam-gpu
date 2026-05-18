#include"dnn.hpp"

namespace ac_dnn {
    
    
    static cv::Mat buildHardenedFloatAlpha(const cv::Mat& image, const cv::Mat& mask, float blackPoint, float whitePoint)
    {
        cv::Mat soft;
        if (mask.type() == CV_32FC1) {
            soft = mask;
        } else if (mask.channels() == 1) {
            mask.convertTo(soft, CV_32F,
                           mask.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
        } else {
            cv::Mat gray;
            cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(soft, CV_32F,
                           gray.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
        }
        if (soft.size() != image.size())
            cv::resize(soft, soft, image.size(), 0, 0, cv::INTER_LINEAR);
        cv::threshold(soft, soft, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(soft, soft, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::Mat binary;
        cv::threshold(soft, binary, 0.5f, 1.0f, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8U, 255.0);
        const cv::Mat kOpen  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        const cv::Mat kClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN,  kOpen);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kClose);
        cv::Mat labels, stats, centroids;
        const int nLabels = cv::connectedComponentsWithStats(binary, labels, stats,
                                                         centroids, 8, CV_32S);
        if (nLabels > 1) {
            int bestLabel = -1;
            int bestArea = 0;
            for (int i = 1; i < nLabels; ++i) {
                const int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > bestArea) {
                    bestArea = area;
                    bestLabel = i;
                }
            }
            const int minArea = (image.cols * image.rows) / 200;
            if (bestLabel > 0 && bestArea >= minArea) {
                binary = (labels == bestLabel);
            }
        }
        const cv::Mat kErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(binary, binary, kErode);
        cv::Mat silhouette;
        binary.convertTo(silhouette, CV_32F, 1.0 / 255.0);
        cv::Mat gated;
        cv::multiply(soft, silhouette, gated);
        cv::Mat feathered;
        cv::GaussianBlur(gated, feathered, cv::Size(0, 0), 1.2);
        cv::Mat hardenedMask = (feathered - blackPoint) / (whitePoint - blackPoint);
        cv::threshold(hardenedMask, hardenedMask, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(hardenedMask, hardenedMask, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::pow(hardenedMask, 1.6, hardenedMask);
        return hardenedMask;
    }

    cv::Mat hardenedAlphaMask(const cv::Mat& image, const cv::Mat& mask, float blackPoint, float whitePoint)
    {
        if (image.empty() || mask.empty())
            return cv::Mat();
        cv::Mat alphaFloat = buildHardenedFloatAlpha(image, mask, blackPoint, whitePoint);
        cv::Mat alpha8;
        alphaFloat.convertTo(alpha8, CV_8U, 255.0);
        return alpha8;
    }

    

    cv::Mat isolateBody(const cv::Mat& image, const cv::Mat& mask, float blackPoint, float whitePoint)
    {
        if (image.empty() || mask.empty())
            return image.clone();
        cv::Mat hardenedMask = buildHardenedFloatAlpha(image, mask, blackPoint, whitePoint);
        cv::Mat alpha;
        cv::cvtColor(hardenedMask, alpha, cv::COLOR_GRAY2BGR);
        cv::Mat foreground;
        image.convertTo(foreground, CV_32FC3, 1.0 / 255.0);
        cv::Mat finalFloat;
        cv::multiply(foreground, alpha, finalFloat);
        cv::Mat output_image;
        finalFloat.convertTo(output_image, CV_8UC3, 255.0);
        return output_image;
    }
}
