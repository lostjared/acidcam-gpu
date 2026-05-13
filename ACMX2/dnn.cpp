#include"dnn.hpp"

namespace ac_dnn {
    // Internal: produces a CV_32F alpha mask in [0,1] from the raw PPHS
    // soft probability mask, shared by isolateBody() and hardenedAlphaMask().
    static Mat buildHardenedFloatAlpha(const Mat& image, const Mat& mask)
    {
        // ----- 1. Normalise mask to CV_32F single channel ---------------
        Mat soft;
        if (mask.type() == CV_32FC1) {
            soft = mask;
        } else if (mask.channels() == 1) {
            mask.convertTo(soft, CV_32F,
                           mask.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
        } else {
            Mat gray;
            cvtColor(mask, gray, COLOR_BGR2GRAY);
            gray.convertTo(soft, CV_32F,
                           gray.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
        }
        if (soft.size() != image.size())
            resize(soft, soft, image.size(), 0, 0, INTER_LINEAR);

        threshold(soft, soft, 1.0, 1.0, THRESH_TRUNC);
        threshold(soft, soft, 0.0, 0.0, THRESH_TOZERO);

        // ----- 2. Binary silhouette + morphological cleanup --------------
        Mat binary;
        threshold(soft, binary, 0.5f, 1.0f, THRESH_BINARY);
        binary.convertTo(binary, CV_8U, 255.0);

        const Mat kOpen  = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        const Mat kClose = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
        morphologyEx(binary, binary, MORPH_OPEN,  kOpen);
        morphologyEx(binary, binary, MORPH_CLOSE, kClose);

        // ----- 3. Largest connected component only -----------------------
        Mat labels, stats, centroids;
        const int nLabels = connectedComponentsWithStats(binary, labels, stats,
                                                         centroids, 8, CV_32S);
        if (nLabels > 1) {
            int bestLabel = -1;
            int bestArea = 0;
            for (int i = 1; i < nLabels; ++i) {
                const int area = stats.at<int>(i, CC_STAT_AREA);
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

        // ----- 4. Slight erosion to kill the halo ------------------------
        const Mat kErode = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        erode(binary, binary, kErode);

        Mat silhouette;
        binary.convertTo(silhouette, CV_32F, 1.0 / 255.0);

        // ----- 5. Combine silhouette with soft probability + feather -----
        Mat gated;
        multiply(soft, silhouette, gated);

        Mat feathered;
        GaussianBlur(gated, feathered, Size(0, 0), 1.2);

        // ----- 6. Black/white point + gamma remap ------------------------
        constexpr float blackPoint = 0.30f;
        constexpr float whitePoint = 0.75f;
        Mat hardenedMask = (feathered - blackPoint) / (whitePoint - blackPoint);
        threshold(hardenedMask, hardenedMask, 1.0, 1.0, THRESH_TRUNC);
        threshold(hardenedMask, hardenedMask, 0.0, 0.0, THRESH_TOZERO);
        pow(hardenedMask, 1.6, hardenedMask);
        return hardenedMask;
    }

    Mat hardenedAlphaMask(const Mat& image, const Mat& mask)
    {
        if (image.empty() || mask.empty())
            return Mat();
        Mat alphaFloat = buildHardenedFloatAlpha(image, mask);
        Mat alpha8;
        alphaFloat.convertTo(alpha8, CV_8U, 255.0);
        return alpha8;
    }

    Mat isolateBody(const Mat& image, const Mat& mask)
    {
        if (image.empty() || mask.empty())
            return image.clone();

        Mat hardenedMask = buildHardenedFloatAlpha(image, mask);

        // ----- 7. Alpha composite over a black background ----------------
        Mat alpha;
        cvtColor(hardenedMask, alpha, COLOR_GRAY2BGR);

        Mat foreground;
        image.convertTo(foreground, CV_32FC3, 1.0 / 255.0);

        Mat finalFloat;
        multiply(foreground, alpha, finalFloat);

        Mat output_image;
        finalFloat.convertTo(output_image, CV_8UC3, 255.0);
        return output_image;
    }
}
