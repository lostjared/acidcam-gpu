#ifndef ACMXVK_APP_MEDIA_HELPERS_HPP
#define ACMXVK_APP_MEDIA_HELPERS_HPP

#include "options.hpp"

#include <opencv2/core.hpp>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mxvk {
    class VK_Capture;
}

namespace acmxvk {
    struct PlaylistNode {
        std::string name;
        std::vector<fs::path> shaders;
    };

    class LatestCameraFrame {
      public:
        LatestCameraFrame() = default;
        ~LatestCameraFrame();
        LatestCameraFrame(const LatestCameraFrame &) = delete;
        LatestCameraFrame &operator=(const LatestCameraFrame &) = delete;
        LatestCameraFrame(LatestCameraFrame &&) = delete;
        LatestCameraFrame &operator=(LatestCameraFrame &&) = delete;

        void start(mxvk::VK_Capture &source);
        void stop() noexcept;
        [[nodiscard]] bool takeLatest(cv::Mat &frame, bool wait_for_first);

      private:
        void captureLoop() noexcept;

        mxvk::VK_Capture *capture_source = nullptr;
        cv::Mat latest_frame;
        std::thread capture_thread;
        std::mutex frame_mutex;
        std::condition_variable frame_condition;
        std::uint64_t published_generation = 0;
        std::uint64_t consumed_generation = 0;
        bool stopping = true;
    };
} // namespace acmxvk

#endif
