    // Playlist data and asynchronous camera capture state.
    struct PlaylistNode {
        std::string name;
        std::vector<fs::path> shaders;
    };

    class LatestCameraFrame {
      public:
        LatestCameraFrame() = default;
        ~LatestCameraFrame() { stop(); }
        LatestCameraFrame(const LatestCameraFrame &) = delete;
        LatestCameraFrame &operator=(const LatestCameraFrame &) = delete;
        LatestCameraFrame(LatestCameraFrame &&) = delete;
        LatestCameraFrame &operator=(LatestCameraFrame &&) = delete;

        void start(mxvk::VK_Capture &source) {
            stop();
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                capture_source = &source;
                stopping = false;
                latest_frame.release();
                published_generation = 0;
                consumed_generation = 0;
            }
            capture_thread = std::thread(&LatestCameraFrame::captureLoop, this);
        }

        void stop() noexcept {
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                stopping = true;
            }
            frame_condition.notify_all();
            if (capture_thread.joinable()) {
                capture_thread.join();
            }
            std::lock_guard<std::mutex> lock(frame_mutex);
            capture_source = nullptr;
            latest_frame.release();
            published_generation = 0;
            consumed_generation = 0;
        }

        [[nodiscard]] bool takeLatest(cv::Mat &frame, bool wait_for_first) {
            std::unique_lock<std::mutex> lock(frame_mutex);
            if (wait_for_first && published_generation == 0 && !stopping) {
                frame_condition.wait_for(lock, std::chrono::seconds(3), [&] {
                    return stopping || published_generation > 0;
                });
            }
            if (stopping || published_generation == consumed_generation ||
                latest_frame.empty()) {
                return false;
            }
            frame = latest_frame;
            consumed_generation = published_generation;
            return true;
        }

      private:
        void captureLoop() noexcept {
            while (true) {
                mxvk::VK_Capture *source = nullptr;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (stopping) {
                        return;
                    }
                    source = capture_source;
                }
                if (source == nullptr) {
                    return;
                }

                cv::Mat captured;
                bool read_frame = false;
                try {
                    read_frame = source->read(captured);
                } catch (const std::exception &error) {
                    std::cerr << "acmxvk: asynchronous camera read failed: "
                              << error.what() << '\n';
                } catch (...) {
                    std::cerr << "acmxvk: asynchronous camera read failed\n";
                }
                if (!read_frame || captured.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (stopping) {
                        return;
                    }
                    latest_frame = std::move(captured);
                    ++published_generation;
                }
                frame_condition.notify_one();
            }
        }

        mxvk::VK_Capture *capture_source = nullptr;
        cv::Mat latest_frame;
        std::thread capture_thread;
        std::mutex frame_mutex;
        std::condition_variable frame_condition;
        std::uint64_t published_generation = 0;
        std::uint64_t consumed_generation = 0;
        bool stopping = true;
    };
