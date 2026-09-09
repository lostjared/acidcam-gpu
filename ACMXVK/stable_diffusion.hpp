#ifndef ACMXVK_STABLE_DIFFUSION_HPP
#define ACMXVK_STABLE_DIFFUSION_HPP

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace acmxvk::stable_diffusion {
    struct Settings {
        std::filesystem::path server_executable;
        std::filesystem::path model;
        std::string prompt;
        std::string negative_prompt;
        std::string sampler;
        std::string scheduler;
        int width = 576;
        int height = 320;
        int steps = 12;
        int seed = 1234;
        int port = 1234;
        double strength = 0.35;
        double cfg_scale = 5.0;
        bool resize_to_input = true;
        bool quiet = false;
        std::function<bool()> cancelled;
    };

    class Server final {
      public:
        explicit Server(Settings settings);
        ~Server();

        Server(const Server &) = delete;
        Server &operator=(const Server &) = delete;
        Server(Server &&) = delete;
        Server &operator=(Server &&) = delete;

        [[nodiscard]] cv::Mat process(const cv::Mat &rgba) const;

      private:
        Settings settings;
        std::int64_t process_id = -1;
        std::string endpoint;

        void start();
        void stop() noexcept;
        void waitUntilReady();
    };
} // namespace acmxvk::stable_diffusion

#endif
