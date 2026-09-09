#include "stable_diffusion.hpp"

#include <curl/curl.h>
#include <json/json.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <spawn.h>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace acmxvk::stable_diffusion {
    namespace {
        constexpr std::size_t MAX_RESPONSE_SIZE = 64U * 1024U * 1024U;

        struct ResponseBuffer {
            std::string value;
            bool overflow = false;
        };

        [[nodiscard]] std::string curlError(CURLcode code) {
            return curl_easy_strerror(code);
        }

        std::size_t appendResponse(char *data, std::size_t size,
                                   std::size_t count, void *context) {
            auto *buffer = static_cast<ResponseBuffer *>(context);
            if (size != 0U && count >
                                  std::numeric_limits<std::size_t>::max() /
                                      size) {
                buffer->overflow = true;
                return 0U;
            }
            const std::size_t bytes = size * count;
            if (bytes > MAX_RESPONSE_SIZE -
                            std::min(buffer->value.size(), MAX_RESPONSE_SIZE)) {
                buffer->overflow = true;
                return 0U;
            }
            buffer->value.append(data, bytes);
            return bytes;
        }

        int checkCancelled(void *context, curl_off_t, curl_off_t, curl_off_t,
                           curl_off_t) {
            const auto *cancelled =
                static_cast<const std::function<bool()> *>(context);
            return cancelled != nullptr && *cancelled && (*cancelled)() ? 1
                                                                        : 0;
        }

        [[nodiscard]] std::string encodeBase64(
            const std::vector<std::uint8_t> &bytes) {
            static constexpr std::string_view ALPHABET =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string output;
            output.reserve(((bytes.size() + 2U) / 3U) * 4U);
            for (std::size_t index = 0; index < bytes.size(); index += 3U) {
                const std::uint32_t first = bytes[index];
                const std::uint32_t second =
                    index + 1U < bytes.size() ? bytes[index + 1U] : 0U;
                const std::uint32_t third =
                    index + 2U < bytes.size() ? bytes[index + 2U] : 0U;
                const std::uint32_t value =
                    (first << 16U) | (second << 8U) | third;
                output.push_back(ALPHABET[(value >> 18U) & 0x3fU]);
                output.push_back(ALPHABET[(value >> 12U) & 0x3fU]);
                output.push_back(index + 1U < bytes.size()
                                     ? ALPHABET[(value >> 6U) & 0x3fU]
                                     : '=');
                output.push_back(index + 2U < bytes.size()
                                     ? ALPHABET[value & 0x3fU]
                                     : '=');
            }
            return output;
        }

        [[nodiscard]] int decodeBase64Character(unsigned char character) {
            if (character >= 'A' && character <= 'Z') {
                return character - 'A';
            }
            if (character >= 'a' && character <= 'z') {
                return character - 'a' + 26;
            }
            if (character >= '0' && character <= '9') {
                return character - '0' + 52;
            }
            if (character == '+') {
                return 62;
            }
            if (character == '/') {
                return 63;
            }
            return -1;
        }

        [[nodiscard]] std::vector<std::uint8_t> decodeBase64(
            std::string_view encoded) {
            const std::size_t comma = encoded.find(',');
            if (encoded.starts_with("data:") && comma != std::string_view::npos) {
                encoded.remove_prefix(comma + 1U);
            }
            std::vector<std::uint8_t> output;
            output.reserve((encoded.size() / 4U) * 3U);
            std::uint32_t accumulator = 0U;
            int bits = 0;
            for (const unsigned char character : encoded) {
                if (character == '=') {
                    break;
                }
                const int value = decodeBase64Character(character);
                if (value < 0) {
                    if (character == ' ' || character == '\n' ||
                        character == '\r' || character == '\t') {
                        continue;
                    }
                    throw std::runtime_error(
                        "sd-server returned invalid base64 image data");
                }
                accumulator = (accumulator << 6U) |
                              static_cast<std::uint32_t>(value);
                bits += 6;
                if (bits >= 8) {
                    bits -= 8;
                    output.push_back(static_cast<std::uint8_t>(
                        (accumulator >> static_cast<unsigned int>(bits)) &
                        0xffU));
                }
            }
            return output;
        }

        [[nodiscard]] Json::Value parseJson(std::string_view text,
                                            std::string_view context) {
            Json::CharReaderBuilder builder;
            builder["collectComments"] = false;
            std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
            Json::Value value;
            std::string error;
            if (!reader->parse(text.data(), text.data() + text.size(), &value,
                               &error)) {
                throw std::runtime_error(std::string(context) +
                                         " returned invalid JSON: " + error);
            }
            return value;
        }

        [[nodiscard]] std::string writeJson(const Json::Value &value) {
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "";
            return Json::writeString(builder, value);
        }

        [[nodiscard]] ResponseBuffer request(
            std::string_view url, const std::string *body,
            long timeout_seconds, long &status,
            const std::function<bool()> *cancelled) {
            CURL *handle = curl_easy_init();
            if (handle == nullptr) {
                throw std::runtime_error("unable to initialize libcurl");
            }
            ResponseBuffer response;
            curl_slist *headers = nullptr;
            if (body != nullptr) {
                headers = curl_slist_append(headers,
                                            "Content-Type: application/json");
            }
            const std::string request_url(url);
            curl_easy_setopt(handle, CURLOPT_URL, request_url.c_str());
            curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, appendResponse);
            curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 2L);
            curl_easy_setopt(handle, CURLOPT_TIMEOUT, timeout_seconds);
            curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, checkCancelled);
            curl_easy_setopt(handle, CURLOPT_XFERINFODATA, cancelled);
            if (body != nullptr) {
                curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
                curl_easy_setopt(handle, CURLOPT_POST, 1L);
                curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body->data());
                curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE_LARGE,
                                 static_cast<curl_off_t>(body->size()));
            }
            const CURLcode result = curl_easy_perform(handle);
            curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
            curl_slist_free_all(headers);
            curl_easy_cleanup(handle);
            if (result != CURLE_OK) {
                if (result == CURLE_ABORTED_BY_CALLBACK) {
                    throw std::runtime_error(
                        "Stable Diffusion request cancelled");
                }
                if (response.overflow) {
                    throw std::runtime_error(
                        "sd-server response exceeded 64 MiB");
                }
                throw std::runtime_error("sd-server request failed: " +
                                         curlError(result));
            }
            return response;
        }

        [[nodiscard]] std::string serverError(const Json::Value &root) {
            if (root.isMember("message") && root["message"].isString()) {
                return root["message"].asString();
            }
            if (root.isMember("error") && root["error"].isString()) {
                return root["error"].asString();
            }
            return "unknown server error";
        }
    } // namespace

    Server::Server(Settings settings) : settings(std::move(settings)) {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("unable to initialize libcurl runtime");
        }
        endpoint = "http://127.0.0.1:" + std::to_string(this->settings.port);
        try {
            start();
            waitUntilReady();
        } catch (...) {
            stop();
            curl_global_cleanup();
            throw;
        }
    }

    Server::~Server() {
        stop();
        curl_global_cleanup();
    }

    void Server::start() {
        const std::string port = std::to_string(settings.port);
        const std::string executable = settings.server_executable.string();
        const std::string model = settings.model.string();
        std::array<char *, 16> arguments{
            const_cast<char *>(executable.c_str()),
            const_cast<char *>("--listen-ip"),
            const_cast<char *>("127.0.0.1"),
            const_cast<char *>("--listen-port"),
            const_cast<char *>(port.c_str()),
            const_cast<char *>("--model"),
            const_cast<char *>(model.c_str()),
            const_cast<char *>("--type"),
            const_cast<char *>("f16"),
            const_cast<char *>("--mmap"),
            const_cast<char *>("--fa"),
            const_cast<char *>("--diffusion-conv-direct"),
            const_cast<char *>("--vae-conv-direct"), nullptr};
        pid_t child = -1;
        posix_spawn_file_actions_t file_actions;
        posix_spawn_file_actions_t *actions = nullptr;
        if (settings.quiet) {
            const int init_result =
                posix_spawn_file_actions_init(&file_actions);
            if (init_result != 0) {
                throw std::runtime_error(
                    "unable to configure sd-server output: " +
                    std::string(std::strerror(init_result)));
            }
            actions = &file_actions;
            const int stdout_result = posix_spawn_file_actions_addopen(
                actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
            const int stderr_result = posix_spawn_file_actions_addopen(
                actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
            if (stdout_result != 0 || stderr_result != 0) {
                posix_spawn_file_actions_destroy(actions);
                const int error =
                    stdout_result != 0 ? stdout_result : stderr_result;
                throw std::runtime_error(
                    "unable to redirect sd-server output: " +
                    std::string(std::strerror(error)));
            }
        }
        const int result = posix_spawnp(&child, executable.c_str(), actions,
                                        nullptr, arguments.data(), environ);
        if (actions != nullptr) {
            posix_spawn_file_actions_destroy(actions);
        }
        if (result != 0) {
            throw std::runtime_error("unable to launch sd-server: " +
                                     std::string(std::strerror(result)));
        }
        process_id = child;
        std::cout << "acmxvk: launched sd-server process " << process_id
                  << " on 127.0.0.1:" << settings.port << '\n';
    }

    void Server::stop() noexcept {
        if (process_id <= 0) {
            return;
        }
        const pid_t child = static_cast<pid_t>(process_id);
        if (kill(child, SIGTERM) != 0 && errno != ESRCH) {
            std::cerr << "acmxvk: unable to stop sd-server process "
                      << process_id << ": " << std::strerror(errno) << '\n';
        }
        for (int attempt = 0; attempt < 50; ++attempt) {
            int status = 0;
            const pid_t result = waitpid(child, &status, WNOHANG);
            if (result == child || (result < 0 && errno == ECHILD)) {
                process_id = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (kill(child, SIGKILL) == 0 || errno == ESRCH) {
            int status = 0;
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
            }
        }
        process_id = -1;
    }

    void Server::waitUntilReady() {
        const std::string url = endpoint + "/sdcpp/v1/capabilities";
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::minutes(5);
        while (std::chrono::steady_clock::now() < deadline) {
            if (settings.cancelled && settings.cancelled()) {
                throw std::runtime_error(
                    "Stable Diffusion startup cancelled");
            }
            int child_status = 0;
            const pid_t result = waitpid(static_cast<pid_t>(process_id),
                                         &child_status, WNOHANG);
            if (result == static_cast<pid_t>(process_id)) {
                process_id = -1;
                throw std::runtime_error(
                    "sd-server exited before accepting requests");
            }
            try {
                long status = 0;
                const ResponseBuffer response = request(
                    url, nullptr, 3L, status, &settings.cancelled);
                if (status == 200 && !response.value.empty()) {
                    static_cast<void>(parseJson(response.value, "sd-server"));
                    std::cout << "acmxvk: sd-server model ready; processing "
                              << settings.width << 'x' << settings.height
                              << " frames at " << settings.steps
                              << " configured steps, strength "
                              << settings.strength << '\n';
                    return;
                }
            } catch (const std::exception &) {
                if (settings.cancelled && settings.cancelled()) {
                    throw;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        throw std::runtime_error(
            "sd-server did not become ready within five minutes");
    }

    cv::Mat Server::process(const cv::Mat &rgba) const {
        if (rgba.empty() || rgba.type() != CV_8UC4) {
            throw std::runtime_error(
                "Stable Diffusion input must be a non-empty RGBA8 frame");
        }
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        std::vector<std::uint8_t> png;
        if (!cv::imencode(".png", bgr, png)) {
            throw std::runtime_error(
                "unable to encode Stable Diffusion input frame");
        }

        Json::Value root;
        root["prompt"] = settings.prompt;
        root["negative_prompt"] = settings.negative_prompt;
        root["width"] = settings.width;
        root["height"] = settings.height;
        root["steps"] = settings.steps;
        root["cfg_scale"] = settings.cfg_scale;
        root["seed"] = settings.seed;
        root["batch_size"] = 1;
        root["sampler_name"] = settings.sampler;
        root["scheduler"] = settings.scheduler;
        root["denoising_strength"] = settings.strength;
        root["init_images"] = Json::arrayValue;
        root["init_images"].append(encodeBase64(png));
        const std::string body = writeJson(root);

        long status = 0;
        const ResponseBuffer response = request(
            endpoint + "/sdapi/v1/img2img", &body, 3600L, status,
            &settings.cancelled);
        const Json::Value document = parseJson(response.value, "sd-server");
        if (status != 200) {
            throw std::runtime_error("sd-server rejected frame: " +
                                     serverError(document));
        }
        if (!document["images"].isArray() ||
            document["images"].empty() ||
            !document["images"][0].isString()) {
            throw std::runtime_error(
                "sd-server response did not contain an output image");
        }
        const std::vector<std::uint8_t> decoded =
            decodeBase64(document["images"][0].asString());
        cv::Mat encoded(1, static_cast<int>(decoded.size()), CV_8UC1,
                        const_cast<std::uint8_t *>(decoded.data()));
        cv::Mat output_bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
        if (output_bgr.empty()) {
            throw std::runtime_error(
                "unable to decode image returned by sd-server");
        }
        cv::Mat output_rgba;
        cv::cvtColor(output_bgr, output_rgba, cv::COLOR_BGR2RGBA);
        if (settings.resize_to_input && output_rgba.size() != rgba.size()) {
            cv::resize(output_rgba, output_rgba, rgba.size(), 0.0, 0.0,
                       cv::INTER_LINEAR);
        }
        return output_rgba;
    }
} // namespace acmxvk::stable_diffusion
