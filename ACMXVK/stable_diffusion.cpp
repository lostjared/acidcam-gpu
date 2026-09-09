#include "stable_diffusion.hpp"

#include <curl/curl.h>
#include <json/json.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <fstream>
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
            curl_easy_setopt(handle, CURLOPT_NOPROXY,
                             "127.0.0.1,localhost");
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

        [[nodiscard]] cv::Size neuralUpscaleWorkingSize(
            const Settings &settings, const cv::Size &fallback) {
            int width = settings.upscale_width > 0 ? settings.upscale_width
                                                   : fallback.width;
            int height = settings.upscale_height > 0 ? settings.upscale_height
                                                     : fallback.height;
            constexpr double MAX_WORKING_PIXELS = 1280.0 * 720.0;
            const double pixels = static_cast<double>(width) * height;
            if (pixels > MAX_WORKING_PIXELS) {
                const double scale = std::sqrt(MAX_WORKING_PIXELS / pixels);
                width = std::max(64, static_cast<int>(std::floor(width * scale)));
                height =
                    std::max(64, static_cast<int>(std::floor(height * scale)));
            }
            return {width, height};
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
        std::vector<std::string> argument_storage{
            executable, "--listen-ip", "127.0.0.1",
            "--listen-port", port, "--model",
            model, "--type", "f16",
            "--mmap", "--fa", "--diffusion-conv-direct",
            "--vae-conv-direct", "--lora-model-dir", ""};
        if (!settings.upscale_model.empty()) {
            argument_storage.emplace_back("--hires-upscalers-dir");
            argument_storage.push_back(
                settings.upscale_model.parent_path().string());
        }
        std::vector<char *> arguments;
        arguments.reserve(argument_storage.size() + 1U);
        for (std::string &argument : argument_storage) {
            arguments.push_back(argument.data());
        }
        arguments.push_back(nullptr);
        pid_t child = -1;
        posix_spawn_file_actions_t file_actions;
        posix_spawn_file_actions_t *actions = nullptr;
        if (settings.quiet) {
            std::string log_template =
                (std::filesystem::temp_directory_path() /
                 "acmxvk-sd-server-XXXXXX")
                    .string();
            std::vector<char> mutable_template(log_template.begin(),
                                               log_template.end());
            mutable_template.push_back('\0');
            const int log_fd = ::mkstemp(mutable_template.data());
            if (log_fd < 0) {
                throw std::runtime_error(
                    "unable to create sd-server diagnostic log: " +
                    std::string(std::strerror(errno)));
            }
            ::close(log_fd);
            diagnostic_log_path = mutable_template.data();

            const int init_result =
                posix_spawn_file_actions_init(&file_actions);
            if (init_result != 0) {
                throw std::runtime_error(
                    "unable to configure sd-server output: " +
                    std::string(std::strerror(init_result)));
            }
            actions = &file_actions;
            const int stdout_result = posix_spawn_file_actions_addopen(
                actions, STDOUT_FILENO, diagnostic_log_path.c_str(),
                O_WRONLY | O_APPEND, 0600);
            const int stderr_result = posix_spawn_file_actions_adddup2(
                actions, STDOUT_FILENO, STDERR_FILENO);
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
        if (!diagnostic_log_path.empty()) {
            std::cout << "acmxvk: sd-server diagnostic log: "
                      << diagnostic_log_path << '\n';
        }
    }

    void Server::resetDiagnosticLog() const noexcept {
        if (diagnostic_log_path.empty()) {
            return;
        }
        std::error_code error;
        std::filesystem::resize_file(diagnostic_log_path, 0U, error);
    }

    std::string Server::diagnosticLogDetails() const {
        if (diagnostic_log_path.empty()) {
            return {};
        }
        constexpr std::streamoff MAX_LOG_TAIL = 16 * 1024;
        std::ifstream input(diagnostic_log_path,
                            std::ios::binary | std::ios::ate);
        std::string details = "; sd-server diagnostic log: " +
                              diagnostic_log_path.string();
        if (!input) {
            return details;
        }
        const std::streamoff size = input.tellg();
        if (size <= 0) {
            return details;
        }
        const std::streamoff start = std::max<std::streamoff>(
            0, size - MAX_LOG_TAIL);
        input.seekg(start);
        std::string tail(static_cast<std::size_t>(size - start), '\0');
        input.read(tail.data(), static_cast<std::streamsize>(tail.size()));
        return details + "\n--- sd-server log tail ---\n" + tail;
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
        const bool require_hires_api = !settings.upscale_model.empty();
        const std::string url =
            endpoint + (require_hires_api ? "/sdcpp/v1/capabilities"
                                          : "/sdapi/v1/options");
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
                    "sd-server exited before accepting requests" +
                    diagnosticLogDetails());
            }
            long status = 0;
            try {
                const ResponseBuffer response = request(
                    url, nullptr, 3L, status, &settings.cancelled);
                if (status == 200) {
                    if (!response.value.empty()) {
                        const Json::Value document =
                            parseJson(response.value, "sd-server");
                        if (require_hires_api) {
                            const std::string upscaler_name =
                                settings.upscale_model.stem().string();
                            bool found = false;
                            for (const Json::Value &entry :
                                 document["upscalers"]) {
                                if (entry["name"].asString() ==
                                    upscaler_name) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                throw std::runtime_error(
                                    "sd-server did not discover the requested "
                                    "upscaler model: " +
                                    upscaler_name);
                            }
                        }
                    }
                    std::cout << "acmxvk: sd-server model ready; processing "
                              << settings.width << 'x' << settings.height
                              << " frames at " << settings.steps
                              << " configured steps, strength "
                              << settings.strength;
                    if (!settings.upscale_model.empty()) {
                        const cv::Size working = neuralUpscaleWorkingSize(
                            settings, {settings.width, settings.height});
                        std::cout << "; ESRGAN working resolution "
                                  << working.width << 'x' << working.height
                                  << ", final resolution "
                                  << settings.upscale_width << 'x'
                                  << settings.upscale_height;
                    }
                    std::cout << '\n';
                    resetDiagnosticLog();
                    return;
                }
                if (status >= 400) {
                    throw std::runtime_error(
                        "sd-server readiness check returned HTTP " +
                        std::to_string(status) + diagnosticLogDetails());
                }
            } catch (const std::exception &) {
                if (status != 0 ||
                    (settings.cancelled && settings.cancelled())) {
                    throw;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        throw std::runtime_error(
            "sd-server did not become ready within five minutes" +
            diagnosticLogDetails());
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

        resetDiagnosticLog();

        const std::string encoded_input = encodeBase64(png);
        Json::Value root;
        root["prompt"] = settings.prompt;
        root["negative_prompt"] = settings.negative_prompt;
        root["width"] = settings.width;
        root["height"] = settings.height;
        root["seed"] = settings.seed;
        std::string encoded_output;
        if (settings.upscale_model.empty()) {
            root["steps"] = settings.steps;
            root["cfg_scale"] = settings.cfg_scale;
            root["batch_size"] = 1;
            root["sampler_name"] = settings.sampler;
            root["scheduler"] = settings.scheduler;
            root["denoising_strength"] = settings.strength;
            root["init_images"] = Json::arrayValue;
            root["init_images"].append(encoded_input);
            const std::string body = writeJson(root);
            long status = 0;
            const ResponseBuffer response = request(
                endpoint + "/sdapi/v1/img2img", &body, 3600L, status,
                &settings.cancelled);
            const Json::Value document =
                parseJson(response.value, "sd-server");
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
            encoded_output = document["images"][0].asString();
        } else {
            root["strength"] = settings.strength;
            root["batch_count"] = 1;
            root["init_image"] = encoded_input;
            root["output_format"] = "png";
            root["output_compression"] = 100;
            Json::Value &sample = root["sample_params"];
            sample["scheduler"] = settings.scheduler;
            sample["sample_method"] = settings.sampler;
            sample["sample_steps"] = settings.steps;
            sample["guidance"]["txt_cfg"] = settings.cfg_scale;
            Json::Value &hires = root["hires"];
            const cv::Size working_size =
                neuralUpscaleWorkingSize(settings, rgba.size());
            hires["enabled"] = true;
            hires["upscaler"] = settings.upscale_model.stem().string();
            hires["scale"] = 4.0;
            hires["target_width"] = working_size.width;
            hires["target_height"] = working_size.height;
            hires["steps"] = settings.steps;
            hires["denoising_strength"] = settings.strength;
            hires["upscale_tile_size"] = 128;
            const std::string body = writeJson(root);
            long status = 0;
            const ResponseBuffer submission = request(
                endpoint + "/sdcpp/v1/img_gen", &body, 30L, status,
                &settings.cancelled);
            const Json::Value accepted =
                parseJson(submission.value, "sd-server");
            if (status != 202 || !accepted["id"].isString()) {
                throw std::runtime_error("sd-server rejected frame: " +
                                         serverError(accepted));
            }
            const std::string job_id = accepted["id"].asString();
            const std::string job_url =
                endpoint + "/sdcpp/v1/jobs/" + job_id;
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::hours(1);
            while (std::chrono::steady_clock::now() < deadline) {
                if (settings.cancelled && settings.cancelled()) {
                    const std::string cancel_body = "{}";
                    long cancel_status = 0;
                    try {
                        static_cast<void>(request(
                            job_url + "/cancel", &cancel_body, 3L,
                            cancel_status, nullptr));
                    } catch (const std::exception &) {
                    }
                    throw std::runtime_error(
                        "Stable Diffusion request cancelled");
                }
                long poll_status = 0;
                const ResponseBuffer poll = request(
                    job_url, nullptr, 10L, poll_status,
                    &settings.cancelled);
                const Json::Value job =
                    parseJson(poll.value, "sd-server job");
                if (poll_status != 200) {
                    throw std::runtime_error(
                        "sd-server job polling failed: " +
                        serverError(job));
                }
                const std::string state = job["status"].asString();
                if (state == "completed") {
                    const Json::Value &images = job["result"]["images"];
                    if (!images.isArray() || images.empty() ||
                        !images[0]["b64_json"].isString()) {
                        throw std::runtime_error(
                            "sd-server job did not contain an output image");
                    }
                    encoded_output = images[0]["b64_json"].asString();
                    break;
                }
                if (state == "failed" || state == "cancelled") {
                    throw std::runtime_error(
                        "sd-server frame job " + state + ": " +
                        serverError(job["error"]));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (encoded_output.empty()) {
                throw std::runtime_error("sd-server frame job timed out");
            }
        }
        const std::vector<std::uint8_t> decoded =
            decodeBase64(encoded_output);
        cv::Mat encoded(1, static_cast<int>(decoded.size()), CV_8UC1,
                        const_cast<std::uint8_t *>(decoded.data()));
        cv::Mat output_bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
        if (output_bgr.empty()) {
            throw std::runtime_error(
                "unable to decode image returned by sd-server");
        }
        cv::Mat output_rgba;
        cv::cvtColor(output_bgr, output_rgba, cv::COLOR_BGR2RGBA);
        if (!settings.upscale_model.empty() &&
            settings.upscale_width > 0 && settings.upscale_height > 0 &&
            output_rgba.size() !=
                cv::Size(settings.upscale_width, settings.upscale_height)) {
            cv::resize(output_rgba, output_rgba,
                       {settings.upscale_width, settings.upscale_height}, 0.0,
                       0.0, cv::INTER_LANCZOS4);
        } else if (settings.resize_to_input &&
                   output_rgba.size() != rgba.size()) {
            cv::resize(output_rgba, output_rgba, rgba.size(), 0.0, 0.0,
                       cv::INTER_LINEAR);
        }
        return output_rgba;
    }
} // namespace acmxvk::stable_diffusion
