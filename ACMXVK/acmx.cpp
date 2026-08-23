/**
 * @file acmx.cpp
 * @brief ACMXVK real-time Vulkan video shader application.
 */

#include <mxvk/mxvk.hpp>
#include <mxvk/mxvk_cv.hpp>
#include <mxvk/mxvk_exception.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef ACMXVK_BUILD_SPRITE_VERTEX_SHADER
#define ACMXVK_BUILD_SPRITE_VERTEX_SHADER "sprite.vert.spv"
#endif

#ifndef ACMXVK_INSTALL_SPRITE_VERTEX_SHADER
#define ACMXVK_INSTALL_SPRITE_VERTEX_SHADER "sprite.vert.spv"
#endif

namespace acmxvk {
    namespace fs = std::filesystem;

    struct Options {
        int width = 1280;
        int height = 720;
        int camera_width = 1280;
        int camera_height = 720;
        int camera_device = 0;
        int shader_index = 0;
        double requested_fps = 0.0;
        bool resolution_specified = false;
        bool fullscreen = false;
        bool repeat = false;
        bool enable_vsync = false;
        bool enable_screenshot = false;
        bool show_help = false;
        std::string input_file;
        std::string graphic_file;
        std::string shader_directory;
        std::string fragment_shader;
        std::string shader_file;
    };

    [[nodiscard]] std::string optionValue(int &index, int argc, char **argv,
                                          std::string_view option) {
        if (++index >= argc) {
            throw std::runtime_error("missing value for " + std::string(option));
        }
        return argv[index];
    }

    [[nodiscard]] int parseInteger(std::string_view text, std::string_view option) {
        std::size_t parsed = 0;
        int value = 0;
        try {
            value = std::stoi(std::string(text), &parsed);
        } catch (const std::exception &) {
            throw std::runtime_error("invalid integer for " + std::string(option) + ": " +
                                     std::string(text));
        }
        if (parsed != text.size()) {
            throw std::runtime_error("invalid integer for " + std::string(option) + ": " +
                                     std::string(text));
        }
        return value;
    }

    [[nodiscard]] double parseNumber(std::string_view text, std::string_view option) {
        std::size_t parsed = 0;
        double value = 0.0;
        try {
            value = std::stod(std::string(text), &parsed);
        } catch (const std::exception &) {
            throw std::runtime_error("invalid number for " + std::string(option) + ": " +
                                     std::string(text));
        }
        if (parsed != text.size() || !std::isfinite(value)) {
            throw std::runtime_error("invalid number for " + std::string(option) + ": " +
                                     std::string(text));
        }
        return value;
    }

    void parseDimensions(std::string_view text, int &width, int &height,
                         std::string_view option) {
        const std::size_t separator = text.find_first_of("xX");
        if (separator == std::string_view::npos) {
            throw std::runtime_error("invalid dimensions for " + std::string(option) +
                                     "; expected WidthxHeight");
        }

        width = parseInteger(text.substr(0, separator), option);
        height = parseInteger(text.substr(separator + 1), option);
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("dimensions must be positive for " + std::string(option));
        }
    }

    [[nodiscard]] Options parseOptions(int argc, char **argv) {
        Options options;
        if (argc == 1) {
            options.show_help = true;
            return options;
        }

        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "-h" || option == "-v" || option == "--help" ||
                option == "--version") {
                options.show_help = true;
            } else if (option == "-i" || option == "--input") {
                options.input_file = optionValue(index, argc, argv, option);
            } else if (option == "-g" || option == "--graphic") {
                options.graphic_file = optionValue(index, argc, argv, option);
            } else if (option == "-d" || option == "--device") {
                options.camera_device =
                    parseInteger(optionValue(index, argc, argv, option), option);
            } else if (option == "-c" || option == "--camera-res") {
                parseDimensions(optionValue(index, argc, argv, option),
                                options.camera_width, options.camera_height, option);
            } else if (option == "-r" || option == "--resolution") {
                parseDimensions(optionValue(index, argc, argv, option), options.width,
                                options.height, option);
                options.resolution_specified = true;
            } else if (option == "-s" || option == "--shaders") {
                options.shader_directory = optionValue(index, argc, argv, option);
            } else if (option == "-f" || option == "--fragment") {
                options.fragment_shader = optionValue(index, argc, argv, option);
            } else if (option == "-H" || option == "--shader-index") {
                options.shader_index =
                    parseInteger(optionValue(index, argc, argv, option), option);
            } else if (option == "--shader-file") {
                options.shader_file = optionValue(index, argc, argv, option);
            } else if (option == "-u" || option == "--fps") {
                options.requested_fps =
                    parseNumber(optionValue(index, argc, argv, option), option);
                if (options.requested_fps <= 0.0) {
                    throw std::runtime_error("FPS must be positive");
                }
            } else if (option == "-n" || option == "--fullscreen") {
                options.fullscreen = true;
            } else if (option == "-a" || option == "--repeat") {
                options.repeat = true;
            } else if (option == "--enable-vsync") {
                options.enable_vsync = true;
            } else if (option == "--enable-screenshot") {
                options.enable_screenshot = true;
            } else {
                throw std::runtime_error("unknown option: " + std::string(option));
            }
        }

        if (!options.input_file.empty() && !options.graphic_file.empty()) {
            throw std::runtime_error("--input and --graphic cannot be used together");
        }
        if (!options.shader_directory.empty() && !options.fragment_shader.empty()) {
            throw std::runtime_error("--shaders and --fragment cannot be used together");
        }
        return options;
    }

    void printHelp(std::ostream &output) {
        output << "ACMXVK - Vulkan video shader engine (Increment 2)\n\n"
               << "Usage:\n"
               << "  acmxvk -i video.mp4 -s shader-directory [options]\n"
               << "  acmxvk -g image.png -f shader.spv [options]\n"
               << "  acmxvk -d 0 -s shader-directory [options]\n\n"
               << "Input:\n"
               << "  -i, --input <file>          Read a video file\n"
               << "  -g, --graphic <file>        Read a still image\n"
               << "  -d, --device <index>        Camera device (default 0)\n"
               << "  -c, --camera-res <WxH>      Requested camera dimensions\n"
               << "  -u, --fps <rate>            Requested camera FPS\n\n"
               << "Shaders:\n"
               << "  -s, --shaders <directory>   SPIR-V library containing index.txt\n"
               << "  -f, --fragment <file.spv>   Use one SPIR-V fragment shader\n"
               << "  -H, --shader-index <index>  Initial library shader index\n"
               << "      --shader-file <name>    Initial library shader filename\n\n"
               << "Window:\n"
               << "  -r, --resolution <WxH>      Window resolution\n"
               << "  -n, --fullscreen            Start fullscreen\n"
               << "  -a, --repeat                Repeat video input\n"
               << "      --enable-vsync          Use FIFO presentation\n"
               << "      --enable-screenshot     Enable MXVK F10 screenshots\n\n"
               << "Keys: Up/Down shader, Space bypass, Escape quit\n";
    }

    [[nodiscard]] std::string trim(std::string text) {
        const auto first = std::find_if_not(text.begin(), text.end(), [](unsigned char value) {
            return std::isspace(value) != 0;
        });
        const auto last = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char value) {
                              return std::isspace(value) != 0;
                          }).base();
        if (first >= last) {
            return {};
        }
        return std::string(first, last);
    }

    [[nodiscard]] cv::Mat loadRgbaImage(const std::string &filename) {
        const cv::Mat source = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (source.empty()) {
            throw std::runtime_error("unable to load image: " + filename);
        }

        cv::Mat rgba;
        switch (source.channels()) {
        case 1:
            cv::cvtColor(source, rgba, cv::COLOR_GRAY2RGBA);
            break;
        case 3:
            cv::cvtColor(source, rgba, cv::COLOR_BGR2RGBA);
            break;
        case 4:
            cv::cvtColor(source, rgba, cv::COLOR_BGRA2RGBA);
            break;
        default:
            throw std::runtime_error("unsupported image channel count: " +
                                     std::to_string(source.channels()));
        }
        return rgba;
    }

    class MainWindow final : public mxvk::VK_Window {
      public:
        explicit MainWindow(Options options)
            : mxvk::VK_Window("ACMXVK", options.width, options.height,
                              options.fullscreen, MXVK_VALIDATION, options.enable_vsync),
              options(std::move(options)) {
            setClearColor(0.0F, 0.0F, 0.0F, 1.0F);
            setEnableScreenshot(this->options.enable_screenshot);
            loadShaders();
            openInput();
            initializeSprite();
        }

        ~MainWindow() override {
            if (capture.is_open()) {
                capture.close();
            }
        }

        void event(SDL_Event &event) override {
            mxvk::VK_Window::event(event);
            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                switch (event.key.key) {
                case SDLK_UP:
                    selectShader(-1);
                    break;
                case SDLK_DOWN:
                    selectShader(1);
                    break;
                case SDLK_SPACE:
                    effects_enabled = !effects_enabled;
                    if (frame_sprite != nullptr) {
                        vkDeviceWaitIdle(getDevice());
                        frame_sprite->setFragmentShaderPath(effects_enabled ? currentShader()
                                                                            : std::string{});
                    }
                    std::cout << "acmxvk: shader effects "
                              << (effects_enabled ? "enabled" : "bypassed") << '\n';
                    break;
                default:
                    break;
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                mouse_x = event.motion.x;
                mouse_y = event.motion.y;
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = true;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = false;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
            }
        }

        void onSwapchainRecreated() override {
            initializeSprite();
        }

        void proc() override {
            if (source_kind != SourceKind::Graphic && !capture.readToSprite(*frame_sprite, false)) {
                handleCaptureEnd();
            }

            const VkExtent2D extent = getSwapchainExtent();
            const int target_width = extent.width > 0U ? static_cast<int>(extent.width) : options.width;
            const int target_height =
                extent.height > 0U ? static_cast<int>(extent.height) : options.height;

            updateShaderUniforms(target_width, target_height);
            frame_sprite->drawSpriteRect(0, 0, target_width, target_height);
        }

      private:
        enum class SourceKind { Camera,
                                Video,
                                Graphic };

        Options options;
        SourceKind source_kind = SourceKind::Camera;
        mxvk::VK_Capture capture;
        mxvk::VK_Sprite *frame_sprite = nullptr;
        cv::Mat graphic_rgba;
        std::vector<fs::path> shaders;
        std::size_t shader_index = 0;
        bool effects_enabled = true;
        float mouse_x = 0.0F;
        float mouse_y = 0.0F;
        bool mouse_pressed = false;
        std::uint64_t frame_count = 0;
        std::chrono::steady_clock::time_point shader_start{std::chrono::steady_clock::now()};
        std::chrono::steady_clock::time_point previous_frame{shader_start};

        void loadShaders() {
            if (!options.fragment_shader.empty()) {
                const fs::path fragment = fs::absolute(options.fragment_shader).lexically_normal();
                if (fragment.extension() != ".spv" || !fs::is_regular_file(fragment)) {
                    throw std::runtime_error("fragment shader is not a readable .spv file: " +
                                             fragment.string());
                }
                shaders.push_back(fragment);
                return;
            }
            if (options.shader_directory.empty()) {
                return;
            }

            const fs::path directory = fs::absolute(options.shader_directory).lexically_normal();
            std::ifstream index_file(directory / "index.txt");
            if (!index_file) {
                throw std::runtime_error("unable to open shader index: " +
                                         (directory / "index.txt").string());
            }

            std::string line;
            while (std::getline(index_file, line)) {
                const std::string entry = trim(std::move(line));
                if (entry.empty() || entry.front() == '#') {
                    continue;
                }
                const fs::path shader = (directory / entry).lexically_normal();
                if (shader.extension() == ".spv" && fs::is_regular_file(shader)) {
                    shaders.push_back(shader);
                }
            }
            if (shaders.empty()) {
                throw std::runtime_error("shader index contains no readable SPIR-V files");
            }

            if (!options.shader_file.empty()) {
                const auto selected = std::find_if(shaders.begin(), shaders.end(), [&](const fs::path &path) {
                    return path.filename() == options.shader_file;
                });
                if (selected == shaders.end()) {
                    throw std::runtime_error("shader file is not listed in index.txt: " +
                                             options.shader_file);
                }
                shader_index = static_cast<std::size_t>(std::distance(shaders.begin(), selected));
            } else {
                const int count = static_cast<int>(shaders.size());
                const int wrapped_index = ((options.shader_index % count) + count) % count;
                shader_index = static_cast<std::size_t>(wrapped_index);
            }
        }

        [[nodiscard]] std::string currentShader() const {
            return shaders.empty() ? std::string{} : shaders[shader_index].string();
        }

        [[nodiscard]] std::string spriteVertexShader() const {
            if (fs::is_regular_file(ACMXVK_INSTALL_SPRITE_VERTEX_SHADER)) {
                return ACMXVK_INSTALL_SPRITE_VERTEX_SHADER;
            }
            return ACMXVK_BUILD_SPRITE_VERTEX_SHADER;
        }

        void openInput() {
            if (!options.graphic_file.empty()) {
                source_kind = SourceKind::Graphic;
                graphic_rgba = loadRgbaImage(options.graphic_file);
                return;
            }

            source_kind = options.input_file.empty() ? SourceKind::Camera : SourceKind::Video;
            const bool opened = source_kind == SourceKind::Video
                                    ? capture.open(options.input_file)
                                    : capture.open(options.camera_device);
            if (!opened) {
                const std::string source = source_kind == SourceKind::Video
                                               ? options.input_file
                                               : std::to_string(options.camera_device);
                throw std::runtime_error("unable to open capture source: " + source);
            }

            if (source_kind == SourceKind::Camera) {
                capture.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
                if (options.requested_fps > 0.0) {
                    capture.set(cv::CAP_PROP_FPS, options.requested_fps);
                }
            }
        }

        void initializeSprite() {
            if (!ensureRenderResources()) {
                throw std::runtime_error("MXVK failed to initialize render resources");
            }

            int source_width = options.width;
            int source_height = options.height;
            if (source_kind == SourceKind::Graphic) {
                source_width = graphic_rgba.cols;
                source_height = graphic_rgba.rows;
            } else {
                source_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
                source_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
                if (source_width <= 0 || source_height <= 0) {
                    source_width = options.camera_width;
                    source_height = options.camera_height;
                }
            }

            if (frame_sprite == nullptr) {
                frame_sprite = createSprite(source_width, source_height);
            }
            frame_sprite->enableExtendedUBO();
            frame_sprite->createEmptySprite(source_width, source_height,
                                            spriteVertexShader(),
                                            effects_enabled ? currentShader() : std::string{});

            if (source_kind == SourceKind::Graphic) {
                frame_sprite->updateTexture(graphic_rgba.ptr(), graphic_rgba.cols,
                                            graphic_rgba.rows,
                                            static_cast<int>(graphic_rgba.step));
            } else if (!capture.readToSprite(*frame_sprite, false)) {
                std::cerr << "acmxvk: capture did not provide an initial frame\n";
            }

            if (!currentShader().empty()) {
                std::cout << "acmxvk: shader " << (shader_index + 1) << '/' << shaders.size()
                          << ": " << currentShader() << '\n';
            }
        }

        void selectShader(int direction) {
            if (shaders.size() < 2 || frame_sprite == nullptr) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(shaders.size());
            auto index = static_cast<std::ptrdiff_t>(shader_index) + direction;
            index = (index % count + count) % count;
            shader_index = static_cast<std::size_t>(index);

            if (effects_enabled) {
                vkDeviceWaitIdle(getDevice());
                frame_sprite->setFragmentShaderPath(currentShader());
            }
            shader_start = std::chrono::steady_clock::now();
            previous_frame = shader_start;
            frame_count = 0;
            std::cout << "acmxvk: shader " << (shader_index + 1) << '/' << shaders.size()
                      << ": " << currentShader() << '\n';
        }

        void handleCaptureEnd() {
            if (source_kind == SourceKind::Camera) {
                return;
            }
            if (!options.repeat) {
                exit();
                return;
            }

            capture.close();
            if (!capture.open(options.input_file) || !capture.readToSprite(*frame_sprite, false)) {
                throw std::runtime_error("unable to restart video input: " + options.input_file);
            }
        }

        void updateShaderUniforms(int width, int height) {
            const auto now = std::chrono::steady_clock::now();
            const float elapsed = std::chrono::duration<float>(now - shader_start).count();
            const float delta = std::chrono::duration<float>(now - previous_frame).count();
            previous_frame = now;
            ++frame_count;

            const float frame_rate = delta > 0.0F ? 1.0F / delta : 0.0F;
            frame_sprite->setShaderParams(1.0F, 1.0F, 1.0F, elapsed);
            frame_sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
            frame_sprite->setUniform0(1.0F, 1.0F, static_cast<float>(width),
                                      static_cast<float>(height));
            frame_sprite->setUniform1(delta, 0.0F, 0.0F, frame_rate);
            frame_sprite->setUniform2(static_cast<float>(frame_count), elapsed, 48000.0F,
                                      0.0F);
            frame_sprite->setUniform3(0.0F, 0.0F, 0.0F, 0.0F);
        }
    };
} // namespace acmxvk

int main(int argc, char **argv) {
    try {
        acmxvk::Options options = acmxvk::parseOptions(argc, argv);
        if (options.show_help) {
            acmxvk::printHelp(std::cout);
            return EXIT_SUCCESS;
        }

        if (!options.graphic_file.empty() && !options.resolution_specified) {
            const cv::Mat image = cv::imread(options.graphic_file, cv::IMREAD_UNCHANGED);
            if (!image.empty()) {
                options.width = image.cols;
                options.height = image.rows;
            }
        }

        acmxvk::MainWindow main_window(std::move(options));
        main_window.loop();
    } catch (const mxvk::Exception &error) {
        std::cerr << "acmxvk: MXVK exception: " << error.text() << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "acmxvk: exception: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
