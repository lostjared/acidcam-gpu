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
        bool enable_playlist = false;
        bool show_help = false;
        std::vector<int> shader_pass_indices;
        std::vector<std::string> shader_pass_files;
        std::string input_file;
        std::string graphic_file;
        std::string shader_directory;
        std::string fragment_shader;
        std::string shader_file;
        std::string playlist_file;
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
            } else if (option == "--shader-pass") {
                const std::string values = optionValue(index, argc, argv, option);
                std::size_t start = 0;
                while (start <= values.size()) {
                    const std::size_t separator = values.find(',', start);
                    const std::string_view value(
                        values.data() + start,
                        (separator == std::string::npos ? values.size() : separator) - start);
                    if (!value.empty()) {
                        options.shader_pass_indices.push_back(parseInteger(value, option));
                    }
                    if (separator == std::string::npos) {
                        break;
                    }
                    start = separator + 1;
                }
            } else if (option == "--shader-pass-files") {
                const std::string payload = optionValue(index, argc, argv, option);
                std::size_t start = 0;
                while (start < payload.size()) {
                    const std::size_t separator = payload.find(':', start);
                    if (separator == std::string::npos) {
                        throw std::runtime_error("invalid --shader-pass-files payload");
                    }
                    const int length = parseInteger(
                        std::string_view(payload).substr(start, separator - start), option);
                    const std::size_t name_start = separator + 1;
                    if (length < 0 || static_cast<std::size_t>(length) >
                                          payload.size() - name_start) {
                        throw std::runtime_error("invalid --shader-pass-files payload");
                    }
                    options.shader_pass_files.push_back(
                        payload.substr(name_start, static_cast<std::size_t>(length)));
                    start = name_start + static_cast<std::size_t>(length);
                }
            } else if (option == "--playlist") {
                options.playlist_file = optionValue(index, argc, argv, option);
            } else if (option == "--enable-playlist") {
                options.enable_playlist = true;
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
        if ((!options.shader_pass_indices.empty() || !options.shader_pass_files.empty() ||
             !options.playlist_file.empty() || options.enable_playlist) &&
            options.shader_directory.empty()) {
            throw std::runtime_error(
                "shader passes and playlists require --shaders <directory>");
        }
        if (options.enable_playlist && options.playlist_file.empty()) {
            throw std::runtime_error("--enable-playlist requires --playlist <file>");
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
               << "  --shader-pass <indices>     Comma-separated pre-shader pass chain\n"
               << "  --shader-pass-files <data>  ACMX2 length-prefixed shader filenames\n"
               << "  --playlist <file>           Shader or named multipass playlist\n\n"
               << "  --enable-playlist           Enable the playlist immediately\n\n"
               << "Window:\n"
               << "  -r, --resolution <WxH>      Window resolution\n"
               << "  -n, --fullscreen            Start fullscreen\n"
               << "  -a, --repeat                Repeat video input\n"
               << "      --enable-vsync          Use FIFO presentation\n"
               << "      --enable-screenshot     Enable MXVK F10 screenshots\n\n"
               << "Keys: Up/Down shader or playlist node, Shift+Up/Down final shader,\n"
               << "      P playlist, M multipass, Space bypass, Escape quit\n";
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

    struct PlaylistNode {
        std::string name;
        std::vector<fs::path> shaders;
    };

    class MainWindow final : public mxvk::VK_Window {
      public:
        explicit MainWindow(Options options)
            : mxvk::VK_Window("ACMXVK", options.width, options.height,
                              options.fullscreen, MXVK_VALIDATION, options.enable_vsync),
              options(std::move(options)) {
            setClearColor(0.0F, 0.0F, 0.0F, 1.0F);
            setEnableScreenshot(this->options.enable_screenshot);
            loadShaders();
            loadShaderPasses();
            loadPlaylist();
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
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0 || !playlist_enabled) {
                        selectShader(-1);
                    } else {
                        selectPlaylistNode(-1);
                    }
                    break;
                case SDLK_DOWN:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0 || !playlist_enabled) {
                        selectShader(1);
                    } else {
                        selectPlaylistNode(1);
                    }
                    break;
                case SDLK_SPACE:
                    effects_enabled = !effects_enabled;
                    applyShaderPipeline();
                    std::cout << "acmxvk: shader effects "
                              << (effects_enabled ? "enabled" : "bypassed") << '\n';
                    break;
                case SDLK_P:
                    if (!playlist.empty()) {
                        playlist_enabled = !playlist_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: playlist "
                                  << (playlist_enabled ? "enabled" : "disabled") << '\n';
                    }
                    break;
                case SDLK_M:
                    if (!configured_passes.empty()) {
                        multipass_enabled = !multipass_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: multipass "
                                  << (multipass_enabled ? "enabled" : "disabled") << '\n';
                    }
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
        std::vector<fs::path> configured_passes;
        std::vector<PlaylistNode> playlist;
        std::vector<mxvk::VK_Sprite *> post_process_sprites;
        std::size_t shader_index = 0;
        std::size_t playlist_index = 0;
        bool effects_enabled = true;
        bool multipass_enabled = false;
        bool playlist_enabled = false;
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

        [[nodiscard]] fs::path findShader(std::string name) const {
            name = trim(std::move(name));
            if (name.empty()) {
                return {};
            }

            fs::path requested(name);
            if (requested.extension() != ".spv") {
                requested.replace_extension(".spv");
            }
            const auto match = std::find_if(shaders.begin(), shaders.end(),
                                            [&](const fs::path &shader) {
                                                return shader.filename() == requested.filename();
                                            });
            return match == shaders.end() ? fs::path{} : *match;
        }

        void loadShaderPasses() {
            for (const int index : options.shader_pass_indices) {
                if (index < 0 || index >= static_cast<int>(shaders.size())) {
                    throw std::runtime_error("shader pass index is out of range: " +
                                             std::to_string(index));
                }
                configured_passes.push_back(shaders[static_cast<std::size_t>(index)]);
            }
            for (const std::string &name : options.shader_pass_files) {
                const fs::path shader = findShader(name);
                if (shader.empty()) {
                    throw std::runtime_error("shader pass file is not listed in index.txt: " +
                                             name);
                }
                configured_passes.push_back(shader);
            }
            multipass_enabled = !configured_passes.empty();
        }

        void loadPlaylist() {
            if (options.playlist_file.empty()) {
                return;
            }

            std::ifstream input(options.playlist_file);
            if (!input) {
                throw std::runtime_error("unable to open playlist: " + options.playlist_file);
            }

            PlaylistNode *current_node = nullptr;
            std::vector<fs::path> default_entries;
            std::string line;
            while (std::getline(input, line)) {
                line = trim(std::move(line));
                if (line.empty() || line.front() == '#') {
                    continue;
                }
                if (line.size() >= 2 && line.front() == '[' && line.back() == ']') {
                    playlist.push_back({line.substr(1, line.size() - 2), {}});
                    current_node = &playlist.back();
                    continue;
                }

                const fs::path shader = findShader(line);
                if (shader.empty()) {
                    std::cerr << "acmxvk: playlist shader not found: " << line << '\n';
                    continue;
                }
                if (current_node != nullptr) {
                    current_node->shaders.push_back(shader);
                } else {
                    default_entries.push_back(shader);
                }
            }

            playlist.erase(std::remove_if(playlist.begin(), playlist.end(),
                                          [](const PlaylistNode &node) {
                                              return node.shaders.empty();
                                          }),
                           playlist.end());
            if (!default_entries.empty()) {
                playlist.insert(playlist.begin(), {"Default", std::move(default_entries)});
            }
            if (playlist.empty()) {
                throw std::runtime_error("playlist contains no shaders available in the SPIR-V library");
            }
            playlist_enabled = options.enable_playlist;

            std::size_t shader_count = 0;
            for (const PlaylistNode &node : playlist) {
                shader_count += node.shaders.size();
            }
            std::cout << "acmxvk: playlist loaded " << shader_count << " shaders in "
                      << playlist.size() << " nodes from " << options.playlist_file << '\n';
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
            frame_sprite->createEmptySprite(source_width, source_height, spriteVertexShader(), {});

            if (source_kind == SourceKind::Graphic) {
                frame_sprite->updateTexture(graphic_rgba.ptr(), graphic_rgba.cols,
                                            graphic_rgba.rows,
                                            static_cast<int>(graphic_rgba.step));
            } else if (!capture.readToSprite(*frame_sprite, false)) {
                std::cerr << "acmxvk: capture did not provide an initial frame\n";
            }

            applyShaderPipeline();
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

            applyShaderPipeline();
            shader_start = std::chrono::steady_clock::now();
            previous_frame = shader_start;
            frame_count = 0;
            std::cout << "acmxvk: shader " << (shader_index + 1) << '/' << shaders.size()
                      << ": " << currentShader() << '\n';
        }

        void selectPlaylistNode(int direction) {
            if (playlist.empty()) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(playlist.size());
            auto index = static_cast<std::ptrdiff_t>(playlist_index) + direction;
            index = (index % count + count) % count;
            playlist_index = static_cast<std::size_t>(index);
            applyShaderPipeline();
            std::cout << "acmxvk: playlist node " << (playlist_index + 1) << '/'
                      << playlist.size() << ": " << playlist[playlist_index].name << " ("
                      << playlist[playlist_index].shaders.size() << " passes)\n";
        }

        [[nodiscard]] std::vector<fs::path> activeShaderPipeline() const {
            std::vector<fs::path> pipeline;
            if (playlist_enabled && !playlist.empty()) {
                pipeline = playlist[playlist_index].shaders;
            } else if (multipass_enabled) {
                pipeline = configured_passes;
            }
            if (!currentShader().empty()) {
                pipeline.emplace_back(currentShader());
            }
            return pipeline;
        }

        void applyShaderPipeline() {
            if (getDevice() == VK_NULL_HANDLE) {
                return;
            }
            vkDeviceWaitIdle(getDevice());
            detachPostProcessingShader();
            post_process_sprites.clear();
            if (!effects_enabled) {
                return;
            }

            const std::vector<fs::path> pipeline = activeShaderPipeline();
            if (pipeline.empty()) {
                return;
            }

            std::vector<PostProcessingEffect> effects;
            effects.reserve(pipeline.size());
            for (const fs::path &shader : pipeline) {
                effects.push_back({shader.string(), {1.0F, 1.0F, 1.0F, 0.0F}, false});
            }
            post_process_sprites = attachPostProcessingShaders(effects);
            for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                sprite->enableExtendedUBO();
            }

            std::cout << "acmxvk: Vulkan shader pipeline (" << pipeline.size() << " passes):\n";
            for (std::size_t index = 0; index < pipeline.size(); ++index) {
                std::cout << "  " << (index + 1) << ": " << pipeline[index].filename().string()
                          << '\n';
            }
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

            for (std::size_t index = 0; index < post_process_sprites.size(); ++index) {
                mxvk::VK_Sprite *sprite = post_process_sprites[index];
                setPostProcessingShaderParams(index, 1.0F, 1.0F, 1.0F, elapsed);
                sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
                sprite->setUniform0(1.0F, 1.0F, static_cast<float>(width),
                                    static_cast<float>(height));
                sprite->setUniform1(delta, 0.0F, 0.0F, frame_rate);
                sprite->setUniform2(static_cast<float>(frame_count), elapsed, 48000.0F,
                                    0.0F);
                sprite->setUniform3(0.0F, 0.0F, 0.0F, 0.0F);
            }
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
