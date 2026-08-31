        // Resource resolution, HUD/watermark drawing, and DNN overlays.
        void resolveConfiguredResourcePaths() {
            const auto resolve = [&](std::string &path,
                                     const fs::path &resource_subdirectory,
                                     std::string_view label) {
                if (path.empty() || fs::is_regular_file(path) ||
                    fs::path(path).is_absolute()) {
                    return;
                }
                fs::path resolved = findResource(options, fs::path(path));
                if (resolved.empty()) {
                    resolved = findResource(
                        options, resource_subdirectory / fs::path(path));
                }
                if (!resolved.empty()) {
                    path = resolved.string();
                    std::cout << "acmxvk: " << label << " (resource path): "
                              << path << '\n';
                }
            };
            resolve(options.playlist_file, "playlists", "playlist");
            resolve(options.midi_map_file, "midi-examples", "MIDI map");
            if (options.enable_3d) {
                if (options.model_file.empty()) {
                    options.model_file = defaultModel().string();
                    std::cout << "acmxvk: 3D model (default): "
                              << options.model_file << '\n';
                } else {
                    resolve(options.model_file, "models", "3D model");
                }

                std::string model_name =
                    fs::path(options.model_file).filename().string();
                std::transform(
                    model_name.begin(), model_name.end(), model_name.begin(),
                    [](unsigned char character) {
                        return static_cast<char>(std::tolower(character));
                    });
                if (!model_name.ends_with(".obj") &&
                    !model_name.ends_with(".mxmod") &&
                    !model_name.ends_with(".mxmod.z")) {
                    throw std::runtime_error(
                        "--model requires an .obj, .mxmod, or .mxmod.z file");
                }
                if (!fs::is_regular_file(options.model_file)) {
                    throw std::runtime_error(
                        "3D model was not found: " + options.model_file);
                }
                constexpr std::uintmax_t MAX_MODEL_BYTES =
                    1024U * 1024U * 1024U;
                input::validate_file_size(options.model_file, "3D model",
                                          MAX_MODEL_BYTES);
            }
        }

        void initializeOverlayFont() {
            if (counter_disabled && !options.display_filter &&
                options.watermark_text.empty() && !options.interface_shm) {
                return;
            }

            const fs::path font = overlayFont();
            if (!fs::is_regular_file(font)) {
                throw std::runtime_error("overlay font was not found: " +
                                         font.string());
            }
            const VkExtent2D preview_extent = getSwapchainExtent();
            const int preview_height = preview_extent.height > 0U
                                           ? static_cast<int>(preview_extent.height)
                                           : options.height;
            constexpr int FONT_HEIGHT_DIVISOR = 60;
            overlay_font_size =
                std::max(12, preview_height / FONT_HEIGHT_DIVISOR);
            preview_overlay_font_size = overlay_font_size;
            setFont(font.string(), overlay_font_size);
            setPreviewFont(font.string(), preview_overlay_font_size);
            std::cout << "acmxvk: window-scaled output/HUD font "
                      << font.string() << " at " << overlay_font_size
                      << " points\n";
        }

        [[nodiscard]] static std::string clipOverlayText(std::string text) {
            constexpr std::size_t MAX_OVERLAY_CHARACTERS = 120;
            return input::truncate_utf8(text, MAX_OVERLAY_CHARACTERS);
        }

        [[nodiscard]] const std::vector<fs::path> *activePasses() const {
            if (playlist_enabled && !playlist.empty()) {
                return &playlist[playlist_index].shaders;
            }
            if (multipass_enabled && !configured_passes.empty()) {
                return &configured_passes;
            }
            return nullptr;
        }

        [[nodiscard]] std::string_view activeShaderRole() const {
            const std::vector<fs::path> *passes = activePasses();
            return passes != nullptr && !passes->empty() ? "Post-shader"
                                                         : "Shader";
        }

        [[nodiscard]] std::string activePassDescription() const {
            const std::vector<fs::path> *passes = activePasses();
            if (passes == nullptr || passes->empty()) {
                return {};
            }

            std::string description = "Multipass: ";
            for (std::size_t index = 0; index < passes->size(); ++index) {
                if (index > 0U) {
                    description += ", ";
                }
                description += (*passes)[index].filename().string();
            }
            return clipOverlayText(std::move(description));
        }

        [[nodiscard]] std::string activePlaylistDescription() const {
            if (!playlist_enabled || playlist.empty()) {
                return {};
            }
            std::ostringstream description;
            description << "Playlist [" << (playlist_index + 1) << '/'
                        << playlist.size() << "]: "
                        << playlist[playlist_index].name;
            return clipOverlayText(description.str());
        }

        [[nodiscard]] static std::string formatHudTime(double seconds_value) {
            const double finite_seconds =
                std::isfinite(seconds_value) ? seconds_value : 0.0;
            const auto elapsed = static_cast<std::uint64_t>(
                std::floor(std::max(0.0, finite_seconds)));
            const std::uint64_t hours = elapsed / 3600U;
            const std::uint64_t minutes = (elapsed / 60U) % 60U;
            const std::uint64_t seconds = elapsed % 60U;
            std::ostringstream text;
            text << std::setfill('0') << std::setw(2) << hours << ':'
                 << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
            return text.str();
        }

        void updateWindowTitle(bool force = false) {
            SDL_Window *window = getSDLWindow();
            if (window == nullptr) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            constexpr auto UPDATE_INTERVAL = std::chrono::milliseconds(500);
            if (!force && window_title_last_update.time_since_epoch().count() != 0 &&
                now - window_title_last_update < UPDATE_INTERVAL) {
                return;
            }
            window_title_last_update = now;

            const bool recording = writer.is_open() || options.png_output;
            double elapsed_seconds = hudWallElapsedSeconds();
            std::uint64_t displayed_frames = frame_count;
            if (recording && recording_fps > 0.0) {
                displayed_frames = output_frame_count;
                elapsed_seconds = writer.is_open()
                                      ? writer.get_duration()
                                      : static_cast<double>(output_frame_count) /
                                            recording_fps;
            } else if (source_kind == SourceKind::Video) {
                displayed_frames = video_source_frame_count;
                elapsed_seconds = hudVideoPositionSeconds();
            }

            std::ostringstream title;
            if (source_kind == SourceKind::Graphic) {
                title << "ACMXVK - Graphics Mode - "
                      << formatHudTime(elapsed_seconds) << " ["
                      << displayed_frames << " frames]";
            } else if (source_kind == SourceKind::Video) {
                const std::uint64_t total_frames =
                    video_duration_seconds > 0.0 && video_source_fps > 0.0
                        ? static_cast<std::uint64_t>(std::llround(
                              video_duration_seconds * video_source_fps))
                        : 0U;
                title << "ACMXVK - [" << video_source_frame_count << '/';
                if (total_frames > 0U) {
                    title << total_frames;
                } else {
                    title << '?';
                }
                title << "] - " << formatHudTime(elapsed_seconds)
                      << " - Video Mode";
            } else {
                title << "ACMXVK - Capture Mode - "
                      << formatHudTime(elapsed_seconds) << " ["
                      << displayed_frames << " frames]";
            }

            if (recording) {
                title << " (Recording)";
                if (writer.is_open()) {
                    constexpr double BYTES_PER_MEGABYTE = 1024.0 * 1024.0;
                    const double file_size_mb =
                        static_cast<double>(writer.get_bytes_written()) /
                        BYTES_PER_MEGABYTE;
                    title << " [File: " << std::fixed << std::setprecision(2)
                          << file_size_mb << " MB]";
                }
            } else {
                title << " (Preview)";
            }

            const std::string text = title.str();
            SDL_SetWindowTitle(window, text.c_str());
        }

        [[nodiscard]] double hudWallElapsedSeconds() const {
            return std::max(
                0.0,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              hud_session_start)
                    .count());
        }

        [[nodiscard]] bool currentVideoTimeline(
            double &timeline,
            std::uint64_t *frame_index = nullptr) const {
            if (source_kind != SourceKind::Video ||
                video_source_frame_count == 0U ||
                !std::isfinite(video_source_fps) || video_source_fps <= 0.0) {
                return false;
            }
            const std::uint64_t index = video_source_frame_count - 1U;
            timeline = static_cast<double>(index) / video_source_fps;
            if (frame_index != nullptr) {
                *frame_index = index;
            }
            return true;
        }

        [[nodiscard]] double hudVideoPositionSeconds() const {
            double position = 0.0;
            if (!currentVideoTimeline(position)) {
                return 0.0;
            }
            if (video_duration_seconds > 0.0) {
                position = std::min(position, video_duration_seconds);
            }
            return std::max(0.0, position);
        }

        [[nodiscard]] std::string hudVideoTimeString() const {
            std::string text = "Video: " +
                               formatHudTime(hudVideoPositionSeconds()) +
                               " / ";
            text += video_duration_seconds > 0.0
                        ? formatHudTime(video_duration_seconds)
                        : "--:--:--";
            return text;
        }

        [[nodiscard]] std::string hudElapsedTimeString() const {
            return "Elapsed: " + formatHudTime(hudWallElapsedSeconds());
        }

        void updateHudFrameRate() {
            ++hud_fps_frame_count;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - hud_fps_last_tick).count();
            if (elapsed < 0.5) {
                return;
            }
            hud_display_fps = static_cast<double>(hud_fps_frame_count) / elapsed;
            hud_fps_frame_count = 0;
            hud_fps_last_tick = now;
        }

        void paceMaximizedRendering() {
            if (!options.maximize_fps || options.requested_fps <= 0.0) {
                return;
            }

            const auto interval = std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / options.requested_fps));
            const auto now = std::chrono::steady_clock::now();
            if (!render_pacing_started) {
                render_pacing_started = true;
                next_render_tick = now;
                return;
            }

            next_render_tick += interval;
            if (next_render_tick > now) {
                std::this_thread::sleep_until(next_render_tick);
                return;
            }

            if (now - next_render_tick > interval * 4) {
                next_render_tick = now;
            }
        }

        void updateCameraFrameRate() {
            if (source_kind != SourceKind::Camera) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (camera_fps_frame_count == 0) {
                camera_fps_frame_count = 1;
                camera_fps_last_tick = now;
                return;
            }

            ++camera_fps_frame_count;
            const double elapsed = std::chrono::duration<double>(
                                       now - camera_fps_last_tick)
                                       .count();
            if (elapsed < 1.0) {
                return;
            }

            camera_delivered_fps =
                static_cast<double>(camera_fps_frame_count - 1) / elapsed;
            camera_fps_frame_count = 1;
            camera_fps_last_tick = now;

            const double log_threshold = std::max(
                5.0, camera_last_logged_fps * 0.2);
            if (camera_last_logged_fps <= 0.0 ||
                std::abs(camera_delivered_fps - camera_last_logged_fps) >=
                    log_threshold) {
                std::ostringstream status;
                status << "acmxvk: camera delivery: " << std::fixed
                       << std::setprecision(1) << camera_delivered_fps
                       << " FPS measured";
                if (camera_reported_fps > 0.0) {
                    status << " (driver reports " << camera_reported_fps
                           << " FPS)";
                }
                std::cout << status.str() << '\n';
                camera_last_logged_fps = camera_delivered_fps;
            }
        }

        void queueRuntimeHud(int &y, int line_height) {
            if (counter_disabled) {
                return;
            }
            updateHudFrameRate();

            const SDL_Color shader_color{0U, 96U, 255U, 255U};
            std::string shader = effects_enabled
                                     ? fs::path(currentShader()).filename().string()
                                     : "bypassed";
            if (shader_locked) {
                shader += " [locked]";
            }
            printPreviewText(clipOverlayText(
                                 std::string(activeShaderRole()) + ": " +
                                 std::move(shader)),
                             10, y, shader_color);
            y += line_height;

            const SDL_Color crossfade_color{255U, 192U, 0U, 255U};
            std::ostringstream crossfade_status;
            crossfade_status << "XFade [" << (crossfade_shader_index + 1)
                             << '/' << CROSSFADE_NAMES.size() << "]: "
                             << CROSSFADE_NAMES[crossfade_shader_index];
            printPreviewText(clipOverlayText(crossfade_status.str()), 10, y,
                             crossfade_color);
            y += line_height;

            const std::string playlist_description =
                activePlaylistDescription();
            if (!playlist_description.empty()) {
                const SDL_Color playlist_color{255U, 0U, 255U, 255U};
                printPreviewText(playlist_description, 10, y,
                                 playlist_color);
                y += line_height;
            }

            const std::vector<fs::path> *passes = activePasses();
            if (passes != nullptr && !passes->empty()) {
                constexpr std::size_t MAX_HUD_PASS_LINES = 8U;
                const std::size_t displayed_passes =
                    std::min(passes->size(), MAX_HUD_PASS_LINES);
                for (std::size_t index = 0; index < displayed_passes;
                     ++index) {
                    std::ostringstream pass;
                    pass << "Pass [" << (index + 1) << '/' << passes->size()
                         << "]: " << (*passes)[index].filename().string();
                    printPreviewText(clipOverlayText(pass.str()), 10, y,
                                     shader_color);
                    y += line_height;
                }
                if (displayed_passes < passes->size()) {
                    const std::string remaining =
                        "Passes: +" +
                        std::to_string(passes->size() - displayed_passes) +
                        " more";
                    printPreviewText(remaining, 10, y, shader_color);
                    y += line_height;
                }
            }

            if (model_initialized) {
                const SDL_Color model_color{0U, 220U, 180U, 255U};
                std::string model_status =
                    model_3d_active ? "Model: " : "Model (2D bypass): ";
                model_status +=
                    fs::path(options.model_file).filename().string();
                if (model_wave_active) {
                    model_status += " [wave]";
                }
                if (model_scale_oscillation_active) {
                    model_status += " [oscillate]";
                }
                printPreviewText(clipOverlayText(std::move(model_status)), 10,
                                 y, model_color);
                y += line_height;
            }

#ifdef ACMXVK_WITH_DNN
            const SDL_Color dnn_color{64U, 220U, 128U, 255U};
            if (human_segmenter != nullptr) {
                printPreviewText(
                    options.human_background
                        ? "DNN: PP-HumanSeg [background]"
                        : "DNN: PP-HumanSeg [foreground]",
                    10, y, dnn_color);
                y += line_height;
            }
            if (edge_detector != nullptr) {
                printPreviewText("DNN: DexiNed edge", 10, y, dnn_color);
                y += line_height;
            }
            if (generic_onnx_processor != nullptr) {
                printPreviewText(
                    clipOverlayText(
                        "DNN: ONNX " +
                        fs::path(options.onnx_configuration)
                            .filename()
                            .string()),
                    10, y, dnn_color);
                y += line_height;
            }
#endif

#ifdef AUDIO_ENABLED
            if (file_audio_source != nullptr && file_audio_source->is_open()) {
                const std::string track = fs::path(
                                              file_audio_source
                                                  ->current_track_path())
                                              .filename()
                                              .string();
                if (!track.empty()) {
                    const SDL_Color track_color{255U, 0U, 255U, 255U};
                    printPreviewText(clipOverlayText("Track: " + track), 10,
                                     y, track_color);
                    y += line_height;
                }
            }
#endif

#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                const SDL_Color gpu_color{255U, 0U, 255U, 255U};
                printPreviewText(
                    clipOverlayText(
                        "GPU: " +
                        gpu_filter_engine->active_filter_description()),
                    10, y, gpu_color);
                y += line_height;
            }
#endif

            if (autopilot_enabled) {
                const int remaining =
                    std::max(0, autopilot_interval_frames - autopilot_counter);
                std::ostringstream status;
                status << "Autopilot "
                       << (autopilot_sequential ? "seq" : "rnd") << ' ';
                if (options.autopilot_random_timeout > 0) {
                    status << "[4-" << options.autopilot_random_timeout
                           << "] cur=" << autopilot_interval_frames;
                } else {
                    status << "every " << autopilot_interval_frames << 'f';
                }
                status << " next=" << remaining << "f";
                if (!playlist.empty()) {
                    status << " idx=" << (playlist_index + 1) << '/'
                           << playlist.size();
                }
                const SDL_Color autopilot_color{0U, 255U, 255U, 255U};
                printPreviewText(clipOverlayText(status.str()), 10, y,
                                 autopilot_color);
                y += line_height;
            }

            const SDL_Color status_color{255U, 255U, 255U, 255U};
            if (source_kind == SourceKind::Video) {
                printPreviewText(hudVideoTimeString(), 10, y, status_color);
                y += line_height;
            }
            printPreviewText(hudElapsedTimeString(), 10, y, status_color);
            y += line_height;
            std::ostringstream fps;
            fps << "Render: " << std::fixed << std::setprecision(1)
                << hud_display_fps << " FPS";
            printPreviewText(fps.str(), 10, y, status_color);
            y += line_height;
            if (source_kind == SourceKind::Camera) {
                std::ostringstream camera_fps;
                camera_fps << "Camera: ";
                if (camera_delivered_fps > 0.0) {
                    camera_fps << std::fixed << std::setprecision(1)
                               << camera_delivered_fps << " FPS measured";
                } else {
                    camera_fps << "measuring...";
                }
                printPreviewText(camera_fps.str(), 10, y, status_color);
                y += line_height;
            }
            const SDL_Color hint_color{128U, 128U, 128U, 255U};
            printPreviewText("F9: Toggle overlay", 10, y, hint_color);
            y += line_height;
        }

        void queueOverlayText() {
            if (counter_disabled && !options.display_filter &&
                (!watermark_enabled || options.watermark_text.empty())) {
                return;
            }

            constexpr int LEFT_MARGIN = 10;
            constexpr int TOP_MARGIN = 10;
            const int line_height = overlay_font_size + 4;
            const int preview_line_height = preview_overlay_font_size + 4;
            int preview_y =
                TOP_MARGIN +
                (!counter_disabled && watermark_enabled &&
                         !options.watermark_text.empty()
                     ? preview_line_height
                     : 0);
            queueRuntimeHud(preview_y, preview_line_height);
            int y = TOP_MARGIN;
            if (options.display_filter) {
                const SDL_Color filter_color{255U, 0U, 255U, 255U};
                std::string shader = effects_enabled
                                         ? fs::path(currentShader()).filename().string()
                                         : "bypassed";
                printText(clipOverlayText(
                              std::string(activeShaderRole()) + ": " +
                              std::move(shader)),
                          LEFT_MARGIN, y, filter_color);
                y += line_height;

                if (playlist_enabled && !playlist.empty()) {
                    printText(clipOverlayText("Playlist: " +
                                              playlist[playlist_index].name),
                              LEFT_MARGIN, y, filter_color);
                    y += line_height;
                }
                const std::string passes = activePassDescription();
                if (!passes.empty()) {
                    printText(passes, LEFT_MARGIN, y, filter_color);
                    y += line_height;
                }
#ifdef ACMXVK_WITH_CUDA
                if (gpu_filter_engine != nullptr) {
                    printText(clipOverlayText(
                                  "GPU: " + gpu_filter_engine
                                                ->active_filter_description()),
                              LEFT_MARGIN, y, filter_color);
                    y += line_height;
                }
#endif
            }

            if (watermark_enabled && !options.watermark_text.empty()) {
                const SDL_Color watermark_color{
                    options.watermark_color[0], options.watermark_color[1],
                    options.watermark_color[2], 255U};
                printText(clipOverlayText(options.watermark_text), LEFT_MARGIN,
                          y, watermark_color);
            }
        }

        [[nodiscard]] static std::string captureFourccName(double value) {
            if (!std::isfinite(value) || value <= 0.0 ||
                value > static_cast<double>(
                            std::numeric_limits<std::uint32_t>::max())) {
                return "unknown";
            }
            const auto fourcc = static_cast<std::uint32_t>(std::llround(value));
            std::string name(4, ' ');
            for (std::size_t index = 0; index < name.size(); ++index) {
                const auto byte = static_cast<unsigned char>(
                    (fourcc >> (index * 8U)) & 0xffU);
                if (!std::isprint(byte)) {
                    return "unknown";
                }
                name[index] = static_cast<char>(byte);
            }
            return name;
        }

        [[nodiscard]] bool dnnHostProcessingEnabled() const {
#ifdef ACMXVK_WITH_DNN
            return edge_detector != nullptr || human_segmenter != nullptr ||
                   generic_onnx_processor != nullptr;
#else
            return false;
#endif
        }

        void applyDnnEffects(cv::Mat &rgba) {
#ifdef ACMXVK_WITH_DNN
            if (human_segmenter != nullptr && !rgba.empty()) {
                cv::Mat bgr;
                cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                const cv::Mat mask = human_segmenter->infer(bgr);
                if (mask.empty()) {
                    throw std::runtime_error(
                        "PP-HumanSeg produced an empty person mask");
                }
                const float black_point =
                    static_cast<float>(options.human_black_point);
                const float white_point =
                    static_cast<float>(options.human_white_point);
                if (options.human_background) {
                    const cv::Mat alpha = dnn::hardenedAlphaMask(
                        bgr, mask, black_point, white_point);
                    cv::cvtColor(bgr, human_overlay_rgba,
                                 cv::COLOR_BGR2RGBA);
                    std::vector<cv::Mat> overlay_channels;
                    cv::split(human_overlay_rgba, overlay_channels);
                    alpha.copyTo(overlay_channels[3]);
                    cv::merge(overlay_channels, human_overlay_rgba);

                    const cv::Mat foreground = dnn::isolateBody(
                        bgr, mask, black_point, white_point);
                    cv::Mat background;
                    cv::subtract(bgr, foreground, background);
                    cv::cvtColor(background, rgba, cv::COLOR_BGR2RGBA);
                } else {
                    const cv::Mat foreground = dnn::isolateBody(
                        bgr, mask, black_point, white_point);
                    cv::cvtColor(foreground, rgba, cv::COLOR_BGR2RGBA);
                }
            }
            if (edge_detector != nullptr && !rgba.empty()) {
                try {
                    cv::Mat bgr;
                    cv::Mat edge;
                    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                    edge_detector->process(bgr, edge);
                    if (edge.empty()) {
                        throw std::runtime_error(
                            "DexiNed produced an empty edge frame");
                    }
                    if (edge.channels() == 1) {
                        cv::cvtColor(edge, rgba, cv::COLOR_GRAY2RGBA);
                    } else {
                        cv::cvtColor(edge, rgba, cv::COLOR_BGR2RGBA);
                    }
                } catch (const std::exception &error) {
                    std::cerr
                        << "acmxvk: edge inference failed; disabling DNN "
                           "effect: "
                        << error.what() << '\n';
                    edge_detector.reset();
                }
            }
            if (generic_onnx_processor != nullptr && !rgba.empty()) {
                try {
                    cv::Mat bgr;
                    cv::Mat processed;
                    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                    generic_onnx_processor->process(bgr, processed);
                    if (processed.empty()) {
                        throw std::runtime_error(
                            "generic ONNX model produced an empty frame");
                    }
                    if (processed.channels() == 1) {
                        cv::cvtColor(processed, rgba,
                                     cv::COLOR_GRAY2RGBA);
                    } else {
                        cv::cvtColor(processed, rgba,
                                     cv::COLOR_BGR2RGBA);
                    }
                } catch (const std::exception &error) {
                    std::cerr
                        << "acmxvk: generic ONNX inference failed; disabling "
                           "model: "
                        << error.what() << '\n';
                    generic_onnx_processor.reset();
                }
            }
#else
            static_cast<void>(rgba);
#endif
        }

        void updateHumanOverlayTexture() {
#ifdef ACMXVK_WITH_DNN
            if (!options.human_background || human_overlay_rgba.empty() ||
                getDevice() == VK_NULL_HANDLE) {
                return;
            }
            if (human_overlay_sprite == nullptr) {
                human_overlay_sprite = createSprite(1, 1);
                human_overlay_sprite->enableHistoryTexture(
                    static_cast<std::uint32_t>(human_overlay_rgba.cols),
                    static_cast<std::uint32_t>(human_overlay_rgba.rows), 1U);
            }
            cv::Mat upload = human_overlay_rgba;
            cv::Mat flipped;
            if (options.flip_output) {
                cv::flip(human_overlay_rgba, flipped, 0);
                upload = flipped;
            }
            human_overlay_sprite->updateHistoryTexture(
                upload.ptr(), upload.cols, upload.rows,
                static_cast<int>(upload.step));
#endif
        }
