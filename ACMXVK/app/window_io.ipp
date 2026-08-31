        // Input setup, output encoding, snapshots, and readback handling.
        void openInput() {
            if (!options.graphic_file.empty()) {
                source_kind = SourceKind::Graphic;
                graphic_rgba = loadRgbaImage(options.graphic_file);
                applyDnnEffects(graphic_rgba);
                rotateFrame(graphic_rgba, options.frame_rotation);
                if (!human_overlay_rgba.empty()) {
                    rotateFrame(human_overlay_rgba, options.frame_rotation);
                }
                return;
            }

            source_kind = options.input_file.empty() ? SourceKind::Camera : SourceKind::Video;
            bool opened = false;
            if (source_kind == SourceKind::Video) {
                opened = openVideoCapture();
            } else {
                opened = capture.open(options.camera_device);
            }
            if (!opened) {
                const std::string source = source_kind == SourceKind::Video
                                               ? options.input_file
                                               : std::to_string(options.camera_device);
                throw std::runtime_error("unable to open capture source: " + source);
            }

            if (source_kind == SourceKind::Video) {
                video_duration_seconds =
                    probeVideoDuration(options.input_file);
                std::ostringstream timeline;
                timeline << "acmxvk: video timeline: " << std::fixed
                         << std::setprecision(3) << video_source_fps
                         << " FPS";
                if (video_duration_seconds > 0.0) {
                    timeline << ", " << video_duration_seconds
                             << " seconds";
                } else {
                    timeline << ", duration unavailable";
                }
                std::cout << timeline.str() << '\n';
                if (options.use_source_fps) {
                    std::cout
                        << "acmxvk: source-FPS playback enabled at "
                        << video_source_fps
                        << " FPS; early frames wait and late frames are skipped\n";
                }
            }

            if (source_kind == SourceKind::Camera) {
                // Match ACMX2's ordering. Some V4L2 drivers renegotiate the
                // frame interval when dimensions or pixel format change.
                capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);
                capture.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
                const int requested_fourcc = options.use_yuv
                                                 ? cv::VideoWriter::fourcc(
                                                       'Y', 'U', 'Y', 'V')
                                                 : cv::VideoWriter::fourcc(
                                                       'M', 'J', 'P', 'G');
                capture.set(cv::CAP_PROP_FOURCC,
                            static_cast<double>(requested_fourcc));
                if (options.requested_fps > 0.0) {
                    capture.set(cv::CAP_PROP_FPS, options.requested_fps);
                }

                camera_reported_width = static_cast<int>(
                    std::lround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
                camera_reported_height = static_cast<int>(
                    std::lround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
                camera_reported_fps = capture.get(cv::CAP_PROP_FPS);
                if (!std::isfinite(camera_reported_fps) ||
                    camera_reported_fps < 0.0) {
                    camera_reported_fps = 0.0;
                }
                const std::string reported_fourcc = captureFourccName(
                    capture.get(cv::CAP_PROP_FOURCC));

                std::cout << "acmxvk: camera opened: "
                          << camera_reported_width << 'x'
                          << camera_reported_height;
                if (camera_reported_fps > 0.0) {
                    std::cout << " at reported " << camera_reported_fps
                              << " FPS";
                } else {
                    std::cout << " at an unreported frame rate";
                }
                std::cout << ", format=" << reported_fourcc << '\n';

                if (camera_reported_width != options.camera_width ||
                    camera_reported_height != options.camera_height) {
                    std::cerr << "acmxvk: camera mode warning: requested "
                              << options.camera_width << 'x'
                              << options.camera_height << " but driver reports "
                              << camera_reported_width << 'x'
                              << camera_reported_height
                              << '\n';
                }
                if (options.requested_fps > 0.0 &&
                    camera_reported_fps > 0.0 &&
                    std::abs(camera_reported_fps - options.requested_fps) >
                        0.05) {
                    std::cerr << "acmxvk: camera mode warning: requested "
                              << options.requested_fps
                              << " FPS but driver reports "
                              << camera_reported_fps << " FPS\n";
                }
                const std::string requested_format =
                    options.use_yuv ? "YUYV" : "MJPG";
                if (reported_fourcc != "unknown" &&
                    reported_fourcc != requested_format) {
                    std::cerr << "acmxvk: camera mode warning: requested "
                              << requested_format << " but driver reports "
                              << reported_fourcc << '\n';
                }
                if (options.maximize_fps) {
                    latest_camera_frame.start(capture);
                    std::cout
                        << "acmxvk: maximize FPS active: asynchronous camera "
                           "capture, Vulkan render target "
                        << options.requested_fps << " FPS\n";
                    if (options.enable_vsync) {
                        std::cout
                            << "acmxvk: maximize FPS note: VSync may cap the "
                               "render rate to the display refresh\n";
                    }
                }
            }
        }

        [[nodiscard]] std::pair<int, int> source_dimensions() {
            int source_width = options.width;
            int source_height = options.height;
            if (source_kind == SourceKind::Graphic) {
                source_width = graphic_rgba.cols;
                source_height = graphic_rgba.rows;
            } else {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                if (using_ffmpeg_capture) {
                    source_width = ffmpeg_capture.width();
                    source_height = ffmpeg_capture.height();
                } else
#endif
                {
                    if (source_kind == SourceKind::Camera &&
                        camera_reported_width > 0 &&
                        camera_reported_height > 0) {
                        source_width = camera_reported_width;
                        source_height = camera_reported_height;
                    } else {
                        source_width = static_cast<int>(
                            std::lround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
                        source_height = static_cast<int>(
                            std::lround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
                    }
                }
                if (source_width <= 0 || source_height <= 0) {
                    source_width = source_kind == SourceKind::Camera
                                       ? options.camera_width
                                       : options.width;
                    source_height = source_kind == SourceKind::Camera
                                        ? options.camera_height
                                        : options.height;
                }
                if (rotationSwapsDimensions(options.frame_rotation)) {
                    std::swap(source_width, source_height);
                }
            }
            return {source_width, source_height};
        }

        void configureRenderResolution() {
            int render_width = options.width;
            int render_height = options.height;
            if (!options.resolution_specified) {
                const auto [source_width, source_height] = source_dimensions();
                if (!dimensions_supported(source_width, source_height)) {
                    throw std::runtime_error(
                        "input source dimensions are outside the supported range");
                }

                render_width = source_width;
                render_height = source_height;
                options.width = render_width;
                options.height = render_height;
                const char *source_name = source_kind == SourceKind::Video
                                              ? "video"
                                          : source_kind == SourceKind::Camera
                                              ? "camera"
                                              : "graphic";
                std::cout << "acmxvk: automatic output resolution: "
                          << render_width << 'x' << render_height << " from "
                          << source_name;
                if (rotationSwapsDimensions(options.frame_rotation)) {
                    std::cout << " after input rotation";
                }
                std::cout << '\n';
            } else {
                std::cout << "acmxvk: requested output resolution: "
                          << render_width << 'x' << render_height << '\n';
            }
            setRenderExtent(static_cast<std::uint32_t>(render_width),
                            static_cast<std::uint32_t>(render_height));

            if (options.fullscreen) {
                std::cout << "acmxvk: fullscreen presentation uses the display "
                             "extent without changing the output resolution\n";
                return;
            }

            SDL_Window *window = getSDLWindow();
            if (window == nullptr) {
                throw std::runtime_error(
                    "unable to configure preview without an SDL window");
            }

            int preview_width = render_width;
            int preview_height = render_height;
            SDL_Rect usable_bounds{};
            SDL_DisplayID display = SDL_GetDisplayForWindow(window);
            if (display == 0) {
                display = SDL_GetPrimaryDisplay();
            }
            if (display != 0 &&
                SDL_GetDisplayUsableBounds(display, &usable_bounds) &&
                usable_bounds.w > 0 && usable_bounds.h > 0) {
                constexpr double PREVIEW_DISPLAY_FRACTION = 0.9;
                const double width_scale =
                    (static_cast<double>(usable_bounds.w) *
                     PREVIEW_DISPLAY_FRACTION) /
                    render_width;
                const double height_scale =
                    (static_cast<double>(usable_bounds.h) *
                     PREVIEW_DISPLAY_FRACTION) /
                    render_height;
                const double preview_scale =
                    std::min({1.0, width_scale, height_scale});
                preview_width = std::max(
                    1, static_cast<int>(std::lround(render_width * preview_scale)));
                preview_height = std::max(
                    1, static_cast<int>(std::lround(render_height * preview_scale)));
            }

            const float render_aspect = static_cast<float>(render_width) /
                                        static_cast<float>(render_height);
            if (!SDL_SetWindowAspectRatio(window, render_aspect,
                                          render_aspect)) {
                std::cerr << "acmxvk: unable to lock preview aspect ratio: "
                          << SDL_GetError() << '\n';
            }
            if (!SDL_SetWindowSize(window, preview_width, preview_height)) {
                throw std::runtime_error(
                    std::string("unable to apply preview resolution: ") +
                    SDL_GetError());
            }
            SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED);
            if (!SDL_SyncWindow(window)) {
                std::cerr << "acmxvk: window resize sync warning: "
                          << SDL_GetError() << '\n';
            }

            int actual_width = 0;
            int actual_height = 0;
            SDL_GetWindowSizeInPixels(window, &actual_width, &actual_height);
            std::cout << "acmxvk: preview resolution: " << actual_width << 'x'
                      << actual_height;
            if (preview_width != render_width ||
                preview_height != render_height) {
                std::cout << " (" << render_width << 'x' << render_height
                          << " output, preview fitted to display)";
            }
            std::cout << '\n';
        }

        [[nodiscard]] double outputFrameRate() {
            if (options.requested_fps > 0.0) {
                return options.requested_fps;
            }
            if (source_kind != SourceKind::Graphic) {
                double source_fps = 0.0;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                if (using_ffmpeg_capture) {
                    source_fps = ffmpeg_capture.fps();
                } else
#endif
                {
                    source_fps = capture.get(cv::CAP_PROP_FPS);
                }
                if (std::isfinite(source_fps) && source_fps > 0.0) {
                    return source_fps;
                }
            }
            return 30.0;
        }

        [[nodiscard]] static fs::path outputFrameDirectory(const std::string &filename,
                                                           std::string_view suffix) {
            const fs::path output_path(filename);
            const fs::path parent = output_path.has_parent_path()
                                        ? output_path.parent_path()
                                        : fs::path(".");
            const std::string name = output_path.filename().empty()
                                         ? std::string("output")
                                         : output_path.filename().string();
            return parent / ("video_file-" + name + "-" + std::string(suffix));
        }

        static void createOutputDirectory(const fs::path &directory) {
            std::error_code error;
            fs::create_directories(directory, error);
            if (error || !fs::is_directory(directory)) {
                throw std::runtime_error("unable to create PNG output directory: " +
                                         directory.string());
            }
        }

        [[nodiscard]] static fs::path framePath(const fs::path &directory,
                                                std::uint64_t index) {
            std::ostringstream filename;
            filename << "frame-" << std::setfill('0') << std::setw(8) << index << ".png";
            return directory / filename.str();
        }

        [[nodiscard]] fs::path snapshotPath(std::uint32_t width,
                                            std::uint32_t height,
                                            SnapshotFormat format) {
            const auto now = std::chrono::system_clock::now();
            const std::time_t now_time =
                std::chrono::system_clock::to_time_t(now);
            std::tm local_time{};
#ifdef _WIN32
            localtime_s(&local_time, &now_time);
#else
            localtime_r(&now_time, &local_time);
#endif
            const fs::path directory(options.snapshot_directory);
            while (true) {
                std::ostringstream filename;
                filename << "ACMXVK.Snapshot-"
                         << std::put_time(&local_time, "%Y.%m.%d-%H.%M.%S")
                         << '-' << width << 'x' << height << '-'
                         << snapshot_count << SnapshotWriter::extension(format);
                const fs::path candidate = directory / filename.str();
                if (!fs::exists(candidate)) {
                    return candidate;
                }
                ++snapshot_count;
            }
        }

        void requestSnapshot(SnapshotFormat format) {
#ifndef ACMXVK_WITH_TIFF
            if (format == SnapshotFormat::Tiff) {
                std::cerr << "acmxvk: TIFF snapshots require a build configured "
                             "with -DTIFF=ON\n";
                return;
            }
#endif
#ifndef ACMXVK_WITH_WEBP
            if (format == SnapshotFormat::WebP) {
                std::cerr << "acmxvk: WebP snapshots require a build configured "
                             "with -DWEBP=ON\n";
                return;
            }
#endif
            if (snapshot_pending) {
                return;
            }
            if (snapshot_writer.queueFull()) {
                std::cerr << "acmxvk: snapshot queue is full; request ignored\n";
                return;
            }
            std::error_code error;
            const fs::path directory(options.snapshot_directory);
            fs::create_directories(directory, error);
            if (error || !fs::is_directory(directory)) {
                std::cerr << "acmxvk: unable to create snapshot directory: "
                          << directory.string() << '\n';
                return;
            }
            if (!snapshot_writer.start()) {
                return;
            }
            snapshot_pending = true;
            pending_snapshot_format = format;
            setFrameReadbackEnabled(true);
            std::cout << "acmxvk: " << SnapshotWriter::formatName(format)
                      << " snapshot requested\n";
        }

        [[nodiscard]] bool continuousReadbackEnabled() const {
            return writer.is_open() || options.png_output ||
                   options.generate_interval > 0;
        }

        void openOutput() {
            if (options.output_file.empty() && options.generate_interval <= 0) {
                return;
            }

            const VkExtent2D extent = getRenderExtent();
            if (options.resolution_specified) {
                recording_width = extent.width > 0U
                                      ? static_cast<int>(extent.width)
                                      : options.width;
                recording_height = extent.height > 0U
                                       ? static_cast<int>(extent.height)
                                       : options.height;
            } else {
                recording_width = options.width;
                recording_height = options.height;
            }
            recording_fps = outputFrameRate();

            if (options.png_output) {
                png_output_directory = outputFrameDirectory(options.output_file, "png");
                createOutputDirectory(png_output_directory);
                std::cout << "acmxvk: writing PNG sequence to "
                          << png_output_directory.string() << '\n';
            }

            if (options.generate_interval > 0) {
                if (!options.output_file.empty()) {
                    generate_output_directory =
                        outputFrameDirectory(options.output_file, "generate");
                } else if (!options.input_file.empty()) {
                    generate_output_directory =
                        outputFrameDirectory(options.input_file, "generate");
                } else {
                    generate_output_directory = "camera-generate";
                }
                createOutputDirectory(generate_output_directory);
                std::cout << "acmxvk: saving every " << options.generate_interval
                          << "th frame to " << generate_output_directory.string() << '\n';
            }

            if (!options.output_file.empty() && !options.png_output) {
                EncodeOptions encode_options;
                encode_options.preset = options.encode_preset;
                encode_options.tune = options.encode_tune;
                encode_options.crf = options.encode_crf;
                encode_options.codec = options.encode_codec;
                encode_options.ffmpeg_options = options.encode_params;
                encode_options.realtime = options.encode_realtime;
                encode_options.block_when_full = options.no_drop;

                if (!writer.open(options.output_file, recording_width, recording_height,
                                 static_cast<float>(recording_fps), encode_options)) {
                    throw std::runtime_error("unable to open output video: " +
                                             options.output_file);
                }
                writer.set_block_when_full(options.no_drop);
                std::cout << "acmxvk: recording " << recording_width << 'x'
                          << recording_height << " at " << recording_fps << " FPS to "
                          << options.output_file
                          << (options.no_drop ? " (no-drop)\n" : "\n");
                if (options.mute_output) {
                    std::cout
                        << "acmxvk: recorded video audio disabled (--mute-output); "
                           "reactivity and pass-through remain active\n";
                }
            }

            setFrameReadbackEnabled(true);
        }

        void onFrameReadbackScheduled() override {
            ReadbackRequest request;
            request.snapshot = snapshot_pending;
            request.snapshot_format = pending_snapshot_format;
            request.continuous = continuousReadbackEnabled();
            request.frame_due = recording_frame_due;
            request.has_pts = recording_frame_has_pts;
            request.pts = recording_frame_pts;
            readback_requests.push_back(request);

            if (snapshot_pending) {
                snapshot_pending = false;
                if (!request.continuous) {
                    setFrameReadbackEnabled(false);
                }
            }
        }

        void onFrameReadback(std::vector<std::uint8_t> &rgba, uint32_t width,
                             uint32_t height) override {
            if (readback_requests.empty()) {
                std::cerr << "acmxvk: received frame readback without queued metadata\n";
                return;
            }
            const ReadbackRequest request = readback_requests.front();
            readback_requests.pop_front();

            if (request.snapshot) {
                const fs::path path =
                    snapshotPath(width, height, request.snapshot_format);
                SnapshotJob job;
                job.path = path;
                job.width = width;
                job.height = height;
                job.format = request.snapshot_format;
                if (request.continuous) {
                    job.rgba = rgba;
                } else {
                    job.rgba = std::move(rgba);
                }
                snapshot_writer.enqueue(std::move(job));
                ++snapshot_count;
                std::cout << "acmxvk: queued "
                          << SnapshotWriter::formatName(request.snapshot_format)
                          << " snapshot: " << path.string() << '\n';
            }

            if (!request.continuous || recording_complete ||
                !request.frame_due) {
                return;
            }

            std::uint8_t *output_pixels = rgba.data();
            cv::Mat resized;
            if (static_cast<int>(width) != recording_width ||
                static_cast<int>(height) != recording_height) {
                const cv::Mat source(static_cast<int>(height), static_cast<int>(width),
                                     CV_8UC4, rgba.data());
                cv::resize(source, resized, cv::Size(recording_width, recording_height),
                           0.0, 0.0, cv::INTER_LINEAR);
                output_pixels = resized.ptr();
            }

            if (writer.is_open()) {
                if (request.has_pts) {
                    writer.write_at_pts(
                        output_pixels,
                        static_cast<std::int64_t>(request.pts));
                } else {
                    writer.write(output_pixels);
                }
            }
            if (options.png_output) {
                SnapshotWriter::savePng(
                    framePath(png_output_directory, png_frame_count),
                    output_pixels, recording_width, recording_height);
                ++png_frame_count;
            }
            if (options.generate_interval > 0 &&
                (request.has_pts ? request.pts : output_frame_count) %
                        static_cast<std::uint64_t>(options.generate_interval) ==
                    0) {
                SnapshotWriter::savePng(
                    framePath(generate_output_directory, generated_frame_count),
                    output_pixels, recording_width, recording_height);
                ++generated_frame_count;
            }
            ++output_frame_count;

            if (options.duration > 0.0) {
                double output_duration = 0.0;
                if (request.has_pts) {
                    output_duration =
                        static_cast<double>(request.pts + 1) / recording_fps;
                } else if (writer.is_open()) {
                    output_duration = writer.get_duration();
                } else {
                    output_duration =
                        static_cast<double>(output_frame_count) / recording_fps;
                }
                if (output_duration >= options.duration) {
                    recording_complete = true;
                    exit();
                }
            }

            if (options.max_size_mb > 0.0 && writer.is_open()) {
                const double maximum_bytes = options.max_size_mb * 1024.0 * 1024.0;
                if (static_cast<double>(writer.get_bytes_written()) >=
                    maximum_bytes) {
                    std::cout << "acmxvk: maximum output size reached ("
                              << options.max_size_mb << " MB)\n";
                    recording_complete = true;
                    exit();
                }
            }
        }
