        // 3D rendering, crossfades, pipelines, history, and frame uploads.
        void initializeModel() {
            if (!options.enable_3d || model_initialized) {
                return;
            }

            try {
                input_model.enableExtendedFragmentUniforms();
                input_model.load(this, options.model_file, "", "", 1.0F);
                input_model.setShaders(
                    this, modelVertexShader().string(),
                    modelFragmentShader().string());
                model_effect_shader = modelFragmentShader();
                input_model.setBackfaceCulling(false);
                model_initialized = true;
                model_3d_active = true;
                model_last_render_time = std::chrono::steady_clock::now();
                std::cout << "acmxvk: loaded 3D model: "
                          << options.model_file << " ("
                          << input_model.model().vertices().size()
                          << " vertices, "
                          << input_model.model().indexCount()
                          << " indices; skybox camera centered; view rotation "
                          << (model_auto_rotate ? "enabled" : "disabled")
                          << ")\n";
            } catch (...) {
                if (getDevice() != VK_NULL_HANDLE) {
                    vkDeviceWaitIdle(getDevice());
                    input_model.cleanup(this);
                }
                throw;
            }
        }

        void initializeSprite() {
            if (!ensureRenderResources()) {
                throw std::runtime_error("MXVK failed to initialize render resources");
            }

            const auto [source_width, source_height] = source_dimensions();

            if (frame_sprite == nullptr) {
                frame_sprite = createSprite(source_width, source_height);
            }
            frame_sprite->enableExtendedUBO();
            frame_sprite->setCustomUniforms(custom_uniform_values);
            if (spectrumTextureEnabledForShaders()) {
                frame_sprite->enableSpectrumTexture(spectrumBinCount());
            }
            if (spectrumHistoryEnabledForShaders()) {
                frame_sprite->enableSpectrumHistoryTexture(
                    spectrumBinCount(),
                    static_cast<std::uint32_t>(options.audio_buffers));
            }
            if (historyCacheEnabled()) {
                frame_sprite->enableHistoryTexture(source_width, source_height,
                                                   static_cast<uint32_t>(
                                                       options.texture_cache_size));
            }
            frame_sprite->createEmptySprite(
                source_width, source_height, spriteVertexShader(),
                options.history_test ? echoCacheShader() : std::string{});

            if (options.human_background &&
                human_overlay_sprite == nullptr) {
                human_overlay_sprite = createSprite(1, 1);
                human_overlay_sprite->enableHistoryTexture(
                    static_cast<std::uint32_t>(source_width),
                    static_cast<std::uint32_t>(source_height), 1U);
                const cv::Mat transparent(source_height, source_width,
                                          CV_8UC4, cv::Scalar::all(0));
                human_overlay_sprite->updateHistoryTexture(
                    transparent.ptr(), transparent.cols, transparent.rows,
                    static_cast<int>(transparent.step));
            }

            initializeModel();

            if (source_kind == SourceKind::Graphic) {
                initial_frame_pending = false;
                uploadInputFrame(graphic_rgba);
                updateHumanOverlayTexture();
                initializeHistory(graphic_rgba);
            } else if (!readTrackedInputFrame()) {
                std::cerr << "acmxvk: capture did not provide an initial frame\n";
            } else {
                initial_frame_pending = true;
            }

            applyShaderPipeline();
            if (!currentShader().empty()) {
                std::cout << "acmxvk: " << activeShaderRole() << ' '
                          << (shader_index + 1) << '/' << shaders.size()
                          << ": " << currentShader() << '\n';
            }
        }

        void resetShaderTime() {
            previous_frame = std::chrono::steady_clock::now();
            previous_video_shader_timeline = 0.0;
            video_shader_timeline_initialized = false;
            shader_time = 0.0;
            frame_count = 0;
        }

        void beginCrossfade() {
            if (options.cross_fade_duration <= 0.0 || frame_count == 0 ||
                getDevice() == VK_NULL_HANDLE) {
                crossfade_active = false;
                crossfade_alpha = 1.0F;
                crossfade_uses_video_timeline = false;
                return;
            }

            try {
                std::vector<std::uint8_t> captured;
                std::uint32_t captured_width = 0;
                std::uint32_t captured_height = 0;
                captureSnapshotPixels(captured, captured_width,
                                      captured_height);
                const VkExtent2D extent = getRenderExtent();
                if (captured.empty() || captured_width == 0U ||
                    captured_height == 0U || extent.width == 0U ||
                    extent.height == 0U) {
                    throw std::runtime_error(
                        "the previous rendered frame is unavailable");
                }

                cv::Mat captured_rgba(static_cast<int>(captured_height),
                                      static_cast<int>(captured_width),
                                      CV_8UC4, captured.data());
                cv::Mat previous_rgba;
                if (captured_width == extent.width &&
                    captured_height == extent.height) {
                    previous_rgba = captured_rgba;
                } else {
                    const double captured_aspect =
                        static_cast<double>(captured_width) / captured_height;
                    const double target_aspect =
                        static_cast<double>(extent.width) / extent.height;
                    cv::Rect crop(0, 0, static_cast<int>(captured_width),
                                  static_cast<int>(captured_height));
                    if (captured_aspect > target_aspect) {
                        crop.width = std::max(
                            1, static_cast<int>(std::lround(
                                   captured_height * target_aspect)));
                        crop.x =
                            (static_cast<int>(captured_width) - crop.width) / 2;
                    } else if (captured_aspect < target_aspect) {
                        crop.height = std::max(
                            1, static_cast<int>(std::lround(
                                   captured_width / target_aspect)));
                        crop.y = (static_cast<int>(captured_height) -
                                  crop.height) /
                                 2;
                    }
                    cv::resize(captured_rgba(crop), previous_rgba,
                               cv::Size(static_cast<int>(extent.width),
                                        static_cast<int>(extent.height)),
                               0.0, 0.0, cv::INTER_LINEAR);
                }

                if (crossfade_previous_sprite == nullptr) {
                    crossfade_previous_sprite = createSprite(1, 1);
                }
                crossfade_previous_sprite->enableHistoryTexture(
                    extent.width, extent.height, 1U);
                crossfade_previous_sprite->updateHistoryTexture(
                    previous_rgba.ptr(), static_cast<int>(extent.width),
                    static_cast<int>(extent.height),
                    static_cast<int>(previous_rgba.step));
                crossfade_alpha = 0.0F;
                crossfade_active = true;
                crossfade_start_time = std::chrono::steady_clock::now();
                crossfade_uses_video_timeline = currentVideoTimeline(
                    crossfade_start_video_timeline);
            } catch (const std::exception &error) {
                crossfade_active = false;
                crossfade_alpha = 1.0F;
                crossfade_uses_video_timeline = false;
                std::cerr << "acmxvk: crossfade snapshot unavailable: "
                          << error.what() << "; switching immediately\n";
            }
        }

        void updateCrossfade(const std::chrono::steady_clock::time_point now) {
            if (!crossfade_active) {
                return;
            }
            double elapsed = 0.0;
            double video_timeline = 0.0;
            if (crossfade_uses_video_timeline &&
                currentVideoTimeline(video_timeline)) {
                if (video_timeline < crossfade_start_video_timeline) {
                    crossfade_start_video_timeline = video_timeline;
                }
                elapsed = video_timeline - crossfade_start_video_timeline;
            } else {
                elapsed = std::chrono::duration<double>(
                              now - crossfade_start_time)
                              .count();
            }
            crossfade_alpha = static_cast<float>(std::clamp(
                elapsed / options.cross_fade_duration, 0.0, 1.0));
            if (crossfade_alpha >= 1.0F) {
                crossfade_active = false;
                crossfade_uses_video_timeline = false;
                applyShaderPipeline();
            }
        }

        void cycleCrossfade(int direction) {
            const auto count =
                static_cast<std::ptrdiff_t>(CROSSFADE_NAMES.size());
            auto index =
                static_cast<std::ptrdiff_t>(crossfade_shader_index) + direction;
            index = (index % count + count) % count;
            crossfade_shader_index = static_cast<std::size_t>(index);
            std::cout << "acmxvk: crossfade shader: "
                      << CROSSFADE_NAMES[crossfade_shader_index] << " ("
                      << (crossfade_shader_index + 1) << '/'
                      << CROSSFADE_NAMES.size() << ")\n";
        }

        void adjustModelScale(float amount) {
            if (!model_initialized || model_scale_oscillation_active) {
                return;
            }
            model_scale = std::clamp(model_scale + amount, 0.05F, 20.0F);
            std::cout << "acmxvk: model scale " << model_scale << '\n';
        }

        void maybeRandomizeCrossfade() {
            if (!autopilot_random_crossfade || CROSSFADE_NAMES.empty()) {
                return;
            }
            std::uniform_int_distribution<std::size_t> distribution(
                0, CROSSFADE_NAMES.size() - 1);
            std::size_t next = distribution(autopilot_rng);
            if (CROSSFADE_NAMES.size() > 1 &&
                next == crossfade_shader_index) {
                next = (next + 1) % CROSSFADE_NAMES.size();
            }
            crossfade_shader_index = next;
        }

        void togglePause() {
            if (source_kind == SourceKind::Camera) {
                std::cout << "acmxvk: pause is available for video and graphic input\n";
                return;
            }
            input_paused = !input_paused;
            setSourcePlaybackClockPaused(input_paused || rendering_frozen);
            std::cout << "acmxvk: input pause "
                      << (input_paused ? "enabled" : "disabled") << '\n';
        }

        void toggleFreeze() {
            if (source_kind == SourceKind::Camera) {
                std::cout << "acmxvk: freeze is available for video and graphic input\n";
                return;
            }
            rendering_frozen = !rendering_frozen;
            setSourcePlaybackClockPaused(input_paused || rendering_frozen);
            previous_frame = std::chrono::steady_clock::now();
            std::cout << "acmxvk: rendering freeze "
                      << (rendering_frozen ? "enabled" : "disabled") << '\n';
        }

        void stepShaderTime(double amount) {
            shader_time += amount;
            std::cout << "acmxvk: shader time stepped to " << shader_time << '\n';
        }

        void adjustTimeSpeed(double amount) {
            options.time_speed += amount;
            if (std::abs(options.time_speed) < 0.01) {
                options.time_speed = 0.0;
            }
            std::cout << "acmxvk: shader time speed " << options.time_speed << '\n';
        }

        void toggleFullscreen() {
            SDL_Window *window = getSDLWindow();
            if (window == nullptr) {
                return;
            }
            const bool fullscreen =
                (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
            if (!SDL_SetWindowFullscreen(window, !fullscreen)) {
                std::cerr << "acmxvk: unable to toggle fullscreen: "
                          << SDL_GetError() << '\n';
                return;
            }
            std::cout << "acmxvk: fullscreen "
                      << (!fullscreen ? "enabled" : "disabled") << '\n';
        }

        void resetAutopilotInterval() {
            if (options.autopilot_random_timeout > 0) {
                std::uniform_int_distribution<int> distribution(
                    4, std::max(4, options.autopilot_random_timeout));
                autopilot_interval_frames = distribution(autopilot_rng);
            } else {
                autopilot_interval_frames = options.autopilot_frames;
            }
        }

        void logSelectedPlaylistNode(std::string_view action) const {
            if (playlist.empty()) {
                return;
            }
            std::cout << "acmxvk: " << action << " playlist node "
                      << (playlist_index + 1) << '/' << playlist.size() << ": "
                      << playlist[playlist_index].name << " ("
                      << playlist[playlist_index].shaders.size()
                      << " passes)\n";
        }

        [[nodiscard]] std::uint64_t autopilotFrameAdvance() {
            double video_timeline = 0.0;
            std::uint64_t video_frame_index = 0U;
            if (!currentVideoTimeline(video_timeline, &video_frame_index)) {
                autopilot_video_timeline_initialized = false;
                return 1U;
            }

            if (!autopilot_video_timeline_initialized ||
                video_frame_index < previous_autopilot_video_frame) {
                previous_autopilot_video_frame = video_frame_index;
                autopilot_video_timeline_initialized = true;
                return 1U;
            }

            const std::uint64_t advance =
                video_frame_index - previous_autopilot_video_frame;
            previous_autopilot_video_frame = video_frame_index;
            return advance;
        }

        void toggleAutopilot(bool sequential) {
            if (!playlist_enabled) {
                std::cout << "acmxvk: "
                          << (sequential ? "sequential autopilot" : "autopilot")
                          << " requires playlist mode (press P first)\n";
                return;
            }
            if (playlist.empty()) {
                std::cout << "acmxvk: autopilot has no playlist entries\n";
                return;
            }

            if (autopilot_enabled && autopilot_sequential == sequential) {
                autopilot_enabled = false;
                autopilot_sequential = false;
                std::cout << "acmxvk: autopilot disabled\n";
                return;
            }

            autopilot_enabled = true;
            autopilot_sequential = sequential;
            autopilot_counter = 0;
            autopilot_video_timeline_initialized = false;
            if (options.autopilot_random_timeout <= 0 && options.autopilot_frames <= 0) {
                options.autopilot_frames = 300;
            }
            resetAutopilotInterval();
            std::cout << "acmxvk: " << (sequential ? "sequential " : "random ")
                      << "autopilot enabled (";
            if (options.autopilot_random_timeout > 0) {
                std::cout << "random interval 4-" << options.autopilot_random_timeout
                          << ", current " << autopilot_interval_frames;
            } else {
                std::cout << "every " << autopilot_interval_frames << " frames";
            }
            std::cout << ")\n";
        }

        void updateAutopilot() {
            const std::uint64_t frame_advance = autopilotFrameAdvance();
            if (shader_locked || !autopilot_enabled || !playlist_enabled ||
                playlist.empty() || autopilot_interval_frames <= 0) {
                return;
            }
            const std::uint64_t remaining = static_cast<std::uint64_t>(
                std::max(0, autopilot_interval_frames - autopilot_counter));
            if (frame_advance < remaining) {
                autopilot_counter += static_cast<int>(frame_advance);
                return;
            }
            autopilot_counter = 0;

            maybeRandomizeCrossfade();
            beginCrossfade();
            if (autopilot_sequential && options.autopilot_random_timeout <= 0) {
                playlist_index = (playlist_index + 1) % playlist.size();
            } else {
                std::uniform_int_distribution<std::size_t> distribution(0,
                                                                        playlist.size() - 1);
                std::size_t next = distribution(autopilot_rng);
                if (playlist.size() > 1 && next == playlist_index) {
                    next = (next + 1) % playlist.size();
                }
                playlist_index = next;
            }

            applyShaderPipeline();
            resetShaderTime();
            if (options.autopilot_random_timeout > 0) {
                resetAutopilotInterval();
            }
            logSelectedPlaylistNode("autopilot selected");
        }

        void selectShader(int direction) {
            if (shader_locked || shaders.size() < 2 || frame_sprite == nullptr) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(shaders.size());
            beginCrossfade();
            auto index = static_cast<std::ptrdiff_t>(shader_index) + direction;
            index = (index % count + count) % count;
            shader_index = static_cast<std::size_t>(index);

            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            std::cout << "acmxvk: " << activeShaderRole() << ' '
                      << (shader_index + 1) << '/' << shaders.size() << ": "
                      << currentShader() << '\n';
        }

        void selectPlaylistNode(int direction) {
            if (shader_locked || playlist.empty()) {
                return;
            }
            const auto count = static_cast<std::ptrdiff_t>(playlist.size());
            beginCrossfade();
            auto index = static_cast<std::ptrdiff_t>(playlist_index) + direction;
            index = (index % count + count) % count;
            playlist_index = static_cast<std::size_t>(index);
            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            logSelectedPlaylistNode("selected");
        }

        [[nodiscard]] std::vector<fs::path> activeShaderPipeline() const {
            std::vector<fs::path> pipeline;
            if (effects_enabled) {
                if (playlist_enabled && !playlist.empty()) {
                    pipeline = playlist[playlist_index].shaders;
                } else if (multipass_enabled) {
                    pipeline = configured_passes;
                }
                if (!currentShader().empty()) {
                    pipeline.emplace_back(currentShader());
                }
            }
            if (options.flip_output) {
                pipeline.emplace_back(flipShader());
            }
            if (crossfade_active) {
                pipeline.emplace_back(crossfadeShader());
            }
            if (pipeline.empty()) {
                pipeline.emplace_back(passthroughShader());
            }
            if (options.human_background) {
                pipeline.emplace_back(humanCompositeShader());
            }
            return pipeline;
        }

        [[nodiscard]] fs::path directModelFragmentShader() const {
            if (!model_3d_active || !model_initialized || !effects_enabled ||
                playlist_enabled || multipass_enabled ||
                currentShader().empty()) {
                return {};
            }

            const fs::path shader(currentShader());
            const mxvk::ShaderModuleInfo module_info =
                mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
            if (module_info.stage != mxvk::ShaderStage::Fragment ||
                module_info.usesHistoryTexture ||
                module_info.usesSpectrumTexture ||
                module_info.usesSpectrumHistoryTexture) {
                return {};
            }
            return shader;
        }

        void applyShaderPipeline() {
            if (getDevice() == VK_NULL_HANDLE) {
                return;
            }
            vkDeviceWaitIdle(getDevice());
            detachPostProcessingShader();
            post_process_sprites.clear();
            frame_sprite->setEffectsEnabled(effects_enabled);

            const fs::path direct_model_shader = directModelFragmentShader();
            model_texture_prepass_active =
                model_3d_active && direct_model_shader.empty();
            setPostProcessingTextureConsumerEnabled(
                model_texture_prepass_active);
            if (model_initialized) {
                const fs::path desired_model_shader =
                    direct_model_shader.empty() ? modelFragmentShader()
                                                : direct_model_shader;
                if (desired_model_shader != model_effect_shader) {
                    input_model.setShaders(this, modelVertexShader().string(),
                                           desired_model_shader.string());
                    model_effect_shader = desired_model_shader;
                }
            }

            std::vector<fs::path> pipeline = activeShaderPipeline();
            if (!direct_model_shader.empty()) {
                const auto selected = std::find(
                    pipeline.begin(), pipeline.end(), direct_model_shader);
                if (selected != pipeline.end()) {
                    pipeline.erase(selected);
                }
                if (pipeline.empty()) {
                    pipeline.emplace_back(passthroughShader());
                }
                std::cout << "acmxvk: 3D texture effect: "
                          << direct_model_shader.filename().string()
                          << " [fragment, evaluated on model UVs]\n";
            } else if (model_3d_active && effects_enabled &&
                       !currentShader().empty()) {
                std::cout << "acmxvk: 3D texture prepass: fragment/compute "
                             "chain output mapped onto model UVs\n";
            }
            if (pipeline.empty()) {
                return;
            }

            std::vector<PostProcessingEffect> effects;
            effects.reserve(pipeline.size());
            crossfade_post_process_index =
                std::numeric_limits<std::size_t>::max();
            for (std::size_t index = 0; index < pipeline.size(); ++index) {
                const fs::path &shader = pipeline[index];
                PostProcessingEffect effect{
                    shader.string(), {1.0F, 1.0F, 1.0F, 0.0F}, false};
                if (crossfade_active && shader == crossfadeShader()) {
                    crossfade_post_process_index = index;
                    effect.historySource = crossfade_previous_sprite;
                    effect.params[0] = crossfade_alpha;
                } else if (options.human_background &&
                           shader == humanCompositeShader()) {
                    effect.historySource = human_overlay_sprite;
                } else if (historyCacheEnabled()) {
                    effect.historySource = frame_sprite;
                }
                if (spectrumTextureEnabledForShaders()) {
                    effect.spectrumBinCount = spectrumBinCount();
                }
                if (spectrumHistoryEnabledForShaders()) {
                    effect.spectrumHistoryLayerCount =
                        static_cast<std::uint32_t>(options.audio_buffers);
                }
                effects.push_back(effect);
            }
            post_process_sprites = attachPostProcessingShaders(effects);
            for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                sprite->enableExtendedUBO();
                sprite->setCustomUniforms(custom_uniform_values);
                if (spectrumTextureEnabledForShaders()) {
                    sprite->enableSpectrumTexture(spectrumBinCount());
                }
                if (spectrumHistoryEnabledForShaders()) {
                    sprite->enableSpectrumHistoryTexture(
                        spectrumBinCount(),
                        static_cast<std::uint32_t>(options.audio_buffers));
                }
            }

            std::cout << "acmxvk: Vulkan shader pipeline (" << pipeline.size() << " passes):\n";
            for (std::size_t index = 0; index < pipeline.size(); ++index) {
                const bool compute =
                    index < post_process_effect_stages.size() &&
                    post_process_effect_stages[index] ==
                        mxvk::ShaderStage::Compute;
                std::cout << "  " << (index + 1) << ": "
                          << pipeline[index].filename().string() << " ["
                          << (compute ? "compute" : "fragment") << "]\n";
            }
        }

        [[nodiscard]] bool readTrackedInputFrame() {
            if (!readInputFrame()) {
                return false;
            }
            if (source_kind == SourceKind::Video) {
                ++decoded_video_frame_count;
                ++video_source_frame_count;
            } else if (source_kind == SourceKind::Camera) {
                updateCameraFrameRate();
            }
            return true;
        }

        [[nodiscard]] bool skipInputFrame() {
            if (source_kind != SourceKind::Video) {
                return false;
            }
            bool skipped = false;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (using_ffmpeg_capture) {
                skipped = ffmpeg_capture.skip();
            } else
#endif
            {
                skipped = capture.grab();
            }
            if (skipped) {
                ++decoded_video_frame_count;
                ++video_source_frame_count;
            }
            return skipped;
        }

        [[nodiscard]] bool handleCaptureEnd(bool discard = false) {
            if (source_kind == SourceKind::Camera) {
                return true;
            }
            if (!options.repeat) {
                setFrameReadbackEnabled(false);
                exit();
                return false;
            }

#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (using_ffmpeg_capture && ffmpeg_capture.seek_start()) {
                video_source_frame_count = 0;
                const bool restarted =
                    discard ? skipInputFrame() : readTrackedInputFrame();
                if (restarted) {
                    if (!ffmpeg_seek_repeat_logged) {
                        std::cout
                            << "acmxvk: video repeat: in-place FFmpeg seek; "
                            << (ffmpeg_capture.using_hardware_decode()
                                    ? "NVDEC decoder and CUDA device preserved\n"
                                    : "software decoder preserved\n");
                        ffmpeg_seek_repeat_logged = true;
                    }
                    return true;
                }
                std::cerr << "acmxvk: in-place FFmpeg repeat did not produce a "
                             "frame; reopening the input\n";
            }
#endif
            closeVideoCapture();
            if (!openVideoCapture() ||
                !(discard ? skipInputFrame() : readTrackedInputFrame())) {
                throw std::runtime_error("unable to restart video input: " + options.input_file);
            }
            return true;
        }

        [[nodiscard]] bool readClockedVideoFrame(double clock_seconds) {
            const double rate = outputFrameRate();
            if (!std::isfinite(rate) || rate <= 0.0) {
                return readTrackedInputFrame();
            }

            std::uint64_t target_frame = static_cast<std::uint64_t>(
                std::floor(std::max(clock_seconds, 0.0) * rate));
            if (target_frame < decoded_video_frame_count) {
                const double next_frame_time =
                    static_cast<double>(decoded_video_frame_count) / rate;
                const double wait_seconds = next_frame_time - clock_seconds;
                if (wait_seconds > 0.0) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(wait_seconds));
                }
                double updated_clock = 0.0;
                if (mediaClockSeconds(updated_clock)) {
                    target_frame = static_cast<std::uint64_t>(
                        std::floor(std::max(updated_clock, 0.0) * rate));
                }
            }
            if (target_frame < decoded_video_frame_count) {
                return true;
            }

            const std::uint64_t frames_to_advance =
                target_frame - decoded_video_frame_count + 1;
            for (std::uint64_t frame = 0; frame < frames_to_advance; ++frame) {
                const bool discard = frame + 1 < frames_to_advance;
                bool advanced = discard ? skipInputFrame()
                                        : readTrackedInputFrame();
                if (!advanced) {
                    advanced = handleCaptureEnd(discard);
                }
                if (!advanced) {
                    return false;
                }
            }

            source_frame_received = true;
            recording_frame_due = true;
            recording_frame_has_pts = true;
            recording_frame_pts = decoded_video_frame_count - 1;
            if (!media_clock_sync_logged) {
                std::cout << "acmxvk: media-clock synchronization active; "
                             "late video frames will be skipped and encoded "
                             "with timeline PTS\n";
                media_clock_sync_logged = true;
            }
            return true;
        }

        void closeVideoCapture() {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.is_open()) {
                ffmpeg_capture.close();
            }
            using_ffmpeg_capture = false;
#endif
            if (capture.is_open()) {
                capture.close();
            }
        }

        [[nodiscard]] bool openVideoCapture() {
            video_source_frame_count = 0;
            video_source_fps = 0.0;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.open(options.input_file, options.cuda_device)) {
                using_ffmpeg_capture = true;
                video_source_fps = ffmpeg_capture.fps();
                std::cout << "acmxvk: video capture: FFmpeg ";
                if (ffmpeg_capture.using_hardware_decode()) {
                    std::cout << "with CUDA/NVDEC";
                    if (ffmpeg_capture.hardware_decode_device() >= 0) {
                        std::cout << " on device "
                                  << ffmpeg_capture.hardware_decode_device();
                    }
                    std::cout << '\n';
                } else {
                    std::cout << "software decode\n";
                }
                return true;
            }
#endif
            const bool opened = capture.open(options.input_file);
            if (opened) {
                video_source_fps = capture.get(cv::CAP_PROP_FPS);
                std::cout << "acmxvk: video capture: OpenCV fallback\n";
            }
            if (!std::isfinite(video_source_fps) || video_source_fps <= 0.0) {
                video_source_fps = 30.0;
            }
            return opened;
        }

        [[nodiscard]] bool readHostRgba(cv::Mat &rgba) {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (using_ffmpeg_capture) {
                int width = 0;
                int height = 0;
                int pitch = 0;
                if (!ffmpeg_capture.readRgba(ffmpeg_rgba, width, height, pitch,
                                             false) ||
                    ffmpeg_rgba.empty() || width <= 0 || height <= 0 ||
                    pitch < width * 4) {
                    return false;
                }
                rgba = cv::Mat(height, width, CV_8UC4, ffmpeg_rgba.data(),
                               static_cast<std::size_t>(pitch));
                return true;
            }
#endif
            return capture.readRgba(rgba, false);
        }

        void initializeHistory(const cv::Mat &rgba) {
            if (!historyCacheEnabled() || history_initialized) {
                return;
            }
            for (uint32_t layer = 0; layer < frame_sprite->getHistoryLayerCount(); ++layer) {
                updateHistoryFrame(rgba);
            }
            history_initialized = true;
            history_delay_counter = 0;
            camera_history_clock_started = false;
            std::cout << "acmxvk: initialized " << frame_sprite->getHistoryLayerCount()
                      << " Vulkan history-cache layers (delay " << options.cache_delay
                      << ")\n";
        }

        void updateHistoryFrame(const cv::Mat &rgba) {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                updateFilteredCudaHistoryFrame();
                return;
            }
#endif
            frame_sprite->updateHistoryTexture(
                rgba.ptr(), rgba.cols, rgba.rows,
                static_cast<int>(rgba.step));
        }

        void updateCameraHistory() {
            if (source_kind != SourceKind::Camera || rendering_frozen ||
                input_paused || !history_initialized) {
                return;
            }

            const double rate = outputFrameRate();
            if (!std::isfinite(rate) || rate <= 0.0) {
                return;
            }
            const auto interval = std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(std::chrono::duration<double>(
                static_cast<double>(options.cache_delay + 1) / rate));
            const auto now = std::chrono::steady_clock::now();
            if (!camera_history_clock_started) {
                camera_history_next_update = now + interval;
                camera_history_clock_started = true;
                return;
            }
            if (now < camera_history_next_update) {
                return;
            }

            bool history_updated = false;
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                updateFilteredCudaHistoryFrame();
                history_updated = true;
            }
#endif
            if (!history_updated && !latest_camera_history_rgba.empty()) {
                updateHistoryFrame(latest_camera_history_rgba);
                history_updated = true;
            }
            if (!history_updated) {
                return;
            }

            camera_history_next_update += interval;
            if (camera_history_next_update <= now) {
                camera_history_next_update = now + interval;
            }
        }

#ifdef ACMXVK_WITH_MXVK_CUDA
        void updateModelTextureCuda(const cv::cuda::GpuMat &rgba,
                                    cv::cuda::Stream &source_stream) {
            if (!model_initialized) {
                return;
            }
            if (input_model.updatePrimaryTextureCuda(rgba, source_stream)) {
                return;
            }

            rgba.download(cuda_model_fallback_rgba, source_stream);
            source_stream.waitForCompletion();
            if (!cuda_model_fallback_logged) {
                std::cerr << "acmxvk: direct CUDA model-texture upload "
                             "unavailable; using host staging\n";
                cuda_model_fallback_logged = true;
            }
            if (!input_model.updatePrimaryTexture(
                    cuda_model_fallback_rgba.ptr(),
                    cuda_model_fallback_rgba.cols,
                    cuda_model_fallback_rgba.rows,
                    static_cast<int>(cuda_model_fallback_rgba.step))) {
                throw std::runtime_error(
                    "MXVK could not update the 3D model texture");
            }
        }

        void updateCudaHistoryFrame(const cv::cuda::GpuMat &rgba,
                                    cv::cuda::Stream &source_stream) {
            if (frame_sprite->updateHistoryTextureCuda(rgba, source_stream)) {
                return;
            }

            rgba.download(cuda_history_fallback_rgba, source_stream);
            source_stream.waitForCompletion();
            if (!cuda_history_fallback_logged) {
                std::cerr << "acmxvk: direct CUDA history upload unavailable; "
                             "using a host-staging fallback\n";
                cuda_history_fallback_logged = true;
            }
            frame_sprite->updateHistoryTexture(
                cuda_history_fallback_rgba.ptr(),
                cuda_history_fallback_rgba.cols,
                cuda_history_fallback_rgba.rows,
                static_cast<int>(cuda_history_fallback_rgba.step));
        }

#ifdef ACMXVK_WITH_CUDA
        void updateFilteredCudaHistoryFrame() {
            updateCudaHistoryFrame(gpu_filter_engine->output(),
                                   gpu_filter_engine->stream());
        }
#endif

        void initializeCudaHistory(const cv::cuda::GpuMat &rgba,
                                   cv::cuda::Stream &source_stream,
                                   bool filtered) {
            if (!historyCacheEnabled() || history_initialized) {
                return;
            }
            for (uint32_t layer = 0;
                 layer < frame_sprite->getHistoryLayerCount(); ++layer) {
                updateCudaHistoryFrame(rgba, source_stream);
            }
            history_initialized = true;
            history_delay_counter = 0;
            camera_history_clock_started = false;
            std::cout << "acmxvk: initialized "
                      << frame_sprite->getHistoryLayerCount()
                      << (filtered ? " filtered" : " NVDEC")
                      << " Vulkan history-cache layers (delay "
                      << options.cache_delay << ")\n";
        }

#ifdef ACMXVK_WITH_CUDA
        void uploadInputFrame(const cv::cuda::GpuMat &rgba,
                              cv::cuda::Stream &source_stream) {
            if (!gpu_filter_engine->process(rgba, source_stream)) {
                throw std::runtime_error(
                    "acidcam-gpu rejected the CUDA RGBA input frame");
            }
            if (!frame_sprite->updateTextureCuda(
                    gpu_filter_engine->output(),
                    gpu_filter_engine->stream())) {
                throw std::runtime_error(
                    "MXVK could not upload the CUDA-filtered frame");
            }
            updateModelTextureCuda(gpu_filter_engine->output(),
                                   gpu_filter_engine->stream());
        }
#endif

        [[nodiscard]] const cv::cuda::GpuMat &
        rotateCudaFrame(const cv::cuda::GpuMat &rgba,
                        cv::cuda::Stream &source_stream) {
            switch (options.frame_rotation) {
            case FrameRotation::None:
                return rgba;
            case FrameRotation::Clockwise90:
                cv::cuda::transpose(rgba, cuda_rotation_transpose, source_stream);
                cv::cuda::flip(cuda_rotation_transpose, cuda_rotated_rgba, 1,
                               source_stream);
                break;
            case FrameRotation::Rotate180:
                cv::cuda::flip(rgba, cuda_rotated_rgba, -1, source_stream);
                break;
            case FrameRotation::Counterclockwise90:
                cv::cuda::transpose(rgba, cuda_rotation_transpose, source_stream);
                cv::cuda::flip(cuda_rotation_transpose, cuda_rotated_rgba, 0,
                               source_stream);
                break;
            }
            return cuda_rotated_rgba;
        }
#endif

        void uploadInputFrame(const cv::Mat &rgba) {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                if (!gpu_filter_engine->process(rgba)) {
                    throw std::runtime_error(
                        "acidcam-gpu rejected the RGBA input frame");
                }
                if (!frame_sprite->updateTextureCuda(
                        gpu_filter_engine->output(),
                        gpu_filter_engine->stream())) {
                    throw std::runtime_error(
                        "MXVK could not upload the CUDA-filtered frame");
                }
                updateModelTextureCuda(gpu_filter_engine->output(),
                                       gpu_filter_engine->stream());
                return;
            }
#endif
            frame_sprite->updateTexture(rgba.ptr(), rgba.cols, rgba.rows,
                                        static_cast<int>(rgba.step));
            if (model_initialized &&
                !input_model.updatePrimaryTexture(
                    rgba.ptr(), rgba.cols, rgba.rows,
                    static_cast<int>(rgba.step))) {
                throw std::runtime_error(
                    "MXVK could not update the 3D model texture");
            }
        }

        [[nodiscard]] bool readLatestCameraFrame() {
            cv::Mat bgr;
            const bool wait_for_first = !async_camera_frame_uploaded &&
                                        !async_camera_initial_wait_completed;
            async_camera_initial_wait_completed = true;
            if (!latest_camera_frame.takeLatest(bgr, wait_for_first)) {
                return false;
            }

            cv::Mat rgba;
            cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
            applyDnnEffects(rgba);
            rotateFrame(rgba, options.frame_rotation);
            if (!human_overlay_rgba.empty()) {
                rotateFrame(human_overlay_rgba, options.frame_rotation);
            }
            uploadInputFrame(rgba);
            updateHumanOverlayTexture();
            latest_camera_history_rgba = rgba;
            async_camera_frame_uploaded = true;

#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr) {
                initializeCudaHistory(gpu_filter_engine->output(),
                                      gpu_filter_engine->stream(), true);
                return true;
            }
#endif

            initializeHistory(rgba);
            return true;
        }

        [[nodiscard]] bool readInputFrame() {
            if (source_kind == SourceKind::Camera && options.maximize_fps) {
                return readLatestCameraFrame();
            }
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr && !dnnHostProcessingEnabled()) {
                cv::cuda::Stream *capture_stream = nullptr;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                if (using_ffmpeg_capture) {
                    if (!ffmpeg_capture.readGpuRgba(cuda_input_rgba,
                                                    ffmpeg_cuda_stream, false)) {
                        return false;
                    }
                    capture_stream = &ffmpeg_cuda_stream;
                } else
#endif
                {
                    if (!capture.readGpuRgba(cuda_input_rgba, false)) {
                        return false;
                    }
                    capture_stream = &capture.cudaStream();
                }
                const cv::cuda::GpuMat &filter_input =
                    rotateCudaFrame(cuda_input_rgba, *capture_stream);
                uploadInputFrame(filter_input, *capture_stream);
                if (!cuda_input_path_logged) {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                    if (using_ffmpeg_capture) {
                        std::cout << "acmxvk: CUDA input path active: FFmpeg "
                                  << (ffmpeg_capture.using_hardware_decode()
                                          ? "NVDEC -> CUDA RGBA -> "
                                          : "software decode -> CUDA upload -> ");
                    } else
#endif
                    {
                        std::cout
                            << "acmxvk: CUDA input path active: MXVK capture -> ";
                    }
                    if (options.frame_rotation != FrameRotation::None) {
                        std::cout << "CUDA rotation -> ";
                    }
                    std::cout
                        << "acidcam-gpu temporal buffer -> Vulkan texture\n";
                    cuda_input_path_logged = true;
                }
                const bool history_was_initialized = history_initialized;
                initializeCudaHistory(gpu_filter_engine->output(),
                                      gpu_filter_engine->stream(), true);
                if (source_kind != SourceKind::Camera &&
                    history_was_initialized &&
                    ++history_delay_counter > options.cache_delay) {
                    updateFilteredCudaHistoryFrame();
                    history_delay_counter = 0;
                }
                return true;
            }
#endif
#ifdef ACMXVK_WITH_MXVK_CUDA
#if defined(MXVK_WITH_FFMPEG_CAPTURE)
            if (using_ffmpeg_capture &&
                ffmpeg_capture.using_hardware_decode() &&
                !dnnHostProcessingEnabled()) {
                if (!ffmpeg_capture.readGpuRgba(cuda_input_rgba,
                                                ffmpeg_cuda_stream, false)) {
                    return false;
                }
                const cv::cuda::GpuMat &render_input =
                    rotateCudaFrame(cuda_input_rgba, ffmpeg_cuda_stream);
                if (!frame_sprite->updateTextureCuda(render_input,
                                                     ffmpeg_cuda_stream)) {
                    render_input.download(cuda_input_fallback_rgba,
                                          ffmpeg_cuda_stream);
                    ffmpeg_cuda_stream.waitForCompletion();
                    if (!cuda_input_fallback_logged) {
                        std::cerr
                            << "acmxvk: direct NVDEC/Vulkan upload unavailable; "
                               "using host staging\n";
                        cuda_input_fallback_logged = true;
                    }
                    frame_sprite->updateTexture(
                        cuda_input_fallback_rgba.ptr(),
                        cuda_input_fallback_rgba.cols,
                        cuda_input_fallback_rgba.rows,
                        static_cast<int>(cuda_input_fallback_rgba.step));
                }
                updateModelTextureCuda(render_input, ffmpeg_cuda_stream);
                if (!cuda_input_path_logged) {
                    std::cout << "acmxvk: CUDA input path active: FFmpeg "
                                 "NVDEC -> CUDA RGBA -> ";
                    if (options.frame_rotation != FrameRotation::None) {
                        std::cout << "CUDA rotation -> ";
                    }
                    std::cout << "Vulkan texture";
                    if (cuda_input_fallback_logged) {
                        std::cout << " (host-staging fallback)";
                    }
                    std::cout << '\n';
                    cuda_input_path_logged = true;
                }
                const bool history_was_initialized = history_initialized;
                initializeCudaHistory(render_input, ffmpeg_cuda_stream, false);
                if (history_was_initialized &&
                    ++history_delay_counter > options.cache_delay) {
                    updateCudaHistoryFrame(render_input, ffmpeg_cuda_stream);
                    history_delay_counter = 0;
                }
                return true;
            }
#endif
#endif

            bool requires_host_frame = dnnHostProcessingEnabled() ||
                                       historyCacheEnabled() ||
                                       options.frame_rotation != FrameRotation::None ||
                                       model_initialized;
            if (!requires_host_frame
#ifdef MXVK_WITH_FFMPEG_CAPTURE
                && !using_ffmpeg_capture
#endif
            ) {
                return capture.readToSprite(*frame_sprite, false);
            }

            cv::Mat rgba;
            if (!readHostRgba(rgba)) {
                return false;
            }
            applyDnnEffects(rgba);
            rotateFrame(rgba, options.frame_rotation);
            if (!human_overlay_rgba.empty()) {
                rotateFrame(human_overlay_rgba, options.frame_rotation);
            }
            uploadInputFrame(rgba);
            updateHumanOverlayTexture();
            if (source_kind == SourceKind::Camera) {
                latest_camera_history_rgba = rgba;
            }
            const bool history_was_initialized = history_initialized;
            initializeHistory(rgba);
            if (source_kind != SourceKind::Camera && history_was_initialized &&
                ++history_delay_counter > options.cache_delay) {
                updateHistoryFrame(rgba);
                history_delay_counter = 0;
            }
            return true;
        }

        void updateShaderUniforms(int width, int height) {
            const auto now = std::chrono::steady_clock::now();
            updateCrossfade(now);
            const float wall_delta =
                std::chrono::duration<float>(now - previous_frame).count();
            previous_frame = now;
            ++frame_count;

            double video_timeline = 0.0;
            const bool video_timeline_available =
                currentVideoTimeline(video_timeline);
            float delta = wall_delta;
            if (video_timeline_available) {
                if (!video_shader_clock_logged) {
                    std::cout
                        << "acmxvk: shader clock: decoded video timeline; "
                           "effects are independent of processing speed\n";
                    video_shader_clock_logged = true;
                }
                if (!video_shader_timeline_initialized ||
                    video_timeline < previous_video_shader_timeline) {
                    if (video_shader_timeline_initialized &&
                        video_timeline < previous_video_shader_timeline) {
                        shader_time = 0.0;
                        frame_count = 1;
                    }
                    delta = 0.0F;
                    video_shader_timeline_initialized = true;
                } else {
                    delta = static_cast<float>(
                        video_timeline - previous_video_shader_timeline);
                }
                previous_video_shader_timeline = video_timeline;
            } else if (options.normalized_time) {
                delta = static_cast<float>(1.0 / outputFrameRate());
            }
            const float frame_rate =
                video_timeline_available
                    ? static_cast<float>(video_source_fps)
                    : (delta > 0.0F ? 1.0F / delta : 0.0F);
            float raw_audio_amplitude = 0.0F;
            float audio_sensitivity = 1.0F;
            float audio_amplitude = 0.0F;
            float audio_frequency = 0.0F;
            float audio_peak = 0.0F;
            float audio_rms = 0.0F;
            float audio_smooth = 0.0F;
            float audio_low = 0.0F;
            float audio_mid = 0.0F;
            float audio_high = 0.0F;
            float audio_sample_rate = 44100.0F;
#ifdef AUDIO_ENABLED
            std::vector<float> spectrum_values;
            if (file_audio_source != nullptr && audio_engine != nullptr &&
                media_timeline_started &&
                (file_audio_source->has_output_clock() ||
                 source_frame_received)) {
                double source_audio_time = 0.0;
                if (options.use_source_audio &&
                    !file_audio_source->has_output_clock() &&
                    mediaClockSeconds(source_audio_time)) {
                    file_audio_source->process_at_time(
                        source_audio_time, outputFrameRate(), *audio_engine);
                } else {
                    file_audio_source->process_frame(outputFrameRate(),
                                                     *audio_engine);
                }
                if (options.audio_trunc && !file_audio_source->is_active()) {
                    std::cout << "acmxvk: audio source finished, stopping "
                                 "(--audio-trunc)\n";
                    exit();
                }
            }
            if (audioSourceOpen()) {
                const audio::AudioMetrics metrics = audio_engine->metrics();
                const float warmup = updateAudioWarmup(now);
                raw_audio_amplitude = metrics.amplitude;
                audio_sensitivity = audio_engine->sensitivity();
                const float delta_scale = audio_delta_time ? delta : 1.0F;
                const float sense = audio_sensitivity * 4.0F * warmup;
                audio_amplitude = raw_audio_amplitude * audio_sensitivity *
                                  static_cast<float>(options.time_speed) *
                                  delta_scale * warmup;
                audio_frequency = metrics.frequency;
                audio_peak = std::sqrt(std::max(metrics.peak, 0.0F)) * sense;
                audio_rms = std::sqrt(std::max(metrics.rms, 0.0F)) * sense;
                audio_smooth = std::sqrt(std::max(metrics.smooth, 0.0F)) * sense;
                audio_low = std::sqrt(std::max(metrics.low, 0.0F)) * sense;
                audio_mid = std::sqrt(std::max(metrics.mid, 0.0F)) * sense;
                audio_high = std::sqrt(std::max(metrics.high, 0.0F)) * sense;
                audio_sample_rate = static_cast<float>(audio_engine->sample_rate());
                spectrum_values = audio_engine->spectrum();
                const float spectrum_scale =
                    warmup *
                    (spectrum_scale_by_sensitivity ? audio_sensitivity : 1.0F);
                for (float &value : spectrum_values) {
                    value *= spectrum_scale;
                }
            }
#endif
            if (audio_time_active) {
                const float delta_scale = audio_delta_time ? delta : 1.0F;
                shader_time += static_cast<double>(raw_audio_amplitude) *
                               static_cast<double>(audio_sensitivity) *
                               options.time_speed *
                               static_cast<double>(delta_scale);
            } else if (shader_time_active) {
                shader_time += static_cast<double>(delta) * options.time_speed;
            }
            if (!std::isfinite(shader_time)) {
                shader_time = 0.0;
            }
            model_wave_audio_step =
                audio_amplitude * raw_audio_amplitude;
            if (video_timeline_available) {
                const std::uint64_t source_frame =
                    video_source_frame_count - 1U;
                if (source_frame <= 58U) {
                    legacy_alpha =
                        0.2F + 0.1F * static_cast<float>(source_frame);
                } else {
                    const std::uint64_t phase = (source_frame - 59U) % 100U;
                    legacy_alpha =
                        phase < 50U
                            ? 5.9F - 0.1F * static_cast<float>(phase)
                            : 1.1F +
                                  0.1F * static_cast<float>(phase - 50U);
                }
            } else if (legacy_alpha_increasing) {
                legacy_alpha += 0.1F;
                if (legacy_alpha >= 6.0F) {
                    legacy_alpha = 6.0F;
                    legacy_alpha_increasing = false;
                }
            } else {
                legacy_alpha -= 0.1F;
                if (legacy_alpha <= 1.0F) {
                    legacy_alpha = 1.0F;
                    legacy_alpha_increasing = true;
                }
            }
            const float elapsed = static_cast<float>(shader_time);
            const float compatibility_time = video_timeline_available
                                                 ? static_cast<float>(
                                                       video_timeline)
                                                 : std::chrono::duration<float>(
                                                       now - compatibility_clock_start)
                                                       .count();
            const float shader_frame =
                video_timeline_available
                    ? static_cast<float>(video_source_frame_count - 1U)
                    : static_cast<float>(frame_count);
            frame_sprite->setShaderParams(1.0F, 1.0F, 1.0F, elapsed);
            frame_sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
            frame_sprite->setUniform0(legacy_alpha, compatibility_time,
                                      static_cast<float>(width),
                                      static_cast<float>(height));
            frame_sprite->setUniform1(delta, audio_amplitude, audio_frequency,
                                      frame_rate);
            frame_sprite->setUniform2(shader_frame, elapsed,
                                      audio_sample_rate, audio_peak);
            frame_sprite->setUniform3(static_cast<float>(frame_sprite->getHistoryHead()),
                                      static_cast<float>(frame_sprite->getHistoryLayerCount()),
                                      audio_rms, audio_smooth);
            frame_sprite->setAudioBands(audio_low, audio_mid, audio_high);

            model_fragment_uniforms = {};
            model_fragment_uniforms.mouse = glm::vec4(
                mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F, 0.0F);
            model_fragment_uniforms.u0 =
                glm::vec4(legacy_alpha, compatibility_time,
                          static_cast<float>(width),
                          static_cast<float>(height));
            model_fragment_uniforms.u1 =
                glm::vec4(delta, audio_amplitude, audio_frequency,
                          frame_rate);
            model_fragment_uniforms.u2 = glm::vec4(
                shader_frame, elapsed, audio_sample_rate, audio_peak);
            model_fragment_uniforms.u3 = glm::vec4(
                static_cast<float>(frame_sprite->getHistoryHead()),
                static_cast<float>(frame_sprite->getHistoryLayerCount()),
                audio_rms, audio_smooth);
            for (std::size_t index = 0;
                 index < custom_uniform_values.size() && index < 64U;
                 ++index) {
                model_fragment_uniforms.custom_uniforms[index / 4U]
                                                       [index % 4U] =
                    custom_uniform_values[index];
            }
            model_fragment_uniforms.audio_bands =
                glm::vec4(audio_low, audio_mid, audio_high, 0.0F);

            for (std::size_t index = 0; index < post_process_sprites.size(); ++index) {
                mxvk::VK_Sprite *sprite = post_process_sprites[index];
                if (crossfade_active &&
                    index == crossfade_post_process_index) {
                    setPostProcessingShaderParams(index, crossfade_alpha, 0.0F,
                                                  0.0F, 0.0F);
                } else {
                    setPostProcessingShaderParams(index, 1.0F, 1.0F, 1.0F,
                                                  elapsed);
                }
                sprite->setMouseState(mouse_x, mouse_y, mouse_pressed ? 1.0F : 0.0F);
                sprite->setUniform0(legacy_alpha, compatibility_time,
                                    static_cast<float>(width),
                                    static_cast<float>(height));
                sprite->setUniform1(delta, audio_amplitude, audio_frequency,
                                    frame_rate);
                sprite->setUniform2(shader_frame, elapsed,
                                    audio_sample_rate, audio_peak);
                sprite->setUniform3(
                    static_cast<float>(frame_sprite->getHistoryHead()),
                    static_cast<float>(frame_sprite->getHistoryLayerCount()),
                    audio_rms, audio_smooth);
                sprite->setAudioBands(audio_low, audio_mid, audio_high);
            }
#ifdef AUDIO_ENABLED
            if (!spectrum_values.empty()) {
                frame_sprite->updateSpectrumTexture(
                    spectrum_values.data(),
                    static_cast<std::uint32_t>(spectrum_values.size()));
                if (options.audio_buffers > 0) {
                    frame_sprite->updateSpectrumHistoryTexture(
                        spectrum_values.data(),
                        static_cast<std::uint32_t>(spectrum_values.size()));
                }
                for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                    sprite->updateSpectrumTexture(
                        spectrum_values.data(),
                        static_cast<std::uint32_t>(spectrum_values.size()));
                    if (options.audio_buffers > 0) {
                        sprite->updateSpectrumHistoryTexture(
                            spectrum_values.data(),
                            static_cast<std::uint32_t>(spectrum_values.size()));
                    }
                }
            }
#endif
        }
    };
