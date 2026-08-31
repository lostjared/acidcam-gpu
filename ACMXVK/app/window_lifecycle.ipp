    // Window construction, event handling, rendering callbacks, and main loop.
    class MainWindow final : public mxvk::VK_Window {
      public:
        explicit MainWindow(Options options)
            : mxvk::VK_Window("ACMXVK", options.width, options.height,
                              options.fullscreen, MXVK_VALIDATION, options.enable_vsync),
              options(std::move(options)) {
            setClearColor(0.0F, 0.0F, 0.0F, 1.0F);
            setEnableScreenshot(this->options.enable_screenshot);
            resolveConfiguredResourcePaths();
            initializeDnn();
            initializeGpuFilters();
            openAudio();
            loadShaders();
            loadShaderPasses();
            initialize_interface_control();
            configureMidiMappings();
            openMidi();
            loadPlaylist();
            resetAutopilotInterval();
            openInput();
            configureRenderResolution();
            initializeSprite();
            initializeOverlayFont();
            start_requested_audio_recording();
            openOutput();
            updateWindowTitle(true);
        }

        ~MainWindow() override {
            interface_client.close();
            latest_camera_frame.stop();
            try {
                flushFrameReadbacks();
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: unable to flush pending frame readbacks: "
                          << error.what() << '\n';
            }
            if (model_initialized && getDevice() != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(getDevice());
                input_model.cleanup(this);
                model_initialized = false;
                std::cout << "acmxvk: released 3D model resources\n";
            }
            const bool should_copy_audio =
                options.copy_audio && !options.mute_output && writer.is_open();
#ifdef AUDIO_ENABLED
            const bool should_mux_file_audio =
                file_audio_source != nullptr && writer.is_open() &&
                !options.output_file.empty() && !options.png_output &&
                !options.mute_output && output_frame_count > 0;
            const bool should_mux_live_audio =
                audio_engine != nullptr && file_audio_source == nullptr &&
                audio_engine->is_recording() && writer.is_open() &&
                !options.output_file.empty() && !options.png_output &&
                !options.copy_audio && !options.mute_output &&
                output_frame_count > 0;
            const bool should_write_live_audio =
                audio_engine != nullptr && audio_engine->is_recording() &&
                !options.record_audio_file.empty();
            audio::AudioRecording live_audio_recording;
            if (audio_engine != nullptr && audio_engine->is_recording()) {
                live_audio_recording = audio_engine->stop_recording();
            }
            if (file_audio_source != nullptr) {
                file_audio_source->stop_output();
            }
#endif
            if (writer.is_open()) {
                writer.close();
                std::cout << "acmxvk: recording closed after " << output_frame_count
                          << " frames\n";
            }
            if (options.png_output) {
                std::cout << "acmxvk: PNG sequence closed after " << png_frame_count
                          << " frames\n";
            }
            if (options.generate_interval > 0) {
                std::cout << "acmxvk: generated " << generated_frame_count
                          << " periodic PNG frames\n";
            }
            if (capture.is_open()) {
                capture.close();
            }
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            if (ffmpeg_capture.is_open()) {
                ffmpeg_capture.close();
            }
#endif
            if (should_copy_audio) {
                transfer_audio(options.input_file, options.output_file);
                std::cout << "acmxvk: copied audio track from " << options.input_file
                          << " to " << options.output_file << '\n';
            }
#ifdef AUDIO_ENABLED
            if (should_mux_file_audio) {
                const double video_duration = writer.get_duration();
                if (!file_audio_source->mux_into_video(options.output_file,
                                                       video_duration)) {
                    std::cerr << "acmxvk: file-audio mux failed; preserving the "
                                 "encoded video without audio\n";
                }
            }
            if (should_write_live_audio) {
                if (live_audio_recording.empty()) {
                    std::cerr << "acmxvk: standalone audio recording was empty; "
                                 "no WAV file was written\n";
                } else if (!audio::write_wav_file(live_audio_recording,
                                                  options.record_audio_file)) {
                    std::cerr << "acmxvk: could not write WAV recording: "
                              << options.record_audio_file << '\n';
                } else {
                    std::cout << "acmxvk: wrote "
                              << live_audio_recording.duration_seconds()
                              << " seconds of microphone audio to "
                              << options.record_audio_file << '\n';
                }
            }
            if (should_mux_live_audio) {
                const double video_duration = writer.get_duration();
                if (live_audio_recording.empty()) {
                    std::cerr << "acmxvk: live audio recording was empty; preserving "
                                 "the encoded video without audio\n";
                } else if (!audio::FileAudioSource::mux_recording_into_video(
                               std::move(live_audio_recording.samples),
                               live_audio_recording.sample_rate,
                               options.output_file, video_duration)) {
                    std::cerr << "acmxvk: live-audio mux failed; preserving the "
                                 "encoded video without audio\n";
                }
            }
#endif
            snapshot_writer.stop();
        }

        void event(SDL_Event &event) override {
            mxvk::VK_Window::event(event);
            if (event.type == SDL_EVENT_KEY_DOWN &&
                event.key.key == SDLK_PAGEUP) {
                adjustTimeSpeed(0.1);
            } else if (event.type == SDL_EVENT_KEY_DOWN &&
                       event.key.key == SDLK_PAGEDOWN) {
                adjustTimeSpeed(-0.1);
            }

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
                case SDLK_LEFT:
                    selectGpuFilter(-1);
                    break;
                case SDLK_RIGHT:
                    selectGpuFilter(1);
                    break;
                case SDLK_SPACE:
                    beginCrossfade();
                    effects_enabled = !effects_enabled;
                    applyShaderPipeline();
                    std::cout << "acmxvk: shader effects "
                              << (effects_enabled ? "enabled" : "bypassed") << '\n';
                    break;
                case SDLK_P:
                    if (!playlist.empty()) {
                        beginCrossfade();
                        playlist_enabled = !playlist_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: playlist "
                                  << (playlist_enabled ? "enabled" : "disabled") << '\n';
                        if (playlist_enabled) {
                            logSelectedPlaylistNode("selected");
                        }
                    } else {
                        togglePause();
                    }
                    break;
                case SDLK_L:
                    toggleFreeze();
                    break;
                case SDLK_T:
                    shader_time_active = !shader_time_active;
                    previous_frame = std::chrono::steady_clock::now();
                    std::cout << "acmxvk: shader time "
                              << (shader_time_active ? "enabled" : "disabled") << '\n';
                    break;
                case SDLK_Q:
#ifdef AUDIO_ENABLED
                    if (audioSourceOpen()) {
                        audio_time_active = !audio_time_active;
                        previous_frame = std::chrono::steady_clock::now();
                        std::cout << "acmxvk: audio-reactive shader time "
                                  << (audio_time_active ? "enabled" : "disabled")
                                  << '\n';
                    }
#endif
                    break;
                case SDLK_HOME:
#ifdef AUDIO_ENABLED
                    if (audioSourceOpen()) {
                        audio_delta_time = !audio_delta_time;
                        std::cout << "acmxvk: audio delta-time scaling "
                                  << (audio_delta_time ? "enabled" : "disabled")
                                  << '\n';
                    }
#endif
                    break;
                case SDLK_END:
#ifdef AUDIO_ENABLED
                    if (audioSourceOpen()) {
                        spectrum_scale_by_sensitivity =
                            !spectrum_scale_by_sensitivity;
                        std::cout << "acmxvk: spectrum sensitivity scaling "
                                  << (spectrum_scale_by_sensitivity ? "enabled"
                                                                    : "disabled")
                                  << '\n';
                    }
#endif
                    break;
                case SDLK_U:
                    stepShaderTime(0.05);
                    break;
                case SDLK_I:
                    stepShaderTime(-0.05);
                    break;
                case SDLK_F:
                    toggleFullscreen();
                    break;
                case SDLK_F9:
                    counter_disabled = !counter_disabled;
                    hud_fps_frame_count = 0;
                    hud_fps_last_tick = std::chrono::steady_clock::now();
                    if (!counter_disabled) {
                        initializeOverlayFont();
                    }
                    std::cout << "acmxvk: runtime HUD "
                              << (counter_disabled ? "hidden" : "shown")
                              << " (F9)\n";
                    break;
                case SDLK_E:
                    if (!options.watermark_text.empty()) {
                        watermark_enabled = !watermark_enabled;
                        std::cout << "acmxvk: watermark "
                                  << (watermark_enabled ? "enabled" : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_INSERT:
                    adjustAudioSensitivity(0.1F);
                    break;
                case SDLK_DELETE:
                    adjustAudioSensitivity(-0.1F);
                    break;
                case SDLK_M:
                    if (!configured_passes.empty()) {
                        beginCrossfade();
                        multipass_enabled = !multipass_enabled;
                        applyShaderPipeline();
                        std::cout << "acmxvk: multipass "
                                  << (multipass_enabled ? "enabled" : "disabled") << '\n';
                    }
                    break;
                case SDLK_J:
                    toggleAutopilot(false);
                    break;
                case SDLK_N:
                    autopilot_random_crossfade =
                        !autopilot_random_crossfade;
                    std::cout << "acmxvk: random autopilot crossfade "
                              << (autopilot_random_crossfade ? "enabled"
                                                             : "disabled")
                              << '\n';
                    break;
                case SDLK_K:
                    shader_locked = !shader_locked;
                    std::cout << "acmxvk: shader lock "
                              << (shader_locked ? "enabled" : "disabled")
                              << '\n';
                    break;
                case SDLK_3:
                    if (model_initialized) {
                        model_3d_active = !model_3d_active;
                        model_video_timeline_initialized = false;
                        model_last_render_time =
                            std::chrono::steady_clock::now();
                        applyShaderPipeline();
                        std::cout << "acmxvk: "
                                  << (model_3d_active ? "3D model" : "2D sprite")
                                  << " rendering enabled\n";
                    }
                    break;
                case SDLK_V:
                    if (model_initialized) {
                        model_auto_rotate = !model_auto_rotate;
                        std::cout << "acmxvk: 3D view rotation "
                                  << (model_auto_rotate ? "enabled" : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_C:
                    if (model_initialized) {
                        model_wave_active = !model_wave_active;
                        std::cout << "acmxvk: 3D wave effect "
                                  << (model_wave_active ? "enabled"
                                                        : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_O:
                    if (model_initialized) {
                        model_scale_oscillation_active =
                            !model_scale_oscillation_active;
                        std::cout << "acmxvk: 3D scale oscillation "
                                  << (model_scale_oscillation_active
                                          ? "enabled"
                                          : "disabled")
                                  << '\n';
                    }
                    break;
                case SDLK_X:
                    if (model_initialized) {
                        model_pitch_degrees = 0.0F;
                        model_yaw_degrees = 270.0F;
                        model_rotation_x_degrees = 0.0F;
                        model_rotation_y_degrees = 0.0F;
                        model_rotation_z_degrees = 0.0F;
                        model_camera_distance = 0.0F;
                        model_scale = 1.0F;
                        model_view_rotation_degrees = 0.0F;
                        std::cout << "acmxvk: model view reset\n";
                    }
                    break;
                case SDLK_LEFTBRACKET:
                    cycleCrossfade(-1);
                    break;
                case SDLK_RIGHTBRACKET:
                    cycleCrossfade(1);
                    break;
                case SDLK_MINUS:
                case SDLK_UNDERSCORE:
                case SDLK_KP_MINUS:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0) {
                        adjustModelScale(-0.05F);
                    }
                    break;
                case SDLK_PLUS:
                case SDLK_EQUALS:
                case SDLK_KP_PLUS:
                    if ((event.key.mod & SDL_KMOD_SHIFT) != 0) {
                        adjustModelScale(0.05F);
                    }
                    break;
                case SDLK_COMMA:
                    if (model_initialized) {
                        model_rotation_speed =
                            std::max(0.0F, model_rotation_speed - 5.0F);
                        std::cout << "acmxvk: 3D view rotation speed "
                                  << model_rotation_speed << " degrees/second\n";
                    }
                    break;
                case SDLK_PERIOD:
                    if (model_initialized) {
                        model_rotation_speed =
                            std::min(360.0F, model_rotation_speed + 5.0F);
                        std::cout << "acmxvk: 3D view rotation speed "
                                  << model_rotation_speed << " degrees/second\n";
                    }
                    break;
                case SDLK_Y:
                    toggleAutopilot(true);
                    break;
                case SDLK_Z:
                    requestSnapshot(SnapshotFormat::Png);
                    break;
                case SDLK_4:
                    requestSnapshot(SnapshotFormat::Tiff);
                    break;
                case SDLK_5:
                    requestSnapshot(SnapshotFormat::WebP);
                    break;
                case SDLK_6:
                    requestSnapshot(SnapshotFormat::Raw);
                    break;
                default:
                    break;
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                mouse_x = event.motion.x;
                mouse_y = event.motion.y;
                if (model_mouse_dragging && model_initialized) {
                    const int x = static_cast<int>(event.motion.x);
                    const int y = static_cast<int>(event.motion.y);
                    model_yaw_degrees +=
                        static_cast<float>(x - model_last_mouse_x) * 0.35F;
                    model_pitch_degrees = std::clamp(
                        model_pitch_degrees +
                            static_cast<float>(y - model_last_mouse_y) * 0.35F,
                        -89.0F, 89.0F);
                    model_last_mouse_x = x;
                    model_last_mouse_y = y;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = true;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
                model_mouse_dragging = model_initialized;
                model_last_mouse_x = static_cast<int>(event.button.x);
                model_last_mouse_y = static_cast<int>(event.button.y);
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
                       event.button.button == SDL_BUTTON_LEFT) {
                mouse_pressed = false;
                model_mouse_dragging = false;
                mouse_x = event.button.x;
                mouse_y = event.button.y;
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL &&
                       model_initialized &&
                       !model_scale_oscillation_active) {
                const float wheel = event.wheel.y != 0.0F
                                        ? event.wheel.y
                                        : static_cast<float>(
                                              event.wheel.integer_y);
                model_camera_distance = std::clamp(
                    model_camera_distance - wheel * 0.2F, -20.0F, 20.0F);
            }
        }

        void onSwapchainRecreated() override {
            initializeSprite();
            if (model_initialized) {
                input_model.resize(this);
            }
            initializeOverlayFont();
        }

        void onRecordCustomRendering(VkCommandBuffer command_buffer,
                                     std::uint32_t image_index) override {
            if (model_texture_prepass_active) {
                return;
            }
            recordModel(command_buffer, image_index, VK_NULL_HANDLE);
        }

        void onRecordPostProcessingTexture(
            VkCommandBuffer command_buffer, std::uint32_t image_index,
            VkImageView texture_view,
            [[maybe_unused]] VkExtent2D texture_extent) override {
            if (!model_texture_prepass_active) {
                return;
            }
            recordModel(command_buffer, image_index, texture_view);
        }

        void recordModel(VkCommandBuffer command_buffer,
                         std::uint32_t image_index,
                         VkImageView texture_view) {
            if (!model_3d_active || !model_initialized) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            float delta = std::chrono::duration<float>(
                              now - model_last_render_time)
                              .count();
            model_last_render_time = now;
            delta = std::clamp(delta, 0.0F, 0.1F);

            float animation_delta = delta;
            std::uint64_t animation_steps = 1U;
            double video_timeline = 0.0;
            std::uint64_t video_frame_index = 0U;
            if (currentVideoTimeline(video_timeline, &video_frame_index)) {
                if (!model_video_timeline_initialized ||
                    video_frame_index < previous_model_video_frame) {
                    if (model_video_timeline_initialized &&
                        video_frame_index < previous_model_video_frame) {
                        model_wave_phase = 0.0F;
                        model_wave_amplitude_x = 0.0F;
                        model_wave_amplitude_y = 0.0F;
                        model_wave_amplitude_z = 0.0F;
                        model_wave_direction_x = 1.0F;
                        model_wave_direction_y = 1.0F;
                        model_wave_direction_z = 1.0F;
                        model_scale_oscillation_phase = 0.0F;
                        if (model_auto_rotate) {
                            model_view_rotation_degrees = 0.0F;
                        }
                    }
                    animation_delta = 0.0F;
                    animation_steps = 0U;
                    model_video_timeline_initialized = true;
                } else {
                    animation_steps =
                        video_frame_index - previous_model_video_frame;
                    animation_delta = static_cast<float>(
                        static_cast<double>(animation_steps) /
                        video_source_fps);
                }
                previous_model_video_frame = video_frame_index;
            } else {
                model_video_timeline_initialized = false;
            }
            if (model_auto_rotate && !rendering_frozen) {
                model_view_rotation_degrees = std::fmod(
                    model_view_rotation_degrees +
                        model_rotation_speed * animation_delta,
                    360.0F);
            }
            if (model_wave_active) {
                const float wave_step =
                    audio_time_active && audioSourceOpen()
                        ? model_wave_audio_step
                        : 0.05F;
                model_wave_phase = std::fmod(
                    model_wave_phase +
                        wave_step * static_cast<float>(animation_steps),
                    360.0F);

                const auto advance_amplitude = [](float &amplitude,
                                                  float &direction,
                                                  std::uint64_t steps) {
                    constexpr float AMPLITUDE_RANGE = 0.5F;
                    constexpr float AMPLITUDE_PERIOD =
                        AMPLITUDE_RANGE * 2.0F;
                    float phase = direction >= 0.0F
                                      ? amplitude
                                      : AMPLITUDE_PERIOD - amplitude;
                    phase = std::fmod(
                        phase + 0.005F * static_cast<float>(steps),
                        AMPLITUDE_PERIOD);
                    if (phase < AMPLITUDE_RANGE) {
                        amplitude = phase;
                        direction = 1.0F;
                    } else {
                        amplitude = AMPLITUDE_PERIOD - phase;
                        direction = -1.0F;
                    }
                };
                advance_amplitude(model_wave_amplitude_x,
                                  model_wave_direction_x, animation_steps);
                advance_amplitude(model_wave_amplitude_y,
                                  model_wave_direction_y, animation_steps);
                advance_amplitude(model_wave_amplitude_z,
                                  model_wave_direction_z, animation_steps);
            }
            if (model_scale_oscillation_active) {
                model_scale_oscillation_phase = std::fmod(
                    model_scale_oscillation_phase +
                        0.016F * static_cast<float>(animation_steps),
                    2.0F * std::numbers::pi_v<float>);
            }

            const bool *keyboard = SDL_GetKeyboardState(nullptr);
            const bool model_scale_modifier =
                (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
            if (!model_scale_oscillation_active &&
                keyboard[SDL_SCANCODE_1]) {
                model_camera_movement_speed = std::clamp(
                    model_camera_movement_speed + 0.1F * delta * 30.0F,
                    0.01F, 20.0F);
            }
            if (!model_scale_oscillation_active &&
                keyboard[SDL_SCANCODE_2]) {
                model_camera_movement_speed = std::clamp(
                    model_camera_movement_speed - 0.1F * delta * 30.0F,
                    0.01F, 20.0F);
            }
            if (!model_scale_oscillation_active && !model_scale_modifier &&
                (keyboard[SDL_SCANCODE_EQUALS] ||
                 keyboard[SDL_SCANCODE_KP_PLUS])) {
                model_camera_distance = std::clamp(
                    model_camera_distance +
                        model_camera_movement_speed * delta,
                    -20.0F, 20.0F);
            }
            if (!model_scale_oscillation_active && !model_scale_modifier &&
                (keyboard[SDL_SCANCODE_MINUS] ||
                 keyboard[SDL_SCANCODE_KP_MINUS])) {
                model_camera_distance = std::clamp(
                    model_camera_distance -
                        model_camera_movement_speed * delta,
                    -20.0F, 20.0F);
            }
            if (!model_auto_rotate) {
                if (keyboard[SDL_SCANCODE_W]) {
                    model_pitch_degrees +=
                        model_camera_rotation_speed * 0.3F * delta * 30.0F;
                }
                if (keyboard[SDL_SCANCODE_S]) {
                    model_pitch_degrees -=
                        model_camera_rotation_speed * 0.33F * delta * 30.0F;
                }
                model_pitch_degrees =
                    std::fmod(model_pitch_degrees, 360.0F);
                if (model_pitch_degrees < 0.0F) {
                    model_pitch_degrees += 360.0F;
                }
                if (keyboard[SDL_SCANCODE_A]) {
                    model_yaw_degrees -=
                        model_camera_rotation_speed * 0.3F * delta * 30.0F;
                }
                if (keyboard[SDL_SCANCODE_D]) {
                    model_yaw_degrees +=
                        model_camera_rotation_speed * 0.3F * delta * 30.0F;
                }
                model_yaw_degrees = std::fmod(model_yaw_degrees, 360.0F);
                if (model_yaw_degrees < 0.0F) {
                    model_yaw_degrees += 360.0F;
                }
            }

            const VkExtent2D extent = getRenderExtent();
            const float aspect = extent.height > 0U
                                     ? static_cast<float>(extent.width) /
                                           static_cast<float>(extent.height)
                                     : 1.0F;

            mxvk::UniformBufferObject uniforms{};
            uniforms.model = glm::scale(
                glm::mat4(1.0F),
                glm::vec3(input_model.modelRenderScale() * model_scale));
            uniforms.model = glm::rotate(
                uniforms.model, glm::radians(model_rotation_x_degrees),
                glm::vec3(1.0F, 0.0F, 0.0F));
            uniforms.model = glm::rotate(
                uniforms.model, glm::radians(model_rotation_y_degrees),
                glm::vec3(0.0F, 1.0F, 0.0F));
            uniforms.model = glm::rotate(
                uniforms.model, glm::radians(model_rotation_z_degrees),
                glm::vec3(0.0F, 0.0F, 1.0F));
            uniforms.model = glm::translate(
                uniforms.model, input_model.modelCenterOffset());

            glm::vec3 look_direction{};
            glm::vec3 camera_up(0.0F, 1.0F, 0.0F);
            if (model_auto_rotate) {
                const float rotation =
                    glm::radians(model_view_rotation_degrees);
                look_direction = glm::vec3(
                    0.48F * std::sin(rotation),
                    0.48F * std::sin(rotation * 0.7F),
                    0.48F * std::cos(rotation));
            } else {
                const float pitch = glm::radians(model_pitch_degrees);
                const float yaw = glm::radians(model_yaw_degrees);
                look_direction = glm::normalize(glm::vec3(
                                     std::cos(pitch) * std::cos(yaw),
                                     std::sin(pitch),
                                     std::cos(pitch) * std::sin(yaw))) *
                                 0.48F;
                camera_up = glm::vec3(-std::sin(pitch) * std::cos(yaw),
                                      std::cos(pitch),
                                      -std::sin(pitch) * std::sin(yaw));
            }
            const float camera_offset =
                model_scale_oscillation_active
                    ? 0.3F * std::sin(model_scale_oscillation_phase)
                    : model_camera_distance;
            const glm::vec3 camera_position =
                -glm::normalize(look_direction) * camera_offset;
            uniforms.view = glm::lookAt(camera_position,
                                        camera_position + look_direction,
                                        camera_up);
            uniforms.proj = glm::perspective(
                glm::radians(120.0F), aspect, 0.01F, 1000.0F);
            uniforms.proj[1][1] *= -1.0F;
            uniforms.fx =
                model_wave_active
                    ? glm::vec4(model_wave_amplitude_x,
                                model_wave_amplitude_y,
                                model_wave_amplitude_z, model_wave_phase)
                    : glm::vec4(0.0F);

            input_model.updateFragmentUBO(image_index,
                                          model_fragment_uniforms);

            mxvk::ModelFragmentPushConstants fragment_constants{};
            fragment_constants.screenWidth = static_cast<float>(extent.width);
            fragment_constants.screenHeight = static_cast<float>(extent.height);
            fragment_constants.spriteSizeW = static_cast<float>(extent.width);
            fragment_constants.spriteSizeH = static_cast<float>(extent.height);
            fragment_constants.effectsOn = effects_enabled ? 1.0F : 0.0F;
            fragment_constants.params = glm::vec4(
                1.0F, 1.0F, 1.0F, static_cast<float>(shader_time));
            input_model.setFragmentPushConstants(fragment_constants);

            if (texture_view != VK_NULL_HANDLE) {
                input_model.renderWithExternalTexture(
                    command_buffer, image_index, texture_view, uniforms,
                    false);
            } else {
                input_model.renderWithPushConstants(
                    command_buffer, image_index, 0U, uniforms, false);
            }
        }

        void proc() override {
            if (recording_complete) {
                return;
            }

            paceMaximizedRendering();

            pollMidi();
            sync_interface_control();

            source_frame_received = false;
            recording_frame_due = false;
            recording_frame_has_pts = false;
            bool clocked_video_handled = false;

            if (!rendering_frozen && !input_paused) {
                if (source_kind == SourceKind::Graphic) {
                    source_frame_received = true;
                } else if (initial_frame_pending) {
                    initial_frame_pending = false;
                    source_frame_received = true;
                } else {
                    double clock_seconds = 0.0;
                    if (source_kind == SourceKind::Video &&
                        media_timeline_started &&
                        mediaClockSeconds(clock_seconds)) {
                        clocked_video_handled = true;
                        if (!readClockedVideoFrame(clock_seconds)) {
                            return;
                        }
                    } else {
                        const bool read_frame = readTrackedInputFrame();
                        if (!read_frame && !handleCaptureEnd()) {
                            return;
                        }
                        source_frame_received =
                            read_frame || source_kind == SourceKind::Video;
                    }
                }
            }

            startMediaTimelineIfReady();
            const bool render_latest_camera_frame =
                options.maximize_fps && source_kind == SourceKind::Camera &&
                media_timeline_started;
            if ((source_frame_received || render_latest_camera_frame) &&
                !clocked_video_handled) {
                recording_frame_due = true;
                if (source_kind == SourceKind::Video) {
                    recording_frame_has_pts = true;
                    recording_frame_pts = decoded_video_frame_count - 1;
                } else {
                    double clock_seconds = 0.0;
                    if (mediaClockSeconds(clock_seconds)) {
                        const double rate = outputFrameRate();
                        const std::uint64_t target_frame =
                            static_cast<std::uint64_t>(std::floor(
                                std::max(clock_seconds, 0.0) * rate));
                        if (target_frame < next_clock_output_frame) {
                            recording_frame_due = false;
                        } else {
                            recording_frame_has_pts = true;
                            recording_frame_pts = target_frame;
                            next_clock_output_frame = target_frame + 1;
                            if (source_kind == SourceKind::Camera &&
                                writer.is_open() &&
                                !camera_recording_clock_logged) {
                                std::cout
                                    << "acmxvk: camera recording uses real-time "
                                       "PTS; slow frames preserve capture duration\n";
                                camera_recording_clock_logged = true;
                            }
                        }
                    }
                }
            }

            if (!rendering_frozen) {
                updateAutopilot();
            }
            updateCameraHistory();
            const VkExtent2D extent = getRenderExtent();
            const int target_width = extent.width > 0U ? static_cast<int>(extent.width) : options.width;
            const int target_height =
                extent.height > 0U ? static_cast<int>(extent.height) : options.height;

            if (!rendering_frozen) {
                updateShaderUniforms(target_width, target_height);
            }
            if (!model_3d_active || model_texture_prepass_active) {
                frame_sprite->drawSpriteRect(0, 0, target_width,
                                             target_height);
            }
            queueOverlayText();
            updateWindowTitle();
            setFrameReadbackEnabled(
                snapshot_pending ||
                (continuousReadbackEnabled() && recording_frame_due));
        }
