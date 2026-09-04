#include "main_window.hpp"

#include <csignal>

namespace acmxvk {
    namespace {
        volatile std::sig_atomic_t HEADLESS_SHUTDOWN_REQUESTED = 0;
        constexpr int COLOR_TRANSFER_SMPTE2084 = 16;
        constexpr int COLOR_TRANSFER_ARIB_STD_B67 = 18;

        [[nodiscard]] float decode_pq(float encoded) {
            constexpr float M1 = 2610.0F / 16384.0F;
            constexpr float M2 = 2523.0F / 32.0F;
            constexpr float C1 = 3424.0F / 4096.0F;
            constexpr float C2 = 2413.0F / 128.0F;
            constexpr float C3 = 2392.0F / 128.0F;
            const float power_value =
                std::pow(std::clamp(encoded, 0.0F, 1.0F), 1.0F / M2);
            const float numerator = std::max(power_value - C1, 0.0F);
            const float denominator =
                std::max(C2 - C3 * power_value, 1.0e-6F);
            return std::pow(numerator / denominator, 1.0F / M1);
        }

        [[nodiscard]] float decode_hlg(float encoded) {
            constexpr float A = 0.17883277F;
            constexpr float B = 0.28466892F;
            constexpr float C = 0.55991073F;
            encoded = std::max(encoded, 0.0F);
            if (encoded <= 0.5F) {
                return encoded * encoded / 3.0F;
            }
            return (std::exp((encoded - C) / A) + B) / 12.0F;
        }

        [[nodiscard]] cv::Mat decode_hdr_transfer(const cv::Mat &rgba,
                                                  bool hlg) {
            if (rgba.empty() ||
                (rgba.type() != CV_8UC4 && rgba.type() != CV_16UC4)) {
                return rgba;
            }
            cv::Mat linear(rgba.rows, rgba.cols, CV_16UC4);
            const float scale = rgba.type() == CV_16UC4
                                    ? 1.0F / 65535.0F
                                    : 1.0F / 255.0F;
            for (int row = 0; row < rgba.rows; ++row) {
                auto *destination = linear.ptr<std::uint16_t>(row);
                for (int column = 0; column < rgba.cols; ++column) {
                    for (int channel = 0; channel < 3; ++channel) {
                        const std::size_t offset =
                            static_cast<std::size_t>(column) * 4U +
                            static_cast<std::size_t>(channel);
                        const float encoded =
                            rgba.type() == CV_16UC4
                                ? static_cast<float>(
                                      rgba.ptr<std::uint16_t>(row)[offset]) *
                                      scale
                                : static_cast<float>(
                                      rgba.ptr<std::uint8_t>(row)[offset]) *
                                      scale;
                        const float decoded =
                            hlg ? decode_hlg(encoded) : decode_pq(encoded);
                        destination[offset] = static_cast<std::uint16_t>(
                            std::lround(std::clamp(decoded, 0.0F, 1.0F) *
                                        65535.0F));
                    }
                    const std::size_t alpha_offset =
                        static_cast<std::size_t>(column) * 4U + 3U;
                    destination[alpha_offset] =
                        rgba.type() == CV_16UC4
                            ? rgba.ptr<std::uint16_t>(row)[alpha_offset]
                            : static_cast<std::uint16_t>(
                                  rgba.ptr<std::uint8_t>(row)[alpha_offset] *
                                  257U);
                }
            }
            return linear;
        }

        [[nodiscard]] cv::Mat rgba16ToRgba8(const cv::Mat &rgba) {
            if (rgba.empty() || rgba.type() != CV_16UC4) {
                return rgba;
            }
            cv::Mat converted;
            rgba.convertTo(converted, CV_8UC4, 1.0 / 257.0);
            return converted;
        }

        [[nodiscard]] float tone_map_channel(float value) {
            value = std::max(value, 0.0F);
            return std::clamp(
                (value * (2.51F * value + 0.03F)) /
                    (value * (2.43F * value + 0.59F) + 0.14F),
                0.0F, 1.0F);
        }

        [[nodiscard]] float encode_srgb(float value) {
            value = std::clamp(value, 0.0F, 1.0F);
            return value <= 0.0031308F
                       ? 12.92F * value
                       : 1.055F * std::pow(value, 1.0F / 2.4F) - 0.055F;
        }

        [[nodiscard]] std::vector<std::uint8_t> tone_map_hdr_rgba16(
            const std::vector<std::uint16_t> &rgba, bool hlg) {
            std::vector<std::uint8_t> converted(rgba.size());
            const float reference_scale =
                hlg ? 1000.0F / 203.0F : 10000.0F / 203.0F;
            for (std::size_t offset = 0; offset + 3U < rgba.size();
                 offset += 4U) {
                const auto decode = [hlg](std::uint16_t sample) {
                    const float encoded =
                        static_cast<float>(sample) / 65535.0F;
                    return hlg ? decode_hlg(encoded) : decode_pq(encoded);
                };
                const float red = decode(rgba[offset]) * reference_scale;
                const float green =
                    decode(rgba[offset + 1U]) * reference_scale;
                const float blue =
                    decode(rgba[offset + 2U]) * reference_scale;
                const float bt709_red =
                    1.660491F * red - 0.587641F * green - 0.072850F * blue;
                const float bt709_green =
                    -0.124550F * red + 1.132900F * green - 0.008349F * blue;
                const float bt709_blue =
                    -0.018151F * red - 0.100579F * green + 1.118730F * blue;
                converted[offset] = static_cast<std::uint8_t>(std::lround(
                    encode_srgb(tone_map_channel(bt709_red)) * 255.0F));
                converted[offset + 1U] = static_cast<std::uint8_t>(
                    std::lround(encode_srgb(tone_map_channel(bt709_green)) *
                                255.0F));
                converted[offset + 2U] = static_cast<std::uint8_t>(
                    std::lround(encode_srgb(tone_map_channel(bt709_blue)) *
                                255.0F));
                converted[offset + 3U] = static_cast<std::uint8_t>(
                    (static_cast<std::uint32_t>(rgba[offset + 3U]) + 128U) /
                    257U);
            }
            return converted;
        }
    } // namespace

    void request_headless_shutdown([[maybe_unused]] int signal_number) noexcept {
        HEADLESS_SHUTDOWN_REQUESTED = 1;
    }

    // Window construction, event handling, rendering callbacks, and main loop.
    MainWindow::MainWindow(Options options)
        : mxvk::VK_Window("ACMXVK", options.width, options.height,
                          options.fullscreen, MXVK_VALIDATION,
                          options.enable_vsync
                              ? PresentModePreference::Vsync
                              : PresentModePreference::LowLatency,
                          options.headless ? RuntimeMode::Headless
                                           : RuntimeMode::Windowed),
          options(std::move(options)) {
        if (this->options.headless) {
            std::cout << "acmxvk: headless mode enabled: surface-free Vulkan "
                         "rendering without an SDL window\n";
        }
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

    MainWindow::~MainWindow() {
        interface_client.close();
        latest_camera_frame.stop();
        try {
            flushFrameReadbacks();
        } catch (const std::exception &error) {
            std::cerr << "acmxvk: unable to flush pending frame readbacks: "
                      << error.what() << '\n';
        }
        if (headless_progress_complete) {
            emitHeadlessProgress(true);
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

    void MainWindow::event(SDL_Event &event) {
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

    void MainWindow::onSwapchainRecreated() {
        // Initial headless extent configuration creates MXVK's surface-free
        // targets before MainWindow performs its normal sprite setup. Do not
        // initialize here: doing so consumes video frame zero, after which the
        // constructor consumes frame one and leaves an empty encoded PTS zero.
        if (frame_sprite == nullptr) {
            return;
        }
        initializeSprite();
        if (model_initialized) {
            input_model.resize(this);
        }
        initializeOverlayFont();
    }

    void MainWindow::onRecordCustomRendering(VkCommandBuffer command_buffer,
                                             std::uint32_t image_index) {
        if (model_texture_prepass_active) {
            return;
        }
        recordModel(command_buffer, image_index, VK_NULL_HANDLE);
    }

    void MainWindow::onRecordPostProcessingTexture(
        VkCommandBuffer command_buffer, std::uint32_t image_index,
        VkImageView texture_view,
        [[maybe_unused]] VkExtent2D texture_extent) {
        if (!model_texture_prepass_active) {
            return;
        }
        recordModel(command_buffer, image_index, texture_view);
    }

    void MainWindow::recordModel(VkCommandBuffer command_buffer,
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

    void MainWindow::proc() {
        if (options.headless && HEADLESS_SHUTDOWN_REQUESTED != 0) {
            if (!headless_shutdown_logged) {
                std::cout << "acmxvk: Ctrl+C received; draining rendered "
                             "frames and closing output\n";
                headless_shutdown_logged = true;
            }
            setFrameReadbackEnabled(false);
            exit();
            return;
        }
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
#ifdef AUDIO_ENABLED

    // Audio, MIDI, custom controls, and media-clock coordination.
    void MainWindow::resetAudioWarmup() {
        audio_warmup_envelope = 0.0F;
        audio_warmup_started = false;
        if (options.audio_warm_rate <= 0.0) {
            std::cout << "acmxvk: audio shader warmup disabled\n";
        } else {
            std::cout << "acmxvk: audio shader warmup "
                      << options.audio_warm_rate << "/second (~"
                      << 1.0 / options.audio_warm_rate
                      << " seconds to full strength)\n";
        }
    }

    [[nodiscard]] float MainWindow::updateAudioWarmup(
        std::chrono::steady_clock::time_point now) {
        if (options.audio_warm_rate <= 0.0) {
            audio_warmup_envelope = 1.0F;
            return audio_warmup_envelope;
        }
        if (!audio_warmup_started) {
            audio_warmup_started = true;
            audio_warmup_last_tick = now;
            return audio_warmup_envelope;
        }

        const float delta = std::max(
            std::chrono::duration<float>(now - audio_warmup_last_tick).count(),
            0.0F);
        audio_warmup_last_tick = now;
        audio_warmup_envelope = std::min(
            audio_warmup_envelope +
                delta * static_cast<float>(options.audio_warm_rate),
            1.0F);
        return audio_warmup_envelope;
    }
#endif

    void MainWindow::initializeDnn() {
#ifdef ACMXVK_WITH_DNN
        if (!options.human_model.empty()) {
            human_segmenter =
                std::make_unique<dnn::HumanSegmenter>(options.human_model);
            std::cout << "acmxvk: PP-HumanSeg enabled: "
                      << options.human_model << " ("
                      << (options.human_background
                              ? "background-only shader composition"
                              : "foreground isolation")
                      << ", automatic CPU/CUDA backend selection)\n";
        }
        if (!options.edge_model.empty()) {
            edge_detector =
                std::make_unique<dnn::EdgeDetector>(options.edge_model);
            std::cout << "acmxvk: DexiNed edge detection enabled: "
                      << options.edge_model
                      << " (automatic CPU/CUDA backend selection)\n";
        }
        if (!options.onnx_configuration.empty()) {
            generic_onnx_processor =
                std::make_unique<dnn::GenericOnnxProcessor>(
                    options.onnx_configuration);
            std::cout << "acmxvk: generic ONNX processing enabled: "
                      << options.onnx_configuration
                      << " (automatic CPU/CUDA backend selection)\n";
        }
#endif
    }

    void MainWindow::initializeGpuFilters() {
#ifdef ACMXVK_WITH_CUDA
        if (options.gpu_filter_indices.empty()) {
            return;
        }
        gpu_filter_engine = std::make_unique<gpu::FilterEngine>(
            options.gpu_filter_indices, options.gpu_frame_buffer_size);
#endif
    }

    void MainWindow::selectGpuFilter(int direction) {
#ifdef ACMXVK_WITH_CUDA
        if (gpu_filter_engine != nullptr &&
            gpu_filter_engine->select_relative_filter(direction) &&
            source_kind == SourceKind::Graphic && !graphic_rgba.empty()) {
            uploadInputFrame(graphic_rgba);
            if (history_initialized) {
                updateHistoryFrame(graphic_rgba);
                history_delay_counter = 0;
            }
        }
#else
        static_cast<void>(direction);
#endif
    }

    void MainWindow::openMidi() {
#ifdef MIDI_ENABLED
        if (!options.midi_device_specified && !options.midi_monitor &&
            options.midi_map_file.empty() && midi_cc_mappings.empty()) {
            return;
        }
        midi_input = std::make_unique<midi::MidiInput>();
        const int port = options.midi_device_specified ? options.midi_device : 0;
        if (!midi_input->open(port)) {
            throw std::runtime_error("could not open MIDI input port " +
                                     std::to_string(port));
        }
#endif
    }

    void MainWindow::configureMidiMappings() {
#ifdef MIDI_ENABLED
        if (!options.midi_map_file.empty()) {
            midi_action_mappings =
                midi::load_mapping_file(options.midi_map_file);
            midi_knob_states.resize(midi_action_mappings.size());
            std::cout << "acmxvk: loaded " << midi_action_mappings.size()
                      << " MIDI mapping(s) from " << options.midi_map_file
                      << '\n';

            for (int slider = 0; slider < 4; ++slider) {
                const int action = 600 + slider * 2;
                const bool mapped = std::any_of(
                    midi_action_mappings.begin(),
                    midi_action_mappings.end(),
                    [&](const midi::MidiMapping &mapping) {
                        return mapping.primary_action == action &&
                               mapping.secondary_action == action + 1;
                    });
                if (!mapped) {
                    continue;
                }
                const std::string name =
                    "slider" + std::to_string(slider + 1);
                const auto uniform = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &candidate) {
                        return candidate.name == name;
                    });
                if (uniform == custom_uniforms.end()) {
                    std::cerr << "acmxvk: MIDI " << name
                              << " mapping has no matching custom uniform in "
                                 "library.json\n";
                    continue;
                }
                midi_slider_uniform_indices[slider] = static_cast<int>(
                    std::distance(custom_uniforms.begin(), uniform));
                std::cout << "acmxvk: MIDI Slider " << (slider + 1)
                          << " -> " << name << " [" << uniform->minimum
                          << ", " << uniform->maximum << "]\n";
            }

            std::size_t active_mappings = 0;
            for (const midi::MidiMapping &mapping :
                 midi_action_mappings) {
                if (isMidiMappingSupported(mapping)) {
                    ++active_mappings;
                } else if (options.midi_monitor) {
                    std::cerr
                        << "acmxvk: MIDI map action unavailable in this build: "
                        << mapping.primary_action << ':'
                        << mapping.secondary_action << '\n';
                }
            }
            std::cout << "acmxvk: MIDI map has " << active_mappings
                      << " active mapping(s)";
            if (active_mappings != midi_action_mappings.size()) {
                std::cout << " and "
                          << (midi_action_mappings.size() - active_mappings)
                          << " mapping(s) reserved for unported ACMX2 controls";
            }
            std::cout << '\n';
        }

        for (const std::string &mapping_text : options.midi_cc_mappings) {
            const std::size_t equals = mapping_text.find('=');
            if (equals == std::string::npos || equals == 0 ||
                equals + 1 >= mapping_text.size() ||
                mapping_text.find('=', equals + 1) != std::string::npos) {
                throw std::runtime_error(
                    "--midi-cc requires [channel:]CC=uniform: " +
                    mapping_text);
            }

            const std::string source = trim(mapping_text.substr(0, equals));
            const std::string uniform_name =
                trim(mapping_text.substr(equals + 1));
            if (!isValidCustomUniformName(uniform_name)) {
                throw std::runtime_error(
                    "--midi-cc contains an invalid uniform name: " +
                    uniform_name);
            }

            int channel = -1;
            int controller = 0;
            const std::size_t colon = source.find(':');
            if (colon == std::string::npos) {
                controller = parseInteger(source, "--midi-cc");
            } else {
                if (colon == 0 || colon + 1 >= source.size() ||
                    source.find(':', colon + 1) != std::string::npos) {
                    throw std::runtime_error(
                        "--midi-cc requires [channel:]CC=uniform: " +
                        mapping_text);
                }
                channel = parseInteger(
                    std::string_view(source).substr(0, colon), "--midi-cc");
                controller = parseInteger(
                    std::string_view(source).substr(colon + 1), "--midi-cc");
                if (channel < 1 || channel > 16) {
                    throw std::runtime_error(
                        "--midi-cc channel must be between 1 and 16");
                }
                --channel;
            }
            if (controller < 0 || controller > 127) {
                throw std::runtime_error(
                    "--midi-cc controller must be between 0 and 127");
            }

            const auto uniform = std::find_if(
                custom_uniforms.begin(), custom_uniforms.end(),
                [&](const ShaderManifest::CustomUniform &candidate) {
                    return candidate.name == uniform_name;
                });
            if (uniform == custom_uniforms.end()) {
                throw std::runtime_error(
                    "--midi-cc target is not defined in library.json: " +
                    uniform_name);
            }
            const std::size_t uniform_index = static_cast<std::size_t>(
                std::distance(custom_uniforms.begin(), uniform));
            const auto duplicate = std::find_if(
                midi_cc_mappings.begin(), midi_cc_mappings.end(),
                [&](const MidiCcMapping &mapping) {
                    return mapping.uniform_index == uniform_index;
                });
            if (duplicate != midi_cc_mappings.end()) {
                throw std::runtime_error(
                    "custom uniform has more than one --midi-cc mapping: " +
                    uniform_name);
            }

            midi_cc_mappings.push_back(
                {channel, controller, uniform_index, uniform_name});
            std::cout << "acmxvk: MIDI "
                      << (channel < 0
                              ? std::string("any channel")
                              : "channel " + std::to_string(channel + 1))
                      << " CC " << controller << " -> " << uniform_name
                      << " [" << uniform->minimum << ", "
                      << uniform->maximum << "]\n";
        }
#endif
    }

#ifdef MIDI_ENABLED
    [[nodiscard]] bool MainWindow::applyMidiCc(const midi::MidiMessage &message) {
        if (message.bytes.size() < 3 ||
            (message.bytes[0] & 0xF0U) != 0xB0U) {
            return false;
        }
        const int channel = message.bytes[0] & 0x0FU;
        const int controller = message.bytes[1] & 0x7FU;
        const int value = message.bytes[2] & 0x7FU;
        bool changed = false;
        for (const MidiCcMapping &mapping : midi_cc_mappings) {
            if (mapping.controller != controller ||
                (mapping.channel >= 0 && mapping.channel != channel)) {
                continue;
            }
            const ShaderManifest::CustomUniform &uniform =
                custom_uniforms[mapping.uniform_index];
            const double normalized = static_cast<double>(value) / 127.0;
            const float mapped = static_cast<float>(
                uniform.minimum + normalized *
                                      (uniform.maximum - uniform.minimum));
            custom_uniform_values[mapping.uniform_index] = mapped;
            changed = true;
            if (options.midi_monitor) {
                std::cout << "acmxvk: MIDI CC " << controller << " -> "
                          << mapping.uniform_name << '=' << mapped << '\n';
            }
        }
        return changed;
    }

    [[nodiscard]] SDL_Keycode MainWindow::midiActionKey(int action) const {
        switch (action) {
        case 262:
#ifdef ACMXVK_WITH_CUDA
            return gpu_filter_engine != nullptr ? SDLK_RIGHT : SDLK_UNKNOWN;
#else
            return SDLK_UNKNOWN;
#endif
        case 263:
#ifdef ACMXVK_WITH_CUDA
            return gpu_filter_engine != nullptr ? SDLK_LEFT : SDLK_UNKNOWN;
#else
            return SDLK_UNKNOWN;
#endif
        case 264:
            return SDLK_DOWN;
        case 265:
            return SDLK_UP;
        case 266:
        case 504:
            return SDLK_PAGEUP;
        case 267:
        case 505:
            return SDLK_PAGEDOWN;
        case 268:
#ifdef AUDIO_ENABLED
            return SDLK_HOME;
#else
            return SDLK_UNKNOWN;
#endif
        case 269:
#ifdef AUDIO_ENABLED
            return SDLK_END;
#else
            return SDLK_UNKNOWN;
#endif
        case 260:
            return SDLK_INSERT;
        case 261:
            return SDLK_DELETE;
        case 500:
            return SDLK_U;
        case 501:
            return SDLK_I;
        case 298:
            return SDLK_F9;
        case 32:
            return SDLK_SPACE;
        case 44:
            return options.enable_3d ? SDLK_COMMA : SDLK_UNKNOWN;
        case 46:
            return options.enable_3d ? SDLK_PERIOD : SDLK_UNKNOWN;
        case 51:
            return options.enable_3d ? SDLK_3 : SDLK_UNKNOWN;
        case 52:
            return SDLK_4;
        case 53:
            return SDLK_5;
        case 54:
            return SDLK_6;
        case 67:
            return options.enable_3d ? SDLK_C : SDLK_UNKNOWN;
        case 79:
            return options.enable_3d ? SDLK_O : SDLK_UNKNOWN;
        case 91:
            return options.enable_3d ? SDLK_MINUS : SDLK_UNKNOWN;
        case 93:
            return options.enable_3d ? SDLK_EQUALS : SDLK_UNKNOWN;
        case 69:
            return options.watermark_text.empty() ? SDLK_UNKNOWN : SDLK_E;
        case 74:
            return SDLK_J;
        case 78:
            return SDLK_N;
        case 75:
            return SDLK_K;
        case 76:
            return SDLK_L;
        case 77:
            return SDLK_M;
        case 80:
            return SDLK_P;
        case 81:
#ifdef AUDIO_ENABLED
            return SDLK_Q;
#else
            return SDLK_UNKNOWN;
#endif
        case 70:
            return SDLK_F;
        case 73:
            return SDLK_I;
        case 84:
            return SDLK_T;
        case 85:
            return SDLK_U;
        case 86:
            return options.enable_3d ? SDLK_V : SDLK_UNKNOWN;
        case 88:
            return options.enable_3d ? SDLK_X : SDLK_UNKNOWN;
        case 89:
            return SDLK_Y;
        case 90:
            return SDLK_Z;
        default:
            return SDLK_UNKNOWN;
        }
    }

    [[nodiscard]] bool MainWindow::isMidiSliderMapping(
        const midi::MidiMapping &mapping) const {
        return mapping.primary_action >= 600 &&
               mapping.primary_action <= 606 &&
               mapping.primary_action % 2 == 0 &&
               mapping.secondary_action == mapping.primary_action + 1;
    }

    [[nodiscard]] bool MainWindow::usesMidiDeltaDirection(
        const midi::MidiMapping &mapping) {
        return mapping.primary_action == 506 ||
               mapping.primary_action == 508 ||
               mapping.primary_action == 512;
    }

    [[nodiscard]] bool MainWindow::isMidiModelAction(int action) const {
        return options.enable_3d && action >= 506 && action <= 515;
    }

    [[nodiscard]] bool MainWindow::isMidiMappingSupported(
        const midi::MidiMapping &mapping) const {
        if (isMidiSliderMapping(mapping)) {
            const int slider = (mapping.primary_action - 600) / 2;
            return midi_slider_uniform_indices[slider] >= 0;
        }
        if (mapping.secondary_action == 0) {
            return isMidiModelAction(mapping.primary_action) ||
                   midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
        }
        const bool primary_supported =
            isMidiModelAction(mapping.primary_action) ||
            midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
        const bool secondary_supported =
            isMidiModelAction(mapping.secondary_action) ||
            midiActionKey(mapping.secondary_action) != SDLK_UNKNOWN;
        return primary_supported && secondary_supported;
    }

    [[nodiscard]] std::string_view MainWindow::midiActionName(int action) const {
        switch (action) {
        case 262:
            return "select next CUDA filter";
        case 263:
            return "select previous CUDA filter";
        case 264:
            return "next shader or playlist node";
        case 265:
            return "previous shader or playlist node";
        case 266:
        case 504:
            return "increase shader time speed";
        case 267:
        case 505:
            return "decrease shader time speed";
        case 268:
            return "toggle audio delta-time scaling";
        case 269:
            return "toggle spectrum sensitivity scaling";
        case 260:
            return "increase audio sensitivity";
        case 261:
            return "decrease audio sensitivity";
        case 500:
            return "step shader time forward";
        case 501:
            return "step shader time backward";
        case 506:
            return "rotate model X forward";
        case 507:
            return "rotate model X backward";
        case 508:
            return "rotate model Y forward";
        case 509:
            return "rotate model Y backward";
        case 510:
            return "increase 3D manual rotation speed";
        case 511:
            return "decrease 3D manual rotation speed";
        case 512:
            return "rotate model Z forward";
        case 513:
            return "rotate model Z backward";
        case 514:
            return "increase model scale";
        case 515:
            return "decrease model scale";
        case 298:
            return "toggle runtime HUD";
        case 32:
            return "toggle shader bypass";
        case 44:
            return "decrease 3D view rotation speed";
        case 46:
            return "increase 3D view rotation speed";
        case 51:
            return "toggle 2D/3D rendering";
        case 52:
            return "take TIFF snapshot";
        case 53:
            return "take WebP snapshot";
        case 54:
            return "take raw RGBA snapshot";
        case 67:
            return "toggle 3D wave effect";
        case 79:
            return "toggle 3D scale oscillation";
        case 91:
            return "decrease model scale";
        case 93:
            return "increase model scale";
        case 69:
            return "toggle watermark";
        case 74:
            return "toggle random autopilot";
        case 78:
            return "toggle random autopilot crossfade";
        case 75:
            return "toggle shader lock";
        case 76:
            return "toggle rendering freeze";
        case 77:
            return "toggle multipass";
        case 80:
            return "toggle playlist or input pause";
        case 81:
            return "toggle audio-reactive shader time";
        case 70:
            return "toggle fullscreen";
        case 73:
            return "step shader time backward";
        case 84:
            return "toggle shader time";
        case 85:
            return "step shader time forward";
        case 86:
            return "toggle 3D view rotation";
        case 88:
            return "reset model view";
        case 89:
            return "toggle sequential autopilot";
        case 90:
            return "take screenshot";
        default:
            return "unsupported action";
        }
    }

    void MainWindow::dispatchMidiModelAction(int action) {
        if (!model_initialized || !isMidiModelAction(action)) {
            return;
        }

        const auto rotate = [](float &degrees, float amount) {
            degrees = std::fmod(degrees + amount, 360.0F);
            if (degrees < 0.0F) {
                degrees += 360.0F;
            }
        };
        switch (action) {
        case 506:
            rotate(model_rotation_x_degrees,
                   model_camera_rotation_speed * 0.3F);
            break;
        case 507:
            rotate(model_rotation_x_degrees,
                   model_camera_rotation_speed * -0.33F);
            break;
        case 508:
            rotate(model_rotation_y_degrees,
                   model_camera_rotation_speed * 0.3F);
            break;
        case 509:
            rotate(model_rotation_y_degrees,
                   model_camera_rotation_speed * -0.3F);
            break;
        case 510:
            model_camera_rotation_speed = std::clamp(
                model_camera_rotation_speed + 0.5F, 0.5F, 50.0F);
            std::cout << "acmxvk: 3D manual rotation speed "
                      << model_camera_rotation_speed << '\n';
            break;
        case 511:
            model_camera_rotation_speed = std::clamp(
                model_camera_rotation_speed - 0.5F, 0.5F, 50.0F);
            std::cout << "acmxvk: 3D manual rotation speed "
                      << model_camera_rotation_speed << '\n';
            break;
        case 512:
            rotate(model_rotation_z_degrees,
                   model_camera_rotation_speed * 0.3F);
            break;
        case 513:
            rotate(model_rotation_z_degrees,
                   model_camera_rotation_speed * -0.3F);
            break;
        case 514:
            adjustModelScale(0.05F);
            break;
        case 515:
            adjustModelScale(-0.05F);
            break;
        default:
            break;
        }
    }

    void MainWindow::dispatchMidiAction(int action) {
        if (isMidiModelAction(action)) {
            if (options.midi_monitor) {
                std::cout << "acmxvk: MIDI action: "
                          << midiActionName(action) << '\n';
            }
            dispatchMidiModelAction(action);
            return;
        }
        const SDL_Keycode key = midiActionKey(action);
        if (key == SDLK_UNKNOWN) {
            return;
        }
        if (options.midi_monitor) {
            std::cout << "acmxvk: MIDI action: " << midiActionName(action)
                      << '\n';
        }
        SDL_Event midi_event{};
        midi_event.type = SDL_EVENT_KEY_DOWN;
        midi_event.key.type = SDL_EVENT_KEY_DOWN;
        midi_event.key.key = key;
        midi_event.key.mod =
            action == 91 || action == 93 ? SDL_KMOD_SHIFT
                                         : SDL_KMOD_NONE;
        midi_event.key.repeat = false;
        event(midi_event);
    }

    [[nodiscard]] bool MainWindow::setMidiUniform(std::size_t uniform_index, int value,
                                                  std::string_view label) {
        if (uniform_index >= custom_uniforms.size() ||
            uniform_index >= custom_uniform_values.size()) {
            return false;
        }
        const ShaderManifest::CustomUniform &uniform =
            custom_uniforms[uniform_index];
        const double normalized = static_cast<double>(value) / 127.0;
        const float mapped = static_cast<float>(
            uniform.minimum +
            normalized * (uniform.maximum - uniform.minimum));
        custom_uniform_values[uniform_index] = mapped;
        if (options.midi_monitor) {
            std::cout << "acmxvk: MIDI " << label << " -> "
                      << uniform.name << '=' << mapped << '\n';
        }
        return true;
    }

    [[nodiscard]] bool MainWindow::applyMidiMap(const midi::MidiMessage &message) {
        if (message.bytes.size() < 3) {
            return false;
        }
        bool changed = false;
        for (std::size_t index = 0; index < midi_action_mappings.size();
             ++index) {
            const midi::MidiMapping &mapping = midi_action_mappings[index];
            if (message.bytes[0] != mapping.status ||
                message.bytes[1] != mapping.data1) {
                continue;
            }
            const int value = message.bytes[2] & 0x7FU;
            if (mapping.secondary_action == 0) {
                if (message.bytes[2] == mapping.data2) {
                    dispatchMidiAction(mapping.primary_action);
                }
                continue;
            }

            if (isMidiSliderMapping(mapping)) {
                const int slider = (mapping.primary_action - 600) / 2;
                const int uniform_index =
                    midi_slider_uniform_indices[slider];
                if (uniform_index >= 0) {
                    changed =
                        setMidiUniform(
                            static_cast<std::size_t>(uniform_index), value,
                            "Slider " + std::to_string(slider + 1)) ||
                        changed;
                }
                continue;
            }

            MidiKnobState &state = midi_knob_states[index];
            if (usesMidiDeltaDirection(mapping) &&
                value != state.previous_value) {
                state.direction_action =
                    value > state.previous_value
                        ? mapping.primary_action
                        : mapping.secondary_action;
            }
            state.previous_value = value;
            state.value = value;
            state.active = value != 64;
            if (!state.active) {
                state.frame_counter = 0;
            }
        }
        return changed;
    }

    void MainWindow::dispatchMidiKnobs() {
        for (std::size_t index = 0; index < midi_action_mappings.size();
             ++index) {
            const midi::MidiMapping &mapping = midi_action_mappings[index];
            MidiKnobState &state = midi_knob_states[index];
            if (!state.active || mapping.secondary_action == 0 ||
                isMidiSliderMapping(mapping) ||
                !isMidiMappingSupported(mapping)) {
                continue;
            }

            const int distance = std::abs(state.value - 64);
            const int frame_skip =
                std::max(1, 17 - (distance * 16 / 63));
            if (++state.frame_counter < frame_skip) {
                continue;
            }
            state.frame_counter = 0;
            int action = state.value > 64
                             ? mapping.primary_action
                             : mapping.secondary_action;
            if (usesMidiDeltaDirection(mapping)) {
                action = state.direction_action;
                if (action == 0) {
                    continue;
                }
            }
            dispatchMidiAction(action);
        }
    }
#endif

    void MainWindow::uploadCustomUniforms() {
        if (frame_sprite != nullptr) {
            frame_sprite->setCustomUniforms(custom_uniform_values);
        }
        for (mxvk::VK_Sprite *sprite : post_process_sprites) {
            sprite->setCustomUniforms(custom_uniform_values);
        }
    }

    void MainWindow::pollMidi() {
#ifdef MIDI_ENABLED
        if (midi_input == nullptr || !midi_input->is_open()) {
            return;
        }
        const std::vector<midi::MidiMessage> messages =
            midi_input->poll_messages();
        bool custom_uniforms_changed = false;
        for (const midi::MidiMessage &message : messages) {
            custom_uniforms_changed =
                applyMidiCc(message) || custom_uniforms_changed;
            custom_uniforms_changed =
                applyMidiMap(message) || custom_uniforms_changed;
        }
        dispatchMidiKnobs();
        if (custom_uniforms_changed) {
            uploadCustomUniforms();
        }
        if (options.midi_monitor) {
            for (const midi::MidiMessage &message : messages) {
                std::ostringstream text;
                text << "acmxvk: MIDI #" << message.sequence << " +"
                     << std::fixed << std::setprecision(6)
                     << message.delta_seconds << "s [";
                for (std::size_t index = 0; index < message.bytes.size();
                     ++index) {
                    if (index > 0) {
                        text << ' ';
                    }
                    text << std::hex << std::uppercase << std::setfill('0')
                         << std::setw(2)
                         << static_cast<unsigned int>(message.bytes[index]);
                }
                text << ']';
                std::cout << text.str() << '\n';
            }
        }
        const std::uint64_t dropped = midi_input->dropped_message_count();
        if (dropped != observed_midi_drops) {
            std::cerr << "acmxvk: MIDI queue dropped " << dropped
                      << " message(s) total\n";
            observed_midi_drops = dropped;
        }
#endif
    }

    void MainWindow::openAudio() {
        if (!options.enable_audio) {
            return;
        }
#ifdef AUDIO_ENABLED
        audio_engine = std::make_unique<audio::AudioEngine>();
        audio_engine->set_sensitivity(
            static_cast<float>(options.audio_sensitivity));
        if (!options.audio_file.empty()) {
            file_audio_source = std::make_unique<audio::FileAudioSource>();
            if (!file_audio_source->open(options.audio_file)) {
                if (options.use_source_audio) {
                    std::cerr
                        << "acmxvk: source video has no decodable audio "
                           "track; continuing with silent audio-reactive "
                           "values";
                    if (options.audio_pass_through) {
                        std::cerr << " and pass-through disabled";
                    }
                    std::cerr << '\n';
                    file_audio_source.reset();
                    return;
                }
                throw std::runtime_error("could not decode --audio-file: " +
                                         options.audio_file);
            }
            if (options.use_source_audio) {
                std::cout << "acmxvk: source video audio drives shader "
                             "reactivity\n";
            }
            file_audio_source->set_repeat(options.audio_repeat);
            if (options.audio_pass_through &&
                !file_audio_source->enable_output(
                    options.audio_output_device,
                    static_cast<float>(options.audio_pass_through_gain))) {
                std::cerr << "acmxvk: file audio output could not be "
                             "initialized; continuing with silent analysis\n";
            }
            resetAudioWarmup();
            return;
        }
        const audio::AudioStreamConfig config{
            static_cast<unsigned int>(options.audio_channels),
            static_cast<float>(options.audio_sensitivity),
            options.audio_input_device,
            options.audio_output_device,
            options.audio_pass_through,
            static_cast<float>(options.audio_pass_through_gain),
            static_cast<float>(options.audio_recording_gain),
        };
        if (!audio_engine->open(config)) {
            std::cerr << "acmxvk: audio input could not be initialized; "
                         "continuing with zero-valued audio metrics\n";
            audio_engine.reset();
        } else {
            resetAudioWarmup();
        }
#endif
    }

    void MainWindow::start_requested_audio_recording() {
        if (options.record_audio_file.empty()) {
            return;
        }
#ifdef AUDIO_ENABLED
        if (audio_engine == nullptr || file_audio_source != nullptr ||
            !audio_engine->is_open()) {
            throw std::runtime_error(
                "--record-audio requires an active live audio input");
        }
        if (!options.output_file.empty() && !options.png_output) {
            return;
        }
        if (!audio_engine->is_recording() &&
            !audio_engine->start_recording()) {
            throw std::runtime_error(
                "could not start standalone microphone recording");
        }
#endif
    }

    void MainWindow::adjustAudioSensitivity(float amount) {
#ifdef AUDIO_ENABLED
        if (audioSourceOpen()) {
            audio_engine->set_sensitivity(audio_engine->sensitivity() + amount);
            options.audio_sensitivity = audio_engine->sensitivity();
            std::cout << "acmxvk: audio sensitivity "
                      << options.audio_sensitivity << '\n';
            return;
        }
#else
        static_cast<void>(amount);
#endif
        std::cout << "acmxvk: audio input is not active\n";
    }

    [[nodiscard]] bool MainWindow::audioSourceOpen() const {
#ifdef AUDIO_ENABLED
        return audio_engine != nullptr &&
               (audio_engine->is_open() ||
                (file_audio_source != nullptr && file_audio_source->is_open()));
#else
        return false;
#endif
    }

    void MainWindow::startLiveAudioRecordingIfNeeded() {
#ifdef AUDIO_ENABLED
        if (audio_engine == nullptr || file_audio_source != nullptr ||
            !audio_engine->is_open() || audio_engine->is_recording() ||
            !writer.is_open() || options.png_output ||
            (options.copy_audio && !options.mute_output &&
             options.record_audio_file.empty())) {
            return;
        }
        if (!audio_engine->start_recording()) {
            std::cerr << "acmxvk: could not start live audio recording; "
                         "continuing with video-only output\n";
        }
#endif
    }

    void MainWindow::startMediaTimelineIfReady() {
        if (media_timeline_started || !source_frame_received) {
            return;
        }
        media_timeline_started = true;
        hud_session_start = std::chrono::steady_clock::now();
        hud_fps_last_tick = hud_session_start;
        hud_fps_frame_count = 0;
        source_playback_clock_start = hud_session_start;
        source_playback_pause_start = {};
        source_playback_paused_duration = {};
        source_playback_clock_paused = false;
#ifdef AUDIO_ENABLED
        resetAudioWarmup();
#endif
        startLiveAudioRecordingIfNeeded();
        std::cout << "acmxvk: media timeline started on first source frame\n";
    }

    void MainWindow::setSourcePlaybackClockPaused(bool paused) {
        if (!options.use_source_fps || source_kind != SourceKind::Video ||
            !media_timeline_started ||
            paused == source_playback_clock_paused) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (paused) {
            source_playback_pause_start = now;
        } else {
            source_playback_paused_duration +=
                now - source_playback_pause_start;
        }
        source_playback_clock_paused = paused;
    }

    [[nodiscard]] bool MainWindow::mediaClockSeconds(double &seconds) const {
#ifdef AUDIO_ENABLED
        if (file_audio_source != nullptr &&
            file_audio_source->has_output_clock()) {
            seconds = file_audio_source->playback_time();
            return true;
        }
        if ((!options.copy_audio || options.mute_output) && writer.is_open() &&
            audio_engine != nullptr && file_audio_source == nullptr &&
            audio_engine->is_recording()) {
            seconds = audio_engine->recording_time();
            return true;
        }
#endif
        if (source_kind == SourceKind::Camera &&
            media_timeline_started) {
            seconds = hudWallElapsedSeconds();
            return true;
        }
        if (options.use_source_fps && source_kind == SourceKind::Video &&
            media_timeline_started) {
            const auto clock_end = source_playback_clock_paused
                                       ? source_playback_pause_start
                                       : std::chrono::steady_clock::now();
            const auto active_time =
                clock_end - source_playback_clock_start -
                source_playback_paused_duration;
            seconds = std::max(
                0.0, std::chrono::duration<double>(active_time).count());
            return true;
        }
        seconds = 0.0;
        return false;
    }
    // Shader discovery, custom uniforms, interface IPC, and playlists.
    void MainWindow::loadShaders() {
        if (!options.fragment_shader.empty() ||
            !options.compute_shader.empty()) {
            const bool compute = !options.compute_shader.empty();
            const fs::path shader = fs::absolute(
                                        compute ? options.compute_shader : options.fragment_shader)
                                        .lexically_normal();
            const std::string label =
                compute ? "compute shader" : "fragment shader";
            if (shader.extension() != ".spv" ||
                !fs::is_regular_file(shader)) {
                throw std::runtime_error(
                    label + " is not a readable .spv file: " +
                    shader.string());
            }
            input::validate_spirv_file(shader, label);
            const mxvk::ShaderModuleInfo module_info =
                mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
            const mxvk::ShaderStage expected_stage =
                compute ? mxvk::ShaderStage::Compute
                        : mxvk::ShaderStage::Fragment;
            if (module_info.stage != expected_stage) {
                throw std::runtime_error(
                    label + " SPIR-V entry point has the wrong shader stage: " +
                    shader.string());
            }
            recordShaderResources(module_info, "shader");
            shaders.push_back(shader);
            return;
        }
        if (options.shader_directory.empty()) {
            return;
        }

        shader_library_directory =
            fs::absolute(options.shader_directory).lexically_normal();
        const ShaderManifest manifest =
            loadShaderManifest(shader_library_directory);
        shader_manifest_path = manifest.path;
        custom_uniforms = manifest.custom_uniforms;
        applyCustomUniformOverrides();
        for (const std::string &entry : manifest.entries) {
            const fs::path shader =
                resolveShaderManifestEntry(shader_library_directory, entry);
            if (!shader.empty()) {
                input::validate_spirv_file(shader,
                                           "shader manifest entry");
                const mxvk::ShaderModuleInfo module_info =
                    mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
                recordShaderResources(module_info, "shader library");
                shaders.push_back(shader);
            }
        }
        std::sort(shaders.begin(), shaders.end(), [](const fs::path &left, const fs::path &right) {
            std::string left_text = left.generic_string();
            std::string right_text = right.generic_string();
            std::transform(left_text.begin(), left_text.end(), left_text.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            std::transform(right_text.begin(), right_text.end(), right_text.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            return left_text < right_text;
        });
        if (shaders.empty()) {
            throw std::runtime_error("shader manifest contains no readable SPIR-V files: " +
                                     shader_manifest_path.string());
        }
        std::cout << "acmxvk: loaded " << shaders.size() << " shaders from "
                  << shader_manifest_path.string() << '\n';
        printCustomUniforms();

        if (!options.shader_file.empty()) {
            const auto selected = std::find_if(
                shaders.begin(), shaders.end(), [&](const fs::path &path) {
                    fs::path requested(options.shader_file);
                    if (requested.extension() != ".spv") {
                        requested.replace_extension(".spv");
                    }
                    return path.filename() == requested.filename() ||
                           path.lexically_relative(shader_library_directory) == requested;
                });
            if (selected == shaders.end()) {
                throw std::runtime_error("shader file is not listed in the manifest: " +
                                         options.shader_file);
            }
            shader_index = static_cast<std::size_t>(std::distance(shaders.begin(), selected));
        } else {
            const int count = static_cast<int>(shaders.size());
            const int wrapped_index = ((options.shader_index % count) + count) % count;
            shader_index = static_cast<std::size_t>(wrapped_index);
        }
    }

    void MainWindow::applyCustomUniformOverrides() {
        for (const std::string &override_text :
             options.custom_uniform_overrides) {
            const std::size_t separator = override_text.find('=');
            if (separator == std::string::npos || separator == 0 ||
                separator + 1 >= override_text.size()) {
                throw std::runtime_error(
                    "--uniform requires name=value: " + override_text);
            }
            const std::string name = trim(override_text.substr(0, separator));
            const double value = parseNumber(
                trim(override_text.substr(separator + 1)), "--uniform");
            const auto match = std::find_if(
                custom_uniforms.begin(), custom_uniforms.end(),
                [&](const ShaderManifest::CustomUniform &uniform) {
                    return uniform.name == name;
                });
            if (match == custom_uniforms.end()) {
                throw std::runtime_error(
                    "custom uniform is not defined in library.json: " + name);
            }
            match->value = std::clamp(value, match->minimum, match->maximum);
        }

        custom_uniform_values.clear();
        custom_uniform_values.reserve(custom_uniforms.size());
        for (const ShaderManifest::CustomUniform &uniform : custom_uniforms) {
            custom_uniform_values.push_back(static_cast<float>(uniform.value));
        }
    }

    void MainWindow::printCustomUniforms() const {
        if (custom_uniforms.empty()) {
            return;
        }
        constexpr std::string_view COMPONENTS = "xyzw";
        std::cout << "acmxvk: custom uniforms (binding 1):\n";
        for (std::size_t index = 0; index < custom_uniforms.size(); ++index) {
            const ShaderManifest::CustomUniform &uniform = custom_uniforms[index];
            std::cout << "  " << uniform.name << '=' << uniform.value
                      << " -> custom_uniforms[" << (index / 4) << "]."
                      << COMPONENTS[index % 4] << '\n';
        }
    }

    [[nodiscard]] std::string MainWindow::currentShader() const {
        return shaders.empty() ? std::string{} : shaders[shader_index].string();
    }

    [[nodiscard]] bool MainWindow::historyCacheEnabled() const {
        return options.enable_texture_cache || shader_history_required;
    }

    void MainWindow::recordShaderResources(const mxvk::ShaderModuleInfo &module_info,
                                           std::string_view source) {
        if (module_info.usesHistoryTexture &&
            !shader_history_required) {
            shader_history_required = true;
            std::cout << "acmxvk: enabled shared history for " << source
                      << " binding 2\n";
        }
        if (module_info.usesSpectrumTexture &&
            !shader_spectrum_required) {
            shader_spectrum_required = true;
            std::cout << "acmxvk: enabled spectrum descriptor for " << source
                      << " binding 3\n";
        }
        if (module_info.usesSpectrumHistoryTexture &&
            !shader_spectrum_history_required) {
            shader_spectrum_history_required = true;
            if (options.audio_buffers == 0) {
                options.audio_buffers = 8;
            }
            std::cout << "acmxvk: enabled " << options.audio_buffers
                      << " spectrum-history layers for " << source
                      << " binding 4\n";
        }
    }

    [[nodiscard]] std::uint32_t MainWindow::spectrumBinCount() const {
#ifdef AUDIO_ENABLED
        return audio::AudioEngine::spectrum_bin_count();
#else
        return COMPATIBILITY_SPECTRUM_BIN_COUNT;
#endif
    }

    [[nodiscard]] bool MainWindow::spectrumTextureEnabledForShaders() const {
#ifdef AUDIO_ENABLED
        return true;
#else
        return shader_spectrum_required;
#endif
    }

    [[nodiscard]] bool MainWindow::spectrumHistoryEnabledForShaders() const {
        return options.audio_buffers > 0;
    }

    void MainWindow::initialize_interface_control() {
        if (!options.interface_shm) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < interface_next_connect_attempt) {
            return;
        }
        interface_next_connect_attempt = now + std::chrono::seconds(2);
        if (!interface_client.open()) {
            return;
        }

        InterfaceState state;
        if (!interface_client.read(state)) {
            if (!interface_connection_warning_reported) {
                std::cerr << "acmxvk: could not read compatible interface "
                             "control state; retrying\n";
                interface_connection_warning_reported = true;
            }
            interface_client.close();
            return;
        }
        const bool reconnected = interface_connection_warning_reported;
        interface_connection_warning_reported = false;
        interface_last_sequence = state.sequence;
        apply_interface_shader_selection(state.selected_shader_name);
        apply_interface_multipass_state(state.multipass);
        apply_interface_uniform_values(state.uniform_values);
        apply_interface_playback_state(state.playback, false);
        apply_interface_overlay_state(state.overlay, false);
        apply_interface_gpu_filter_state(state.gpu_filters, false);
        interface_last_audio_file_sequence =
            state.audio_file.request_sequence;
        interface_last_reload_sequence = state.reload.request_sequence;
        std::cout << "acmxvk: interface live shader, multipass, playback, "
                     "overlay, GPU-filter, and audio-file control enabled"
                  << (reconnected ? " (reconnected)" : "") << '\n';
    }

    void MainWindow::sync_interface_control() {
        if (!options.interface_shm) {
            return;
        }
        if (!interface_client.is_open()) {
            initialize_interface_control();
            return;
        }

        InterfaceState state;
        if (!interface_client.read(state)) {
            if (!interface_connection_warning_reported) {
                std::cerr << "acmxvk: interface control connection lost; "
                             "retrying\n";
                interface_connection_warning_reported = true;
            }
            interface_client.close();
            interface_next_connect_attempt =
                std::chrono::steady_clock::now() + std::chrono::seconds(2);
            return;
        }
        if (state.sequence == interface_last_sequence) {
            return;
        }
        interface_last_sequence = state.sequence;
        apply_interface_shader_selection(state.selected_shader_name);
        apply_interface_multipass_state(state.multipass);
        apply_interface_uniform_values(state.uniform_values);
        apply_interface_playback_state(state.playback, true);
        apply_interface_overlay_state(state.overlay, true);
        apply_interface_gpu_filter_state(state.gpu_filters, true);
        if (state.audio_file.request_sequence !=
            interface_last_audio_file_sequence) {
            interface_last_audio_file_sequence =
                state.audio_file.request_sequence;
            apply_interface_audio_file_state(state.audio_file);
        }
        if (state.reload.request_sequence !=
            interface_last_reload_sequence) {
            interface_last_reload_sequence = state.reload.request_sequence;
            apply_interface_shader_reload(state.reload);
        }
    }

    void MainWindow::apply_interface_playback_state(
        const InterfacePlaybackState &requested, bool announce) {
        if (options.repeat != requested.repeat) {
            options.repeat = requested.repeat;
            if (announce) {
                std::cout << "acmxvk: interface video repeat "
                          << (options.repeat ? "enabled" : "disabled")
                          << '\n';
            }
        }
        if (options.normalized_time != requested.normalized_time) {
            options.normalized_time = requested.normalized_time;
            if (announce) {
                std::cout << "acmxvk: interface normalized time "
                          << (options.normalized_time ? "enabled"
                                                      : "disabled")
                          << '\n';
            }
        }
    }

    void MainWindow::apply_interface_overlay_state(const InterfaceOverlayState &requested,
                                                   bool announce) {
        if (options.display_filter != requested.display_filter) {
            options.display_filter = requested.display_filter;
            if (announce) {
                std::cout << "acmxvk: interface display-filter overlay "
                          << (options.display_filter ? "enabled"
                                                     : "disabled")
                          << '\n';
            }
        }

        try {
            input::validate_string(requested.watermark_text,
                                   input::StringKind::DisplayText,
                                   "interface watermark", true);
        } catch (const std::exception &error) {
            std::cerr << "acmxvk: rejected interface watermark: "
                      << error.what() << '\n';
            return;
        }

        const bool was_enabled =
            watermark_enabled && !options.watermark_text.empty();
        const bool requested_enabled =
            requested.watermark_enabled &&
            !requested.watermark_text.empty();
        const bool changed =
            watermark_enabled != requested_enabled ||
            options.watermark_text != requested.watermark_text ||
            options.watermark_color != requested.watermark_color;
        if (!changed) {
            return;
        }

        options.watermark_text = requested.watermark_text;
        options.watermark_color = requested.watermark_color;
        watermark_enabled = requested_enabled;
        if (!was_enabled && watermark_enabled) {
            counter_disabled = true;
        }
        if (announce) {
            std::cout << "acmxvk: interface watermark "
                      << (watermark_enabled ? "enabled" : "disabled");
            if (watermark_enabled) {
                std::cout << " (color="
                          << static_cast<int>(options.watermark_color[0])
                          << ','
                          << static_cast<int>(options.watermark_color[1])
                          << ','
                          << static_cast<int>(options.watermark_color[2])
                          << ')';
            }
            std::cout << '\n';
        }
    }

    void MainWindow::apply_interface_gpu_filter_state(
        const InterfaceGpuFilterState &requested, bool announce) {
#ifdef ACMXVK_WITH_CUDA
        const bool requested_enabled =
            requested.enabled && !requested.filter_indices.empty();
        const bool currently_enabled = gpu_filter_engine != nullptr;
        const std::vector<int> effective_indices =
            requested_enabled ? requested.filter_indices
                              : std::vector<int>{};
        if (requested_enabled == currently_enabled &&
            options.gpu_filter_indices == effective_indices &&
            (!requested_enabled ||
             options.gpu_frame_buffer_size ==
                 requested.frame_buffer_size)) {
            return;
        }

        if (requested.enabled && requested.filter_indices.empty()) {
            std::cerr << "acmxvk: rejected enabled interface GPU-filter "
                         "state without any filter indices\n";
            return;
        }

        std::unique_ptr<gpu::FilterEngine> replacement;
        if (requested_enabled) {
            try {
                replacement = std::make_unique<gpu::FilterEngine>(
                    requested.filter_indices,
                    requested.frame_buffer_size);
            } catch (const std::exception &error) {
                std::cerr
                    << "acmxvk: rejected interface GPU-filter state: "
                    << error.what() << '\n';
                return;
            }
        }

        gpu_filter_engine = std::move(replacement);
        options.gpu_filter_indices = effective_indices;
        if (requested_enabled) {
            options.gpu_frame_buffer_size = requested.frame_buffer_size;
        }

        if (frame_sprite != nullptr &&
            source_kind == SourceKind::Graphic && !graphic_rgba.empty()) {
            uploadInputFrame(graphic_rgba);
            if (history_initialized) {
                updateHistoryFrame(graphic_rgba);
                history_delay_counter = 0;
            }
        }

        if (announce) {
            std::cout << "acmxvk: interface CUDA filter chain "
                      << (requested_enabled ? "enabled" : "disabled");
            if (requested_enabled) {
                std::cout << " (" << requested.filter_indices.size()
                          << " filters, " << requested.frame_buffer_size
                          << " history frames)";
            }
            std::cout << '\n';
        }
#else
        if (announce && requested.enabled) {
            std::cerr << "acmxvk: ignored interface GPU-filter state: this "
                         "build does not include acidcam-gpu\n";
        }
#endif
    }

    void MainWindow::apply_interface_audio_file_state(
        const InterfaceAudioFileState &requested) {
#ifdef AUDIO_ENABLED
        if (file_audio_source == nullptr || audio_engine == nullptr) {
            std::cerr
                << "acmxvk: ignored live audio-file change because this "
                   "process was not started in audio-file mode\n";
            return;
        }
        if (requested.path.empty()) {
            std::cerr
                << "acmxvk: rejected empty interface audio-file request\n";
            return;
        }

        auto replacement = std::make_unique<audio::FileAudioSource>();
        try {
            if (!replacement->open(requested.path)) {
                std::cerr << "acmxvk: could not switch file audio to: "
                          << requested.path << '\n';
                return;
            }
        } catch (const std::exception &error) {
            std::cerr << "acmxvk: rejected interface audio-file request: "
                      << error.what() << '\n';
            return;
        }

        replacement->set_repeat(requested.repeat);
        if (requested.pass_through &&
            !replacement->enable_output(
                requested.output_device,
                static_cast<float>(options.audio_pass_through_gain))) {
            std::cerr
                << "acmxvk: live audio-file output could not be opened; "
                   "continuing with visual reactivity only\n";
        }

        file_audio_source->stop_output();
        file_audio_source = std::move(replacement);
        options.audio_file = requested.path;
        options.audio_output_device = requested.output_device;
        options.audio_pass_through = requested.pass_through;
        options.audio_trunc = requested.trunc;
        options.audio_repeat = requested.repeat;
        audio_engine->reset();
        resetAudioWarmup();
        std::cout << "acmxvk: switched file audio to: "
                  << file_audio_source->path() << " (repeat="
                  << (options.audio_repeat ? "on" : "off")
                  << ", trunc=" << (options.audio_trunc ? "on" : "off")
                  << ", pass-through="
                  << (options.audio_pass_through ? "on" : "off")
                  << ")\n";
#else
        static_cast<void>(requested);
        std::cerr << "acmxvk: ignored interface audio-file request: this "
                     "build does not include audio support\n";
#endif
    }

    void MainWindow::apply_interface_shader_reload(
        const InterfaceReloadState &requested) {
        if (requested.path.empty()) {
            std::cerr
                << "acmxvk: rejected empty interface shader reload\n";
            return;
        }

        try {
            input::validate_string(requested.path,
                                   input::StringKind::Path,
                                   "interface shader reload path");
        } catch (const std::exception &error) {
            std::cerr << "acmxvk: rejected interface shader reload: "
                      << error.what() << '\n';
            return;
        }

        std::error_code error;
        const fs::path requested_path =
            fs::weakly_canonical(requested.path, error);
        if (error || requested_path.empty() ||
            !fs::is_regular_file(requested_path)) {
            std::cerr << "acmxvk: interface shader reload file is not "
                         "readable: "
                      << requested.path << '\n';
            return;
        }

        const auto shader_match = std::find_if(
            shaders.begin(), shaders.end(),
            [&](const fs::path &shader) {
                std::error_code shader_error;
                const fs::path canonical_shader =
                    fs::weakly_canonical(shader, shader_error);
                return !shader_error &&
                       canonical_shader == requested_path;
            });
        if (shader_match == shaders.end()) {
            std::cerr << "acmxvk: interface shader reload is outside the "
                         "active runtime library: "
                      << requested_path.string() << '\n';
            return;
        }

        mxvk::ShaderModuleInfo module_info;
        try {
            input::validate_spirv_file(requested_path,
                                       "interface shader reload");
            module_info = mxvk::inspect_spirv(
                mxvk::load_spv(requested_path.string()));
        } catch (const std::exception &reload_error) {
            std::cerr << "acmxvk: rejected compiled shader reload: "
                      << reload_error.what() << '\n';
            return;
        }

        const bool history_before = shader_history_required;
        const bool spectrum_before = shader_spectrum_required;
        const bool spectrum_history_before =
            shader_spectrum_history_required;
        recordShaderResources(module_info, "live shader reload");
        const bool resources_grew =
            history_before != shader_history_required ||
            spectrum_before != shader_spectrum_required ||
            spectrum_history_before !=
                shader_spectrum_history_required;

        const std::vector<fs::path> active_pipeline =
            activeShaderPipeline();
        const bool active =
            std::find(active_pipeline.begin(), active_pipeline.end(),
                      *shader_match) != active_pipeline.end();
        if (active && frame_sprite != nullptr) {
            if (model_effect_shader == *shader_match) {
                model_effect_shader.clear();
            }
            if (resources_grew) {
                initializeSprite();
            } else {
                beginCrossfade();
                applyShaderPipeline();
            }
            std::cout << "acmxvk: live reloaded active "
                      << (module_info.stage == mxvk::ShaderStage::Compute
                              ? "compute"
                              : "fragment")
                      << " shader: " << requested_path.string() << '\n';
        } else {
            std::cout << "acmxvk: live compiled shader ready for its next "
                         "use: "
                      << requested_path.string() << '\n';
        }
    }

    void MainWindow::apply_interface_multipass_state(
        const InterfaceMultipassState &requested) {
        std::vector<fs::path> requested_passes;
        if (requested.enabled) {
            if (requested.shader_names.empty()) {
                std::cerr << "acmxvk: rejected enabled interface multipass "
                             "state without any shader passes\n";
                return;
            }
            requested_passes.reserve(requested.shader_names.size());
            for (const std::string &name : requested.shader_names) {
                const fs::path requested_path(name);
                const bool has_parent_reference = std::any_of(
                    requested_path.begin(), requested_path.end(),
                    [](const fs::path &part) { return part == ".."; });
                if (requested_path.is_absolute() || has_parent_reference) {
                    std::cerr << "acmxvk: rejected unsafe interface "
                                 "multipass shader name: "
                              << name << '\n';
                    return;
                }
                const fs::path shader = find_shader_path(
                    shaders, shader_library_directory, name);
                if (shader.empty()) {
                    std::cerr << "acmxvk: interface multipass shader is not "
                                 "in the active library: "
                              << name << '\n';
                    return;
                }
                requested_passes.push_back(shader);
            }
        }

        const bool requested_enabled =
            requested.enabled && !requested_passes.empty();
        if (multipass_enabled == requested_enabled &&
            configured_passes == requested_passes) {
            return;
        }
        if (frame_sprite != nullptr && shader_locked) {
            std::cerr << "acmxvk: interface multipass update ignored while "
                         "shader switching is locked\n";
            return;
        }

        if (frame_sprite != nullptr) {
            beginCrossfade();
        }
        configured_passes = std::move(requested_passes);
        multipass_enabled = requested_enabled;
        if (frame_sprite != nullptr) {
            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
        }

        if (multipass_enabled) {
            std::cout << "acmxvk: interface multipass enabled ("
                      << configured_passes.size() << " passes)";
            for (const fs::path &shader : configured_passes) {
                std::cout << "\n  " << shader.filename().string();
            }
            std::cout << '\n';
        } else {
            std::cout << "acmxvk: interface multipass disabled\n";
        }
    }

    void MainWindow::apply_interface_shader_selection(
        const std::string &requested_name) {
        if (requested_name.empty()) {
            return;
        }

        const fs::path requested(requested_name);
        const bool has_parent_reference =
            std::any_of(requested.begin(), requested.end(),
                        [](const fs::path &part) { return part == ".."; });
        if (requested.is_absolute() || has_parent_reference) {
            std::cerr << "acmxvk: rejected unsafe interface shader name: "
                      << requested_name << '\n';
            return;
        }

        const fs::path shader = find_shader_path(
            shaders, shader_library_directory, requested_name);
        const auto match = std::find(shaders.begin(), shaders.end(), shader);
        if (shader.empty() || match == shaders.end()) {
            std::cerr << "acmxvk: interface shader is not in the active "
                         "library: "
                      << requested_name << '\n';
            return;
        }

        const std::size_t next_index =
            static_cast<std::size_t>(std::distance(shaders.begin(), match));
        if (next_index == shader_index) {
            return;
        }
        if (shader_locked || frame_sprite == nullptr) {
            std::cerr << "acmxvk: interface shader selection ignored while "
                         "shader switching is locked\n";
            return;
        }

        beginCrossfade();
        shader_index = next_index;
        applyShaderPipeline();
        resetShaderTime();
        autopilot_counter = 0;
        std::cout << "acmxvk: interface selected " << activeShaderRole()
                  << ' ' << (shader_index + 1) << '/' << shaders.size()
                  << ": " << currentShader() << '\n';
    }

    void MainWindow::apply_interface_uniform_values(
        const std::vector<InterfaceUniformValue> &uniform_values) {
        if (uniform_values.empty()) {
            return;
        }

        std::size_t changed_count = 0;
        std::size_t ignored_count = 0;
        for (const InterfaceUniformValue &incoming : uniform_values) {
            if (!isValidCustomUniformName(incoming.name) ||
                !std::isfinite(incoming.value)) {
                ++ignored_count;
                continue;
            }
            const auto match = std::find_if(
                custom_uniforms.begin(), custom_uniforms.end(),
                [&](const ShaderManifest::CustomUniform &uniform) {
                    return uniform.name == incoming.name;
                });
            if (match == custom_uniforms.end()) {
                ++ignored_count;
                continue;
            }
            const std::size_t index = static_cast<std::size_t>(
                std::distance(custom_uniforms.begin(), match));
            if (index >= custom_uniform_values.size()) {
                ++ignored_count;
                continue;
            }
            const float value = static_cast<float>(std::clamp(
                static_cast<double>(incoming.value), match->minimum,
                match->maximum));
            if (custom_uniform_values[index] == value) {
                continue;
            }
            custom_uniform_values[index] = value;
            ++changed_count;
        }

        if (changed_count > 0) {
            uploadCustomUniforms();
            std::cout << "acmxvk: interface updated " << changed_count
                      << " custom uniform(s)\n";
        }
        if (ignored_count > 0) {
            std::cerr << "acmxvk: interface ignored " << ignored_count
                      << " unknown or invalid custom uniform(s)\n";
        }
    }

    void MainWindow::loadShaderPasses() {
        for (const int index : options.shader_pass_indices) {
            if (index < 0 || index >= static_cast<int>(shaders.size())) {
                throw std::runtime_error("shader pass index is out of range: " +
                                         std::to_string(index));
            }
            configured_passes.push_back(shaders[static_cast<std::size_t>(index)]);
        }
        for (const std::string &name : options.shader_pass_files) {
            const fs::path shader = find_shader_path(
                shaders, shader_library_directory, name);
            if (shader.empty()) {
                throw std::runtime_error("shader pass file is not listed in the manifest: " +
                                         name);
            }
            configured_passes.push_back(shader);
        }
        multipass_enabled = !configured_passes.empty();
    }

    void MainWindow::loadPlaylist() {
        if (options.playlist_file.empty()) {
            return;
        }
        playlist = load_playlist(options.playlist_file, shaders,
                                 shader_library_directory, std::cerr);
        playlist_enabled = options.enable_playlist;
        std::cout << "acmxvk: playlist loaded "
                  << playlist_shader_count(playlist) << " shaders in "
                  << playlist.size() << " nodes from "
                  << options.playlist_file << '\n';
        logSelectedPlaylistNode("selected");
    }
    // Resource resolution, HUD/watermark drawing, and DNN overlays.
    void MainWindow::resolveConfiguredResourcePaths() {
        const auto resolve = [&](std::string &path,
                                 const fs::path &resource_subdirectory,
                                 std::string_view label) {
            if (path.empty() || fs::is_regular_file(path) ||
                fs::path(path).is_absolute()) {
                return;
            }
            fs::path resolved = find_resource(options, fs::path(path));
            if (resolved.empty()) {
                resolved = find_resource(
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
                options.model_file = default_model_path(options).string();
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

    void MainWindow::initializeOverlayFont() {
        if (counter_disabled && !options.display_filter &&
            options.watermark_text.empty() && !options.interface_shm) {
            return;
        }

        const fs::path font = overlay_font_path(options);
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

    [[nodiscard]] std::string MainWindow::clipOverlayText(std::string text) {
        constexpr std::size_t MAX_OVERLAY_CHARACTERS = 120;
        return input::truncate_utf8(text, MAX_OVERLAY_CHARACTERS);
    }

    [[nodiscard]] const std::vector<fs::path> *MainWindow::activePasses() const {
        if (playlist_enabled && !playlist.empty()) {
            return &playlist[playlist_index].shaders;
        }
        if (multipass_enabled && !configured_passes.empty()) {
            return &configured_passes;
        }
        return nullptr;
    }

    [[nodiscard]] std::string_view MainWindow::activeShaderRole() const {
        const std::vector<fs::path> *passes = activePasses();
        return passes != nullptr && !passes->empty() ? "Post-shader"
                                                     : "Shader";
    }

    [[nodiscard]] std::string MainWindow::activePassDescription() const {
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

    [[nodiscard]] std::string MainWindow::activePlaylistDescription() const {
        if (!playlist_enabled || playlist.empty()) {
            return {};
        }
        std::ostringstream description;
        description << "Playlist [" << (playlist_index + 1) << '/'
                    << playlist.size() << "]: "
                    << playlist[playlist_index].name;
        return clipOverlayText(description.str());
    }

    [[nodiscard]] std::string MainWindow::formatHudTime(double seconds_value) {
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

    void MainWindow::updateWindowTitle(bool force) {
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

    void MainWindow::emitHeadlessProgress(bool complete) {
        if (!options.headless || recording_fps <= 0.0 ||
            output_frame_count == 0U) {
            return;
        }

        std::uint64_t expected_frames = 0U;
        if (options.duration > 0.0) {
            const auto duration_frames = static_cast<std::uint64_t>(
                std::ceil(options.duration * recording_fps));
            expected_frames = std::max<std::uint64_t>(1U, duration_frames);
        }
        if (source_kind == SourceKind::Video &&
            video_duration_seconds > 0.0) {
            const auto source_frames = static_cast<std::uint64_t>(
                std::ceil(video_duration_seconds * recording_fps));
            if (expected_frames == 0U) {
                expected_frames = source_frames;
            } else if (!options.repeat) {
                expected_frames = std::min(expected_frames, source_frames);
            }
        }
        if (complete && expected_frames == 0U) {
            expected_frames = output_frame_count;
        }

        const auto now = std::chrono::steady_clock::now();
        int percent = -1;
        if (expected_frames > 0U) {
            const std::uint64_t processed_frames = complete
                                                       ? expected_frames
                                                       : std::min(
                                                             output_frame_count,
                                                             expected_frames);
            percent = static_cast<int>(
                (static_cast<double>(processed_frames) /
                 static_cast<double>(expected_frames)) *
                100.0);
            if (!complete) {
                percent = std::min(percent, 99);
            }
        }

        const bool percent_changed =
            percent >= 0 && percent > headless_progress_last_percent;
        const bool time_elapsed =
            headless_progress_last_emit.time_since_epoch().count() == 0 ||
            now - headless_progress_last_emit >=
                std::chrono::milliseconds(500);
        if (!complete && !percent_changed && !time_elapsed) {
            return;
        }

        headless_progress_last_percent = percent;
        headless_progress_last_emit = now;
        const std::uint64_t processed_frames =
            complete && expected_frames > 0U ? expected_frames
                                             : output_frame_count;
        const std::uint64_t written_frames =
            writer.is_open()
                ? static_cast<std::uint64_t>(
                      std::max<std::int64_t>(0, writer.get_frame_count()))
                : png_frame_count;
        const double elapsed_seconds =
            static_cast<double>(processed_frames) / recording_fps;

        std::cout << "acmxvk: [";
        if (percent >= 0) {
            std::cout << std::setw(3) << percent << '%';
        } else {
            std::cout << "  ?%";
        }
        std::cout << "] Frame " << processed_frames << '/';
        if (expected_frames > 0U) {
            std::cout << expected_frames;
        } else {
            std::cout << '?';
        }
        std::cout << " | Written: " << written_frames
                  << " | Time: " << formatHudTime(elapsed_seconds);
        if (writer.is_open()) {
            constexpr double BYTES_PER_MEGABYTE = 1024.0 * 1024.0;
            const double file_size_mb =
                static_cast<double>(writer.get_bytes_written()) /
                BYTES_PER_MEGABYTE;
            std::ostringstream size_text;
            size_text << std::fixed << std::setprecision(2) << file_size_mb;
            std::cout << " | Size: " << size_text.str() << " MB";
        }
        std::cout << '\n'
                  << std::flush;
    }

    [[nodiscard]] double MainWindow::hudWallElapsedSeconds() const {
        return std::max(
            0.0,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          hud_session_start)
                .count());
    }

    [[nodiscard]] bool MainWindow::currentVideoTimeline(
        double &timeline,
        std::uint64_t *frame_index) const {
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

    [[nodiscard]] double MainWindow::hudVideoPositionSeconds() const {
        double position = 0.0;
        if (!currentVideoTimeline(position)) {
            return 0.0;
        }
        if (video_duration_seconds > 0.0) {
            position = std::min(position, video_duration_seconds);
        }
        return std::max(0.0, position);
    }

    [[nodiscard]] std::string MainWindow::hudVideoTimeString() const {
        std::string text = "Video: " +
                           formatHudTime(hudVideoPositionSeconds()) +
                           " / ";
        text += video_duration_seconds > 0.0
                    ? formatHudTime(video_duration_seconds)
                    : "--:--:--";
        return text;
    }

    [[nodiscard]] std::string MainWindow::hudElapsedTimeString() const {
        return "Elapsed: " + formatHudTime(hudWallElapsedSeconds());
    }

    void MainWindow::updateHudFrameRate() {
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

    void MainWindow::paceMaximizedRendering() {
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

    void MainWindow::updateCameraFrameRate() {
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

    void MainWindow::queueRuntimeHud(int &y, int line_height) {
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

    void MainWindow::queueOverlayText() {
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

    [[nodiscard]] std::string MainWindow::captureFourccName(double value) {
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

    [[nodiscard]] bool MainWindow::dnnHostProcessingEnabled() const {
#ifdef ACMXVK_WITH_DNN
        return edge_detector != nullptr || human_segmenter != nullptr ||
               generic_onnx_processor != nullptr;
#else
        return false;
#endif
    }

    void MainWindow::applyDnnEffects(cv::Mat &rgba) {
#ifdef ACMXVK_WITH_DNN
        if (rgba.type() == CV_16UC4 &&
            (human_segmenter != nullptr || edge_detector != nullptr ||
             generic_onnx_processor != nullptr)) {
            cv::Mat compatible = rgba16ToRgba8(rgba);
            if (!hdr_dnn_compatibility_logged) {
                std::cout
                    << "acmxvk: HDR increment 2: DNN preprocessing uses an "
                       "RGBA8 compatibility copy before RGBA16 upload\n";
                hdr_dnn_compatibility_logged = true;
            }
            applyDnnEffects(compatible);
            compatible.convertTo(rgba, CV_16UC4, 257.0);
            return;
        }
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

    void MainWindow::updateHumanOverlayTexture() {
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
    // Input setup, output encoding, snapshots, and readback handling.
    void MainWindow::openInput() {
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
            video_hdr_info = probeVideoHdrInfo(options.input_file);
#ifdef MXVK_WITH_FFMPEG_CAPTURE
            hdr_input_precision_enabled =
                video_hdr_info.valid && video_hdr_info.hdr &&
                using_ffmpeg_capture;
#else
            hdr_input_precision_enabled = false;
#endif
            hdr_transfer_processing_enabled =
                hdr_input_precision_enabled &&
                (video_hdr_info.color_transfer ==
                     COLOR_TRANSFER_SMPTE2084 ||
                 video_hdr_info.color_transfer ==
                     COLOR_TRANSFER_ARIB_STD_B67);
            hdr_transfer_hlg =
                video_hdr_info.color_transfer ==
                COLOR_TRANSFER_ARIB_STD_B67;
            setHdrRenderIntermediatesEnabled(hdr_input_precision_enabled);
            setFrameReadbackRgba16Enabled(hdr_input_precision_enabled);
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
            if (video_hdr_info.valid && video_hdr_info.hdr) {
                std::cout << "acmxvk: HDR input metadata detected\n";
                printVideoHdrInfo(video_hdr_info, std::cout);
                if (hdr_input_precision_enabled) {
                    std::cout << "acmxvk: HDR processing active: native "
                                 "RGBA16 input, RGBA16F effects/history, and "
                                 "normalized RGBA16 Vulkan readback\n";
                    if (hdr_transfer_processing_enabled) {
                        std::cout
                            << "acmxvk: HDR transfer: "
                            << (hdr_transfer_hlg ? "HLG" : "PQ")
                            << " decoded to linear BT.2020 before effects and "
                               "encoded after effects\n";
                        if (!options.headless) {
                            std::cout
                                << "acmxvk: HDR preview: presentation-only "
                                   "BT.2020-to-BT.709 SDR tone mapping active; "
                                   "Main10 recording remains unchanged\n";
                        }
                    } else {
                        std::cerr
                            << "acmxvk: HDR transfer "
                            << video_hdr_info.color_transfer
                            << " is not PQ or HLG; preserving transfer-encoded "
                               "values through the precision path\n";
                    }
                } else {
                    std::cout
                        << "acmxvk: HDR precision path unavailable because this "
                           "build is not using MXVK FFmpeg capture; falling "
                           "back to RGBA8\n";
                }
            }
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

    [[nodiscard]] std::pair<int, int> MainWindow::source_dimensions() {
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

    void MainWindow::configureRenderResolution() {
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

        if (options.headless) {
            std::cout << "acmxvk: headless output resolution: "
                      << render_width << 'x' << render_height << '\n';
            return;
        }

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

    [[nodiscard]] double MainWindow::outputFrameRate() {
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

    void MainWindow::requestSnapshot(SnapshotFormat format) {
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

    [[nodiscard]] bool MainWindow::continuousReadbackEnabled() const {
        return writer.is_open() || options.png_output ||
               options.generate_interval > 0;
    }

    void MainWindow::openOutput() {
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
            png_output_directory =
                output_frame_directory(options.output_file, "png");
            create_output_directory(png_output_directory);
            std::cout << "acmxvk: writing PNG sequence to "
                      << png_output_directory.string() << '\n';
        }

        if (options.generate_interval > 0) {
            if (!options.output_file.empty()) {
                generate_output_directory =
                    output_frame_directory(options.output_file,
                                           "generate");
            } else if (!options.input_file.empty()) {
                generate_output_directory =
                    output_frame_directory(options.input_file,
                                           "generate");
            } else {
                generate_output_directory = "camera-generate";
            }
            create_output_directory(generate_output_directory);
            std::cout << "acmxvk: saving every " << options.generate_interval
                      << "th frame to " << generate_output_directory.string() << '\n';
        }

        if (!options.output_file.empty() && !options.png_output) {
            EncodeOptions encode_options;
            encode_options.preset = options.encode_preset;
            encode_options.tune = options.encode_tune;
            encode_options.crf = options.encode_crf;
            encode_options.bit_rate = options.encode_bitrate;
            encode_options.codec = options.encode_codec;
            encode_options.ffmpeg_options = options.encode_params;
            encode_options.realtime = options.encode_realtime;
            encode_options.block_when_full = options.no_drop;
            hdr_output_enabled = hdr_transfer_processing_enabled;
            if (hdr_output_enabled) {
                if ((recording_width & 1) != 0 ||
                    (recording_height & 1) != 0) {
                    throw std::runtime_error(
                        "HDR Main10 output requires even width and height");
                }
                encode_options.hdr.enabled = true;
                encode_options.hdr.color_primaries =
                    video_hdr_info.color_primaries;
                encode_options.hdr.color_trc =
                    video_hdr_info.color_transfer;
                encode_options.hdr.color_space =
                    video_hdr_info.color_space;
                encode_options.hdr.color_range =
                    video_hdr_info.color_range;
                encode_options.hdr.mastering_display =
                    video_hdr_info.mastering_display;
                encode_options.hdr.content_light =
                    video_hdr_info.content_light;
                std::cout
                    << "acmxvk: HDR output: HEVC Main10 with captured "
                    << (hdr_transfer_hlg ? "BT.2020/HLG" : "BT.2020/PQ")
                    << " color metadata (software libx265)\n";
            }

            if (options.encode_bitrate > 0) {
                std::cout << "acmxvk: encoder rate control: target VBR "
                          << options.encode_bitrate << " bits/s\n";
            } else {
                std::cout << "acmxvk: encoder rate control: CRF/CQ "
                          << options.encode_crf << '\n';
            }

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

    void MainWindow::onFrameReadbackScheduled() {
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

    void MainWindow::onFrameReadback(std::vector<std::uint8_t> &rgba, uint32_t width,
                                     uint32_t height) {
        handleFrameReadback(rgba, nullptr, width, height);
    }

    void MainWindow::handleFrameReadback(
        std::vector<std::uint8_t> &rgba,
        const std::vector<std::uint16_t> *rgba16, uint32_t width,
        uint32_t height) {
        if (readback_requests.empty()) {
            std::cerr << "acmxvk: received frame readback without queued metadata\n";
            return;
        }
        const ReadbackRequest request = readback_requests.front();
        readback_requests.pop_front();

        if (request.snapshot) {
            const fs::path path = snapshot_path(
                options.snapshot_directory, width, height, snapshot_count,
                request.snapshot_format);
            SnapshotJob job;
            job.path = path;
            job.width = width;
            job.height = height;
            job.format = request.snapshot_format;
            if (rgba16 != nullptr &&
                (request.snapshot_format == SnapshotFormat::Tiff ||
                 request.snapshot_format == SnapshotFormat::Raw)) {
                job.rgba16 = *rgba16;
            }
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
        const std::uint16_t *hdr_output_pixels =
            rgba16 != nullptr ? rgba16->data() : nullptr;
        cv::Mat hdr_resized;
        if (static_cast<int>(width) != recording_width ||
            static_cast<int>(height) != recording_height) {
            const cv::Mat source(static_cast<int>(height), static_cast<int>(width),
                                 CV_8UC4, rgba.data());
            cv::resize(source, resized, cv::Size(recording_width, recording_height),
                       0.0, 0.0, cv::INTER_LINEAR);
            output_pixels = resized.ptr();
            if (rgba16 != nullptr) {
                const cv::Mat hdr_source(
                    static_cast<int>(height), static_cast<int>(width),
                    CV_16UC4,
                    const_cast<std::uint16_t *>(rgba16->data()));
                cv::resize(hdr_source, hdr_resized,
                           cv::Size(recording_width, recording_height), 0.0,
                           0.0, cv::INTER_LINEAR);
                hdr_output_pixels = hdr_resized.ptr<std::uint16_t>();
            }
        }

        if (writer.is_open()) {
            if (hdr_output_enabled) {
                if (hdr_output_pixels == nullptr) {
                    throw std::runtime_error(
                        "HDR Main10 recording did not receive an RGBA16 "
                        "Vulkan readback");
                }
                if (request.has_pts) {
                    writer.write_hdr_rgba16_at_pts(
                        const_cast<std::uint16_t *>(hdr_output_pixels),
                        static_cast<std::int64_t>(request.pts));
                } else {
                    writer.write_hdr_rgba16(
                        const_cast<std::uint16_t *>(hdr_output_pixels));
                }
            } else if (request.has_pts) {
                writer.write_at_pts(output_pixels,
                                    static_cast<std::int64_t>(request.pts));
            } else {
                writer.write(output_pixels);
            }
        }
        if (options.png_output) {
            SnapshotWriter::savePng(
                frame_path(png_output_directory, png_frame_count),
                output_pixels, recording_width, recording_height);
            ++png_frame_count;
        }
        if (options.generate_interval > 0 &&
            (request.has_pts ? request.pts : output_frame_count) %
                    static_cast<std::uint64_t>(options.generate_interval) ==
                0) {
            SnapshotWriter::savePng(
                frame_path(generate_output_directory,
                           generated_frame_count),
                output_pixels, recording_width, recording_height);
            ++generated_frame_count;
        }
        ++output_frame_count;
        emitHeadlessProgress(false);

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
                headless_progress_complete = options.headless;
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

    void MainWindow::onFrameReadbackRgba16(
        std::vector<std::uint16_t> &rgba, uint32_t width, uint32_t height) {
        if (!hdr_readback_logged) {
            std::cout
                << "acmxvk: HDR readback: normalized RGBA16 received from "
                   "the final Vulkan HDR intermediate"
                << (hdr_output_enabled
                        ? "; feeding MXWrite's HEVC Main10 encoder\n"
                        : "; converting to RGBA8 for snapshots/output\n");
            hdr_readback_logged = true;
        }
        std::vector<std::uint8_t> rgba8 =
            tone_map_hdr_rgba16(rgba, hdr_transfer_hlg);
        handleFrameReadback(rgba8, &rgba, width, height);
    }
    // 3D rendering, crossfades, pipelines, history, and frame uploads.
    void MainWindow::initializeModel() {
        if (!options.enable_3d || model_initialized) {
            return;
        }

        try {
            input_model.enableExtendedFragmentUniforms();
            input_model.load(this, options.model_file, "", "", 1.0F);
            input_model.setShaders(
                this, model_vertex_shader_path(options).string(),
                model_fragment_shader_path(options).string());
            model_effect_shader = model_fragment_shader_path(options);
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

    void MainWindow::initializeSprite() {
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
            if (hdr_input_precision_enabled) {
                frame_sprite->enableHistoryTextureRgba16Float(
                    source_width, source_height,
                    static_cast<uint32_t>(options.texture_cache_size));
            } else {
                frame_sprite->enableHistoryTexture(
                    source_width, source_height,
                    static_cast<uint32_t>(options.texture_cache_size));
            }
        }
        const std::string initial_fragment =
            options.history_test
                ? echo_cache_shader_path(options).string()
            : hdr_input_precision_enabled
                ? passthrough_shader_path(options).string()
                : std::string{};
        if (hdr_input_precision_enabled) {
            frame_sprite->createEmptySpriteRgba16(
                source_width, source_height,
                sprite_vertex_shader_path(options).string(), initial_fragment);
        } else {
            frame_sprite->createEmptySprite(
                source_width, source_height,
                sprite_vertex_shader_path(options).string(), initial_fragment);
        }

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

    void MainWindow::resetShaderTime() {
        previous_frame = std::chrono::steady_clock::now();
        previous_video_shader_timeline = 0.0;
        video_shader_timeline_initialized = false;
        shader_time = 0.0;
        frame_count = 0;
    }

    void MainWindow::beginCrossfade() {
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
            if (hdr_input_precision_enabled) {
                crossfade_previous_sprite->enableHistoryTextureRgba16Float(
                    extent.width, extent.height, 1U);
            } else {
                crossfade_previous_sprite->enableHistoryTexture(
                    extent.width, extent.height, 1U);
            }
            if (hdr_transfer_processing_enabled) {
                const cv::Mat linear_previous =
                    decode_hdr_transfer(previous_rgba, hdr_transfer_hlg);
                crossfade_previous_sprite->updateHistoryTextureRgba16(
                    linear_previous.ptr<std::uint16_t>(),
                    static_cast<int>(extent.width),
                    static_cast<int>(extent.height),
                    static_cast<int>(linear_previous.step));
            } else {
                crossfade_previous_sprite->updateHistoryTexture(
                    previous_rgba.ptr(), static_cast<int>(extent.width),
                    static_cast<int>(extent.height),
                    static_cast<int>(previous_rgba.step));
            }
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

    void MainWindow::updateCrossfade(const std::chrono::steady_clock::time_point now) {
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

    void MainWindow::cycleCrossfade(int direction) {
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

    void MainWindow::adjustModelScale(float amount) {
        if (!model_initialized || model_scale_oscillation_active) {
            return;
        }
        model_scale = std::clamp(model_scale + amount, 0.05F, 20.0F);
        std::cout << "acmxvk: model scale " << model_scale << '\n';
    }

    void MainWindow::maybeRandomizeCrossfade() {
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

    void MainWindow::togglePause() {
        if (source_kind == SourceKind::Camera) {
            std::cout << "acmxvk: pause is available for video and graphic input\n";
            return;
        }
        input_paused = !input_paused;
        setSourcePlaybackClockPaused(input_paused || rendering_frozen);
        std::cout << "acmxvk: input pause "
                  << (input_paused ? "enabled" : "disabled") << '\n';
    }

    void MainWindow::toggleFreeze() {
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

    void MainWindow::stepShaderTime(double amount) {
        shader_time += amount;
        std::cout << "acmxvk: shader time stepped to " << shader_time << '\n';
    }

    void MainWindow::adjustTimeSpeed(double amount) {
        options.time_speed += amount;
        if (std::abs(options.time_speed) < 0.01) {
            options.time_speed = 0.0;
        }
        std::cout << "acmxvk: shader time speed " << options.time_speed << '\n';
    }

    void MainWindow::toggleFullscreen() {
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

    void MainWindow::resetAutopilotInterval() {
        if (options.autopilot_random_timeout > 0) {
            std::uniform_int_distribution<int> distribution(
                4, std::max(4, options.autopilot_random_timeout));
            autopilot_interval_frames = distribution(autopilot_rng);
        } else {
            autopilot_interval_frames = options.autopilot_frames;
        }
    }

    void MainWindow::logSelectedPlaylistNode(std::string_view action) const {
        if (playlist.empty()) {
            return;
        }
        std::cout << "acmxvk: " << action << " playlist node "
                  << (playlist_index + 1) << '/' << playlist.size() << ": "
                  << playlist[playlist_index].name << " ("
                  << playlist[playlist_index].shaders.size()
                  << " passes)\n";
    }

    [[nodiscard]] std::uint64_t MainWindow::autopilotFrameAdvance() {
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

    void MainWindow::toggleAutopilot(bool sequential) {
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

    void MainWindow::updateAutopilot() {
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

    void MainWindow::selectShader(int direction) {
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

    void MainWindow::selectPlaylistNode(int direction) {
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

    [[nodiscard]] std::vector<fs::path> MainWindow::activeShaderPipeline() const {
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
            pipeline.emplace_back(flip_shader_path(options));
        }
        if (crossfade_active) {
            pipeline.emplace_back(
                crossfade_shader_path(options, crossfade_shader_index));
        }
        if (pipeline.empty()) {
            pipeline.emplace_back(passthrough_shader_path(options));
        }
        if (options.human_background) {
            pipeline.emplace_back(human_composite_shader_path(options));
        }
        return pipeline;
    }

    [[nodiscard]] fs::path MainWindow::directModelFragmentShader() const {
        if (!model_3d_active || !model_initialized || !effects_enabled ||
            hdr_input_precision_enabled || playlist_enabled ||
            multipass_enabled ||
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

    void MainWindow::applyShaderPipeline() {
        if (getDevice() == VK_NULL_HANDLE) {
            return;
        }
        vkDeviceWaitIdle(getDevice());
        detachPostProcessingShader();
        post_process_sprites.clear();
        setPostProcessingPresentFragmentShader(
            hdr_transfer_processing_enabled
                ? hdr_preview_shader_path(options, hdr_transfer_hlg).string()
                : std::string{});
        frame_sprite->setEffectsEnabled(effects_enabled);

        const fs::path direct_model_shader = directModelFragmentShader();
        model_texture_prepass_active =
            model_3d_active && direct_model_shader.empty();
        setPostProcessingTextureConsumerEnabled(
            model_texture_prepass_active);
        if (model_initialized) {
            input_model.setColorAttachmentFormat(
                model_texture_prepass_active &&
                        !hdr_transfer_processing_enabled
                    ? getSwapchainFormat()
                    : getSceneColorFormat());
            const fs::path desired_model_shader =
                direct_model_shader.empty()
                    ? model_fragment_shader_path(options)
                    : direct_model_shader;
            if (desired_model_shader != model_effect_shader) {
                input_model.setShaders(
                    this, model_vertex_shader_path(options).string(),
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
                pipeline.emplace_back(passthrough_shader_path(options));
            }
            std::cout << "acmxvk: 3D texture effect: "
                      << direct_model_shader.filename().string()
                      << " [fragment, evaluated on model UVs]\n";
        } else if (model_3d_active && effects_enabled &&
                   !currentShader().empty()) {
            std::cout << "acmxvk: 3D texture prepass: fragment/compute "
                         "chain output mapped onto model UVs\n";
        }
        if (hdr_transfer_processing_enabled) {
            pipeline.insert(
                pipeline.begin(),
                hdr_transfer_shader_path(options, hdr_transfer_hlg, false));
            pipeline.emplace_back(
                hdr_transfer_shader_path(options, hdr_transfer_hlg, true));
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
            if (crossfade_active &&
                shader == crossfade_shader_path(options,
                                                crossfade_shader_index)) {
                crossfade_post_process_index = index;
                effect.historySource = crossfade_previous_sprite;
                effect.params[0] = crossfade_alpha;
            } else if (options.human_background &&
                       shader == human_composite_shader_path(options)) {
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

    [[nodiscard]] bool MainWindow::readTrackedInputFrame() {
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

    [[nodiscard]] bool MainWindow::skipInputFrame() {
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

    [[nodiscard]] bool MainWindow::handleCaptureEnd(bool discard) {
        if (source_kind == SourceKind::Camera) {
            return true;
        }
        if (!options.repeat) {
            setFrameReadbackEnabled(false);
            headless_progress_complete = options.headless;
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

    [[nodiscard]] bool MainWindow::readClockedVideoFrame(double clock_seconds) {
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

    void MainWindow::closeVideoCapture() {
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

    [[nodiscard]] bool MainWindow::openVideoCapture() {
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

    [[nodiscard]] bool MainWindow::readHostRgba(cv::Mat &rgba) {
#ifdef MXVK_WITH_FFMPEG_CAPTURE
        if (using_ffmpeg_capture) {
            int width = 0;
            int height = 0;
            int pitch = 0;
            if (hdr_input_precision_enabled) {
                if (!ffmpeg_capture.readRgba16(ffmpeg_rgba16, width, height,
                                               pitch, false) ||
                    ffmpeg_rgba16.empty() || width <= 0 || height <= 0 ||
                    pitch < width * 8) {
                    return false;
                }
                rgba = cv::Mat(height, width, CV_16UC4,
                               ffmpeg_rgba16.data(),
                               static_cast<std::size_t>(pitch));
                return true;
            }
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

    void MainWindow::initializeHistory(const cv::Mat &rgba) {
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

    void MainWindow::updateHistoryFrame(const cv::Mat &rgba) {
#ifdef ACMXVK_WITH_CUDA
        if (gpu_filter_engine != nullptr && rgba.type() == CV_8UC4) {
            updateFilteredCudaHistoryFrame();
            return;
        }
#endif
        if (rgba.type() == CV_16UC4) {
            if (hdr_transfer_processing_enabled) {
                const cv::Mat linear_history =
                    decode_hdr_transfer(rgba, hdr_transfer_hlg);
                frame_sprite->updateHistoryTextureRgba16(
                    linear_history.ptr<std::uint16_t>(), linear_history.cols,
                    linear_history.rows,
                    static_cast<int>(linear_history.step));
                return;
            }
            frame_sprite->updateHistoryTextureRgba16(
                rgba.ptr<uint16_t>(), rgba.cols, rgba.rows,
                static_cast<int>(rgba.step));
            return;
        }
        frame_sprite->updateHistoryTexture(rgba.ptr(), rgba.cols, rgba.rows,
                                           static_cast<int>(rgba.step));
    }

    void MainWindow::updateCameraHistory() {
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
    void MainWindow::updateModelTextureCuda(const cv::cuda::GpuMat &rgba,
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

    void MainWindow::updateCudaHistoryFrame(const cv::cuda::GpuMat &rgba,
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
    void MainWindow::updateFilteredCudaHistoryFrame() {
        updateCudaHistoryFrame(gpu_filter_engine->output(),
                               gpu_filter_engine->stream());
    }
#endif

    void MainWindow::initializeCudaHistory(const cv::cuda::GpuMat &rgba,
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
    void MainWindow::uploadInputFrame(const cv::cuda::GpuMat &rgba,
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
    MainWindow::rotateCudaFrame(const cv::cuda::GpuMat &rgba,
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

    void MainWindow::uploadInputFrame(const cv::Mat &rgba) {
#ifdef ACMXVK_WITH_CUDA
        if (gpu_filter_engine != nullptr && rgba.type() == CV_8UC4) {
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
        if (gpu_filter_engine != nullptr && rgba.type() == CV_16UC4 &&
            !hdr_cuda_filter_bypass_logged) {
            std::cout
                << "acmxvk: HDR increment 2: bypassing RGBA8 CUDA filters to "
                   "preserve the 16-bit source texture\n";
            hdr_cuda_filter_bypass_logged = true;
        }
#endif
        if (rgba.type() == CV_16UC4) {
            frame_sprite->updateTextureRgba16(
                rgba.ptr<std::uint16_t>(), rgba.cols, rgba.rows,
                static_cast<int>(rgba.step));
            if (!hdr_input_upload_logged) {
                std::cout << "acmxvk: uploaded first RGBA16 HDR source frame "
                             "to Vulkan\n";
                hdr_input_upload_logged = true;
            }
        } else {
            frame_sprite->updateTexture(rgba.ptr(), rgba.cols, rgba.rows,
                                        static_cast<int>(rgba.step));
        }

        cv::Mat model_compatible;
        const cv::Mat *model_input = &rgba;
        if (model_initialized && rgba.type() == CV_16UC4) {
            model_compatible = rgba16ToRgba8(rgba);
            model_input = &model_compatible;
        }
        if (model_initialized &&
            !input_model.updatePrimaryTexture(
                model_input->ptr(), model_input->cols, model_input->rows,
                static_cast<int>(model_input->step))) {
            throw std::runtime_error(
                "MXVK could not update the 3D model texture");
        }
    }

    [[nodiscard]] bool MainWindow::readLatestCameraFrame() {
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

    [[nodiscard]] bool MainWindow::readInputFrame() {
        if (source_kind == SourceKind::Camera && options.maximize_fps) {
            return readLatestCameraFrame();
        }
#ifdef ACMXVK_WITH_CUDA
        if (gpu_filter_engine != nullptr && !dnnHostProcessingEnabled() &&
            !hdr_input_precision_enabled) {
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
            !dnnHostProcessingEnabled() && !hdr_input_precision_enabled) {
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

        bool requires_host_frame = hdr_input_precision_enabled ||
                                   dnnHostProcessingEnabled() ||
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

    void MainWindow::updateShaderUniforms(int width, int height) {
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

} // namespace acmxvk
