      private:
        // Runtime state shared by the focused MainWindow implementation sections.
        enum class SourceKind { Camera,
                                Video,
                                Graphic };

        struct ReadbackRequest {
            bool snapshot = false;
            SnapshotFormat snapshot_format = SnapshotFormat::Png;
            bool continuous = false;
            bool frame_due = false;
            bool has_pts = false;
            std::uint64_t pts = 0;
        };

        static constexpr std::uint32_t COMPATIBILITY_SPECTRUM_BIN_COUNT = 256;

        Options options;
        SourceKind source_kind = SourceKind::Camera;
        mxvk::VK_Capture capture;
        LatestCameraFrame latest_camera_frame;
        SnapshotWriter snapshot_writer;
#ifdef MXVK_WITH_FFMPEG_CAPTURE
        mxvk::VK_FF_Capture ffmpeg_capture;
        std::vector<std::uint8_t> ffmpeg_rgba;
        bool using_ffmpeg_capture = false;
        bool ffmpeg_seek_repeat_logged = false;
#ifdef ACMXVK_WITH_MXVK_CUDA
        cv::cuda::Stream ffmpeg_cuda_stream;
#endif
#endif
        Writer writer;
        mxvk::VKAbstractModel input_model;
        mxvk::VK_Sprite *frame_sprite = nullptr;
        mxvk::VK_Sprite *crossfade_previous_sprite = nullptr;
        mxvk::VK_Sprite *human_overlay_sprite = nullptr;
        cv::Mat graphic_rgba;
        cv::Mat human_overlay_rgba;
        cv::Mat latest_camera_history_rgba;
        std::vector<fs::path> shaders;
        std::vector<fs::path> configured_passes;
        std::vector<PlaylistNode> playlist;
        std::vector<mxvk::VK_Sprite *> post_process_sprites;
        std::vector<ShaderManifest::CustomUniform> custom_uniforms;
        std::vector<float> custom_uniform_values;
        fs::path shader_library_directory;
        fs::path shader_manifest_path;
        fs::path png_output_directory;
        fs::path generate_output_directory;
        InterfaceClient interface_client;
        std::uint32_t interface_last_sequence = 0;
        std::uint32_t interface_last_audio_file_sequence = 0;
        std::uint32_t interface_last_reload_sequence = 0;
        std::size_t shader_index = 0;
        std::size_t playlist_index = 0;
        std::size_t crossfade_post_process_index =
            std::numeric_limits<std::size_t>::max();
        bool effects_enabled = true;
        bool multipass_enabled = false;
        bool playlist_enabled = false;
        bool shader_locked = false;
        bool model_initialized = false;
        bool model_3d_active = false;
        bool model_texture_prepass_active = false;
        bool model_auto_rotate = false;
        bool model_wave_active = false;
        bool model_scale_oscillation_active = false;
        bool model_mouse_dragging = false;
        bool shader_history_required = false;
        bool shader_spectrum_required = false;
        bool shader_spectrum_history_required = false;
        float mouse_x = 0.0F;
        float mouse_y = 0.0F;
        bool mouse_pressed = false;
        bool history_initialized = false;
        bool initial_frame_pending = false;
        bool async_camera_frame_uploaded = false;
        bool async_camera_initial_wait_completed = false;
        bool render_pacing_started = false;
        bool media_timeline_started = false;
        bool source_frame_received = false;
        bool recording_frame_due = false;
        bool recording_frame_has_pts = false;
        bool media_clock_sync_logged = false;
        bool camera_recording_clock_logged = false;
        bool recording_complete = false;
        bool input_paused = false;
        bool rendering_frozen = false;
        bool source_playback_clock_paused = false;
        bool shader_time_active = true;
        bool audio_time_active = false;
        bool audio_delta_time = false;
        bool spectrum_scale_by_sensitivity = false;
        bool watermark_enabled = !options.watermark_text.empty();
        bool counter_disabled =
            options.disable_counter || !options.watermark_text.empty();
        int overlay_font_size = 18;
        int preview_overlay_font_size = 18;
        bool snapshot_pending = false;
        SnapshotFormat pending_snapshot_format = SnapshotFormat::Png;
        bool autopilot_enabled = false;
        bool autopilot_sequential = false;
        bool autopilot_random_crossfade = false;
        bool crossfade_active = false;
        int recording_width = 0;
        int recording_height = 0;
        int camera_reported_width = 0;
        int camera_reported_height = 0;
        int autopilot_counter = 0;
        int autopilot_interval_frames = 0;
        std::uint64_t previous_autopilot_video_frame = 0;
        int history_delay_counter = 0;
        std::size_t crossfade_shader_index = 0;
        bool camera_history_clock_started = false;
        double recording_fps = 0.0;
        double video_source_fps = 0.0;
        double video_duration_seconds = 0.0;
        double camera_reported_fps = 0.0;
        double camera_delivered_fps = 0.0;
        double camera_last_logged_fps = 0.0;
        double shader_time = 0.0;
        float legacy_alpha = 0.1F;
        float crossfade_alpha = 1.0F;
        float model_pitch_degrees = 0.0F;
        float model_yaw_degrees = 270.0F;
        float model_rotation_x_degrees = 0.0F;
        float model_rotation_y_degrees = 0.0F;
        float model_rotation_z_degrees = 0.0F;
        float model_camera_distance = 0.0F;
        float model_camera_movement_speed = 0.1F;
        float model_camera_rotation_speed = 5.0F;
        float model_scale = 1.0F;
        float model_rotation_speed = 18.0F;
        float model_view_rotation_degrees = 0.0F;
        float model_wave_amplitude_x = 0.0F;
        float model_wave_amplitude_y = 0.0F;
        float model_wave_amplitude_z = 0.0F;
        float model_wave_direction_x = 1.0F;
        float model_wave_direction_y = 1.0F;
        float model_wave_direction_z = 1.0F;
        float model_wave_phase = 0.0F;
        float model_wave_audio_step = 0.0F;
        float model_scale_oscillation_phase = 0.0F;
        fs::path model_effect_shader;
        mxvk::ModelFragmentUniforms model_fragment_uniforms{};
        bool legacy_alpha_increasing = true;
        int model_last_mouse_x = 0;
        int model_last_mouse_y = 0;
        std::chrono::steady_clock::time_point compatibility_clock_start =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point model_last_render_time =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point crossfade_start_time =
            std::chrono::steady_clock::now();
        double crossfade_start_video_timeline = 0.0;
        std::uint64_t output_frame_count = 0;
        std::uint64_t decoded_video_frame_count = 0;
        std::uint64_t video_source_frame_count = 0;
        std::uint64_t recording_frame_pts = 0;
        std::uint64_t next_clock_output_frame = 0;
        std::uint64_t png_frame_count = 0;
        std::uint64_t generated_frame_count = 0;
        std::uint64_t snapshot_count = 0;
        std::uint64_t frame_count = 0;
        std::uint64_t previous_model_video_frame = 0;
        std::uint64_t hud_fps_frame_count = 0;
        std::uint64_t camera_fps_frame_count = 0;
        double hud_display_fps = 0.0;
        std::deque<ReadbackRequest> readback_requests;
        std::chrono::steady_clock::time_point hud_session_start{
            std::chrono::steady_clock::now()};
        std::chrono::steady_clock::time_point hud_fps_last_tick{
            hud_session_start};
        std::chrono::steady_clock::time_point camera_fps_last_tick{};
        std::chrono::steady_clock::time_point camera_history_next_update{};
        std::chrono::steady_clock::time_point window_title_last_update{};
        std::chrono::steady_clock::time_point next_render_tick{};
        std::chrono::steady_clock::time_point source_playback_clock_start{};
        std::chrono::steady_clock::time_point source_playback_pause_start{};
        std::chrono::steady_clock::duration source_playback_paused_duration{};
        std::chrono::steady_clock::time_point previous_frame{std::chrono::steady_clock::now()};
        double previous_video_shader_timeline = 0.0;
        bool video_shader_timeline_initialized = false;
        bool video_shader_clock_logged = false;
        bool model_video_timeline_initialized = false;
        bool crossfade_uses_video_timeline = false;
        bool autopilot_video_timeline_initialized = false;
        std::mt19937 autopilot_rng{std::random_device{}()};
#ifdef ACMXVK_WITH_CUDA
        std::unique_ptr<gpu::FilterEngine> gpu_filter_engine;
#endif
#ifdef ACMXVK_WITH_DNN
        std::unique_ptr<dnn::EdgeDetector> edge_detector;
        std::unique_ptr<dnn::HumanSegmenter> human_segmenter;
        std::unique_ptr<dnn::GenericOnnxProcessor> generic_onnx_processor;
#endif
#ifdef ACMXVK_WITH_MXVK_CUDA
        cv::cuda::GpuMat cuda_input_rgba;
        cv::cuda::GpuMat cuda_rotated_rgba;
        cv::cuda::GpuMat cuda_rotation_transpose;
        cv::Mat cuda_input_fallback_rgba;
        cv::Mat cuda_history_fallback_rgba;
        cv::Mat cuda_model_fallback_rgba;
        bool cuda_input_path_logged = false;
        bool cuda_input_fallback_logged = false;
        bool cuda_history_fallback_logged = false;
        bool cuda_model_fallback_logged = false;
#endif
#ifdef MIDI_ENABLED
        struct MidiCcMapping {
            int channel = -1;
            int controller = 0;
            std::size_t uniform_index = 0;
            std::string uniform_name;
        };

        struct MidiKnobState {
            int value = 64;
            int previous_value = 64;
            int direction_action = 0;
            int frame_counter = 0;
            bool active = false;
        };

        std::unique_ptr<midi::MidiInput> midi_input;
        std::vector<midi::MidiMapping> midi_action_mappings;
        std::vector<MidiKnobState> midi_knob_states;
        std::vector<MidiCcMapping> midi_cc_mappings;
        std::array<int, 4> midi_slider_uniform_indices{-1, -1, -1, -1};
        std::uint64_t observed_midi_drops = 0;
#endif
