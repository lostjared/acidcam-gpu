#ifndef ACMXVK_MAIN_WINDOW_HPP
#define ACMXVK_MAIN_WINDOW_HPP

#include <mxvk/mxvk.hpp>
#include <mxvk/mxvk_abstract_model.hpp>
#include <mxvk/mxvk_cv.hpp>
#ifdef MXVK_WITH_FFMPEG_CAPTURE
#include <mxvk/mxvk_ff_capture.hpp>
#endif
#include <mxwrite.hpp>

#ifdef AUDIO_ENABLED
#include "audio.hpp"
#include "file_audio.hpp"
#endif
#ifdef MIDI_ENABLED
#include "midi.hpp"
#endif
#ifdef ACMXVK_WITH_CUDA
#include "gpu_filters.hpp"
#endif
#ifdef ACMXVK_WITH_DNN
#include "edge_dnn.hpp"
#endif
#include "app/interface_client.hpp"
#include "app/media_helpers.hpp"
#include "app/media_utils.hpp"
#include "app/options.hpp"
#include "app/output_paths.hpp"
#include "app/playlist.hpp"
#include "app/resource_paths.hpp"
#include "app/shader_library.hpp"
#include "app/snapshot_writer.hpp"
#include "input_validation.hpp"

#ifdef ACMXVK_WITH_MXVK_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace acmxvk {
    void request_headless_shutdown([[maybe_unused]] int signal_number) noexcept;

    // MainWindow's declaration and runtime state live here; method definitions
    // are implemented in main_window.cpp.
    class MainWindow final : public mxvk::VK_Window {
      public:
        explicit MainWindow(Options options);
        ~MainWindow() override;

        void event(SDL_Event &event) override;
        void onSwapchainRecreated() override;
        void onRecordCustomRendering(VkCommandBuffer command_buffer,
                                     std::uint32_t image_index) override;
        void onRecordPostProcessingTexture(
            VkCommandBuffer command_buffer, std::uint32_t image_index,
            VkImageView texture_view,
            [[maybe_unused]] VkExtent2D texture_extent) override;
        void recordModel(VkCommandBuffer command_buffer,
                         std::uint32_t image_index, VkImageView texture_view);
        void proc() override;

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
        std::vector<std::uint16_t> ffmpeg_rgba16;
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
        std::chrono::steady_clock::time_point interface_next_connect_attempt{};
        bool interface_connection_warning_reported = false;
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
        bool headless_shutdown_logged = false;
        bool headless_progress_complete = false;
        bool input_paused = false;
        bool rendering_frozen = false;
        bool source_playback_clock_paused = false;
        bool shader_time_active = true;
        bool audio_time_active = false;
        bool audio_delta_time = false;
        bool spectrum_scale_by_sensitivity = false;
        bool watermark_enabled = !options.watermark_text.empty();
        bool counter_disabled =
            options.headless || options.disable_counter ||
            !options.watermark_text.empty();
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
        VideoHdrInfo video_hdr_info;
        bool hdr_input_precision_enabled = false;
        bool hdr_transfer_processing_enabled = false;
        bool hdr_transfer_hlg = false;
        bool hdr_output_enabled = false;
        bool hdr_readback_logged = false;
        bool hdr_dnn_compatibility_logged = false;
        bool hdr_cuda_filter_bypass_logged = false;
        bool hdr_input_upload_logged = false;
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
        std::chrono::steady_clock::time_point headless_progress_last_emit{};
        int headless_progress_last_percent = -1;
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
#ifdef AUDIO_ENABLED
        std::unique_ptr<audio::AudioEngine> audio_engine;
        std::unique_ptr<audio::FileAudioSource> file_audio_source;
        float audio_warmup_envelope = 0.0F;
        bool audio_warmup_started = false;
        std::chrono::steady_clock::time_point audio_warmup_last_tick{};
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

#ifdef AUDIO_ENABLED
        void resetAudioWarmup();
        [[nodiscard]] float updateAudioWarmup(
            std::chrono::steady_clock::time_point now);
#endif
        void initializeDnn();
        void initializeGpuFilters();
        void selectGpuFilter(int direction);
        void openMidi();
        void configureMidiMappings();
#ifdef MIDI_ENABLED
        [[nodiscard]] bool applyMidiCc(const midi::MidiMessage &message);
        [[nodiscard]] SDL_Keycode midiActionKey(int action) const;
        [[nodiscard]] bool isMidiSliderMapping(
            const midi::MidiMapping &mapping) const;
        [[nodiscard]] static bool usesMidiDeltaDirection(
            const midi::MidiMapping &mapping);
        [[nodiscard]] bool isMidiModelAction(int action) const;
        [[nodiscard]] bool isMidiMappingSupported(
            const midi::MidiMapping &mapping) const;
        [[nodiscard]] std::string_view midiActionName(int action) const;
        void dispatchMidiModelAction(int action);
        void dispatchMidiAction(int action);
        [[nodiscard]] bool setMidiUniform(std::size_t uniform_index, int value,
                                          std::string_view label);
        [[nodiscard]] bool applyMidiMap(const midi::MidiMessage &message);
        void dispatchMidiKnobs();
#endif
        void uploadCustomUniforms();
        void pollMidi();
        void openAudio();
        void start_requested_audio_recording();
        void adjustAudioSensitivity(float amount);
        [[nodiscard]] bool audioSourceOpen() const;
        void startLiveAudioRecordingIfNeeded();
        void startMediaTimelineIfReady();
        void setSourcePlaybackClockPaused(bool paused);
        [[nodiscard]] bool mediaClockSeconds(double &seconds) const;

        void loadShaders();
        void applyCustomUniformOverrides();
        void printCustomUniforms() const;
        [[nodiscard]] std::string currentShader() const;
        [[nodiscard]] bool historyCacheEnabled() const;
        void recordShaderResources(const mxvk::ShaderModuleInfo &module_info,
                                   std::string_view source);
        [[nodiscard]] std::uint32_t spectrumBinCount() const;
        [[nodiscard]] bool spectrumTextureEnabledForShaders() const;
        [[nodiscard]] bool spectrumHistoryEnabledForShaders() const;
        void initialize_interface_control();
        void sync_interface_control();
        void apply_interface_playback_state(
            const InterfacePlaybackState &requested, bool announce);
        void apply_interface_overlay_state(
            const InterfaceOverlayState &requested, bool announce);
        void apply_interface_gpu_filter_state(
            const InterfaceGpuFilterState &requested, bool announce);
        void apply_interface_audio_file_state(
            const InterfaceAudioFileState &requested);
        void apply_interface_shader_reload(
            const InterfaceReloadState &requested);
        void apply_interface_multipass_state(
            const InterfaceMultipassState &requested);
        void apply_interface_shader_selection(
            const std::string &requested_name);
        void apply_interface_uniform_values(
            const std::vector<InterfaceUniformValue> &uniform_values);
        void loadShaderPasses();
        void loadPlaylist();

        void resolveConfiguredResourcePaths();
        void initializeOverlayFont();
        [[nodiscard]] static std::string clipOverlayText(std::string text);
        [[nodiscard]] const std::vector<fs::path> *activePasses() const;
        [[nodiscard]] std::string_view activeShaderRole() const;
        [[nodiscard]] std::string activePassDescription() const;
        [[nodiscard]] std::string activePlaylistDescription() const;
        [[nodiscard]] static std::string formatHudTime(double seconds_value);
        void updateWindowTitle(bool force = false);
        void emitHeadlessProgress(bool complete);
        [[nodiscard]] double hudWallElapsedSeconds() const;
        [[nodiscard]] bool currentVideoTimeline(
            double &timeline,
            std::uint64_t *frame_index = nullptr) const;
        [[nodiscard]] double hudVideoPositionSeconds() const;
        [[nodiscard]] std::string hudVideoTimeString() const;
        [[nodiscard]] std::string hudElapsedTimeString() const;
        void updateHudFrameRate();
        void paceMaximizedRendering();
        void updateCameraFrameRate();
        void queueRuntimeHud(int &y, int line_height);
        void queueOverlayText();
        [[nodiscard]] static std::string captureFourccName(double value);
        [[nodiscard]] bool dnnHostProcessingEnabled() const;
        void applyDnnEffects(cv::Mat &rgba);
        void updateHumanOverlayTexture();

        void openInput();
        [[nodiscard]] std::pair<int, int> source_dimensions();
        void configureRenderResolution();
        [[nodiscard]] double outputFrameRate();
        void requestSnapshot(SnapshotFormat format);
        [[nodiscard]] bool continuousReadbackEnabled() const;
        void openOutput();
        void onFrameReadbackScheduled() override;
        void onFrameReadback(std::vector<std::uint8_t> &rgba, uint32_t width,
                             uint32_t height) override;
        void onFrameReadbackRgba16(std::vector<std::uint16_t> &rgba,
                                   uint32_t width,
                                   uint32_t height) override;
        void handleFrameReadback(
            std::vector<std::uint8_t> &rgba,
            const std::vector<std::uint16_t> *rgba16, uint32_t width,
            uint32_t height);

        void initializeModel();
        void initializeSprite();
        void resetShaderTime();
        void beginCrossfade();
        void updateCrossfade(const std::chrono::steady_clock::time_point now);
        void cycleCrossfade(int direction);
        void adjustModelScale(float amount);
        void maybeRandomizeCrossfade();
        void togglePause();
        void toggleFreeze();
        void stepShaderTime(double amount);
        void adjustTimeSpeed(double amount);
        void toggleFullscreen();
        void resetAutopilotInterval();
        void logSelectedPlaylistNode(std::string_view action) const;
        [[nodiscard]] std::uint64_t autopilotFrameAdvance();
        void toggleAutopilot(bool sequential);
        void updateAutopilot();
        void selectShader(int direction);
        void selectPlaylistNode(int direction);
        [[nodiscard]] std::vector<fs::path> activeShaderPipeline() const;
        [[nodiscard]] fs::path directModelFragmentShader() const;
        void applyShaderPipeline();
        [[nodiscard]] bool readTrackedInputFrame();
        [[nodiscard]] bool skipInputFrame();
        [[nodiscard]] bool handleCaptureEnd(bool discard = false);
        [[nodiscard]] bool readClockedVideoFrame(double clock_seconds);
        void closeVideoCapture();
        [[nodiscard]] bool openVideoCapture();
        [[nodiscard]] bool readHostRgba(cv::Mat &rgba);
        void initializeHistory(const cv::Mat &rgba);
        void updateHistoryFrame(const cv::Mat &rgba);
        void updateCameraHistory();
#ifdef ACMXVK_WITH_MXVK_CUDA
        void updateModelTextureCuda(const cv::cuda::GpuMat &rgba,
                                    cv::cuda::Stream &source_stream);
        void updateCudaHistoryFrame(const cv::cuda::GpuMat &rgba,
                                    cv::cuda::Stream &source_stream);
        void updateFilteredCudaHistoryFrame();
        void initializeCudaHistory(const cv::cuda::GpuMat &rgba,
                                   cv::cuda::Stream &source_stream,
                                   bool filtered);
        void uploadInputFrame(const cv::cuda::GpuMat &rgba,
                              cv::cuda::Stream &source_stream);
        [[nodiscard]] const cv::cuda::GpuMat &
        rotateCudaFrame(const cv::cuda::GpuMat &rgba,
                        cv::cuda::Stream &source_stream);
#endif
        void uploadInputFrame(const cv::Mat &rgba);
        [[nodiscard]] bool readLatestCameraFrame();
        [[nodiscard]] bool readInputFrame();
        void updateShaderUniforms(int width, int height);
    };
} // namespace acmxvk

#endif
