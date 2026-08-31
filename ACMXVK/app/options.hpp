#ifndef ACMXVK_APP_OPTIONS_HPP
#define ACMXVK_APP_OPTIONS_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace acmxvk {
    namespace fs = std::filesystem;

    constexpr std::array<std::string_view, 35> CROSSFADE_NAMES{
        "xfade_01_linear", "xfade_02_block",
        "xfade_03_wipe", "xfade_04_radial",
        "xfade_05_pixelate", "xfade_06_dissolve",
        "xfade_07_swirl", "xfade_08_glitch",
        "xfade_09_diamond", "xfade_10_burn",
        "xfade_11_fade_black", "xfade_12_fade_white",
        "xfade_13_slide_left", "xfade_14_slide_right",
        "xfade_15_slide_up", "xfade_16_slide_down",
        "xfade_17_diagonal_wipe", "xfade_18_iris_open",
        "xfade_19_iris_close", "xfade_20_checker",
        "xfade_21_blinds_h", "xfade_22_blinds_v",
        "xfade_23_zoom_in", "xfade_24_zoom_out",
        "xfade_25_rotate", "xfade_26_ripple",
        "xfade_27_wave", "xfade_28_chroma",
        "xfade_29_invert", "xfade_30_flash",
        "xfade_31_explode", "xfade_32_mosaic",
        "xfade_33_shutter", "xfade_34_luma",
        "xfade_35_noise"};

    enum class FrameRotation { None,
                               Clockwise90,
                               Rotate180,
                               Counterclockwise90 };

    struct Options {
        int width = 1280;
        int height = 720;
        int camera_width = 1280;
        int camera_height = 720;
        int camera_device = 0;
        int enumerate_camera_device = -1;
        int shader_index = 0;
        int encode_crf = 18;
        int autopilot_frames = 0;
        int autopilot_random_timeout = 0;
        int generate_interval = 0;
        int cache_delay = 1;
        int texture_cache_size = 8;
        int audio_channels = 2;
        int audio_input_device = -1;
        int audio_output_device = -1;
        int audio_buffers = 0;
        int midi_device = -1;
        int gpu_frame_buffer_size = 10;
        int cuda_device = 0;
        double requested_fps = 0.0;
        double duration = 0.0;
        double cross_fade_duration = 0.5;
        double time_speed = 1.0;
        double max_size_mb = 0.0;
        double audio_sensitivity = 1.0;
        double audio_warm_rate = 0.5;
        double audio_pass_through_gain = 1.0;
        double audio_recording_gain = 1.0;
        double human_black_point = 0.35;
        double human_white_point = 0.75;
        bool resolution_specified = false;
        bool use_yuv = false;
        bool maximize_fps = false;
        bool use_source_fps = false;
        bool use_source_audio = false;
        bool fullscreen = false;
        bool repeat = false;
        bool enable_vsync = false;
        bool enable_screenshot = false;
        bool enable_playlist = false;
        bool enable_texture_cache = false;
        bool history_test = false;
        bool enable_3d = false;
        bool normalized_time = false;
        bool flip_output = false;
        bool png_output = false;
        bool encode_realtime = false;
        bool no_drop = false;
        bool copy_audio = false;
        bool mute_output = false;
        bool enable_audio = false;
        bool audio_input_specified = false;
        bool audio_warm_rate_specified = false;
        bool audio_output_specified = false;
        bool audio_pass_through_gain_specified = false;
        bool audio_recording_gain_specified = false;
        bool audio_pass_through = false;
        bool audio_repeat = false;
        bool audio_trunc = false;
        bool list_audio_devices = false;
        bool list_camera_devices = false;
        bool check_audio = false;
        bool midi_device_specified = false;
        bool midi_monitor = false;
        bool list_midi_devices = false;
        bool check_midi = false;
        bool gpu_buffer_specified = false;
        bool cuda_device_specified = false;
        bool list_gpu_filters = false;
        bool list_cuda_devices = false;
        bool check_cuda = false;
        bool check_dnn = false;
        bool human_background = false;
        bool human_black_specified = false;
        bool human_white_specified = false;
        bool list_encoders = false;
        bool display_filter = false;
        bool disable_counter = false;
        bool build_fix = false;
        bool build_prune = false;
        bool build_force = false;
        bool unbuffered_output = false;
        bool interface_shm = false;
        bool show_help = false;
        FrameRotation frame_rotation = FrameRotation::None;
        std::vector<int> shader_pass_indices;
        std::vector<std::string> shader_pass_files;
        std::vector<std::string> custom_uniform_overrides;
        std::vector<std::string> midi_cc_mappings;
        std::vector<int> gpu_filter_indices;
        std::string input_file;
        std::string graphic_file;
        std::string shader_directory;
        std::string fragment_shader;
        std::string compute_shader;
        std::string shader_file;
        std::string build_manifest;
        std::string build_directory;
        std::string glslc_executable = "glslc";
        std::string model_file;
        std::string playlist_file;
        std::string output_file;
        std::string encode_preset = "medium";
        std::string encode_tune;
        std::string encode_codec = "auto";
        std::string encode_params;
        std::string list_encoder_options;
        std::string audio_file;
        std::string record_audio_file;
        std::string midi_map_file;
        std::string edge_model;
        std::string human_model;
        std::string onnx_configuration;
        std::string snapshot_directory = ".";
        std::string resource_directory;
        std::string watermark_text;
        std::array<std::uint8_t, 3> watermark_color{255U, 0U, 150U};
    };

    [[nodiscard]] bool dimensions_supported(int width, int height);
    [[nodiscard]] int parseInteger(std::string_view text, std::string_view option);
    [[nodiscard]] double parseNumber(std::string_view text, std::string_view option);
    [[nodiscard]] Options parseOptions(int argc, char **argv);
    void printHelp(std::ostream &output);
    void printEncoders(std::ostream &output);
    [[nodiscard]] bool printEncoderOptions(std::string_view encoder_name, std::ostream &output, std::ostream &error);
    [[nodiscard]] std::string trim(std::string text);

} // namespace acmxvk

#endif
