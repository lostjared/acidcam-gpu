#ifndef FFWRITE_HPP
#define FFWRITE_HPP
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>

struct Frame_Data {
    void *data;
    std::chrono::steady_clock::time_point capture_time;
};

class Writer {
  public:
    Writer() = default;

    bool open(const std::string &filename, int width, int height, float fps, const char *crf);
    void write(void *rgba_buffer);
    bool write_cuda_rgba(void *cuda_rgba_buffer, int src_stride, bool bottom_up = false);
    bool open_ts(const std::string &filename, int width, int height, float fps, const char *crf);
    void write_ts(void *rgba_buffer);
    void close();
    bool is_open() const { return opened; }
    /// @brief True when the active encoder backend is hardware (NVENC).
    bool is_hardware_encode() const { return use_hw_encode; }
    int64_t get_frame_count() const { return frame_count; }
    double get_duration() const;
    ~Writer() {
        if (is_open()) {
            close();
            opened = false;
        }
    }

  private:
    bool opened{false};
    int width = 0;
    int height = 0;
    int fps_num = 0;
    int fps_den = 0;
    int64_t frame_count = 0;
    double last_duration = 0.0;
    AVFormatContext *format_ctx = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    AVStream *stream = nullptr;
    AVFrame *frameYUV = nullptr;
    AVFrame *frameRGBA = nullptr;
    AVFrame *upload_sw_frame = nullptr;
    AVBufferRef *hw_device_ctx = nullptr;
    AVBufferRef *hw_frames_ctx = nullptr;
    bool use_hw_encode = false;
    SwsContext *sws_ctx = nullptr;
    AVRational time_base;
    void calculateFPSFraction(float fps, int &fps_num, int &fps_den);
    std::chrono::steady_clock::time_point recordingStart;

    std::queue<AVFrame *> encode_queue;
    static constexpr size_t MAX_QUEUE_SIZE = 30;
    std::condition_variable queue_cv;
    std::jthread encode_thread;

    std::mutex queue_mutex{};
    std::mutex writer_mutex{};
    bool stop_requested = false;

    bool openInternal(const std::string &filename, int w, int h, float fps, const char *crf, bool ts_mode);
    bool initHardwareEncoding();
    void startEncoderThread();
    void stopEncoderThread();
    void encodeLoop(std::stop_token stop_token);
    void encodeAndWriteFrame(AVFrame *in_frame);
    void drainEncoderPackets();
    void releaseFrame(AVFrame *f);
};

extern void transfer_audio(std::string_view sourceAudioFile, std::string_view destVideoFile);
extern void cleanup_contexts(AVFormatContext *source_ctx, AVFormatContext *dest_ctx, AVFormatContext *output_ctx);

#endif