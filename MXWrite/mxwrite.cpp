#include "mxwrite.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>
#ifdef MXWRITE_HAS_CUDA_COPY
#include <cuda_runtime.h>
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

std::mutex transfer_audio_mutex;

bool is_format_supported(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext)
        return false;
    return (strcmp(ext, ".mp4") == 0 ||
            strcmp(ext, ".mkv") == 0 ||
            strcmp(ext, ".avi") == 0 ||
            strcmp(ext, ".mov") == 0);
}

void cleanup_contexts(AVFormatContext *source_ctx,
                      AVFormatContext *dest_ctx,
                      AVFormatContext *output_ctx) {
    if (source_ctx)
        avformat_close_input(&source_ctx);
    if (dest_ctx)
        avformat_close_input(&dest_ctx);
    if (output_ctx) {
        if (!(output_ctx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&output_ctx->pb);
        avformat_free_context(output_ctx);
    }
}

void transfer_audio(std::string_view sourceAudioFile, std::string_view destVideoFile) {
    std::lock_guard<std::mutex> lock(transfer_audio_mutex);
    if (!is_format_supported(destVideoFile.data())) {
        std::cerr << "Unsupported output format. Supported formats: .mp4, .mkv, .avi, .mov\n";
        return;
    }

    AVFormatContext *source_ctx = nullptr, *dest_ctx = nullptr, *output_ctx = nullptr;
    int source_audio_idx = -1, dest_video_idx = -1, dest_audio_idx = -1;
    std::string temp_output = std::string(destVideoFile) + ".tmp";

    if (avformat_open_input(&source_ctx, sourceAudioFile.data(), nullptr, nullptr) != 0 ||
        avformat_open_input(&dest_ctx, destVideoFile.data(), nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input files\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    if (avformat_find_stream_info(source_ctx, nullptr) < 0 ||
        avformat_find_stream_info(dest_ctx, nullptr) < 0) {
        std::cerr << "Failed to find stream info\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    for (unsigned i = 0; i < source_ctx->nb_streams; ++i) {
        if (source_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            source_audio_idx = i;
            break;
        }
    }

    for (unsigned i = 0; i < dest_ctx->nb_streams; ++i) {
        AVMediaType type = dest_ctx->streams[i]->codecpar->codec_type;
        if (type == AVMEDIA_TYPE_VIDEO)
            dest_video_idx = i;
        else if (type == AVMEDIA_TYPE_AUDIO)
            dest_audio_idx = i;
    }

    if (source_audio_idx == -1 || dest_video_idx == -1) {
        std::cerr << "Required streams not found\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    const AVOutputFormat *output_fmt = av_guess_format(nullptr, destVideoFile.data(), nullptr);
    if (!output_fmt) {
        output_fmt = av_guess_format("mp4", nullptr, nullptr);
        if (!output_fmt) {
            std::cerr << "Failed to determine output format\n";
            cleanup_contexts(source_ctx, dest_ctx, output_ctx);
            return;
        }
    }

    if (avformat_alloc_output_context2(&output_ctx, output_fmt, nullptr, temp_output.c_str()) < 0) {
        std::cerr << "Failed to create output context\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    const AVCodec *audio_codec = avcodec_find_decoder(source_ctx->streams[source_audio_idx]->codecpar->codec_id);
    if (!audio_codec) {
        std::cerr << "Failed to find audio decoder\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    for (unsigned i = 0; i < dest_ctx->nb_streams; ++i) {
        if (dest_ctx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
            continue;
        }

        AVStream *dest_stream = dest_ctx->streams[i];
        AVStream *out_stream = avformat_new_stream(output_ctx, nullptr);
        if (!out_stream) {
            std::cerr << "Failed to create output stream\n";
            cleanup_contexts(source_ctx, dest_ctx, output_ctx);
            return;
        }

        if (avcodec_parameters_copy(out_stream->codecpar, dest_stream->codecpar) < 0) {
            std::cerr << "Failed to copy video parameters\n";
            cleanup_contexts(source_ctx, dest_ctx, output_ctx);
            return;
        }

        out_stream->time_base = dest_stream->time_base;
        out_stream->codecpar->codec_tag = 0;
    }

    AVStream *out_stream = avformat_new_stream(output_ctx, audio_codec);
    if (!out_stream) {
        std::cerr << "Failed to create audio stream\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    AVCodecParameters *source_params = source_ctx->streams[source_audio_idx]->codecpar;
    if (avcodec_parameters_copy(out_stream->codecpar, source_params) < 0) {
        std::cerr << "Failed to copy audio parameters\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }

    if (source_params->frame_size == 0) {
        out_stream->codecpar->frame_size = 1024;
    } else {
        out_stream->codecpar->frame_size = source_params->frame_size;
    }

    out_stream->time_base = source_ctx->streams[source_audio_idx]->time_base;
    out_stream->codecpar->codec_tag = 0;
    dest_audio_idx = out_stream->index;

    if (!(output_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&output_ctx->pb, temp_output.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Failed to open output file\n";
            cleanup_contexts(source_ctx, dest_ctx, output_ctx);
            return;
        }
    }
    if (avformat_write_header(output_ctx, nullptr) < 0) {
        std::cerr << "Failed to write header\n";
        cleanup_contexts(source_ctx, dest_ctx, output_ctx);
        return;
    }
    AVPacket packet;
    while (av_read_frame(dest_ctx, &packet) >= 0) {
        if (packet.stream_index == dest_audio_idx) {
            av_packet_unref(&packet);
            continue;
        }

        AVStream *in_stream = dest_ctx->streams[packet.stream_index];
        AVStream *out_stream = output_ctx->streams[packet.stream_index];
        av_packet_rescale_ts(&packet, in_stream->time_base, out_stream->time_base);

        if (av_interleaved_write_frame(output_ctx, &packet) < 0) {
            std::cerr << "Failed to write packet\n";
            av_packet_unref(&packet);
            cleanup_contexts(source_ctx, dest_ctx, output_ctx);
            return;
        }
        av_packet_unref(&packet);
    }

    int64_t video_duration_ts = 0;
    {
        AVStream *vid_stream = dest_ctx->streams[0];
        if (vid_stream->duration > 0) {
            video_duration_ts = av_rescale_q(vid_stream->duration, vid_stream->time_base, source_ctx->streams[source_audio_idx]->time_base);
        } else if (dest_ctx->duration > 0) {
            AVRational av_tb = {1, AV_TIME_BASE};
            video_duration_ts = av_rescale_q(dest_ctx->duration, av_tb, source_ctx->streams[source_audio_idx]->time_base);
        }
    }

    av_seek_frame(source_ctx, source_audio_idx, 0, AVSEEK_FLAG_BACKWARD);
    while (av_read_frame(source_ctx, &packet) >= 0) {
        if (packet.stream_index == source_audio_idx) {
            if (video_duration_ts > 0 && packet.pts != AV_NOPTS_VALUE && packet.pts > video_duration_ts) {
                av_packet_unref(&packet);
                break;
            }
            AVStream *in_stream = source_ctx->streams[packet.stream_index];
            AVStream *out_stream = output_ctx->streams[dest_audio_idx];
            av_packet_rescale_ts(&packet, in_stream->time_base, out_stream->time_base);
            packet.stream_index = dest_audio_idx;

            if (av_interleaved_write_frame(output_ctx, &packet) < 0) {
                std::cerr << "Failed to write audio packet\n";
                av_packet_unref(&packet);
                cleanup_contexts(source_ctx, dest_ctx, output_ctx);
                return;
            }
        }
        av_packet_unref(&packet);
    }
    av_write_trailer(output_ctx);
    cleanup_contexts(source_ctx, dest_ctx, output_ctx);
    std::remove(destVideoFile.data());
    std::rename(temp_output.c_str(), destVideoFile.data());
}

void Writer::calculateFPSFraction(float fps, int &fps_num, int &fps_den) {
    const float epsilon = 0.001f;
    fps_den = 1001;
    if (std::fabs(fps - 29.97f) < epsilon) {
        fps_num = 30000;
        fps_den = 1001;
    } else if (std::fabs(fps - 59.94f) < epsilon) {
        fps_num = 60000;
        fps_den = 1001;
    } else {
        float precision = 1000.0f;
        fps_num = static_cast<int>(std::round(fps * precision));
        fps_den = static_cast<int>(precision);
        int gcd = std::gcd(fps_num, fps_den);
        fps_num /= gcd;
        fps_den /= gcd;
    }
}

bool Writer::open(const std::string &filename, int w, int h, float fps, const char *crf) {
    std::lock_guard<std::mutex> lock(writer_mutex);
    return openInternal(filename, w, h, fps, crf, false);
}

bool Writer::open_ts(const std::string &filename, int w, int h, float fps, const char *crf) {
    std::lock_guard<std::mutex> lock(writer_mutex);
    return openInternal(filename, w, h, fps, crf, true);
}

bool Writer::initHardwareEncoding() {
    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        return false;
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
    codec_ctx->sw_pix_fmt = AV_PIX_FMT_RGBA;

    hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ctx) {
        return false;
    }

    auto *frames_ctx = reinterpret_cast<AVHWFramesContext *>(hw_frames_ctx->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_RGBA;
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        return false;
    }

    codec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);

    upload_sw_frame = av_frame_alloc();
    if (!upload_sw_frame) {
        return false;
    }
    upload_sw_frame->format = AV_PIX_FMT_RGBA;
    upload_sw_frame->width = width;
    upload_sw_frame->height = height;
    if (av_frame_get_buffer(upload_sw_frame, 32) < 0) {
        return false;
    }

    return true;
}

bool Writer::openInternal(const std::string &filename, int w, int h, float fps, const char *crf, bool ts_mode) {
    avformat_network_init();
    av_log_set_level(AV_LOG_ERROR);
    opened = false;
    stop_requested = false;
    frame_count = 0;
    last_duration = 0.0;

    while (!encode_queue.empty()) {
        releaseFrame(encode_queue.front());
        encode_queue.pop();
    }

    if (avformat_alloc_output_context2(&format_ctx, nullptr, "mp4", filename.c_str()) < 0) {
        std::cerr << "Could not allocate output context.\n";
        return false;
    }

    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    bool wants_hw = (codec != nullptr);
    if (!codec) {
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    }
    if (!codec) {
        std::cerr << "Could not find H.264 encoder.\n";
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return false;
    }

    stream = avformat_new_stream(format_ctx, codec);
    if (!stream) {
        std::cerr << "Could not create new stream.\n";
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return false;
    }

    width = w;
    height = h;
    calculateFPSFraction(fps, fps_num, fps_den);

    AVRational tb = {fps_den, fps_num};
    stream->time_base = tb;

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Could not allocate codec context.\n";
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return false;
    }

    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->time_base = stream->time_base;
    codec_ctx->framerate = AVRational{fps_num, fps_den};
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->gop_size = 30;
    codec_ctx->max_b_frames = 0;
    codec_ctx->thread_count = std::max(1u, std::thread::hardware_concurrency());
    codec_ctx->thread_type = FF_THREAD_SLICE;
    codec_ctx->slices = 4;
    codec_ctx->delay = 0;

    if (ts_mode) {
        codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    }

    if (wants_hw) {
        av_opt_set(codec_ctx->priv_data, "preset", "p4", 0);
        av_opt_set(codec_ctx->priv_data, "tune", "ll", 0);
        av_opt_set(codec_ctx->priv_data, "rc", "vbr", 0);
        if (crf && *crf) {
            av_opt_set(codec_ctx->priv_data, "cq", crf, 0);
        }
        av_opt_set(codec_ctx->priv_data, "zerolatency", "1", 0);

        if (initHardwareEncoding()) {
            use_hw_encode = true;
            std::cout << "MXWrite: hardware encoder selected (h264_nvenc)\n";
        } else {
            std::cerr << "MXWrite: h264_nvenc present but CUDA context failed, falling back to software H.264\n";
            av_buffer_unref(&hw_frames_ctx);
            av_buffer_unref(&hw_device_ctx);
            avcodec_free_context(&codec_ctx);

            codec = avcodec_find_encoder(AV_CODEC_ID_H264);
            if (!codec) {
                std::cerr << "Could not find software H.264 encoder fallback.\n";
                avformat_free_context(format_ctx);
                format_ctx = nullptr;
                return false;
            }

            codec_ctx = avcodec_alloc_context3(codec);
            if (!codec_ctx) {
                std::cerr << "Could not allocate fallback codec context.\n";
                avformat_free_context(format_ctx);
                format_ctx = nullptr;
                return false;
            }

            codec_ctx->width = width;
            codec_ctx->height = height;
            codec_ctx->time_base = stream->time_base;
            codec_ctx->framerate = AVRational{fps_num, fps_den};
            codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
            codec_ctx->gop_size = 30;
            codec_ctx->max_b_frames = 0;
            codec_ctx->thread_count = std::max(1u, std::thread::hardware_concurrency());
            codec_ctx->thread_type = FF_THREAD_SLICE;
            codec_ctx->slices = 4;
            codec_ctx->delay = 0;

            if (ts_mode) {
                codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
            }
        }
    }

    if (!use_hw_encode) {
        av_opt_set(codec_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(codec_ctx->priv_data, "tune", "zerolatency", 0);
        av_opt_set(codec_ctx->priv_data, "crf", crf, 0);
        av_opt_set(codec_ctx->priv_data, "x264-params", "bframes=0:ref=1:me=dia:subme=0", 0);
        av_opt_set(codec_ctx->priv_data, "force_cfr", "1", 0);
    }

    time_base = tb;

    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec.\n";
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return false;
    }
    if (avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) {
        std::cerr << "Could not copy codec parameters.\n";
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return false;
    }
    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&format_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Could not open output file: " << filename << "\n";
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            format_ctx = nullptr;
            return false;
        }
    }
    if (avformat_write_header(format_ctx, nullptr) < 0) {
        std::cerr << "Error writing MP4 header.\n";
        avio_closep(&format_ctx->pb);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        format_ctx = nullptr;
        return false;
    }

    if (!use_hw_encode) {
        frameYUV = av_frame_alloc();
        if (!frameYUV) {
            std::cerr << "Could not allocate YUV frame.\n";
            avio_closep(&format_ctx->pb);
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            format_ctx = nullptr;
            return false;
        }
        frameYUV->format = AV_PIX_FMT_YUV420P;
        frameYUV->width = width;
        frameYUV->height = height;
        if (av_frame_get_buffer(frameYUV, 32) < 0) {
            std::cerr << "Could not allocate frame buffer for YUV frame.\n";
            av_frame_free(&frameYUV);
            avio_closep(&format_ctx->pb);
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            format_ctx = nullptr;
            return false;
        }

        sws_ctx = sws_getContext(width, height, AV_PIX_FMT_RGBA, width, height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) {
            std::cerr << "Could not initialize conversion context.\n";
            av_frame_free(&frameYUV);
            avio_closep(&format_ctx->pb);
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            format_ctx = nullptr;
            return false;
        }
    }

    opened = true;
    recordingStart = std::chrono::steady_clock::now();
    startEncoderThread();
    return true;
}

void Writer::write(void *rgba_buffer) {
    if (!rgba_buffer) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(writer_mutex);
        if (!opened) {
            return;
        }
    }

    AVFrame *queued_frame = av_frame_alloc();
    if (!queued_frame) {
        std::cerr << "Writer: failed to allocate queued frame\n";
        return;
    }

    if (use_hw_encode) {
        queued_frame->format = AV_PIX_FMT_CUDA;
        queued_frame->width = width;
        queued_frame->height = height;
        if (av_hwframe_get_buffer(hw_frames_ctx, queued_frame, 0) < 0) {
            std::cerr << "Writer: failed to allocate CUDA frame from hardware pool\n";
            releaseFrame(queued_frame);
            return;
        }

        if (av_frame_make_writable(upload_sw_frame) < 0) {
            std::cerr << "Writer: software upload frame not writable\n";
            releaseFrame(queued_frame);
            return;
        }

        const auto *src = static_cast<const uint8_t *>(rgba_buffer);
        for (int y = 0; y < height; ++y) {
            std::memcpy(upload_sw_frame->data[0] + static_cast<size_t>(y) * upload_sw_frame->linesize[0],
                        src + static_cast<size_t>(y) * static_cast<size_t>(width) * 4,
                        static_cast<size_t>(width) * 4);
        }

        if (av_hwframe_transfer_data(queued_frame, upload_sw_frame, 0) < 0) {
            std::cerr << "Writer: failed to transfer RGBA system frame to CUDA frame\n";
            releaseFrame(queued_frame);
            return;
        }
    } else {
        queued_frame->format = AV_PIX_FMT_RGBA;
        queued_frame->width = width;
        queued_frame->height = height;
        if (av_frame_get_buffer(queued_frame, 32) < 0) {
            std::cerr << "Writer: failed to allocate queued RGBA frame buffer\n";
            releaseFrame(queued_frame);
            return;
        }
        if (av_frame_make_writable(queued_frame) < 0) {
            std::cerr << "Writer: queued RGBA frame not writable\n";
            releaseFrame(queued_frame);
            return;
        }

        const auto *src = static_cast<const uint8_t *>(rgba_buffer);
        for (int y = 0; y < height; ++y) {
            std::memcpy(queued_frame->data[0] + static_cast<size_t>(y) * queued_frame->linesize[0],
                        src + static_cast<size_t>(y) * static_cast<size_t>(width) * 4,
                        static_cast<size_t>(width) * 4);
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (stop_requested || encode_queue.size() >= MAX_QUEUE_SIZE) {
            static int drop_counter = 0;
            if (++drop_counter % 30 == 0) {
                std::cerr << "Writer: dropped " << drop_counter << " frames (encoder queue full)\n";
            }
            releaseFrame(queued_frame);
            return;
        }
        queued_frame->pts = frame_count++;
        encode_queue.push(queued_frame);
    }

    queue_cv.notify_one();
}

bool Writer::write_cuda_rgba(void *cuda_rgba_buffer, int src_stride, bool bottom_up) {
    if (!cuda_rgba_buffer || src_stride <= 0) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(writer_mutex);
        if (!opened || !use_hw_encode) {
            return false;
        }
    }

    AVFrame *queued_frame = av_frame_alloc();
    if (!queued_frame) {
        std::cerr << "Writer: failed to allocate queued CUDA frame\n";
        return false;
    }

    queued_frame->format = AV_PIX_FMT_CUDA;
    queued_frame->width = width;
    queued_frame->height = height;

    if (av_hwframe_get_buffer(hw_frames_ctx, queued_frame, 0) < 0) {
        std::cerr << "Writer: failed to allocate CUDA frame from hardware pool\n";
        releaseFrame(queued_frame);
        return false;
    }

#ifdef MXWRITE_HAS_CUDA_COPY
    if (!bottom_up) {
        const auto copy_err = cudaMemcpy2D(
            queued_frame->data[0],
            static_cast<size_t>(queued_frame->linesize[0]),
            cuda_rgba_buffer,
            static_cast<size_t>(src_stride),
            static_cast<size_t>(width) * 4,
            static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice);

        if (copy_err != cudaSuccess) {
            std::cerr << "Writer: cudaMemcpy2D device upload failed: " << cudaGetErrorString(copy_err) << "\n";
            releaseFrame(queued_frame);
            return false;
        }
    } else {
        auto *src_base = static_cast<unsigned char *>(cuda_rgba_buffer);
        auto *dst_base = queued_frame->data[0];
        const size_t row_bytes = static_cast<size_t>(width) * 4;

        for (int y = 0; y < height; ++y) {
            auto *src_row = src_base + static_cast<size_t>(height - 1 - y) * static_cast<size_t>(src_stride);
            auto *dst_row = dst_base + static_cast<size_t>(y) * static_cast<size_t>(queued_frame->linesize[0]);
            const auto row_copy_err = cudaMemcpy(dst_row, src_row, row_bytes, cudaMemcpyDeviceToDevice);
            if (row_copy_err != cudaSuccess) {
                std::cerr << "Writer: cudaMemcpy row upload failed: " << cudaGetErrorString(row_copy_err) << "\n";
                releaseFrame(queued_frame);
                return false;
            }
        }
    }
#else
    (void)bottom_up;
    std::cerr << "Writer: CUDA copy support disabled at build time\n";
    releaseFrame(queued_frame);
    return false;
#endif

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (stop_requested || encode_queue.size() >= MAX_QUEUE_SIZE) {
            static int drop_counter = 0;
            if (++drop_counter % 30 == 0) {
                std::cerr << "Writer: dropped " << drop_counter << " frames (encoder queue full)\n";
            }
            releaseFrame(queued_frame);
            return false;
        }
        queued_frame->pts = frame_count++;
        encode_queue.push(queued_frame);
    }

    queue_cv.notify_one();
    return true;
}

void Writer::write_ts(void *rgba_buffer) {
    write(rgba_buffer);
}

void Writer::startEncoderThread() {
    stop_requested = false;
    encode_thread = std::jthread([this](std::stop_token st) {
        encodeLoop(st);
    });
}

void Writer::stopEncoderThread() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop_requested = true;
    }
    queue_cv.notify_all();

    if (encode_thread.joinable()) {
        encode_thread.request_stop();
        encode_thread.join();
    }
}

void Writer::releaseFrame(AVFrame *f) {
    if (!f) {
        return;
    }
    av_frame_free(&f);
}

void Writer::drainEncoderPackets() {
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        std::cerr << "Writer: failed to allocate packet\n";
        return;
    }

    while (true) {
        int ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            std::cerr << "Writer: error receiving packet: " << ret << "\n";
            break;
        }

        av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);
        pkt->stream_index = stream->index;

        if (av_interleaved_write_frame(format_ctx, pkt) < 0) {
            std::cerr << "Writer: error writing frame\n";
            av_packet_unref(pkt);
            break;
        }
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
}

void Writer::encodeAndWriteFrame(AVFrame *in_frame) {
    if (!in_frame) {
        return;
    }

    AVFrame *encode_frame = in_frame;
    if (!use_hw_encode) {
        const uint8_t *src_data[1] = {in_frame->data[0]};
        int src_linesize[1] = {in_frame->linesize[0]};
        sws_scale(sws_ctx, src_data, src_linesize, 0, height, frameYUV->data, frameYUV->linesize);
        frameYUV->pts = in_frame->pts;
        encode_frame = frameYUV;
    }

    int ret = avcodec_send_frame(codec_ctx, encode_frame);
    if (ret < 0) {
        std::cerr << "Writer: error sending frame to encoder: " << ret << "\n";
        return;
    }

    drainEncoderPackets();
}

void Writer::encodeLoop(std::stop_token stop_token) {
    while (true) {
        AVFrame *frame = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this, &stop_token]() {
                return stop_requested || stop_token.stop_requested() || !encode_queue.empty();
            });

            if ((stop_requested || stop_token.stop_requested()) && encode_queue.empty()) {
                break;
            }

            frame = encode_queue.front();
            encode_queue.pop();
        }

        encodeAndWriteFrame(frame);
        releaseFrame(frame);
    }

    if (codec_ctx) {
        const int flush_ret = avcodec_send_frame(codec_ctx, nullptr);
        if (flush_ret >= 0) {
            drainEncoderPackets();
        }
    }
}
void Writer::close() {
    std::lock_guard<std::mutex> lock(writer_mutex);
    if (!opened) {
        return;
    }

    stopEncoderThread();

    if (stream && stream->duration > 0) {
        last_duration = static_cast<double>(stream->duration) * av_q2d(stream->time_base);
    } else if (fps_num > 0 && fps_den > 0) {
        last_duration = static_cast<double>(frame_count) * static_cast<double>(fps_den) / static_cast<double>(fps_num);
    }

    av_write_trailer(format_ctx);

    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&format_ctx->pb);
    }

    av_frame_free(&frameRGBA);
    av_frame_free(&frameYUV);
    sws_freeContext(sws_ctx);
    av_frame_free(&upload_sw_frame);
    avcodec_free_context(&codec_ctx);
    av_buffer_unref(&hw_frames_ctx);
    av_buffer_unref(&hw_device_ctx);
    avformat_free_context(format_ctx);


    while (!encode_queue.empty()) {
        releaseFrame(encode_queue.front());
        encode_queue.pop();
    }
    opened = false;
    format_ctx = nullptr;
    codec_ctx = nullptr;
    sws_ctx = nullptr;
    frameRGBA = nullptr;
    frameYUV = nullptr;
    upload_sw_frame = nullptr;
    use_hw_encode = false;
    stop_requested = false;
}

double Writer::get_duration() const {
    if (!opened && last_duration > 0.0) {
        return last_duration;
    }
    if (stream && stream->duration > 0) {
        return static_cast<double>(stream->duration) * av_q2d(stream->time_base);
    }
    if (fps_num > 0 && fps_den > 0) {
        return static_cast<double>(frame_count) * static_cast<double>(fps_den) / static_cast<double>(fps_num);
    }
    return 0.0;
}