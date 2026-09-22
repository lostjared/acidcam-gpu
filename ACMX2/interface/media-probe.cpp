#include "media-probe.hpp"

#include <QByteArray>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/display.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
#include <libavutil/samplefmt.h>
}

namespace acmx2::media {
namespace {

QString avError(int code) {
    char text[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, text, sizeof(text));
    return QString::fromUtf8(text);
}

QString rational(AVRational value) {
    return QStringLiteral("%1/%2").arg(value.num).arg(value.den);
}

QString seconds(qint64 timestamp, AVRational timeBase) {
    if (timestamp == AV_NOPTS_VALUE || !timeBase.den)
        return {};
    return QString::number(timestamp * av_q2d(timeBase), 'f', 6);
}

void addString(QJsonObject &object, const char *key, const char *value) {
    if (value && *value)
        object.insert(QString::fromLatin1(key), QString::fromUtf8(value));
}

QJsonObject dictionaryJson(const AVDictionary *dictionary) {
    QJsonObject result;
    const AVDictionaryEntry *entry = nullptr;
    while ((entry = av_dict_get(dictionary, "", entry, AV_DICT_IGNORE_SUFFIX)))
        result.insert(QString::fromUtf8(entry->key), QString::fromUtf8(entry->value));
    return result;
}

QJsonObject dispositionJson(int disposition) {
    struct Entry { int flag; const char *name; };
    static const Entry entries[] = {
        {AV_DISPOSITION_DEFAULT, "default"}, {AV_DISPOSITION_DUB, "dub"},
        {AV_DISPOSITION_ORIGINAL, "original"}, {AV_DISPOSITION_COMMENT, "comment"},
        {AV_DISPOSITION_LYRICS, "lyrics"}, {AV_DISPOSITION_KARAOKE, "karaoke"},
        {AV_DISPOSITION_FORCED, "forced"}, {AV_DISPOSITION_HEARING_IMPAIRED, "hearing_impaired"},
        {AV_DISPOSITION_VISUAL_IMPAIRED, "visual_impaired"}, {AV_DISPOSITION_CLEAN_EFFECTS, "clean_effects"},
        {AV_DISPOSITION_ATTACHED_PIC, "attached_pic"}, {AV_DISPOSITION_TIMED_THUMBNAILS, "timed_thumbnails"},
#ifdef AV_DISPOSITION_CAPTIONS
        {AV_DISPOSITION_CAPTIONS, "captions"}, {AV_DISPOSITION_DESCRIPTIONS, "descriptions"},
        {AV_DISPOSITION_METADATA, "metadata"}, {AV_DISPOSITION_DEPENDENT, "dependent"},
        {AV_DISPOSITION_STILL_IMAGE, "still_image"},
#endif
    };
    QJsonObject result;
    for (const Entry &entry : entries)
        result.insert(QString::fromLatin1(entry.name), !!(disposition & entry.flag));
    return result;
}

QString codecTag(unsigned int tag) {
    char text[AV_FOURCC_MAX_STRING_SIZE] = {};
    av_fourcc_make_string(text, tag);
    return QString::fromLatin1(text);
}

QJsonObject streamJson(const AVFormatContext *format, const AVStream *stream) {
    const AVCodecParameters *par = stream->codecpar;
    QJsonObject out;
    out.insert("index", stream->index);
    addString(out, "codec_name", avcodec_get_name(par->codec_id));
    if (const AVCodecDescriptor *descriptor = avcodec_descriptor_get(par->codec_id))
        addString(out, "codec_long_name", descriptor->long_name);
    addString(out, "profile", avcodec_profile_name(par->codec_id, par->profile));
    addString(out, "codec_type", av_get_media_type_string(par->codec_type));
    out.insert("codec_tag_string", codecTag(par->codec_tag));
    out.insert("codec_tag", QStringLiteral("0x%1").arg(par->codec_tag, 8, 16, QLatin1Char('0')));

    if (par->codec_type == AVMEDIA_TYPE_VIDEO) {
        out.insert("width", par->width);
        out.insert("height", par->height);
        if (par->sample_aspect_ratio.den) {
            out.insert("sample_aspect_ratio", rational(par->sample_aspect_ratio));
            const AVRational dar = av_mul_q(par->sample_aspect_ratio, AVRational{par->width, par->height});
            out.insert("display_aspect_ratio", rational(dar));
        }
        if (par->format >= 0)
            addString(out, "pix_fmt", av_get_pix_fmt_name(static_cast<AVPixelFormat>(par->format)));
        out.insert("level", par->level);
        addString(out, "color_range", av_color_range_name(par->color_range));
        addString(out, "color_space", av_color_space_name(par->color_space));
        addString(out, "color_transfer", av_color_transfer_name(par->color_trc));
        addString(out, "color_primaries", av_color_primaries_name(par->color_primaries));
        addString(out, "chroma_location", av_chroma_location_name(par->chroma_location));
        if (par->video_delay)
            out.insert("has_b_frames", par->video_delay);
        if (par->bits_per_coded_sample)
            out.insert("bits_per_raw_sample", QString::number(par->bits_per_coded_sample));
    } else if (par->codec_type == AVMEDIA_TYPE_AUDIO) {
        if (par->format >= 0)
            addString(out, "sample_fmt", av_get_sample_fmt_name(static_cast<AVSampleFormat>(par->format)));
        out.insert("sample_rate", QString::number(par->sample_rate));
#if LIBAVUTIL_VERSION_MAJOR >= 57
        out.insert("channels", par->ch_layout.nb_channels);
        char layout[256] = {};
        if (av_channel_layout_describe(&par->ch_layout, layout, sizeof(layout)) >= 0)
            addString(out, "channel_layout", layout);
#else
        out.insert("channels", par->channels);
        addString(out, "channel_layout", av_get_channel_name(par->channel_layout));
#endif
        if (par->frame_size)
            out.insert("frame_size", par->frame_size);
        if (par->bits_per_raw_sample)
            out.insert("bits_per_raw_sample", QString::number(par->bits_per_raw_sample));
    }

    out.insert("r_frame_rate", rational(stream->r_frame_rate));
    out.insert("avg_frame_rate", rational(stream->avg_frame_rate));
    out.insert("time_base", rational(stream->time_base));
    if (stream->start_time != AV_NOPTS_VALUE) {
        out.insert("start_pts", QString::number(stream->start_time));
        out.insert("start_time", seconds(stream->start_time, stream->time_base));
    }
    if (stream->duration != AV_NOPTS_VALUE) {
        out.insert("duration_ts", QString::number(stream->duration));
        out.insert("duration", seconds(stream->duration, stream->time_base));
    }
    if (par->bit_rate > 0)
        out.insert("bit_rate", QString::number(par->bit_rate));
    if (stream->nb_frames)
        out.insert("nb_frames", QString::number(stream->nb_frames));
    if (par->extradata_size)
        out.insert("extradata_size", par->extradata_size);
    out.insert("disposition", dispositionJson(stream->disposition));
    const QJsonObject tags = dictionaryJson(stream->metadata);
    if (!tags.isEmpty())
        out.insert("tags", tags);
    Q_UNUSED(format);
    return out;
}

QString ratioString(AVRational value) {
    return value.den ? rational(value) : QString();
}

QJsonArray frameSideDataJson(const AVFrame *frame) {
    QJsonArray items;
    for (int i = 0; i < frame->nb_side_data; ++i) {
        const AVFrameSideData *side = frame->side_data[i];
        QJsonObject item;
        item.insert("side_data_type", QString::fromUtf8(av_frame_side_data_name(side->type)));
        if (side->type == AV_FRAME_DATA_MASTERING_DISPLAY_METADATA &&
            side->size >= sizeof(AVMasteringDisplayMetadata)) {
            const auto *md = reinterpret_cast<const AVMasteringDisplayMetadata *>(side->data);
            if (md->has_primaries) {
                item.insert("red_x", ratioString(md->display_primaries[0][0]));
                item.insert("red_y", ratioString(md->display_primaries[0][1]));
                item.insert("green_x", ratioString(md->display_primaries[1][0]));
                item.insert("green_y", ratioString(md->display_primaries[1][1]));
                item.insert("blue_x", ratioString(md->display_primaries[2][0]));
                item.insert("blue_y", ratioString(md->display_primaries[2][1]));
                item.insert("white_point_x", ratioString(md->white_point[0]));
                item.insert("white_point_y", ratioString(md->white_point[1]));
            }
            if (md->has_luminance) {
                item.insert("min_luminance", ratioString(md->min_luminance));
                item.insert("max_luminance", ratioString(md->max_luminance));
            }
        } else if (side->type == AV_FRAME_DATA_CONTENT_LIGHT_LEVEL &&
                   side->size >= sizeof(AVContentLightMetadata)) {
            const auto *cll = reinterpret_cast<const AVContentLightMetadata *>(side->data);
            item.insert("max_content", static_cast<int>(cll->MaxCLL));
            item.insert("max_average", static_cast<int>(cll->MaxFALL));
        }
        items.append(item);
    }
    return items;
}

QJsonObject decodeFirstVideoFrame(AVFormatContext *format, int streamIndex) {
    QJsonObject result;
    AVStream *stream = format->streams[streamIndex];
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec)
        return result;
    AVCodecContext *context = avcodec_alloc_context3(codec);
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!context || !packet || !frame)
        goto cleanup;
    if (avcodec_parameters_to_context(context, stream->codecpar) < 0 || avcodec_open2(context, codec, nullptr) < 0)
        goto cleanup;
    while (av_read_frame(format, packet) >= 0) {
        if (packet->stream_index == streamIndex && avcodec_send_packet(context, packet) >= 0) {
            av_packet_unref(packet);
            if (avcodec_receive_frame(context, frame) >= 0) {
                const QJsonArray sideData = frameSideDataJson(frame);
                if (!sideData.isEmpty())
                    result.insert("side_data_list", sideData);
                break;
            }
        } else {
            av_packet_unref(packet);
        }
    }
cleanup:
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&context);
    return result;
}

} // namespace

ProbeResult probe(const QString &path, bool readFrameSideData) {
    ProbeResult result;
    AVFormatContext *format = nullptr;
    const QByteArray encodedPath = QFile::encodeName(QFileInfo(path).absoluteFilePath());
    int status = avformat_open_input(&format, encodedPath.constData(), nullptr, nullptr);
    if (status < 0) {
        result.error = QStringLiteral("Could not open media file: %1").arg(avError(status));
        return result;
    }
    status = avformat_find_stream_info(format, nullptr);
    if (status < 0) {
        result.error = QStringLiteral("Could not read stream information: %1").arg(avError(status));
        avformat_close_input(&format);
        return result;
    }

    QJsonObject formatJson;
    formatJson.insert("filename", QFileInfo(path).absoluteFilePath());
    formatJson.insert("nb_streams", static_cast<int>(format->nb_streams));
    formatJson.insert("nb_programs", static_cast<int>(format->nb_programs));
    if (format->iformat) {
        addString(formatJson, "format_name", format->iformat->name);
        addString(formatJson, "format_long_name", format->iformat->long_name);
    }
    if (format->start_time != AV_NOPTS_VALUE)
        formatJson.insert("start_time", QString::number(format->start_time / static_cast<double>(AV_TIME_BASE), 'f', 6));
    if (format->duration != AV_NOPTS_VALUE)
        formatJson.insert("duration", QString::number(format->duration / static_cast<double>(AV_TIME_BASE), 'f', 6));
    const qint64 fileSize = QFileInfo(path).size();
    if (fileSize >= 0)
        formatJson.insert("size", QString::number(fileSize));
    if (format->bit_rate > 0)
        formatJson.insert("bit_rate", QString::number(format->bit_rate));
    formatJson.insert("probe_score", format->probe_score);
    const QJsonObject formatTags = dictionaryJson(format->metadata);
    if (!formatTags.isEmpty())
        formatJson.insert("tags", formatTags);
    result.json.insert("format", formatJson);

    QJsonArray streams;
    int firstVideo = -1;
    for (unsigned int i = 0; i < format->nb_streams; ++i) {
        streams.append(streamJson(format, format->streams[i]));
        if (firstVideo < 0 && format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            firstVideo = static_cast<int>(i);
    }
    result.json.insert("streams", streams);

    if (readFrameSideData && firstVideo >= 0) {
        const QJsonObject frame = decodeFirstVideoFrame(format, firstVideo);
        result.json.insert("frames", QJsonArray{frame});
    }
    avformat_close_input(&format);
    return result;
}

VideoColorInfo probeFirstVideoColor(const QString &path) {
    VideoColorInfo result;
    const ProbeResult media = probe(path, false);
    if (!media) {
        result.error = media.error;
        return result;
    }
    const QJsonArray streams = media.json.value("streams").toArray();
    for (const QJsonValue &value : streams) {
        const QJsonObject stream = value.toObject();
        if (stream.value("codec_type").toString() != QStringLiteral("video"))
            continue;
        result.transfer = stream.value("color_transfer").toString();
        result.primaries = stream.value("color_primaries").toString();
        result.space = stream.value("color_space").toString();
        return result;
    }
    result.error = QStringLiteral("The file contains no video stream.");
    return result;
}

} // namespace acmx2::media
