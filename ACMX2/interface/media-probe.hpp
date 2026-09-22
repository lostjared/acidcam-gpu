#ifndef ACMX2_MEDIA_PROBE_HPP
#define ACMX2_MEDIA_PROBE_HPP

#include <QJsonObject>
#include <QString>

namespace acmx2::media {

struct ProbeResult {
    QJsonObject json;
    QString error;

    explicit operator bool() const { return error.isEmpty(); }
};

struct VideoColorInfo {
    QString transfer;
    QString primaries;
    QString space;
    QString error;

    explicit operator bool() const { return error.isEmpty(); }
};

// Uses the linked FFmpeg libraries directly; no ffprobe executable is needed.
ProbeResult probe(const QString &path, bool readFrameSideData = true);
VideoColorInfo probeFirstVideoColor(const QString &path);

} // namespace acmx2::media

#endif
