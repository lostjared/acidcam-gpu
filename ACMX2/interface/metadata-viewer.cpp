/**
 * @file metadata-viewer.cpp
 * @brief Implementation of the media metadata viewer dialog.
 */

#include "metadata-viewer.hpp"
#include "custom_style.hpp"

#include <QApplication>
#include <QClipboard>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QTabBar>
#include <QTabWidget>
#include <QTextStream>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

namespace {

    QString jsonValueToString(const QJsonValue &v) {
        switch (v.type()) {
        case QJsonValue::Bool:
            return v.toBool() ? QStringLiteral("true") : QStringLiteral("false");
        case QJsonValue::Double: {
            const double d = v.toDouble();
            if (d == static_cast<qint64>(d))
                return QString::number(static_cast<qint64>(d));
            return QString::number(d, 'g', 10);
        }
        case QJsonValue::String:
            return v.toString();
        case QJsonValue::Null:
            return QStringLiteral("null");
        case QJsonValue::Array:
        case QJsonValue::Object:
            return QString::fromUtf8(QJsonDocument(v.isArray()
                                                       ? QJsonDocument(v.toArray())
                                                       : QJsonDocument(v.toObject()))
                                         .toJson(QJsonDocument::Compact));
        default:
            return QString();
        }
    }

    QString humanBitrate(qint64 bps) {
        if (bps <= 0)
            return QString();
        if (bps >= 1'000'000)
            return QString::number(bps / 1'000'000.0, 'f', 2) + " Mb/s";
        if (bps >= 1'000)
            return QString::number(bps / 1'000.0, 'f', 1) + " kb/s";
        return QString::number(bps) + " b/s";
    }

    QString humanSize(qint64 bytes) {
        if (bytes <= 0)
            return QString();
        constexpr double KiB = 1024.0;
        constexpr double MiB = KiB * 1024.0;
        constexpr double GiB = MiB * 1024.0;
        if (bytes >= GiB)
            return QString::number(bytes / GiB, 'f', 2) + " GiB";
        if (bytes >= MiB)
            return QString::number(bytes / MiB, 'f', 2) + " MiB";
        if (bytes >= KiB)
            return QString::number(bytes / KiB, 'f', 1) + " KiB";
        return QString::number(bytes) + " B";
    }

    struct StreamRow {
        int index = 0;
        QString type;
        QString codec;
        QString profile;
        QString fmtPix;
        QString sizeRate;
        QString color;
        QString bitrate;
    };

    QList<StreamRow> collectStreams(const QJsonArray &streams) {
        QList<StreamRow> rows;
        rows.reserve(streams.size());
        for (int i = 0; i < streams.size(); ++i) {
            const QJsonObject s = streams.at(i).toObject();
            StreamRow r;
            r.index = i;
            r.type = s.value("codec_type").toString();
            r.codec = s.value("codec_name").toString();
            r.profile = s.value("profile").toString();
            if (r.type == "video") {
                r.fmtPix = s.value("pix_fmt").toString();
                const int w = s.value("width").toInt();
                const int h = s.value("height").toInt();
                const QString fps = s.value("avg_frame_rate").toString();
                r.sizeRate = QString("%1x%2 @ %3").arg(w).arg(h).arg(fps);
                QStringList tags;
                const QString cs = s.value("color_space").toString();
                const QString cp = s.value("color_primaries").toString();
                const QString tr = s.value("color_transfer").toString();
                if (!cp.isEmpty())
                    tags << cp;
                if (!tr.isEmpty())
                    tags << tr;
                if (!cs.isEmpty())
                    tags << cs;
                r.color = tags.join('/');
            } else if (r.type == "audio") {
                r.fmtPix = s.value("sample_fmt").toString();
                const int sr = s.value("sample_rate").toString().toInt();
                const int ch = s.value("channels").toInt();
                r.sizeRate = QString("%1 Hz, %2 ch").arg(sr).arg(ch);
            }
            const qint64 sBrate = s.value("bit_rate").toString().toLongLong();
            r.bitrate = humanBitrate(sBrate);
            rows.append(r);
        }
        return rows;
    }

    struct HdrInfo {
        bool hasMastering = false;
        QString redX, redY, greenX, greenY, blueX, blueY, whiteX, whiteY, maxLum, minLum;
        bool hasCll = false;
        QString maxCll, maxFall;
    };

    HdrInfo collectHdr(const QJsonArray &frames) {
        HdrInfo info;
        if (frames.isEmpty())
            return info;
        const QJsonArray sideData = frames.first().toObject().value("side_data_list").toArray();
        for (const QJsonValue &sv : sideData) {
            const QJsonObject so = sv.toObject();
            const QString type = so.value("side_data_type").toString();
            if (type == "Mastering display metadata") {
                info.hasMastering = true;
                info.redX = so.value("red_x").toString();
                info.redY = so.value("red_y").toString();
                info.greenX = so.value("green_x").toString();
                info.greenY = so.value("green_y").toString();
                info.blueX = so.value("blue_x").toString();
                info.blueY = so.value("blue_y").toString();
                info.whiteX = so.value("white_point_x").toString();
                info.whiteY = so.value("white_point_y").toString();
                info.maxLum = so.value("max_luminance").toString();
                info.minLum = so.value("min_luminance").toString();
            } else if (type == "Content light level metadata") {
                info.hasCll = true;
                info.maxCll = so.value("max_content").toVariant().toString();
                info.maxFall = so.value("max_average").toVariant().toString();
            }
        }
        return info;
    }

} // namespace

MetadataViewer::MetadataViewer(QWidget *parent) : QDialog(parent) {
    setWindowTitle(tr("Media Metadata Viewer"));
    resize(900, 700);
    acmx2::applyCustomStyleIfEnabled(this);

    auto *root = new QVBoxLayout(this);

    // File selection row.
    auto *fileRow = new QHBoxLayout();
    pathEdit = new QLineEdit(this);
    pathEdit->setPlaceholderText(tr("Path to a media file (mp4, mkv, mov, ...)"));
    browseButton = new QPushButton(tr("Browse..."), this);
    analyzeButton = new QPushButton(tr("Analyze"), this);
    fileRow->addWidget(pathEdit, 1);
    fileRow->addWidget(browseButton);
    fileRow->addWidget(analyzeButton);
    root->addLayout(fileRow);

    // Tabbed display: parsed tree + Markdown/HTML/text previews.
    tabs = new QTabWidget(this);

    tree = new QTreeWidget(this);
    tree->setColumnCount(2);
    tree->setHeaderLabels({tr("Field"), tr("Value")});
    tree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    tree->header()->setSectionResizeMode(1, QHeaderView::Stretch);
    tabs->addTab(tree, tr("Tree"));

    markdownPreview = new QPlainTextEdit(this);
    markdownPreview->setLineWrapMode(QPlainTextEdit::NoWrap);
    markdownTabIndex = tabs->addTab(markdownPreview, tr("Markdown"));

    htmlPreview = new QPlainTextEdit(this);
    htmlPreview->setLineWrapMode(QPlainTextEdit::NoWrap);
    htmlTabIndex = tabs->addTab(htmlPreview, tr("HTML"));

    textPreview = new QPlainTextEdit(this);
    textPreview->setLineWrapMode(QPlainTextEdit::NoWrap);
    textTabIndex = tabs->addTab(textPreview, tr("Text"));

    // Monospace font for previews.
    QFont monoFont("Courier New");
    monoFont.setStyleHint(QFont::Monospace);
    monoFont.setPointSize(10);
    markdownPreview->setFont(monoFont);
    htmlPreview->setFont(monoFont);
    textPreview->setFont(monoFont);

    // Match application palette so previews and tabs blend with the dialog
    // background instead of falling back to the style's default light Base/Button.
    const QPalette appPalette = qApp->palette();
    QPalette viewPalette = appPalette;
    viewPalette.setColor(QPalette::Base, appPalette.color(QPalette::Window));
    viewPalette.setColor(QPalette::Text, Qt::white);
    viewPalette.setColor(QPalette::WindowText, Qt::white);
    viewPalette.setColor(QPalette::ButtonText, Qt::white);
    viewPalette.setColor(QPalette::HighlightedText, Qt::white);
    for (auto *w : {static_cast<QWidget *>(markdownPreview),
                    static_cast<QWidget *>(htmlPreview),
                    static_cast<QWidget *>(textPreview),
                    static_cast<QWidget *>(tree)}) {
        w->setPalette(viewPalette);
        if (auto *vp = w->findChild<QWidget *>("qt_scrollarea_viewport"))
            vp->setPalette(viewPalette);
    }

    root->addWidget(tabs, 1);

    // Action row.
    auto *actionRow = new QHBoxLayout();
    copyButton = new QPushButton(this);
    closeButton = new QPushButton(tr("Close"), this);
    copyButton->setEnabled(false);
    actionRow->addStretch(1);
    actionRow->addWidget(copyButton);
    actionRow->addWidget(closeButton);
    root->addLayout(actionRow);

    connect(browseButton, &QPushButton::clicked, this, &MetadataViewer::browseFile);
    connect(analyzeButton, &QPushButton::clicked, this, &MetadataViewer::analyzeFile);
    connect(copyButton, &QPushButton::clicked, this, &MetadataViewer::copyCurrentTab);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(pathEdit, &QLineEdit::returnPressed, this, &MetadataViewer::analyzeFile);
    connect(tabs, &QTabWidget::currentChanged, this, [this](int) { updateCopyButtonLabel(); });

    updateCopyButtonLabel();
}

void MetadataViewer::updateCopyButtonLabel() {
    const int idx = tabs ? tabs->currentIndex() : -1;
    QString label;
    if (idx == markdownTabIndex)
        label = tr("Copy Markdown to Clipboard");
    else if (idx == htmlTabIndex)
        label = tr("Copy HTML to Clipboard");
    else if (idx == textTabIndex)
        label = tr("Copy Text to Clipboard");
    else
        label = tr("Copy to Clipboard");
    copyButton->setText(label);
}

void MetadataViewer::browseFile() {
    const QString start = pathEdit->text().isEmpty()
                              ? QString()
                              : QFileInfo(pathEdit->text()).absolutePath();
    const QString chosen = QFileDialog::getOpenFileName(
        this, tr("Select Media File"), start,
        tr("Media files (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.ts *.flv);;All files (*.*)"));
    if (!chosen.isEmpty()) {
        pathEdit->setText(chosen);
    }
}

void MetadataViewer::analyzeFile() {
    const QString path = pathEdit->text().trimmed();
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        QMessageBox::warning(this, tr("Metadata Viewer"),
                             tr("Please select an existing media file."));
        return;
    }

    QProcess proc;
    QStringList args;
    // Pull format + streams + frame-level side data (HDR mastering, MaxCLL, etc).
    args << "-v" << "error"
         << "-print_format" << "json"
         << "-show_format"
         << "-show_streams"
         << "-show_frames"
         << "-read_intervals" << "%+#1"
         << "-show_entries" << "frame=side_data_list:format:stream"
         << path;
    proc.start("ffprobe", args);
    if (!proc.waitForStarted(5000)) {
        QMessageBox::critical(this, tr("Metadata Viewer"),
                              tr("Failed to launch ffprobe. Is it installed and on PATH?"));
        return;
    }
    if (!proc.waitForFinished(30000)) {
        proc.kill();
        QMessageBox::critical(this, tr("Metadata Viewer"),
                              tr("ffprobe timed out."));
        return;
    }
    if (proc.exitStatus() != QProcess::NormalExit || proc.exitCode() != 0) {
        const QString err = QString::fromUtf8(proc.readAllStandardError());
        QMessageBox::critical(this, tr("Metadata Viewer"),
                              tr("ffprobe failed:\n%1").arg(err.isEmpty() ? tr("(no error output)") : err));
        return;
    }

    QJsonParseError parseErr;
    const QByteArray out = proc.readAllStandardOutput();
    const QJsonDocument doc = QJsonDocument::fromJson(out, &parseErr);
    if (parseErr.error != QJsonParseError::NoError || !doc.isObject()) {
        QMessageBox::critical(this, tr("Metadata Viewer"),
                              tr("Failed to parse ffprobe JSON: %1").arg(parseErr.errorString()));
        return;
    }

    const QJsonObject root = doc.object();
    populateTree(root);
    markdownPreview->setPlainText(buildMarkdown(root));
    htmlPreview->setPlainText(buildHtml(root));
    textPreview->setPlainText(buildPlainText(root));
    copyButton->setEnabled(true);
}

void MetadataViewer::addJsonValue(QTreeWidgetItem *parent, const QString &key,
                                  const QJsonValue &value) {
    auto *item = new QTreeWidgetItem(parent);
    item->setText(0, key);
    if (value.isObject()) {
        const QJsonObject obj = value.toObject();
        item->setText(1, tr("{%1 fields}").arg(obj.size()));
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            addJsonValue(item, it.key(), it.value());
        }
    } else if (value.isArray()) {
        const QJsonArray arr = value.toArray();
        item->setText(1, tr("[%1 items]").arg(arr.size()));
        for (int i = 0; i < arr.size(); ++i) {
            addJsonValue(item, QString::number(i), arr.at(i));
        }
    } else {
        item->setText(1, jsonValueToString(value));
    }
}

void MetadataViewer::populateTree(const QJsonObject &root) {
    tree->clear();

    auto addRoot = [&](const QString &label) -> QTreeWidgetItem * {
        auto *item = new QTreeWidgetItem(tree);
        item->setText(0, label);
        item->setFirstColumnSpanned(false);
        return item;
    };

    if (root.contains("format")) {
        auto *fmt = addRoot(tr("format"));
        const QJsonObject obj = root.value("format").toObject();
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            addJsonValue(fmt, it.key(), it.value());
        }
        fmt->setExpanded(true);
    }
    if (root.contains("streams")) {
        const QJsonArray arr = root.value("streams").toArray();
        for (int i = 0; i < arr.size(); ++i) {
            const QJsonObject sObj = arr.at(i).toObject();
            const QString codec = sObj.value("codec_type").toString();
            auto *node = addRoot(tr("stream[%1] (%2)").arg(i).arg(codec));
            for (auto it = sObj.begin(); it != sObj.end(); ++it) {
                addJsonValue(node, it.key(), it.value());
            }
            node->setExpanded(i == 0);
        }
    }
    if (root.contains("frames")) {
        const QJsonArray arr = root.value("frames").toArray();
        if (!arr.isEmpty()) {
            auto *node = addRoot(tr("frame side data"));
            for (int i = 0; i < arr.size(); ++i) {
                addJsonValue(node, QString::number(i), arr.at(i));
            }
            node->setExpanded(true);
        }
    }
}

QString MetadataViewer::buildMarkdown(const QJsonObject &root) const {
    QString md;
    QTextStream out(&md);

    const QJsonObject fmt = root.value("format").toObject();
    const QString filename = fmt.value("filename").toString();
    const QString fmtName = fmt.value("format_long_name").toString();
    const double duration = fmt.value("duration").toString().toDouble();
    const qint64 size = fmt.value("size").toString().toLongLong();
    const qint64 brate = fmt.value("bit_rate").toString().toLongLong();

    out << "## Media Metadata\n\n";
    if (!filename.isEmpty())
        out << "**File**: `" << QFileInfo(filename).fileName() << "`  \n";
    if (!fmtName.isEmpty())
        out << "**Container**: " << fmtName << "  \n";
    if (duration > 0.0)
        out << "**Duration**: " << QString::number(duration, 'f', 3) << " s  \n";
    if (size > 0)
        out << "**Size**: " << humanSize(size) << "  \n";
    if (brate > 0)
        out << "**Overall bitrate**: " << humanBitrate(brate) << "  \n";
    out << "\n";

    const QList<StreamRow> rows = collectStreams(root.value("streams").toArray());
    if (!rows.isEmpty()) {
        out << "### Streams\n\n";
        out << "| # | Type | Codec | Profile | Pix/Sample fmt | Resolution / Rate | Color | Bitrate |\n";
        out << "|---|------|-------|---------|----------------|-------------------|-------|---------|\n";
        for (const StreamRow &r : rows) {
            out << "| " << r.index << " | " << r.type << " | " << r.codec << " | " << r.profile
                << " | " << r.fmtPix << " | " << r.sizeRate
                << " | " << r.color << " | " << r.bitrate << " |\n";
        }
        out << "\n";
    }

    const HdrInfo hdr = collectHdr(root.value("frames").toArray());
    if (hdr.hasMastering || hdr.hasCll) {
        out << "### HDR Static Metadata\n\n";
        if (hdr.hasMastering) {
            out << QString("- **Mastering display**: R(%1, %2) G(%3, %4) B(%5, %6) "
                           "WP(%7, %8) L(%9, %10)\n")
                       .arg(hdr.redX, hdr.redY, hdr.greenX, hdr.greenY,
                            hdr.blueX, hdr.blueY, hdr.whiteX, hdr.whiteY)
                       .arg(hdr.maxLum, hdr.minLum);
        }
        if (hdr.hasCll) {
            out << "- **MaxCLL**: " << hdr.maxCll << " cd/m^2\n";
            out << "- **MaxFALL**: " << hdr.maxFall << " cd/m^2\n";
        }
        out << "\n";
    }

    return md;
}

QString MetadataViewer::buildHtml(const QJsonObject &root) const {
    QString html;
    QTextStream out(&html);

    const QJsonObject fmt = root.value("format").toObject();
    const QString filename = fmt.value("filename").toString();
    const QString fmtName = fmt.value("format_long_name").toString();
    const double duration = fmt.value("duration").toString().toDouble();
    const qint64 size = fmt.value("size").toString().toLongLong();
    const qint64 brate = fmt.value("bit_rate").toString().toLongLong();

    auto esc = [](const QString &s) { return s.toHtmlEscaped(); };

    out << "<h2>Media Metadata</h2>\n";
    out << "<ul>\n";
    if (!filename.isEmpty())
        out << "  <li><b>File:</b> <code>" << esc(QFileInfo(filename).fileName()) << "</code></li>\n";
    if (!fmtName.isEmpty())
        out << "  <li><b>Container:</b> " << esc(fmtName) << "</li>\n";
    if (duration > 0.0)
        out << "  <li><b>Duration:</b> " << QString::number(duration, 'f', 3) << " s</li>\n";
    if (size > 0)
        out << "  <li><b>Size:</b> " << esc(humanSize(size)) << "</li>\n";
    if (brate > 0)
        out << "  <li><b>Overall bitrate:</b> " << esc(humanBitrate(brate)) << "</li>\n";
    out << "</ul>\n";

    const QList<StreamRow> rows = collectStreams(root.value("streams").toArray());
    if (!rows.isEmpty()) {
        out << "<h3>Streams</h3>\n";
        out << "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">\n";
        out << "  <thead><tr>"
               "<th>#</th><th>Type</th><th>Codec</th><th>Profile</th>"
               "<th>Pix/Sample fmt</th><th>Resolution / Rate</th>"
               "<th>Color</th><th>Bitrate</th></tr></thead>\n";
        out << "  <tbody>\n";
        for (const StreamRow &r : rows) {
            out << "    <tr>"
                << "<td>" << r.index << "</td>"
                << "<td>" << esc(r.type) << "</td>"
                << "<td>" << esc(r.codec) << "</td>"
                << "<td>" << esc(r.profile) << "</td>"
                << "<td>" << esc(r.fmtPix) << "</td>"
                << "<td>" << esc(r.sizeRate) << "</td>"
                << "<td>" << esc(r.color) << "</td>"
                << "<td>" << esc(r.bitrate) << "</td>"
                << "</tr>\n";
        }
        out << "  </tbody>\n</table>\n";
    }

    const HdrInfo hdr = collectHdr(root.value("frames").toArray());
    if (hdr.hasMastering || hdr.hasCll) {
        out << "<h3>HDR Static Metadata</h3>\n<ul>\n";
        if (hdr.hasMastering) {
            out << "  <li><b>Mastering display:</b> R("
                << esc(hdr.redX) << ", " << esc(hdr.redY) << ") G("
                << esc(hdr.greenX) << ", " << esc(hdr.greenY) << ") B("
                << esc(hdr.blueX) << ", " << esc(hdr.blueY) << ") WP("
                << esc(hdr.whiteX) << ", " << esc(hdr.whiteY) << ") L("
                << esc(hdr.maxLum) << ", " << esc(hdr.minLum) << ")</li>\n";
        }
        if (hdr.hasCll) {
            out << "  <li><b>MaxCLL:</b> " << esc(hdr.maxCll) << " cd/m&sup2;</li>\n";
            out << "  <li><b>MaxFALL:</b> " << esc(hdr.maxFall) << " cd/m&sup2;</li>\n";
        }
        out << "</ul>\n";
    }

    return html;
}

QString MetadataViewer::buildPlainText(const QJsonObject &root) const {
    QString txt;
    QTextStream out(&txt);

    const QJsonObject fmt = root.value("format").toObject();
    const QString filename = fmt.value("filename").toString();
    const QString fmtName = fmt.value("format_long_name").toString();
    const double duration = fmt.value("duration").toString().toDouble();
    const qint64 size = fmt.value("size").toString().toLongLong();
    const qint64 brate = fmt.value("bit_rate").toString().toLongLong();

    out << "Media Metadata\n";
    out << "==============\n\n";
    if (!filename.isEmpty())
        out << "File           : " << QFileInfo(filename).fileName() << "\n";
    if (!fmtName.isEmpty())
        out << "Container      : " << fmtName << "\n";
    if (duration > 0.0)
        out << "Duration       : " << QString::number(duration, 'f', 3) << " s\n";
    if (size > 0)
        out << "Size           : " << humanSize(size) << "\n";
    if (brate > 0)
        out << "Overall bitrate: " << humanBitrate(brate) << "\n";
    out << "\n";

    const QList<StreamRow> rows = collectStreams(root.value("streams").toArray());
    if (!rows.isEmpty()) {
        out << "Streams\n";
        out << "-------\n";
        for (const StreamRow &r : rows) {
            out << "[" << r.index << "] " << r.type;
            if (!r.codec.isEmpty())
                out << " | codec=" << r.codec;
            if (!r.profile.isEmpty())
                out << " | profile=" << r.profile;
            if (!r.fmtPix.isEmpty())
                out << " | fmt=" << r.fmtPix;
            if (!r.sizeRate.isEmpty())
                out << " | " << r.sizeRate;
            if (!r.color.isEmpty())
                out << " | color=" << r.color;
            if (!r.bitrate.isEmpty())
                out << " | bitrate=" << r.bitrate;
            out << "\n";
        }
        out << "\n";
    }

    const HdrInfo hdr = collectHdr(root.value("frames").toArray());
    if (hdr.hasMastering || hdr.hasCll) {
        out << "HDR Static Metadata\n";
        out << "-------------------\n";
        if (hdr.hasMastering) {
            out << "Mastering display: "
                << "R(" << hdr.redX << ", " << hdr.redY << ") "
                << "G(" << hdr.greenX << ", " << hdr.greenY << ") "
                << "B(" << hdr.blueX << ", " << hdr.blueY << ") "
                << "WP(" << hdr.whiteX << ", " << hdr.whiteY << ") "
                << "L(" << hdr.maxLum << ", " << hdr.minLum << ")\n";
        }
        if (hdr.hasCll) {
            out << "MaxCLL : " << hdr.maxCll << " cd/m^2\n";
            out << "MaxFALL: " << hdr.maxFall << " cd/m^2\n";
        }
        out << "\n";
    }

    return txt;
}

void MetadataViewer::copyCurrentTab() {
    const int idx = tabs->currentIndex();
    QPlainTextEdit *src = nullptr;
    QString format;
    if (idx == markdownTabIndex) {
        src = markdownPreview;
        format = tr("Markdown");
    } else if (idx == htmlTabIndex) {
        src = htmlPreview;
        format = tr("HTML");
    } else if (idx == textTabIndex) {
        src = textPreview;
        format = tr("text");
    } else {
        QMessageBox::information(this, tr("Metadata Viewer"),
                                 tr("Switch to the Markdown, HTML, or Text tab to copy."));
        return;
    }

    const QString text = src ? src->toPlainText() : QString();
    if (text.isEmpty())
        return;

    QClipboard *clip = QGuiApplication::clipboard();
    if (idx == htmlTabIndex) {
        // For HTML, also publish a rich-HTML representation so HTML-aware
        // targets (mail composers, rich-text editors) get formatted output
        // while plain-text targets get the raw HTML source.
        auto *mime = new QMimeData();
        mime->setText(text);
        mime->setHtml(text);
        clip->setMimeData(mime, QClipboard::Clipboard);
#ifdef __linux__
        if (clip->supportsSelection()) {
            auto *selMime = new QMimeData();
            selMime->setText(text);
            selMime->setHtml(text);
            clip->setMimeData(selMime, QClipboard::Selection);
        }
#endif
    } else {
        clip->setText(text, QClipboard::Clipboard);
#ifdef __linux__
        if (clip->supportsSelection())
            clip->setText(text, QClipboard::Selection);
#endif
    }

    QMessageBox::information(this, tr("Metadata Viewer"),
                             tr("%1 copied to clipboard.").arg(format));
}
