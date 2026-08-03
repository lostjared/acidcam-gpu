#include "settings.hpp"
#include "custom_style.hpp"
#include <QApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHeaderView>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QProcess>
#include <QRegularExpression>
#include <QScreen>
#include <QScrollArea>
#include <QSet>
#include <QSettings>
#include <QTimer>
#include <QTreeWidget>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#ifdef _WIN32
#include <dshow.h>
#include <dvdmedia.h> // VIDEOINFOHEADER2, FORMAT_VideoInfo2 (not pulled in by dshow.h in MinGW)
#include <olectl.h>
#include <optional>
#include <windows.h>
// MEDIASUBTYPE_I420 is absent from MinGW DirectShow headers — define it manually.
#ifndef MEDIASUBTYPE_I420
static const GUID MEDIASUBTYPE_I420 = {0x30323449, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
#endif
#endif
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
    bool parse_even_resolution(const QString &text, QSize &resolution) {
        const QString trimmed = text.trimmed();
        if (trimmed.compare("Default", Qt::CaseInsensitive) == 0) {
            resolution = QSize(0, 0);
            return true;
        }

        static const QRegularExpression resolution_pattern(
            R"(^(\d+)\s*[xX]\s*(\d+)$)");
        const QRegularExpressionMatch match =
            resolution_pattern.match(trimmed);
        if (!match.hasMatch()) {
            return false;
        }

        bool width_ok = false;
        bool height_ok = false;
        const int width = match.captured(1).toInt(&width_ok);
        const int height = match.captured(2).toInt(&height_ok);
        if (!width_ok || !height_ok || width <= 0 || height <= 0 ||
            width % 2 != 0 || height % 2 != 0) {
            return false;
        }

        resolution = QSize(width, height);
        return true;
    }

    QString resolution_text(const QSize &resolution) {
        if (resolution.isEmpty()) {
            return "Default";
        }
        return QString("%1x%2")
            .arg(resolution.width())
            .arg(resolution.height());
    }

    bool pathsReferToSameFile(const QString &firstPath, const QString &secondPath) {
        const QFileInfo firstInfo(firstPath);
        const QFileInfo secondInfo(secondPath);

        QString normalizedFirst = firstInfo.canonicalFilePath();
        if (normalizedFirst.isEmpty()) {
            normalizedFirst = QDir::cleanPath(firstInfo.absoluteFilePath());
        }

        QString normalizedSecond = secondInfo.canonicalFilePath();
        if (normalizedSecond.isEmpty()) {
            normalizedSecond = QDir::cleanPath(secondInfo.absoluteFilePath());
        }

#ifdef _WIN32
        return normalizedFirst.compare(normalizedSecond, Qt::CaseInsensitive) == 0;
#else
        return normalizedFirst == normalizedSecond;
#endif
    }

    void appendUniqueFps(QList<double> &fpsList, double fps) {
        if (fps <= 0.0) {
            return;
        }

        for (double existing : fpsList) {
            if (qAbs(existing - fps) < 0.05) {
                return;
            }
        }
        fpsList.append(fps);
    }

#ifdef __APPLE__
    QStringList appleCameraNamesFromSystemProfiler() {
        QProcess process;
        process.start("/usr/sbin/system_profiler", {"-json", "SPCameraDataType"});
        if (!process.waitForFinished(8000)) {
            return {};
        }

        const QByteArray output = process.readAllStandardOutput();
        const QJsonDocument document = QJsonDocument::fromJson(output);
        if (!document.isObject()) {
            return {};
        }

        QStringList cameraNames;
        const QJsonArray cameras = document.object().value("SPCameraDataType").toArray();
        for (const QJsonValue &cameraValue : cameras) {
            const QJsonObject cameraObject = cameraValue.toObject();
            const QString cameraName = cameraObject.value("_name").toString().trimmed();
            if (!cameraName.isEmpty()) {
                cameraNames.append(cameraName);
            }
        }
        return cameraNames;
    }

    void populateAppleDefaultCapabilities(QMap<QString, QList<double>> &deviceCapabilities) {
        static const QStringList kDefaultResolutions = {
            "640x360",
            "640x480",
            "1280x720",
            "1920x1080",
            "3840x2160"};
        static const QList<double> kDefaultFps = {24.0, 30.0, 60.0};

        for (const QString &resolution : kDefaultResolutions) {
            deviceCapabilities.insert(resolution, kDefaultFps);
        }
    }
#endif

#ifdef _WIN32
    std::optional<int> parseIndexedCameraLabel(const QString &label) {
        const int openPos = label.lastIndexOf("[");
        const int closePos = label.lastIndexOf("]");
        if (openPos < 0 || closePos < 0 || closePos <= openPos + 1) {
            return std::nullopt;
        }

        bool ok = false;
        const int parsed = label.mid(openPos + 1, closePos - openPos - 1).trimmed().toInt(&ok);
        if (!ok) {
            return std::nullopt;
        }
        return parsed;
    }

    int resolveWindowsSelectedCameraIndex(const QComboBox *comboBox) {
        if (comboBox == nullptr || comboBox->currentIndex() < 0) {
            return -1;
        }

        const QString currentLabel = comboBox->currentText();
        if (const auto parsedIndex = parseIndexedCameraLabel(currentLabel); parsedIndex.has_value()) {
            return *parsedIndex;
        }

        return comboBox->currentData().toInt();
    }

    template <typename T>
    struct ComReleaser {
        void operator()(T *value) const {
            if (value != nullptr) {
                value->Release();
            }
        }
    };

    template <typename T>
    using ComPtr = std::unique_ptr<T, ComReleaser<T>>;

    class ComInitScope {
      public:
        ComInitScope()
            : result(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)), shouldUninitialize(SUCCEEDED(result)) {
        }

        ~ComInitScope() {
            if (shouldUninitialize) {
                CoUninitialize();
            }
        }

        bool ready() const {
            return SUCCEEDED(result) || result == RPC_E_CHANGED_MODE;
        }

      private:
        HRESULT result;
        bool shouldUninitialize;
    };

    struct WindowsCameraDeviceInfo {
        int index = -1;
        QString name;
    };

    void freeMediaType(AM_MEDIA_TYPE *mediaType) {
        if (mediaType == nullptr) {
            return;
        }

        if (mediaType->cbFormat != 0 && mediaType->pbFormat != nullptr) {
            CoTaskMemFree(mediaType->pbFormat);
            mediaType->cbFormat = 0;
            mediaType->pbFormat = nullptr;
        }

        if (mediaType->pUnk != nullptr) {
            mediaType->pUnk->Release();
            mediaType->pUnk = nullptr;
        }

        CoTaskMemFree(mediaType);
    }

    QString cameraNameFromPropertyBag(IPropertyBag *propertyBag) {
        if (propertyBag == nullptr) {
            return {};
        }

        VARIANT value;
        VariantInit(&value);

        QString deviceName;
        if (SUCCEEDED(propertyBag->Read(L"FriendlyName", &value, nullptr)) && value.vt == VT_BSTR && value.bstrVal != nullptr) {
            deviceName = QString::fromWCharArray(value.bstrVal).trimmed();
        }
        VariantClear(&value);

        if (!deviceName.isEmpty()) {
            return deviceName;
        }

        VariantInit(&value);
        if (SUCCEEDED(propertyBag->Read(L"Description", &value, nullptr)) && value.vt == VT_BSTR && value.bstrVal != nullptr) {
            deviceName = QString::fromWCharArray(value.bstrVal).trimmed();
        }
        VariantClear(&value);

        return deviceName;
    }

    std::vector<WindowsCameraDeviceInfo> enumerateWindowsCameraDevices() {
        ComInitScope comScope;
        if (!comScope.ready()) {
            return {};
        }

        ICreateDevEnum *deviceEnumeratorRaw = nullptr;
        if (FAILED(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                    IID_ICreateDevEnum, reinterpret_cast<void **>(&deviceEnumeratorRaw)))) {
            return {};
        }
        ComPtr<ICreateDevEnum> deviceEnumerator(deviceEnumeratorRaw);

        IEnumMoniker *enumMonikerRaw = nullptr;
        const HRESULT createResult = deviceEnumerator->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
                                                                             &enumMonikerRaw, 0);
        if (createResult != S_OK || enumMonikerRaw == nullptr) {
            return {};
        }
        ComPtr<IEnumMoniker> enumMoniker(enumMonikerRaw);

        std::vector<WindowsCameraDeviceInfo> devices;
        IMoniker *monikerRaw = nullptr;
        ULONG fetched = 0;
        int currentIndex = 0;
        while (enumMoniker->Next(1, &monikerRaw, &fetched) == S_OK) {
            ComPtr<IMoniker> moniker(monikerRaw);
            monikerRaw = nullptr;

            IPropertyBag *propertyBagRaw = nullptr;
            if (SUCCEEDED(moniker->BindToStorage(nullptr, nullptr, IID_IPropertyBag,
                                                 reinterpret_cast<void **>(&propertyBagRaw)))) {
                ComPtr<IPropertyBag> propertyBag(propertyBagRaw);
                const QString deviceName = cameraNameFromPropertyBag(propertyBag.get());
                if (!deviceName.isEmpty()) {
                    devices.push_back({currentIndex, deviceName});
                } else {
                    devices.push_back({currentIndex, QString("Camera %1").arg(currentIndex)});
                }
            } else {
                devices.push_back({currentIndex, QString("Camera %1").arg(currentIndex)});
            }
            ++currentIndex;
        }

        return devices;
    }

    ComPtr<IAMStreamConfig> openWindowsStreamConfig(int deviceIndex) {
        ICreateDevEnum *deviceEnumeratorRaw = nullptr;
        if (FAILED(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                    IID_ICreateDevEnum, reinterpret_cast<void **>(&deviceEnumeratorRaw)))) {
            return {};
        }
        ComPtr<ICreateDevEnum> deviceEnumerator(deviceEnumeratorRaw);

        IEnumMoniker *enumMonikerRaw = nullptr;
        if (deviceEnumerator->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMonikerRaw, 0) != S_OK ||
            enumMonikerRaw == nullptr) {
            return {};
        }
        ComPtr<IEnumMoniker> enumMoniker(enumMonikerRaw);

        IMoniker *monikerRaw = nullptr;
        ULONG fetched = 0;
        int currentIndex = 0;
        while (enumMoniker->Next(1, &monikerRaw, &fetched) == S_OK) {
            ComPtr<IMoniker> moniker(monikerRaw);
            monikerRaw = nullptr;

            if (currentIndex != deviceIndex) {
                ++currentIndex;
                continue;
            }

            IBaseFilter *filterRaw = nullptr;
            if (FAILED(moniker->BindToObject(nullptr, nullptr, IID_IBaseFilter,
                                             reinterpret_cast<void **>(&filterRaw)))) {
                return {};
            }
            ComPtr<IBaseFilter> filter(filterRaw);

            IEnumPins *enumPinsRaw = nullptr;
            if (FAILED(filter->EnumPins(&enumPinsRaw)) || enumPinsRaw == nullptr) {
                return {};
            }
            ComPtr<IEnumPins> enumPins(enumPinsRaw);

            IPin *pinRaw = nullptr;
            ULONG pinFetched = 0;
            while (enumPins->Next(1, &pinRaw, &pinFetched) == S_OK) {
                ComPtr<IPin> pin(pinRaw);
                pinRaw = nullptr;

                PIN_DIRECTION direction = PINDIR_INPUT;
                if (FAILED(pin->QueryDirection(&direction)) || direction != PINDIR_OUTPUT) {
                    continue;
                }

                IAMStreamConfig *streamConfigRaw = nullptr;
                if (SUCCEEDED(pin->QueryInterface(IID_IAMStreamConfig, reinterpret_cast<void **>(&streamConfigRaw))) &&
                    streamConfigRaw != nullptr) {
                    return ComPtr<IAMStreamConfig>(streamConfigRaw);
                }
            }

            return {};
        }

        return {};
    }

    QString mediaSubtypeName(const GUID &subtype) {
        if (subtype == MEDIASUBTYPE_YUY2)
            return "YUY2";
        if (subtype == MEDIASUBTYPE_UYVY)
            return "UYVY";
        if (subtype == MEDIASUBTYPE_YV12)
            return "YV12";
        if (subtype == MEDIASUBTYPE_NV12)
            return "NV12";
        if (subtype == MEDIASUBTYPE_I420)
            return "I420";
        if (subtype == MEDIASUBTYPE_MJPG)
            return "MJPG";
        if (subtype == MEDIASUBTYPE_RGB24)
            return "RGB24";
        if (subtype == MEDIASUBTYPE_RGB32)
            return "RGB32";
        return {};
    }

    bool isYuvSubtype(const GUID &subtype) {
        return subtype == MEDIASUBTYPE_YUY2 || subtype == MEDIASUBTYPE_UYVY || subtype == MEDIASUBTYPE_YV12 ||
               subtype == MEDIASUBTYPE_NV12 || subtype == MEDIASUBTYPE_I420;
    }

    bool extractDirectShowFormat(const AM_MEDIA_TYPE *mediaType, QSize &resolution, QString &formatName, double &fps) {
        if (mediaType == nullptr) {
            return false;
        }

        formatName = mediaSubtypeName(mediaType->subtype);
        fps = 0.0;

        if (mediaType->formattype == FORMAT_VideoInfo && mediaType->cbFormat >= sizeof(VIDEOINFOHEADER)) {
            const auto *videoInfo = reinterpret_cast<const VIDEOINFOHEADER *>(mediaType->pbFormat);
            resolution = QSize(videoInfo->bmiHeader.biWidth, std::abs(videoInfo->bmiHeader.biHeight));
            if (videoInfo->AvgTimePerFrame > 0) {
                fps = 10000000.0 / static_cast<double>(videoInfo->AvgTimePerFrame);
            }
            return resolution.width() > 0 && resolution.height() > 0;
        }

        if (mediaType->formattype == FORMAT_VideoInfo2 && mediaType->cbFormat >= sizeof(VIDEOINFOHEADER2)) {
            const auto *videoInfo = reinterpret_cast<const VIDEOINFOHEADER2 *>(mediaType->pbFormat);
            resolution = QSize(videoInfo->bmiHeader.biWidth, std::abs(videoInfo->bmiHeader.biHeight));
            if (videoInfo->AvgTimePerFrame > 0) {
                fps = 10000000.0 / static_cast<double>(videoInfo->AvgTimePerFrame);
            }
            return resolution.width() > 0 && resolution.height() > 0;
        }

        return false;
    }
#endif
} // namespace

SettingsWindow::SettingsWindow(const QString &execPath, QWidget *parent)
    : QDialog(parent),
      selectedCameraIndex(0),
      selectedCameraResolution(1280, 720),
      selectedScreenResolution(0, 0),
      cameraFPS(30),
      inputVideoFile(""),
      outputVideoFile(""),
      useInputVideoFile(false),
      useGraphicsFile(false),
      saveOutputVideoFile(false),
      graphicsFile(""),
      graphicsDuration(10),
      modelFile("data/cube.mxmod.z"),
      selectedCudaDevice(0),
      maxDuration(0.0),
      maxSizeLimit(0.0),
      executablePath(execPath) {
    init();
}

void SettingsWindow::populateCameraDevices() {
    QSet<QString> addedCameras;

#ifdef __APPLE__
    const QStringList cameraNames = appleCameraNamesFromSystemProfiler();
    for (int i = 0; i < cameraNames.size(); ++i) {
        const QString cameraName = cameraNames.at(i).trimmed();
        if (!cameraName.isEmpty() && !addedCameras.contains(cameraName)) {
            cameraIndexComboBox->addItem(QString("%1 [%2]").arg(cameraName).arg(i), i);
            addedCameras.insert(cameraName);
        }
    }
#elif defined(_WIN32)
    const auto devices = enumerateWindowsCameraDevices();
    for (const auto &device : devices) {
        const QString cameraName = device.name.trimmed();
        if (!cameraName.isEmpty() && !addedCameras.contains(cameraName)) {
            cameraIndexComboBox->addItem(QString("%1 [%2]").arg(cameraName).arg(device.index), device.index);
            addedCameras.insert(cameraName);
        }
    }
#else
    for (int i = 0; i < 20; ++i) {
        QString sysfs_path = QString("/sys/class/video4linux/video%1/name").arg(i);
        QFile file(sysfs_path);
        if (file.exists()) {
            QString cameraName = getCameraName(i);
            if (!addedCameras.contains(cameraName)) {
                cameraIndexComboBox->addItem(QString("%1 [%2]").arg(cameraName).arg(i), i);
                addedCameras.insert(cameraName);
            }
        }
    }
#endif

    if (cameraIndexComboBox->count() == 0) {
        cameraIndexComboBox->addItem("No cameras found", -1);
    }
}

void SettingsWindow::populateCudaDevices() {
    QProcess process;
    process.start("acmx2", QStringList() << "--list-cuda-devices");
    process.waitForFinished(5000);

    QString output = process.readAllStandardOutput();

    if (output.isEmpty() || process.exitCode() != 0) {
        cudaDeviceComboBox->addItem("No CUDA devices found", 0);
        return;
    }

    QStringList lines = output.split('\n');
    QRegularExpression deviceRegex("Device\\s+(\\d+):\\s*\"?([^\"\\n]+)\"?");

    bool foundDevice = false;
    for (const QString &line : lines) {
        QRegularExpressionMatch match = deviceRegex.match(line);
        if (match.hasMatch()) {
            int deviceIndex = match.captured(1).toInt();
            QString deviceName = match.captured(2).trimmed();
            int parenPos = deviceName.indexOf('(');
            if (parenPos > 0) {
                deviceName = deviceName.left(parenPos).trimmed();
            }
            cudaDeviceComboBox->addItem(QString("%1 [%2]").arg(deviceName).arg(deviceIndex), deviceIndex);
            foundDevice = true;
        }
    }

    if (!foundDevice) {
        cudaDeviceComboBox->addItem("No CUDA devices found", 0);
    }
}

void SettingsWindow::populateVideoEncoders(const QString &savedEncoder) {
    encodeCodecComboBox->clear();
    encodeCodecComboBox->addItem("Automatic (NVENC, then software)", "auto");
    encodeCodecComboBox->addItem("Automatic software H.264/H.265", "software");
    encodeCodecComboBox->addItem("Automatic NVIDIA NVENC", "nvenc");

    QProcess process;
    process.start(executablePath, QStringList() << "--list-encoders");
    const bool finished = process.waitForFinished(5000);
    const QString output = QString::fromUtf8(process.readAllStandardOutput());
    if (finished && process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
        const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
        for (const QString &line : lines) {
            const QStringList fields = line.split('\t', Qt::KeepEmptyParts);
            if (fields.size() < 7 || fields.at(0) != "ENCODER") {
                continue;
            }
            const QString name = fields.at(1);
            const QString longName = fields.at(2);
            const QString codecName = fields.at(3);
            const bool hardware = fields.at(4) == "hardware";
            const QString stability = fields.at(5);
            const QString pixelFormats = fields.at(6);
            const QString backend = hardware ? "hardware" : "software";
            const QString label = QString("%1 — %2 [%3]").arg(name, longName, backend);
            encodeCodecComboBox->addItem(label, name);
            const int index = encodeCodecComboBox->count() - 1;
            encodeCodecComboBox->setItemData(index, longName, Qt::UserRole + 1);
            encodeCodecComboBox->setItemData(index, codecName, Qt::UserRole + 2);
            encodeCodecComboBox->setItemData(index, backend, Qt::UserRole + 3);
            encodeCodecComboBox->setItemData(index, stability, Qt::UserRole + 4);
            encodeCodecComboBox->setItemData(index, pixelFormats, Qt::UserRole + 5);
            encodeCodecComboBox->setItemData(
                index,
                QString("%1\nCodec: %2; backend: %3; status: %4\nPixel formats: %5")
                    .arg(longName, codecName, backend, stability,
                         pixelFormats.isEmpty() ? QString("encoder-defined") : pixelFormats),
                Qt::ToolTipRole);
        }
    }

    // Keep useful exact choices visible even when the configured executable
    // is temporarily unavailable. They will fail with a clear MXWrite error
    // if the linked FFmpeg build does not provide them.
    const QList<QPair<QString, QString>> fallbackEncoders = {
        {"libx264 — software H.264", "libx264"},
        {"libx265 — software H.265/HEVC", "libx265"},
        {"h264_nvenc — NVIDIA H.264", "h264_nvenc"},
        {"hevc_nvenc — NVIDIA H.265/HEVC", "hevc_nvenc"}};
    for (const auto &encoder : fallbackEncoders) {
        if (encodeCodecComboBox->findData(encoder.second) < 0) {
            encodeCodecComboBox->addItem(encoder.first, encoder.second);
        }
    }

    int savedIndex = encodeCodecComboBox->findData(savedEncoder);
    if (savedIndex < 0 && !savedEncoder.isEmpty()) {
        encodeCodecComboBox->addItem(savedEncoder + " — unavailable in current FFmpeg build",
                                     savedEncoder);
        savedIndex = encodeCodecComboBox->count() - 1;
    }
    encodeCodecComboBox->setCurrentIndex(savedIndex >= 0 ? savedIndex : 0);
}

void SettingsWindow::updateEncoderDetails() {
    if (!encodeCodecComboBox || !encodeCodecDetailsLabel || !encodeOptionsButton) {
        return;
    }
    const QString name = encodeCodecComboBox->currentData().toString();
    const QString description = encodeCodecComboBox->currentData(Qt::UserRole + 1).toString();
    const QString codecName = encodeCodecComboBox->currentData(Qt::UserRole + 2).toString();
    const QString backend = encodeCodecComboBox->currentData(Qt::UserRole + 3).toString();
    const QString pixelFormats = encodeCodecComboBox->currentData(Qt::UserRole + 5).toString();
    const bool exactEncoder = name != "auto" && name != "software" && name != "nvenc";
    encodeOptionsButton->setEnabled(exactEncoder);
    if (!description.isEmpty()) {
        encodeCodecDetailsLabel->setText(
            QString("%1 | codec: %2 | %3 | pixel formats: %4")
                .arg(description, codecName, backend,
                     pixelFormats.isEmpty() ? QString("encoder-defined") : pixelFormats));
    } else if (exactEncoder) {
        encodeCodecDetailsLabel->setText("Exact FFmpeg encoder: " + name);
    } else {
        encodeCodecDetailsLabel->setText(
            "Selection policy; MXWrite chooses the concrete H.264/H.265 encoder at startup.");
    }
}

void SettingsWindow::showEncoderOptions() {
    const QString encoderName = encodeCodecComboBox->currentData().toString();
    if (encoderName.isEmpty() || encoderName == "auto" || encoderName == "software" ||
        encoderName == "nvenc") {
        return;
    }

    QProcess process;
    process.start(executablePath,
                  QStringList() << "--list-encoder-options" << encoderName);
    if (!process.waitForFinished(5000) || process.exitStatus() != QProcess::NormalExit ||
        process.exitCode() != 0) {
        const QString error = QString::fromUtf8(process.readAllStandardError()).trimmed();
        QMessageBox::warning(this, "Encoder Options",
                             error.isEmpty() ? "Unable to query this encoder." : error);
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(encoderName + " Encoder Options");
    dialog.resize(1000, 560);
    dialog.setStyleSheet(styleSheet());
    auto *layout = new QVBoxLayout(&dialog);
    auto *summary = new QLabel(
        "Double-click an option to add it to Extra FFmpeg parameters, or use the shown "
        "name manually as -option value.",
        &dialog);
    summary->setWordWrap(true);
    layout->addWidget(summary);

    auto *tree = new QTreeWidget(&dialog);
    tree->setColumnCount(6);
    tree->setHeaderLabels({"Option", "Type", "Default", "Range", "Named values", "Description"});
    tree->setRootIsDecorated(false);
    tree->setAlternatingRowColors(false);
    tree->setUniformRowHeights(false);
    tree->setWordWrap(true);
    tree->setTextElideMode(Qt::ElideNone);
    tree->setSelectionBehavior(QAbstractItemView::SelectRows);
    tree->setStyleSheet(
        "QTreeWidget::item { background-color: transparent; padding: 4px; }"
        "QTreeWidget::item:selected { background-color: palette(highlight); }");
    const QString output = QString::fromUtf8(process.readAllStandardOutput());
    for (const QString &line : output.split('\n', Qt::SkipEmptyParts)) {
        const QStringList fields = line.split('\t', Qt::KeepEmptyParts);
        if (fields.size() < 8 || fields.at(0) != "OPTION") {
            continue;
        }
        QString range;
        if (!fields.at(4).isEmpty() || !fields.at(5).isEmpty()) {
            range = fields.at(4) + " … " + fields.at(5);
        }
        new QTreeWidgetItem(tree, {"-" + fields.at(1), fields.at(2), fields.at(3),
                                   range, fields.at(6), fields.at(7)});
    }
    if (tree->topLevelItemCount() == 0) {
        new QTreeWidgetItem(tree, {"(No private video AVOptions reported)"});
    }
    tree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    tree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    tree->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    tree->header()->setSectionResizeMode(3, QHeaderView::Interactive);
    tree->header()->setSectionResizeMode(4, QHeaderView::Interactive);
    tree->header()->setSectionResizeMode(5, QHeaderView::Stretch);
    tree->setColumnWidth(3, 175);
    tree->setColumnWidth(4, 250);

    const auto updateRowHeights = [tree]() {
        const QFontMetrics metrics(tree->font());
        for (int row = 0; row < tree->topLevelItemCount(); ++row) {
            QTreeWidgetItem *item = tree->topLevelItem(row);
            int height = metrics.height() + 10;
            for (int column : {3, 4, 5}) {
                const int textWidth = std::max(40, tree->columnWidth(column) - 12);
                const QRect bounds = metrics.boundingRect(
                    QRect(0, 0, textWidth, std::numeric_limits<int>::max()),
                    Qt::TextWordWrap | Qt::AlignLeft, item->text(column));
                height = std::max(height, bounds.height() + 10);
            }
            item->setSizeHint(0, QSize(tree->columnWidth(0), height));
        }
    };
    connect(tree->header(), &QHeaderView::sectionResized, &dialog,
            [updateRowHeights](int logicalIndex, int, int) {
                if (logicalIndex >= 3) {
                    updateRowHeights();
                }
            });
    QTimer::singleShot(0, &dialog, updateRowHeights);
    connect(tree, &QTreeWidget::itemDoubleClicked, &dialog,
            [this, &dialog](QTreeWidgetItem *item, int) {
                QString optionName = item->text(0);
                if (!optionName.startsWith('-')) {
                    return;
                }
                bool accepted = false;
                const QString value = QInputDialog::getText(
                    &dialog, "Set Encoder Option", optionName + " value:",
                    QLineEdit::Normal, item->text(2), &accepted);
                if (!accepted) {
                    return;
                }
                QString escapedValue = value;
                if (escapedValue.contains(' ') || escapedValue.contains('\t') ||
                    escapedValue.contains('"')) {
                    escapedValue.replace('\\', "\\\\");
                    escapedValue.replace('"', "\\\"");
                    escapedValue = '"' + escapedValue + '"';
                }
                QString parameters = encodeParametersLineEdit->text().trimmed();
                if (!parameters.isEmpty()) {
                    parameters += ' ';
                }
                parameters += optionName;
                if (!escapedValue.isEmpty()) {
                    parameters += ' ' + escapedValue;
                }
                encodeParametersLineEdit->setText(parameters);
            });
    layout->addWidget(tree);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    dialog.exec();
}

void SettingsWindow::init() {
    setStyleSheet(qApp->styleSheet());
    acmx2::applyCustomStyleIfEnabled(this);

    // ── Create all widgets ────────────────────────────────────────────
    cameraOptionRadioButton = new QRadioButton("Use Camera", this);
    inputVideoOptionRadioButton = new QRadioButton("Use Video File as Input", this);
    graphicsFileOptionRadioButton = new QRadioButton("Use Graphics File as Input", this);
    cameraOptionRadioButton->setChecked(true);

    cameraIndexComboBox = new QComboBox(this);
    populateCameraDevices();

    cudaDeviceLabel = new QLabel("CUDA Device:", this);
    cudaDeviceComboBox = new QComboBox(this);
    populateCudaDevices();

    cameraResolutionComboBox = new QComboBox(this);
    cameraResolutionComboBox->addItem("Default");
    cameraFPSComboBox = new QComboBox(this);
    cameraFPSComboBox->addItem("30");

    if (cameraIndexComboBox->count() > 0)
        enumerateDevice(cameraIndexComboBox->currentData().toInt());

    useYuvCheckBox = new QCheckBox("Use YUV (YUYV) camera format", this);
    useYuvCheckBox->setChecked(false);
    useYuvCheckBox->setEnabled(false);

    inputVideoFileLineEdit = new QLineEdit(this);
    inputVideoFileLineEdit->setReadOnly(true);
    browseInputVideoButton = new QPushButton("Browse", this);

    hdrStatusLabel = new QLabel("HDR: not checked", this);
    convertHdr10CheckBox = new QCheckBox("Convert HLG to HDR10 after processing", this);
    convertHdr10CheckBox->setChecked(false);
    convertHdr10CheckBox->setEnabled(false);
    convertHdr10CheckBox->setToolTip(
        "When the input is detected as HDR (HLG or PQ), enable this to run an "
        "ffmpeg pass after acmx2 finishes that re-encodes the output to HDR10 "
        "(HEVC NVENC, bt2020/PQ).");

    graphicsFileLineEdit = new QLineEdit(this);
    graphicsFileLineEdit->setReadOnly(true);
    browseGraphicsButton = new QPushButton("Browse", this);

    screenResolutionComboBox = new QComboBox(this);
    screenResolutionComboBox->setEditable(true);
    screenResolutionComboBox->setInsertPolicy(QComboBox::NoInsert);
    screenResolutionComboBox->lineEdit()->setPlaceholderText("Default or WxH");
    screenResolutionComboBox->addItems({"Default",
                                        "320x240", "240x320", "400x300", "300x400", "512x384", "384x512",
                                        "640x360", "360x640", "640x480", "480x640", "720x480", "480x720",
                                        "800x600", "600x800", "960x720", "720x960", "1024x768", "768x1024",
                                        "1152x864", "864x1152", "1280x720", "720x1280", "1280x960", "960x1280",
                                        "1280x1024", "1024x1280", "1366x768", "768x1366", "1440x900", "900x1440",
                                        "1600x900", "900x1600", "1600x1200", "1200x1600", "1440x1080", "1080x1440",
                                        "1920x1080", "1080x1920", "1920x1200", "1200x1920", "2048x1536", "1536x2048",
                                        "2560x1440", "1440x2560", "2560x1600", "1600x2560", "2560x1920", "1920x2560",
                                        "3440x1440", "1440x3440", "3840x1600", "1600x3840", "3840x2160", "2160x3840",
                                        "7680x4320", "4320x7680"});
    screenResolutionComboBox->setCurrentIndex(0);

    saveOutputVideoCheckBox = new QCheckBox("Save Output to Video File", this);
    outputVideoFileLineEdit = new QLineEdit(this);
    outputVideoFileLineEdit->setReadOnly(true);
    browseOutputVideoButton = new QPushButton("Browse", this);

    copyAudioCheckBox = new QCheckBox("Copy Audio Track", this);
    copyAudioCheckBox->setChecked(false);
    copyAudioCheckBox->setEnabled(false);

    writePngCheckBox = new QCheckBox("Write PNG", this);

    generateCheckBox = new QCheckBox("Generate every", this);
    generateCheckBox->setToolTip("Save a PNG frame every N frames to an output subdirectory (passes --generate <N>).");
    generateIntervalSpinBox = new QSpinBox(this);
    generateIntervalSpinBox->setRange(1, 100000);
    generateIntervalSpinBox->setValue(30);
    generateIntervalSpinBox->setSuffix(" frames");
    generateIntervalSpinBox->setEnabled(false);
    connect(generateCheckBox, &QCheckBox::toggled, generateIntervalSpinBox, &QWidget::setEnabled);

    timeSpeedSpinBox = new QDoubleSpinBox(this);
    timeSpeedSpinBox->setRange(-100.0, 100.0);
    timeSpeedSpinBox->setSingleStep(0.1);
    timeSpeedSpinBox->setDecimals(2);
    timeSpeedSpinBox->setValue(1.0);

    durationLimitCheckBox = new QCheckBox("Max Duration (sec):", this);
    durationLimitSpinBox = new QDoubleSpinBox(this);
    durationLimitSpinBox->setRange(0.1, 86400.0);
    durationLimitSpinBox->setSingleStep(1.0);
    durationLimitSpinBox->setDecimals(1);
    durationLimitSpinBox->setValue(60.0);
    durationLimitSpinBox->setEnabled(false);

    maxSizeLimitCheckBox = new QCheckBox("Max Size: MB", this);
    maxSizeLimitSpinBox = new QDoubleSpinBox(this);
    maxSizeLimitSpinBox->setRange(0.1, 1048576.0);
    maxSizeLimitSpinBox->setSingleStep(10.0);
    maxSizeLimitSpinBox->setDecimals(2);
    maxSizeLimitSpinBox->setValue(500.0);
    maxSizeLimitSpinBox->setEnabled(false);

    crossFadeSpinBox = new QDoubleSpinBox(this);
    crossFadeSpinBox->setRange(0.0, 10.0);
    crossFadeSpinBox->setSingleStep(0.1);
    crossFadeSpinBox->setDecimals(2);
    crossFadeSpinBox->setValue(0.5);

    flipCheckBox = new QCheckBox("Flip", this);

    fullscreenCheckBox = new QCheckBox("Fullscreen", this);

    enable3dCheckBox = new QCheckBox("Enable 3D", this);
    enable3dCheckBox->setChecked(false);

    modelFileLineEdit = new QLineEdit(this);
    modelFileLineEdit->setText("data/cube.mxmod.z");
    modelFileLineEdit->setReadOnly(true);
    modelFileLineEdit->setEnabled(false);
    browseModelButton = new QPushButton("Model", this);
    browseModelButton->setEnabled(false);

    useOnnxModelCheckBox = new QCheckBox("Use ONNX Model", this);
    useOnnxModelCheckBox->setChecked(false);
    onnxModelFileLineEdit = new QLineEdit(this);
    onnxModelFileLineEdit->setReadOnly(true);
    onnxModelFileLineEdit->setEnabled(false);
    onnxModelFileLineEdit->setPlaceholderText("Select YAML config file...");
    browseOnnxModelButton = new QPushButton("YAML", this);
    browseOnnxModelButton->setEnabled(false);

    textureCacheCheckBox = new QCheckBox("Texture Cache", this);
    textureCacheCheckBox->setEnabled(true);
    auto *textureCacheArrayCheckBox =
        new QCheckBox("Use sampler2DArray history", this);
    textureCacheArrayCheckBox->setObjectName("textureCacheArrayCheckBox");
    textureCacheArrayCheckBox->setToolTip(
        "Pass --texture-cache-array and bind frame history as one array texture");
    textureCacheArrayCheckBox->setEnabled(
        textureCacheCheckBox->isChecked());
    cacheDelaySpinBox = new QSpinBox(this);
    cacheDelaySpinBox->setRange(1, 8);
    cacheDelaySpinBox->setValue(1);
    cacheDelaySpinBox->setEnabled(textureCacheCheckBox->isChecked());
    cacheSizeSpinBox = new QSpinBox(this);
    cacheSizeSpinBox->setRange(1, 64);
    cacheSizeSpinBox->setValue(8);
    cacheSizeSpinBox->setToolTip("Number of frames to keep in the texture ring buffer (1-64, default 8)");
    cacheSizeSpinBox->setEnabled(textureCacheCheckBox->isChecked());
    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);
    // ── Encoding quality widgets ──────────────────────────────────────
    QSettings encSettings("LostSideDead", "acmx2");
    encodePresetComboBox = new QComboBox(this);
    encodePresetComboBox->addItems({"ultrafast", "superfast", "veryfast", "faster", "fast",
                                    "medium", "slow", "slower", "veryslow", "p1", "p2", "p3",
                                    "p4", "p5", "p6", "p7"});
    encodePresetComboBox->setCurrentText(encSettings.value("recording/preset", "medium").toString());

    encodeTuneComboBox = new QComboBox(this);
    encodeTuneComboBox->addItems({"none", "film", "animation", "grain", "stillimage",
                                  "psnr", "ssim", "fastdecode", "zerolatency", "hq", "uhq",
                                  "ll", "ull", "lossless"});
    encodeTuneComboBox->setCurrentText(encSettings.value("recording/tune", "none").toString());

    encodeCrfSpinBox = new QSpinBox(this);
    encodeCrfSpinBox->setRange(0, 51);
    encodeCrfSpinBox->setValue(encSettings.value("recording/crf", 18).toInt());
    encodeCrfSpinBox->setToolTip("Constant Rate Factor: 0 = lossless, 18 = visually lossless, 23 = default, 28 = small file");

    encodeCodecComboBox = new QComboBox(this);
    encodeCodecComboBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    encodeCodecComboBox->setMinimumContentsLength(28);
    populateVideoEncoders(encSettings.value("recording/codec", "auto").toString());
    encodeCodecDetailsLabel = new QLabel(this);
    encodeCodecDetailsLabel->setWordWrap(true);
    encodeCodecDetailsLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    encodeOptionsButton = new QPushButton("Show Encoder Options...", this);
    connect(encodeCodecComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { updateEncoderDetails(); });
    connect(encodeOptionsButton, &QPushButton::clicked,
            this, &SettingsWindow::showEncoderOptions);
    updateEncoderDetails();

    encodeParametersLineEdit = new QLineEdit(this);
    encodeParametersLineEdit->setText(encSettings.value("recording/parameters", "").toString());
    encodeParametersLineEdit->setPlaceholderText(
        "-preset p6 -tune lossless -profile:v rext -pix_fmt yuv444p");
    encodeParametersLineEdit->setToolTip(
        "Additional FFmpeg-style video encoder options passed through MXWrite. "
        "Do not include an input or output filename.");

    encodeRealtimeCheckBox = new QCheckBox("Realtime (low-latency)", this);
    encodeRealtimeCheckBox->setChecked(encSettings.value("recording/realtime", false).toBool());
    encodeRealtimeCheckBox->setToolTip("Enable low-latency encoding. Required for live camera capture.");

    encodeNoDropCheckBox = new QCheckBox("No Drop", this);
    encodeNoDropCheckBox->setChecked(encSettings.value("recording/no_drop", false).toBool());
    encodeNoDropCheckBox->setToolTip(
        "File and graphics modes only: keep one pending frame and process the next frame "
        "as encoder capacity becomes available. Webcam recording always uses wall-clock "
        "timestamps and drops late frames to remain synchronized.");
    encodeNoDropCheckBox->setEnabled(false);

    // ── Input Source group ────────────────────────────────────────────
    auto *sourceGroup = new QGroupBox("Input Source", this);
    auto *sourceGrid = new QGridLayout(sourceGroup);
    sourceGrid->setVerticalSpacing(6);
    sourceGrid->setColumnStretch(1, 1);
    int r = 0;
    sourceGrid->addWidget(cameraOptionRadioButton, r, 0, 1, 2);
    sourceGrid->addWidget(inputVideoOptionRadioButton, ++r, 0, 1, 2);
    sourceGrid->addWidget(graphicsFileOptionRadioButton, ++r, 0, 1, 2);
    sourceGrid->addWidget(new QLabel("Camera:", this), ++r, 0);
    sourceGrid->addWidget(cameraIndexComboBox, r, 1);
    sourceGrid->addWidget(new QLabel("Resolution:", this), ++r, 0);
    sourceGrid->addWidget(cameraResolutionComboBox, r, 1);
    sourceGrid->addWidget(new QLabel("FPS:", this), ++r, 0);
    sourceGrid->addWidget(cameraFPSComboBox, r, 1);
    sourceGrid->addWidget(useYuvCheckBox, ++r, 0, 1, 2);
    sourceGrid->addWidget(new QLabel("Input Video:", this), ++r, 0);
    auto *inputRow = new QHBoxLayout;
    inputRow->setSpacing(4);
    inputRow->addWidget(inputVideoFileLineEdit);
    inputRow->addWidget(browseInputVideoButton);
    sourceGrid->addLayout(inputRow, r, 1);
    sourceGrid->addWidget(hdrStatusLabel, ++r, 1);
    sourceGrid->addWidget(convertHdr10CheckBox, ++r, 1);
    sourceGrid->addWidget(new QLabel("Graphics:", this), ++r, 0);
    auto *graphicsRow = new QHBoxLayout;
    graphicsRow->setSpacing(4);
    graphicsRow->addWidget(graphicsFileLineEdit);
    graphicsRow->addWidget(browseGraphicsButton);
    sourceGrid->addLayout(graphicsRow, r, 1);

    // ── Output group ──────────────────────────────────────────────────
    auto *outputGroup = new QGroupBox("Output", this);
    auto *outputGrid = new QGridLayout(outputGroup);
    outputGrid->setVerticalSpacing(6);
    outputGrid->setColumnStretch(1, 1);
    r = 0;
    outputGrid->addWidget(new QLabel("Screen Resolution:", this), r, 0);
    outputGrid->addWidget(screenResolutionComboBox, r, 1);
    outputGrid->addWidget(saveOutputVideoCheckBox, ++r, 0, 1, 2);
    outputGrid->addWidget(new QLabel("Output File:", this), ++r, 0);
    auto *outputRow = new QHBoxLayout;
    outputRow->setSpacing(4);
    outputRow->addWidget(outputVideoFileLineEdit);
    outputRow->addWidget(browseOutputVideoButton);
    outputGrid->addLayout(outputRow, r, 1);
    outputGrid->addWidget(copyAudioCheckBox, ++r, 0, 1, 2);
    outputGrid->addWidget(writePngCheckBox, ++r, 0, 1, 2);
    outputGrid->addWidget(generateCheckBox, ++r, 0);
    outputGrid->addWidget(generateIntervalSpinBox, r, 1);

    // ── Encoding group ────────────────────────────────────────────────
    auto *encodingGroup = new QGroupBox("Encoding Quality", this);
    auto *encodingGrid = new QGridLayout(encodingGroup);
    encodingGrid->setVerticalSpacing(6);
    encodingGrid->setColumnStretch(1, 1);
    r = 0;
    encodingGrid->addWidget(new QLabel("Preset:", this), r, 0);
    encodingGrid->addWidget(encodePresetComboBox, r, 1);
    encodingGrid->addWidget(new QLabel("Tune:", this), ++r, 0);
    encodingGrid->addWidget(encodeTuneComboBox, r, 1);
    encodingGrid->addWidget(new QLabel("CRF (quality):", this), ++r, 0);
    encodingGrid->addWidget(encodeCrfSpinBox, r, 1);
    encodingGrid->addWidget(new QLabel("Codec:", this), ++r, 0);
    encodingGrid->addWidget(encodeCodecComboBox, r, 1);
    encodingGrid->addWidget(encodeOptionsButton, ++r, 1);
    encodingGrid->addWidget(encodeCodecDetailsLabel, ++r, 0, 1, 2);
    encodingGrid->addWidget(new QLabel("Extra FFmpeg parameters:", this), ++r, 0);
    encodingGrid->addWidget(encodeParametersLineEdit, r, 1);
    encodingGrid->addWidget(encodeRealtimeCheckBox, ++r, 0, 1, 2);
    encodingGrid->addWidget(encodeNoDropCheckBox, ++r, 0, 1, 2);

    // ── Playback group ────────────────────────────────────────────────
    auto *playbackGroup = new QGroupBox("Playback", this);
    auto *playbackGrid = new QGridLayout(playbackGroup);
    playbackGrid->setVerticalSpacing(6);
    playbackGrid->setColumnStretch(1, 1);
    r = 0;
    playbackGrid->addWidget(cudaDeviceLabel, r, 0);
    playbackGrid->addWidget(cudaDeviceComboBox, r, 1);
    playbackGrid->addWidget(new QLabel("Time Speed:", this), ++r, 0);
    playbackGrid->addWidget(timeSpeedSpinBox, r, 1);
    playbackGrid->addWidget(new QLabel("Crossfade (sec):", this), ++r, 0);
    playbackGrid->addWidget(crossFadeSpinBox, r, 1);
    playbackGrid->addWidget(flipCheckBox, ++r, 0, 1, 2);
    playbackGrid->addWidget(fullscreenCheckBox, ++r, 0, 1, 2);
    auto *durationRow = new QHBoxLayout;
    durationRow->addWidget(durationLimitCheckBox);
    durationRow->addWidget(durationLimitSpinBox);
    playbackGrid->addLayout(durationRow, ++r, 0, 1, 2);
    auto *maxSizeRow = new QHBoxLayout;
    maxSizeRow->addWidget(maxSizeLimitCheckBox);
    maxSizeRow->addWidget(maxSizeLimitSpinBox);
    playbackGrid->addLayout(maxSizeRow, ++r, 0, 1, 2);
    auto *cacheRow = new QHBoxLayout;
    cacheRow->addWidget(textureCacheCheckBox);
    cacheRow->addWidget(new QLabel("Delay:", this));
    cacheRow->addWidget(cacheDelaySpinBox);
    cacheRow->addWidget(new QLabel("Size:", this));
    cacheRow->addWidget(cacheSizeSpinBox);
    cacheRow->addStretch();
    playbackGrid->addLayout(cacheRow, ++r, 0, 1, 2);
    playbackGrid->addWidget(textureCacheArrayCheckBox, ++r, 0, 1, 2);

    // ── Display & 3D group ────────────────────────────────────────────
    auto *displayGroup = new QGroupBox("Display & 3D", this);
    auto *displayGrid = new QGridLayout(displayGroup);
    displayGrid->setVerticalSpacing(6);
    displayGrid->setColumnStretch(1, 1);
    r = 0;
    displayGrid->addWidget(enable3dCheckBox, r, 0, 1, 2);
    displayGrid->addWidget(new QLabel("3D Model:", this), ++r, 0);
    auto *modelRow = new QHBoxLayout;
    modelRow->setSpacing(4);
    modelRow->addWidget(modelFileLineEdit);
    modelRow->addWidget(browseModelButton);
    displayGrid->addLayout(modelRow, r, 1);
    displayGrid->addWidget(useOnnxModelCheckBox, ++r, 0, 1, 2);
    displayGrid->addWidget(new QLabel("ONNX Model:", this), ++r, 0);
    auto *onnxModelRow = new QHBoxLayout;
    onnxModelRow->setSpacing(4);
    onnxModelRow->addWidget(onnxModelFileLineEdit);
    onnxModelRow->addWidget(browseOnnxModelButton);
    displayGrid->addLayout(onnxModelRow, r, 1);

    // ── Assemble responsive group layout ──────────────────────────────
    // Groups are organised into independent left/right column VBoxes so
    // each column is laid out top-to-bottom by its own contents (Output
    // -> Playback -> Display sit flush on the right regardless of the
    // left column's height). resizeEvent() reflows between two columns
    // and a single column based on the current dialog width.
    reflowGroups = {sourceGroup, outputGroup, encodingGroup, playbackGroup, displayGroup};

    leftColumn = new QVBoxLayout;
    leftColumn->setSpacing(8);
    rightColumn = new QVBoxLayout;
    rightColumn->setSpacing(8);

    groupsRow = new QHBoxLayout;
    groupsRow->setSpacing(12);
    groupsRow->addLayout(leftColumn, 1);
    groupsRow->addLayout(rightColumn, 1);

    auto *buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch();
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(8);
    mainLayout->addLayout(groupsRow, 1);
    mainLayout->addStretch();
    mainLayout->addLayout(buttonLayout);
    setLayout(mainLayout);
    setWindowTitle("Settings");

    // Allow the dialog to be resized down to fit small/high-DPI displays.
    // The preferred size is intentionally modest (the layout itself will
    // grow naturally based on its contents). On displays where Qt is
    // applying scaling (high-DPI / fractional scaling) the available
    // screen size already accounts for the scale factor, so clamping
    // against availableSize() keeps the dialog inside the screen.
    setSizeGripEnabled(true);
    setMinimumSize(420, 320);
    // Give both responsive columns enough room for descriptive encoder names,
    // option buttons, and status text without requiring an initial resize.
    QSize preferred(960, 840);
    if (QScreen *scr = QGuiApplication::primaryScreen()) {
        const QSize avail = scr->availableSize();
        preferred.setWidth(std::min(preferred.width(), avail.width() - 40));
        preferred.setHeight(std::min(preferred.height(), avail.height() - 80));
    }
    resize(preferred);
    // Initial flow uses 2 columns; resizeEvent will adapt as needed.
    reflowGroupColumns(2);

    // ── Signals ───────────────────────────────────────────────────────
    connect(cameraIndexComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SettingsWindow::onCameraDeviceChanged);
    connect(cameraResolutionComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SettingsWindow::onCameraResolutionChanged);
    connect(cameraFPSComboBox, &QComboBox::currentTextChanged, this, [this](const QString &fpsText) {
        if (fpsText.isEmpty()) {
            return;
        }
        preferredFpsText = fpsText;
        QSettings appSettings("LostSideDead", "acmx2");
        appSettings.setValue("interface/preferred_fps", preferredFpsText);
    });

    connect(textureCacheCheckBox, &QCheckBox::toggled, cacheDelaySpinBox, &QSpinBox::setEnabled);
    connect(textureCacheCheckBox, &QCheckBox::toggled, cacheSizeSpinBox, &QSpinBox::setEnabled);
    connect(textureCacheCheckBox, &QCheckBox::toggled,
            textureCacheArrayCheckBox, &QCheckBox::setEnabled);
    connect(durationLimitCheckBox, &QCheckBox::toggled, durationLimitSpinBox, &QDoubleSpinBox::setEnabled);
    connect(maxSizeLimitCheckBox, &QCheckBox::toggled, maxSizeLimitSpinBox, &QDoubleSpinBox::setEnabled);

    connect(enable3dCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        modelFileLineEdit->setEnabled(checked);
        browseModelButton->setEnabled(checked);
        if (!checked)
            modelFileLineEdit->clear();
    });

    connect(useOnnxModelCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        onnxModelFileLineEdit->setEnabled(checked);
        browseOnnxModelButton->setEnabled(checked);
    });

    connect(browseOnnxModelButton, &QPushButton::clicked, this, &SettingsWindow::browseOnnxModelFile);

    connect(cameraOptionRadioButton, &QRadioButton::toggled, this, [this](bool checked) {
        encodeNoDropCheckBox->setEnabled(!checked);
        if (checked) {
            encodeNoDropCheckBox->setChecked(false);
            cameraIndexComboBox->setEnabled(true);
            cameraResolutionComboBox->setEnabled(true);
            cameraFPSComboBox->setEnabled(true);
            inputVideoFileLineEdit->setEnabled(false);
            browseInputVideoButton->setEnabled(false);
            graphicsFileLineEdit->setEnabled(false);
            browseGraphicsButton->setEnabled(false);
            textureCacheCheckBox->setEnabled(true);
            cacheDelaySpinBox->setEnabled(textureCacheCheckBox->isChecked());
            cacheSizeSpinBox->setEnabled(textureCacheCheckBox->isChecked());
            populateFPS();
            QString currentRes = cameraResolutionComboBox->currentText();
            useYuvCheckBox->setEnabled(yuvResolutions.contains(currentRes));
            if (!useYuvCheckBox->isEnabled())
                useYuvCheckBox->setChecked(false);
        }
    });

    connect(inputVideoOptionRadioButton, &QRadioButton::toggled, this, [this](bool checked) {
        if (checked)
            encodeNoDropCheckBox->setEnabled(true);
        if (checked) {
            cameraIndexComboBox->setEnabled(false);
            cameraResolutionComboBox->setEnabled(false);
            cameraFPSComboBox->setEnabled(true);
            inputVideoFileLineEdit->setEnabled(true);
            browseInputVideoButton->setEnabled(true);
            graphicsFileLineEdit->setEnabled(false);
            browseGraphicsButton->setEnabled(false);
            textureCacheCheckBox->setEnabled(true);
            cacheDelaySpinBox->setEnabled(textureCacheCheckBox->isChecked());
            cacheSizeSpinBox->setEnabled(textureCacheCheckBox->isChecked());
            useYuvCheckBox->setEnabled(false);
            useYuvCheckBox->setChecked(false);
        }
        bool enableAudio = checked && saveOutputVideoCheckBox->isChecked();
        copyAudioCheckBox->setEnabled(enableAudio);
        if (!enableAudio)
            copyAudioCheckBox->setChecked(false);
    });

    connect(graphicsFileOptionRadioButton, &QRadioButton::toggled, this, [this](bool checked) {
        if (checked)
            encodeNoDropCheckBox->setEnabled(true);
        if (checked) {
            cameraIndexComboBox->setEnabled(false);
            cameraResolutionComboBox->setEnabled(false);
            cameraFPSComboBox->setEnabled(true);
            inputVideoFileLineEdit->setEnabled(false);
            browseInputVideoButton->setEnabled(false);
            graphicsFileLineEdit->setEnabled(true);
            browseGraphicsButton->setEnabled(true);
            textureCacheCheckBox->setEnabled(false);
            cacheDelaySpinBox->setEnabled(false);
            cacheSizeSpinBox->setEnabled(false);
            cameraFPSComboBox->clear();
            cameraFPSComboBox->addItems({"24", "30", "60"});
            int preferredIdx = cameraFPSComboBox->findText(preferredFpsText);
            if (preferredIdx >= 0) {
                cameraFPSComboBox->setCurrentIndex(preferredIdx);
            } else {
                cameraFPSComboBox->setCurrentIndex(1);
            }
            useYuvCheckBox->setEnabled(false);
            useYuvCheckBox->setChecked(false);
        }
    });

    connect(saveOutputVideoCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        outputVideoFileLineEdit->setEnabled(checked);
        browseOutputVideoButton->setEnabled(checked);
        bool enableAudio = checked && inputVideoOptionRadioButton->isChecked();
        copyAudioCheckBox->setEnabled(enableAudio);
        if (!enableAudio)
            copyAudioCheckBox->setChecked(false);
    });

    connect(okButton, &QPushButton::clicked, this, &SettingsWindow::acceptSettings);
    connect(cancelButton, &QPushButton::clicked, this, &SettingsWindow::rejectSettings);
    connect(browseInputVideoButton, &QPushButton::clicked, this, &SettingsWindow::browseInputVideoFile);
    connect(browseOutputVideoButton, &QPushButton::clicked, this, &SettingsWindow::browseOutputVideoFile);
    connect(browseGraphicsButton, &QPushButton::clicked, this, &SettingsWindow::browseGraphicsFile);
    connect(browseModelButton, &QPushButton::clicked, this, &SettingsWindow::browseModelFile);

    // ── Initial enabled states ────────────────────────────────────────
    inputVideoFileLineEdit->setEnabled(false);
    browseInputVideoButton->setEnabled(false);
    graphicsFileLineEdit->setEnabled(false);
    browseGraphicsButton->setEnabled(false);
    outputVideoFileLineEdit->setEnabled(false);
    browseOutputVideoButton->setEnabled(false);

    loadUiState();
    if (cameraOptionRadioButton->isChecked()) {
        encodeNoDropCheckBox->setChecked(false);
        encodeNoDropCheckBox->setEnabled(false);
    }
}

void SettingsWindow::reflowGroupColumns(int columns) {
    if (!leftColumn || !rightColumn || reflowGroups.isEmpty()) {
        return;
    }
    columns = std::max(1, columns);
    if (columns == currentColumnCount) {
        return;
    }
    currentColumnCount = columns;

    auto detachAll = [](QVBoxLayout *col) {
        while (QLayoutItem *item = col->takeAt(0)) {
            // Stretch / spacer items have no widget; they are owned here.
            if (!item->widget()) {
                delete item;
            } else {
                delete item; // widget stays alive; reparented when re-added
            }
        }
    };
    detachAll(leftColumn);
    detachAll(rightColumn);

    // reflowGroups order: 0=source, 1=output, 2=encoding, 3=playback, 4=display.
    if (columns >= 2) {
        // Left:  Source, Encoding
        leftColumn->addWidget(reflowGroups[0]);
        leftColumn->addWidget(reflowGroups[2]);
        leftColumn->addStretch();
        // Right: Output -> Playback -> Display (flush, top-aligned)
        rightColumn->addWidget(reflowGroups[1]);
        rightColumn->addWidget(reflowGroups[3]);
        rightColumn->addWidget(reflowGroups[4]);
        rightColumn->addStretch();
        rightColumn->parentWidget(); // no-op; keep code consistent
        if (groupsRow) {
            groupsRow->setStretch(0, 1);
            groupsRow->setStretch(1, 1);
        }
        for (QGroupBox *g : reflowGroups) {
            g->setVisible(true);
        }
    } else {
        // Single column: Source, Output, Playback, Display, Encoding.
        const int singleOrder[] = {0, 1, 3, 4, 2};
        for (int idx : singleOrder) {
            leftColumn->addWidget(reflowGroups[idx]);
        }
        leftColumn->addStretch();
        if (groupsRow) {
            groupsRow->setStretch(0, 1);
            groupsRow->setStretch(1, 0);
        }
    }
}

void SettingsWindow::resizeEvent(QResizeEvent *event) {
    QDialog::resizeEvent(event);
    const int w = width();
    const int columns = (w >= 720) ? 2 : 1;
    reflowGroupColumns(columns);
}

void SettingsWindow::loadUiState() {
    QSettings appSettings("LostSideDead", "acmx2");

    preferredFpsText = appSettings.value(
                                      "interface/preferred_fps",
                                      appSettings.value("interface/camera_fps", "30"))
                           .toString();
    if (preferredFpsText.isEmpty()) {
        preferredFpsText = "30";
    }

    QString inputMode = appSettings.value("interface/input_mode", "camera").toString();
    if (inputMode == "video") {
        inputVideoOptionRadioButton->setChecked(true);
    } else if (inputMode == "graphic") {
        graphicsFileOptionRadioButton->setChecked(true);
    } else {
        cameraOptionRadioButton->setChecked(true);
    }

    int cameraDevice = appSettings.value("interface/camera_device", 0).toInt();
    int camIdx = cameraIndexComboBox->findData(cameraDevice);
    if (camIdx >= 0) {
        cameraIndexComboBox->setCurrentIndex(camIdx);
    }

    QString cameraRes = appSettings.value("interface/camera_resolution", "1280x720").toString();
    int camResIdx = cameraResolutionComboBox->findText(cameraRes);
    if (camResIdx >= 0) {
        cameraResolutionComboBox->setCurrentIndex(camResIdx);
    }

    QString cameraFps = appSettings.value("interface/camera_fps", preferredFpsText).toString();
    int camFpsIdx = cameraFPSComboBox->findText(cameraFps);
    if (camFpsIdx >= 0) {
        cameraFPSComboBox->setCurrentIndex(camFpsIdx);
        preferredFpsText = cameraFps;
    }

    QString screenRes = appSettings.value("interface/screen_resolution", "Default").toString();
    QSize restored_screen_resolution;
    if (parse_even_resolution(screenRes, restored_screen_resolution)) {
        const QString normalized_screen_resolution =
            resolution_text(restored_screen_resolution);
        const int screenResIdx =
            screenResolutionComboBox->findText(normalized_screen_resolution);
        if (screenResIdx >= 0) {
            screenResolutionComboBox->setCurrentIndex(screenResIdx);
        } else {
            screenResolutionComboBox->setCurrentText(
                normalized_screen_resolution);
        }
    }

    inputVideoFileLineEdit->setText(appSettings.value("interface/input_video", "").toString());
    graphicsFileLineEdit->setText(appSettings.value("interface/graphics_file", "").toString());

    saveOutputVideoCheckBox->setChecked(appSettings.value("interface/save_output", false).toBool());
    outputVideoFileLineEdit->setText(appSettings.value("interface/output_video", "").toString());
    copyAudioCheckBox->setChecked(appSettings.value("interface/copy_audio", false).toBool());
    writePngCheckBox->setChecked(appSettings.value("interface/write_png", false).toBool());
    generateCheckBox->setChecked(appSettings.value("interface/generate_enabled", false).toBool());
    generateIntervalSpinBox->setValue(appSettings.value("interface/generate_interval", 30).toInt());
    generateIntervalSpinBox->setEnabled(generateCheckBox->isChecked());

    // Re-probe HDR for whatever video file we just restored so the checkbox
    // reflects the actual capabilities of the cached path.
    detectInputHdr();
    if (convertHdr10CheckBox) {
        const bool wantHdr10 =
            appSettings.value("interface/convert_to_hdr10", false).toBool();
        if (wantHdr10 && convertHdr10CheckBox->isEnabled()) {
            convertHdr10CheckBox->setChecked(true);
        }
    }

    fullscreenCheckBox->setChecked(appSettings.value("interface/fullscreen", false).toBool());
    enable3dCheckBox->setChecked(appSettings.value("interface/enable_3d", false).toBool());
    modelFileLineEdit->setText(appSettings.value("interface/model_file", "cube.mxmod.z").toString());
    useOnnxModelCheckBox->setChecked(appSettings.value("interface/use_onnx_model", false).toBool());
    onnxModelFileLineEdit->setText(appSettings.value("interface/onnx_model_file", "").toString());
    onnxModelFileLineEdit->setEnabled(useOnnxModelCheckBox->isChecked());
    browseOnnxModelButton->setEnabled(useOnnxModelCheckBox->isChecked());

    textureCacheCheckBox->setChecked(appSettings.value("interface/texture_cache", false).toBool());
    if (auto *arrayCheck =
            findChild<QCheckBox *>("textureCacheArrayCheckBox")) {
        arrayCheck->setChecked(
            appSettings.value("interface/texture_cache_array", false).toBool());
        arrayCheck->setEnabled(textureCacheCheckBox->isChecked());
    }
    cacheDelaySpinBox->setValue(appSettings.value("interface/cache_delay", 1).toInt());
    cacheSizeSpinBox->setValue(appSettings.value("interface/cache_size", 8).toInt());
    useYuvCheckBox->setChecked(appSettings.value("interface/use_yuv", false).toBool());

    int cudaDevice = appSettings.value("interface/cuda_device", 0).toInt();
    int cudaIdx = cudaDeviceComboBox->findData(cudaDevice);
    if (cudaIdx >= 0) {
        cudaDeviceComboBox->setCurrentIndex(cudaIdx);
    }

    timeSpeedSpinBox->setValue(appSettings.value("interface/time_speed", 1.0).toDouble());
    durationLimitCheckBox->setChecked(appSettings.value("interface/duration_enabled", false).toBool());
    durationLimitSpinBox->setValue(appSettings.value("interface/duration_seconds", 60.0).toDouble());
    maxSizeLimitCheckBox->setChecked(appSettings.value("interface/max_size_enabled", false).toBool());
    maxSizeLimitSpinBox->setValue(appSettings.value("interface/max_size_mb", 500.0).toDouble());
    crossFadeSpinBox->setValue(appSettings.value("interface/crossfade", 0.5).toDouble());
    flipCheckBox->setChecked(appSettings.value("interface/flip", false).toBool());

    // Recompute YUV availability for the restored camera/resolution even when
    // the combo-box index did not change (Qt won't emit change signals then).
    onCameraResolutionChanged(cameraResolutionComboBox->currentIndex());
}

void SettingsWindow::saveUiState() {
    QSettings appSettings("LostSideDead", "acmx2");

    QString inputMode = "camera";
    if (inputVideoOptionRadioButton->isChecked()) {
        inputMode = "video";
    } else if (graphicsFileOptionRadioButton->isChecked()) {
        inputMode = "graphic";
    }
    appSettings.setValue("interface/input_mode", inputMode);

#ifdef _WIN32
    appSettings.setValue("interface/camera_device", resolveWindowsSelectedCameraIndex(cameraIndexComboBox));
#else
    appSettings.setValue("interface/camera_device", cameraIndexComboBox->currentData().toInt());
#endif
    appSettings.setValue("interface/camera_resolution", cameraResolutionComboBox->currentText());
    appSettings.setValue("interface/camera_fps", cameraFPSComboBox->currentText());
    appSettings.setValue("interface/preferred_fps", cameraFPSComboBox->currentText());
    QSize screen_resolution;
    if (parse_even_resolution(screenResolutionComboBox->currentText(),
                              screen_resolution)) {
        appSettings.setValue("interface/screen_resolution",
                             resolution_text(screen_resolution));
    }

    appSettings.setValue("interface/input_video", inputVideoFileLineEdit->text());
    appSettings.setValue("interface/graphics_file", graphicsFileLineEdit->text());

    appSettings.setValue("interface/save_output", saveOutputVideoCheckBox->isChecked());
    appSettings.setValue("interface/output_video", outputVideoFileLineEdit->text());
    appSettings.setValue("interface/copy_audio", copyAudioCheckBox->isChecked());
    appSettings.setValue("interface/write_png", writePngCheckBox->isChecked());
    appSettings.setValue("interface/generate_enabled", generateCheckBox->isChecked());
    appSettings.setValue("interface/generate_interval", generateIntervalSpinBox->value());
    if (convertHdr10CheckBox) {
        appSettings.setValue("interface/convert_to_hdr10",
                             convertHdr10CheckBox->isChecked());
    }

    appSettings.setValue("interface/fullscreen", fullscreenCheckBox->isChecked());
    appSettings.setValue("interface/enable_3d", enable3dCheckBox->isChecked());
    appSettings.setValue("interface/model_file", modelFileLineEdit->text());
    appSettings.setValue("interface/use_onnx_model", useOnnxModelCheckBox->isChecked());
    appSettings.setValue("interface/onnx_model_file", onnxModelFileLineEdit->text());

    appSettings.setValue("interface/texture_cache", textureCacheCheckBox->isChecked());
    if (auto *arrayCheck =
            findChild<QCheckBox *>("textureCacheArrayCheckBox")) {
        appSettings.setValue("interface/texture_cache_array",
                             arrayCheck->isChecked());
    }
    appSettings.setValue("interface/cache_delay", cacheDelaySpinBox->value());
    appSettings.setValue("interface/cache_size", cacheSizeSpinBox->value());
    appSettings.setValue("interface/use_yuv", useYuvCheckBox->isChecked());

    appSettings.setValue("interface/cuda_device", cudaDeviceComboBox->currentData().toInt());
    appSettings.setValue("interface/time_speed", timeSpeedSpinBox->value());
    appSettings.setValue("interface/duration_enabled", durationLimitCheckBox->isChecked());
    appSettings.setValue("interface/duration_seconds", durationLimitSpinBox->value());
    appSettings.setValue("interface/max_size_enabled", maxSizeLimitCheckBox->isChecked());
    appSettings.setValue("interface/max_size_mb", maxSizeLimitSpinBox->value());
    appSettings.setValue("interface/crossfade", crossFadeSpinBox->value());
    appSettings.setValue("interface/flip", flipCheckBox->isChecked());
}

bool SettingsWindow::is3dEnabled() const {
    return enable3dCheckBox->isChecked();
}

int SettingsWindow::getSelectedCameraIndex() const {
    return selectedCameraIndex;
}

QSize SettingsWindow::getSelectedCameraResolution() const {
    return selectedCameraResolution;
}

QSize SettingsWindow::getSelectedScreenResolution() const {
    return selectedScreenResolution;
}

int SettingsWindow::getCameraFPS() const {
    return cameraFPS;
}

QString SettingsWindow::getInputVideoFile() const {
    return inputVideoFile;
}

QString SettingsWindow::getOutputVideoFile() const {
    return outputVideoFile;
}

QString SettingsWindow::getGraphicsFile() const {
    return graphicsFile;
}

bool SettingsWindow::isUsingInputVideoFile() const {
    return useInputVideoFile;
}

bool SettingsWindow::isUsingGraphicsFile() const {
    return useGraphicsFile;
}

bool SettingsWindow::isSavingToOutputVideoFile() const {
    return saveOutputVideoFile;
}

bool SettingsWindow::isInputHdrDetected() const {
    return inputHdrDetected;
}

bool SettingsWindow::isConvertToHdr10Enabled() const {
    return convertHdr10CheckBox && convertHdr10CheckBox->isChecked() &&
           convertHdr10CheckBox->isEnabled();
}

bool SettingsWindow::isTextureCacheEnabled() const {
    return textureCacheCheckBox->isChecked();
}

int SettingsWindow::getCacheDelay() const {
    return cacheDelaySpinBox->value();
}

int SettingsWindow::getCacheSize() const {
    return cacheSizeSpinBox->value();
}

bool SettingsWindow::isFullscreen() const {
    return fullscreenCheckBox->isChecked();
}

bool SettingsWindow::isCopyAudioEnabled() const {
    return copyAudioCheckBox->isChecked();
}

bool SettingsWindow::isPngOutputEnabled() const {
    return writePngCheckBox->isChecked();
}

bool SettingsWindow::isGenerateEnabled() const {
    return generateCheckBox && generateCheckBox->isChecked();
}

int SettingsWindow::getGenerateInterval() const {
    return generateIntervalSpinBox ? generateIntervalSpinBox->value() : 0;
}

bool SettingsWindow::isUseYuvEnabled() const {
    return useYuvCheckBox->isChecked();
}

QString SettingsWindow::getModelFile() const {
    return modelFile;
}

bool SettingsWindow::isOnnxModelEnabled() const {
    return useOnnxModelCheckBox->isChecked();
}

QString SettingsWindow::getOnnxModelFile() const {
    return onnxModelFile;
}

int SettingsWindow::getSelectedCudaDevice() const {
    return selectedCudaDevice;
}

void SettingsWindow::setCudaAvailable(bool available) {
    if (cudaDeviceComboBox) {
        cudaDeviceComboBox->setEnabled(available);
        if (!available) {
            cudaDeviceComboBox->clear();
            cudaDeviceComboBox->addItem("CUDA disabled (acmx2 built without CUDA)", 0);
            cudaDeviceComboBox->setToolTip("CUDA support is not compiled into this acmx2 build.");
        }
    }
    if (cudaDeviceLabel) {
        cudaDeviceLabel->setEnabled(available);
    }
}

float SettingsWindow::getTimeSpeed() const {
    return static_cast<float>(timeSpeedSpinBox->value());
}

bool SettingsWindow::isDurationLimitEnabled() const {
    return durationLimitCheckBox->isChecked();
}

double SettingsWindow::getDurationLimit() const {
    return durationLimitSpinBox->value();
}

bool SettingsWindow::isMaxSizeLimitEnabled() const {
    return maxSizeLimitCheckBox->isChecked();
}

double SettingsWindow::getMaxSizeLimit() const {
    return maxSizeLimitSpinBox->value();
}

float SettingsWindow::getCrossFadeDuration() const {
    return static_cast<float>(crossFadeSpinBox->value());
}

bool SettingsWindow::isFlipEnabled() const {
    return flipCheckBox->isChecked();
}

QString SettingsWindow::getEncodePreset() const {
    return encodePresetComboBox ? encodePresetComboBox->currentText() : QString("medium");
}

QString SettingsWindow::getEncodeTune() const {
    if (!encodeTuneComboBox)
        return QString();
    QString t = encodeTuneComboBox->currentText();
    return (t == "none") ? QString() : t;
}

int SettingsWindow::getEncodeCrf() const {
    return encodeCrfSpinBox ? encodeCrfSpinBox->value() : 18;
}

QString SettingsWindow::getEncodeCodec() const {
    if (!encodeCodecComboBox) {
        return QString("auto");
    }
    const QString encoderName = encodeCodecComboBox->currentData().toString();
    return encoderName.isEmpty() ? encodeCodecComboBox->currentText() : encoderName;
}

QString SettingsWindow::getEncodeParameters() const {
    return encodeParametersLineEdit ? encodeParametersLineEdit->text().trimmed() : QString();
}

bool SettingsWindow::isEncodeRealtime() const {
    return encodeRealtimeCheckBox && encodeRealtimeCheckBox->isChecked();
}

bool SettingsWindow::isEncodeNoDrop() const {
    return encodeNoDropCheckBox && encodeNoDropCheckBox->isEnabled() &&
           encodeNoDropCheckBox->isChecked();
}

QString SettingsWindow::getCameraName(int device_index) {
#ifdef __APPLE__
    for (int i = 0; i < cameraIndexComboBox->count(); ++i) {
        if (cameraIndexComboBox->itemData(i).toInt() == device_index) {
            const QString label = cameraIndexComboBox->itemText(i);
            const int suffixPos = label.lastIndexOf(" [");
            return suffixPos > 0 ? label.left(suffixPos).trimmed() : label;
        }
    }

    const QStringList cameraNames = appleCameraNamesFromSystemProfiler();
    if (device_index >= 0 && device_index < cameraNames.size()) {
        return cameraNames.at(device_index);
    }
    return "Unknown Camera";
#elif defined(_WIN32)
    for (int i = 0; i < cameraIndexComboBox->count(); ++i) {
        if (cameraIndexComboBox->itemData(i).toInt() == device_index) {
            const QString label = cameraIndexComboBox->itemText(i);
            const int suffixPos = label.lastIndexOf(" [");
            return suffixPos > 0 ? label.left(suffixPos).trimmed() : label;
        }
    }

    const auto devices = enumerateWindowsCameraDevices();
    for (const auto &device : devices) {
        if (device.index == device_index) {
            return device.name;
        }
    }
    return QString("Camera %1").arg(device_index);
#else
    QString sysfs_path = QString("/sys/class/video4linux/video%1/name").arg(device_index);
    QFile file(sysfs_path);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QString name = QString::fromUtf8(file.readLine()).trimmed();
        file.close();
        if (!name.isEmpty()) {
            return name;
        }
    }
    return "Unknown Camera";
#endif
}

void SettingsWindow::acceptSettings() {
    QSize screen_resolution;
    if (!parse_even_resolution(screenResolutionComboBox->currentText(),
                               screen_resolution)) {
        QMessageBox::warning(
            this,
            "Invalid window resolution",
            "Enter Default or a resolution in WxH format. Width and height "
            "must be positive numbers divisible by 2 (for example, 1920x1080).");
        screenResolutionComboBox->setFocus();
        if (screenResolutionComboBox->lineEdit()) {
            screenResolutionComboBox->lineEdit()->selectAll();
        }
        return;
    }

    useInputVideoFile = inputVideoOptionRadioButton->isChecked();
    useGraphicsFile = graphicsFileOptionRadioButton->isChecked();
    saveOutputVideoFile = saveOutputVideoCheckBox->isChecked();

    if (useInputVideoFile) {
        if (inputVideoFileLineEdit->text().isEmpty()) {
            QMessageBox::information(this, "Video file required", "When using video file mode, a selected video file is required");
            return;
        }
        inputVideoFile = inputVideoFileLineEdit->text();
    } else if (useGraphicsFile) {
        if (graphicsFileLineEdit->text().isEmpty()) {
            QMessageBox::information(this, "Graphics file required", "When using graphics file mode, a selected graphics file is required");
            return;
        }
        graphicsFile = graphicsFileLineEdit->text();
    } else {

#ifdef _WIN32
        selectedCameraIndex = resolveWindowsSelectedCameraIndex(cameraIndexComboBox);
#else
        selectedCameraIndex = cameraIndexComboBox->currentData().toInt();
#endif
        QStringList cameraResParts = cameraResolutionComboBox->currentText().split('x');
        if (cameraResParts.size() == 2) {
            selectedCameraResolution = QSize(cameraResParts[0].toInt(), cameraResParts[1].toInt());
        }
    }

    cameraFPS = cameraFPSComboBox->currentText().toInt();

    selectedScreenResolution = screen_resolution;
    screenResolutionComboBox->setCurrentText(
        resolution_text(screen_resolution));

    if (saveOutputVideoFile) {
        outputVideoFile = outputVideoFileLineEdit->text();
        if (outputVideoFile.isEmpty()) {
            QMessageBox::information(this, "Output required", "Requires you set a output filename");
            reject();
            return;
        }

        if (useInputVideoFile &&
            pathsReferToSameFile(inputVideoFile, outputVideoFile)) {
            QMessageBox::critical(
                this,
                "Input and output files must be different",
                "You cannot process and write to the same video file. Select a different output file.");
            return;
        }
    }

    if (enable3dCheckBox->isChecked()) {
        modelFile = modelFileLineEdit->text();
    }

    if (useOnnxModelCheckBox->isChecked()) {
        onnxModelFile = onnxModelFileLineEdit->text();
    } else {
        onnxModelFile.clear();
    }

    selectedCudaDevice = cudaDeviceComboBox->currentData().toInt();

    saveUiState();

    // Persist encoding quality settings for next session.
    QSettings encSettings("LostSideDead", "acmx2");
    if (encodePresetComboBox)
        encSettings.setValue("recording/preset", encodePresetComboBox->currentText());
    if (encodeTuneComboBox)
        encSettings.setValue("recording/tune", encodeTuneComboBox->currentText());
    if (encodeCrfSpinBox)
        encSettings.setValue("recording/crf", encodeCrfSpinBox->value());
    if (encodeCodecComboBox)
        encSettings.setValue("recording/codec", getEncodeCodec());
    if (encodeParametersLineEdit)
        encSettings.setValue("recording/parameters", encodeParametersLineEdit->text().trimmed());
    if (encodeRealtimeCheckBox)
        encSettings.setValue("recording/realtime", encodeRealtimeCheckBox->isChecked());
    if (encodeNoDropCheckBox)
        encSettings.setValue("recording/no_drop", encodeNoDropCheckBox->isChecked());

    accept();
}

void SettingsWindow::rejectSettings() {
    saveUiState();
    reject();
}

void SettingsWindow::browseInputVideoFile() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastInputVideoDir", "").toString();
    QString fileName = QFileDialog::getOpenFileName(this, "Select Input Video File", lastDir, "Video Files (*.mp4 *.avi *.mkv *.mov)");
    if (!fileName.isEmpty()) {
        appSettings.setValue("lastInputVideoDir", QFileInfo(fileName).absolutePath());
        inputVideoFileLineEdit->setText(fileName);
        detectInputHdr();
    }
}

void SettingsWindow::detectInputHdr() {
    inputHdrDetected = false;
    if (!convertHdr10CheckBox || !hdrStatusLabel) {
        return;
    }

    const QString file = inputVideoFileLineEdit ? inputVideoFileLineEdit->text() : QString();
    if (file.isEmpty() || !QFileInfo::exists(file)) {
        hdrStatusLabel->setText("HDR: not checked");
        convertHdr10CheckBox->setEnabled(false);
        convertHdr10CheckBox->setChecked(false);
        return;
    }

    QProcess probe;
    QStringList args;
    args << "-v" << "error"
         << "-select_streams" << "v:0"
         << "-show_entries" << "stream=color_transfer,color_primaries,color_space"
         << "-of" << "default=noprint_wrappers=1:nokey=0"
         << file;
    probe.start("ffprobe", args);
    if (!probe.waitForStarted(3000)) {
        hdrStatusLabel->setText("HDR: ffprobe not available");
        convertHdr10CheckBox->setEnabled(false);
        convertHdr10CheckBox->setChecked(false);
        return;
    }
    if (!probe.waitForFinished(8000)) {
        probe.kill();
        probe.waitForFinished(1000);
        hdrStatusLabel->setText("HDR: ffprobe timed out");
        convertHdr10CheckBox->setEnabled(false);
        convertHdr10CheckBox->setChecked(false);
        return;
    }

    const QString output = QString::fromUtf8(probe.readAllStandardOutput()).toLower();
    QString transfer;
    QString primaries;
    QString space;
    const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    for (const QString &line : lines) {
        const QString trimmed = line.trimmed();
        const qsizetype eq = trimmed.indexOf('=');
        if (eq <= 0) {
            continue;
        }
        const QString key = trimmed.left(eq);
        const QString value = trimmed.mid(eq + 1).trimmed();
        if (key == "color_transfer") {
            transfer = value;
        } else if (key == "color_primaries") {
            primaries = value;
        } else if (key == "color_space") {
            space = value;
        }
    }

    const bool isHlg = transfer.contains("arib-std-b67") ||
                       transfer.contains("arib_std_b67") ||
                       transfer.contains("hlg");
    const bool isPq = transfer.contains("smpte2084") ||
                      transfer.contains("smpte-2084") ||
                      transfer.contains("pq");
    const bool isBt2020 = primaries.contains("bt2020") || space.contains("bt2020");
    inputHdrDetected = isHlg || isPq || isBt2020;

    QString label;
    if (isPq) {
        label = "HDR: detected (PQ / SMPTE2084)";
    } else if (isHlg) {
        label = "HDR: detected (HLG) - can convert to HDR10";
    } else if (isBt2020) {
        label = "HDR: detected (BT.2020)";
    } else {
        QString t = transfer.isEmpty() ? QStringLiteral("unknown") : transfer;
        label = "HDR: not detected (transfer=" + t + ")";
    }
    hdrStatusLabel->setText(label);
    // Only enable HDR10 conversion checkbox for HLG sources
    convertHdr10CheckBox->setEnabled(isHlg);
    if (!isHlg) {
        convertHdr10CheckBox->setChecked(false);
    }
}

void SettingsWindow::browseOutputVideoFile() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastOutputVideoDir", "").toString();
    static const QStringList kVideoExts = {
        "mp4", "mkv", "mov", "avi", "m4v",
        "ts", "mts", "m2ts", "mpg", "mpeg",
        "flv", "f4v", "3gp", "3g2", "wmv",
        "asf", "vob"};
    QStringList allPattern;
    for (const QString &e : kVideoExts)
        allPattern << ("*." + e);
    QString filter = QString("Video Files (%1)").arg(allPattern.join(' '));
    filter += ";;MP4 Files (*.mp4);;Matroska Files (*.mkv);;QuickTime Files (*.mov);;AVI Files (*.avi);;MPEG-TS Files (*.ts *.mts *.m2ts);;MPEG Files (*.mpg *.mpeg);;Flash Video (*.flv *.f4v);;3GPP Files (*.3gp *.3g2);;Windows Media (*.wmv *.asf);;DVD VOB (*.vob);;All Files (*)";
    QString fileName = QFileDialog::getSaveFileName(this, "Select Output Video File", lastDir, filter);
    if (!fileName.isEmpty()) {
        appSettings.setValue("lastOutputVideoDir", QFileInfo(fileName).absolutePath());
        bool hasKnownExt = false;
        for (const QString &e : kVideoExts) {
            if (fileName.endsWith("." + e, Qt::CaseInsensitive)) {
                hasKnownExt = true;
                break;
            }
        }
        if (!hasKnownExt) {
            fileName += ".mp4";
        }
        outputVideoFileLineEdit->setText(fileName);
    }
}

void SettingsWindow::browseGraphicsFile() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastGraphicsDir", "").toString();
    QString fileName = QFileDialog::getOpenFileName(this, "Select Graphics File", lastDir, "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.gif)");
    if (!fileName.isEmpty()) {
        appSettings.setValue("lastGraphicsDir", QFileInfo(fileName).absolutePath());
        graphicsFileLineEdit->setText(fileName);
    }
}

void SettingsWindow::browseModelFile() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastModelDir", "").toString();
    QString fileName = QFileDialog::getOpenFileName(this, "Select 3D Model File", lastDir, "Model Files (*.mxmod *.mxmod.z *.obj)");
    if (!fileName.isEmpty()) {
        appSettings.setValue("lastModelDir", QFileInfo(fileName).absolutePath());
        modelFileLineEdit->setText(fileName);
    }
}

void SettingsWindow::browseOnnxModelFile() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastOnnxModelDir", "").toString();
    if (lastDir.isEmpty())
        lastDir = QDir::homePath();
    QString fileName = QFileDialog::getOpenFileName(this, "Select ONNX Model YAML Config", lastDir, "YAML Config Files (*.yaml *.yml)");
    if (!fileName.isEmpty()) {
        appSettings.setValue("lastOnnxModelDir", QFileInfo(fileName).absolutePath());
        onnxModelFileLineEdit->setText(fileName);
    }
}

void SettingsWindow::enumerateDevice(int deviceIndex) {
    deviceCapabilities.clear();
    yuvResolutions.clear();

#ifdef __APPLE__
    QProcess process;
    process.start("ffmpeg", {"-hide_banner", "-f", "avfoundation", "-list_formats", "true", "-i",
                             QString("%1:none").arg(deviceIndex)});
    process.waitForFinished(5000);

    const QString output = QString::fromUtf8(process.readAllStandardError()) +
                           QString::fromUtf8(process.readAllStandardOutput());

    if (output.isEmpty() || process.error() == QProcess::FailedToStart) {
        populateAppleDefaultCapabilities(deviceCapabilities);
        populateResolutions();
        return;
    }

    const QRegularExpression rangeRegex(R"((\d+x\d+)\s*@\s*\[\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*\]\s*fps)");
    const QRegularExpression singleRegex(R"((\d+x\d+)\s*@\s*(\d+(?:\.\d+)?)\s*fps)");

    const QStringList lines = output.split('\n');
    for (const QString &line : lines) {
        QRegularExpressionMatch rangeMatch = rangeRegex.match(line);
        if (rangeMatch.hasMatch()) {
            const QString resolution = rangeMatch.captured(1);
            const double maxFps = rangeMatch.captured(3).toDouble();
            appendUniqueFps(deviceCapabilities[resolution], maxFps);
            continue;
        }

        QRegularExpressionMatch singleMatch = singleRegex.match(line);
        if (singleMatch.hasMatch()) {
            const QString resolution = singleMatch.captured(1);
            const double fps = singleMatch.captured(2).toDouble();
            appendUniqueFps(deviceCapabilities[resolution], fps);
        }
    }

    if (deviceCapabilities.isEmpty()) {
        populateAppleDefaultCapabilities(deviceCapabilities);
    }

    populateResolutions();
    return;
#elif defined(_WIN32)
    ComInitScope comScope;
    if (!comScope.ready()) {
        populateResolutions();
        return;
    }

    const auto streamConfig = openWindowsStreamConfig(deviceIndex);
    if (!streamConfig) {
        populateResolutions();
        return;
    }

    int capabilityCount = 0;
    int capabilitySize = 0;
    if (FAILED(streamConfig->GetNumberOfCapabilities(&capabilityCount, &capabilitySize)) ||
        capabilityCount <= 0 || capabilitySize <= 0) {
        populateResolutions();
        return;
    }

    QByteArray capabilityBuffer(capabilitySize, 0);
    for (int capabilityIndex = 0; capabilityIndex < capabilityCount; ++capabilityIndex) {
        AM_MEDIA_TYPE *mediaType = nullptr;
        if (FAILED(streamConfig->GetStreamCaps(capabilityIndex, &mediaType,
                                               reinterpret_cast<BYTE *>(capabilityBuffer.data()))) ||
            mediaType == nullptr) {
            continue;
        }

        const std::unique_ptr<AM_MEDIA_TYPE, decltype(&freeMediaType)> mediaTypeGuard(mediaType, &freeMediaType);

        QSize resolution;
        QString formatName;
        double formatFps = 0.0;
        if (!extractDirectShowFormat(mediaType, resolution, formatName, formatFps)) {
            continue;
        }

        const QString resolutionKey = QString("%1x%2").arg(resolution.width()).arg(resolution.height());
        if (isYuvSubtype(mediaType->subtype)) {
            yuvResolutions.insert(resolutionKey);
        }

        appendUniqueFps(deviceCapabilities[resolutionKey], formatFps);

        if (capabilitySize >= static_cast<int>(sizeof(VIDEO_STREAM_CONFIG_CAPS))) {
            const auto *caps = reinterpret_cast<const VIDEO_STREAM_CONFIG_CAPS *>(capabilityBuffer.constData());
            if (caps->MinFrameInterval > 0) {
                appendUniqueFps(deviceCapabilities[resolutionKey],
                                10000000.0 / static_cast<double>(caps->MinFrameInterval));
            }
            if (caps->MaxFrameInterval > 0) {
                appendUniqueFps(deviceCapabilities[resolutionKey],
                                10000000.0 / static_cast<double>(caps->MaxFrameInterval));
            }
        }
    }

    populateResolutions();
    return;
#else
    QProcess process;
    process.start(executablePath, QStringList() << "--enumerate-device" << QString::number(deviceIndex));
    process.waitForFinished(5000);

    QString output = process.readAllStandardOutput();
    if (output.isEmpty() || process.exitCode() != 0) {
        populateResolutions();
        return;
    }

    // Parse lines like: "    1920x1080 @ 30.0 fps, 24.0 fps"
    QRegularExpression resRegex(R"(^\s+(\d+x\d+)\s*@\s*(.+)$)");
    QRegularExpression fpsRegex(R"((\d+(?:\.\d+)?)\s*fps)");
    QRegularExpression formatRegex(R"(^\s*Format:\s*(\S+))");

    QString currentFormat;
    QStringList lines = output.split('\n');
    for (const QString &line : lines) {
        QRegularExpressionMatch fmtMatch = formatRegex.match(line);
        if (fmtMatch.hasMatch()) {
            currentFormat = fmtMatch.captured(1).toUpper();
            continue;
        }

        QRegularExpressionMatch resMatch = resRegex.match(line);
        if (resMatch.hasMatch()) {
            QString resolution = resMatch.captured(1);
            QString fpsStr = resMatch.captured(2);
            QList<double> fpsList;

            QRegularExpressionMatchIterator it = fpsRegex.globalMatch(fpsStr);
            while (it.hasNext()) {
                QRegularExpressionMatch fpsMatch = it.next();
                fpsList.append(fpsMatch.captured(1).toDouble());
            }

            if (currentFormat == "YUYV") {
                yuvResolutions.insert(resolution);
            }

            if (deviceCapabilities.contains(resolution)) {
                // Merge FPS values from different formats
                QList<double> &existing = deviceCapabilities[resolution];
                for (double fps : fpsList) {
                    if (!existing.contains(fps)) {
                        existing.append(fps);
                    }
                }
            } else {
                deviceCapabilities[resolution] = fpsList;
            }
        }
    }

    populateResolutions();
#endif
}

void SettingsWindow::populateResolutions() {
    cameraResolutionComboBox->blockSignals(true);
    cameraResolutionComboBox->clear();
    cameraResolutionComboBox->addItem("Default");

    if (deviceCapabilities.isEmpty()) {
        cameraResolutionComboBox->blockSignals(false);
        populateFPS();
        return;
    }

    // Sort resolutions by pixel count descending
    QStringList resolutions = deviceCapabilities.keys();
    std::sort(resolutions.begin(), resolutions.end(), [](const QString &a, const QString &b) {
        QStringList pa = a.split('x');
        QStringList pb = b.split('x');
        int pixA = pa[0].toInt() * pa[1].toInt();
        int pixB = pb[0].toInt() * pb[1].toInt();
        return pixA > pixB;
    });

    for (const QString &res : resolutions) {
        cameraResolutionComboBox->addItem(res);
    }

    // Try to select 1280x720 by default, otherwise first real resolution
    int idx = cameraResolutionComboBox->findText("1280x720");
    if (idx < 0 && cameraResolutionComboBox->count() > 1) {
        idx = 1;
    }
    if (idx >= 0) {
        cameraResolutionComboBox->setCurrentIndex(idx);
    }

    cameraResolutionComboBox->blockSignals(false);
    populateFPS();
}

void SettingsWindow::populateFPS() {
    const QString previousSelection = cameraFPSComboBox->currentText();
    cameraFPSComboBox->clear();

    QString currentRes = cameraResolutionComboBox->currentText();
    if (currentRes == "Default" || !deviceCapabilities.contains(currentRes)) {
        cameraFPSComboBox->addItems({"24", "30", "60"});
        int preferredIdx = cameraFPSComboBox->findText(preferredFpsText);
        if (preferredIdx >= 0) {
            cameraFPSComboBox->setCurrentIndex(preferredIdx);
        } else {
            cameraFPSComboBox->setCurrentText("30");
        }
        return;
    }

    QList<double> fpsList = deviceCapabilities[currentRes];
    std::sort(fpsList.begin(), fpsList.end(), std::greater<double>());

    for (double fps : fpsList) {
        int iFps = static_cast<int>(fps);
        if (qAbs(fps - iFps) < 0.05) {
            cameraFPSComboBox->addItem(QString::number(iFps));
        } else {
            cameraFPSComboBox->addItem(QString::number(fps, 'f', 1));
        }
    }

    // Keep previous/preferred FPS when available, otherwise fall back to 30.
    int idx = cameraFPSComboBox->findText(previousSelection);
    if (idx < 0) {
        idx = cameraFPSComboBox->findText(preferredFpsText);
    }
    if (idx < 0) {
        idx = cameraFPSComboBox->findText("30");
    }
    if (idx >= 0) {
        cameraFPSComboBox->setCurrentIndex(idx);
    }
}

void SettingsWindow::onCameraDeviceChanged(int comboIndex) {
    if (comboIndex < 0)
        return;
#ifdef _WIN32
    int deviceIndex = resolveWindowsSelectedCameraIndex(cameraIndexComboBox);
#else
    int deviceIndex = cameraIndexComboBox->currentData().toInt();
#endif
    enumerateDevice(deviceIndex);
    onCameraResolutionChanged(cameraResolutionComboBox->currentIndex());
}

void SettingsWindow::onCameraResolutionChanged(int comboIndex) {
    Q_UNUSED(comboIndex);
    populateFPS();
    QString currentRes = cameraResolutionComboBox->currentText();
    bool yuvSupported = yuvResolutions.contains(currentRes);
    useYuvCheckBox->setEnabled(cameraOptionRadioButton->isChecked() && yuvSupported);
    if (!useYuvCheckBox->isEnabled())
        useYuvCheckBox->setChecked(false);
}
