#include "audio-window.hpp"
#include "audio-playlist.hpp"
#include "custom_style.hpp"
#include <QFileDialog>
#include <QFileInfo>
#include <QProcess>
#include <QRegularExpression>
#include <QSettings>

AudioSettings::AudioSettings(QWidget *parent)
    : QDialog(parent) {
    setWindowTitle("Audio Settings");

    audioReactivityCheckBox = new QCheckBox("Enable Audio Reactivity", this);
    audioPassThroughCheckBox =
        new QCheckBox("Enable Audio Pass Through / File Playback", this);
    audioPassThroughCheckBox->setToolTip(
        "Play live microphone input or the selected audio source through the output device.");
    recordAudioCheckBox = new QCheckBox("Record Audio to File", this);

    QLabel *recordVolumeLabel = new QLabel("Recording Volume:", this);
    recordVolumeSlider = new QSlider(Qt::Horizontal, this);
    recordVolumeSlider->setRange(0, 200);
    recordVolumeSlider->setValue(100);
    QLabel *recordVolumeValueLabel = new QLabel("100%", this);
    connect(recordVolumeSlider, &QSlider::valueChanged, this, [recordVolumeValueLabel](int value) {
        recordVolumeValueLabel->setText(QString::number(value) + "%");
    });

    QLabel *channelLabel = new QLabel("Number of Channels:", this);
    channelSpinBox = new QSpinBox(this);
    channelSpinBox->setRange(1, 32);
    channelSpinBox->setValue(2);

    QLabel *sensitivityLabel = new QLabel("Sensitivity:", this);
    sensitivitySlider = new QSlider(Qt::Horizontal, this);
    sensitivitySlider->setRange(1, 50);
    sensitivitySlider->setValue(10);

    QLabel *sensitivityValueLabel = new QLabel("1.0", this);
    connect(sensitivitySlider, &QSlider::valueChanged, this, [this, sensitivityValueLabel](int value) {
        double floatValue = value / 10.0;
        sensitivityValueLabel->setText(QString::number(floatValue, 'f', 1));
    });

    acmx2::applyCustomStyleIfEnabled(this);

    QLabel *inputDeviceLabel = new QLabel("Input Device:", this);
    inputDeviceComboBox = new QComboBox(this);

    QLabel *outputDeviceLabel = new QLabel("Output Device:", this);
    outputDeviceComboBox = new QComboBox(this);

    populateAudioDevices();
    outputDeviceComboBox->setEnabled(false);

    connect(audioPassThroughCheckBox, &QCheckBox::toggled, this,
            [this](bool checked) {
                outputDeviceComboBox->setEnabled(checked);
            });

    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);

    audioFileCheckBox = new QCheckBox("Use Audio File for Reactivity (instead of mic)", this);
    audioFileLineEdit = new QLineEdit(this);
    audioFileLineEdit->setReadOnly(true);
    audioFileLineEdit->setEnabled(false);
    audioFileBrowseButton = new QPushButton("Browse", this);
    audioFileBrowseButton->setEnabled(false);
    audioPlaylistCheckBox = new QCheckBox("Use M3U Audio Playlist", this);
    audioPlaylistCheckBox->setToolTip(
        "Use the tracks in an M3U playlist instead of the selected audio file.");
    audioPlaylistLineEdit = new QLineEdit(this);
    audioPlaylistLineEdit->setReadOnly(true);
    audioPlaylistLineEdit->setEnabled(false);
    audioPlaylistBrowseButton = new QPushButton("Browse", this);
    audioPlaylistBrowseButton->setEnabled(false);
    audioPlaylistEditButton = new QPushButton("Create / Edit...", this);
    audioTruncCheckBox = new QCheckBox("Stop video when audio source completes", this);
    audioTruncCheckBox->setEnabled(false);
    audioRepeatCheckBox = new QCheckBox("Repeat", this);
    audioRepeatCheckBox->setToolTip(
        "Restart the audio file, or the full playlist, when it reaches the end.");
    audioRepeatCheckBox->setEnabled(false);
    audioBuffersCheckBox = new QCheckBox("Enable Audio Spectrum History Buffers", this);
    audioBuffersSpinBox = new QSpinBox(this);
    audioBuffersSpinBox->setRange(1, 512);
    audioBuffersSpinBox->setValue(8);
    audioBuffersSpinBox->setEnabled(false);

    audioWarmRateSpinBox = new QDoubleSpinBox(this);
    audioWarmRateSpinBox->setRange(0.0, 10.0);
    audioWarmRateSpinBox->setSingleStep(0.05);
    audioWarmRateSpinBox->setDecimals(2);
    audioWarmRateSpinBox->setValue(0.5);
    audioWarmRateSpinBox->setToolTip("Audio startup warmup rate (1/sec). 0.5 is about a 2 second fade-in; 0 disables warmup.");

    connect(audioBuffersCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        audioBuffersSpinBox->setEnabled(checked);
    });

    connect(audioFileCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked)
            audioPlaylistCheckBox->setChecked(false);
        updateAudioSourceControls();
    });

    connect(audioTruncCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked)
            audioRepeatCheckBox->setChecked(false);
    });

    connect(audioPlaylistCheckBox, &QCheckBox::toggled, this,
            [this](bool checked) {
                if (checked)
                    audioFileCheckBox->setChecked(false);
                updateAudioSourceControls();
            });
    connect(audioRepeatCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked)
            audioTruncCheckBox->setChecked(false);
    });

    connect(audioFileBrowseButton, &QPushButton::clicked, this, [this]() {
        QSettings appSettings("LostSideDead");
        QString lastDir = appSettings.value("lastAudioFileDir", "").toString();
        QString fileName = QFileDialog::getOpenFileName(this, "Select Audio File", lastDir,
                                                        "Audio Files (*.wav *.mp3 *.flac *.aac *.ogg *.m4a *.wma *.mp4 *.mkv *.mov *.avi)");
        if (!fileName.isEmpty()) {
            appSettings.setValue("lastAudioFileDir", QFileInfo(fileName).absolutePath());
            audioFileLineEdit->setText(fileName);
        }
    });

    connect(audioPlaylistBrowseButton, &QPushButton::clicked, this, [this]() {
        QSettings appSettings("LostSideDead");
        QString lastDir = appSettings.value("lastAudioPlaylistDir", "").toString();
        QString fileName = QFileDialog::getOpenFileName(
            this, "Select M3U Audio Playlist", lastDir,
            "M3U Playlists (*.m3u *.m3u8)");
        if (!fileName.isEmpty()) {
            appSettings.setValue("lastAudioPlaylistDir",
                                 QFileInfo(fileName).absolutePath());
            audioPlaylistLineEdit->setText(fileName);
        }
    });

    connect(audioPlaylistEditButton, &QPushButton::clicked, this, [this]() {
        AudioPlaylistDialog editor(audioPlaylistLineEdit->text(), this);
        if (editor.exec() != QDialog::Accepted || editor.playlistPath().isEmpty())
            return;
        audioPlaylistLineEdit->setText(editor.playlistPath());
        audioPlaylistCheckBox->setChecked(true);
    });

    connect(okButton, &QPushButton::clicked, this, [this]() {
        saveUiState();
        accept();
    });
    connect(cancelButton, &QPushButton::clicked, this, [this]() {
        saveUiState();
        reject();
    });

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(audioReactivityCheckBox);
    mainLayout->addWidget(audioPassThroughCheckBox);
    mainLayout->addWidget(recordAudioCheckBox);

    QHBoxLayout *recordVolumeLayout = new QHBoxLayout();
    recordVolumeLayout->addWidget(recordVolumeLabel);
    recordVolumeLayout->addWidget(recordVolumeSlider);
    recordVolumeLayout->addWidget(recordVolumeValueLabel);
    mainLayout->addLayout(recordVolumeLayout);

    QHBoxLayout *channelLayout = new QHBoxLayout();
    channelLayout->addWidget(channelLabel);
    channelLayout->addWidget(channelSpinBox);
    mainLayout->addLayout(channelLayout);

    QHBoxLayout *sensitivityLayout = new QHBoxLayout();
    sensitivityLayout->addWidget(sensitivityLabel);
    sensitivityLayout->addWidget(sensitivitySlider);
    sensitivityLayout->addWidget(sensitivityValueLabel);
    mainLayout->addLayout(sensitivityLayout);

    QHBoxLayout *inputDeviceLayout = new QHBoxLayout();
    inputDeviceLayout->addWidget(inputDeviceLabel);
    inputDeviceLayout->addWidget(inputDeviceComboBox);
    mainLayout->addLayout(inputDeviceLayout);

    QHBoxLayout *outputDeviceLayout = new QHBoxLayout();
    outputDeviceLayout->addWidget(outputDeviceLabel);
    outputDeviceLayout->addWidget(outputDeviceComboBox);
    mainLayout->addLayout(outputDeviceLayout);

    QHBoxLayout *audioFileLayout = new QHBoxLayout();
    audioFileLayout->addWidget(audioFileLineEdit);
    audioFileLayout->addWidget(audioFileBrowseButton);
    mainLayout->addWidget(audioFileCheckBox);
    mainLayout->addLayout(audioFileLayout);
    QHBoxLayout *audioPlaylistLayout = new QHBoxLayout();
    audioPlaylistLayout->addWidget(audioPlaylistLineEdit);
    audioPlaylistLayout->addWidget(audioPlaylistBrowseButton);
    audioPlaylistLayout->addWidget(audioPlaylistEditButton);
    mainLayout->addWidget(audioPlaylistCheckBox);
    mainLayout->addLayout(audioPlaylistLayout);
    mainLayout->addWidget(audioTruncCheckBox);
    mainLayout->addWidget(audioRepeatCheckBox);

    QHBoxLayout *audioBuffersLayout = new QHBoxLayout();
    audioBuffersLayout->addWidget(audioBuffersCheckBox);
    audioBuffersLayout->addWidget(new QLabel("Frames:", this));
    audioBuffersLayout->addWidget(audioBuffersSpinBox);
    audioBuffersLayout->addStretch();
    mainLayout->addLayout(audioBuffersLayout);

    QHBoxLayout *audioWarmLayout = new QHBoxLayout();
    audioWarmLayout->addWidget(new QLabel("Audio Warm Rate:", this));
    audioWarmLayout->addWidget(audioWarmRateSpinBox);
    audioWarmLayout->addWidget(new QLabel("(1/sec)", this));
    audioWarmLayout->addStretch();
    mainLayout->addLayout(audioWarmLayout);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);
    loadUiState();
}

void AudioSettings::loadUiState() {
    QSettings appSettings("LostSideDead", "acmx2");

    audioReactivityCheckBox->setChecked(appSettings.value("audio/enabled", false).toBool());
    audioPassThroughCheckBox->setChecked(appSettings.value("audio/passthrough", false).toBool());
    recordAudioCheckBox->setChecked(appSettings.value("audio/record", false).toBool());
    recordVolumeSlider->setValue(appSettings.value("audio/record_volume", 100).toInt());
    channelSpinBox->setValue(appSettings.value("audio/channels", 2).toInt());
    sensitivitySlider->setValue(appSettings.value("audio/sensitivity", 10).toInt());

    int inputId = appSettings.value("audio/input_device", -1).toInt();
    int outputId = appSettings.value("audio/output_device", -1).toInt();
    int inputIdx = inputDeviceComboBox->findData(inputId);
    int outputIdx = outputDeviceComboBox->findData(outputId);
    if (inputIdx >= 0) {
        inputDeviceComboBox->setCurrentIndex(inputIdx);
    }
    if (outputIdx >= 0) {
        outputDeviceComboBox->setCurrentIndex(outputIdx);
    }

    audioFileCheckBox->setChecked(appSettings.value("audio/file_enabled", false).toBool());
    audioFileLineEdit->setText(appSettings.value("audio/file_path", "").toString());
    audioPlaylistCheckBox->setChecked(
        appSettings.value("audio/playlist_enabled", false).toBool());
    audioPlaylistLineEdit->setText(
        appSettings.value("audio/playlist_path", "").toString());
    audioTruncCheckBox->setChecked(appSettings.value("audio/file_trunc", false).toBool());
    audioRepeatCheckBox->setChecked(appSettings.value("audio/file_repeat", false).toBool());
    audioBuffersCheckBox->setChecked(appSettings.value("audio/buffers_enabled", false).toBool());
    audioBuffersSpinBox->setValue(appSettings.value("audio/buffers_frames", 8).toInt());
    audioBuffersSpinBox->setEnabled(audioBuffersCheckBox->isChecked());
    audioWarmRateSpinBox->setValue(appSettings.value("audio/warm_rate", 0.5).toDouble());
}

void AudioSettings::saveUiState() {
    QSettings appSettings("LostSideDead", "acmx2");
    appSettings.setValue("audio/enabled", audioReactivityCheckBox->isChecked());
    appSettings.setValue("audio/passthrough", audioPassThroughCheckBox->isChecked());
    appSettings.setValue("audio/record", recordAudioCheckBox->isChecked());
    appSettings.setValue("audio/record_volume", recordVolumeSlider->value());
    appSettings.setValue("audio/channels", channelSpinBox->value());
    appSettings.setValue("audio/sensitivity", sensitivitySlider->value());
    appSettings.setValue("audio/input_device", inputDeviceComboBox->currentData().toInt());
    appSettings.setValue("audio/output_device", outputDeviceComboBox->currentData().toInt());
    appSettings.setValue("audio/file_enabled", audioFileCheckBox->isChecked());
    appSettings.setValue("audio/file_path", audioFileLineEdit->text());
    appSettings.setValue("audio/playlist_enabled",
                         audioPlaylistCheckBox->isChecked());
    appSettings.setValue("audio/playlist_path", audioPlaylistLineEdit->text());
    appSettings.setValue("audio/file_trunc", audioTruncCheckBox->isChecked());
    appSettings.setValue("audio/file_repeat", audioRepeatCheckBox->isChecked());
    appSettings.setValue("audio/buffers_enabled", audioBuffersCheckBox->isChecked());
    appSettings.setValue("audio/buffers_frames", audioBuffersSpinBox->value());
    appSettings.setValue("audio/warm_rate", audioWarmRateSpinBox->value());
}

void AudioSettings::updateAudioSourceControls() {
    const bool fileEnabled = audioFileCheckBox->isChecked();
    const bool playlistEnabled = audioPlaylistCheckBox->isChecked();
    const bool sourceEnabled = fileEnabled || playlistEnabled;

    audioFileLineEdit->setEnabled(fileEnabled);
    audioFileBrowseButton->setEnabled(fileEnabled);
    audioPlaylistLineEdit->setEnabled(playlistEnabled);
    audioPlaylistBrowseButton->setEnabled(playlistEnabled);
    audioTruncCheckBox->setEnabled(sourceEnabled);
    audioRepeatCheckBox->setEnabled(sourceEnabled);
    if (sourceEnabled) {
        audioPassThroughCheckBox->setChecked(true);
        outputDeviceComboBox->setEnabled(true);
    }

    // File and playlist audio replace microphone-specific input controls.
    // Pass-through and output selection remain available for playback.
    channelSpinBox->setEnabled(!sourceEnabled);
    inputDeviceComboBox->setEnabled(!sourceEnabled);
}

void AudioSettings::populateAudioDevices() {
    inputDeviceComboBox->addItem("Default", -1);
    outputDeviceComboBox->addItem("Default", -1);

    QProcess process;
    process.start("acmx2", QStringList() << "--list-devices");
    process.waitForFinished(5000);

    QString output = process.readAllStandardOutput();
    if (output.isEmpty() || process.exitCode() != 0) {
        return;
    }

    // Parse lines like:
    //   Device 132: EMEET SmartCam C950 4K [DEFAULT INPUT]
    //     Input channels: 2
    //     Output channels: 0
    QRegularExpression deviceRegex(
        R"(^\s*Device\s+(\d+):\s*(.+?)\s*$)");
    QRegularExpression inputChRegex(
        R"(^\s*Input channels:\s*(\d+))");
    QRegularExpression outputChRegex(
        R"(^\s*Output channels:\s*(\d+))");

    QStringList lines = output.split('\n');
    int currentId = -1;
    QString currentName;
    bool isDefaultInput = false;
    bool isDefaultOutput = false;

    auto flushDevice = [&]() {
        // called when we hit the next "Device" line or end of output
        // nothing to flush on the first call
    };

    int inputCh = 0;
    int outputCh = 0;

    for (int i = 0; i < lines.size(); ++i) {
        const QString &line = lines[i];
        QRegularExpressionMatch devMatch = deviceRegex.match(line);

        if (devMatch.hasMatch()) {
            // Flush previous device
            if (currentId >= 0) {
                if (inputCh > 0) {
                    QString label = currentName + " (" + QString::number(inputCh) + " ch)";
                    if (isDefaultInput)
                        label += " [Default]";
                    inputDeviceComboBox->addItem(label, currentId);
                }
                if (outputCh > 0) {
                    QString label = currentName + " (" + QString::number(outputCh) + " ch)";
                    if (isDefaultOutput)
                        label += " [Default]";
                    outputDeviceComboBox->addItem(label, currentId);
                }
            }

            currentId = devMatch.captured(1).toInt();
            currentName = devMatch.captured(2).trimmed();
            isDefaultInput = currentName.contains("[DEFAULT INPUT]");
            isDefaultOutput = currentName.contains("[DEFAULT OUTPUT]");
            // Strip the tags from the display name
            currentName.remove("[DEFAULT INPUT]");
            currentName.remove("[DEFAULT OUTPUT]");
            currentName = currentName.trimmed();
            inputCh = 0;
            outputCh = 0;
            continue;
        }

        QRegularExpressionMatch inMatch = inputChRegex.match(line);
        if (inMatch.hasMatch()) {
            inputCh = inMatch.captured(1).toInt();
            continue;
        }

        QRegularExpressionMatch outMatch = outputChRegex.match(line);
        if (outMatch.hasMatch()) {
            outputCh = outMatch.captured(1).toInt();
            continue;
        }
    }

    // Flush last device
    if (currentId >= 0) {
        if (inputCh > 0) {
            QString label = currentName + " (" + QString::number(inputCh) + " ch)";
            if (isDefaultInput)
                label += " [Default]";
            inputDeviceComboBox->addItem(label, currentId);
        }
        if (outputCh > 0) {
            QString label = currentName + " (" + QString::number(outputCh) + " ch)";
            if (isDefaultOutput)
                label += " [Default]";
            outputDeviceComboBox->addItem(label, currentId);
        }
    }

    inputDeviceComboBox->setCurrentIndex(0);
    outputDeviceComboBox->setCurrentIndex(0);
}

bool AudioSettings::isAudioReactivityEnabled() const {
    return audioReactivityCheckBox->isChecked();
}

bool AudioSettings::isAudioPassThroughEnabled() const {
    return audioPassThroughCheckBox->isChecked();
}

bool AudioSettings::isRecordAudioEnabled() const {
    return recordAudioCheckBox->isChecked();
}

double AudioSettings::getRecordVolume() const {
    return recordVolumeSlider->value() / 100.0;
}

int AudioSettings::getNumberOfChannels() const {
    return channelSpinBox->value();
}

double AudioSettings::getSensitivity() const {
    return sensitivitySlider->value() / 10.0;
}

int AudioSettings::getInputDeviceIndex() const {
    return inputDeviceComboBox->currentData().toInt();
}

int AudioSettings::getOutputDeviceIndex() const {
    return outputDeviceComboBox->currentData().toInt();
}

bool AudioSettings::isAudioFileEnabled() const {
    return (audioFileCheckBox->isChecked() &&
            !audioFileLineEdit->text().isEmpty()) ||
           (audioPlaylistCheckBox->isChecked() &&
            !audioPlaylistLineEdit->text().isEmpty());
}

QString AudioSettings::getAudioFilePath() const {
    return isAudioPlaylistEnabled() ? audioPlaylistLineEdit->text()
                                    : audioFileLineEdit->text();
}

bool AudioSettings::isAudioPlaylistEnabled() const {
    return audioPlaylistCheckBox->isChecked() &&
           !audioPlaylistLineEdit->text().isEmpty();
}

QString AudioSettings::getAudioPlaylistPath() const {
    return audioPlaylistLineEdit->text();
}

bool AudioSettings::isAudioTruncEnabled() const {
    return audioTruncCheckBox->isChecked();
}

bool AudioSettings::isAudioRepeatEnabled() const {
    return audioRepeatCheckBox->isChecked();
}

bool AudioSettings::isAudioBuffersEnabled() const {
    return audioBuffersCheckBox->isChecked();
}

int AudioSettings::getAudioBufferFrames() const {
    return audioBuffersSpinBox->value();
}

double AudioSettings::getAudioWarmRate() const {
    return audioWarmRateSpinBox->value();
}
