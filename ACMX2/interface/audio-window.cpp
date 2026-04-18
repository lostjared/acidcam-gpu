#include "audio-window.hpp"
#include <QProcess>
#include <QRegularExpression>
#include <QSettings>

AudioSettings::AudioSettings(QWidget *parent)
    : QDialog(parent) {
    setWindowTitle("Audio Settings");

    audioReactivityCheckBox = new QCheckBox("Enable Audio Reactivity", this);
    audioPassThroughCheckBox = new QCheckBox("Enable Audio Pass Through", this);
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

    QString style = "QMainWindow, QDialog { background-color: black; border: 3px solid red; }"
                    "* { color: red; font-weight: bold; } "
                    "QPushButton { border: 1px solid red; background-color: #110000; padding: 5px; }"
                    "QPushButton:hover { background-color: red; color: black; }";

    QSettings appSettings("LostSideDead");
    if (appSettings.value("useCustomStyle", true).toBool()) {
        setStyleSheet(style);
    }

    QLabel *inputDeviceLabel = new QLabel("Input Device:", this);
    inputDeviceComboBox = new QComboBox(this);

    QLabel *outputDeviceLabel = new QLabel("Output Device:", this);
    outputDeviceComboBox = new QComboBox(this);

    populateAudioDevices();

    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);

    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);

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

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);
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
