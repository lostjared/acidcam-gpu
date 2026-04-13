#include "midi-settings.hpp"
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QProcess>
#include <QRegularExpression>
#include <QSettings>

MidiSettings::MidiSettings(const QString &executablePath, QWidget *parent)
    : QDialog(parent), execPath(executablePath) {
    setWindowTitle("MIDI Settings");
    setMinimumWidth(500);

    QString style = "QMainWindow, QDialog { background-color: black; border: 3px solid red; }"
                    "* { color: red; font-weight: bold; } "
                    "QPushButton { border: 1px solid red; background-color: #110000; padding: 5px; }"
                    "QPushButton:hover { background-color: red; color: black; }";

    QSettings appSettings("LostSideDead");
    if (appSettings.value("useCustomStyle", true).toBool()) {
        setStyleSheet(style);
    }

    auto *mainLayout = new QVBoxLayout(this);

    // --- Enable MIDI ---
    enableCheckBox = new QCheckBox("Enable MIDI Input", this);
    mainLayout->addWidget(enableCheckBox);

    // --- Config file ---
    auto *configGroup = new QGroupBox("MIDI Configuration File", this);
    auto *configLayout = new QHBoxLayout(configGroup);
    configFileEdit = new QLineEdit(this);
    configFileEdit->setPlaceholderText("Select a .midi_cfg file...");
    browseButton = new QPushButton("Browse...", this);
    configLayout->addWidget(configFileEdit, 1);
    configLayout->addWidget(browseButton);
    mainLayout->addWidget(configGroup);

    connect(browseButton, &QPushButton::clicked, this, &MidiSettings::browseConfigFile);

    // --- Launch midi-map tool ---
    auto *toolGroup = new QGroupBox("MIDI Map Tool", this);
    auto *toolLayout = new QHBoxLayout(toolGroup);
    auto *toolLabel = new QLabel("Create/edit MIDI mappings:", this);
    launchMapToolButton = new QPushButton("Launch midi-map", this);
    toolLayout->addWidget(toolLabel, 1);
    toolLayout->addWidget(launchMapToolButton);
    mainLayout->addWidget(toolGroup);

    connect(launchMapToolButton, &QPushButton::clicked, this, &MidiSettings::launchMidiMapTool);

    // --- Device selection ---
    auto *deviceGroup = new QGroupBox("MIDI Input Device", this);
    auto *deviceLayout = new QHBoxLayout(deviceGroup);
    deviceComboBox = new QComboBox(this);
    deviceComboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    refreshButton = new QPushButton("Refresh", this);
    deviceLayout->addWidget(deviceComboBox, 1);
    deviceLayout->addWidget(refreshButton);
    mainLayout->addWidget(deviceGroup);

    connect(refreshButton, &QPushButton::clicked, this, &MidiSettings::refreshDevices);

    // --- OK / Cancel ---
    auto *buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);
    mainLayout->addLayout(buttonLayout);

    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);

    // Load saved settings
    configFileEdit->setText(appSettings.value("midiConfigFile", "").toString());
    enableCheckBox->setChecked(appSettings.value("midiEnabled", false).toBool());

    populateMidiDevices();

    int savedDevice = appSettings.value("midiDevice", 0).toInt();
    if (savedDevice >= 0 && savedDevice < deviceComboBox->count()) {
        deviceComboBox->setCurrentIndex(savedDevice);
    }
}

bool MidiSettings::isMidiEnabled() const {
    return enableCheckBox->isChecked();
}

QString MidiSettings::getMidiConfigFile() const {
    return configFileEdit->text();
}

int MidiSettings::getMidiDeviceIndex() const {
    if (deviceComboBox->count() == 0)
        return -1;
    return deviceComboBox->currentIndex();
}

void MidiSettings::browseConfigFile() {
    QString fileName = QFileDialog::getOpenFileName(
        this, "Select MIDI Config File", configFileEdit->text(),
        "MIDI Config (*.midi_cfg);;All Files (*)");
    if (!fileName.isEmpty()) {
        configFileEdit->setText(fileName);
    }
}

void MidiSettings::launchMidiMapTool() {
    if (!QProcess::startDetached("midi-map", {})) {
        QMessageBox::warning(this, "Launch Failed",
            "Could not launch midi-map.\n"
            "Make sure it is installed and available in your system PATH.");
    }
}

void MidiSettings::refreshDevices() {
    populateMidiDevices();
}

void MidiSettings::populateMidiDevices() {
    deviceComboBox->clear();

    QProcess proc;
    proc.start(execPath, {"--list-midi"});
    if (!proc.waitForFinished(5000)) {
        deviceComboBox->addItem("(could not query devices)");
        return;
    }

    QString output = proc.readAllStandardOutput();
    if (output.isEmpty()) {
        output = proc.readAllStandardError();
    }

    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    QRegularExpression re(R"(^\s*(\d+):\s*(.+)$)");

    bool found = false;
    for (const QString &line : lines) {
        QRegularExpressionMatch match = re.match(line);
        if (match.hasMatch()) {
            QString idx = match.captured(1);
            QString name = match.captured(2).trimmed();
            deviceComboBox->addItem(QString("%1: %2").arg(idx, name));
            found = true;
        }
    }

    if (!found) {
        deviceComboBox->addItem("No MIDI devices found");
    }
}
