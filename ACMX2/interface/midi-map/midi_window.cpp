#include "midi_window.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QFont>
#include <fstream>
#include <sstream>

MidiMapWindow::MidiMapWindow(QWidget *parent)
    : QMainWindow(parent), midiIn(nullptr) {
    setupUi();
    applyStyleSheet();
    populateActions();
    updateTable();

    pollTimer = new QTimer(this);
    connect(pollTimer, &QTimer::timeout, this, &MidiMapWindow::pollMidi);

    try {
        midiIn = new RtMidiIn();
        midiIn->ignoreTypes(false, false, false);
        refreshDevices();
        setStatus("MIDI initialized — select a device");
    } catch (RtMidiError &e) {
        setStatus(QString("MIDI init failed: %1").arg(e.getMessage().c_str()));
    }
}

MidiMapWindow::~MidiMapWindow() {
    if (pollTimer->isActive())
        pollTimer->stop();
    delete midiIn;
}

void MidiMapWindow::setupUi() {
    setWindowTitle("ACMX2 MIDI Map Configuration");
    setMinimumSize(820, 640);

    auto *central = new QWidget(this);
    auto *mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(8);

    // --- Device section ---
    auto *deviceGroup = new QGroupBox("MIDI Device", this);
    auto *deviceLayout = new QHBoxLayout(deviceGroup);
    deviceCombo = new QComboBox(this);
    deviceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    refreshButton = new QPushButton("Refresh", this);
    deviceLayout->addWidget(deviceCombo);
    deviceLayout->addWidget(refreshButton);
    mainLayout->addWidget(deviceGroup);

    connect(refreshButton, &QPushButton::clicked, this, &MidiMapWindow::refreshDevices);
    connect(deviceCombo, QOverload<int>::of(&QComboBox::activated), this, &MidiMapWindow::openDevice);

    // --- MIDI monitor ---
    midiMonitorLabel = new QLabel("Last MIDI: —", this);
    midiMonitorLabel->setObjectName("midiMonitor");
    mainLayout->addWidget(midiMonitorLabel);

    // --- Mapping table ---
    auto *tableGroup = new QGroupBox("Action Mappings", this);
    auto *tableLayout = new QVBoxLayout(tableGroup);
    table = new QTableWidget(this);
    table->setColumnCount(5);
    table->setHorizontalHeaderLabels({"Action", "Description", "Key Codes", "MIDI Bytes", "Status"});
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    tableLayout->addWidget(table);
    mainLayout->addWidget(tableGroup, 1);

    // --- Buttons ---
    auto *btnLayout = new QHBoxLayout();
    captureButton = new QPushButton("Capture Selected", this);
    clearButton = new QPushButton("Clear Selected", this);
    btnLayout->addWidget(captureButton);
    btnLayout->addWidget(clearButton);
    btnLayout->addStretch();
    mainLayout->addLayout(btnLayout);

    connect(captureButton, &QPushButton::clicked, this, &MidiMapWindow::captureMapping);
    connect(clearButton, &QPushButton::clicked, this, &MidiMapWindow::clearMapping);

    // --- File section ---
    auto *fileGroup = new QGroupBox("Configuration File", this);
    auto *fileLayout = new QHBoxLayout(fileGroup);
    fileEdit = new QLineEdit("midi.midi_cfg", this);
    saveButton = new QPushButton("Save", this);
    loadButton = new QPushButton("Load", this);
    fileLayout->addWidget(fileEdit, 1);
    fileLayout->addWidget(loadButton);
    fileLayout->addWidget(saveButton);
    mainLayout->addWidget(fileGroup);

    connect(saveButton, &QPushButton::clicked, this, &MidiMapWindow::saveConfig);
    connect(loadButton, &QPushButton::clicked, this, &MidiMapWindow::loadConfig);

    setCentralWidget(central);
    statusBar()->showMessage("Ready");
}

void MidiMapWindow::applyStyleSheet() {
    setStyleSheet(
        "QMainWindow { background-color: rgb(0, 0, 0); }"
        "* { color: white; font-family: 'Courier New', Courier, monospace; font-size: 13px; }"
        "QGroupBox { border: 1px solid #444444; border-radius: 4px; margin-top: 8px; padding-top: 14px; font-weight: bold; color: white; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: white; }"
        "QPushButton { border: 1px solid #555555; background-color: #1a1a1a; padding: 6px 14px; border-radius: 3px; font-weight: bold; color: white; }"
        "QPushButton:hover { background-color: #333333; color: white; }"
        "QPushButton:pressed { background-color: #555555; color: white; }"
        "QPushButton:disabled { border: 1px solid #2a2a2a; color: #444444; background-color: #0d0d0d; }"
        "QComboBox { border: 1px solid #555555; background-color: #1a1a1a; padding: 4px; color: white; selection-background-color: #444444; }"
        "QComboBox::drop-down { border-left: 1px solid #555555; }"
        "QComboBox QAbstractItemView { background-color: #1a1a1a; border: 1px solid #555555; selection-background-color: #444444; color: white; }"
        "QLineEdit { border: 1px solid #555555; background-color: #1a1a1a; padding: 4px; color: white; selection-background-color: #444444; }"
        "QTableWidget { background-color: black; color: white; font-size: 13px; gridline-color: #333333; border: 1px solid #444444; }"
        "QTableWidget::item { padding: 4px; }"
        "QTableWidget::item:selected { background-color: #444444; color: white; }"
        "QHeaderView::section { background-color: #1a1a1a; border: 1px solid #444444; padding: 4px; font-weight: bold; color: white; }"
        "QStatusBar { background-color: black; border-top: 1px solid #444444; color: lime; font-size: 13px; }"
        "QLabel#midiMonitor { background-color: black; border: 1px solid #444444; padding: 6px; font-size: 14px; color: lime; }"
        "QScrollBar:vertical { background: #0d0d0d; width: 12px; border: 1px solid #333333; }"
        "QScrollBar::handle:vertical { background: #555555; min-height: 20px; border-radius: 3px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        "QMessageBox { background-color: #1a1a1a; color: white; }"
        "QMessageBox QLabel { color: white; }"
        "QMessageBox QPushButton { min-width: 70px; }"
    );
}

void MidiMapWindow::populateActions() {
    // Knob mappings (paired key codes)
    mappings.push_back({"Left/Right Knob", "Filter index left/right", 262, 263, false, 0, 0, 0});
    mappings.push_back({"Up/Down Knob", "Shader index down/up", 264, 265, false, 0, 0, 0});
    mappings.push_back({"Time Fwd/Back Knob", "Time forward/backward", 500, 501, false, 0, 0, 0});

    // Key mappings (single key, key2 = 0)
    mappings.push_back({"Left", "Filter index left", 263, 0, false, 0, 0, 0});
    mappings.push_back({"Right", "Filter index right", 262, 0, false, 0, 0, 0});
    mappings.push_back({"Up", "Shader index up", 265, 0, false, 0, 0, 0});
    mappings.push_back({"Down", "Shader index down", 264, 0, false, 0, 0, 0});
    mappings.push_back({"Space", "Toggle acid cam filters", 32, 0, false, 0, 0, 0});
    mappings.push_back({"Time Forward", "Move time forward", 500, 0, false, 0, 0, 0});
    mappings.push_back({"Time Backward", "Move time backward", 501, 0, false, 0, 0, 0});
    mappings.push_back({"Time Pause", "Pause time", 502, 0, false, 0, 0, 0});
    mappings.push_back({"Time On/Off", "Toggle time on/off", 503, 0, false, 0, 0, 0});
    mappings.push_back({"Plus/Equal", "Increase blend %", 61, 0, false, 0, 0, 0});
    mappings.push_back({"Minus", "Decrease blend %", 45, 0, false, 0, 0, 0});
    mappings.push_back({"H", "Shuffle playlist", 72, 0, false, 0, 0, 0});
    mappings.push_back({"L", "Enable/disable playlist", 76, 0, false, 0, 0, 0});
    mappings.push_back({"N", "Set index to end", 78, 0, false, 0, 0, 0});
    mappings.push_back({"P", "Reset index to zero", 80, 0, false, 0, 0, 0});
    mappings.push_back({"Page Down", "Restore position index", 267, 0, false, 0, 0, 0});
    mappings.push_back({"Page Up", "Store index position", 266, 0, false, 0, 0, 0});
    mappings.push_back({"Comma", "Color map decrease", 44, 0, false, 0, 0, 0});
    mappings.push_back({"Period", "Color map increase", 46, 0, false, 0, 0, 0});
    mappings.push_back({"Slash", "Random shader toggle", 47, 0, false, 0, 0, 0});
    mappings.push_back({"W", "Camera pitch up", 87, 0, false, 0, 0, 0});
    mappings.push_back({"S", "Camera pitch down", 83, 0, false, 0, 0, 0});
    mappings.push_back({"A", "Camera yaw left", 65, 0, false, 0, 0, 0});
    mappings.push_back({"D", "Camera yaw right", 68, 0, false, 0, 0, 0});
    mappings.push_back({"B", "Increase movement speed", 66, 0, false, 0, 0, 0});
}

void MidiMapWindow::updateTable() {
    table->setRowCount(static_cast<int>(mappings.size()));
    for (int i = 0; i < static_cast<int>(mappings.size()); ++i) {
        const auto &m = mappings[i];
        table->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(m.actionName)));
        table->setItem(i, 1, new QTableWidgetItem(QString::fromStdString(m.description)));

        QString keys = (m.key2 != 0)
            ? QString("%1:%2").arg(m.key1).arg(m.key2)
            : QString("%1:0").arg(m.key1);
        table->setItem(i, 2, new QTableWidgetItem(keys));

        if (m.captured) {
            if (m.key2 != 0) {
                // Knob: only byte0:byte1 matter for matching
                table->setItem(i, 3, new QTableWidgetItem(
                    QString("CC %1 %2 (knob)").arg(m.byte0).arg(m.byte1)));
                auto *statusItem = new QTableWidgetItem("Knob mapped");
                statusItem->setForeground(QBrush(QColor("#00ff00")));
                table->setItem(i, 4, statusItem);
            } else {
                table->setItem(i, 3, new QTableWidgetItem(
                    QString("%1 %2 %3").arg(m.byte0).arg(m.byte1).arg(m.byte2)));
                auto *statusItem = new QTableWidgetItem("Mapped");
                statusItem->setForeground(QBrush(QColor("#00ff00")));
                table->setItem(i, 4, statusItem);
            }
        } else {
            table->setItem(i, 3, new QTableWidgetItem("—"));
            auto *statusItem = new QTableWidgetItem("Not mapped");
            statusItem->setForeground(QBrush(QColor("#666666")));
            table->setItem(i, 4, statusItem);
        }
    }
}

void MidiMapWindow::refreshDevices() {
    deviceCombo->clear();
    if (!midiIn) return;

    unsigned int ports = midiIn->getPortCount();
    if (ports == 0) {
        deviceCombo->addItem("No MIDI devices found");
        captureButton->setEnabled(false);
        setStatus("No MIDI devices detected");
        return;
    }

    for (unsigned int i = 0; i < ports; ++i) {
        deviceCombo->addItem(QString::fromStdString(midiIn->getPortName(i)));
    }
    captureButton->setEnabled(true);
    setStatus(QString("%1 MIDI device(s) found").arg(ports));
}

void MidiMapWindow::openDevice(int index) {
    if (!midiIn || index < 0) return;

    if (deviceOpen) {
        midiIn->closePort();
        pollTimer->stop();
        deviceOpen = false;
    }

    try {
        midiIn->openPort(static_cast<unsigned int>(index));
        deviceOpen = true;
        pollTimer->start(10);
        setStatus(QString("Opened: %1").arg(deviceCombo->currentText()));
    } catch (RtMidiError &e) {
        QMessageBox::critical(this, "Error",
            QString("Could not open port: %1").arg(e.getMessage().c_str()));
        setStatus("Failed to open device");
    }
}

void MidiMapWindow::captureMapping() {
    if (!deviceOpen) {
        QMessageBox::warning(this, "No Device",
            "Please select and open a MIDI device first.\n"
            "Click a device in the dropdown to connect.");
        return;
    }

    int row = table->currentRow();
    if (row < 0 || row >= static_cast<int>(mappings.size())) {
        QMessageBox::information(this, "Select Action",
            "Select a row in the table, then click Capture.\n"
            "Move a knob or press a button on your MIDI controller.");
        return;
    }

    capturing = true;
    captureRow = row;
    setStatus(QString("Waiting for MIDI input for: %1 — move a knob or press a button...")
        .arg(QString::fromStdString(mappings[row].actionName)));
    captureButton->setText("Listening...");
    captureButton->setEnabled(false);
}

void MidiMapWindow::clearMapping() {
    int row = table->currentRow();
    if (row < 0 || row >= static_cast<int>(mappings.size())) {
        QMessageBox::information(this, "Select Action",
            "Select a row in the table to clear its mapping.");
        return;
    }

    mappings[row].captured = false;
    mappings[row].byte0 = 0;
    mappings[row].byte1 = 0;
    mappings[row].byte2 = 0;
    updateTable();
    table->selectRow(row);
    setStatus(QString("Cleared mapping for: %1").arg(QString::fromStdString(mappings[row].actionName)));
}

void MidiMapWindow::pollMidi() {
    if (!midiIn || !deviceOpen) return;

    std::vector<unsigned char> message;
    midiIn->getMessage(&message);

    if (message.size() < 3) return;

    midiMonitorLabel->setText(QString("Last MIDI: [%1 %2 %3]  (ch %4)")
        .arg(message[0]).arg(message[1]).arg(message[2])
        .arg((message[0] & 0x0F) + 1));

    if (capturing && captureRow >= 0 && captureRow < static_cast<int>(mappings.size())) {
        mappings[captureRow].byte0 = message[0];
        mappings[captureRow].byte1 = message[1];
        // For knob actions (key2 != 0), byte2 is ignored at runtime —
        // acmx2 uses the live value to determine direction (>64 = key1, <=64 = key2).
        // Store 0 to make it clear only byte0:byte1 matter for matching.
        mappings[captureRow].byte2 = (mappings[captureRow].key2 != 0) ? 0 : message[2];
        mappings[captureRow].captured = true;

        updateTable();
        table->selectRow(captureRow);

        QString statusMsg;
        if (mappings[captureRow].key2 != 0) {
            statusMsg = QString("Captured knob CC [%1 %2] for: %3 — value >64 = %4, <=64 = %5")
                .arg(message[0]).arg(message[1])
                .arg(QString::fromStdString(mappings[captureRow].actionName))
                .arg(QString::fromStdString(mappings[captureRow].actionName).split('/').first().trimmed())
                .arg(QString::fromStdString(mappings[captureRow].actionName).split('/').last().trimmed());
        } else {
            statusMsg = QString("Captured [%1 %2 %3] for: %4")
                .arg(message[0]).arg(message[1]).arg(message[2])
                .arg(QString::fromStdString(mappings[captureRow].actionName));
        }
        setStatus(statusMsg);

        capturing = false;
        captureRow = -1;
        captureButton->setText("Capture Selected");
        captureButton->setEnabled(true);
    }
}

void MidiMapWindow::saveConfig() {
    QString fileName = QFileDialog::getSaveFileName(this, "Save MIDI Config",
        fileEdit->text(), "MIDI Config (*.midi_cfg);;All Files (*)");
    if (fileName.isEmpty()) return;

    std::ofstream file(fileName.toStdString());
    if (!file.is_open()) {
        QMessageBox::critical(this, "Error", "Could not open file for writing.");
        return;
    }

    int count = 0;
    for (const auto &m : mappings) {
        if (!m.captured) continue;
        file << m.key1 << ":" << m.key2 << " {"
             << static_cast<int>(m.byte0) << " "
             << static_cast<int>(m.byte1) << " "
             << static_cast<int>(m.byte2) << "}\n";
        ++count;
    }
    file.close();
    fileEdit->setText(fileName);
    setStatus(QString("Saved %1 mapping(s) to %2").arg(count).arg(fileName));
}

void MidiMapWindow::loadConfig() {
    QString fileName = QFileDialog::getOpenFileName(this, "Load MIDI Config",
        fileEdit->text(), "MIDI Config (*.midi_cfg);;All Files (*)");
    if (fileName.isEmpty()) return;

    std::ifstream file(fileName.toStdString());
    if (!file.is_open()) {
        QMessageBox::critical(this, "Error", "Could not open file for reading.");
        return;
    }

    // Clear existing captures
    for (auto &m : mappings) {
        m.captured = false;
        m.byte0 = 0;
        m.byte1 = 0;
        m.byte2 = 0;
    }

    std::string line;
    int loaded = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string keyPair;
        if (!(iss >> keyPair)) continue;

        auto colonPos = keyPair.find(':');
        if (colonPos == std::string::npos) continue;

        int k1 = std::stoi(keyPair.substr(0, colonPos));
        int k2 = std::stoi(keyPair.substr(colonPos + 1));

        char brace;
        if (!(iss >> brace) || brace != '{') continue;

        int b0, b1, b2;
        if (!(iss >> b0 >> b1 >> b2)) continue;

        for (auto &m : mappings) {
            if (m.key1 == k1 && m.key2 == k2) {
                m.byte0 = static_cast<unsigned char>(b0);
                m.byte1 = static_cast<unsigned char>(b1);
                m.byte2 = static_cast<unsigned char>(b2);
                m.captured = true;
                ++loaded;
                break;
            }
        }
    }
    file.close();

    updateTable();
    fileEdit->setText(fileName);
    setStatus(QString("Loaded %1 mapping(s) from %2").arg(loaded).arg(fileName));
}

void MidiMapWindow::setStatus(const QString &msg) {
    statusBar()->showMessage(msg);
}
