#include "midi_window.hpp"
#include <QFileDialog>
#include <QFont>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QVBoxLayout>
#include <algorithm>
#include <fstream>
#include <sstream>

MidiMapWindow::MidiMapWindow(QWidget *parent)
    : QMainWindow(parent), midiIn(nullptr) {
    setupUi();
    applyStyleSheet();
    populateActions();
    profile_mappings[0] = mappings;
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

    // --- Target application ---
    auto *target_group = new QGroupBox("Target Application", this);
    auto *target_layout = new QHBoxLayout(target_group);
    target_combo = new QComboBox(this);
    target_combo->addItem("ACMX2");
    target_combo->addItem("ACMXVK");
    target_combo->setToolTip(
        "Selects target-specific action names while preserving the shared "
        ".midi_cfg file format.");
    target_layout->addWidget(target_combo);
    mainLayout->addWidget(target_group);

    connect(target_combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MidiMapWindow::change_target);

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
        "QMessageBox QPushButton { min-width: 70px; }");
}

void MidiMapWindow::populateActions() {
    mappings.clear();
    if (target_profile == MidiTargetProfile::Acmxvk) {
        // Knob mappings supported by ACMXVK.
        mappings.push_back({"Left/Right Knob", "GPU filter index left/right", 262, 263, false, 0, 0, 0});
        mappings.push_back({"Up/Down Knob", "Shader index or playlist node prev/next", 264, 265, false, 0, 0, 0});
        mappings.push_back({"Time Fwd/Back Knob", "Shader time step forward/backward", 500, 501, false, 0, 0, 0});
        mappings.push_back({"TimeSpeed Knob", "Shader time speed increase/decrease", 504, 505, false, 0, 0, 0});
        mappings.push_back({"Rotate X Axis Knob", "Rotate the 3D model forward/backward on X", 506, 507, false, 0, 0, 0});
        mappings.push_back({"Rotate Y Axis Knob", "Rotate the 3D model forward/backward on Y", 508, 509, false, 0, 0, 0});
        mappings.push_back({"Rotate Z Axis Knob", "Rotate the 3D model forward/backward on Z", 512, 513, false, 0, 0, 0});
        mappings.push_back({"Model Scale Knob", "Increase/decrease 3D model scale", 514, 515, false, 0, 0, 0});
        mappings.push_back({"Slider 1 Knob", "Shader uniform slider1", 600, 601, false, 0, 0, 0});
        mappings.push_back({"Slider 2 Knob", "Shader uniform slider2", 602, 603, false, 0, 0, 0});
        mappings.push_back({"Slider 3 Knob", "Shader uniform slider3", 604, 605, false, 0, 0, 0});
        mappings.push_back({"Slider 4 Knob", "Shader uniform slider4", 606, 607, false, 0, 0, 0});

        // ACMXVK key/action mappings.
        mappings.push_back({"Left", "Previous CUDA filter", 263, 0, false, 0, 0, 0});
        mappings.push_back({"Right", "Next CUDA filter", 262, 0, false, 0, 0, 0});
        mappings.push_back({"Up", "Previous shader or playlist node", 265, 0, false, 0, 0, 0});
        mappings.push_back({"Down", "Next shader or playlist node", 264, 0, false, 0, 0, 0});
        mappings.push_back({"Space", "Toggle shader bypass", 32, 0, false, 0, 0, 0});
        mappings.push_back({"Time Forward", "Step shader time forward (U)", 500, 0, false, 0, 0, 0});
        mappings.push_back({"Time Backward", "Step shader time backward (I)", 501, 0, false, 0, 0, 0});
        mappings.push_back({"Page Up", "Increase shader time speed", 266, 0, false, 0, 0, 0});
        mappings.push_back({"Page Down", "Decrease shader time speed", 267, 0, false, 0, 0, 0});
        mappings.push_back({"P", "Toggle playlist mode or input pause", 80, 0, false, 0, 0, 0});
        mappings.push_back({"L", "Toggle rendering freeze", 76, 0, false, 0, 0, 0});
        mappings.push_back({"M", "Toggle configured multipass chain", 77, 0, false, 0, 0, 0});
        mappings.push_back({"J", "Toggle random autopilot", 74, 0, false, 0, 0, 0});
        mappings.push_back({"Y", "Toggle sequential autopilot", 89, 0, false, 0, 0, 0});
        mappings.push_back({"N", "Toggle random autopilot XFade", 78, 0, false, 0, 0, 0});
        mappings.push_back({"K", "Toggle shader lock", 75, 0, false, 0, 0, 0});
        mappings.push_back({"T", "Toggle normal shader time", 84, 0, false, 0, 0, 0});
        mappings.push_back({"Q", "Toggle audio-reactive shader time", 81, 0, false, 0, 0, 0});
        mappings.push_back({"Home", "Toggle audio delta-time scaling", 268, 0, false, 0, 0, 0});
        mappings.push_back({"End", "Toggle spectrum sensitivity scaling", 269, 0, false, 0, 0, 0});
        mappings.push_back({"Insert", "Increase audio sensitivity", 260, 0, false, 0, 0, 0});
        mappings.push_back({"Delete", "Decrease audio sensitivity", 261, 0, false, 0, 0, 0});
        mappings.push_back({"E", "Toggle configured watermark", 69, 0, false, 0, 0, 0});
        mappings.push_back({"F", "Toggle fullscreen", 70, 0, false, 0, 0, 0});
        mappings.push_back({"F9", "Toggle preview-only runtime HUD", 298, 0, false, 0, 0, 0});
        mappings.push_back({"Z", "Take PNG snapshot", 90, 0, false, 0, 0, 0});
        mappings.push_back({"4", "Take optional TIFF snapshot", 52, 0, false, 0, 0, 0});
        mappings.push_back({"5", "Take optional WebP snapshot", 53, 0, false, 0, 0, 0});
        mappings.push_back({"6", "Take raw RGBA snapshot", 54, 0, false, 0, 0, 0});
        mappings.push_back({"3", "Toggle 2D/3D rendering", 51, 0, false, 0, 0, 0});
        mappings.push_back({"C", "Toggle 3D wave effect", 67, 0, false, 0, 0, 0});
        mappings.push_back({"O", "Toggle 3D scale oscillation", 79, 0, false, 0, 0, 0});
        mappings.push_back({"V", "Toggle automatic 3D view rotation", 86, 0, false, 0, 0, 0});
        mappings.push_back({"X", "Reset 3D model view", 88, 0, false, 0, 0, 0});
        mappings.push_back({"Comma", "Decrease 3D rotation speed", 44, 0, false, 0, 0, 0});
        mappings.push_back({"Period", "Increase 3D rotation speed", 46, 0, false, 0, 0, 0});
        mappings.push_back({"Scale Down", "Decrease 3D model scale", 91, 0, false, 0, 0, 0});
        mappings.push_back({"Scale Up", "Increase 3D model scale", 93, 0, false, 0, 0, 0});
        mappings.push_back({"RotSpeed Up", "Increase manual 3D rotation speed", 510, 0, false, 0, 0, 0});
        mappings.push_back({"RotSpeed Down", "Decrease manual 3D rotation speed", 511, 0, false, 0, 0, 0});
        return;
    }

    // Knob mappings (paired key codes)
    mappings.push_back({"Left/Right Knob", "GPU filter index left/right", 262, 263, false, 0, 0, 0});
    mappings.push_back({"Up/Down Knob", "Shader index prev/next (or playlist node)", 264, 265, false, 0, 0, 0});
    mappings.push_back({"Time Fwd/Back Knob", "Time step forward/backward", 500, 501, false, 0, 0, 0});
    mappings.push_back({"TimeSpeed Knob", "Time speed increase/decrease", 504, 505, false, 0, 0, 0});
    mappings.push_back({"Rotate X Axis W/S Knob", "Camera pitch up/down (3D)", 506, 507, false, 0, 0, 0});
    mappings.push_back({"Rotate Y Axis A/D Knob", "Camera yaw right/left (3D)", 508, 509, false, 0, 0, 0});
    mappings.push_back({"Rotate Z Axis Knob", "Camera roll right/left (3D)", 512, 513, false, 0, 0, 0});
    mappings.push_back({"Model Scale Knob", "Model scale increase/decrease (3D)", 514, 515, false, 0, 0, 0});
    mappings.push_back({"Slider 1 Knob", "Shader uniform slider1 (0.0-1.0)", 600, 601, false, 0, 0, 0});
    mappings.push_back({"Slider 2 Knob", "Shader uniform slider2 (0.0-1.0)", 602, 603, false, 0, 0, 0});
    mappings.push_back({"Slider 3 Knob", "Shader uniform slider3 (0.0-1.0)", 604, 605, false, 0, 0, 0});
    mappings.push_back({"Slider 4 Knob", "Shader uniform slider4 (0.0-1.0)", 606, 607, false, 0, 0, 0});
    mappings.push_back({"RotSpeed Up", "Camera rotation speed increase", 510, 0, false, 0, 0, 0});
    mappings.push_back({"RotSpeed Down", "Camera rotation speed decrease", 511, 0, false, 0, 0, 0});

    // Key mappings (single key, key2 = 0)
    mappings.push_back({"Left", "GPU filter index left", 263, 0, false, 0, 0, 0});
    mappings.push_back({"Right", "GPU filter index right", 262, 0, false, 0, 0, 0});
    mappings.push_back({"Up", "Previous shader (or playlist node)", 265, 0, false, 0, 0, 0});
    mappings.push_back({"Down", "Next shader (or playlist node)", 264, 0, false, 0, 0, 0});
    mappings.push_back({"Space", "Toggle shader bypass", 32, 0, false, 0, 0, 0});
    mappings.push_back({"Time Forward", "Time step forward (U)", 500, 0, false, 0, 0, 0});
    mappings.push_back({"Time Backward", "Time step backward (I)", 501, 0, false, 0, 0, 0});
    mappings.push_back({"P", "Toggle playlist enable/disable", 80, 0, false, 0, 0, 0});
    mappings.push_back({"L", "Freeze frame (video/graphic mode)", 76, 0, false, 0, 0, 0});
    mappings.push_back({"M", "Toggle multi-pass shader", 77, 0, false, 0, 0, 0});
    mappings.push_back({"Z", "Take PNG snapshot", 90, 0, false, 0, 0, 0});
    mappings.push_back({"E", "Toggle watermark", 69, 0, false, 0, 0, 0});
    mappings.push_back({"V", "Toggle view rotation (3D)", 86, 0, false, 0, 0, 0});
    mappings.push_back({"O", "Toggle scale oscillation (3D)", 79, 0, false, 0, 0, 0});
    mappings.push_back({"C", "Toggle wave effect (3D)", 67, 0, false, 0, 0, 0});
    mappings.push_back({"X", "Reset camera distance (3D)", 88, 0, false, 0, 0, 0});
    mappings.push_back({"3", "Toggle 2D/3D mode", 51, 0, false, 0, 0, 0});
    mappings.push_back({"Plus/Equal", "Camera distance increase (3D)", 61, 0, false, 0, 0, 0});
    mappings.push_back({"Minus", "Camera distance decrease (3D)", 45, 0, false, 0, 0, 0});
    mappings.push_back({"N", "Toggle random autopilot crossfade", 78, 0, false, 0, 0, 0});
    mappings.push_back({"Page Up", "Time speed increase", 266, 0, false, 0, 0, 0});
    mappings.push_back({"Page Down", "Time speed decrease", 267, 0, false, 0, 0, 0});
    mappings.push_back({"W", "Camera pitch up (3D)", 87, 0, false, 0, 0, 0});
    mappings.push_back({"S", "Camera pitch down (3D)", 83, 0, false, 0, 0, 0});
    mappings.push_back({"A", "Camera yaw left (3D)", 65, 0, false, 0, 0, 0});
    mappings.push_back({"D", "Camera yaw right (3D)", 68, 0, false, 0, 0, 0});
    mappings.push_back({"1", "Camera movement speed increase (3D)", 49, 0, false, 0, 0, 0});
    mappings.push_back({"2", "Camera movement speed decrease (3D)", 50, 0, false, 0, 0, 0});
    mappings.push_back({"B", "(3D) unused / available", 66, 0, false, 0, 0, 0});
    mappings.push_back({"K", "Toggle shader lock", 75, 0, false, 0, 0, 0});
    mappings.push_back({"R", "Toggle random multipass mode", 82, 0, false, 0, 0, 0});
    mappings.push_back({"G", "Generate new random shader chain", 71, 0, false, 0, 0, 0});
    mappings.push_back({"H", "Generate long random chain (up to 10)", 72, 0, false, 0, 0, 0});
    mappings.push_back({"F", "Generate short random pair (2 shaders)", 70, 0, false, 0, 0, 0});
    mappings.push_back({"End", "Toggle spectrum sensitivity scaling", 269, 0, false, 0, 0, 0});
    mappings.push_back({"J", "Toggle autopilot (random)", 74, 0, false, 0, 0, 0});
    mappings.push_back({"Y", "Toggle sequential autopilot", 89, 0, false, 0, 0, 0});
    mappings.push_back({"4", "Take TIFF snapshot", 52, 0, false, 0, 0, 0});
    mappings.push_back({"5", "Take HDR snapshot", 53, 0, false, 0, 0, 0});
    mappings.push_back({"6", "Take RAW snapshot", 54, 0, false, 0, 0, 0});
    mappings.push_back({"T", "Toggle active time", 84, 0, false, 0, 0, 0});
    mappings.push_back({"Q", "Toggle audio time", 81, 0, false, 0, 0, 0});
    mappings.push_back({"Home", "Toggle audio delta", 268, 0, false, 0, 0, 0});
    mappings.push_back({"Insert", "Audio sensitivity increase", 260, 0, false, 0, 0, 0});
    mappings.push_back({"Delete", "Audio sensitivity decrease", 261, 0, false, 0, 0, 0});
    mappings.push_back({"F9", "Toggle HUD overlay visibility", 298, 0, false, 0, 0, 0});
    mappings.push_back({"Left Bracket", "Crossfade shader previous", 91, 0, false, 0, 0, 0});
    mappings.push_back({"Right Bracket", "Crossfade shader next", 93, 0, false, 0, 0, 0});
}

void MidiMapWindow::change_target(int index) {
    if (index < 0 || index > 1) {
        return;
    }

    const MidiTargetProfile next_profile =
        index == 0 ? MidiTargetProfile::Acmx2 : MidiTargetProfile::Acmxvk;
    if (next_profile == target_profile) {
        return;
    }

    if (capturing) {
        capturing = false;
        captureRow = -1;
        captureButton->setText("Capture Selected");
        captureButton->setEnabled(deviceOpen);
    }

    const std::size_t previous_profile_index =
        target_profile == MidiTargetProfile::Acmx2 ? 0U : 1U;
    const std::size_t next_profile_index =
        next_profile == MidiTargetProfile::Acmx2 ? 0U : 1U;
    profile_mappings[previous_profile_index] = mappings;
    const std::vector<MidiMapping> previous_mappings = mappings;
    target_profile = next_profile;

    if (profile_mappings[next_profile_index].empty()) {
        populateActions();
        for (MidiMapping &mapping : mappings) {
            const bool target_specific_meaning =
                mapping.key2 == 0 &&
                (mapping.key1 == 53 || mapping.key1 == 70 ||
                 mapping.key1 == 91 || mapping.key1 == 93);
            if (target_specific_meaning) {
                continue;
            }
            const auto previous = std::find_if(
                previous_mappings.begin(), previous_mappings.end(),
                [&](const MidiMapping &candidate) {
                    return candidate.captured &&
                           candidate.key1 == mapping.key1 &&
                           candidate.key2 == mapping.key2;
                });
            if (previous == previous_mappings.end()) {
                continue;
            }
            mapping.captured = true;
            mapping.byte0 = previous->byte0;
            mapping.byte1 = previous->byte1;
            mapping.byte2 = previous->byte2;
        }
        profile_mappings[next_profile_index] = mappings;
    } else {
        mappings = profile_mappings[next_profile_index];
    }

    const QString target_name =
        target_profile == MidiTargetProfile::Acmx2 ? "ACMX2" : "ACMXVK";
    setWindowTitle(target_name + " MIDI Map Configuration");
    updateTable();
    setStatus(QString("%1 profile selected — %2 actions available; shared "
                      "captured mappings preserved")
                  .arg(target_name)
                  .arg(mappings.size()));
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
                table->setItem(i, 3, new QTableWidgetItem(QString("CC %1 %2 (knob)").arg(m.byte0).arg(m.byte1)));
                auto *statusItem = new QTableWidgetItem("Knob mapped");
                statusItem->setForeground(QBrush(QColor("#00ff00")));
                table->setItem(i, 4, statusItem);
            } else {
                table->setItem(i, 3, new QTableWidgetItem(QString("%1 %2 %3").arg(m.byte0).arg(m.byte1).arg(m.byte2)));
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
    if (!midiIn)
        return;

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
    if (!midiIn || index < 0)
        return;

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
    if (!midiIn || !deviceOpen)
        return;

    std::vector<unsigned char> message;
    midiIn->getMessage(&message);

    if (message.size() < 3)
        return;

    midiMonitorLabel->setText(QString("Last MIDI: [%1 %2 %3]  (ch %4)")
                                  .arg(message[0])
                                  .arg(message[1])
                                  .arg(message[2])
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
                            .arg(message[0])
                            .arg(message[1])
                            .arg(QString::fromStdString(mappings[captureRow].actionName))
                            .arg(QString::fromStdString(mappings[captureRow].actionName).split('/').first().trimmed())
                            .arg(QString::fromStdString(mappings[captureRow].actionName).split('/').last().trimmed());
        } else {
            statusMsg = QString("Captured [%1 %2 %3] for: %4")
                            .arg(message[0])
                            .arg(message[1])
                            .arg(message[2])
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
    if (fileName.isEmpty())
        return;

    std::ofstream file(fileName.toStdString());
    if (!file.is_open()) {
        QMessageBox::critical(this, "Error", "Could not open file for writing.");
        return;
    }

    int count = 0;
    for (const auto &m : mappings) {
        if (!m.captured)
            continue;
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
    if (fileName.isEmpty())
        return;

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
        if (!(iss >> keyPair))
            continue;

        auto colonPos = keyPair.find(':');
        if (colonPos == std::string::npos)
            continue;

        int k1 = std::stoi(keyPair.substr(0, colonPos));
        int k2 = std::stoi(keyPair.substr(colonPos + 1));

        char brace;
        if (!(iss >> brace) || brace != '{')
            continue;

        int b0, b1, b2;
        if (!(iss >> b0 >> b1 >> b2))
            continue;

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
