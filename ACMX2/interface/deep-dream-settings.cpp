#include "deep-dream-settings.hpp"

#include "custom_style.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>

#include <array>

DeepDreamSettingsDialog::DeepDreamSettingsDialog(bool gpu_filter_enabled,
                                                 QWidget *parent)
    : QDialog(parent), gpu_filter_available(gpu_filter_enabled) {
    setWindowTitle("Deep Dream Settings");
    setMinimumSize(560, 620);

    enable_check_box = new QCheckBox("Enable Deep Dream", this);
    model_file_edit = new QLineEdit(this);
    model_file_edit->setReadOnly(true);
    model_file_edit->setPlaceholderText("Select a TorchScript .pt model...");
    browse_model_button = new QPushButton("Browse...", this);

    layer_combo_box = new QComboBox(this);
    layer_combo_box->setEditable(true);
    layer_combo_box->addItems(
        {"relu1_1", "relu1_2", "relu2_1", "relu2_2", "relu3_1",
         "relu3_2", "relu3_3", "relu4_1", "relu4_2", "relu4_3",
         "relu5_1", "relu5_2", "relu5_3"});

    iterations_spin_box = new QSpinBox(this);
    iterations_spin_box->setRange(1, 100);
    strength_spin_box = new QDoubleSpinBox(this);
    strength_spin_box->setRange(0.0001, 10.0);
    strength_spin_box->setDecimals(4);
    strength_spin_box->setSingleStep(0.01);
    feedback_spin_box = new QDoubleSpinBox(this);
    feedback_spin_box->setRange(0.0, 0.99);
    feedback_spin_box->setDecimals(3);
    feedback_spin_box->setSingleStep(0.05);
    zoom_spin_box = new QDoubleSpinBox(this);
    zoom_spin_box->setRange(0.9, 1.1);
    zoom_spin_box->setDecimals(4);
    zoom_spin_box->setSingleStep(0.005);
    rotation_spin_box = new QDoubleSpinBox(this);
    rotation_spin_box->setRange(-5.0, 5.0);
    rotation_spin_box->setDecimals(3);
    rotation_spin_box->setSingleStep(0.1);

    native_size_check_box = new QCheckBox("Use native input size", this);
    maximum_dimension_spin_box = new QSpinBox(this);
    maximum_dimension_spin_box->setRange(64, 4096);
    maximum_dimension_spin_box->setSingleStep(64);
    maximum_dimension_spin_box->setSuffix(" px");
    fp16_check_box = new QCheckBox("Use FP16", this);
    fp16_check_box->setToolTip(
        "Use half-precision model and working tensors to reduce CUDA memory "
        "and improve performance.");
    channel_spin_box = new QSpinBox(this);
    channel_spin_box->setRange(-1, 65535);
    channel_spin_box->setSpecialValueText("All channels");
    octaves_spin_box = new QSpinBox(this);
    octaves_spin_box->setRange(1, 8);
    octave_scale_spin_box = new QDoubleSpinBox(this);
    octave_scale_spin_box->setRange(1.1, 3.0);
    octave_scale_spin_box->setDecimals(2);
    octave_scale_spin_box->setSingleStep(0.1);
    jitter_spin_box = new QSpinBox(this);
    jitter_spin_box->setRange(0, 64);
    jitter_spin_box->setSuffix(" px");
    smoothing_spin_box = new QSpinBox(this);
    smoothing_spin_box->setRange(0, 16);
    smoothing_spin_box->setSuffix(" px");
    gpu_filter_first_check_box =
        new QCheckBox("Run acidcam-gpu filters before Deep Dream", this);
    gpu_filter_first_check_box->setToolTip(
        gpu_filter_available
            ? "Transform the CUDA source with acidcam-gpu before LibTorch."
            : "Enable an acidcam-gpu filter chain first.");

    auto *model_group = new QGroupBox("Model", this);
    auto *model_layout = new QFormLayout(model_group);
    auto *model_row = new QHBoxLayout;
    model_row->addWidget(model_file_edit, 1);
    model_row->addWidget(browse_model_button);
    model_layout->addRow("TorchScript model:", model_row);
    model_layout->addRow("Feature layer:", layer_combo_box);

    auto *dream_group = new QGroupBox("Gradient Ascent", this);
    auto *dream_layout = new QFormLayout(dream_group);
    dream_layout->addRow("Iterations:", iterations_spin_box);
    dream_layout->addRow("Strength:", strength_spin_box);
    dream_layout->addRow("Target channel:", channel_spin_box);
    dream_layout->addRow("Octaves:", octaves_spin_box);
    dream_layout->addRow("Octave scale:", octave_scale_spin_box);
    dream_layout->addRow("Spatial jitter:", jitter_spin_box);
    dream_layout->addRow("Gradient smoothing:", smoothing_spin_box);

    auto *feedback_group = new QGroupBox("Temporal Feedback", this);
    auto *feedback_layout = new QFormLayout(feedback_group);
    feedback_layout->addRow("Previous-frame blend:", feedback_spin_box);
    feedback_layout->addRow("Feedback zoom:", zoom_spin_box);
    feedback_layout->addRow("Feedback rotation:", rotation_spin_box);

    auto *performance_group = new QGroupBox("Performance and Pipeline", this);
    auto *performance_layout = new QFormLayout(performance_group);
    performance_layout->addRow(native_size_check_box);
    performance_layout->addRow("Maximum dimension:",
                               maximum_dimension_spin_box);
    performance_layout->addRow(fp16_check_box);
    performance_layout->addRow(gpu_filter_first_check_box);

    auto *contents = new QWidget(this);
    auto *contents_layout = new QVBoxLayout(contents);
    contents_layout->addWidget(enable_check_box);
    contents_layout->addWidget(model_group);
    contents_layout->addWidget(dream_group);
    contents_layout->addWidget(feedback_group);
    contents_layout->addWidget(performance_group);
    contents_layout->addStretch();

    auto *scroll_area = new QScrollArea(this);
    scroll_area->setWidgetResizable(true);
    scroll_area->setWidget(contents);

    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    auto *layout = new QVBoxLayout(this);
    layout->addWidget(scroll_area, 1);
    layout->addWidget(buttons);

    connect(enable_check_box, &QCheckBox::toggled, this,
            [this](bool) { update_enabled_state(); });
    connect(native_size_check_box, &QCheckBox::toggled, this,
            [this](bool) { update_enabled_state(); });
    connect(browse_model_button, &QPushButton::clicked, this,
            &DeepDreamSettingsDialog::browse_model);
    connect(buttons, &QDialogButtonBox::accepted, this,
            &DeepDreamSettingsDialog::accept_settings);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    load_ui_state();
    update_enabled_state();
    acmx2::applyCustomStyleIfEnabled(this);
}

DeepDreamConfiguration DeepDreamSettingsDialog::configuration() const {
    DeepDreamConfiguration result;
    result.enabled = enable_check_box->isChecked();
    result.model_file = model_file_edit->text().trimmed();
    result.layer = layer_combo_box->currentText().trimmed();
    result.iterations = iterations_spin_box->value();
    result.strength = strength_spin_box->value();
    result.feedback = feedback_spin_box->value();
    result.zoom = zoom_spin_box->value();
    result.rotation = rotation_spin_box->value();
    result.maximum_dimension =
        native_size_check_box->isChecked()
            ? 0
            : maximum_dimension_spin_box->value();
    result.fp16 = fp16_check_box->isChecked();
    result.channel = channel_spin_box->value();
    result.octaves = octaves_spin_box->value();
    result.octave_scale = octave_scale_spin_box->value();
    result.jitter = jitter_spin_box->value();
    result.smoothing = smoothing_spin_box->value();
    result.gpu_filter_first = gpu_filter_first_check_box->isChecked();
    return result;
}

void DeepDreamSettingsDialog::browse_model() {
    QSettings settings("LostSideDead", "acmx2");
    QString directory = settings.value("deep_dream/last_model_directory")
                            .toString();
    if (directory.isEmpty()) {
        directory = QFileInfo(model_file_edit->text()).absolutePath();
    }
    const QString filename = QFileDialog::getOpenFileName(
        this, "Select Deep Dream TorchScript Model", directory,
        "TorchScript Models (*.pt *.pth);;All Files (*)");
    if (filename.isEmpty()) {
        return;
    }
    model_file_edit->setText(filename);
    settings.setValue("deep_dream/last_model_directory",
                      QFileInfo(filename).absolutePath());
}

void DeepDreamSettingsDialog::accept_settings() {
    if (enable_check_box->isChecked()) {
        const QFileInfo model(model_file_edit->text().trimmed());
        if (!model.isFile()) {
            QMessageBox::warning(this, "Deep Dream Model Required",
                                 "Select an existing TorchScript model file.");
            model_file_edit->setFocus();
            return;
        }
        static const QRegularExpression layer_pattern(
            QStringLiteral("^[A-Za-z0-9_.-]+$"));
        if (!layer_pattern.match(layer_combo_box->currentText().trimmed())
                 .hasMatch()) {
            QMessageBox::warning(
                this, "Invalid Deep Dream Layer",
                "Enter a named layer such as relu4_2 or a numeric layer index.");
            layer_combo_box->setFocus();
            return;
        }
        if (gpu_filter_first_check_box->isChecked() &&
            !gpu_filter_available) {
            QMessageBox::warning(
                this, "GPU Filter Required",
                "Configure and enable an acidcam-gpu filter chain before "
                "selecting GPU filters before Deep Dream.");
            return;
        }
    }
    save_ui_state();
    accept();
}

void DeepDreamSettingsDialog::load_ui_state() {
    QSettings settings("LostSideDead", "acmx2");
    enable_check_box->setChecked(
        settings.value("deep_dream/enabled", false).toBool());
    model_file_edit->setText(
        settings.value("deep_dream/model_file", QString()).toString());
    layer_combo_box->setCurrentText(
        settings.value("deep_dream/layer", "relu4_2").toString());
    iterations_spin_box->setValue(
        settings.value("deep_dream/iterations", 1).toInt());
    strength_spin_box->setValue(
        settings.value("deep_dream/strength", 0.05).toDouble());
    feedback_spin_box->setValue(
        settings.value("deep_dream/feedback", 0.9).toDouble());
    zoom_spin_box->setValue(
        settings.value("deep_dream/zoom", 1.01).toDouble());
    rotation_spin_box->setValue(
        settings.value("deep_dream/rotation", 0.1).toDouble());
    const int maximum_dimension =
        settings.value("deep_dream/maximum_dimension", 512).toInt();
    native_size_check_box->setChecked(maximum_dimension == 0);
    maximum_dimension_spin_box->setValue(
        maximum_dimension == 0 ? 512 : maximum_dimension);
    fp16_check_box->setChecked(
        settings.value("deep_dream/fp16", false).toBool());
    channel_spin_box->setValue(
        settings.value("deep_dream/channel", -1).toInt());
    octaves_spin_box->setValue(
        settings.value("deep_dream/octaves", 1).toInt());
    octave_scale_spin_box->setValue(
        settings.value("deep_dream/octave_scale", 1.4).toDouble());
    jitter_spin_box->setValue(
        settings.value("deep_dream/jitter", 0).toInt());
    smoothing_spin_box->setValue(
        settings.value("deep_dream/smoothing", 0).toInt());
    gpu_filter_first_check_box->setChecked(
        gpu_filter_available &&
        settings.value("deep_dream/gpu_filter_first", false).toBool());
}

void DeepDreamSettingsDialog::save_ui_state() {
    const DeepDreamConfiguration current = configuration();
    QSettings settings("LostSideDead", "acmx2");
    settings.setValue("deep_dream/enabled", current.enabled);
    settings.setValue("deep_dream/model_file", current.model_file);
    settings.setValue("deep_dream/layer", current.layer);
    settings.setValue("deep_dream/iterations", current.iterations);
    settings.setValue("deep_dream/strength", current.strength);
    settings.setValue("deep_dream/feedback", current.feedback);
    settings.setValue("deep_dream/zoom", current.zoom);
    settings.setValue("deep_dream/rotation", current.rotation);
    settings.setValue("deep_dream/maximum_dimension",
                      current.maximum_dimension);
    settings.setValue("deep_dream/fp16", current.fp16);
    settings.setValue("deep_dream/channel", current.channel);
    settings.setValue("deep_dream/octaves", current.octaves);
    settings.setValue("deep_dream/octave_scale", current.octave_scale);
    settings.setValue("deep_dream/jitter", current.jitter);
    settings.setValue("deep_dream/smoothing", current.smoothing);
    settings.setValue("deep_dream/gpu_filter_first",
                      current.gpu_filter_first);
}

void DeepDreamSettingsDialog::update_enabled_state() {
    const bool enabled = enable_check_box->isChecked();
    const std::array<QWidget *, 15> controls = {
        model_file_edit, browse_model_button, layer_combo_box,
        iterations_spin_box, strength_spin_box, feedback_spin_box,
        zoom_spin_box, rotation_spin_box, native_size_check_box,
        fp16_check_box, channel_spin_box, octaves_spin_box,
        octave_scale_spin_box, jitter_spin_box, smoothing_spin_box};
    for (QWidget *widget : controls) {
        widget->setEnabled(enabled);
    }
    maximum_dimension_spin_box->setEnabled(
        enabled && !native_size_check_box->isChecked());
    gpu_filter_first_check_box->setEnabled(enabled && gpu_filter_available);
    if (!gpu_filter_available) {
        gpu_filter_first_check_box->setChecked(false);
    }
}
