#include "stable-diffusion-settings.hpp"

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
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>

#include <limits>

StableDiffusionSettingsDialog::StableDiffusionSettingsDialog(QWidget *parent)
    : QDialog(parent) {
    setWindowTitle("Stable Diffusion Settings");
    setMinimumSize(600, 600);

    enable_check_box = new QCheckBox("Enable Stable Diffusion", this);
    model_file_edit = new QLineEdit(this);
    model_file_edit->setReadOnly(true);
    model_file_edit->setPlaceholderText("Select a safetensors model...");
    browse_model_button = new QPushButton("Browse...", this);
    prompt_edit = new QLineEdit(this);
    prompt_edit->setPlaceholderText("Describe the desired image style...");
    negative_prompt_edit = new QLineEdit(this);
    negative_prompt_edit->setPlaceholderText("Optional unwanted features...");

    server_edit = new QLineEdit(this);
    server_edit->setPlaceholderText("sd-server");
    browse_server_button = new QPushButton("Browse...", this);
    server_port_spin_box = new QSpinBox(this);
    server_port_spin_box->setRange(1024, 65535);

    width_spin_box = new QSpinBox(this);
    width_spin_box->setRange(64, 2048);
    width_spin_box->setSingleStep(64);
    width_spin_box->setSuffix(" px");
    height_spin_box = new QSpinBox(this);
    height_spin_box->setRange(64, 2048);
    height_spin_box->setSingleStep(64);
    height_spin_box->setSuffix(" px");
    steps_spin_box = new QSpinBox(this);
    steps_spin_box->setRange(1, 150);
    strength_spin_box = new QDoubleSpinBox(this);
    strength_spin_box->setRange(0.01, 1.0);
    strength_spin_box->setDecimals(3);
    strength_spin_box->setSingleStep(0.05);
    cfg_scale_spin_box = new QDoubleSpinBox(this);
    cfg_scale_spin_box->setRange(0.0, 50.0);
    cfg_scale_spin_box->setDecimals(2);
    cfg_scale_spin_box->setSingleStep(0.5);
    seed_spin_box = new QSpinBox(this);
    seed_spin_box->setRange(std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::max());
    sampler_combo_box = new QComboBox(this);
    sampler_combo_box->setEditable(true);
    sampler_combo_box->addItems(
        {"euler_a", "euler", "dpm++2m", "dpm++2mv2", "lcm"});
    scheduler_combo_box = new QComboBox(this);
    scheduler_combo_box->setEditable(true);
    scheduler_combo_box->addItems(
        {"discrete", "karras", "exponential", "ays"});
    upscale_check_box = new QCheckBox(
        "High-quality Vulkan compute upscale before shaders", this);
    upscale_check_box->setToolTip(
        "Keep the native Stable Diffusion image size and run ACMXVK's "
        "bicubic detail-preserving compute stage before user shaders.");

    auto *model_group = new QGroupBox("Image-to-Image Model", this);
    auto *model_layout = new QFormLayout(model_group);
    auto *model_row = new QHBoxLayout;
    model_row->addWidget(model_file_edit, 1);
    model_row->addWidget(browse_model_button);
    model_layout->addRow("Safetensors model:", model_row);
    model_layout->addRow("Prompt:", prompt_edit);
    model_layout->addRow("Negative prompt:", negative_prompt_edit);

    auto *generation_group = new QGroupBox("Generation", this);
    auto *generation_layout = new QFormLayout(generation_group);
    generation_layout->addRow("Width:", width_spin_box);
    generation_layout->addRow("Height:", height_spin_box);
    generation_layout->addRow("Steps:", steps_spin_box);
    generation_layout->addRow("Denoising strength:", strength_spin_box);
    generation_layout->addRow("CFG scale:", cfg_scale_spin_box);
    generation_layout->addRow("Seed:", seed_spin_box);
    generation_layout->addRow("Sampler:", sampler_combo_box);
    generation_layout->addRow("Scheduler:", scheduler_combo_box);
    generation_layout->addRow(upscale_check_box);

    auto *server_group = new QGroupBox("Local sd-server", this);
    auto *server_layout = new QFormLayout(server_group);
    auto *server_row = new QHBoxLayout;
    server_row->addWidget(server_edit, 1);
    server_row->addWidget(browse_server_button);
    server_layout->addRow("Executable:", server_row);
    server_layout->addRow("Loopback port:", server_port_spin_box);

    auto *contents = new QWidget(this);
    auto *contents_layout = new QVBoxLayout(contents);
    contents_layout->addWidget(enable_check_box);
    contents_layout->addWidget(model_group);
    contents_layout->addWidget(generation_group);
    contents_layout->addWidget(server_group);
    contents_layout->addStretch();

    auto *scroll_area = new QScrollArea(this);
    scroll_area->setWidgetResizable(true);
    scroll_area->setWidget(contents);

    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Apply |
            QDialogButtonBox::Cancel,
        this);
    auto *layout = new QVBoxLayout(this);
    layout->addWidget(scroll_area, 1);
    layout->addWidget(buttons);

    connect(enable_check_box, &QCheckBox::toggled, this,
            [this](bool) { update_enabled_state(); });
    connect(browse_model_button, &QPushButton::clicked, this,
            &StableDiffusionSettingsDialog::browse_model);
    connect(browse_server_button, &QPushButton::clicked, this,
            &StableDiffusionSettingsDialog::browse_server);
    connect(buttons, &QDialogButtonBox::accepted, this,
            &StableDiffusionSettingsDialog::accept_settings);
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked,
            this, &StableDiffusionSettingsDialog::apply_settings);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    load_ui_state();
    update_enabled_state();
    acmx2::applyCustomStyleIfEnabled(this);
}

StableDiffusionConfiguration
StableDiffusionSettingsDialog::configuration() const {
    StableDiffusionConfiguration result;
    result.enabled = enable_check_box->isChecked();
    result.model_file = model_file_edit->text().trimmed();
    result.prompt = prompt_edit->text().trimmed();
    result.negative_prompt = negative_prompt_edit->text().trimmed();
    result.server_executable = server_edit->text().trimmed();
    result.server_port = server_port_spin_box->value();
    result.width = width_spin_box->value();
    result.height = height_spin_box->value();
    result.steps = steps_spin_box->value();
    result.strength = strength_spin_box->value();
    result.cfg_scale = cfg_scale_spin_box->value();
    result.seed = seed_spin_box->value();
    result.sampler = sampler_combo_box->currentText().trimmed();
    result.scheduler = scheduler_combo_box->currentText().trimmed();
    result.upscale = upscale_check_box->isChecked();
    return result;
}

void StableDiffusionSettingsDialog::browse_model() {
    QSettings settings("LostSideDead", "acmx2");
    const QString directory =
        settings.value("stable_diffusion/last_model_directory")
            .toString();
    const QString filename = QFileDialog::getOpenFileName(
        this, "Select Stable Diffusion Model", directory,
        "Safetensors Models (*.safetensors);;All Files (*)");
    if (filename.isEmpty()) {
        return;
    }
    model_file_edit->setText(QFileInfo(filename).absoluteFilePath());
    settings.setValue("stable_diffusion/last_model_directory",
                      QFileInfo(filename).absolutePath());
}

void StableDiffusionSettingsDialog::browse_server() {
    const QString filename = QFileDialog::getOpenFileName(
        this, "Select sd-server Executable",
        QFileInfo(server_edit->text()).absolutePath());
    if (!filename.isEmpty()) {
        server_edit->setText(QFileInfo(filename).absoluteFilePath());
    }
}

void StableDiffusionSettingsDialog::apply_settings() {
    if (!validate_settings()) {
        return;
    }
    save_ui_state();
    emit settingsApplied();
}

void StableDiffusionSettingsDialog::accept_settings() {
    if (!validate_settings()) {
        return;
    }
    save_ui_state();
    emit settingsApplied();
    accept();
}

bool StableDiffusionSettingsDialog::validate_settings() {
    if (!enable_check_box->isChecked()) {
        return true;
    }
    if (!QFileInfo(model_file_edit->text().trimmed()).isFile()) {
        QMessageBox::warning(this, "Stable Diffusion Model Required",
                             "Select an existing safetensors model file.");
        return false;
    }
    if (prompt_edit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "Stable Diffusion Prompt Required",
                             "Enter an image-to-image prompt.");
        return false;
    }
    if (server_edit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "sd-server Required",
                             "Enter sd-server or select its executable.");
        return false;
    }
    if ((width_spin_box->value() % 64) != 0 ||
        (height_spin_box->value() % 64) != 0) {
        QMessageBox::warning(
            this, "Invalid Stable Diffusion Size",
            "The Stable Diffusion width and height must be multiples of 64.");
        return false;
    }
    if (sampler_combo_box->currentText().trimmed().isEmpty() ||
        scheduler_combo_box->currentText().trimmed().isEmpty()) {
        QMessageBox::warning(this, "Generation Method Required",
                             "Enter both a sampler and scheduler.");
        return false;
    }
    return true;
}

void StableDiffusionSettingsDialog::load_ui_state() {
    QSettings settings("LostSideDead", "acmx2");
    enable_check_box->setChecked(
        settings.value("stable_diffusion/enabled", false).toBool());
    model_file_edit->setText(
        settings.value("stable_diffusion/model_file").toString());
    prompt_edit->setText(settings.value("stable_diffusion/prompt").toString());
    negative_prompt_edit->setText(
        settings.value("stable_diffusion/negative_prompt").toString());
    server_edit->setText(
        settings.value("stable_diffusion/server", "sd-server").toString());
    server_port_spin_box->setValue(
        settings.value("stable_diffusion/port", 1234).toInt());
    width_spin_box->setValue(
        settings.value("stable_diffusion/width", 576).toInt());
    height_spin_box->setValue(
        settings.value("stable_diffusion/height", 320).toInt());
    steps_spin_box->setValue(
        settings.value("stable_diffusion/steps", 12).toInt());
    strength_spin_box->setValue(
        settings.value("stable_diffusion/strength", 0.35).toDouble());
    cfg_scale_spin_box->setValue(
        settings.value("stable_diffusion/cfg_scale", 5.0).toDouble());
    seed_spin_box->setValue(
        settings.value("stable_diffusion/seed", 1234).toInt());
    sampler_combo_box->setCurrentText(
        settings.value("stable_diffusion/sampler", "euler_a").toString());
    scheduler_combo_box->setCurrentText(
        settings.value("stable_diffusion/scheduler", "discrete").toString());
    upscale_check_box->setChecked(
        settings.value("stable_diffusion/upscale", false).toBool());
}

void StableDiffusionSettingsDialog::save_ui_state() {
    const StableDiffusionConfiguration current = configuration();
    QSettings settings("LostSideDead", "acmx2");
    settings.setValue("stable_diffusion/enabled", current.enabled);
    settings.setValue("stable_diffusion/model_file", current.model_file);
    settings.setValue("stable_diffusion/prompt", current.prompt);
    settings.setValue("stable_diffusion/negative_prompt",
                      current.negative_prompt);
    settings.setValue("stable_diffusion/server", current.server_executable);
    settings.setValue("stable_diffusion/port", current.server_port);
    settings.setValue("stable_diffusion/width", current.width);
    settings.setValue("stable_diffusion/height", current.height);
    settings.setValue("stable_diffusion/steps", current.steps);
    settings.setValue("stable_diffusion/strength", current.strength);
    settings.setValue("stable_diffusion/cfg_scale", current.cfg_scale);
    settings.setValue("stable_diffusion/seed", current.seed);
    settings.setValue("stable_diffusion/sampler", current.sampler);
    settings.setValue("stable_diffusion/scheduler", current.scheduler);
    settings.setValue("stable_diffusion/upscale", current.upscale);
    settings.sync();
}

void StableDiffusionSettingsDialog::update_enabled_state() {
    const bool enabled = enable_check_box->isChecked();
    model_file_edit->setEnabled(enabled);
    browse_model_button->setEnabled(enabled);
    prompt_edit->setEnabled(enabled);
    negative_prompt_edit->setEnabled(enabled);
    server_edit->setEnabled(enabled);
    browse_server_button->setEnabled(enabled);
    server_port_spin_box->setEnabled(enabled);
    width_spin_box->setEnabled(enabled);
    height_spin_box->setEnabled(enabled);
    steps_spin_box->setEnabled(enabled);
    strength_spin_box->setEnabled(enabled);
    cfg_scale_spin_box->setEnabled(enabled);
    seed_spin_box->setEnabled(enabled);
    sampler_combo_box->setEnabled(enabled);
    scheduler_combo_box->setEnabled(enabled);
    upscale_check_box->setEnabled(enabled);
}
