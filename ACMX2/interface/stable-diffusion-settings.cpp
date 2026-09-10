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
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <limits>

namespace {
    constexpr int STABLE_DIMENSION_MINIMUM = 64;
    constexpr int STABLE_DIMENSION_MAXIMUM = 2048;
    constexpr int LORA_PATH_ROLE = Qt::UserRole;
    constexpr int LORA_MULTIPLIER_ROLE = Qt::UserRole + 1;

    bool parse_stable_resolution(const QString &text, int &width, int &height) {
        static const QRegularExpression RESOLUTION_PATTERN(QStringLiteral("^\\s*(\\d+)\\s*[xX]\\s*(\\d+)\\s*$"));
        const QRegularExpressionMatch match = RESOLUTION_PATTERN.match(text);
        if (!match.hasMatch()) {
            return false;
        }

        bool width_ok = false;
        bool height_ok = false;
        const int parsed_width = match.captured(1).toInt(&width_ok);
        const int parsed_height = match.captured(2).toInt(&height_ok);
        if (!width_ok || !height_ok) {
            return false;
        }

        width = parsed_width;
        height = parsed_height;
        return true;
    }

    QString resolution_text(int width, int height) { return QStringLiteral("%1x%2").arg(width).arg(height); }
} // namespace

StableDiffusionSettingsDialog::StableDiffusionSettingsDialog(QWidget *parent) : QDialog(parent) {
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
    lora_list_widget = new QListWidget(this);
    lora_list_widget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    lora_list_widget->setMinimumHeight(100);
    add_lora_button = new QPushButton("Add...", this);
    remove_lora_button = new QPushButton("Remove", this);
    lora_multiplier_spin_box = new QDoubleSpinBox(this);
    lora_multiplier_spin_box->setRange(-10.0, 10.0);
    lora_multiplier_spin_box->setDecimals(3);
    lora_multiplier_spin_box->setSingleStep(0.05);
    lora_multiplier_spin_box->setValue(1.0);

    server_edit = new QLineEdit(this);
    server_edit->setPlaceholderText("sd-server");
    server_arguments_edit = new QLineEdit(this);
    server_arguments_edit->setPlaceholderText("Optional flags, for example --vae-tiling --offload-to-cpu");
    server_arguments_edit->setToolTip("Additional command-line arguments passed directly to sd-server. "
                                      "Use quotes around values containing spaces.");
    browse_server_button = new QPushButton("Browse...", this);
    server_port_spin_box = new QSpinBox(this);
    server_port_spin_box->setRange(1024, 65535);

    resolution_combo_box = new QComboBox(this);
    resolution_combo_box->setEditable(true);
    resolution_combo_box->setInsertPolicy(QComboBox::NoInsert);
    resolution_combo_box->lineEdit()->setPlaceholderText("WIDTHxHEIGHT");
    resolution_combo_box->addItems({"512x512", "576x320", "640x384", "704x448", "768x448", "768x512", "832x512", "896x512", "1024x576", "1024x1024", "1152x640", "1280x768", "1344x768", "1536x896", "1920x1088"});
    resolution_combo_box->setToolTip("Choose a preset or enter WIDTHxHEIGHT. Both dimensions must be "
                                     "multiples of 64 between 64 and 2048 pixels.");
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
    seed_spin_box->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    sampler_combo_box = new QComboBox(this);
    sampler_combo_box->setEditable(true);
    sampler_combo_box->addItems({"euler_a", "euler", "dpm++2m", "dpm++2mv2", "lcm"});
    scheduler_combo_box = new QComboBox(this);
    scheduler_combo_box->setEditable(true);
    scheduler_combo_box->addItems({"discrete", "karras", "exponential", "ays"});
    upscale_check_box = new QCheckBox("High-quality Vulkan compute upscale before shaders", this);
    upscale_check_box->setToolTip("Keep the native Stable Diffusion image size and run ACMXVK's "
                                  "bicubic detail-preserving compute stage before user shaders.");
    server_upscale_check_box = new QCheckBox("Use an sd-server ESRGAN upscale model", this);
    server_upscale_check_box->setToolTip("Load an ESRGAN or RealESRGAN model in sd-server and use its neural "
                                         "upscaler instead of ACMXVK's Vulkan compute upscaler.");
    upscale_model_edit = new QLineEdit(this);
    upscale_model_edit->setReadOnly(true);
    upscale_model_edit->setPlaceholderText("Select an ESRGAN/RealESRGAN model...");
    browse_upscale_model_button = new QPushButton("Browse...", this);

    auto *model_group = new QGroupBox("Image-to-Image Model", this);
    auto *model_layout = new QFormLayout(model_group);
    auto *model_row = new QHBoxLayout;
    model_row->addWidget(model_file_edit, 1);
    model_row->addWidget(browse_model_button);
    model_layout->addRow("Safetensors model:", model_row);
    model_layout->addRow("Prompt:", prompt_edit);
    model_layout->addRow("Negative prompt:", negative_prompt_edit);

    auto *lora_group = new QGroupBox("LoRA Models", this);
    lora_group->setToolTip("Optional adapters applied by sd-server. All selected LoRA files "
                           "must be in the same folder.");
    auto *lora_layout = new QVBoxLayout(lora_group);
    lora_layout->addWidget(lora_list_widget);
    auto *lora_controls = new QHBoxLayout;
    lora_controls->addWidget(add_lora_button);
    lora_controls->addWidget(remove_lora_button);
    lora_controls->addStretch();
    lora_controls->addWidget(new QLabel("Selected multiplier:", this));
    lora_controls->addWidget(lora_multiplier_spin_box);
    lora_layout->addLayout(lora_controls);

    auto *generation_group = new QGroupBox("Generation", this);
    auto *generation_layout = new QFormLayout(generation_group);
    generation_layout->addRow("Resolution:", resolution_combo_box);
    generation_layout->addRow("Steps:", steps_spin_box);
    generation_layout->addRow("Denoising strength:", strength_spin_box);
    generation_layout->addRow("CFG scale:", cfg_scale_spin_box);
    generation_layout->addRow("Seed:", seed_spin_box);
    generation_layout->addRow("Sampler:", sampler_combo_box);
    generation_layout->addRow("Scheduler:", scheduler_combo_box);
    generation_layout->addRow(upscale_check_box);
    generation_layout->addRow(server_upscale_check_box);
    auto *upscale_model_row = new QHBoxLayout;
    upscale_model_row->addWidget(upscale_model_edit, 1);
    upscale_model_row->addWidget(browse_upscale_model_button);
    generation_layout->addRow("Upscale model:", upscale_model_row);

    auto *server_group = new QGroupBox("Local sd-server", this);
    auto *server_layout = new QFormLayout(server_group);
    auto *server_row = new QHBoxLayout;
    server_row->addWidget(server_edit, 1);
    server_row->addWidget(browse_server_button);
    server_layout->addRow("Executable:", server_row);
    server_layout->addRow("Extra flags:", server_arguments_edit);
    server_layout->addRow("Loopback port:", server_port_spin_box);

    auto *contents = new QWidget(this);
    auto *contents_layout = new QVBoxLayout(contents);
    contents_layout->addWidget(enable_check_box);
    contents_layout->addWidget(model_group);
    contents_layout->addWidget(lora_group);
    contents_layout->addWidget(generation_group);
    contents_layout->addWidget(server_group);
    contents_layout->addStretch();

    auto *scroll_area = new QScrollArea(this);
    scroll_area->setWidgetResizable(true);
    scroll_area->setWidget(contents);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Apply | QDialogButtonBox::Cancel, this);
    auto *layout = new QVBoxLayout(this);
    layout->addWidget(scroll_area, 1);
    layout->addWidget(buttons);

    connect(enable_check_box, &QCheckBox::toggled, this, [this](bool) { update_enabled_state(); });
    connect(browse_model_button, &QPushButton::clicked, this, &StableDiffusionSettingsDialog::browse_model);
    connect(add_lora_button, &QPushButton::clicked, this, &StableDiffusionSettingsDialog::add_lora_models);
    connect(remove_lora_button, &QPushButton::clicked, this, &StableDiffusionSettingsDialog::remove_lora_models);
    connect(lora_list_widget, &QListWidget::itemSelectionChanged, this, &StableDiffusionSettingsDialog::select_lora_model);
    connect(lora_multiplier_spin_box, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &StableDiffusionSettingsDialog::update_lora_multiplier);
    connect(browse_upscale_model_button, &QPushButton::clicked, this, &StableDiffusionSettingsDialog::browse_upscale_model);
    connect(upscale_check_box, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) {
            server_upscale_check_box->setChecked(false);
        }
        update_enabled_state();
    });
    connect(server_upscale_check_box, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) {
            upscale_check_box->setChecked(false);
        }
        update_enabled_state();
    });
    connect(browse_server_button, &QPushButton::clicked, this, &StableDiffusionSettingsDialog::browse_server);
    connect(buttons, &QDialogButtonBox::accepted, this, &StableDiffusionSettingsDialog::accept_settings);
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked, this, &StableDiffusionSettingsDialog::apply_settings);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    load_ui_state();
    update_enabled_state();
    acmx2::applyCustomStyleIfEnabled(this);
}

StableDiffusionConfiguration StableDiffusionSettingsDialog::configuration() const {
    StableDiffusionConfiguration result;
    result.enabled = enable_check_box->isChecked();
    result.model_file = model_file_edit->text().trimmed();
    for (int row = 0; row < lora_list_widget->count(); ++row) {
        const QListWidgetItem *item = lora_list_widget->item(row);
        result.lora_files.append(item->data(LORA_PATH_ROLE).toString());
        result.lora_multipliers.append(item->data(LORA_MULTIPLIER_ROLE).toDouble());
    }
    if (server_upscale_check_box->isChecked()) {
        result.upscale_model_file = upscale_model_edit->text().trimmed();
    }
    result.prompt = prompt_edit->text().trimmed();
    result.negative_prompt = negative_prompt_edit->text().trimmed();
    result.server_executable = server_edit->text().trimmed();
    result.server_arguments = server_arguments_edit->text().trimmed();
    result.server_port = server_port_spin_box->value();
    parse_stable_resolution(resolution_combo_box->currentText(), result.width, result.height);
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
    const QString directory = settings.value("stable_diffusion/last_model_directory").toString();
    const QString filename = QFileDialog::getOpenFileName(this, "Select Stable Diffusion Model", directory, "Safetensors Models (*.safetensors);;All Files (*)");
    if (filename.isEmpty()) {
        return;
    }
    model_file_edit->setText(QFileInfo(filename).absoluteFilePath());
    settings.setValue("stable_diffusion/last_model_directory", QFileInfo(filename).absolutePath());
}

void StableDiffusionSettingsDialog::add_lora_item(const QString &filename, double multiplier) {
    const QString absolute_file = QFileInfo(filename).absoluteFilePath();
    for (int row = 0; row < lora_list_widget->count(); ++row) {
        if (lora_list_widget->item(row)->data(LORA_PATH_ROLE).toString() == absolute_file) {
            return;
        }
    }
    auto *item = new QListWidgetItem(lora_list_widget);
    item->setData(LORA_PATH_ROLE, absolute_file);
    item->setData(LORA_MULTIPLIER_ROLE, multiplier);
    item->setToolTip(absolute_file);
    update_lora_item_text(lora_list_widget->row(item));
}

void StableDiffusionSettingsDialog::update_lora_item_text(int row) {
    QListWidgetItem *item = lora_list_widget->item(row);
    if (item == nullptr) {
        return;
    }
    const QString filename = QFileInfo(item->data(LORA_PATH_ROLE).toString()).fileName();
    const double multiplier = item->data(LORA_MULTIPLIER_ROLE).toDouble();
    item->setText(QStringLiteral("%1  —  %2").arg(filename).arg(multiplier, 0, 'f', 3));
}

void StableDiffusionSettingsDialog::add_lora_models() {
    QSettings settings("LostSideDead", "acmx2");
    const QString directory = settings.value("stable_diffusion/last_lora_directory").toString();
    const QStringList filenames = QFileDialog::getOpenFileNames(this, "Select LoRA Models", directory, "LoRA Models (*.safetensors *.ckpt *.pt *.pth *.gguf);;All Files (*)");
    if (filenames.isEmpty()) {
        return;
    }
    for (const QString &filename : filenames) {
        add_lora_item(filename, 1.0);
    }
    settings.setValue("stable_diffusion/last_lora_directory", QFileInfo(filenames.first()).absolutePath());
    lora_list_widget->setCurrentRow(lora_list_widget->count() - 1);
}

void StableDiffusionSettingsDialog::remove_lora_models() {
    const QList<QListWidgetItem *> selected = lora_list_widget->selectedItems();
    for (QListWidgetItem *item : selected) {
        delete lora_list_widget->takeItem(lora_list_widget->row(item));
    }
    select_lora_model();
}

void StableDiffusionSettingsDialog::select_lora_model() {
    const QList<QListWidgetItem *> selected = lora_list_widget->selectedItems();
    const bool one_selected = selected.size() == 1;
    remove_lora_button->setEnabled(enable_check_box->isChecked() && !selected.isEmpty());
    lora_multiplier_spin_box->setEnabled(enable_check_box->isChecked() && one_selected);
    if (one_selected) {
        const QSignalBlocker blocker(lora_multiplier_spin_box);
        lora_multiplier_spin_box->setValue(selected.front()->data(LORA_MULTIPLIER_ROLE).toDouble());
    }
}

void StableDiffusionSettingsDialog::update_lora_multiplier(double multiplier) {
    const QList<QListWidgetItem *> selected = lora_list_widget->selectedItems();
    if (selected.size() != 1) {
        return;
    }
    QListWidgetItem *item = selected.front();
    item->setData(LORA_MULTIPLIER_ROLE, multiplier);
    update_lora_item_text(lora_list_widget->row(item));
}

void StableDiffusionSettingsDialog::browse_upscale_model() {
    QSettings settings("LostSideDead", "acmx2");
    const QString directory = settings.value("stable_diffusion/last_upscale_model_directory").toString();
    const QString filename = QFileDialog::getOpenFileName(this, "Select ESRGAN Upscale Model", directory, "Upscale Models (*.safetensors *.pth *.pt);;All Files (*)");
    if (filename.isEmpty()) {
        return;
    }
    upscale_model_edit->setText(QFileInfo(filename).absoluteFilePath());
    server_upscale_check_box->setChecked(true);
    settings.setValue("stable_diffusion/last_upscale_model_directory", QFileInfo(filename).absolutePath());
}

void StableDiffusionSettingsDialog::browse_server() {
    const QString filename = QFileDialog::getOpenFileName(this, "Select sd-server Executable", QFileInfo(server_edit->text()).absolutePath());
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
        QMessageBox::warning(this, "Stable Diffusion Model Required", "Select an existing safetensors model file.");
        return false;
    }
    QString lora_directory;
    for (int row = 0; row < lora_list_widget->count(); ++row) {
        const QString filename = lora_list_widget->item(row)->data(LORA_PATH_ROLE).toString();
        const QFileInfo file_info(filename);
        if (!file_info.isFile()) {
            QMessageBox::warning(this, "LoRA Model Not Found", QStringLiteral("The selected LoRA model does not exist:\n%1").arg(filename));
            return false;
        }
        if (lora_directory.isEmpty()) {
            lora_directory = file_info.absolutePath();
        } else if (lora_directory != file_info.absolutePath()) {
            QMessageBox::warning(this,
                                 "LoRA Folder Mismatch",
                                 "All selected LoRA models must be in the same folder. "
                                 "sd-server scans one LoRA model directory per launch.");
            return false;
        }
    }
    if (prompt_edit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "Stable Diffusion Prompt Required", "Enter an image-to-image prompt.");
        return false;
    }
    if (server_edit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "sd-server Required", "Enter sd-server or select its executable.");
        return false;
    }
    const QStringList server_arguments = QProcess::splitCommand(server_arguments_edit->text());
    constexpr int MAX_SERVER_ARGUMENTS = 256;
    constexpr int MAX_SERVER_ARGUMENT_BYTES = 8192;
    if (server_arguments.size() > MAX_SERVER_ARGUMENTS) {
        QMessageBox::warning(this, "Too Many sd-server Flags", "Extra sd-server flags are limited to 256 arguments.");
        return false;
    }
    for (const QString &argument : server_arguments) {
        if (argument.isEmpty() || argument.toUtf8().size() > MAX_SERVER_ARGUMENT_BYTES) {
            QMessageBox::warning(this,
                                 "Invalid sd-server Flag",
                                 "Each extra sd-server argument must be non-empty and no "
                                 "larger than 8192 UTF-8 bytes.");
            return false;
        }
    }
    if (server_upscale_check_box->isChecked() && !QFileInfo(upscale_model_edit->text().trimmed()).isFile()) {
        QMessageBox::warning(this, "Upscale Model Required", "Select an existing ESRGAN or RealESRGAN model.");
        return false;
    }
    int width = 0;
    int height = 0;
    if (!parse_stable_resolution(resolution_combo_box->currentText(), width, height)) {
        QMessageBox::warning(this, "Invalid Stable Diffusion Resolution", "Enter the resolution as WIDTHxHEIGHT, for example 640x384.");
        return false;
    }
    if (width < STABLE_DIMENSION_MINIMUM || width > STABLE_DIMENSION_MAXIMUM || height < STABLE_DIMENSION_MINIMUM || height > STABLE_DIMENSION_MAXIMUM || (width % 64) != 0 || (height % 64) != 0) {
        QMessageBox::warning(this,
                             "Invalid Stable Diffusion Resolution",
                             "Width and height must be multiples of 64 between 64 and 2048 "
                             "pixels.");
        return false;
    }
    resolution_combo_box->setCurrentText(resolution_text(width, height));
    if (sampler_combo_box->currentText().trimmed().isEmpty() || scheduler_combo_box->currentText().trimmed().isEmpty()) {
        QMessageBox::warning(this, "Generation Method Required", "Enter both a sampler and scheduler.");
        return false;
    }
    return true;
}

void StableDiffusionSettingsDialog::load_ui_state() {
    QSettings settings("LostSideDead", "acmx2");
    enable_check_box->setChecked(settings.value("stable_diffusion/enabled", false).toBool());
    model_file_edit->setText(settings.value("stable_diffusion/model_file").toString());
    const QStringList lora_files = settings.value("stable_diffusion/lora_files").toStringList();
    const QStringList lora_multiplier_values = settings.value("stable_diffusion/lora_multipliers").toStringList();
    for (int index = 0; index < lora_files.size(); ++index) {
        bool multiplier_ok = false;
        const double saved_multiplier = index < lora_multiplier_values.size() ? lora_multiplier_values.at(index).toDouble(&multiplier_ok) : 1.0;
        add_lora_item(lora_files.at(index), multiplier_ok ? saved_multiplier : 1.0);
    }
    prompt_edit->setText(settings.value("stable_diffusion/prompt").toString());
    negative_prompt_edit->setText(settings.value("stable_diffusion/negative_prompt").toString());
    server_edit->setText(settings.value("stable_diffusion/server", "sd-server").toString());
    server_arguments_edit->setText(settings.value("stable_diffusion/server_arguments").toString());
    server_port_spin_box->setValue(settings.value("stable_diffusion/port", 1234).toInt());
    const int width = settings.value("stable_diffusion/width", 576).toInt();
    const int height = settings.value("stable_diffusion/height", 320).toInt();
    const QString saved_resolution = settings.value("stable_diffusion/resolution", resolution_text(width, height)).toString();
    int saved_width = 0;
    int saved_height = 0;
    if (parse_stable_resolution(saved_resolution, saved_width, saved_height)) {
        resolution_combo_box->setCurrentText(resolution_text(saved_width, saved_height));
    } else {
        resolution_combo_box->setCurrentText(resolution_text(width, height));
    }
    steps_spin_box->setValue(settings.value("stable_diffusion/steps", 12).toInt());
    strength_spin_box->setValue(settings.value("stable_diffusion/strength", 0.35).toDouble());
    cfg_scale_spin_box->setValue(settings.value("stable_diffusion/cfg_scale", 5.0).toDouble());
    seed_spin_box->setValue(settings.value("stable_diffusion/seed", 1234).toInt());
    sampler_combo_box->setCurrentText(settings.value("stable_diffusion/sampler", "euler_a").toString());
    scheduler_combo_box->setCurrentText(settings.value("stable_diffusion/scheduler", "discrete").toString());
    upscale_check_box->setChecked(settings.value("stable_diffusion/upscale", false).toBool());
    upscale_model_edit->setText(settings.value("stable_diffusion/upscale_model_file").toString());
    server_upscale_check_box->setChecked(settings.value("stable_diffusion/server_upscale", false).toBool());
}

void StableDiffusionSettingsDialog::save_ui_state() {
    const StableDiffusionConfiguration current = configuration();
    QSettings settings("LostSideDead", "acmx2");
    settings.setValue("stable_diffusion/enabled", current.enabled);
    settings.setValue("stable_diffusion/model_file", current.model_file);
    settings.setValue("stable_diffusion/lora_files", current.lora_files);
    QStringList lora_multiplier_values;
    for (const double multiplier : current.lora_multipliers) {
        lora_multiplier_values.append(QString::number(multiplier, 'g', 12));
    }
    settings.setValue("stable_diffusion/lora_multipliers", lora_multiplier_values);
    settings.setValue("stable_diffusion/prompt", current.prompt);
    settings.setValue("stable_diffusion/negative_prompt", current.negative_prompt);
    settings.setValue("stable_diffusion/server", current.server_executable);
    settings.setValue("stable_diffusion/server_arguments", current.server_arguments);
    settings.setValue("stable_diffusion/port", current.server_port);
    settings.setValue("stable_diffusion/width", current.width);
    settings.setValue("stable_diffusion/height", current.height);
    settings.setValue("stable_diffusion/resolution", resolution_text(current.width, current.height));
    settings.setValue("stable_diffusion/steps", current.steps);
    settings.setValue("stable_diffusion/strength", current.strength);
    settings.setValue("stable_diffusion/cfg_scale", current.cfg_scale);
    settings.setValue("stable_diffusion/seed", current.seed);
    settings.setValue("stable_diffusion/sampler", current.sampler);
    settings.setValue("stable_diffusion/scheduler", current.scheduler);
    settings.setValue("stable_diffusion/upscale", current.upscale);
    settings.setValue("stable_diffusion/server_upscale", server_upscale_check_box->isChecked());
    settings.setValue("stable_diffusion/upscale_model_file", upscale_model_edit->text().trimmed());
    settings.sync();
}

void StableDiffusionSettingsDialog::update_enabled_state() {
    const bool enabled = enable_check_box->isChecked();
    model_file_edit->setEnabled(enabled);
    browse_model_button->setEnabled(enabled);
    prompt_edit->setEnabled(enabled);
    negative_prompt_edit->setEnabled(enabled);
    lora_list_widget->setEnabled(enabled);
    add_lora_button->setEnabled(enabled);
    remove_lora_button->setEnabled(enabled && !lora_list_widget->selectedItems().isEmpty());
    select_lora_model();
    server_edit->setEnabled(enabled);
    server_arguments_edit->setEnabled(enabled);
    browse_server_button->setEnabled(enabled);
    server_port_spin_box->setEnabled(enabled);
    resolution_combo_box->setEnabled(enabled);
    steps_spin_box->setEnabled(enabled);
    strength_spin_box->setEnabled(enabled);
    cfg_scale_spin_box->setEnabled(enabled);
    seed_spin_box->setEnabled(enabled);
    sampler_combo_box->setEnabled(enabled);
    scheduler_combo_box->setEnabled(enabled);
    upscale_check_box->setEnabled(enabled);
    server_upscale_check_box->setEnabled(enabled);
    const bool server_upscale_enabled = enabled && server_upscale_check_box->isChecked();
    upscale_model_edit->setEnabled(server_upscale_enabled);
    browse_upscale_model_button->setEnabled(server_upscale_enabled);
}
