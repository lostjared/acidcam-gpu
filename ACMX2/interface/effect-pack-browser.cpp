#include "effect-pack-browser.hpp"
#include "effect-pack-models.hpp"

#include <QComboBox>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QImageReader>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QProcess>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QSettings>
#include <QStandardPaths>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <memory>
#include <utility>

namespace {
    QString roots_key() { return QStringLiteral("effect_packs/roots"); }
    QString last_selected_key() { return QStringLiteral("effect_packs/last_selected"); }
    QString model_override_key(const QString &id) { return QStringLiteral("effect_packs/model_overrides/") + QString::fromLatin1(QCryptographicHash::hash(id.toUtf8(), QCryptographicHash::Sha256).toHex()); }

    QStringList configured_roots() {
        QSettings settings("LostSideDead", "acmx2");
        QStringList roots = settings.value(roots_key()).toStringList();
        roots.prepend(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/effect-packs"));
        roots << QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../share/acmxvk/effect-packs"));
        roots.removeDuplicates();
        return roots;
    }

    QString user_pack_root() {
        const QString path = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/effect-packs");
        return QDir().mkpath(path) ? path : QString();
    }

    QJsonObject read_pack_manifest(const QString &root) {
        QFile file(QDir(root).filePath(QStringLiteral("effect.json")));
        if (!file.open(QIODevice::ReadOnly) || file.size() > 1024 * 1024) {
            return {};
        }
        return QJsonDocument::fromJson(file.readAll()).object();
    }

    QString pack_folder_name(const QString &name) {
        QString folder = name.toLower().trimmed();
        folder.replace(QRegularExpression(QStringLiteral("[^a-z0-9_-]+")), QStringLiteral("-"));
        folder.remove(QRegularExpression(QStringLiteral("^-+|-+$")));
        return folder.left(80);
    }
} // namespace

EffectPackBrowser::EffectPackBrowser(QWidget *parent) : QDialog(parent) {
    setWindowTitle(tr("ACMXVK Effect Packs"));
    setModal(false);
    resize(860, 620);

    auto *layout = new QVBoxLayout(this);
    auto *toolbar = new QHBoxLayout;
    root_combo = new QComboBox(this);
    root_combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolbar->addWidget(root_combo);
    auto *add_button = new QPushButton(tr("Add Folder..."), this);
    auto *model_folder_button = new QPushButton(tr("Model Folder..."), this);
    auto *remove_button = new QPushButton(tr("Remove Folder"), this);
    refresh_button = new QPushButton(tr("Refresh"), this);
    toolbar->addWidget(add_button);
    toolbar->addWidget(model_folder_button);
    toolbar->addWidget(remove_button);
    toolbar->addWidget(refresh_button);
    layout->addLayout(toolbar);

    pack_list = new QListWidget(this);
    pack_list->setViewMode(QListView::IconMode);
    pack_list->setResizeMode(QListView::Adjust);
    pack_list->setMovement(QListView::Static);
    pack_list->setWrapping(true);
    pack_list->setSpacing(10);
    pack_list->setIconSize(QSize(128, 96));
    pack_list->setGridSize(QSize(176, 144));
    pack_list->setWordWrap(true);
    layout->addWidget(pack_list, 1);

    details = new QLabel(this);
    details->setWordWrap(true);
    layout->addWidget(details);
    auto *transfer_bar = new QHBoxLayout;
    create_button = new QPushButton(tr("Create from Current..."), this);
    save_button = new QPushButton(tr("Save Pack As..."), this);
    export_button = new QPushButton(tr("Export..."), this);
    import_button = new QPushButton(tr("Import..."), this);
    transfer_bar->addWidget(create_button);
    transfer_bar->addWidget(save_button);
    transfer_bar->addWidget(export_button);
    transfer_bar->addWidget(import_button);
    transfer_bar->addStretch();
    layout->addLayout(transfer_bar);
    progress = new QProgressBar(this);
    progress->setVisible(false);
    layout->addWidget(progress);
    auto *footer = new QHBoxLayout;
    status = new QLabel(tr("Choose a pack to build and activate it."), this);
    footer->addWidget(status, 1);
    build_button = new QPushButton(tr("Build && Activate"), this);
    footer->addWidget(build_button);
    controls_button = new QPushButton(tr("Controls..."), this);
    footer->addWidget(controls_button);
    choose_model_button = new QPushButton(tr("Choose Dream Model..."), this);
    footer->addWidget(choose_model_button);
    library_button = new QPushButton(tr("Use Shader Library"), this);
    footer->addWidget(library_button);
    auto *close_button = new QPushButton(tr("Close"), this);
    footer->addWidget(close_button);
    layout->addLayout(footer);

    build_process = new QProcess(this);
    build_process->setProcessChannelMode(QProcess::MergedChannels);
    connect(build_process, &QProcess::readyRead, this, [this]() {
        build_output.append(build_process->readAll());
        const QRegularExpression expression(QStringLiteral("effect pack build progress: (\\d+)/(\\d+)"));
        auto matches = expression.globalMatch(QString::fromUtf8(build_output));
        int current = 0;
        int total = 0;
        while (matches.hasNext()) {
            const QRegularExpressionMatch match = matches.next();
            current = match.captured(1).toInt();
            total = match.captured(2).toInt();
        }
        if (total > 0) {
            progress->setRange(0, total);
            progress->setValue(current);
            status->setText(tr("Building effect pack: %1/%2 passes").arg(current).arg(total));
        }
    });
    connect(build_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus exit_status) {
        build_output.append(build_process->readAll());
        const QString output = QString::fromUtf8(build_output).trimmed();
        if (pending_row >= 0 && pending_row < packs.size()) {
            packs[pending_row].status = code == 0 && exit_status == QProcess::NormalExit ? tr("Compiled") : tr("Build failed");
            populate();
            if (code == 0 && exit_status == QProcess::NormalExit) {
                QSettings settings("LostSideDead", "acmx2");
                settings.setValue(last_selected_key(), packs[pending_row].manifest);
                if (!control_dialog) {
                    control_dialog = new EffectPackControls(this);
                    connect(control_dialog, &EffectPackControls::values_changed, this, [this](const QVector<EffectPackUniformValue> &values) {
                        if (!active_manifest.isEmpty() && control_manifest == active_manifest) {
                            active_values = values;
                            if (project_manifest == active_manifest) {
                                project_values = control_dialog->project_values();
                            }
                            emit uniform_values_changed(values);
                        }
                    });
                }
                const PackEntry &pack = packs[pending_row];
                const bool project_pack = project_manifest == pack.manifest;
                control_dialog->set_pack(pack.id, pack.name, pack.controls, !project_pack);
                if (project_pack) {
                    control_dialog->set_project_values(project_values);
                }
                control_manifest = pack.manifest;
                active_manifest = pack.manifest;
                active_values = control_dialog->values();
                emit activation_requested(pack.manifest, active_values, pack.dream);
                if (!pack.controls.isEmpty() && has_active_pack()) {
                    control_dialog->show();
                    control_dialog->raise();
                }
                status->setText(tr("Selected %1; it will activate now or on the next ACMXVK launch.").arg(packs[pending_row].name));
            } else {
                status->setText(output.isEmpty() ? tr("Effect pack build failed.") : output.left(500));
            }
        }
        pending_row = -1;
        progress->setVisible(false);
        set_busy(false);
    });
    connect(build_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) {
            status->setText(tr("Could not start ACMXVK: %1").arg(build_process->errorString()));
            pending_row = -1;
            progress->setVisible(false);
            set_busy(false);
        }
    });
    connect(add_button, &QPushButton::clicked, this, [this]() {
        const QString folder = QFileDialog::getExistingDirectory(this, tr("Add Effect Pack Folder"));
        if (folder.isEmpty()) {
            return;
        }
        QSettings settings("LostSideDead", "acmx2");
        QStringList roots = settings.value(roots_key()).toStringList();
        if (!roots.contains(folder)) {
            roots << folder;
            settings.setValue(roots_key(), roots);
        }
        refresh();
    });
    connect(model_folder_button, &QPushButton::clicked, this, [this]() {
        const QString folder = QFileDialog::getExistingDirectory(this, tr("Add Deep Dream Model Folder"));
        if (folder.isEmpty()) {
            return;
        }
        QSettings settings("LostSideDead", "acmx2");
        QStringList roots = settings.value(QStringLiteral("effect_packs/model_roots")).toStringList();
        if (!roots.contains(folder)) {
            roots << folder;
            settings.setValue(QStringLiteral("effect_packs/model_roots"), roots);
        }
        refresh();
    });
    connect(remove_button, &QPushButton::clicked, this, [this]() {
        QSettings settings("LostSideDead", "acmx2");
        QStringList roots = settings.value(roots_key()).toStringList();
        roots.removeAll(root_combo->currentText());
        settings.setValue(roots_key(), roots);
        refresh();
    });
    connect(refresh_button, &QPushButton::clicked, this, &EffectPackBrowser::refresh);
    connect(create_button, &QPushButton::clicked, this, &EffectPackBrowser::create_from_session);
    connect(save_button, &QPushButton::clicked, this, &EffectPackBrowser::save_pack_as);
    connect(export_button, &QPushButton::clicked, this, &EffectPackBrowser::export_pack);
    connect(import_button, &QPushButton::clicked, this, &EffectPackBrowser::import_pack);
    connect(pack_list, &QListWidget::currentRowChanged, this, &EffectPackBrowser::update_details);
    connect(pack_list, &QListWidget::itemClicked, this, [this](QListWidgetItem *item) { activate(pack_list->row(item)); });
    connect(build_button, &QPushButton::clicked, this, [this]() { start_build(pack_list->currentRow()); });
    connect(controls_button, &QPushButton::clicked, this, [this]() {
        const int row = pack_list->currentRow();
        if (row < 0 || row >= packs.size() || !packs[row].valid) {
            return;
        }
        if (!control_dialog) {
            control_dialog = new EffectPackControls(this);
            connect(control_dialog, &EffectPackControls::values_changed, this, [this](const QVector<EffectPackUniformValue> &values) {
                if (!active_manifest.isEmpty() && control_manifest == active_manifest) {
                    active_values = values;
                    if (project_manifest == active_manifest) {
                        project_values = control_dialog->project_values();
                    }
                    emit uniform_values_changed(values);
                }
            });
        }
        const bool project_pack = project_manifest == packs[row].manifest;
        control_dialog->set_pack(packs[row].id, packs[row].name, packs[row].controls, !project_pack);
        if (project_pack) {
            control_dialog->set_project_values(project_values);
        }
        control_manifest = packs[row].manifest;
        control_dialog->show();
        control_dialog->raise();
    });
    connect(choose_model_button, &QPushButton::clicked, this, [this]() {
        const int row = pack_list->currentRow();
        if (row < 0 || row >= packs.size() || packs[row].dream_model_id.isEmpty()) {
            return;
        }
        const QString model = QFileDialog::getOpenFileName(this, tr("Choose Local Deep Dream Model"), QString(), tr("TorchScript models (*.pt *.torchscript);;All files (*)"));
        if (model.isEmpty()) {
            return;
        }
        QSettings settings("LostSideDead", "acmx2");
        settings.setValue(model_override_key(packs[row].id), QFileInfo(model).canonicalFilePath());
        refresh();
    });
    connect(library_button, &QPushButton::clicked, this, [this]() {
        clear_active_pack();
        emit activation_requested(QString(), {}, {});
        status->setText(tr("Requested normal shader library mode."));
    });
    connect(close_button, &QPushButton::clicked, this, &QDialog::hide);
}

void EffectPackBrowser::set_build_tools(const QString &executable, const QString &compiler, int parallel_jobs) {
    executable_path = executable;
    compiler_path = compiler;
    jobs = std::clamp(parallel_jobs, 1, 64);
}

void EffectPackBrowser::set_runtime_context(bool dream, const QString &configured_model, bool audio, bool midi, bool midi_profile) {
    dream_supported = dream;
    configured_dream_model = configured_model;
    audio_supported = audio;
    midi_supported = midi;
    midi_profile_selected = midi_profile;
}

void EffectPackBrowser::set_session_snapshot(const QString &source_root, const QJsonObject &manifest) {
    session_source_root = source_root;
    session_manifest = manifest;
    create_button->setEnabled(!source_root.isEmpty() && !manifest.isEmpty());
}

bool EffectPackBrowser::has_active_pack() const { return !active_manifest.isEmpty(); }

bool EffectPackBrowser::has_project_pack() const { return !project_manifest.isEmpty(); }

void EffectPackBrowser::clear_active_pack() {
    active_manifest.clear();
    active_values.clear();
    if (control_dialog) {
        control_dialog->hide();
    }
}

void EffectPackBrowser::clear_project_pack() {
    clear_active_pack();
    project_manifest.clear();
    project_values = {};
    project_model_file.clear();
}

EffectPackProjectState EffectPackBrowser::project_state() const {
    EffectPackProjectState state;
    if (active_manifest.isEmpty()) {
        return state;
    }
    for (const PackEntry &pack : packs) {
        if (pack.manifest != active_manifest) {
            continue;
        }
        state.manifest_path = pack.manifest;
        state.id = pack.id;
        state.dream_model_file = pack.dream.enabled ? pack.dream.model_file : QString();
        if (control_dialog && control_manifest == active_manifest) {
            state.values = control_dialog->project_values();
        } else {
            for (int index = 0; index < pack.controls.size() && index < active_values.size(); ++index) {
                state.values.insert(pack.controls[index].id, active_values[index].value);
            }
        }
        return state;
    }
    return state;
}

bool EffectPackBrowser::restore_project_pack(const EffectPackProjectState &state, QString &error) {
    if (!acmx2::validate_effect_pack_project_cache(state.manifest_path, state.id, error)) {
        return false;
    }
    const QString manifest = QFileInfo(state.manifest_path).canonicalFilePath();
    const QStringList model_roots = acmx2::effect_pack_model_roots(state.dream_model_file.isEmpty() ? configured_dream_model : state.dream_model_file);
    const QVector<PackEntry> found = discover({QFileInfo(manifest).absolutePath()}, model_roots, dream_supported, audio_supported, midi_supported, midi_profile_selected, state.dream_model_file);
    auto match = std::find_if(found.cbegin(), found.cend(), [&manifest](const PackEntry &pack) { return pack.manifest == manifest; });
    if (match == found.cend() || !match->valid || match->id != state.id) {
        error = match == found.cend() ? tr("Project effect pack is missing.") : tr("Project effect pack is unavailable: %1").arg(match->status);
        return false;
    }
    for (auto it = state.values.constBegin(); it != state.values.constEnd(); ++it) {
        const auto control = std::find_if(match->controls.cbegin(), match->controls.cend(), [&it](const EffectPackControlDefinition &item) { return item.id == it.key(); });
        if (control == match->controls.cend() || !it.value().isDouble() || !std::isfinite(it.value().toDouble()) || it.value().toDouble() < control->minimum || it.value().toDouble() > control->maximum) {
            error = tr("Project effect-pack control override is invalid: %1").arg(it.key());
            return false;
        }
    }
    clear_project_pack();
    project_manifest = manifest;
    project_values = state.values;
    project_model_file = state.dream_model_file;
    auto existing = std::find_if(packs.begin(), packs.end(), [&manifest](const PackEntry &pack) { return pack.manifest == manifest; });
    if (existing == packs.end()) {
        packs.push_back(*match);
        existing = std::prev(packs.end());
    } else {
        *existing = *match;
    }
    if (!control_dialog) {
        control_dialog = new EffectPackControls(this);
        connect(control_dialog, &EffectPackControls::values_changed, this, [this](const QVector<EffectPackUniformValue> &values) {
            if (!active_manifest.isEmpty() && control_manifest == active_manifest) {
                active_values = values;
                if (project_manifest == active_manifest) {
                    project_values = control_dialog->project_values();
                }
                emit uniform_values_changed(values);
            }
        });
    }
    control_dialog->set_pack(existing->id, existing->name, existing->controls, false);
    control_dialog->set_project_values(project_values);
    control_manifest = manifest;
    active_manifest = manifest;
    active_values = control_dialog->values();
    emit activation_requested(manifest, active_values, existing->dream);
    status->setText(tr("Restored project effect pack: %1").arg(existing->name));
    return true;
}

bool EffectPackBrowser::queue_project_pack_for_build(const EffectPackProjectState &state, QString &error) {
    const QString manifest = QFileInfo(state.manifest_path).canonicalFilePath();
    if (manifest.isEmpty()) {
        error = tr("Project effect-pack manifest is missing.");
        return false;
    }
    const QStringList model_roots = acmx2::effect_pack_model_roots(state.dream_model_file.isEmpty() ? configured_dream_model : state.dream_model_file);
    const QVector<PackEntry> found = discover({QFileInfo(manifest).absolutePath()}, model_roots, dream_supported, audio_supported, midi_supported, midi_profile_selected, state.dream_model_file);
    const auto match = std::find_if(found.cbegin(), found.cend(), [&manifest](const PackEntry &pack) { return pack.manifest == manifest; });
    if (match == found.cend() || match->id != state.id || !match->valid) {
        error = tr("Project effect-pack source is missing or invalid.");
        return false;
    }
    for (auto it = state.values.constBegin(); it != state.values.constEnd(); ++it) {
        const auto control = std::find_if(match->controls.cbegin(), match->controls.cend(), [&it](const EffectPackControlDefinition &item) { return item.id == it.key(); });
        if (control == match->controls.cend() || !it.value().isDouble() || !std::isfinite(it.value().toDouble()) || it.value().toDouble() < control->minimum || it.value().toDouble() > control->maximum) {
            error = tr("Project effect-pack control override is invalid: %1").arg(it.key());
            return false;
        }
    }
    clear_project_pack();
    project_manifest = manifest;
    project_values = state.values;
    project_model_file = state.dream_model_file;
    auto existing = std::find_if(packs.begin(), packs.end(), [&manifest](const PackEntry &pack) { return pack.manifest == manifest; });
    if (existing == packs.end()) {
        packs.push_back(*match);
    } else {
        *existing = *match;
    }
    populate();
    status->setText(tr("Project effect pack needs a rebuild. Select it and click Build & Activate."));
    return true;
}

QStringList EffectPackBrowser::search_roots() const { return configured_roots(); }

QVector<EffectPackBrowser::PackEntry> EffectPackBrowser::discover(const QStringList &roots, const QStringList &model_roots, bool dream_supported, bool audio_supported, bool midi_supported, bool midi_profile_selected, const QString &model_override) {
    QVector<PackEntry> entries;
    QSettings model_settings("LostSideDead", "acmx2");
    QSet<QString> paths;
    QSet<QString> ids;
    for (const QString &root : roots) {
        if (!QFileInfo(root).isDir()) {
            continue;
        }
        QStringList manifests;
        const QString direct = QDir(root).filePath(QStringLiteral("effect.json"));
        if (QFileInfo(direct).isFile()) {
            manifests << direct;
        }
        QDirIterator iterator(root, {QStringLiteral("effect.json")}, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
        while (iterator.hasNext() && manifests.size() < 4096) {
            manifests << iterator.next();
        }
        for (const QString &manifest : manifests) {
            const QString canonical = QFileInfo(manifest).canonicalFilePath();
            if (canonical.isEmpty() || paths.contains(canonical)) {
                continue;
            }
            paths.insert(canonical);
            if (entries.size() >= 4096) {
                return entries;
            }
            PackEntry entry;
            entry.manifest = canonical;
            entry.name = QFileInfo(canonical).dir().dirName();
            QFile file(canonical);
            if (!file.open(QIODevice::ReadOnly) || file.size() > 1024 * 1024) {
                entry.status = QObject::tr("Unreadable manifest");
                entries.push_back(std::move(entry));
                continue;
            }
            QJsonParseError parse_error;
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
            const QJsonObject object = document.object();
            if (parse_error.error != QJsonParseError::NoError || !document.isObject() || object.value(QStringLiteral("format")).toString() != QStringLiteral("acmxvk-effect-pack") || object.value(QStringLiteral("version")).toInt() != 1 || !object.value(QStringLiteral("passes")).isArray()) {
                entry.status = QObject::tr("Invalid manifest");
                entries.push_back(std::move(entry));
                continue;
            }
            entry.id = object.value(QStringLiteral("id")).toString();
            entry.name = object.value(QStringLiteral("name")).toString(entry.name);
            entry.description = object.value(QStringLiteral("description")).toString();
            if (entry.id.isEmpty() || ids.contains(entry.id)) {
                entry.status = QObject::tr("Missing or duplicate ID");
                entries.push_back(std::move(entry));
                continue;
            }
            ids.insert(entry.id);
            entry.valid = true;
            const QJsonArray controls = object.value(QStringLiteral("controls")).toArray();
            if (controls.size() > 64) {
                entry.valid = false;
                entry.status = QObject::tr("Too many controls");
            }
            QSet<QString> control_ids;
            QSet<QString> uniform_names;
            for (const QJsonValue &item : controls) {
                const QJsonObject data = item.toObject();
                EffectPackControlDefinition control;
                control.id = data.value(QStringLiteral("id")).toString();
                control.label = data.value(QStringLiteral("label")).toString();
                control.uniform = data.value(QStringLiteral("uniform")).toString();
                control.minimum = data.value(QStringLiteral("minimum")).toDouble();
                control.maximum = data.value(QStringLiteral("maximum")).toDouble();
                control.step = data.value(QStringLiteral("step")).toDouble();
                control.default_value = data.value(QStringLiteral("default")).toDouble();
                if (control.id.isEmpty() || control.label.isEmpty() || control.uniform.isEmpty() || control_ids.contains(control.id) || uniform_names.contains(control.uniform) || !std::isfinite(control.minimum) || !std::isfinite(control.maximum) || !std::isfinite(control.step) || !std::isfinite(control.default_value) || control.minimum >= control.maximum || control.step <= 0.0 || control.default_value < control.minimum || control.default_value > control.maximum) {
                    entry.valid = false;
                    entry.status = QObject::tr("Invalid controls");
                    break;
                }
                control_ids.insert(control.id);
                uniform_names.insert(control.uniform);
                entry.controls.push_back(std::move(control));
            }
            const QJsonObject dream = object.value(QStringLiteral("deep_dream")).toObject();
            if (!dream.isEmpty()) {
                entry.dream_declared = true;
                entry.dream.enabled = dream.value(QStringLiteral("enabled")).toBool(false);
                entry.dream_model_id = dream.value(QStringLiteral("model")).toString();
                entry.dream.layer = dream.value(QStringLiteral("layer")).toString();
                entry.dream.channel = dream.value(QStringLiteral("channel")).toInt(-1);
                entry.dream.iterations = dream.value(QStringLiteral("iterations")).toInt(1);
                entry.dream.strength = dream.value(QStringLiteral("strength")).toDouble(0.05);
                entry.dream.feedback = dream.value(QStringLiteral("feedback")).toDouble(0.9);
                entry.dream.zoom = dream.value(QStringLiteral("zoom")).toDouble(1.01);
                entry.dream.rotation = dream.value(QStringLiteral("rotation")).toDouble(0.1);
                entry.dream.maximum_dimension = dream.value(QStringLiteral("working_size")).toInt(512);
                entry.dream.fp16 = dream.value(QStringLiteral("fp16")).toBool(false);
                entry.dream.octaves = dream.value(QStringLiteral("octaves")).toInt(1);
                entry.dream.octave_scale = dream.value(QStringLiteral("octave_scale")).toDouble(1.4);
                entry.dream.jitter = dream.value(QStringLiteral("jitter")).toInt(0);
                entry.dream.smoothing = dream.value(QStringLiteral("smoothing")).toInt(0);
                entry.dream.gpu_filter_first = dream.value(QStringLiteral("gpu_filter_before_dream")).toBool(false);
                if (!entry.dream_model_id.isEmpty()) {
                    entry.dream.model_file = acmx2::resolve_effect_pack_model(entry.dream_model_id, model_roots, model_override.isEmpty() ? model_settings.value(model_override_key(entry.id)).toString() : model_override);
                }
                if (entry.dream.enabled && !dream_supported) {
                    entry.valid = false;
                    entry.status = QObject::tr("Deep Dream unavailable in this build");
                } else if (entry.dream.enabled && entry.dream.model_file.isEmpty()) {
                    entry.valid = false;
                    entry.status = QObject::tr("Dream model missing: %1").arg(entry.dream_model_id);
                }
            }
            const QString icon_name = object.value(QStringLiteral("icon")).toString();
            if (!icon_name.isEmpty() && !QDir::isAbsolutePath(icon_name) && !icon_name.split('/').contains(QStringLiteral(".."))) {
                const QFileInfo icon_file(QFileInfo(canonical).dir().filePath(icon_name));
                if (icon_file.isFile() && icon_file.size() <= 4 * 1024 * 1024) {
                    QImageReader reader(icon_file.filePath());
                    reader.setAutoTransform(true);
                    const QSize original = reader.size();
                    if (original.isValid() && original.width() <= 4096 && original.height() <= 4096) {
                        reader.setScaledSize(original.scaled(128, 96, Qt::KeepAspectRatio));
                        entry.icon = reader.read();
                    }
                }
            }
            bool cached = true;
            const QDir pack_root = QFileInfo(canonical).dir();
            const QJsonArray passes = object.value(QStringLiteral("passes")).toArray();
            for (const QJsonValue &pass : passes) {
                const QString source_name = pass.toString();
                if (source_name.isEmpty() || QDir::isAbsolutePath(source_name) || source_name.split('/').contains(QStringLiteral(".."))) {
                    cached = false;
                    entry.valid = false;
                    entry.status = QObject::tr("Invalid pass path");
                    break;
                }
                const QFileInfo source(pack_root.filePath(source_name));
                const QFileInfo output(pack_root.filePath(QStringLiteral(".acmxvk-build/") + source_name + (source_name.endsWith(QStringLiteral(".spv")) ? QString() : QStringLiteral(".spv"))));
                if (!source.isFile() || !output.isFile() || output.lastModified() < source.lastModified() || output.lastModified() < QFileInfo(canonical).lastModified()) {
                    cached = false;
                }
            }
            if (cached) {
                QFile cache_file(pack_root.filePath(QStringLiteral(".acmxvk-build/effect-cache.json")));
                if (!cache_file.open(QIODevice::ReadOnly) || cache_file.size() > 1024 * 1024) {
                    cached = false;
                } else {
                    const QJsonObject metadata = QJsonDocument::fromJson(cache_file.readAll()).object();
                    cached = metadata.value(QStringLiteral("format")).toString() == QStringLiteral("acmxvk-effect-cache") && metadata.value(QStringLiteral("version")).toInt() == 2 && metadata.value(QStringLiteral("pack_id")).toString() == entry.id && metadata.value(QStringLiteral("shader_abi")).toString() == QStringLiteral("acmxvk-effect-abi-1") && metadata.value(QStringLiteral("vulkan_target")).toString() == QStringLiteral("vulkan1.0");
                }
            }
            if (entry.status.isEmpty()) {
                entry.status = cached ? QObject::tr("Compiled cache present") : QObject::tr("Needs build");
            }
            if (entry.dream_declared && !entry.dream.enabled && !entry.dream_model_id.isEmpty() && entry.dream.model_file.isEmpty()) {
                entry.status += QObject::tr(" • optional Dream model missing");
            }
            if (!object.value(QStringLiteral("audio_mappings")).toArray().isEmpty() && !audio_supported) {
                entry.status += QObject::tr(" • audio mappings inactive (no audio support)");
            }
            if (!object.value(QStringLiteral("midi_mappings")).toArray().isEmpty()) {
                if (!midi_supported) {
                    entry.status += QObject::tr(" • MIDI mappings inactive (no MIDI support)");
                } else if (!midi_profile_selected) {
                    entry.status += QObject::tr(" • MIDI mappings need a controller profile");
                }
            }
            if (entry.icon.isNull()) {
                entry.status += QObject::tr(" • no icon");
            }
            entries.push_back(std::move(entry));
        }
    }
    std::sort(entries.begin(), entries.end(), [](const PackEntry &left, const PackEntry &right) { return left.name.localeAwareCompare(right.name) < 0; });
    return entries;
}

void EffectPackBrowser::refresh() {
    if (scanning || transferring || build_process->state() != QProcess::NotRunning) {
        return;
    }
    root_combo->clear();
    root_combo->addItems(search_roots());
    scanning = true;
    set_busy(true);
    status->setText(tr("Discovering effect packs..."));
    auto *watcher = new QFutureWatcher<QVector<PackEntry>>(this);
    connect(watcher, &QFutureWatcher<QVector<PackEntry>>::finished, this, [this, watcher]() {
        packs = watcher->result();
        watcher->deleteLater();
        scanning = false;
        if (!project_manifest.isEmpty() && std::none_of(packs.cbegin(), packs.cend(), [this](const PackEntry &pack) { return pack.manifest == project_manifest; })) {
            const QStringList model_roots = acmx2::effect_pack_model_roots(project_model_file.isEmpty() ? configured_dream_model : project_model_file);
            const QVector<PackEntry> project_packs = discover({QFileInfo(project_manifest).absolutePath()}, model_roots, dream_supported, audio_supported, midi_supported, midi_profile_selected, project_model_file);
            for (const PackEntry &pack : project_packs) {
                if (pack.manifest == project_manifest) {
                    packs.push_back(pack);
                    break;
                }
            }
        }
        populate();
        status->setText(transfer_message.isEmpty() ? tr("%1 effect packs found. Click an icon to activate.").arg(packs.size()) : transfer_message);
        transfer_message.clear();
        set_busy(false);
    });
    watcher->setFuture(QtConcurrent::run([roots = search_roots(), models = acmx2::effect_pack_model_roots(configured_dream_model), dream = dream_supported, audio = audio_supported, midi = midi_supported, profile = midi_profile_selected]() { return discover(roots, models, dream, audio, midi, profile); }));
}

void EffectPackBrowser::populate() {
    QSettings settings("LostSideDead", "acmx2");
    const QString selected = pack_list->currentItem() ? pack_list->currentItem()->data(Qt::UserRole).toString() : settings.value(last_selected_key()).toString();
    pack_list->clear();
    for (const PackEntry &pack : packs) {
        const QIcon icon = pack.icon.isNull() ? style()->standardIcon(QStyle::SP_FileIcon) : QIcon(QPixmap::fromImage(pack.icon));
        auto *item = new QListWidgetItem(icon, pack.name + QStringLiteral("\n") + pack.status, pack_list);
        item->setData(Qt::UserRole, pack.manifest);
        item->setToolTip(pack.description + QStringLiteral("\n") + pack.manifest);
        if (!pack.valid) {
            item->setForeground(palette().color(QPalette::Disabled, QPalette::Text));
        }
        if (pack.manifest == selected) {
            pack_list->setCurrentItem(item);
        }
    }
    update_details(pack_list->currentRow());
}

void EffectPackBrowser::update_details(int row) {
    const bool selected = row >= 0 && row < packs.size();
    build_button->setEnabled(selected && packs[row].valid && build_process->state() == QProcess::NotRunning);
    controls_button->setEnabled(selected && packs[row].valid && !packs[row].controls.isEmpty());
    choose_model_button->setEnabled(selected && !packs[row].dream_model_id.isEmpty());
    details->setText(selected ? tr("%1 — %2\n%3").arg(packs[row].name, packs[row].status, packs[row].description) : QString());
}

void EffectPackBrowser::activate(int row) { start_build(row); }

void EffectPackBrowser::start_build(int row) {
    if (row < 0 || row >= packs.size() || !packs[row].valid || scanning || build_process->state() != QProcess::NotRunning) {
        return;
    }
    if (executable_path.isEmpty() || compiler_path.isEmpty()) {
        status->setText(tr("Set an ACMXVK executable and glslc compiler before building effect packs."));
        return;
    }
    pending_row = row;
    build_output.clear();
    progress->setRange(0, 0);
    progress->setVisible(true);
    status->setText(tr("Building %1...").arg(packs[row].name));
    set_busy(true);
    build_process->start(executable_path, {QStringLiteral("--build-effect-pack"), packs[row].manifest, QStringLiteral("--glslc"), compiler_path, QStringLiteral("--parallel"), QString::number(jobs)});
}

void EffectPackBrowser::set_busy(bool busy) {
    refresh_button->setEnabled(!busy);
    pack_list->setEnabled(!busy);
    const int row = pack_list->currentRow();
    build_button->setEnabled(!busy && row >= 0 && row < packs.size() && packs[row].valid);
    controls_button->setEnabled(!busy && row >= 0 && row < packs.size() && packs[row].valid && !packs[row].controls.isEmpty());
    choose_model_button->setEnabled(!busy && row >= 0 && row < packs.size() && !packs[row].dream_model_id.isEmpty());
    library_button->setEnabled(!busy);
    create_button->setEnabled(!busy && !session_manifest.isEmpty());
    save_button->setEnabled(!busy && row >= 0 && row < packs.size() && packs[row].valid);
    export_button->setEnabled(!busy && row >= 0 && row < packs.size() && packs[row].valid);
    import_button->setEnabled(!busy);
}

void EffectPackBrowser::start_transfer(const acmx2::EffectPackTransferRequest &request) {
    if (scanning || transferring || build_process->state() != QProcess::NotRunning) {
        return;
    }
    transferring = true;
    set_busy(true);
    auto *dialog = new QProgressDialog(tr("Copying effect-pack resources..."), QString(), 0, 0, this);
    dialog->setWindowTitle(tr("Effect Pack Transfer"));
    dialog->setCancelButton(nullptr);
    dialog->setMinimumDuration(0);
    dialog->show();
    auto current = std::make_shared<std::atomic<int>>(0);
    auto total = std::make_shared<std::atomic<int>>(0);
    auto *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [dialog, current, total]() {
        const int maximum = total->load();
        if (maximum > 0) {
            dialog->setRange(0, maximum);
            dialog->setValue(current->load());
        }
    });
    timer->start(100);
    auto *watcher = new QFutureWatcher<acmx2::EffectPackTransferResult>(this);
    connect(watcher, &QFutureWatcher<acmx2::EffectPackTransferResult>::finished, this, [this, watcher, timer, dialog]() {
        const acmx2::EffectPackTransferResult result = watcher->result();
        watcher->deleteLater();
        timer->stop();
        timer->deleteLater();
        dialog->close();
        dialog->deleteLater();
        transferring = false;
        transfer_message = result.success ? tr("Effect pack saved to %1").arg(result.destination) : tr("Effect pack transfer failed: %1").arg(result.error);
        refresh();
    });
    watcher->setFuture(QtConcurrent::run([request, current, total]() {
        return acmx2::transfer_effect_pack(request, [current, total](int value, int maximum) {
            total->store(maximum);
            current->store(value);
        });
    }));
}

void EffectPackBrowser::create_from_session() {
    if (!active_manifest.isEmpty()) {
        for (int row = 0; row < packs.size(); ++row) {
            if (packs[row].manifest == active_manifest) {
                pack_list->setCurrentRow(row);
                save_pack_as();
                return;
            }
        }
    }
    if (session_manifest.isEmpty() || session_source_root.isEmpty()) {
        status->setText(tr("Choose a source shader library and shader before creating a pack."));
        return;
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(this, tr("Create Effect Pack"), tr("Pack name:"), QLineEdit::Normal, tr("My Effect"), &accepted).trimmed();
    if (!accepted || name.isEmpty()) {
        return;
    }
    const QString folder = pack_folder_name(name);
    const QString destination = user_pack_root();
    if (folder.isEmpty() || destination.isEmpty()) {
        status->setText(tr("A portable pack name or writable user pack folder is required."));
        return;
    }
    QJsonObject manifest = session_manifest;
    manifest.insert(QStringLiteral("id"), QStringLiteral("user.") + folder);
    manifest.insert(QStringLiteral("name"), name);
    manifest.insert(QStringLiteral("description"), tr("Created from an ACMXVK interface setup."));
    const QString icon = QFileDialog::getOpenFileName(this, tr("Optional Pack Icon — Cancel for None"), QString(), tr("Images (*.png *.webp *.jpg *.jpeg)"));
    acmx2::EffectPackTransferRequest request{session_source_root, destination, folder, manifest};
    request.external_icon = icon;
    request.infer_requirements = true;
    request.assign_new_id = true;
    start_transfer(request);
}

void EffectPackBrowser::save_pack_as() {
    const int row = pack_list->currentRow();
    if (row < 0 || row >= packs.size() || !packs[row].valid) {
        return;
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(this, tr("Save Effect Pack As"), tr("New pack name:"), QLineEdit::Normal, packs[row].name + tr(" Copy"), &accepted).trimmed();
    if (!accepted || name.isEmpty()) {
        return;
    }
    const QString folder = pack_folder_name(name);
    const QString destination = user_pack_root();
    if (folder.isEmpty() || destination.isEmpty()) {
        status->setText(tr("A portable pack name or writable user pack folder is required."));
        return;
    }
    QJsonObject manifest = read_pack_manifest(QFileInfo(packs[row].manifest).absolutePath());
    if (manifest.isEmpty()) {
        status->setText(tr("Cannot read the selected pack manifest."));
        return;
    }
    EffectPackControls current;
    current.set_pack(packs[row].id, packs[row].name, packs[row].controls);
    const QVector<EffectPackUniformValue> values = current.values();
    QJsonArray controls = manifest.value(QStringLiteral("controls")).toArray();
    for (int index = 0; index < controls.size(); ++index) {
        QJsonObject control = controls[index].toObject();
        for (const EffectPackUniformValue &value : values) {
            if (control.value(QStringLiteral("uniform")).toString() == value.name) {
                control.insert(QStringLiteral("default"), value.value);
                break;
            }
        }
        controls[index] = control;
    }
    manifest.insert(QStringLiteral("controls"), controls);
    manifest.insert(QStringLiteral("name"), name);
    acmx2::EffectPackTransferRequest request{QFileInfo(packs[row].manifest).absolutePath(), destination, folder, manifest};
    request.assign_new_id = true;
    start_transfer(request);
}

void EffectPackBrowser::export_pack() {
    const int row = pack_list->currentRow();
    if (row < 0 || row >= packs.size() || !packs[row].valid) {
        return;
    }
    const QString destination = QFileDialog::getExistingDirectory(this, tr("Export Effect Pack To Folder"));
    if (destination.isEmpty()) {
        return;
    }
    const QString source = QFileInfo(packs[row].manifest).absolutePath();
    acmx2::EffectPackTransferRequest request{source, destination, pack_folder_name(QFileInfo(source).fileName()), read_pack_manifest(source)};
    start_transfer(request);
}

void EffectPackBrowser::import_pack() {
    const QString source = QFileDialog::getExistingDirectory(this, tr("Import Effect Pack Folder"));
    if (source.isEmpty()) {
        return;
    }
    const QString destination = user_pack_root();
    const QJsonObject manifest = read_pack_manifest(source);
    if (destination.isEmpty() || manifest.isEmpty()) {
        status->setText(tr("Import requires a readable effect.json and a writable user pack folder."));
        return;
    }
    acmx2::EffectPackTransferRequest request{source, destination, pack_folder_name(QFileInfo(source).fileName()), manifest};
    for (const PackEntry &pack : packs) {
        if (pack.id == manifest.value(QStringLiteral("id")).toString()) {
            request.assign_new_id = true;
            break;
        }
    }
    start_transfer(request);
}
