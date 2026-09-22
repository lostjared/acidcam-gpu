#include "main_window.hpp"
#include "audio-window.hpp"
#include "custom-uniforms.hpp"
#include "custom_style.hpp"
#include "deep-dream-settings.hpp"
#include "effect-pack-browser.hpp"
#include "effect-pack-project.hpp"
#include "find-shader.hpp"
#include "library-builder.hpp"
#include "metadata-viewer.hpp"
#include "settings.hpp"
#include "shader-manifest.hpp"
#include "stable-diffusion-settings.hpp"
#include "uniform-reference.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QColorDialog>
#include <QComboBox>
#include <QDataStream>
#include <QDateTime>
#include <QDebug>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QFutureWatcher>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QIcon>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QLocale>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTabWidget>
#include <QTextStream>
#include <QTimer>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {
    constexpr int RECENT_LIBRARY_LIMIT = 10;
    constexpr int RECENT_PRESET_LIMIT = 10;

    QString parallel_build_jobs_key() { return acmx2::backend_settings_key(acmx2::Backend::Acmxvk, QStringLiteral("parallel_build_jobs")); }

    QString legacy_parallel_build_enabled_key() { return acmx2::backend_settings_key(acmx2::Backend::Acmxvk, QStringLiteral("parallel_build_enabled")); }

    int parallel_build_jobs(QSettings &settings) {
        const QString jobs_key = parallel_build_jobs_key();
        const QString enabled_key = legacy_parallel_build_enabled_key();
        if (settings.contains(enabled_key)) {
            const bool enabled = settings.value(enabled_key, false).toBool();
            const int jobs = qBound(1, settings.value(jobs_key, 2).toInt(), 256);
            settings.setValue(jobs_key, enabled ? jobs : 0);
            settings.remove(enabled_key);
            return enabled ? jobs : 0;
        }
        return qBound(0, settings.value(jobs_key, 0).toInt(), 256);
    }

    void normalize_parallel_build_settings(QJsonObject &settings) {
        const QString jobs_key = parallel_build_jobs_key();
        const QString enabled_key = legacy_parallel_build_enabled_key();
        if (settings.contains(enabled_key)) {
            const bool enabled = settings.value(enabled_key).toBool(false);
            const int jobs = qBound(1, settings.value(jobs_key).toInt(2), 256);
            settings.insert(jobs_key, enabled ? jobs : 0);
            settings.remove(enabled_key);
        } else if (settings.contains(jobs_key)) {
            settings.insert(jobs_key, qBound(0, settings.value(jobs_key).toInt(), 256));
        }
    }

    void normalize_project_parallel_build_settings(QJsonObject &interface_settings, QJsonObject &application_settings) {
        const QString jobs_key = parallel_build_jobs_key();
        const QString enabled_key = legacy_parallel_build_enabled_key();
        normalize_parallel_build_settings(application_settings);
        normalize_parallel_build_settings(interface_settings);
        if (!application_settings.contains(jobs_key) && interface_settings.contains(jobs_key))
            application_settings.insert(jobs_key, interface_settings.value(jobs_key));
        interface_settings.remove(jobs_key);
        interface_settings.remove(enabled_key);
    }

    QString timestamped_output_path(const QString &output_path) {
        const QFileInfo output_info(output_path);
        const QString base_name = output_info.completeBaseName();
        static const QRegularExpression timestamp_suffix(QStringLiteral("-\\d{8}-\\d{6}-\\d{3}$"));
        if (timestamp_suffix.match(base_name).hasMatch())
            return output_info.absoluteFilePath();
        const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz"));
        QString file_name = base_name + QStringLiteral("-") + timestamp;
        if (!output_info.suffix().isEmpty())
            file_name += QStringLiteral(".") + output_info.suffix();
        return output_info.dir().filePath(file_name);
    }

    QString untimestamped_output_base_name(const QString &output_path) {
        QString base_name = QFileInfo(output_path).completeBaseName();
        static const QRegularExpression timestamp_suffix(QStringLiteral("-\\d{8}-\\d{6}-\\d{3}$"));
        base_name.remove(timestamp_suffix);
        return base_name;
    }

    QString output_path_from_arguments(const QStringList &arguments) {
        QString output_path;
        for (qsizetype index = 0; index + 1 < arguments.size(); ++index) {
            if (arguments.at(index) == QStringLiteral("--output")) {
                output_path = arguments.at(index + 1);
                ++index;
            }
        }
        return output_path;
    }

    QJsonValue preset_json_value(const QVariant &value) {
        if (value.metaType().id() == QMetaType::QStringList) {
            QJsonArray array;
            for (const QString &entry : value.toStringList()) {
                array.append(entry);
            }
            return array;
        }
        return QJsonValue::fromVariant(value);
    }

    QVariant preset_variant(const QJsonValue &value) {
        if (value.isArray()) {
            QStringList result;
            for (const QJsonValue &entry : value.toArray()) {
                if (!entry.isString()) {
                    return {};
                }
                result.append(entry.toString());
            }
            return result;
        }
        return value.toVariant();
    }

    QJsonObject preset_settings(const QSettings &settings) {
        QJsonObject result;
        for (const QString &key : settings.allKeys()) {
            result.insert(key, preset_json_value(settings.value(key)));
        }
        return result;
    }

    void apply_preset_settings(QSettings &settings, const QJsonObject &values) {
        for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
            const QVariant value = preset_variant(it.value());
            if (value.isValid()) {
                settings.setValue(it.key(), value);
            }
        }
        settings.sync();
    }

    struct ProjectProgress {
        std::atomic<qint64> total_bytes = 0;
        std::atomic<qint64> copied_bytes = 0;
        std::atomic_bool cancelled = false;
    };

    class ProjectResources {
      public:
        ProjectResources(QString project_root, std::shared_ptr<ProjectProgress> progress) : project_root(std::move(project_root)), progress(std::move(progress)) {}

        QString copy_path(const QString &source_path, const QString &category, QString &error) {
            if (source_path.isEmpty())
                return {};
            const QFileInfo source(source_path);
            if (!source.exists()) {
                error = QStringLiteral("Project resource does not exist: %1").arg(source_path);
                return {};
            }
            const QString absolute_source = source.absoluteFilePath();
            const QString project_relative_source = QDir(project_root).relativeFilePath(absolute_source);
            if (project_relative_source != QStringLiteral("..") && !project_relative_source.startsWith(QStringLiteral("../")) && !project_relative_source.startsWith(QStringLiteral("..\\"))) {
                // Saving an open project should keep resources that are already
                // inside its bundle in place instead of copying a new numbered
                // duplicate on every save.
                return project_relative_source;
            }
            for (auto it = copied_paths.constBegin(); it != copied_paths.constEnd(); ++it) {
                const QFileInfo copied_source(it.key());
                if (copied_source.isDir() && QDir(it.key()).relativeFilePath(absolute_source) != QStringLiteral("..") && !QDir(it.key()).relativeFilePath(absolute_source).startsWith(QStringLiteral("../"))) {
                    return QDir(it.value()).filePath(QDir(it.key()).relativeFilePath(absolute_source));
                }
                if (it.key() == absolute_source)
                    return it.value();
            }

            QString relative_destination = QDir(category).filePath(source.fileName());
            relative_destination = unique_destination(relative_destination);
            const QString destination = QDir(project_root).filePath(relative_destination);
            if (source.isDir()) {
                progress->total_bytes.fetch_add(directory_size(absolute_source));
                if (!copy_directory(absolute_source, destination, error))
                    return {};
            } else {
                progress->total_bytes.fetch_add(source.size());
                if (!copy_file(absolute_source, destination, error))
                    return {};
            }
            copied_paths.insert(absolute_source, relative_destination);
            return relative_destination;
        }

        QString absolute_path(const QString &relative_path) const { return QDir(project_root).filePath(relative_path); }

      private:
        QString unique_destination(const QString &relative_destination) {
            const QFileInfo info(relative_destination);
            const QString directory = info.path();
            const QString suffix = info.completeSuffix();
            const QString stem = suffix.isEmpty() ? info.fileName() : info.completeBaseName();
            QString candidate = relative_destination;
            int index = 2;
            while (used_destinations.contains(candidate) || QFileInfo::exists(QDir(project_root).filePath(candidate))) {
                const QString name = suffix.isEmpty() ? QStringLiteral("%1-%2").arg(stem).arg(index) : QStringLiteral("%1-%2.%3").arg(stem).arg(index).arg(suffix);
                candidate = QDir(directory).filePath(name);
                ++index;
            }
            used_destinations.insert(candidate);
            return candidate;
        }

        qint64 directory_size(const QString &source) const {
            qint64 total = 0;
            const QDir source_directory(source);
            QDirIterator iterator(source, QDir::Files | QDir::NoDotAndDotDot | QDir::Hidden, QDirIterator::Subdirectories);
            while (iterator.hasNext()) {
                const QString entry = iterator.next();
                if (is_temporary_resource(source_directory.relativeFilePath(entry)))
                    continue;
                total += iterator.fileInfo().size();
            }
            return total;
        }

        bool is_temporary_resource(const QString &relative_path) const {
            QString normalized_path = relative_path;
            normalized_path.replace(QLatin1Char('\\'), QLatin1Char('/'));
            const QStringList path_components = normalized_path.split(QLatin1Char('/'), Qt::SkipEmptyParts);
            const QString file_name = QFileInfo(normalized_path).fileName();
            return file_name.contains(QStringLiteral(".acmxvk-tmp-")) || file_name.contains(QStringLiteral(".live-tmp-")) || path_components.contains(QStringLiteral(".editor-preview")) || path_components.contains(QStringLiteral(".acmx2-editor-preview"));
        }

        bool copy_file(const QString &source, const QString &destination, QString &error) const {
            if (!QDir().mkpath(QFileInfo(destination).absolutePath())) {
                error = QStringLiteral("Could not copy project resource: %1").arg(source);
                return false;
            }
            QFile input(source);
            QFile output(destination);
            if (!input.open(QIODevice::ReadOnly) || !output.open(QIODevice::WriteOnly | QIODevice::NewOnly)) {
                error = QStringLiteral("Could not copy project resource: %1").arg(source);
                return false;
            }
            constexpr qint64 buffer_size = 4 * 1024 * 1024;
            while (!input.atEnd()) {
                if (progress->cancelled.load()) {
                    output.close();
                    QFile::remove(destination);
                    error = QStringLiteral("Project save cancelled");
                    return false;
                }
                const QByteArray bytes = input.read(buffer_size);
                if (bytes.isEmpty() && input.error() != QFile::NoError) {
                    error = QStringLiteral("Could not read project resource: %1").arg(source);
                    return false;
                }
                if (output.write(bytes) != bytes.size()) {
                    error = QStringLiteral("Could not copy project resource: %1").arg(source);
                    return false;
                }
                progress->copied_bytes.fetch_add(bytes.size());
            }
            if (!output.flush()) {
                error = QStringLiteral("Could not finish copying project resource: %1").arg(source);
                return false;
            }
            output.close();
            const QDateTime modification_time = QFileInfo(source).lastModified();
            if (modification_time.isValid()) {
                QFile timestamp_file(destination);
                if (!timestamp_file.open(QIODevice::ReadWrite) || !timestamp_file.setFileTime(modification_time, QFileDevice::FileModificationTime)) {
                    error = QStringLiteral("Could not preserve the project resource timestamp: %1").arg(source);
                    return false;
                }
            }
            return true;
        }

        bool copy_directory(const QString &source, const QString &destination, QString &error) const {
            if (!QDir().mkpath(destination)) {
                error = QStringLiteral("Could not create project resource directory: %1").arg(destination);
                return false;
            }
            QDir source_directory(source);
            QDirIterator iterator(source, QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot | QDir::Hidden, QDirIterator::Subdirectories);
            while (iterator.hasNext()) {
                if (progress->cancelled.load()) {
                    error = QStringLiteral("Project save cancelled");
                    return false;
                }
                const QString entry = iterator.next();
                const QFileInfo info(entry);
                const QString relative_path = source_directory.relativeFilePath(entry);
                if (is_temporary_resource(relative_path))
                    continue;
                const QString target = QDir(destination).filePath(relative_path);
                if (info.isDir()) {
                    if (!QDir().mkpath(target)) {
                        error = QStringLiteral("Could not create project resource directory: %1").arg(target);
                        return false;
                    }
                } else if (!copy_file(entry, target, error)) {
                    return false;
                }
            }
            return true;
        }

        QString project_root;
        QHash<QString, QString> copied_paths;
        QSet<QString> used_destinations;
        std::shared_ptr<ProjectProgress> progress;
    };

    QString copy_audio_playlist(ProjectResources &resources, const QString &source_path, QString &error) {
        const QString relative_playlist = resources.copy_path(source_path, QStringLiteral("resources/audio"), error);
        if (relative_playlist.isEmpty())
            return {};
        QFile source(source_path);
        if (!source.open(QIODevice::ReadOnly | QIODevice::Text)) {
            error = QStringLiteral("Could not read audio playlist: %1").arg(source_path);
            return {};
        }
        QTextStream reader(&source);
        QStringList lines;
        const QDir source_directory = QFileInfo(source_path).absoluteDir();
        const QDir destination_directory = QFileInfo(resources.absolute_path(relative_playlist)).absoluteDir();
        while (!reader.atEnd()) {
            const QString line = reader.readLine();
            const QString trimmed = line.trimmed();
            if (trimmed.isEmpty() || trimmed.startsWith(QLatin1Char('#')) || trimmed.contains(QStringLiteral("://"))) {
                lines.append(line);
                continue;
            }
            const QString track = QDir::isAbsolutePath(trimmed) ? trimmed : source_directory.filePath(trimmed);
            const QString relative_track = resources.copy_path(track, QStringLiteral("resources/audio/tracks"), error);
            if (relative_track.isEmpty())
                return {};
            lines.append(destination_directory.relativeFilePath(resources.absolute_path(relative_track)));
        }
        QSaveFile playlist(resources.absolute_path(relative_playlist));
        if (!playlist.open(QIODevice::WriteOnly | QIODevice::Text) || playlist.write(lines.join(QLatin1Char('\n')).toUtf8()) < 0 || !playlist.commit()) {
            error = QStringLiteral("Could not write portable audio playlist: %1").arg(relative_playlist);
            return {};
        }
        return relative_playlist;
    }

    QString project_path(const QString &project_root, const QString &path) {
        if (path.isEmpty() || QDir::isAbsolutePath(path))
            return path;
        return QFileInfo(QDir(project_root).filePath(path)).absoluteFilePath();
    }

    void set_project_path(QJsonObject &settings, const QString &key, const QString &path) {
        if (!path.isEmpty())
            settings.insert(key, path);
    }

    void resolve_project_path(QJsonObject &settings, const QString &key, const QString &project_root) {
        const QJsonValue value = settings.value(key);
        if (value.isString())
            settings.insert(key, project_path(project_root, value.toString()));
    }

    void resolve_project_path_list(QJsonObject &settings, const QString &key, const QString &project_root) {
        const QJsonArray values = settings.value(key).toArray();
        if (values.isEmpty())
            return;
        QJsonArray resolved;
        for (const QJsonValue &value : values)
            resolved.append(value.isString() ? project_path(project_root, value.toString()) : value);
        settings.insert(key, resolved);
    }

    bool is_nonportable_path(const QString &value) {
        const QString path = value.trimmed();
        return QDir::isAbsolutePath(path) || QRegularExpression(QStringLiteral("^[A-Za-z]:[/\\\\]|^//|^\\\\\\\\")).match(path).hasMatch();
    }

    void remove_nonportable_paths(QJsonObject &settings) {
        for (auto it = settings.begin(); it != settings.end();) {
            if (it.value().isString() && is_nonportable_path(it.value().toString())) {
                it = settings.erase(it);
                continue;
            }
            if (it.value().isArray()) {
                QJsonArray portable_values;
                for (const QJsonValue &value : it.value().toArray()) {
                    if (!value.isString() || !is_nonportable_path(value.toString()))
                        portable_values.append(value);
                }
                *it = portable_values;
            }
            ++it;
        }
    }

    void remove_nonportable_project_settings(QJsonObject &interface_settings, QJsonObject &application_settings) {
        remove_nonportable_paths(interface_settings);
        remove_nonportable_paths(application_settings);

        const QStringList local_keys = {QStringLiteral("interface/backend"), QStringLiteral("interface/extra_arguments"), QStringLiteral("exePath"), QStringLiteral("shaders"), QStringLiteral("midiDevice"), QStringLiteral("customStyleSheet"), QStringLiteral("customStylePreset"), QStringLiteral("useCustomStyle"), QStringLiteral("editor/geometry"), QStringLiteral("editor/workspaceGeometry"), QStringLiteral("findInFiles/geometry"), QStringLiteral("lastExeDir"), QStringLiteral("lastShaderDir"), QStringLiteral("lastScreenshotDir"), QStringLiteral("lastShaderCompilerDir"), QStringLiteral("lastLibraryDir"), QStringLiteral("lastPlaylistDir"), QStringLiteral("lastMidiConfigDir"), QStringLiteral("lastGpuFilterDir"), QStringLiteral("lastEditorSaveDir")};
        for (const QString &key : local_keys) {
            interface_settings.remove(key);
            application_settings.remove(key);
        }

        const auto remove_local_keys = [](QJsonObject &settings) {
            for (auto it = settings.begin(); it != settings.end();) {
                if (it.key().startsWith(QStringLiteral("last"), Qt::CaseInsensitive) || it.key().startsWith(QStringLiteral("effect_packs/")) || it.key().startsWith(QStringLiteral("backend/acmx2/")) || it.key().contains(QStringLiteral("/recent"), Qt::CaseInsensitive) || it.key().endsWith(QStringLiteral("/executable")) || it.key().endsWith(QStringLiteral("/shader_compiler_path"))) {
                    it = settings.erase(it);
                    continue;
                }
                ++it;
            }
        };
        remove_local_keys(interface_settings);
        remove_local_keys(application_settings);
    }

    struct ProjectSaveRequest {
        QString path;
        QJsonObject interface_settings;
        QJsonObject application_settings;
        QJsonObject runtime;
        QString shader_library;
        QString selected_shader;
        bool repeat = false;
        QStringList acmxvk_arguments;
        QString video_file;
        QString graphics_file;
        QString audio_file;
        QString model_file;
        QString onnx_model;
        QString deep_dream_model;
        QString stable_diffusion_model;
        QString stable_diffusion_upscale_model;
        QStringList stable_diffusion_loras;
        QString midi_config_file;
        QString playlist_file;
        QString output_file;
        QString output_extension;
        EffectPackProjectState effect_pack;
        std::shared_ptr<ProjectProgress> progress;
        bool enable_3d = false;
        bool onnx_model_enabled = false;
        bool deep_dream_enabled = false;
        bool stable_diffusion_enabled = false;
        bool stable_diffusion_upscale_only = false;
        bool midi_enabled = false;
        bool playlist_enabled = false;
        bool png_output = false;
        bool create_output_directories = true;
    };

    struct ProjectSaveResult {
        bool success = false;
        QString message;
    };

    struct ProjectLoadResult {
        bool success = false;
        QString message;
        QJsonDocument document;
    };

    ProjectLoadResult load_project_document(const QString &path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return {false, QStringLiteral("Could not open project:\n%1").arg(path), {}};

        QJsonParseError error;
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
        if (error.error != QJsonParseError::NoError || !document.isObject())
            return {false, QStringLiteral("The project is not valid JSON:\n%1").arg(error.errorString()), {}};
        return {true, {}, document};
    }

    ProjectSaveResult save_project_bundle(ProjectSaveRequest request) {
        const QString project_root = QFileInfo(request.path).absolutePath();
        if (!QDir().mkpath(project_root))
            return {false, QStringLiteral("Could not create project directory: %1").arg(project_root)};
        if (request.create_output_directories && (!QDir().mkpath(QDir(project_root).filePath(QStringLiteral("output/snapshots"))) || !QDir().mkpath(QDir(project_root).filePath(QStringLiteral("output/logs")))))
            return {false, QStringLiteral("Could not create project output directory")};

        ProjectResources resources(project_root, request.progress);
        QString error;
        const auto copy_resource = [&resources, &error](const QString &source, const QString &category) { return resources.copy_path(source, category, error); };
        const auto require_resource = [](const QString &resource) { return !resource.isEmpty(); };
        QJsonObject interface_settings = request.interface_settings;
        QJsonObject application_settings = request.application_settings;

        QString project_library;
        if (!request.shader_library.isEmpty()) {
            project_library = copy_resource(request.shader_library, QStringLiteral("resources/shaders"));
            if (project_library.isEmpty())
                return {false, error};
            application_settings.insert(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "library"), project_library);
        }
        QJsonObject project_effect_pack;
        if (!request.effect_pack.manifest_path.isEmpty()) {
            QString relative_manifest;
            if (!acmx2::bundle_effect_pack_project(request.effect_pack, project_root, relative_manifest, error))
                return {false, error};
            project_effect_pack.insert(QStringLiteral("id"), request.effect_pack.id);
            project_effect_pack.insert(QStringLiteral("manifest"), relative_manifest);
            project_effect_pack.insert(QStringLiteral("values"), request.effect_pack.values);
            if (!request.effect_pack.dream_model_file.isEmpty()) {
                const QString model = copy_resource(request.effect_pack.dream_model_file, QStringLiteral("resources/models"));
                if (model.isEmpty())
                    return {false, error};
                if (QFileInfo::exists(request.effect_pack.dream_model_file + QStringLiteral(".json")) && copy_resource(request.effect_pack.dream_model_file + QStringLiteral(".json"), QStringLiteral("resources/models")).isEmpty())
                    return {false, error};
                project_effect_pack.insert(QStringLiteral("model"), model);
            }
        }
        if (!request.video_file.isEmpty()) {
            const QString video = copy_resource(request.video_file, QStringLiteral("resources/media"));
            if (video.isEmpty())
                return {false, error};
            set_project_path(interface_settings, QStringLiteral("interface/input_video"), video);
        }
        if (!request.graphics_file.isEmpty()) {
            const QString graphic = copy_resource(request.graphics_file, QStringLiteral("resources/media"));
            if (graphic.isEmpty())
                return {false, error};
            set_project_path(interface_settings, QStringLiteral("interface/graphics_file"), graphic);
        }
        if (!request.audio_file.isEmpty()) {
            const bool audio_playlist = interface_settings.value("audio/playlist_enabled").toBool();
            const QString audio = audio_playlist ? copy_audio_playlist(resources, request.audio_file, error) : copy_resource(request.audio_file, QStringLiteral("resources/audio"));
            if (audio.isEmpty())
                return {false, error};
            set_project_path(interface_settings, audio_playlist ? QStringLiteral("audio/playlist_path") : QStringLiteral("audio/file_path"), audio);
        }
        if (request.enable_3d && !request.model_file.isEmpty()) {
            const QString model = copy_resource(request.model_file, QStringLiteral("resources/models"));
            if (model.isEmpty())
                return {false, error};
            set_project_path(interface_settings, QStringLiteral("interface/model_file"), model);
        }
        if (request.onnx_model_enabled && !request.onnx_model.isEmpty()) {
            const QString model = copy_resource(request.onnx_model, QStringLiteral("resources/models"));
            if (model.isEmpty())
                return {false, error};
            set_project_path(interface_settings, QStringLiteral("interface/onnx_model_file"), model);
        }
        if (request.deep_dream_enabled && !request.deep_dream_model.isEmpty()) {
            const QString model = copy_resource(request.deep_dream_model, QStringLiteral("resources/models"));
            if (model.isEmpty())
                return {false, error};
            if (QFileInfo::exists(request.deep_dream_model + QStringLiteral(".json")) && !require_resource(copy_resource(request.deep_dream_model + QStringLiteral(".json"), QStringLiteral("resources/models"))))
                return {false, error};
            set_project_path(interface_settings, QStringLiteral("deep_dream/model_file"), model);
        }
        if (request.stable_diffusion_enabled) {
            if (!request.stable_diffusion_upscale_only && !request.stable_diffusion_model.isEmpty()) {
                const QString model = copy_resource(request.stable_diffusion_model, QStringLiteral("resources/models"));
                if (model.isEmpty())
                    return {false, error};
                set_project_path(interface_settings, QStringLiteral("stable_diffusion/model_file"), model);
            }
            if (!request.stable_diffusion_upscale_model.isEmpty()) {
                const QString model = copy_resource(request.stable_diffusion_upscale_model, QStringLiteral("resources/models"));
                if (model.isEmpty())
                    return {false, error};
                set_project_path(interface_settings, QStringLiteral("stable_diffusion/upscale_model_file"), model);
            }
            QJsonArray loras;
            for (const QString &lora : request.stable_diffusion_loras) {
                const QString copied_lora = copy_resource(lora, QStringLiteral("resources/models/loras"));
                if (copied_lora.isEmpty())
                    return {false, error};
                loras.append(copied_lora);
            }
            interface_settings.insert("stable_diffusion/lora_files", loras);
        }
        if (request.midi_enabled && !request.midi_config_file.isEmpty()) {
            const QString midi_map = copy_resource(request.midi_config_file, QStringLiteral("resources/midi"));
            if (midi_map.isEmpty())
                return {false, error};
            application_settings.insert("midiConfigFile", midi_map);
        }
        QString project_playlist;
        if (request.playlist_enabled && !request.playlist_file.isEmpty()) {
            project_playlist = copy_resource(request.playlist_file, QStringLiteral("resources/playlists"));
            if (project_playlist.isEmpty())
                return {false, error};
        }
        const QFileInfo source_output(request.output_file.isEmpty() ? QFileInfo(request.path).completeBaseName() + QStringLiteral(".mp4") : request.output_file);
        const QString output_extension = request.output_extension.isEmpty() ? (source_output.completeSuffix().isEmpty() ? QStringLiteral("mp4") : source_output.completeSuffix()) : request.output_extension;
        QString output_name = source_output.fileName();
        if (output_name.isEmpty())
            output_name = source_output.completeBaseName() + QStringLiteral(".") + output_extension;
        interface_settings.insert("interface/save_output", true);
        set_project_path(interface_settings, QStringLiteral("interface/output_video"), QDir(QStringLiteral("output")).filePath(output_name));
        if (request.png_output)
            set_project_path(interface_settings, QStringLiteral("interface/png_output_directory"), QStringLiteral("output/png-sequence"));
        application_settings.insert("prefix_path", QStringLiteral("output/snapshots"));

        QStringList portable_arguments;
        const QSet<QString> resource_options = {"--input", "--graphic", "--audio-file", "--shaders", "--model", "--onnx", "--dream-model", "--sd-model", "--upscale-model", "--sd-lora", "--playlist", "--midi-map", "--edge", "--human"};
        const QSet<QString> local_path_options = {"--fragment", "--compute", "--sd-server", "--glslc", "--build", "--builddir", "--fix", "--probe-hdr"};
        const QSet<QString> interface_only_options = {"--interface-shm"};
        for (int index = 0; index < request.acmxvk_arguments.size(); ++index) {
            const QString argument = request.acmxvk_arguments.at(index);
            if (interface_only_options.contains(argument))
                continue;
            if (argument == QStringLiteral("--path")) {
                ++index;
                continue;
            }
            if (index + 1 >= request.acmxvk_arguments.size()) {
                portable_arguments.append(argument);
                continue;
            }
            const QString value = request.acmxvk_arguments.at(index + 1);
            if (resource_options.contains(argument)) {
                const QString category = argument == QStringLiteral("--shaders") ? QStringLiteral("resources/shaders") : QStringLiteral("resources/external");
                const QString resource = copy_resource(value, category);
                if (resource.isEmpty())
                    return {false, error};
                portable_arguments.append(argument);
                portable_arguments.append(resource);
                ++index;
            } else if (argument == QStringLiteral("--output") || argument == QStringLiteral("--record-audio")) {
                portable_arguments.append(argument);
                portable_arguments.append(QDir(QStringLiteral("output")).filePath(QFileInfo(value).fileName()));
                ++index;
            } else if (argument == QStringLiteral("--prefix")) {
                portable_arguments.append(argument);
                portable_arguments.append(QStringLiteral("output/snapshots"));
                ++index;
            } else if (local_path_options.contains(argument) || (argument.startsWith(QStringLiteral("--")) && is_nonportable_path(value))) {
                ++index;
            } else {
                portable_arguments.append(argument);
            }
        }

        remove_nonportable_project_settings(interface_settings, application_settings);
        normalize_project_parallel_build_settings(interface_settings, application_settings);

        QJsonObject root;
        root.insert("format", "acmx-project");
        root.insert("version", 2);
        root.insert("interface_settings", interface_settings);
        root.insert("application_settings", application_settings);
        root.insert("shader_library", project_library);
        root.insert("selected_shader", request.selected_shader);
        if (!project_effect_pack.isEmpty())
            root.insert(QStringLiteral("effect_pack"), project_effect_pack);
        root.insert("repeat", request.repeat);
        QJsonObject runtime = request.runtime;
        runtime.insert("playlist_file", project_playlist);
        root.insert("runtime", runtime);
        QJsonArray arguments;
        for (const QString &argument : portable_arguments)
            arguments.append(argument);
        root.insert("acmxvk_arguments", arguments);

        QSaveFile file(request.path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text) || file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0 || !file.commit())
            return {false, QStringLiteral("Could not write project: %1").arg(request.path)};
        return {true, request.path};
    }

    bool take_process_output_line(QString &buffer, QString &line) {
        const qsizetype newline = buffer.indexOf('\n');
        const qsizetype carriage = buffer.indexOf('\r');
        qsizetype separator = newline;
        if (separator < 0 || (carriage >= 0 && carriage < separator)) {
            separator = carriage;
        }
        if (separator < 0) {
            return false;
        }

        line = buffer.left(separator);
        qsizetype consumed = 1;
        if (buffer.at(separator) == QLatin1Char('\r') && separator + 1 < buffer.size() && buffer.at(separator + 1) == QLatin1Char('\n')) {
            consumed = 2;
        }
        buffer.remove(0, separator + consumed);
        return true;
    }

    bool is_stable_diffusion_diagnostic(const QString &line) {
        QString normalized = line;
        static const QRegularExpression ANSI_ESCAPE(QStringLiteral("\\x1b\\[[0-9;?]*[ -/]*[@-~]"));
        normalized.remove(ANSI_ESCAPE);
        normalized = normalized.trimmed();
        if (normalized.startsWith("[INFO ]") || normalized.startsWith("[DEBUG]") || normalized.startsWith("[TRACE]")) {
            return true;
        }
        if (normalized.startsWith(QLatin1Char('|')) && normalized.contains(QRegularExpression(QStringLiteral("\\d+\\s*/\\s*\\d+")))) {
            return true;
        }
        return false;
    }

    bool is_acmxvk_interface_diagnostic(const QString &line) {
        const QString normalized = line.trimmed();
        return normalized.startsWith(QStringLiteral("mxvk:")) || normalized.startsWith(QStringLiteral("vk:")) || normalized.startsWith(QStringLiteral("SDL3:")) || normalized.startsWith(QStringLiteral("acmxvk: Vulkan shader pipeline")) || QRegularExpression(QStringLiteral("^\\d+: ")).match(normalized).hasMatch();
    }

    QString shellQuote(const QString &value) {
#ifdef _WIN32
        if (value.isEmpty()) {
            return QStringLiteral("\"\"");
        }

        QString quoted = QStringLiteral("\"");
        qsizetype backslash_count = 0;
        for (const QChar character : value) {
            if (character == QLatin1Char('\\')) {
                ++backslash_count;
                continue;
            }
            if (character == QLatin1Char('"')) {
                quoted += QString(backslash_count * 2 + 1, QLatin1Char('\\'));
                quoted += character;
                backslash_count = 0;
                continue;
            }
            quoted += QString(backslash_count, QLatin1Char('\\'));
            backslash_count = 0;
            quoted += character;
        }
        quoted += QString(backslash_count * 2, QLatin1Char('\\'));
        quoted += QLatin1Char('"');
        return quoted;
#else
        if (value.isEmpty()) {
            return "''";
        }
        QString out = value;
        out.replace("'", "'\\''");
        return "'" + out + "'";
#endif
    }

    QString buildShellCommand(const QStringList &envAssignments, const QString &program, const QStringList &arguments) {
        QStringList parts;
        parts.reserve(envAssignments.size() + 1 + arguments.size());
        for (const QString &entry : envAssignments) {
            int eq = entry.indexOf('=');
            if (eq <= 0) {
                continue;
            }
            QString key = entry.left(eq);
            QString value = entry.mid(eq + 1);
#ifdef _WIN32
            parts << (QStringLiteral("set \"") + key + QLatin1Char('=') + value + QStringLiteral("\" &&"));
#else
            parts << (key + "=" + shellQuote(value));
#endif
        }
        parts << shellQuote(program);
        for (const QString &arg : arguments) {
            parts << shellQuote(arg);
        }
        return parts.join(' ');
    }

    void replace_file(const std::filesystem::path &source, const std::filesystem::path &destination, std::error_code &error) {
#ifdef _WIN32
        if (MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE) {
            error.clear();
            return;
        }
        error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
#else
        std::filesystem::rename(source, destination, error);
#endif
    }

#ifdef __linux__
    QStringList defaultLinuxRunEnvAssignments() {
        QStringList envAssignments;
        QString uid = QString::number(getuid());
        QString userRunPath = "/run/user/" + uid;
        QByteArray display = qgetenv("DISPLAY");
        QByteArray waylandDisplay = qgetenv("WAYLAND_DISPLAY");
        QByteArray sessionType = qgetenv("XDG_SESSION_TYPE");
        if (!waylandDisplay.isEmpty() && sessionType == "wayland") {
            envAssignments << "SDL_VIDEODRIVER=wayland";
        } else if (!display.isEmpty()) {
            envAssignments << "SDL_VIDEODRIVER=x11";
        } else if (!waylandDisplay.isEmpty()) {
            envAssignments << "SDL_VIDEODRIVER=wayland";
        }
        if (QDir(userRunPath).exists()) {
            envAssignments << ("XDG_RUNTIME_DIR=" + userRunPath);
            envAssignments << ("PULSE_SERVER=unix:" + userRunPath + "/pulse/native");
        }
        envAssignments << "vblank_mode=0";
        return envAssignments;
    }
#endif

    QString resolveAssetsPath() {
        QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
        return dirPath + "/../Helpers";
#else
        if (QFileInfo::exists(dirPath + "/data/win-icon.png"))
            return dirPath;
        const QString installedPath = QDir::cleanPath(dirPath + "/../share/acmx2");
        if (QFileInfo::exists(installedPath + "/data/win-icon.png"))
            return installedPath;
        return dirPath;
#endif
    }

    QString resolve_acmxvk_shader_compiler(QString &error) {
        error.clear();
        QSettings settings("LostSideDead");
        const QString mode = settings.value(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "shader_compiler_mode"), "auto").toString();
        if (mode == QStringLiteral("custom")) {
            QString compiler = settings.value(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "shader_compiler_path")).toString().trimmed();
            if (compiler.isEmpty()) {
                error = QStringLiteral("The custom ACMXVK shader compiler path is empty. Select "
                                       "one in Properties (Ctrl+,).");
                return {};
            }
            if (QFileInfo(compiler).isRelative()) {
                const QString resolved = QStandardPaths::findExecutable(compiler);
                if (!resolved.isEmpty())
                    compiler = resolved;
            }
            const QFileInfo compilerInfo(compiler);
            if (!compilerInfo.isFile() || !compilerInfo.isExecutable()) {
                error = QStringLiteral("The configured ACMXVK shader compiler is not an "
                                       "executable file: %1")
                            .arg(compiler);
                return {};
            }
            return compilerInfo.absoluteFilePath();
        }

        QString compiler;
#ifdef _WIN32
        const QString compilerName = QStringLiteral("glslc.exe");
        const QFileInfo bundledCompiler(QDir(QCoreApplication::applicationDirPath()).filePath(compilerName));
        if (bundledCompiler.isFile() && bundledCompiler.isExecutable())
            compiler = bundledCompiler.absoluteFilePath();
#else
        const QString compilerName = QStringLiteral("glslc");
#endif
        if (compiler.isEmpty())
            compiler = QStandardPaths::findExecutable(compilerName);
        if (compiler.isEmpty()) {
            const QString sdk = QString::fromLocal8Bit(qgetenv("VULKAN_SDK"));
            const QString sdkCompiler = QDir(sdk).filePath(QStringLiteral("bin/") + compilerName);
            if (!sdk.isEmpty() && QFileInfo(sdkCompiler).isExecutable())
                compiler = sdkCompiler;
        }
        if (compiler.isEmpty()) {
            error = QStringLiteral("glslc was not found in PATH or VULKAN_SDK. Select a custom "
                                   "compiler in Properties (Ctrl+,).");
        }
        return compiler;
    }

    QString default_stable_diffusion_server() {
#ifdef _WIN32
        const QFileInfo bundled_server(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("sd-server.exe")));
        if (bundled_server.isFile()) {
            return bundled_server.absoluteFilePath();
        }
        return QStringLiteral("sd-server.exe");
#else
        const QFileInfo bundled_server(QStringLiteral("/app/bin/sd-server"));
        if (bundled_server.isExecutable()) {
            return bundled_server.absoluteFilePath();
        }
        return QStringLiteral("sd-server");
#endif
    }

    QString resolve_backend_assets_path(acmx2::Backend backend, const QString &executable, const QString &libraryPath) {
        if (backend == acmx2::Backend::Acmx2)
            return resolveAssetsPath();

        const QString applicationDir = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
        const QString bundleResources = QDir::cleanPath(applicationDir + "/../Resources/acmxvk");
        if (QFileInfo(bundleResources).isDir())
            return bundleResources;
#endif
        QStringList candidates;
        QString resolvedExecutable = executable;
        if (QFileInfo(resolvedExecutable).isRelative()) {
            const QString pathExecutable = QStandardPaths::findExecutable(resolvedExecutable);
            if (!pathExecutable.isEmpty())
                resolvedExecutable = pathExecutable;
        }
        const QFileInfo executableInfo(resolvedExecutable);
        if (!executableInfo.absolutePath().isEmpty()) {
            candidates << QDir::cleanPath(executableInfo.absolutePath() + "/../share/acmxvk");
        }
        if (!libraryPath.isEmpty())
            candidates << QDir::cleanPath(QFileInfo(libraryPath).absolutePath());
        candidates << QDir::cleanPath(applicationDir + "/../share/acmxvk") << QStringLiteral("/usr/local/share/acmxvk") << QStringLiteral("/opt/homebrew/share/acmxvk") << QStringLiteral("/usr/share/acmxvk");
        for (const QString &candidate : candidates) {
            if (QFileInfo(candidate + "/data").isDir())
                return candidate;
        }

        // --path only requires a readable directory. ACMXVK can still use the
        // explicitly selected shader library if no installed data tree exists.
        return applicationDir;
    }

    bool is_acmxvk_source_library(const QString &libraryPath, QString &error) {
        error.clear();
        const std::optional<acmx2::ShaderLibraryType> type = acmx2::shader_manifest_library_type(libraryPath, error);
        if (!error.isEmpty())
            return false;
        if (type)
            return *type == acmx2::ShaderLibraryType::Source;

        // Legacy manifests may not carry library_type. Infer source libraries
        // from GLSL entries while continuing to accept old SPIR-V manifests.
        QStringList entries;
        if (!acmx2::load_shader_manifest(libraryPath, entries, error))
            return false;
        return std::any_of(entries.cbegin(), entries.cend(), [](const QString &entry) { return entry.endsWith(".frag", Qt::CaseInsensitive) || entry.endsWith(".comp", Qt::CaseInsensitive); });
    }

    QString acmxvk_build_directory(const QString &sourceLibrary) { return QDir(sourceLibrary).filePath(QStringLiteral(".acmxvk-build")); }

    QString acmxvk_runtime_shader_name(const QString &sourceName) { return sourceName.endsWith(".spv", Qt::CaseInsensitive) ? sourceName : sourceName + QStringLiteral(".spv"); }

    enum class AcmxvkBuildState { UpToDate, Stale, NotBuilt };

    AcmxvkBuildState acmxvk_shader_build_state(const QString &source_library, const QString &source_name) {
        const QString runtime_library = acmxvk_build_directory(source_library);
        if (!acmx2::shader_manifest_exists(runtime_library))
            return AcmxvkBuildState::NotBuilt;

        const QFileInfo source_file(QDir(source_library).filePath(source_name));
        const QFileInfo runtime_file(QDir(runtime_library).filePath(acmxvk_runtime_shader_name(source_name)));
        if (!runtime_file.isFile())
            return AcmxvkBuildState::NotBuilt;
        if (source_file.isFile() && runtime_file.lastModified() < source_file.lastModified()) {
            return AcmxvkBuildState::Stale;
        }
        return AcmxvkBuildState::UpToDate;
    }

    bool acmxvk_runtime_manifest_matches(const QString &source_library, const QString &runtime_library, QString &error) {
        QStringList source_entries;
        QStringList runtime_entries;
        if (!acmx2::load_shader_manifest(source_library, source_entries, error) || !acmx2::load_shader_manifest(runtime_library, runtime_entries, error)) {
            return false;
        }

        QStringList expected_entries;
        expected_entries.reserve(source_entries.size());
        for (const QString &entry : source_entries)
            expected_entries.append(acmxvk_runtime_shader_name(entry));
        if (runtime_entries != expected_entries) {
            error = QObject::tr("The ACMXVK runtime manifest does not match the source "
                                "shader list. Choose Playback > Build before running.");
            return false;
        }

        bool uniformMetadataMatches = false;
        if (!acmx2::custom_uniform_metadata_matches(source_library, runtime_library, uniformMetadataMatches, error)) {
            return false;
        }
        if (!uniformMetadataMatches) {
            error = QObject::tr("The ACMXVK runtime custom-uniform metadata is out of date. "
                                "Choose Playback > Build before running.");
            return false;
        }
        return true;
    }

    bool resolve_acmxvk_runtime_library(const QString &selectedLibrary, QString &runtimeLibrary, QString &error) {
        error.clear();
        runtimeLibrary = selectedLibrary;
        if (!is_acmxvk_source_library(selectedLibrary, error))
            return error.isEmpty();

        runtimeLibrary = acmxvk_build_directory(selectedLibrary);
        if (!acmx2::shader_manifest_exists(runtimeLibrary)) {
            error = QObject::tr("The ACMXVK source library has not been built yet. "
                                "Choose Playback > Build first.\n\nExpected output: %1")
                        .arg(runtimeLibrary);
            return false;
        }
        const std::optional<acmx2::ShaderLibraryType> type = acmx2::shader_manifest_library_type(runtimeLibrary, error);
        if (!error.isEmpty())
            return false;
        if (type && *type != acmx2::ShaderLibraryType::Runtime) {
            error = QObject::tr("Compiled output is not an ACMXVK runtime library: %1").arg(runtimeLibrary);
            return false;
        }
        QStringList sourceEntries;
        if (!acmx2::load_shader_manifest(selectedLibrary, sourceEntries, error))
            return false;
        if (!acmxvk_runtime_manifest_matches(selectedLibrary, runtimeLibrary, error)) {
            return false;
        }
        for (const QString &sourceEntry : sourceEntries) {
            const QFileInfo sourceFile(QDir(selectedLibrary).filePath(sourceEntry));
            const QFileInfo runtimeFile(QDir(runtimeLibrary).filePath(acmxvk_runtime_shader_name(sourceEntry)));
            if (!runtimeFile.isFile() || runtimeFile.lastModified() < sourceFile.lastModified()) {
                error = QObject::tr("The ACMXVK build is missing or older than %1. "
                                    "Choose Playback > Build before running.")
                            .arg(sourceEntry);
                return false;
            }
        }
        return true;
    }

    bool textureCacheArraySettingEnabled() {
        QSettings settings("LostSideDead", "acmx2");
        return settings.value("interface/texture_cache_array", false).toBool();
    }

    QSize storedResolution(QSettings &settings, const QString &key, const QSize &fallback, bool defaultIsEmpty) {
        const QString text = settings.value(key).toString().trimmed();
        if (text.compare("Default", Qt::CaseInsensitive) == 0) {
            return defaultIsEmpty ? QSize(0, 0) : fallback;
        }

        static const QRegularExpression resolutionPattern(R"(^\s*(\d+)\s*[xX]\s*(\d+)\s*$)");
        const QRegularExpressionMatch match = resolutionPattern.match(text);
        if (!match.hasMatch()) {
            return fallback;
        }

        const int width = match.captured(1).toInt();
        const int height = match.captured(2).toInt();
        return width > 0 && height > 0 ? QSize(width, height) : fallback;
    }

    bool hasPositiveResolution(const QSize &resolution) { return resolution.width() > 0 && resolution.height() > 0; }

    QString shaderCacheFilename(const QString &libraryPath, int cacheSize, bool useArray) {
        std::error_code ec;
        const std::filesystem::path libraryFsPath(libraryPath.toStdString());
        const std::filesystem::path absoluteLibrary = std::filesystem::absolute(libraryFsPath, ec);
        std::string key = ec ? libraryPath.toStdString() : absoluteLibrary.lexically_normal().string();
        key += "|s=" + std::to_string(cacheSize);
        key += "|a=" + std::to_string(useArray ? 1 : 0);
        std::ostringstream nameStream;
        nameStream << ".shader_cache_" << std::hex << std::hash<std::string>{}(key);
        return QString::fromStdString(nameStream.str());
    }

    QString resolveShaderCachePath(const QString &libraryPath, int cacheSize, bool useArray) {
        const QString assets = resolveAssetsPath();
        const QString filename = shaderCacheFilename(libraryPath, cacheSize, useArray);

        // Mirror ShaderLibrary::shaderCacheFilePath: prefer cache in assets dir,
        // then fall back to the library directory itself (acmx2 writes there when
        // assets isn't writable).
        const QString assetsCache = assets + "/" + filename;
        const QString libCache = libraryPath + "/" + filename;
        if (QFileInfo::exists(assetsCache))
            return assetsCache;
        if (QFileInfo::exists(libCache))
            return libCache;
        return assetsCache;
    }

    // Parse the shader cache file produced by ShaderLibrary::buildShaderCache().
    // Returns a map of shader stem -> failed flag. Empty on missing/invalid cache.
    QHash<QString, bool> parseShaderCacheStatus(const QString &cachePath) {
        QHash<QString, bool> result;
        QFile f(cachePath);
        if (!f.open(QIODevice::ReadOnly))
            return result;

        auto readU32 = [&](quint32 &v) -> bool { return f.read(reinterpret_cast<char *>(&v), sizeof(v)) == qint64(sizeof(v)); };
        auto readU64 = [&](quint64 &v) -> bool { return f.read(reinterpret_cast<char *>(&v), sizeof(v)) == qint64(sizeof(v)); };
        auto readU8 = [&](quint8 &v) -> bool { return f.read(reinterpret_cast<char *>(&v), sizeof(v)) == qint64(sizeof(v)); };
        auto readStr = [&](QString &out) -> bool {
            quint32 len = 0;
            if (!readU32(len))
                return false;
            QByteArray buf = f.read(len);
            if (quint32(buf.size()) != len)
                return false;
            out = QString::fromUtf8(buf);
            return true;
        };
        auto skipBytes = [&](quint32 n) -> bool { return f.skip(n) == qint64(n); };

        constexpr quint32 CACHE_MAGIC = 0x53484452;
        constexpr quint32 CACHE_VERSION = 4;

        quint32 magic = 0, version = 0;
        if (!readU32(magic) || !readU32(version))
            return result;
        if (magic != CACHE_MAGIC || version != CACHE_VERSION)
            return result;

        QString tmp;
        if (!readStr(tmp))
            return result; // gl_renderer
        if (!readStr(tmp))
            return result; // gl_version

        quint8 dual_mode = 0;
        if (!readU8(dual_mode))
            return result;

        quint32 count = 0;
        if (!readU32(count))
            return result;

        for (quint32 i = 0; i < count; ++i) {
            QString name;
            if (!readStr(name))
                return result;
            quint8 shader_kind = 0;
            if (!readU8(shader_kind) || shader_kind > 2)
                return result;
            quint8 failed_flag = 0;
            if (!readU8(failed_flag))
                return result;
            quint64 source_hash = 0;
            if (!readU64(source_hash))
                return result;
            quint32 fmt2d = 0, sz2d = 0, fmt3d = 0, sz3d = 0;
            if (!readU32(fmt2d) || !readU32(sz2d) || !skipBytes(sz2d))
                return result;
            if (!readU32(fmt3d) || !readU32(sz3d) || !skipBytes(sz3d))
                return result;
            result.insert(name, failed_flag != 0);
        }
        return result;
    }

    QString formatLastModified(const QDateTime &dt) {
        if (!dt.isValid())
            return QStringLiteral("-");
        return dt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm"));
    }
} // namespace

void MainWindow::initControls() {
    lastFoundIndex = -1;
    lastSearchText = QString();
    process = new QProcess(this);
    auto updateShaderMenuState = [this](QProcess::ProcessState state) {
        const bool running = (state == QProcess::Running);
        if (backendMenu)
            backendMenu->setEnabled(!running);
        if (listMenu_new) {
            listMenu_new->setEnabled(!running);
        }
        if (listMenu_shader) {
            listMenu_shader->setEnabled(!running);
        }
        if (libraryBuilderAction) {
            libraryBuilderAction->setEnabled(!running);
        }
        if (listMenu_remove) {
            listMenu_remove->setEnabled(!running);
        }
        if (listMenu_up) {
            listMenu_up->setEnabled(!running);
        }
        if (listMenu_down) {
            listMenu_down->setEnabled(!running);
        }
        if (listMenu_shuffle) {
            listMenu_shuffle->setEnabled(!running);
        }
        if (listMenu_sort) {
            listMenu_sort->setEnabled(!running);
        }
        if (listMenu_set_current) {
            listMenu_set_current->setEnabled(running);
        }
    };
    connect(process, &QProcess::stateChanged, this, updateShaderMenuState);
    updateShaderMenuState(process->state());
    connect(process, &QProcess::readyReadStandardOutput, this, [this]() {
        stdoutBuffer += QString::fromLocal8Bit(process->readAllStandardOutput());
        QString line;
        while (take_process_output_line(stdoutBuffer, line)) {
            appendOutputRunLog(line);
            if (active_backend == acmx2::Backend::Acmxvk && stable_diffusion_enabled && is_stable_diffusion_diagnostic(line)) {
                continue;
            }
            if (active_backend == acmx2::Backend::Acmxvk && is_acmxvk_interface_diagnostic(line)) {
                continue;
            }
            queueProcessOutput(line + "<br>");
        }
    });

    connect(process, &QProcess::readyReadStandardError, this, [this]() {
        auto writeStderrLine = [this](const QString &line) {
            if (line.contains("GStreamer"))
                return;
            if (active_backend == acmx2::Backend::Acmxvk && stable_diffusion_enabled && is_stable_diffusion_diagnostic(line)) {
                return;
            }
            if (line.contains("[ WARN:") || line.contains("[WARN "))
                queueProcessOutput("<b style='color:#ccaa00;'>Warning:</b> " + line + "<br>");
            else
                queueProcessOutput("<b style='color:red;'>Error:</b> " + line + "<br>");
        };

        stderrBuffer += QString::fromLocal8Bit(process->readAllStandardError());
        QString line;
        while (take_process_output_line(stderrBuffer, line)) {
            appendOutputRunLog(line);
            writeStderrLine(line);
        }
        if (stderrBuffer.size() > 4096) {
            appendOutputRunLog(stderrBuffer);
            writeStderrLine(stderrBuffer);
            stderrBuffer.clear();
        }
    });

    connect(process, static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
        if (!stdoutBuffer.isEmpty() && !(active_backend == acmx2::Backend::Acmxvk && stable_diffusion_enabled && is_stable_diffusion_diagnostic(stdoutBuffer))) {
            queueProcessOutput(stdoutBuffer + "<br>");
        }
        appendOutputRunLog(stdoutBuffer);
        stdoutBuffer.clear();
        if (!stderrBuffer.isEmpty() && !stderrBuffer.contains("GStreamer") && !(active_backend == acmx2::Backend::Acmxvk && stable_diffusion_enabled && is_stable_diffusion_diagnostic(stderrBuffer))) {
            if (stderrBuffer.contains("[ WARN:") || stderrBuffer.contains("[WARN "))
                queueProcessOutput("<b style='color:#ccaa00;'>Warning:</b> " + stderrBuffer + "<br>");
            else
                queueProcessOutput("<b style='color:red;'>Error:</b> " + stderrBuffer + "<br>");
        }
        appendOutputRunLog(stderrBuffer);
        stderrBuffer.clear();
        flushProcessOutput();
        QString text;
        QTextStream stream(&text);
        stream << acmx2::backend_name(active_backend) << ": Exited with Code: " << exitCode;
        Log(text + "<br>");
        // Keep the selected pack in shared memory across a shader-library build
        // and across launches. The next ACMXVK process reads it on startup.
        finishOutputRunLog(exitCode, exitStatus);
        play_stop->setEnabled(false);

        if (exitStatus == QProcess::CrashExit) {
            qDebug() << acmx2::backend_name(active_backend) << "engine crashed.";
            Log("<b style='color:red;'>" + acmx2::backend_name(active_backend) + " engine crashed.</b><br>");
        }

        // Refresh the shader tree's metadata in place now that the child
        // process may have rewritten the binary shader cache.  Do not rebuild
        // the list between runs: its order belongs to the loaded library (or
        // to an explicit user reorder).
        const bool finishedBuildProcess = cacheBuildInProgress;
        if (!finishedBuildProcess) {
            refreshShaderTreeMetadata();
        }
        if (cacheBuildInProgress) {
            const PendingAcmxvkAction resume_action = pending_acmxvk_action;
            const QString pruneLibraryPath = acmxvkPruneLibraryPath;
            pending_acmxvk_action = PendingAcmxvkAction::None;
            acmxvkPruneLibraryPath.clear();
            if (active_backend == acmx2::Backend::Acmxvk) {
                if (exitCode == 0) {
                    Log(tr("ACMXVK build ready: %1").arg(acmxvk_build_directory(shader_path)));
                } else {
                    Log(tr("<b style='color:red;'>ACMXVK build failed "
                           "with exit code %1.</b>")
                            .arg(exitCode));
                }
            }
            cacheBuildInProgress = false;
            update_backend_ui();
            if (!pruneLibraryPath.isEmpty() && exitCode == 0 && exitStatus == QProcess::NormalExit) {
                QStringList sourceShaders;
                QString manifestError;
                if (!acmx2::load_shader_manifest(pruneLibraryPath, sourceShaders, manifestError)) {
                    Log(tr("<b style='color:red;'>Broken sources were "
                           "pruned, but the source manifest could not "
                           "be read: %1</b>")
                            .arg(manifestError.toHtmlEscaped()));
                    QMessageBox::warning(this,
                                         tr("Remove Broken Shaders"),
                                         tr("Broken source files were deleted, but the "
                                            "source manifest could not be updated.\n\n%1")
                                             .arg(manifestError));
                } else {
                    QStringList retainedShaders;
                    int removedCount = 0;
                    const QDir sourceDirectory(pruneLibraryPath);
                    for (const QString &shader : sourceShaders) {
                        const QString suffix = QFileInfo(shader).suffix().toLower();
                        const bool sourceEntry = suffix == QStringLiteral("frag") || suffix == QStringLiteral("comp");
                        if (sourceEntry && !QFileInfo(sourceDirectory.filePath(shader)).isFile()) {
                            ++removedCount;
                        } else {
                            retainedShaders.append(shader);
                        }
                    }

                    if (removedCount > 0 && !acmx2::write_shader_manifest(pruneLibraryPath, retainedShaders, manifestError)) {
                        Log(tr("<b style='color:red;'>Broken sources "
                               "were pruned, but the source manifest "
                               "could not be updated: %1</b>")
                                .arg(manifestError.toHtmlEscaped()));
                        QMessageBox::warning(this,
                                             tr("Remove Broken Shaders"),
                                             tr("%1 source file(s) were permanently "
                                                "deleted, but library.json could not be "
                                                "updated.\n\n%2")
                                                 .arg(removedCount)
                                                 .arg(manifestError));
                    } else {
                        if (shader_path == pruneLibraryPath)
                            loadShaders(pruneLibraryPath, true);
                        Log(tr("Remove Broken completed: %1 source "
                               "shader(s) permanently deleted.")
                                .arg(removedCount));
                        QMessageBox::information(this,
                                                 tr("Remove Broken Shaders"),
                                                 removedCount > 0 ? tr("Removed %1 broken source shader(s) "
                                                                       "and updated library.json.\n\n"
                                                                       "This deletion cannot be undone.")
                                                                        .arg(removedCount)
                                                                  : tr("The build completed and no broken "
                                                                       "source shaders were found."));
                    }
                }
            }
            if (active_backend == acmx2::Backend::Acmxvk && exitCode == 0 && exitStatus == QProcess::NormalExit && resume_action != PendingAcmxvkAction::None) {
                Log(tr("ACMXVK build succeeded; resuming the requested "
                       "action."));
                QTimer::singleShot(0, this, [this, resume_action]() {
                    if (resume_action == PendingAcmxvkAction::RunSelected) {
                        runSelected();
                    } else if (resume_action == PendingAcmxvkAction::RunAll) {
                        runAll();
                    } else if (resume_action == PendingAcmxvkAction::CopyCommand) {
                        copyCommand();
                    }
                });
            }
        }

        // Optional post-process: convert the produced HLG HDR file
        // to HDR10 via ffmpeg and stream its output to the log.
        if (!finishedBuildProcess && convert_to_hdr10 && exitCode == 0 && !output_file.isEmpty() && QFileInfo::exists(output_file)) {
            runHdr10Conversion();
        }
    });

    hdr10Process = new QProcess(this);
    connect(hdr10Process, &QProcess::readyReadStandardOutput, this, [this]() {
        QString output = QString::fromUtf8(hdr10Process->readAllStandardOutput());
        output.replace("\n", "<br>");
        this->Write(output);
    });
    connect(hdr10Process, &QProcess::readyReadStandardError, this, [this]() {
        QString output = QString::fromUtf8(hdr10Process->readAllStandardError());
        output.replace("\n", "<br>");
        // ffmpeg writes progress to stderr; render in a neutral colour rather
        // than the alarming red used for acmx2 errors.
        this->Write("<span style='color:#88aaff;'>" + output + "</span>");
    });
    connect(hdr10Process, static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus) {
        QString text;
        QTextStream stream(&text);
        stream << "ffmpeg (HDR10): Exited with Code: " << exitCode;
        Log(text + "<br>");
        play_stop->setEnabled(false);
    });

    setStyleSheet(" QMainWindow { background-color: rgb(0,0,0); }");
    camera_index = 0;
    camera_res = QSize(1280, 720);
    screen_res = QSize(0, 0);
    setGeometry(150, 150, 1280, 720);
    setWindowTitle("ACMX2 - Interface");
    QMenuBar *menuBarPtr = menuBar();

    menuBar()->setNativeMenuBar(false);
    fileMenu = menuBarPtr->addMenu(tr("File"));
    projectMenu = menuBarPtr->addMenu(tr("Project"));
    cameraMenu = menuBarPtr->addMenu(tr("Session"));
    backendMenu = menuBarPtr->addMenu(tr("Backend"));
    playbackMenu = menuBarPtr->addMenu(tr("Playback"));
    runMenu = menuBarPtr->addMenu(tr("Run"));
    listMenu = menuBarPtr->addMenu(tr("List"));
    viewMenu = menuBarPtr->addMenu(tr("View"));
    helpMenu = menuBarPtr->addMenu(tr("Help"));
    QAction *newProjectAction = projectMenu->addAction(tr("New Project..."));
    newProjectAction->setShortcut(QKeySequence::New);
    connect(newProjectAction, &QAction::triggered, this, &MainWindow::menuNewProject);
    projectMenu->addSeparator();
    QAction *saveProjectAction = projectMenu->addAction(tr("Save Project"));
    saveProjectAction->setShortcut(QKeySequence::Save);
    connect(saveProjectAction, &QAction::triggered, this, &MainWindow::menuSaveProject);
    QAction *saveProjectAsAction = projectMenu->addAction(tr("Save Project As..."));
    connect(saveProjectAsAction, &QAction::triggered, this, &MainWindow::menuSaveProjectAs);
    QAction *exportProjectAction = projectMenu->addAction(tr("Export Project..."));
    connect(exportProjectAction, &QAction::triggered, this, &MainWindow::menuExportProject);
    QAction *importPresetAction = projectMenu->addAction(tr("Load Project..."));
    importPresetAction->setShortcut(QKeySequence("Ctrl+Shift+I"));
    connect(importPresetAction, &QAction::triggered, this, &MainWindow::menuImportPreset);
    recentPresetsMenu = projectMenu->addMenu(tr("Recent Projects"));
    connect(recentPresetsMenu, &QMenu::aboutToShow, this, &MainWindow::updateRecentPresetsMenu);
    projectMenu->addSeparator();
    updateRecentPresetsMenu();
    backendActionGroup = new QActionGroup(this);
    backendActionGroup->setExclusive(true);
    backendAcmx2Action = backendMenu->addAction(tr("ACMX2"));
    backendAcmx2Action->setCheckable(true);
    backendAcmx2Action->setChecked(true);
    backendActionGroup->addAction(backendAcmx2Action);
    backendAcmxvkAction = backendMenu->addAction(tr("ACMXVK"));
    backendAcmxvkAction->setCheckable(true);
    backendActionGroup->addAction(backendAcmxvkAction);
    connect(backendAcmx2Action, &QAction::triggered, this, [this]() { set_backend(acmx2::Backend::Acmx2); });
    connect(backendAcmxvkAction, &QAction::triggered, this, [this]() { set_backend(acmx2::Backend::Acmxvk); });
    stayOnTopAction = new QAction(tr("Stay on Top"), this);
    stayOnTopAction->setShortcut(QKeySequence("Ctrl+Alt+T"));
    stayOnTopAction->setCheckable(true);
    stayOnTopAction->setChecked(false);
    connect(stayOnTopAction, &QAction::toggled, this, [this](bool checked) {
        if (checked) {
            setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);
        } else {
            setWindowFlags(windowFlags() & ~Qt::WindowStaysOnTopHint);
        }
        show();
        if (checked && QGuiApplication::platformName() == "wayland") {
            Log("Stay on Top may not work on Wayland. Launch with QT_QPA_PLATFORM=xcb for X11 support.");
        }
    });
    viewMenu->addAction(stayOnTopAction);
    QAction *metadataAction = new QAction(tr("Media Metadata Viewer..."), this);
    metadataAction->setShortcut(QKeySequence("Ctrl+Alt+V"));
    connect(metadataAction, &QAction::triggered, this, &MainWindow::menuMetadataViewer);
    viewMenu->addSeparator();
    viewMenu->addAction(metadataAction);
    fileMenu_loadLibrary = new QAction(tr("Load Library..."), this);
    fileMenu_loadLibrary->setShortcut(QKeySequence::Open);
    connect(fileMenu_loadLibrary, &QAction::triggered, this, &MainWindow::menuLoadLibrary);
    fileMenu->addAction(fileMenu_loadLibrary);
    loadRecentMenu = fileMenu->addMenu(tr("Load Recent"));
    loadRecentMenu->menuAction()->setShortcut(QKeySequence("Ctrl+Shift+O"));
    connect(loadRecentMenu, &QMenu::aboutToShow, this, &MainWindow::updateRecentLibrariesMenu);
    updateRecentLibrariesMenu();
    fileMenu->addSeparator();
    fileMenu_prop = new QAction(tr("Properties"), this);
    fileMenu_prop->setShortcut(QKeySequence("Ctrl+,"));
    fileMenu->addAction(fileMenu_prop);
    connect(fileMenu_prop, &QAction::triggered, this, &MainWindow::fileOpenProp);
    fileMenu->addSeparator();
    fileMenu_exit = new QAction(tr("Exit"), this);
    fileMenu_exit->setShortcut(QKeySequence::Quit);
    connect(fileMenu_exit, &QAction::triggered, this, &MainWindow::fileExit);
    fileMenu->addAction(fileMenu_exit);
    cameraSet = new QAction(tr("Session Properties"), this);
    cameraSet->setShortcut(QKeySequence("Ctrl+Shift+P"));
    connect(cameraSet, &QAction::triggered, this, &MainWindow::cameraSettings);
    cameraMenu->addAction(cameraSet);
    audioSet = new QAction(tr("Audio Settings"), this);
    audioSet->setShortcut(QKeySequence("Ctrl+Shift+A"));
    connect(audioSet, &QAction::triggered, this, &MainWindow::menuAudioSettings);
    cameraMenu->addAction(audioSet);
    gpuFilterAction = new QAction(tr("GPU Filter Settings"), this);
    gpuFilterAction->setShortcut(QKeySequence("Ctrl+Shift+G"));
    connect(gpuFilterAction, &QAction::triggered, this, &MainWindow::menuGPUFilterSettings);
    cameraMenu->addAction(gpuFilterAction);
    deepDreamAction = new QAction(tr("Deep Dream Settings..."), this);
    deepDreamAction->setShortcut(QKeySequence("Ctrl+Shift+D"));
    connect(deepDreamAction, &QAction::triggered, this, &MainWindow::menuDeepDreamSettings);
    cameraMenu->addAction(deepDreamAction);
    stableDiffusionAction = new QAction(tr("Stable Diffusion Settings..."), this);
    connect(stableDiffusionAction, &QAction::triggered, this, &MainWindow::menuStableDiffusionSettings);
    cameraMenu->addAction(stableDiffusionAction);
    cameraMenu->addSeparator();
    styleSheetAction = new QAction(tr("Use Custom Style"), this);
    styleSheetAction->setShortcut(QKeySequence("Ctrl+Shift+T"));
    styleSheetAction->setCheckable(true);
    styleSheetAction->setChecked(false);
    connect(styleSheetAction, &QAction::triggered, this, &MainWindow::openCustomStyleEditor);
    cameraMenu->addAction(styleSheetAction);
    runMenu_select = new QAction(tr("Run Selected"), this);
    runMenu_select->setShortcut(QKeySequence("F5"));
    connect(runMenu_select, &QAction::triggered, this, &MainWindow::runSelected);
    runMenu->addAction(runMenu_select);
    runMenu->addSeparator();
    runMenu_all = new QAction(tr("Run All"), this);
    runMenu_all->setShortcut(QKeySequence("Ctrl+E"));
    connect(runMenu_all, &QAction::triggered, this, &MainWindow::runAll);
    runMenu->addAction(runMenu_all);
    runMenu->addSeparator();
    runMenu_copyCommand = new QAction(tr("Edit Command"), this);
    runMenu_copyCommand->setShortcut(QKeySequence("Ctrl+Shift+C"));
    connect(runMenu_copyCommand, &QAction::triggered, this, &MainWindow::copyCommand);
    runMenu->addAction(runMenu_copyCommand);
    runMenu->addSeparator();
    QAction *runMenu_clearLog = new QAction(tr("Clear Log"), this);
    runMenu_clearLog->setShortcut(QKeySequence("Ctrl+L"));
    connect(runMenu_clearLog, &QAction::triggered, this, [this]() { bottomTextBox->clear(); });
    runMenu->addAction(runMenu_clearLog);
    play_repeat = new QAction(tr("Repeat"), this);
    effectPacksAction = playbackMenu->addAction(tr("Effect Packs..."));
    connect(effectPacksAction, &QAction::triggered, this, &MainWindow::menuEffectPacks);
    playbackMenu->addSeparator();
    play_repeat->setShortcut(QKeySequence("Ctrl+R"));
    play_repeat->setCheckable(true);
    play_repeat->setChecked(false);
    connect(play_repeat, &QAction::toggled, this, [this](bool) { publishRepeatStateToRunningProcess(); });
    playbackMenu->addAction(play_repeat);
    normalizedTimeAction = new QAction(tr("Normalized Time"), this);
    normalizedTimeAction->setShortcut(QKeySequence("Ctrl+Alt+N"));
    normalizedTimeAction->setCheckable(true);
    normalizedTimeAction->setChecked(false);
    normalizedTimeAction->setToolTip(tr("Advance shader time by a fixed amount per output frame."));
    connect(normalizedTimeAction, &QAction::toggled, this, [this](bool checked) {
        normalized_time = checked;
        QSettings settings("LostSideDead", "acmx2");
        settings.setValue("interface/normalized_time", checked);
        publishRuntimeSettingsToRunningProcess();
    });
    playbackMenu->addAction(normalizedTimeAction);
    play_stop = new QAction(tr("Stop"), this);
    play_stop->setShortcut(QKeySequence("Shift+F5"));
    play_stop->setEnabled(false);
    connect(play_stop, &QAction::triggered, this, [=]() {
        if (process->state() == QProcess::Running) {
            process->terminate();
        }
        if (hdr10Process && hdr10Process->state() == QProcess::Running) {
            hdr10Process->terminate();
        }
    });
    playbackMenu->addAction(play_stop);
    playbackMenu->addSeparator();
    shaderPassAction = new QAction(tr("Multi-Pass Shader Settings..."), this);
    shaderPassAction->setShortcut(QKeySequence("Ctrl+Alt+M"));
    connect(shaderPassAction, &QAction::triggered, this, &MainWindow::menuShaderPassSettings);
    playbackMenu->addAction(shaderPassAction);
    playbackMenu->addSeparator();
    playlistAction = new QAction(tr("Shader Playlist Settings..."), this);
    playlistAction->setShortcut(QKeySequence("Ctrl+Alt+P"));
    connect(playlistAction, &QAction::triggered, this, &MainWindow::menuPlaylistSettings);
    playbackMenu->addAction(playlistAction);
    playbackMenu->addSeparator();
    buildCacheAction = new QAction(tr("Rebuild Shader Cache"), this);
    buildCacheAction->setShortcut(QKeySequence("Ctrl+Alt+B"));
    connect(buildCacheAction, &QAction::triggered, this, &MainWindow::menuBuildShaderCache);
    playbackMenu->addAction(buildCacheAction);
    fixBuildAction = new QAction(tr("Fix Build"), this);
    fixBuildAction->setShortcut(QKeySequence("Ctrl+Alt+F"));
    fixBuildAction->setToolTip(tr("Build ACMXVK while omitting shaders that fail to compile."));
    connect(fixBuildAction, &QAction::triggered, this, &MainWindow::menuFixBuild);
    playbackMenu->addAction(fixBuildAction);
    cleanShaderCacheAction = new QAction(tr("Clean Shader Cache"), this);
    cleanShaderCacheAction->setShortcut(QKeySequence("Ctrl+Alt+C"));
    connect(cleanShaderCacheAction, &QAction::triggered, this, &MainWindow::menuCleanShaderCache);
    playbackMenu->addAction(cleanShaderCacheAction);
#ifdef Q_OS_MACOS
    // macOS does not support the persistent binary shader cache.
    buildCacheAction->setVisible(false);
    buildCacheAction->setEnabled(false);
    fixBuildAction->setVisible(false);
    fixBuildAction->setEnabled(false);
    cleanShaderCacheAction->setVisible(false);
    cleanShaderCacheAction->setEnabled(false);
#endif

    removeBrokenAction = new QAction(tr("Remove Broken"), this);
    removeBrokenAction->setShortcut(QKeySequence("Ctrl+Alt+R"));
    connect(removeBrokenAction, &QAction::triggered, this, &MainWindow::menuRemoveBroken);
    playbackMenu->addAction(removeBrokenAction);

    runFromCacheAction = new QAction(tr("Run from Cache"), this);
    runFromCacheAction->setShortcut(QKeySequence("Ctrl+Alt+K"));
    runFromCacheAction->setCheckable(true);
#ifdef Q_OS_MACOS
    use_shader_cache = false;
    runFromCacheAction->setChecked(false);
    runFromCacheAction->setEnabled(false);
    runFromCacheAction->setToolTip(tr("Shader binary caching is not supported on macOS."));
#else
    runFromCacheAction->setChecked(true);
#endif
    connect(runFromCacheAction, &QAction::toggled, this, [this](bool checked) {
        use_shader_cache = checked;
        if (checked) {
            Log("Shader cache enabled - will use cached shaders if available");
        } else {
            Log("Shader cache disabled - shaders will be recompiled each run");
        }
    });
    playbackMenu->addAction(runFromCacheAction);

    playbackMenu->addSeparator();
    midiSettingsAction = new QAction(tr("MIDI Settings..."), this);
    midiSettingsAction->setShortcut(QKeySequence("Ctrl+Alt+I"));
    connect(midiSettingsAction, &QAction::triggered, this, &MainWindow::menuMidiSettings);
    playbackMenu->addAction(midiSettingsAction);

    playbackMenu->addSeparator();
    watermarkAction = new QAction(tr("Watermark..."), this);
    watermarkAction->setShortcut(QKeySequence("Ctrl+Alt+W"));
    connect(watermarkAction, &QAction::triggered, this, &MainWindow::menuWatermarkSettings);
    playbackMenu->addAction(watermarkAction);

    displayFilterAction = new QAction(tr("Display"), this);
    displayFilterAction->setShortcut(QKeySequence("Ctrl+Alt+D"));
    displayFilterAction->setCheckable(true);
    displayFilterAction->setChecked(false);
    connect(displayFilterAction, &QAction::toggled, this, &MainWindow::menuToggleDisplayFilter);
    playbackMenu->addAction(displayFilterAction);

    listMenu_new = new QAction(tr("New Shader Library"), this);
    listMenu_new->setShortcut(QKeySequence("Ctrl+Shift+N"));
    connect(listMenu_new, &QAction::triggered, this, &MainWindow::newList);
    listMenu->addAction(listMenu_new);
    libraryBuilderAction = new QAction(tr("Shader Library Builder..."), this);
    libraryBuilderAction->setShortcut(QKeySequence("Ctrl+Shift+B"));
    connect(libraryBuilderAction, &QAction::triggered, this, &MainWindow::menuLibraryBuilder);
    listMenu->addAction(libraryBuilderAction);
    listMenu_shader = new QAction(tr("New Shader File..."), this);
    listMenu_shader->setShortcut(QKeySequence::New);
    connect(listMenu_shader, &QAction::triggered, this, &MainWindow::newShader);
    listMenu->addAction(listMenu_shader);
    customUniformsAction = new QAction(tr("Custom Uniforms..."), this);
    customUniformsAction->setShortcut(QKeySequence("Ctrl+U"));
    connect(customUniformsAction, &QAction::triggered, this, &MainWindow::menuCustomUniforms);
    listMenu->addAction(customUniformsAction);
    listMenu->addSeparator();
    listMenu_remove = new QAction(tr("Remove Shader"), this);
    listMenu_remove->setShortcut(QKeySequence::Delete);
    connect(listMenu_remove, &QAction::triggered, this, &MainWindow::menuRemove);
    listMenu->addAction(listMenu_remove);
    listMenu_set_current = new QAction(tr("Set Current Shader"), this);
    listMenu_set_current->setShortcut(QKeySequence("Ctrl+Return"));
    listMenu_set_current->setEnabled(false);
    connect(listMenu_set_current, &QAction::triggered, this, &MainWindow::menuSetCurrentShader);
    listMenu->addAction(listMenu_set_current);
    listMenu->addSeparator();
    listMenu_up = new QAction(tr("Shift Shader Up"), this);
    listMenu_up->setShortcut(QKeySequence("Alt+Up"));
    connect(listMenu_up, &QAction::triggered, this, &MainWindow::menuUp);
    listMenu->addAction(listMenu_up);
    listMenu_down = new QAction(tr("Shift Shader Down"), this);
    listMenu_down->setShortcut(QKeySequence("Alt+Down"));
    connect(listMenu_down, &QAction::triggered, this, &MainWindow::menuDown);
    listMenu->addAction(listMenu_down);
    listMenu_shuffle = new QAction(tr("Shuffle Shaders"), this);
    listMenu_shuffle->setShortcut(QKeySequence("Ctrl+Shift+H"));
    connect(listMenu_shuffle, &QAction::triggered, this, &MainWindow::menuShuffle);
    listMenu->addAction(listMenu_shuffle);

    listMenu_sort = new QAction(tr("Sort Shaders"), this);
    listMenu_sort->setShortcut(QKeySequence("Ctrl+Shift+S"));
    connect(listMenu_sort, &QAction::triggered, this, &MainWindow::menuSort);
    listMenu->addAction(listMenu_sort);
    listMenu->addSeparator();
    listMenu_search = new QAction(tr("Search Shaders"), this);
    listMenu_search->setShortcut(QKeySequence("Ctrl+F"));
    connect(listMenu_search, &QAction::triggered, this, &MainWindow::menuSearch);
    listMenu->addAction(listMenu_search);
    listMenu_findNext = new QAction(tr("Find Next"), this);
    listMenu_findNext->setShortcut(QKeySequence("F3"));
    connect(listMenu_findNext, &QAction::triggered, this, &MainWindow::menuFindNext);
    listMenu->addAction(listMenu_findNext);
    listMenu_findInFiles = new QAction(tr("Find in Files..."), this);
    listMenu_findInFiles->setShortcut(QKeySequence("Ctrl+Shift+F"));
    connect(listMenu_findInFiles, &QAction::triggered, this, [this]() {
        if (shader_path.isEmpty() || !QDir(shader_path).exists()) {
            QMessageBox::information(this, tr("Find in Files"), tr("Load a shader library before searching its files."));
            return;
        }

        auto *dialog = new FindShaderDialog(shader_path, this);
        connect(dialog, &FindShaderDialog::resultActivated, this, [this](const QString &filePath, int lineNumber, int columnNumber, int matchLength) { openShaderEditor(filePath, lineNumber, columnNumber, matchLength); });
        dialog->show();
        dialog->raise();
        dialog->activateWindow();
    });
    listMenu->addAction(listMenu_findInFiles);
    helpMenu_uniformReference = new QAction(tr("Built-in Uniform Reference..."), this);
    helpMenu_uniformReference->setShortcut(QKeySequence::HelpContents);
    connect(helpMenu_uniformReference, &QAction::triggered, this, &MainWindow::menuUniformReference);
    helpMenu->addAction(helpMenu_uniformReference);
    helpMenu->addSeparator();

    helpMenu_about = new QAction("About", this);
    helpMenu_about->setShortcut(QKeySequence("Shift+F1"));

    connect(helpMenu_about, &QAction::triggered, this, [=]() {
        QMessageBox box(this);
        box.setWindowTitle("About ACMX2");
        box.setWindowIcon(QIcon(":/win-icon.png"));
        const QString info = QStringLiteral("<p><b>ACMX %1</b><br>"
                                            "(C) 2026 %2 Software<br>"
                                            "<a href=\"https://lostsidedead.biz\">"
                                            "http://lostsidedead.biz</a><br>"
                                            "This software is dedicated to all that have "
                                            "experienced mental health issues.</p>")
                                 .arg(QStringLiteral(VERSION_INFO), QStringLiteral(VERSION_AUTHOR));
        box.setTextFormat(Qt::RichText);
        box.setTextInteractionFlags(Qt::TextBrowserInteraction);
        box.setText(info);
        for (QLabel *label : box.findChildren<QLabel *>())
            label->setOpenExternalLinks(true);
        QPixmap bigIcon(":/win-icon.png");
        if (!bigIcon.isNull()) {
            QPixmap resizedIcon = bigIcon.scaled(64, 64, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            box.setIconPixmap(resizedIcon);
        }
        box.exec();
    });
    helpMenu->addAction(helpMenu_about);
    customUniformDialog = new CustomUniformDialog(this);
    connect(customUniformDialog, &CustomUniformDialog::uniformsChanged, this, &MainWindow::publishCustomUniformsToRunningProcess);
    connect(customUniformDialog, &CustomUniformDialog::uniformDefinitionsChanged, this, [this]() {
        updateOpenEditorShaderContexts();
        const QString shaderName = currentShaderName();
        if (!shaderName.isEmpty())
            publishShaderReloadToRunningProcess(QDir(shader_path).filePath(shaderName));
    });
    list_view = new QTreeWidget(this);
    list_view->setColumnCount(5);
    list_view->setHeaderLabels({tr("#"), tr("Name"), tr("Last Modified"), tr("Compile Health"), tr("Type")});
    list_view->setRootIsDecorated(false);
    list_view->setUniformRowHeights(true);
    list_view->setAlternatingRowColors(false);
    list_view->setSelectionMode(QAbstractItemView::SingleSelection);
    list_view->setSelectionBehavior(QAbstractItemView::SelectRows);
    list_view->setContextMenuPolicy(Qt::CustomContextMenu);
    list_view->setSortingEnabled(false);
    list_view->setAllColumnsShowFocus(true);
    list_view->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    list_view->header()->setSectionResizeMode(1, QHeaderView::Stretch);
    list_view->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    list_view->header()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    list_view->header()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
#ifdef Q_OS_MACOS
    // macOS does not support the persistent shader cache; hide the column.
    list_view->setColumnHidden(3, true);
#endif
    list_view->setToolTip(tr("Right click while running to change the active shader."));
    bottomTextBox = new QTextEdit(this);
    bottomTextBox->setHtml("<b style='color:red;'>ACMX</b> - Interface: Loaded.");
    bottomTextBox->setReadOnly(true);
    bottomTextBox->document()->setMaximumBlockCount(2000);
    processOutputFlushTimer = new QTimer(this);
    processOutputFlushTimer->setSingleShot(true);
    processOutputFlushTimer->setInterval(35);
    connect(processOutputFlushTimer, &QTimer::timeout, this, &MainWindow::flushProcessOutput);
    connect(list_view, &QTreeWidget::doubleClicked, this, &MainWindow::listClicked);
    connect(list_view, &QTreeWidget::customContextMenuRequested, this, [this](const QPoint &pos) {
        if (!list_view)
            return;
        if (QTreeWidgetItem *item = list_view->itemAt(pos)) {
            list_view->setCurrentItem(item);
            publishSelectedShaderIndexToRunningProcess();
            if (process && process->state() == QProcess::Running) {
                return;
            }
        }
        if (listMenu) {
            listMenu->exec(list_view->viewport()->mapToGlobal(pos));
        }
    });
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);
    layout->addWidget(list_view, 3);
    layout->addWidget(bottomTextBox, 1);
    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);
    QSettings appSettings("LostSideDead");
    const QString last_project_path = appSettings.value("projects/last_open").toString();
    const bool restore_last_project = QFileInfo(last_project_path).isFile();
    active_backend = acmx2::backend_from_id(appSettings.value("interface/backend", "acmx2").toString()).value_or(acmx2::Backend::Acmx2);
    backendAcmx2Action->setChecked(active_backend == acmx2::Backend::Acmx2);
    backendAcmxvkAction->setChecked(active_backend == acmx2::Backend::Acmxvk);
    loadSessionSettings();
    baseAppStyleSheet = qApp->styleSheet();
    const QString legacyLibrary = active_backend == acmx2::Backend::Acmx2 ? appSettings.value("shaders", "").toString() : QString();
    QString path = appSettings.value(acmx2::backend_settings_key(active_backend, "library"), legacyLibrary).toString();
    path = path.trimmed();
    while (path.endsWith("/") || path.endsWith("\\")) {
        path.chop(1);
    }
    const QString legacyExecutable = active_backend == acmx2::Backend::Acmx2 ? appSettings.value("exePath", acmx2::default_backend_executable(acmx2::Backend::Acmx2)).toString() : acmx2::default_backend_executable(active_backend);
    executable_path = appSettings.value(acmx2::backend_settings_key(active_backend, "executable"), legacyExecutable).toString();
    prefix_path = appSettings.value("prefix_path", ".").toString();
    initShaderSelectionSharedMemory();
    detectCudaSupport();
    bool useCustomStyle = appSettings.value("useCustomStyle", false).toBool();
    styleSheetAction->setChecked(useCustomStyle);
    midi_enabled = appSettings.value("midiEnabled", false).toBool();
    midi_config_file = appSettings.value("midiConfigFile", "").toString();
    midi_device = appSettings.value("midiDevice", -1).toInt();
    watermark_enabled = appSettings.value("watermarkEnabled", false).toBool();
    watermark_text = appSettings.value("watermarkText", "").toString();
    watermark_r = appSettings.value("watermarkR", 255).toInt();
    watermark_g = appSettings.value("watermarkG", 0).toInt();
    watermark_b = appSettings.value("watermarkB", 150).toInt();
    display_filter_enabled = appSettings.value("displayFilter", false).toBool();
    autopilot_frames = appSettings.value("playlistAutopilotFrames", 4).toInt();
    if (autopilot_frames < 4) {
        autopilot_frames = 4;
    }
    autopilot_random = appSettings.value("playlistAutopilotRandom", false).toBool();
    if (displayFilterAction) {
        QSignalBlocker blocker(displayFilterAction);
        displayFilterAction->setChecked(display_filter_enabled);
    }
    publishRuntimeSettingsToRunningProcess();
    if (!path.isEmpty() && !restore_last_project) {
        QFileInfo pathInfo(path);
        if (pathInfo.exists() && pathInfo.isDir() && acmx2::shader_manifest_exists(path)) {
            QString backendError;
            const std::optional<acmx2::Backend> libraryBackend = acmx2::shader_manifest_backend(path, backendError);
            if (!backendError.isEmpty()) {
                Log("Warning: Saved shader library backend metadata is invalid: " + backendError);
            } else if (libraryBackend && *libraryBackend != active_backend) {
                Log(tr("Warning: Saved shader library targets %1 while the "
                       "active backend is %2: %3")
                        .arg(acmx2::backend_name(*libraryBackend), acmx2::backend_name(active_backend), path));
            } else {
                shader_path = path;
                loadShaders(path);
                addRecentLibrary(path);
                Log("Successfully loaded saved shader path");
            }
        } else {
            QString errorMsg = "Warning: Saved shader path is invalid: " + path + " - ";
            if (!pathInfo.exists()) {
                errorMsg += "directory does not exist";
            } else if (!pathInfo.isDir()) {
                errorMsg += "path is not a directory";
            } else if (!acmx2::shader_manifest_exists(path)) {
                errorMsg += "library.json or index.txt not found in directory";
            }
            Log(errorMsg);
        }
    }
    update_backend_ui();
    const QString defaultCustomStyleSheet = acmx2::defaultCustomStyleSheet();
    customStyleSheet = appSettings.value("customStyleSheet", defaultCustomStyleSheet).toString();

    applyCustomStyleSheet(useCustomStyle);

    if (restore_last_project) {
        QTimer::singleShot(0, this, [this, last_project_path]() { importPreset(last_project_path); });
    } else if (!last_project_path.isEmpty()) {
        appSettings.remove("projects/last_open");
        appSettings.sync();
    }
}

void MainWindow::loadSessionSettings() {
    QSettings settings("LostSideDead", "acmx2");
    QSettings application_settings("LostSideDead");
    parallel_build_jobs(application_settings);

    const QString inputMode = settings.value("interface/input_mode", "camera").toString();
    const bool videoMode = inputMode == "video";
    const bool graphicsMode = inputMode == "graphic";
    const bool cameraMode = !videoMode && !graphicsMode;

    camera_index = static_cast<unsigned int>(std::max(0, settings.value("interface/camera_device", 0).toInt()));
    camera_res = storedResolution(settings, "interface/camera_resolution", QSize(1280, 720), false);
    screen_res = storedResolution(settings, "interface/screen_resolution", QSize(0, 0), true);

    output_fps = settings.value("interface/camera_fps", 30.0).toDouble();
    if (output_fps <= 0.0)
        output_fps = 30.0;

    video_file = videoMode ? settings.value("interface/input_video", "").toString() : QString();
    graphics_file = graphicsMode ? settings.value("interface/graphics_file", "").toString() : QString();

    const bool saveOutput = settings.value("interface/save_output", false).toBool();
    output_file = saveOutput ? settings.value("interface/output_video", "").toString() : QString();
    save_output_log = saveOutput && settings.value("interface/save_output_log", false).toBool();
    full_screen_value = settings.value("interface/fullscreen", false).toBool();
    enable_vsync = settings.value("interface/acmxvk_vsync", false).toBool();
    monitor_index = std::max(0, settings.value("interface/acmxvk_monitor", 0).toInt());
    copy_audio = videoMode && saveOutput && settings.value("interface/copy_audio", false).toBool();

    cache_enabled = !graphicsMode && settings.value("interface/texture_cache", false).toBool();
    cache_delay = settings.value("interface/cache_delay", 1).toInt();
    cache_size = std::clamp(settings.value("interface/cache_size", 8).toInt(), 1, 64);
    use_yuv = cameraMode && settings.value("interface/use_yuv", false).toBool();

    convert_to_hdr10 = videoMode && saveOutput && settings.value("interface/convert_to_hdr10", false).toBool();
    enable_3d = settings.value("interface/enable_3d", false).toBool();
    model_file = settings.value("interface/model_file", "cube.mxmod.z").toString();
    onnx_model_enabled = settings.value("interface/use_onnx_model", false).toBool();
    onnx_model = settings.value("interface/onnx_model_file", "").toString();
    deep_dream_enabled = settings.value("deep_dream/enabled", false).toBool();
    deep_dream_model = settings.value("deep_dream/model_file", QString()).toString();
    deep_dream_layer = settings.value("deep_dream/layer", "relu4_2").toString();
    deep_dream_iterations = std::clamp(settings.value("deep_dream/iterations", 1).toInt(), 1, 100);
    deep_dream_strength = std::clamp(settings.value("deep_dream/strength", 0.05).toDouble(), 0.0001, 10.0);
    deep_dream_feedback = std::clamp(settings.value("deep_dream/feedback", 0.9).toDouble(), 0.0, 0.99);
    deep_dream_zoom = std::clamp(settings.value("deep_dream/zoom", 1.01).toDouble(), 0.9, 1.1);
    deep_dream_rotation = std::clamp(settings.value("deep_dream/rotation", 0.1).toDouble(), -5.0, 5.0);
    deep_dream_maximum_dimension = settings.value("deep_dream/maximum_dimension", 512).toInt();
    if (deep_dream_maximum_dimension != 0) {
        deep_dream_maximum_dimension = std::clamp(deep_dream_maximum_dimension, 64, 4096);
    }
    deep_dream_fp16 = settings.value("deep_dream/fp16", false).toBool();
    deep_dream_channel = std::clamp(settings.value("deep_dream/channel", -1).toInt(), -1, 65535);
    deep_dream_octaves = std::clamp(settings.value("deep_dream/octaves", 1).toInt(), 1, 8);
    deep_dream_octave_scale = std::clamp(settings.value("deep_dream/octave_scale", 1.4).toDouble(), 1.1, 3.0);
    deep_dream_jitter = std::clamp(settings.value("deep_dream/jitter", 0).toInt(), 0, 64);
    deep_dream_smoothing = std::clamp(settings.value("deep_dream/smoothing", 0).toInt(), 0, 16);
    deep_dream_gpu_filter_first = settings.value("deep_dream/gpu_filter_first", false).toBool();
    deep_dream_original = settings.value("deep_dream/deep_original", false).toBool();
    if (deep_dream_original) {
        deep_dream_feedback = 0.0;
        deep_dream_zoom = 1.0;
        deep_dream_rotation = 0.0;
    }
    stable_diffusion_enabled = settings.value("stable_diffusion/enabled", false).toBool();
    stable_diffusion_model = settings.value("stable_diffusion/model_file").toString();
    stable_diffusion_lora_files = settings.value("stable_diffusion/lora_files").toStringList();
    const QStringList stable_diffusion_lora_multiplier_values = settings.value("stable_diffusion/lora_multipliers").toStringList();
    for (int index = 0; index < stable_diffusion_lora_files.size(); ++index) {
        bool multiplier_ok = false;
        const double multiplier = index < stable_diffusion_lora_multiplier_values.size() ? stable_diffusion_lora_multiplier_values.at(index).toDouble(&multiplier_ok) : 1.0;
        stable_diffusion_lora_multipliers.append(multiplier_ok ? std::clamp(multiplier, -10.0, 10.0) : 1.0);
    }
    stable_diffusion_upscale_only = settings.value("stable_diffusion/upscale_only", false).toBool();
    if (stable_diffusion_upscale_only || settings.value("stable_diffusion/server_upscale", false).toBool()) {
        stable_diffusion_upscale_model = settings.value("stable_diffusion/upscale_model_file").toString();
    }
    stable_diffusion_prompt = settings.value("stable_diffusion/prompt").toString();
    stable_diffusion_negative_prompt = settings.value("stable_diffusion/negative_prompt").toString();
    stable_diffusion_server = settings.value("stable_diffusion/server", default_stable_diffusion_server()).toString();
    stable_diffusion_server_arguments = settings.value("stable_diffusion/server_arguments").toString();
    stable_diffusion_server_port = std::clamp(settings.value("stable_diffusion/port", 1234).toInt(), 1024, 65535);
    stable_diffusion_width = std::clamp(settings.value("stable_diffusion/width", 576).toInt(), 64, 2048);
    stable_diffusion_height = std::clamp(settings.value("stable_diffusion/height", 320).toInt(), 64, 2048);
    stable_diffusion_upscale_working_width = std::clamp(settings.value("stable_diffusion/upscale_working_width", 0).toInt(), 0, 4096);
    stable_diffusion_upscale_working_height = std::clamp(settings.value("stable_diffusion/upscale_working_height", 0).toInt(), 0, 4096);
    stable_diffusion_steps = std::clamp(settings.value("stable_diffusion/steps", 12).toInt(), 1, 150);
    stable_diffusion_strength = std::clamp(settings.value("stable_diffusion/strength", 0.35).toDouble(), 0.01, 1.0);
    stable_diffusion_cfg_scale = std::clamp(settings.value("stable_diffusion/cfg_scale", 5.0).toDouble(), 0.0, 50.0);
    stable_diffusion_seed = settings.value("stable_diffusion/seed", 1234).toInt();
    stable_diffusion_sampler = settings.value("stable_diffusion/sampler", "euler_a").toString();
    stable_diffusion_scheduler = settings.value("stable_diffusion/scheduler", "discrete").toString();
    stable_diffusion_upscale = settings.value("stable_diffusion/upscale", false).toBool() && !stable_diffusion_upscale_only;
    cuda_device = settings.value("interface/cuda_device", 0).toInt();
    time_speed = settings.value("interface/time_speed", 1.0).toFloat();
    normalized_time = settings.value("interface/normalized_time", false).toBool();
    if (normalizedTimeAction) {
        QSignalBlocker blocker(normalizedTimeAction);
        normalizedTimeAction->setChecked(normalized_time);
    }
    duration_limit_enabled = settings.value("interface/duration_enabled", false).toBool();
    max_duration = settings.value("interface/duration_seconds", 60.0).toDouble();
    max_size_limit_enabled = settings.value("interface/max_size_enabled", false).toBool();
    max_size_mb = settings.value("interface/max_size_mb", 500.0).toDouble();
    cross_fade_duration = settings.value("interface/crossfade", 0.5).toFloat();
    flip_enabled = settings.value("interface/flip", false).toBool();
    rotate_enabled = settings.value("interface/rotate", false).toBool();
    rotation_mode = settings.value("interface/rotation_mode", "clockwise").toString();
    png_output = active_backend == acmx2::Backend::Acmxvk && settings.value("interface/write_png", false).toBool();
    png_output_directory = settings.value("interface/png_output_directory", "").toString();
    png_level = std::clamp(settings.value("interface/png_level", 6).toInt(), 1, 9);
    generate_enabled = settings.value("interface/generate_enabled", false).toBool();
    generate_interval = settings.value("interface/generate_interval", 30).toInt();

    encode_preset = settings.value("recording/preset", "medium").toString();
    encode_tune = settings.value("recording/tune", "").toString();
    encode_crf = settings.value("recording/crf", 18).toInt();
    encode_rate_control = settings.value("recording/rate_control", "quality").toString();
    encode_bitrate = settings.value("recording/bitrate", "10M").toString();
    encode_codec = settings.value("recording/codec", "auto").toString();
    encode_parameters = settings.value("recording/parameters", "").toString();
    encode_realtime = settings.value("recording/realtime", false).toBool();
    encode_no_drop = !cameraMode && settings.value("recording/no_drop", false).toBool();
    encode_constant_frame_rate = settings.value("recording/constant_frame_rate", false).toBool();
    encode_fill_pts_gaps = settings.value("recording/fill_pts_gaps", false).toBool();
    maximize_fps = settings.value("interface/acmxvk_maximize_fps", false).toBool();
    use_source_fps = settings.value("interface/acmxvk_use_source_fps", false).toBool();
    use_source_audio = use_source_fps && settings.value("interface/acmxvk_use_source_audio", false).toBool();
    extra_arguments = settings.value("interface/extra_arguments", QString()).toString();
}

void MainWindow::applyMainViewStyles(bool customStyleEnabled) {
    if (list_view) {
        QFont listFont("Courier New");
        listFont.setStyleHint(QFont::Monospace);
        listFont.setPointSize(12);
        list_view->setFont(listFont);

        if (customStyleEnabled) {
            list_view->setStyleSheet("");
        } else {
            list_view->setStyleSheet("QTreeWidget { background-color: black; color: white; font-size: 13px;"
                                     " font-family: 'Courier New', Courier, monospace; }"
                                     "QHeaderView::section { background-color: #110000; color: lime;"
                                     " font-family: 'Courier New', Courier, monospace; padding: 4px;"
                                     " border: 1px solid #330000; }");
        }
    }

    if (bottomTextBox) {
        QFont logFont("Courier New");
        logFont.setStyleHint(QFont::Monospace);
        logFont.setPointSize(11);
        bottomTextBox->setFont(logFont);

        if (customStyleEnabled) {
            bottomTextBox->setStyleSheet("");
        } else {
            bottomTextBox->setStyleSheet("QTextEdit { background-color: black; color: lime; font-size: 13px;"
                                         " font-family: 'Courier New', Courier, monospace; }");
        }
    }
}

void MainWindow::applyCustomStyleSheet(bool enable) {
    QSettings appSettings("LostSideDead");
    appSettings.setValue("useCustomStyle", enable);

    if (baseAppStyleSheet.isEmpty()) {
        baseAppStyleSheet = qApp->styleSheet();
    }

    if (enable) {
        qApp->setStyleSheet(customStyleSheet);
    } else {
        qApp->setStyleSheet(baseAppStyleSheet);
    }

    // Keep this window clean so it follows the global app style consistently.
    setStyleSheet("");
    applyMainViewStyles(enable);
}

void MainWindow::openCustomStyleEditor() {
    QSettings appSettings("LostSideDead");
    const bool currentlyEnabled = appSettings.value("useCustomStyle", false).toBool();
    const QString lastPresetName = appSettings.value("customStylePreset", "Current Style").toString();

    auto makePalette = [](const char *winBg, const char *winFg, const char *accent, const char *fieldBg, const char *fieldFg, const char *fieldBorder, const char *btnBg, const char *btnHover, const char *btnFg, const char *menuBg, const char *menuFg, const char *menuSelBg, const char *menuSelFg, const char *selBg, const char *border) {
        acmx2::CustomStylePalette p;
        p.windowBg = winBg;
        p.windowFg = winFg;
        p.accent = accent;
        p.fieldBg = fieldBg;
        p.fieldFg = fieldFg;
        p.fieldBorder = fieldBorder;
        p.buttonBg = btnBg;
        p.buttonHover = btnHover;
        p.buttonFg = btnFg;
        p.menuBg = menuBg;
        p.menuFg = menuFg;
        p.menuSelBg = menuSelBg;
        p.menuSelFg = menuSelFg;
        p.selectionBg = selBg;
        p.border = border;
        return acmx2::buildStyleSheet(p);
    };

    const std::array<QPair<QString, QString>, 26> presetStyles = {{{"Current Style", customStyleSheet},
                                                                   {"Light: Blue & White", makePalette("#f6fbff", "#143a5c", "#2d7cc4", "#ffffff", "#123b61", "#9cc6ea", "#2d7cc4", "#2368a6", "#ffffff", "#eaf5ff", "#143a5c", "#cfe6ff", "#0b2e4d", "#bcdcff", "1px solid #9cc6ea")},
                                                                   {"Light: Slate", makePalette("#f5f7fa", "#1f2a37", "#4b5563", "#ffffff", "#1f2937", "#b6c3d4", "#4b5563", "#374151", "#ffffff", "#e8edf4", "#1f2a37", "#d2dbe7", "#111827", "#cdd5e0", "1px solid #b6c3d4")},
                                                                   {"Light: White & Red", makePalette("#fffdfd", "#5b1515", "#d63b3b", "#ffffff", "#5a1a1a", "#e8bcbc", "#d63b3b", "#bc2f2f", "#ffffff", "#fff4f4", "#5b1515", "#ffdede", "#4b0f0f", "#ffd1d1", "1px solid #e8bcbc")},
                                                                   {"Light: White & Green", makePalette("#fcfffc", "#164529", "#2e9d57", "#ffffff", "#1a4f2f", "#b8dfc7", "#2e9d57", "#25824a", "#ffffff", "#f1fbf4", "#164529", "#d6f3df", "#11361f", "#c9eecf", "1px solid #b8dfc7")},
                                                                   {"Light: White & Blue", makePalette("#fcfdff", "#16395f", "#2f6ed7", "#ffffff", "#1b446f", "#b7d0f0", "#2f6ed7", "#285db7", "#ffffff", "#f1f6ff", "#16395f", "#d9e8ff", "#102b49", "#cddfff", "1px solid #b7d0f0")},
                                                                   {"Light: White & Cyan", makePalette("#fbfeff", "#12404a", "#1ea9bf", "#ffffff", "#14505d", "#b8e2ea", "#1ea9bf", "#198da0", "#ffffff", "#effbfe", "#12404a", "#d5f3f8", "#0e3138", "#c7edf4", "1px solid #b8e2ea")},
                                                                   {"Light: White & Amber", makePalette("#fffefb", "#5a3a12", "#d18b1f", "#ffffff", "#644317", "#ead7b6", "#d18b1f", "#b37518", "#ffffff", "#fff9ed", "#5a3a12", "#ffebcb", "#4a2f0f", "#ffe2b5", "1px solid #ead7b6")},
                                                                   {"Dark: Crimson", makePalette("#0f0608", "#ff637d", "#a02949", "#1b0b10", "#ff8fa3", "#7f2036", "#6f1630", "#8a1f3d", "#ffdfe6", "#16090d", "#ff637d", "#52111f", "#ffd5dc", "#52111f", "2px solid #a02949")},
                                                                   {"Dark: Emerald", makePalette("#06110c", "#7af7c2", "#2c8e68", "#0d1e16", "#95ffd0", "#2c8e68", "#1c6a4d", "#258961", "#dcfff2", "#08160f", "#7af7c2", "#12402d", "#d9fff0", "#12402d", "2px solid #2c8e68")},
                                                                   {"Dark: Indigo", makePalette("#070713", "#c6c8ff", "#5362ba", "#121634", "#d8daff", "#4956a5", "#36439a", "#4453b4", "#eef0ff", "#0d1022", "#c6c8ff", "#232a5a", "#eef0ff", "#232a5a", "2px solid #5362ba")},
                                                                   {"Dark: Black & Red", makePalette("#050505", "#ff4d4d", "#d90000", "#120808", "#ff7b7b", "#b50000", "#2a0c0c", "#3a1010", "#ffd6d6", "#0b0707", "#ff5a5a", "#6b1111", "#ffe9e9", "#5a0c0c", "2px solid #d90000")},
                                                                   {"Dark: Black & Green", makePalette("#040704", "#6dfb88", "#22b44a", "#0a140b", "#a8ffbe", "#1d9a3e", "#12331b", "#164425", "#e1ffe8", "#08100a", "#74ff95", "#12331b", "#e7ffed", "#10381d", "2px solid #22b44a")},
                                                                   {"Dark: Black & Blue", makePalette("#04060a", "#81b9ff", "#2f6ed7", "#0a1222", "#b4d4ff", "#2a5eb7", "#132749", "#1a3260", "#e7f1ff", "#070d1a", "#8cc0ff", "#1a3260", "#eef5ff", "#17335f", "2px solid #2f6ed7")},
                                                                   {"Dark: Black & Cyan", makePalette("#030809", "#7defff", "#1ba8c3", "#09161a", "#b8f7ff", "#1990a7", "#10323a", "#14414b", "#e7fbff", "#071015", "#89f3ff", "#0f3943", "#e8fcff", "#0f3943", "2px solid #1ba8c3")},
                                                                   {"Dark: Black & Amber", makePalette("#090704", "#ffd77a", "#d88c1d", "#1a1308", "#ffe7b4", "#bf7a19", "#3d2810", "#523618", "#fff3db", "#130e07", "#ffdf8a", "#5a3a16", "#fff4df", "#5a3a16", "2px solid #d88c1d")},
                                                                   {"Light: Lavender Mist", makePalette("#f8f6ff", "#302653", "#7157c8", "#ffffff", "#34295b", "#c9bdea", "#7157c8", "#5d45ae", "#ffffff", "#eee9ff", "#302653", "#ded5ff", "#241a48", "#d9d0ff", "1px solid #c9bdea")},
                                                                   {"Light: Rose Quartz", makePalette("#fff8fa", "#532535", "#c25578", "#ffffff", "#5b293c", "#e8c1cf", "#c25578", "#a94465", "#ffffff", "#fff0f4", "#532535", "#f6d7e1", "#411a28", "#f1ccd8", "1px solid #e8c1cf")},
                                                                   {"Light: Sandstone", makePalette("#fbf7ef", "#493728", "#a66a3f", "#fffdf8", "#4f3929", "#d9c3aa", "#a66a3f", "#895431", "#ffffff", "#f3eadc", "#493728", "#ead8c1", "#35251a", "#e5d1b7", "1px solid #d9c3aa")},
                                                                   {"Light: Mint & Navy", makePalette("#f3fbf8", "#173a3c", "#2b8c7f", "#ffffff", "#173a3c", "#addbd2", "#1d5962", "#287681", "#ffffff", "#e5f6f1", "#173a3c", "#c8eee5", "#102f34", "#bde5dc", "1px solid #addbd2")},
                                                                   {"Light: High Contrast", makePalette("#ffffff", "#111111", "#005fcc", "#ffffff", "#000000", "#4d4d4d", "#111111", "#005fcc", "#ffffff", "#f0f0f0", "#000000", "#005fcc", "#ffffff", "#9dccff", "2px solid #111111")},
                                                                   {"Dark: Cyberpunk Neon", makePalette("#070513", "#f3e7ff", "#ff2bd6", "#100c24", "#5ffbf1", "#6e4cff", "#2a145c", "#ff2bd6", "#ffffff", "#0c081c", "#5ffbf1", "#381b72", "#ffffff", "#381b72", "2px solid #ff2bd6")},
                                                                   {"Dark: Dracula", makePalette("#282a36", "#f8f8f2", "#bd93f9", "#21222c", "#f8f8f2", "#6272a4", "#44475a", "#6272a4", "#f8f8f2", "#21222c", "#f8f8f2", "#44475a", "#f8f8f2", "#44475a", "1px solid #6272a4")},
                                                                   {"Dark: Nord Frost", makePalette("#2e3440", "#eceff4", "#88c0d0", "#3b4252", "#eceff4", "#4c566a", "#4c566a", "#5e81ac", "#eceff4", "#242933", "#d8dee9", "#434c5e", "#eceff4", "#434c5e", "1px solid #88c0d0")},
                                                                   {"Dark: Solarized", makePalette("#002b36", "#93a1a1", "#b58900", "#073642", "#eee8d5", "#586e75", "#07576b", "#268bd2", "#fdf6e3", "#00242d", "#93a1a1", "#07576b", "#fdf6e3", "#07576b", "1px solid #586e75")},
                                                                   {"Dark: Graphite Orange", makePalette("#171717", "#f2f2f2", "#ff8a3d", "#242424", "#f7f7f7", "#5f5f5f", "#3a3a3a", "#ff8a3d", "#ffffff", "#202020", "#f2f2f2", "#59311c", "#ffffff", "#59311c", "2px solid #ff8a3d")}}};

    if (styleSheetAction) {
        QSignalBlocker blocker(styleSheetAction);
        styleSheetAction->setChecked(currentlyEnabled);
    }

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Custom Style Editor"));
    dialog.resize(900, 640);
    // Keep the editor dialog on the application stylesheet so Apply updates it live.
    dialog.setStyleSheet("");

    auto *layout = new QVBoxLayout(&dialog);
    auto *topRow = new QHBoxLayout();
    auto *enableCheck = new QCheckBox(tr("Use custom style"), &dialog);
    enableCheck->setChecked(currentlyEnabled);
    auto *presetLabel = new QLabel(tr("Preset:"), &dialog);
    auto *presetCombo = new QComboBox(&dialog);
    for (const auto &preset : presetStyles) {
        presetCombo->addItem(preset.first);
    }
    int presetIndex = 0;
    for (int i = 0; i < static_cast<int>(presetStyles.size()); ++i) {
        if (presetStyles[static_cast<std::size_t>(i)].first == lastPresetName) {
            presetIndex = i;
            break;
        }
    }
    presetCombo->setCurrentIndex(presetIndex);

    auto *editor = new QPlainTextEdit(&dialog);
    editor->setPlainText(customStyleSheet);
    editor->setLineWrapMode(QPlainTextEdit::NoWrap);
    editor->setPlaceholderText(tr("Enter a Qt stylesheet (QSS) for ACMX2 interface..."));
    {
        QFont qssFont("Courier New");
        qssFont.setStyleHint(QFont::Monospace);
        qssFont.setPointSize(10);
        editor->setFont(qssFont);
    }

    auto *buttonBox = new QDialogButtonBox(&dialog);
    auto *applyButton = buttonBox->addButton(tr("Apply"), QDialogButtonBox::ApplyRole);
    auto *saveButton = buttonBox->addButton(tr("Save"), QDialogButtonBox::ActionRole);
    auto *closeButton = buttonBox->addButton(QDialogButtonBox::Close);

    topRow->addWidget(enableCheck);
    topRow->addSpacing(12);
    topRow->addWidget(presetLabel);
    topRow->addWidget(presetCombo, 1);
    layout->addLayout(topRow);
    layout->addWidget(editor, 1);
    layout->addWidget(buttonBox);

    connect(presetCombo, &QComboBox::currentTextChanged, &dialog, [editor, &presetStyles, &appSettings](const QString &name) {
        for (const auto &preset : presetStyles) {
            if (preset.first == name) {
                editor->setPlainText(preset.second);
                appSettings.setValue("customStylePreset", name);
                break;
            }
        }
    });

    auto applyEditorStyle = [this, &dialog, enableCheck, editor, presetCombo]() {
        customStyleSheet = editor->toPlainText();
        QSettings styleSettings("LostSideDead");
        styleSettings.setValue("customStyleSheet", customStyleSheet);
        styleSettings.setValue("customStylePreset", presetCombo->currentText());
        styleSettings.setValue("useCustomStyle", enableCheck->isChecked());
        applyCustomStyleSheet(enableCheck->isChecked());
        // Ensure no local override remains so the dialog always follows qApp style.
        dialog.setStyleSheet("");
        if (styleSheetAction) {
            QSignalBlocker blocker(styleSheetAction);
            styleSheetAction->setChecked(enableCheck->isChecked());
        }
    };

    connect(applyButton, &QPushButton::clicked, &dialog, applyEditorStyle);
    connect(saveButton, &QPushButton::clicked, &dialog, applyEditorStyle);
    connect(closeButton, &QPushButton::clicked, &dialog, &QDialog::accept);

    dialog.exec();
}

void MainWindow::newList() {
    LibraryWindow library(active_backend, this);

    if (library.exec() == QDialog::Accepted) {
        loadLibraryPath(library.getShaderPath());
    }
}

void MainWindow::menuLibraryBuilder() {
    if (libraryBuilderDialog && libraryBuilderDialog->selectedBackend() != active_backend) {
        libraryBuilderDialog->close();
        libraryBuilderDialog = nullptr;
    }
    if (libraryBuilderDialog) {
        libraryBuilderDialog->show();
        libraryBuilderDialog->raise();
        libraryBuilderDialog->activateWindow();
        return;
    }

    libraryBuilderDialog = new LibraryBuilderDialog(active_backend, this);
    libraryBuilderDialog->setAttribute(Qt::WA_DeleteOnClose);
    connect(libraryBuilderDialog, &LibraryBuilderDialog::libraryExported, this, [this](const QString &directory) {
        if (loadLibraryPath(directory))
            Log(tr("Loaded exported shader library: %1").arg(shader_path));
    });
    libraryBuilderDialog->show();
    libraryBuilderDialog->raise();
    libraryBuilderDialog->activateWindow();
}

void MainWindow::menuSearch() {
    bool ok;
    QString searchText = QInputDialog::getText(this, tr("Search Shaders"), tr("Enter shader name to search:"), QLineEdit::Normal, lastSearchText, &ok);

    if (!ok || searchText.isEmpty()) {
        return;
    }

    lastSearchText = searchText;
    lastFoundIndex = -1;
    if (items.isEmpty()) {
        QMessageBox::information(this, tr("Search Shaders"), tr("No shaders are loaded."));
        return;
    }
    int foundIndex = -1;

    for (int i = 0; i < items.size(); ++i) {
        if (items[i].compare(searchText, Qt::CaseInsensitive) == 0) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex == -1) {
        for (int i = 0; i < items.size(); ++i) {
            if (items[i].contains(searchText, Qt::CaseInsensitive)) {
                foundIndex = i;
                break;
            }
        }
    }

    if (foundIndex != -1) {
        lastFoundIndex = foundIndex;
        selectShaderRow(foundIndex);
        Log("Found shader: " + items[foundIndex] + " at index " + QString::number(foundIndex));
    } else {
        QMessageBox::information(this, tr("Not Found"), tr("Shader \"") + searchText + tr("\" not found in the list."));
        Log("Shader not found: " + searchText);
    }
}

void MainWindow::menuFindNext() {
    if (lastSearchText.isEmpty()) {
        QMessageBox::information(this, tr("No Search"), tr("Please perform a search first (Ctrl+F)."));
        return;
    }

    if (items.isEmpty()) {
        return;
    }

    int foundIndex = -1;
    int startIndex = (lastFoundIndex + 1) % items.size();

    for (int i = startIndex; i < items.size(); ++i) {
        if (items[i].contains(lastSearchText, Qt::CaseInsensitive)) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex == -1 && startIndex > 0) {
        for (int i = 0; i < startIndex; ++i) {
            if (items[i].contains(lastSearchText, Qt::CaseInsensitive)) {
                foundIndex = i;
                break;
            }
        }
    }

    if (foundIndex != -1) {
        lastFoundIndex = foundIndex;
        selectShaderRow(foundIndex);
        Log("Found next: " + items[foundIndex] + " at index " + QString::number(foundIndex));
    } else {
        QMessageBox::information(this, tr("No More Results"), tr("No more matches for \"") + lastSearchText + tr("\"."));
        Log("No more matches for: " + lastSearchText);
    }
}

void MainWindow::newShader() {
    if (shader_path.isEmpty() || !acmx2::shader_manifest_exists(shader_path)) {
        QMessageBox::information(this, tr("New Shader File"), tr("Create or load a shader library first."));
        return;
    }
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString typeError;
        const auto libraryType = acmx2::shader_manifest_library_type(shader_path, typeError);
        if (!typeError.isEmpty()) {
            QMessageBox::warning(this, tr("New Shader File"), typeError);
            return;
        }
        if (libraryType && *libraryType == acmx2::ShaderLibraryType::Runtime) {
            QMessageBox::information(this,
                                     tr("New Shader File"),
                                     tr("New ACMXVK shaders must be added to a source library, not "
                                        "a compiled SPIR-V runtime library."));
            return;
        }
    }
    ShaderDialog new_shader(active_backend, this);
    new_shader.setShaderPath(shader_path);
    if (new_shader.exec() == QDialog::Accepted) {
        QSettings appSettings("LostSideDead");
        appSettings.setValue(acmx2::backend_settings_key(active_backend, "library"), shader_path);
        if (active_backend == acmx2::Backend::Acmx2)
            appSettings.setValue("shaders", shader_path);
        appSettings.sync();
        loadShaders(shader_path, true);
    }
}

void MainWindow::menuRemove() {
    int row = currentShaderRow();
    if (row < 0 || row >= items.size())
        return;
    const QString shaderName = items.at(row);
    QString manifestError;
    if (!acmx2::remove_shader_manifest_entry(shader_path, shaderName, manifestError)) {
        QMessageBox::warning(this, tr("Could Not Remove Shader"), manifestError);
        Log(tr("Could not remove %1 from the library manifest: %2").arg(shaderName, manifestError));
        return;
    }
    items.removeAt(row);
    populateShaderTree();
    indexTimestamp = acmx2::shader_manifest_last_modified(shader_path);
    activeShaderManifestPath = acmx2::shader_manifest_path(shader_path);
    Log(tr("Removed shader from library manifest: %1").arg(shaderName));
    loadShaders(shader_path, true);
}

void MainWindow::menuSetCurrentShader() {
    if (!process || process->state() != QProcess::Running)
        return;
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size()) {
        Log("No shader selected.");
        return;
    }
    publishSelectedShaderIndexToRunningProcess();
}

void MainWindow::updateIndex() {
    QStringList writtenItems;
    const int rowCount = items.size();

    for (int row = 0; row < rowCount; ++row) {
        const QString shaderName = items.at(row).trimmed();
        if (shaderName.isEmpty() || writtenItems.contains(shaderName, Qt::CaseInsensitive)) {
            continue;
        }

        QString fullPath = shader_path + "/" + shaderName;
        QFileInfo fileInfo(fullPath);
        if (fileInfo.exists() && fileInfo.isFile()) {
            writtenItems.append(shaderName);
        } else {
            Log("Warning: File no longer exists, removing from list: " + shaderName);
        }
    }
    QString manifestError;
    QStringList existingItems;
    if (acmx2::load_shader_manifest(shader_path, existingItems, manifestError) && existingItems == writtenItems) {
        indexTimestamp = acmx2::shader_manifest_last_modified(shader_path);
        activeShaderManifestPath = acmx2::shader_manifest_path(shader_path);
        return;
    }
    manifestError.clear();
    if (!acmx2::write_shader_manifest(shader_path, writtenItems, manifestError)) {
        Log("Failed to update shader manifest: " + manifestError);
        return;
    }
    indexTimestamp = acmx2::shader_manifest_last_modified(shader_path);
    activeShaderManifestPath = acmx2::shader_manifest_path(shader_path);

    if (writtenItems.size() != rowCount) {
        items = writtenItems;
        populateShaderTree();
        Log("Updated shader list, removed " + QString::number(rowCount - writtenItems.size()) + " non-existent files");
    }
}

void MainWindow::menuUp() {
    const int row = currentShaderRow();
    if (row <= 0 || row >= items.size())
        return;
    items.swapItemsAt(row, row - 1);
    populateShaderTree();
    selectShaderRow(row - 1);
    updateIndex();
}

void MainWindow::menuDown() {
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size() - 1)
        return;
    items.swapItemsAt(row, row + 1);
    populateShaderTree();
    selectShaderRow(row + 1);
    updateIndex();
}

QString MainWindow::readFileContents(const QString &filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        Log("Failed to open file: " + filePath);
        return QString();
    }

    QTextStream in(&file);
    QString contents = in.readAll();
    file.close();
    return contents;
}

void MainWindow::listClicked(const QModelIndex &i) {
    if (!i.isValid())
        return;
    const int row = i.row();
    if (row < 0 || row >= items.size())
        return;
    QString itemText = sanitizeShaderName(items.at(row));
    if (itemText.isEmpty()) {
        Log("Invalid shader name");
        return;
    }
    QString filePath = shader_path + "/" + itemText;
    openShaderEditor(filePath);
}

void MainWindow::openShaderEditor(const QString &filePath, int lineNumber, int columnNumber, int matchLength) {
    const QFileInfo requestedFile(filePath);
    if (!requestedFile.exists() || !requestedFile.isFile()) {
        QMessageBox::warning(this, tr("Open Shader"), tr("Shader file no longer exists:\n%1").arg(filePath));
        return;
    }

    cleanupClosedEditors();
    const QString canonicalPath = requestedFile.canonicalFilePath();
    for (const QPointer<TextEditor> &openEditor : open_files) {
        if (!openEditor)
            continue;
        const QString openPath = QFileInfo(openEditor->fileName()).canonicalFilePath();
        if (!canonicalPath.isEmpty() && openPath == canonicalPath) {
            ensureShaderEditorWorkspace();
            shaderEditorTabs->setCurrentWidget(openEditor);
            shaderEditorWorkspace->show();
            shaderEditorWorkspace->raise();
            shaderEditorWorkspace->activateWindow();
            openEditor->revealLocation(lineNumber, columnNumber, matchLength);
            return;
        }
    }

    ensureShaderEditorWorkspace();
    TextEditor *editor = new TextEditor(shaderEditorTabs);
    editor->setWindowFlags(Qt::Widget);
    editor->setText(readFileContents(filePath));
    editor->setFileName(filePath);
    connect(editor, &TextEditor::fileSaved, this, [this](const QString &filePath) { handleSavedShader(filePath); });
    connect(editor, &TextEditor::openFileRequested, this, [this](const QString &includePath, int lineNumber) { openShaderEditor(includePath, lineNumber); });
    connect(editor, &TextEditor::previewRequested, this, &MainWindow::queueAcmxvkEditorPreview);
    connect(editor, &TextEditor::uniformValueChanged, this, [this](const QString &name, double value) {
        if (!customUniformDialog || !customUniformDialog->setUniformValue(name, value)) {
            return;
        }
        for (const QPointer<TextEditor> &openEditor : open_files) {
            if (openEditor)
                openEditor->setUniformValue(name, value);
        }
    });
    open_files.append(editor);
    const int tabIndex = shaderEditorTabs->addTab(editor, requestedFile.fileName());
    connect(editor, &QWidget::windowTitleChanged, this, [this, editor](const QString &title) {
        if (!shaderEditorTabs)
            return;
        const int index = shaderEditorTabs->indexOf(editor);
        if (index < 0)
            return;
        QString tabTitle = title;
        const int separator = tabTitle.indexOf(QStringLiteral(" - "));
        if (separator >= 0)
            tabTitle = tabTitle.mid(separator + 3);
        shaderEditorTabs->setTabText(index, tabTitle);
    });
    updateOpenEditorShaderContexts();
    shaderEditorTabs->setCurrentIndex(tabIndex);
    shaderEditorWorkspace->show();
    shaderEditorWorkspace->raise();
    shaderEditorWorkspace->activateWindow();
    editor->show();
    editor->revealLocation(lineNumber, columnNumber, matchLength);
}

void MainWindow::ensureShaderEditorWorkspace() {
    if (shaderEditorWorkspace)
        return;
    shaderEditorWorkspace = new QDialog(this);
    shaderEditorWorkspace->setWindowTitle(tr("ACMX Shader Editor"));
    shaderEditorWorkspace->setModal(false);
    auto *layout = new QVBoxLayout(shaderEditorWorkspace);
    layout->setContentsMargins(4, 4, 4, 4);
    shaderEditorTabs = new QTabWidget(shaderEditorWorkspace);
    shaderEditorTabs->setTabsClosable(true);
    shaderEditorTabs->setMovable(true);
    shaderEditorTabs->setDocumentMode(true);
    layout->addWidget(shaderEditorTabs);
    connect(shaderEditorTabs, &QTabWidget::tabCloseRequested, this, [this](int index) {
        auto *editor = qobject_cast<TextEditor *>(shaderEditorTabs->widget(index));
        if (editor && editor->close())
            shaderEditorTabs->removeTab(index);
    });
    QSettings settings("LostSideDead");
    if (!shaderEditorWorkspace->restoreGeometry(settings.value("editor/workspaceGeometry").toByteArray())) {
        shaderEditorWorkspace->resize(1180, 820);
    }
    connect(shaderEditorWorkspace, &QDialog::finished, this, [this](int) {
        if (shaderEditorWorkspace) {
            QSettings("LostSideDead").setValue("editor/workspaceGeometry", shaderEditorWorkspace->saveGeometry());
        }
    });
}

void MainWindow::updateOpenEditorCompileStatus(const QString &sourcePath, bool pending, bool success, const QString &diagnostics) {
    const QFileInfo sourceInfo(sourcePath);
    const QString sourceCanonical = sourceInfo.canonicalFilePath();
    for (const QPointer<TextEditor> &editor : open_files) {
        if (!editor)
            continue;
        const QFileInfo editorInfo(editor->fileName());
        const QString editorCanonical = editorInfo.canonicalFilePath();
        const bool sameFile = (!sourceCanonical.isEmpty() && !editorCanonical.isEmpty() && sourceCanonical == editorCanonical) || sourceInfo.absoluteFilePath() == editorInfo.absoluteFilePath();
        if (!sameFile)
            continue;
        if (pending)
            editor->setCompilePending();
        else
            editor->setCompileResult(success, diagnostics);
    }
}

void MainWindow::updateOpenEditorShaderContexts() {
    const bool acmxvk = active_backend == acmx2::Backend::Acmxvk;
    QList<acmx2::CustomUniformDefinition> definitions;
    QString error;
    if (acmxvk && !shader_path.isEmpty() && !acmx2::load_custom_uniforms(shader_path, definitions, error)) {
        definitions.clear();
    }

    QVector<ShaderEditorUniform> uniforms;
    uniforms.reserve(definitions.size());
    for (const acmx2::CustomUniformDefinition &definition : definitions)
        uniforms.append({definition.name, definition.slot, definition.minimum, definition.maximum, definition.step, definition.value});

    const QString libraryRoot = QFileInfo(shader_path).canonicalFilePath();
    for (const QPointer<TextEditor> &editor : open_files) {
        if (!editor)
            continue;
        const QString editorPath = QFileInfo(editor->fileName()).canonicalFilePath();
        const QString relative = libraryRoot.isEmpty() || editorPath.isEmpty() ? QStringLiteral("..") : QDir(libraryRoot).relativeFilePath(editorPath);
        const bool inActiveLibrary = relative != QStringLiteral("..") && !relative.startsWith(QStringLiteral("../"));
        if (inActiveLibrary)
            editor->setShaderContext(acmxvk, uniforms, libraryRoot);
    }
}

QString MainWindow::currentShaderName() const {
    QTreeWidgetItem *it = list_view ? list_view->currentItem() : nullptr;
    if (!it)
        return QString();
    const int row = list_view->indexOfTopLevelItem(it);
    if (row < 0 || row >= items.size())
        return it->text(1);
    return items.at(row);
}

int MainWindow::currentShaderRow() const {
    if (!list_view)
        return -1;
    QTreeWidgetItem *it = list_view->currentItem();
    if (!it)
        return -1;
    return list_view->indexOfTopLevelItem(it);
}

void MainWindow::initShaderSelectionSharedMemory() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
#if defined(__linux__) || defined(__APPLE__)
    // A named semaphore can be unlinked while an existing process still owns
    // a usable handle. Verify that new child processes can still discover the
    // name before every launch and recreate it when necessary.
    if (shaderSelectionSemaphore != SEM_FAILED) {
        sem_t *publishedSemaphore = ::sem_open(acmx2::ipc::kShaderSelectionSemaphoreName, 0);
        if (publishedSemaphore != SEM_FAILED) {
            ::sem_close(publishedSemaphore);
        } else {
            ::sem_close(shaderSelectionSemaphore);
            shaderSelectionSemaphore = SEM_FAILED;
        }
    }
    if (shaderSelectionSemaphore == SEM_FAILED) {
        shaderSelectionSemaphore = ::sem_open(acmx2::ipc::kShaderSelectionSemaphoreName, O_CREAT, 0666, 1);
    }
    if (shaderSelectionSemaphore == SEM_FAILED) {
        Log(tr("Shared interface control unavailable: sem_open(%1) failed: %2").arg(acmx2::ipc::kShaderSelectionSemaphoreName, QString::fromLocal8Bit(std::strerror(errno))));
        return;
    }

    if (shaderSelectionShm)
        return;

    shaderSelectionShmFd = ::shm_open(acmx2::ipc::kShaderSelectionShmName, O_CREAT | O_RDWR, 0666);
    if (shaderSelectionShmFd < 0) {
        const int openError = errno;
        Log(tr("Shared interface control unavailable: shm_open(%1) failed: "
               "%2")
                .arg(acmx2::ipc::kShaderSelectionShmName, QString::fromLocal8Bit(std::strerror(openError))));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    constexpr std::size_t SHARED_MEMORY_SIZE = sizeof(acmx2::ipc::ShaderSelectionShmData);
    struct stat shmStat{};
    if (::fstat(shaderSelectionShmFd, &shmStat) != 0) {
        const int statError = errno;
        Log(tr("Shared interface control unavailable: fstat(%1) failed: %2").arg(acmx2::ipc::kShaderSelectionShmName, QString::fromLocal8Bit(std::strerror(statError))));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    if (shmStat.st_size == 0) {
        if (::ftruncate(shaderSelectionShmFd, static_cast<off_t>(SHARED_MEMORY_SIZE)) != 0) {
            const int truncateError = errno;
            Log(tr("Shared interface control unavailable: ftruncate(%1, %2) "
                   "failed: %3")
                    .arg(acmx2::ipc::kShaderSelectionShmName)
                    .arg(static_cast<qulonglong>(SHARED_MEMORY_SIZE))
                    .arg(QString::fromLocal8Bit(std::strerror(truncateError))));
            cleanupShaderSelectionSharedMemory();
            return;
        }
    } else if (shmStat.st_size < static_cast<off_t>(SHARED_MEMORY_SIZE)) {
        Log(tr("Shared interface control unavailable: %1 has size %2 bytes; "
               "expected %3. Refusing to resize an active or stale shared "
               "memory object.")
                .arg(acmx2::ipc::kShaderSelectionShmName)
                .arg(static_cast<qlonglong>(shmStat.st_size))
                .arg(static_cast<qulonglong>(SHARED_MEMORY_SIZE)));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    void *mapped = ::mmap(nullptr, SHARED_MEMORY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shaderSelectionShmFd, 0);
    if (mapped == MAP_FAILED) {
        const int mapError = errno;
        Log(tr("Shared interface control unavailable: mmap(%1, %2) failed: "
               "%3")
                .arg(acmx2::ipc::kShaderSelectionShmName)
                .arg(static_cast<qulonglong>(SHARED_MEMORY_SIZE))
                .arg(QString::fromLocal8Bit(std::strerror(mapError))));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    shaderSelectionShm = static_cast<acmx2::ipc::ShaderSelectionShmData *>(mapped);
#else
    if (shaderSelectionSemaphore == nullptr) {
        shaderSelectionSemaphore = ::CreateMutexW(nullptr, FALSE, acmx2::ipc::kShaderSelectionMutexNameWindows);
    }
    if (shaderSelectionSemaphore == nullptr) {
        Log(tr("Shared interface control unavailable: CreateMutexW failed "
               "with Windows error %1")
                .arg(static_cast<qulonglong>(::GetLastError())));
        return;
    }

    if (shaderSelectionShm)
        return;

    constexpr std::size_t SHARED_MEMORY_SIZE = sizeof(acmx2::ipc::ShaderSelectionShmData);
    shaderSelectionMapping = ::CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(SHARED_MEMORY_SIZE), acmx2::ipc::kShaderSelectionMappingNameWindows);
    if (shaderSelectionMapping == nullptr) {
        Log(tr("Shared interface control unavailable: CreateFileMappingW "
               "failed with Windows error %1")
                .arg(static_cast<qulonglong>(::GetLastError())));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    void *mapped = ::MapViewOfFile(shaderSelectionMapping, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEMORY_SIZE);
    if (mapped == nullptr) {
        Log(tr("Shared interface control unavailable: MapViewOfFile failed "
               "with Windows error %1")
                .arg(static_cast<qulonglong>(::GetLastError())));
        cleanupShaderSelectionSharedMemory();
        return;
    }
    shaderSelectionShm = static_cast<acmx2::ipc::ShaderSelectionShmData *>(mapped);
#endif

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
#if defined(__linux__) || defined(__APPLE__)
        const int lock_error = errno;
        Log(tr("Shared interface control unavailable: could not lock %1: %2").arg(acmx2::ipc::kShaderSelectionSemaphoreName, QString::fromLocal8Bit(std::strerror(lock_error))));
        if (lock_error == ETIMEDOUT) {
            Log(tr("The interface lock may have been left stale by a force-quit. Close all ACMX processes before resetting the shared-memory semaphore."));
        }
#else
        Log(tr("Shared interface control unavailable: could not lock the "
               "Windows control mutex (error %1)")
                .arg(static_cast<qulonglong>(::GetLastError())));
#endif
        cleanupShaderSelectionSharedMemory();
        return;
    }

    if (shaderSelectionShm->magic != acmx2::ipc::kShaderSelectionMagic || shaderSelectionShm->version != acmx2::ipc::kShaderSelectionVersion) {
        shaderSelectionShm->magic = acmx2::ipc::kShaderSelectionMagic;
        shaderSelectionShm->version = acmx2::ipc::kShaderSelectionVersion;
        shaderSelectionShm->selected_index = -1;
        shaderSelectionShm->shader_pass_count = 0;
        shaderSelectionShm->shader_pass_enabled = 0;
        shaderSelectionShm->repeat_enabled = 0;
        shaderSelectionShm->display_filter_enabled = 0;
        shaderSelectionShm->watermark_enabled = 0;
        shaderSelectionShm->normalized_time_enabled = 0;
        std::fill(std::begin(shaderSelectionShm->reserved_flags), std::end(shaderSelectionShm->reserved_flags), 0);
        std::fill(std::begin(shaderSelectionShm->shader_pass_indices), std::end(shaderSelectionShm->shader_pass_indices), -1);
        std::fill(&shaderSelectionShm->shader_pass_names[0][0], &shaderSelectionShm->shader_pass_names[0][0] + acmx2::ipc::kShaderSelectionMaxPassCount * acmx2::ipc::kShaderSelectionMaxShaderName, '\0');
        shaderSelectionShm->gpu_filter_count = 0;
        shaderSelectionShm->gpu_filter_enabled = 0;
        shaderSelectionShm->gpu_buffer_size = 8;
        shaderSelectionShm->watermark_r = 255;
        shaderSelectionShm->watermark_g = 0;
        shaderSelectionShm->watermark_b = 150;
        std::fill(std::begin(shaderSelectionShm->reserved), std::end(shaderSelectionShm->reserved), 0);
        std::fill(std::begin(shaderSelectionShm->gpu_filter_indices), std::end(shaderSelectionShm->gpu_filter_indices), -1);
        std::fill(std::begin(shaderSelectionShm->watermark_text), std::end(shaderSelectionShm->watermark_text), '\0');
        shaderSelectionShm->reload_shader_index = -1;
        std::fill(std::begin(shaderSelectionShm->reload_shader_path), std::end(shaderSelectionShm->reload_shader_path), '\0');
        shaderSelectionShm->reload_sequence = 0;
        shaderSelectionShm->custom_uniform_count = 0;
        std::fill(&shaderSelectionShm->custom_uniform_names[0][0], &shaderSelectionShm->custom_uniform_names[0][0] + acmx2::ipc::kShaderSelectionMaxCustomUniforms * acmx2::ipc::kShaderSelectionMaxUniformName, '\0');
        std::fill(std::begin(shaderSelectionShm->custom_uniform_values), std::end(shaderSelectionShm->custom_uniform_values), 0.0f);
        std::fill(std::begin(shaderSelectionShm->audio_file_path), std::end(shaderSelectionShm->audio_file_path), '\0');
        shaderSelectionShm->audio_output_device = -1;
        shaderSelectionShm->audio_pass_through = 0;
        shaderSelectionShm->audio_trunc = 0;
        shaderSelectionShm->audio_repeat = 0;
        shaderSelectionShm->audio_reserved = 0;
        shaderSelectionShm->audio_file_sequence = 0;
        shaderSelectionShm->dream_enabled = 0;
        shaderSelectionShm->dream_fp16 = 0;
        shaderSelectionShm->dream_gpu_filter_first = 0;
        shaderSelectionShm->dream_reserved = 0;
        shaderSelectionShm->dream_iterations = 1;
        shaderSelectionShm->dream_maximum_dimension = 512;
        shaderSelectionShm->dream_channel = -1;
        shaderSelectionShm->dream_octaves = 1;
        shaderSelectionShm->dream_jitter = 0;
        shaderSelectionShm->dream_smoothing = 0;
        shaderSelectionShm->dream_strength = 0.05F;
        shaderSelectionShm->dream_feedback = 0.9F;
        shaderSelectionShm->dream_zoom = 1.01F;
        shaderSelectionShm->dream_rotation = 0.1F;
        shaderSelectionShm->dream_octave_scale = 1.4F;
        std::fill(std::begin(shaderSelectionShm->dream_model_path), std::end(shaderSelectionShm->dream_model_path), '\0');
        std::fill(std::begin(shaderSelectionShm->dream_layer), std::end(shaderSelectionShm->dream_layer), '\0');
        std::fill(std::begin(shaderSelectionShm->effect_pack_manifest_path), std::end(shaderSelectionShm->effect_pack_manifest_path), '\0');
        shaderSelectionShm->effect_pack_sequence = 0;
        std::fill(std::begin(shaderSelectionShm->selected_shader_name), std::end(shaderSelectionShm->selected_shader_name), '\0');
        shaderSelectionShm->sequence = 0;
    }
#endif
}

void MainWindow::publishSelectedShaderIndexToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size())
        return;
    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->selected_index = row;
    const QByteArray shaderName = items.at(row).toUtf8();
    const qsizetype copyLength = std::min<qsizetype>(shaderName.size(), static_cast<qsizetype>(acmx2::ipc::kShaderSelectionMaxShaderName - 1));
    std::fill(std::begin(shaderSelectionShm->selected_shader_name), std::end(shaderSelectionShm->selected_shader_name), '\0');
    std::copy_n(shaderName.constData(), copyLength, shaderSelectionShm->selected_shader_name);
    ++shaderSelectionShm->sequence;
#endif
}

DeepDreamConfiguration MainWindow::regular_deep_dream_configuration() const {
    DeepDreamConfiguration dream;
    dream.enabled = deep_dream_enabled;
    dream.model_file = deep_dream_model;
    dream.layer = deep_dream_layer;
    dream.iterations = deep_dream_iterations;
    dream.strength = deep_dream_strength;
    dream.feedback = deep_dream_feedback;
    dream.zoom = deep_dream_zoom;
    dream.rotation = deep_dream_rotation;
    dream.maximum_dimension = deep_dream_maximum_dimension;
    dream.fp16 = deep_dream_fp16;
    dream.channel = deep_dream_channel;
    dream.octaves = deep_dream_octaves;
    dream.octave_scale = deep_dream_octave_scale;
    dream.jitter = deep_dream_jitter;
    dream.smoothing = deep_dream_smoothing;
    dream.gpu_filter_first = deep_dream_gpu_filter_first;
    dream.deep_original = deep_dream_original;
    return dream;
}

static bool dream_state_fits(const DeepDreamConfiguration &dream) { return dream.model_file.toUtf8().size() < static_cast<int>(acmx2::ipc::kShaderSelectionMaxDreamModelPath) && dream.layer.toUtf8().size() < static_cast<int>(acmx2::ipc::kShaderSelectionMaxDreamLayer); }

static void write_dream_state(acmx2::ipc::ShaderSelectionShmData *shared, const DeepDreamConfiguration &dream, bool enabled) {
    shared->dream_enabled = enabled && dream.enabled ? 1 : 0;
    shared->dream_fp16 = dream.fp16 ? 1 : 0;
    shared->dream_gpu_filter_first = dream.gpu_filter_first ? 1 : 0;
    shared->dream_iterations = dream.iterations;
    shared->dream_maximum_dimension = dream.maximum_dimension;
    shared->dream_channel = dream.channel;
    shared->dream_octaves = dream.octaves;
    shared->dream_jitter = dream.jitter;
    shared->dream_smoothing = dream.smoothing;
    shared->dream_strength = static_cast<float>(dream.strength);
    shared->dream_feedback = static_cast<float>(dream.feedback);
    shared->dream_zoom = static_cast<float>(dream.zoom);
    shared->dream_rotation = static_cast<float>(dream.rotation);
    shared->dream_octave_scale = static_cast<float>(dream.octave_scale);
    std::fill(std::begin(shared->dream_model_path), std::end(shared->dream_model_path), '\0');
    std::fill(std::begin(shared->dream_layer), std::end(shared->dream_layer), '\0');
    const QByteArray model = dream.model_file.toUtf8();
    const QByteArray layer = dream.layer.toUtf8();
    std::copy(model.cbegin(), model.cend(), shared->dream_model_path);
    std::copy(layer.cbegin(), layer.cend(), shared->dream_layer);
}

static bool valid_effect_pack_uniforms(const QVector<EffectPackUniformValue> &values) {
    if (values.size() > static_cast<int>(acmx2::ipc::kShaderSelectionMaxCustomUniforms)) {
        return false;
    }
    for (const EffectPackUniformValue &value : values) {
        const QByteArray name = value.name.toUtf8();
        if (name.isEmpty() || name.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxUniformName) || !std::isfinite(value.value) || std::abs(value.value) > std::numeric_limits<float>::max()) {
            return false;
        }
    }
    return true;
}

static bool write_effect_pack_uniforms(acmx2::ipc::ShaderSelectionShmData *shared, const QVector<EffectPackUniformValue> &values) {
    if (!valid_effect_pack_uniforms(values)) {
        return false;
    }
    std::fill(&shared->custom_uniform_names[0][0], &shared->custom_uniform_names[0][0] + acmx2::ipc::kShaderSelectionMaxCustomUniforms * acmx2::ipc::kShaderSelectionMaxUniformName, '\0');
    std::fill(std::begin(shared->custom_uniform_values), std::end(shared->custom_uniform_values), 0.0f);
    for (int index = 0; index < values.size(); ++index) {
        const QByteArray name = values[index].name.toUtf8();
        std::copy(name.cbegin(), name.cend(), shared->custom_uniform_names[index]);
        shared->custom_uniform_values[index] = static_cast<float>(values[index].value);
    }
    shared->custom_uniform_count = static_cast<quint32>(values.size());
    return true;
}

void MainWindow::publishEffectPackToRunningProcess(const QString &manifest_path, const QVector<EffectPackUniformValue> &values, const DeepDreamConfiguration &dream) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend != acmx2::Backend::Acmxvk) {
        return;
    }
    const bool running_engine = process && process->state() == QProcess::Running && !cacheBuildInProgress;
    if (!shaderSelectionShm) {
        initShaderSelectionSharedMemory();
    }
    if (!shaderSelectionShm) {
        Log(tr("Unable to publish effect pack: interface shared memory is unavailable."));
        return;
    }
    const QByteArray path = manifest_path.toUtf8();
    if (path.size() >= static_cast<qsizetype>(acmx2::ipc::kShaderSelectionMaxEffectPackPath)) {
        Log(tr("Effect-pack path is too long for the interface protocol."));
        return;
    }
    QVector<EffectPackUniformValue> requested_values = values;
    if (manifest_path.isEmpty() && customUniformDialog) {
        for (const acmx2::CustomUniformDefinition &uniform : customUniformDialog->uniforms()) {
            requested_values.push_back({uniform.name, uniform.value});
        }
    }
    const DeepDreamConfiguration requested_dream = manifest_path.isEmpty() ? regular_deep_dream_configuration() : dream;
    if (!valid_effect_pack_uniforms(requested_values) || !dream_state_fits(requested_dream) || (!manifest_path.isEmpty() && requested_dream.enabled && (!deep_dream_available || !QFileInfo(requested_dream.model_file).isFile() || requested_dream.layer.isEmpty()))) {
        Log(tr("Effect-pack controls or Deep Dream model are unavailable or exceed the interface limits."));
        return;
    }
    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log(tr("Unable to publish effect pack: interface lock failed."));
        return;
    }
    write_effect_pack_uniforms(shaderSelectionShm, requested_values);
    write_dream_state(shaderSelectionShm, requested_dream, active_backend == acmx2::Backend::Acmxvk && deep_dream_available);
    std::fill(std::begin(shaderSelectionShm->effect_pack_manifest_path), std::end(shaderSelectionShm->effect_pack_manifest_path), '\0');
    std::copy(path.cbegin(), path.cend(), shaderSelectionShm->effect_pack_manifest_path);
    ++shaderSelectionShm->effect_pack_sequence;
    ++shaderSelectionShm->sequence;
    active_effect_pack_dream = manifest_path.isEmpty() ? DeepDreamConfiguration{} : requested_dream;
    Log(manifest_path.isEmpty() ? tr("Returned to the shader library.") : running_engine ? tr("Requested effect pack: %1").arg(manifest_path) : tr("Effect pack selected for the next ACMXVK launch: %1").arg(manifest_path));
#else
    Q_UNUSED(manifest_path);
    Q_UNUSED(values);
    Q_UNUSED(dream);
#endif
}

void MainWindow::publishEffectPackUniformsToRunningProcess(const QVector<EffectPackUniformValue> &values) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend != acmx2::Backend::Acmxvk || !shaderSelectionShm || !effectPackBrowser || !effectPackBrowser->has_active_pack()) {
        return;
    }
    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock || !write_effect_pack_uniforms(shaderSelectionShm, values)) {
        Log(tr("Unable to publish effect-pack controls."));
        return;
    }
    ++shaderSelectionShm->sequence;
#else
    Q_UNUSED(values);
#endif
}

void MainWindow::ensureEffectPackBrowser() {
    if (!effectPackBrowser) {
        effectPackBrowser = new EffectPackBrowser(this);
        connect(effectPackBrowser, &EffectPackBrowser::activation_requested, this, &MainWindow::publishEffectPackToRunningProcess);
        connect(effectPackBrowser, &EffectPackBrowser::uniform_values_changed, this, &MainWindow::publishEffectPackUniformsToRunningProcess);
    }
    QString compiler_error;
    const QString compiler = resolve_acmxvk_shader_compiler(compiler_error);
    QSettings settings("LostSideDead");
    const int jobs = parallel_build_jobs(settings);
    effectPackBrowser->set_build_tools(executable_path, compiler, jobs > 0 ? jobs : 2);
    effectPackBrowser->set_runtime_context(deep_dream_available, deep_dream_model, audio_available, midi_available, midi_enabled && !midi_config_file.isEmpty());
}

void MainWindow::menuEffectPacks() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    ensureEffectPackBrowser();
    QJsonObject snapshot;
    QString source_error;
    const bool source_library = !shader_path.isEmpty() && is_acmxvk_source_library(shader_path, source_error);
    const QStringList pass_names = shader_pass_enabled && !shader_pass_names.isEmpty() ? shader_pass_names : QStringList{currentShaderName()};
    if (source_library && !pass_names.isEmpty() && pass_names.size() <= 64) {
        QJsonArray passes;
        bool complete = true;
        for (const QString &name : pass_names) {
            const QString relative = QDir::fromNativeSeparators(name);
            if ((!relative.endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive)) || relative.startsWith(QStringLiteral("../")) || relative.contains(QStringLiteral("/../")) || !QFileInfo(QDir(shader_path).filePath(relative)).isFile()) {
                complete = false;
                break;
            }
            passes.append(relative);
        }
        if (complete) {
            snapshot.insert(QStringLiteral("format"), QStringLiteral("acmxvk-effect-pack"));
            snapshot.insert(QStringLiteral("version"), 1);
            snapshot.insert(QStringLiteral("passes"), passes);
            QJsonArray controls;
            QList<acmx2::CustomUniformDefinition> definitions;
            if (customUniformDialog) {
                definitions = customUniformDialog->uniforms();
            } else {
                acmx2::load_custom_uniforms(shader_path, definitions, source_error);
            }
            for (const acmx2::CustomUniformDefinition &uniform : definitions) {
                if (uniform.name.isEmpty() || !std::isfinite(uniform.minimum) || !std::isfinite(uniform.maximum) || !std::isfinite(uniform.step) || !std::isfinite(uniform.value) || uniform.minimum >= uniform.maximum) {
                    continue;
                }
                controls.append(QJsonObject{{QStringLiteral("id"), uniform.name}, {QStringLiteral("label"), uniform.name}, {QStringLiteral("uniform"), uniform.name}, {QStringLiteral("minimum"), uniform.minimum}, {QStringLiteral("maximum"), uniform.maximum}, {QStringLiteral("step"), uniform.step}, {QStringLiteral("default"), std::clamp(uniform.value, uniform.minimum, uniform.maximum)}});
            }
            snapshot.insert(QStringLiteral("controls"), controls);
            if (deep_dream_enabled && !deep_dream_model.isEmpty()) {
                QString model_id = QFileInfo(deep_dream_model).completeBaseName();
                if (model_id.startsWith(QStringLiteral("deep-dream-"))) {
                    model_id.remove(0, 11);
                }
                snapshot.insert(QStringLiteral("deep_dream"), QJsonObject{{QStringLiteral("enabled"), true}, {QStringLiteral("model"), model_id}, {QStringLiteral("layer"), deep_dream_layer}, {QStringLiteral("channel"), deep_dream_channel}, {QStringLiteral("iterations"), deep_dream_iterations}, {QStringLiteral("strength"), deep_dream_strength}, {QStringLiteral("feedback"), deep_dream_feedback}, {QStringLiteral("zoom"), deep_dream_zoom}, {QStringLiteral("rotation"), deep_dream_rotation}, {QStringLiteral("working_size"), deep_dream_maximum_dimension}, {QStringLiteral("fp16"), deep_dream_fp16}, {QStringLiteral("octaves"), deep_dream_octaves}, {QStringLiteral("octave_scale"), deep_dream_octave_scale}, {QStringLiteral("jitter"), deep_dream_jitter}, {QStringLiteral("smoothing"), deep_dream_smoothing}, {QStringLiteral("gpu_filter_before_dream"), deep_dream_gpu_filter_first}});
            }
        }
    }
    effectPackBrowser->set_session_snapshot(snapshot.isEmpty() ? QString() : shader_path, snapshot);
    effectPackBrowser->refresh();
    effectPackBrowser->show();
    effectPackBrowser->raise();
    effectPackBrowser->activateWindow();
}

void MainWindow::publishShaderReloadToRunningProcess(const QString &filePath) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend != acmx2::Backend::Acmx2 || !shaderSelectionShm || !process || process->state() != QProcess::Running || cacheBuildInProgress) {
        return;
    }

    const QFileInfo savedFile(filePath);
    const QString shaderName = QDir(shader_path).relativeFilePath(savedFile.absoluteFilePath());
    const int shaderIndex = items.indexOf(shaderName, 0, Qt::CaseInsensitive);
    if (shaderIndex < 0) {
        Log("Saved shader is not in the active library; live reload was skipped: " + filePath);
        return;
    }

    const QByteArray reloadPath = savedFile.canonicalFilePath().toUtf8();
    if (reloadPath.isEmpty() || reloadPath.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxReloadPath)) {
        Log("Shader path is too long for live reload: " + filePath);
        return;
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->reload_shader_index = shaderIndex;
    std::fill(std::begin(shaderSelectionShm->reload_shader_path), std::end(shaderSelectionShm->reload_shader_path), '\0');
    std::copy(reloadPath.cbegin(), reloadPath.cend(), shaderSelectionShm->reload_shader_path);
    ++shaderSelectionShm->reload_sequence;
    ++shaderSelectionShm->sequence;
    Log("Requested live shader reload: " + shaderName + "<br>");
#else
    Q_UNUSED(filePath);
#endif
}

void MainWindow::handleSavedShader(const QString &filePath) {
    refreshShaderTreeMetadata();
    if (active_backend == acmx2::Backend::Acmxvk) {
        queueAcmxvkLiveCompile(filePath);
        return;
    }
    publishShaderReloadToRunningProcess(filePath);
}

void MainWindow::queueAcmxvkLiveCompile(const QString &filePath) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    QString typeError;
    if (!is_acmxvk_source_library(shader_path, typeError)) {
        const QString diagnostic = typeError.isEmpty() ? tr("ACMXVK live compile requires a source library.") : typeError;
        Log(diagnostic);
        updateOpenEditorCompileStatus(filePath, false, false, diagnostic);
        return;
    }

    const QFileInfo sourceInfo(filePath);
    const QString sourcePath = sourceInfo.canonicalFilePath();
    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    if (sourcePath.isEmpty() || sourceRoot.isEmpty()) {
        const QString diagnostic = tr("Could not resolve saved ACMXVK shader: %1").arg(filePath);
        Log(diagnostic);
        updateOpenEditorCompileStatus(filePath, false, false, diagnostic);
        return;
    }
    const QString sourceName = sanitizeShaderName(QDir(sourceRoot).relativeFilePath(sourcePath));
    if (sourceName.isEmpty() || (!sourceName.endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) && !sourceName.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive)) || !items.contains(sourceName, Qt::CaseInsensitive)) {
        const QString diagnostic = tr("Saved file is not a fragment or compute source in the active "
                                      "ACMXVK library: %1")
                                       .arg(filePath);
        Log(diagnostic);
        updateOpenEditorCompileStatus(filePath, false, false, diagnostic);
        return;
    }

    liveShaderCompileQueue.removeAll(sourcePath);
    liveShaderCompileQueue.append(sourcePath);
    updateOpenEditorCompileStatus(sourcePath, true);
    startNextAcmxvkLiveCompile();
#else
    Q_UNUSED(filePath);
#endif
}

void MainWindow::startNextAcmxvkLiveCompile() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (liveShaderCompileProcess && liveShaderCompileProcess->state() != QProcess::NotRunning) {
        return;
    }
    if (liveShaderCompileQueue.isEmpty()) {
        return;
    }

    if (!liveShaderCompileProcess) {
        liveShaderCompileProcess = new QProcess(this);
        liveShaderCompileProcess->setProcessChannelMode(QProcess::SeparateChannels);
        connect(liveShaderCompileProcess, &QProcess::readyReadStandardOutput, this, [this]() {
            QString output = QString::fromUtf8(liveShaderCompileProcess->readAllStandardOutput());
            liveShaderCompileStdout += output;
            if (liveShaderCompileStdout.size() > 262144)
                liveShaderCompileStdout = liveShaderCompileStdout.right(262144);
            Write(output.toHtmlEscaped().replace('\n', QStringLiteral("<br>")));
        });
        connect(liveShaderCompileProcess, &QProcess::readyReadStandardError, this, [this]() {
            QString output = QString::fromUtf8(liveShaderCompileProcess->readAllStandardError());
            liveShaderCompileStderr += output;
            if (liveShaderCompileStderr.size() > 262144)
                liveShaderCompileStderr = liveShaderCompileStderr.right(262144);
            Write(QStringLiteral("<b style='color:red;'>") + output.toHtmlEscaped().replace('\n', QStringLiteral("<br>")) + QStringLiteral("</b>"));
        });
        connect(liveShaderCompileProcess, static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
            liveShaderCompileStdout += QString::fromUtf8(liveShaderCompileProcess->readAllStandardOutput());
            liveShaderCompileStderr += QString::fromUtf8(liveShaderCompileProcess->readAllStandardError());
            bool installed = false;
            QString editorDiagnostics;
            QString compilerOutput = liveShaderCompileStderr.trimmed();
            if (!liveShaderCompileStdout.trimmed().isEmpty()) {
                if (!compilerOutput.isEmpty())
                    compilerOutput += QLatin1Char('\n');
                compilerOutput += liveShaderCompileStdout.trimmed();
            }
            if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                QFile compiled(liveShaderCompileTemporary);
                quint32 magic = 0;
                if (compiled.open(QIODevice::ReadOnly)) {
                    QDataStream stream(&compiled);
                    stream.setByteOrder(QDataStream::LittleEndian);
                    stream >> magic;
                }
                compiled.close();
                constexpr quint32 SPIRV_MAGIC = 0x07230203U;
                if (magic != SPIRV_MAGIC) {
                    editorDiagnostics = tr("Compiler did not produce valid SPIR-V.");
                    Log(tr("<b style='color:red;'>Live ACMXVK compile did "
                           "not produce valid SPIR-V for %1.</b>")
                            .arg(liveShaderCompileSource));
                } else if (QFileInfo(liveShaderCompileOutput).isSymLink()) {
                    editorDiagnostics = tr("The compiled output is a symbolic link and cannot "
                                           "be replaced safely.");
                    Log(tr("<b style='color:red;'>Refusing to replace "
                           "symbolic-link shader output: %1</b>")
                            .arg(liveShaderCompileOutput));
                } else {
                    std::error_code error;
                    replace_file(std::filesystem::u8path(liveShaderCompileTemporary.toUtf8().constData()), std::filesystem::u8path(liveShaderCompileOutput.toUtf8().constData()), error);
                    if (error) {
                        editorDiagnostics = tr("Could not install compiled shader: %1").arg(QString::fromStdString(error.message()));
                        Log(tr("<b style='color:red;'>Could not install "
                               "live ACMXVK shader: %1</b>")
                                .arg(QString::fromStdString(error.message())));
                    } else {
                        installed = true;
                    }
                }
            } else {
                Log(tr("<b style='color:red;'>Live ACMXVK compile failed "
                       "for %1 (%2, exit code %3).</b>")
                        .arg(liveShaderCompileSource)
                        .arg(exitStatus == QProcess::CrashExit ? tr("compiler crashed") : tr("compiler error"))
                        .arg(exitCode));
                if (compilerOutput.isEmpty())
                    compilerOutput = liveShaderCompileProcess->errorString();
                editorDiagnostics = compilerOutput;
                Log(tr("<b style='color:red;'>Compiler message:</b>"
                       "<pre style='white-space:pre-wrap;'>%1</pre>")
                        .arg(compilerOutput.toHtmlEscaped()));
            }

            updateOpenEditorCompileStatus(liveShaderCompileSource, false, installed, installed ? compilerOutput : editorDiagnostics);

            QFile::remove(liveShaderCompileInput);
            if (!installed) {
                QFile::remove(liveShaderCompileTemporary);
            } else {
                Log(tr("Compiled and installed ACMXVK shader: %1").arg(liveShaderCompileOutput));
                refreshShaderTreeMetadata();
                publishAcmxvkCompiledShaderReload(liveShaderCompileSource, liveShaderCompileOutput);
            }

            liveShaderCompileSource.clear();
            liveShaderCompileInput.clear();
            liveShaderCompileOutput.clear();
            liveShaderCompileTemporary.clear();
            liveShaderCompileStdout.clear();
            liveShaderCompileStderr.clear();
            QTimer::singleShot(0, this, &MainWindow::startNextAcmxvkLiveCompile);
        });
    }

    QString compilerError;
    const QString glslc = resolve_acmxvk_shader_compiler(compilerError);
    if (glslc.isEmpty()) {
        Log(tr("<b style='color:red;'>Cannot live compile ACMXVK shader: "
               "%1</b>")
                .arg(compilerError.toHtmlEscaped()));
        for (const QString &sourcePath : liveShaderCompileQueue) {
            updateOpenEditorCompileStatus(sourcePath, false, false, compilerError);
        }
        liveShaderCompileQueue.clear();
        return;
    }

    liveShaderCompileSource = liveShaderCompileQueue.takeFirst();
    updateOpenEditorCompileStatus(liveShaderCompileSource, true);
    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    const QString sourceName = QDir(sourceRoot).relativeFilePath(liveShaderCompileSource);
    liveShaderCompileOutput = QDir(acmxvk_build_directory(sourceRoot)).filePath(acmxvk_runtime_shader_name(sourceName));
    if (!QDir().mkpath(QFileInfo(liveShaderCompileOutput).absolutePath())) {
        const QString diagnostic = tr("Could not create live shader output directory for %1.").arg(liveShaderCompileOutput);
        Log(QStringLiteral("<b style='color:red;'>%1</b>").arg(diagnostic.toHtmlEscaped()));
        updateOpenEditorCompileStatus(liveShaderCompileSource, false, false, diagnostic);
        liveShaderCompileSource.clear();
        liveShaderCompileOutput.clear();
        QTimer::singleShot(0, this, &MainWindow::startNextAcmxvkLiveCompile);
        return;
    }

    liveShaderCompileTemporary = liveShaderCompileOutput + QStringLiteral(".live-tmp-%1-%2").arg(QCoreApplication::applicationPid()).arg(++liveShaderCompileSequence);
    const QFileInfo sourceInfo(liveShaderCompileSource);
    liveShaderCompileInput = sourceInfo.absolutePath() + QLatin1Char('/') + QStringLiteral(".acmxvk-live-%1-%2.%3").arg(QCoreApplication::applicationPid()).arg(liveShaderCompileSequence).arg(sourceInfo.suffix());
    QFile sourceFile(liveShaderCompileSource);
    if (!sourceFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        const QString diagnostic = tr("Could not read shader source for live safety processing: %1").arg(sourceFile.errorString());
        Log(QStringLiteral("<b style='color:red;'>%1</b>").arg(diagnostic.toHtmlEscaped()));
        updateOpenEditorCompileStatus(liveShaderCompileSource, false, false, diagnostic);
        liveShaderCompileTemporary.clear();
        liveShaderCompileInput.clear();
        liveShaderCompileSource.clear();
        QTimer::singleShot(0, this, &MainWindow::startNextAcmxvkLiveCompile);
        return;
    }
    const QString source = QString::fromUtf8(sourceFile.readAll());
    const QSettings editorSettings("LostSideDead");
    const QString guardedSource = editorSettings.value("editor/guardLoops", true).toBool() ? inject_safety_counters(source) : source;
    sourceFile.close();
    QSaveFile guardedFile(liveShaderCompileInput);
    const QByteArray guardedBytes = guardedSource.toUtf8();
    if (!guardedFile.open(QIODevice::WriteOnly | QIODevice::Text) || guardedFile.write(guardedBytes) != guardedBytes.size() || !guardedFile.commit()) {
        const QString diagnostic = tr("Could not create temporary shader source for live safety processing.");
        Log(QStringLiteral("<b style='color:red;'>%1</b>").arg(diagnostic.toHtmlEscaped()));
        updateOpenEditorCompileStatus(liveShaderCompileSource, false, false, diagnostic);
        QFile::remove(liveShaderCompileInput);
        liveShaderCompileTemporary.clear();
        liveShaderCompileInput.clear();
        liveShaderCompileSource.clear();
        QTimer::singleShot(0, this, &MainWindow::startNextAcmxvkLiveCompile);
        return;
    }
    const QStringList arguments{QStringLiteral("-I"), sourceRoot, liveShaderCompileInput, QStringLiteral("-o"), liveShaderCompileTemporary};
    Log(tr("Live compiling ACMXVK shader: %1").arg(sourceName));
    Log(tr("Command: %1 %2<br>").arg(glslc, concatList(arguments)));
    liveShaderCompileStdout.clear();
    liveShaderCompileStderr.clear();
    liveShaderCompileProcess->start(glslc, arguments);
    if (!liveShaderCompileProcess->waitForStarted()) {
        const QString diagnostic = tr("Failed to start the ACMXVK shader compiler: %1").arg(liveShaderCompileProcess->errorString());
        Log(QStringLiteral("<b style='color:red;'>%1</b>").arg(diagnostic.toHtmlEscaped()));
        updateOpenEditorCompileStatus(liveShaderCompileSource, false, false, diagnostic);
        QFile::remove(liveShaderCompileTemporary);
        QFile::remove(liveShaderCompileInput);
        liveShaderCompileSource.clear();
        liveShaderCompileInput.clear();
        liveShaderCompileOutput.clear();
        liveShaderCompileTemporary.clear();
        liveShaderCompileStdout.clear();
        liveShaderCompileStderr.clear();
        QTimer::singleShot(0, this, &MainWindow::startNextAcmxvkLiveCompile);
    }
#endif
}

void MainWindow::queueAcmxvkEditorPreview(const QString &filePath, const QString &source) {
    if (active_backend == acmx2::Backend::Acmx2) {
        publishAcmx2EditorPreview(filePath, source);
        return;
    }
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    pendingEditorPreviewPath = QFileInfo(filePath).absoluteFilePath();
    pendingEditorPreviewSource = source;
    startNextAcmxvkEditorPreview();
}

bool MainWindow::publishAcmx2EditorPreview(const QString &filePath, const QString &source) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm || !process || process->state() != QProcess::Running) {
        const QString error = tr("Start ACMX2 before using shader live preview.");
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }

    const QFileInfo sourceInfo(filePath);
    const QString sourcePath = sourceInfo.canonicalFilePath();
    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    if (sourcePath.isEmpty() || sourceRoot.isEmpty()) {
        const QString error = tr("Could not resolve the ACMX2 shader preview path: %1").arg(filePath);
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }

    const QString shaderName = sanitizeShaderName(QDir(sourceRoot).relativeFilePath(sourcePath));
    const int shaderIndex = items.indexOf(shaderName, 0, Qt::CaseInsensitive);
    const QString suffix = sourceInfo.suffix().toLower();
    if (shaderIndex < 0 || (suffix != QStringLiteral("glsl") && suffix != QStringLiteral("frag") && suffix != QStringLiteral("comp"))) {
        const QString error = tr("The editor file is not an active ACMX2 fragment or compute "
                                 "shader: %1")
                                  .arg(filePath);
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }

    const QString previewDirectory = QDir(sourceRoot).filePath(QStringLiteral(".acmx2-editor-preview"));
    if (!QDir().mkpath(previewDirectory)) {
        const QString error = tr("Could not create the ACMX2 editor preview directory.");
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }

    const QString previewPath = QDir(previewDirectory).filePath(QStringLiteral("preview-%1-%2.%3").arg(QCoreApplication::applicationPid()).arg(++editorPreviewSequence).arg(suffix));
    QSaveFile previewFile(previewPath);
    const QByteArray sourceBytes = source.toUtf8();
    if (!previewFile.open(QIODevice::WriteOnly | QIODevice::Text) || previewFile.write(sourceBytes) != sourceBytes.size() || !previewFile.commit()) {
        const QString error = tr("Could not write the temporary ACMX2 shader preview.");
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }

    const QByteArray reloadPath = QFileInfo(previewPath).canonicalFilePath().toUtf8();
    if (reloadPath.isEmpty() || reloadPath.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxReloadPath)) {
        QFile::remove(previewPath);
        const QString error = tr("The ACMX2 shader preview path is too long.");
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        QFile::remove(previewPath);
        const QString error = tr("Could not lock ACMX2 interface control.");
        updateOpenEditorCompileStatus(filePath, false, false, error);
        Log(error);
        return false;
    }
    shaderSelectionShm->reload_shader_index = shaderIndex;
    std::fill(std::begin(shaderSelectionShm->reload_shader_path), std::end(shaderSelectionShm->reload_shader_path), '\0');
    std::copy(reloadPath.cbegin(), reloadPath.cend(), shaderSelectionShm->reload_shader_path);
    ++shaderSelectionShm->reload_sequence;
    ++shaderSelectionShm->sequence;

    editorPreviewTemporaryFiles.append(previewPath);
    while (editorPreviewTemporaryFiles.size() > 16)
        QFile::remove(editorPreviewTemporaryFiles.takeFirst());
    updateOpenEditorCompileStatus(filePath, false, true, tr("Preview source sent to the running ACMX2 backend."));
    Log(tr("Requested ACMX2 editor preview: %1").arg(shaderName));
    return true;
#else
    Q_UNUSED(filePath);
    Q_UNUSED(source);
    return false;
#endif
}

void MainWindow::startNextAcmxvkEditorPreview() {
    if (editorPreviewProcess && editorPreviewProcess->state() != QProcess::NotRunning) {
        return;
    }
    if (pendingEditorPreviewPath.isEmpty())
        return;

    if (!editorPreviewProcess) {
        editorPreviewProcess = new QProcess(this);
        editorPreviewProcess->setProcessChannelMode(QProcess::SeparateChannels);
        connect(editorPreviewProcess, &QProcess::readyReadStandardOutput, this, [this]() { editorPreviewStdout += QString::fromUtf8(editorPreviewProcess->readAllStandardOutput()); });
        connect(editorPreviewProcess, &QProcess::readyReadStandardError, this, [this]() { editorPreviewStderr += QString::fromUtf8(editorPreviewProcess->readAllStandardError()); });
        connect(editorPreviewProcess, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
            editorPreviewStdout += QString::fromUtf8(editorPreviewProcess->readAllStandardOutput());
            editorPreviewStderr += QString::fromUtf8(editorPreviewProcess->readAllStandardError());
            QString diagnostics = editorPreviewStderr.trimmed();
            if (!editorPreviewStdout.trimmed().isEmpty()) {
                if (!diagnostics.isEmpty())
                    diagnostics += QLatin1Char('\n');
                diagnostics += editorPreviewStdout.trimmed();
            }
            diagnostics.replace(editorPreviewInput, editorPreviewPath);
            const bool success = exitStatus == QProcess::NormalExit && exitCode == 0 && QFileInfo(editorPreviewOutput).isFile() && QFileInfo(editorPreviewOutput).size() >= 20;
            updateOpenEditorCompileStatus(editorPreviewPath, false, success, diagnostics);
            if (success) {
                editorPreviewTemporaryFiles.append(editorPreviewOutput);
                while (editorPreviewTemporaryFiles.size() > 16)
                    QFile::remove(editorPreviewTemporaryFiles.takeFirst());
                publishAcmxvkCompiledShaderReload(editorPreviewPath, editorPreviewOutput);
                Log(tr("Compiled ACMXVK editor preview: %1").arg(QFileInfo(editorPreviewPath).fileName()));
            } else {
                QFile::remove(editorPreviewOutput);
                Log(tr("<b style='color:red;'>ACMXVK editor preview failed: "
                       "%1</b><pre style='white-space:pre-wrap;'>%2</pre>")
                        .arg(QFileInfo(editorPreviewPath).fileName(), diagnostics.toHtmlEscaped()));
            }
            QFile::remove(editorPreviewInput);
            editorPreviewPath.clear();
            editorPreviewInput.clear();
            editorPreviewOutput.clear();
            editorPreviewStdout.clear();
            editorPreviewStderr.clear();
            QTimer::singleShot(0, this, &MainWindow::startNextAcmxvkEditorPreview);
        });
    }

    QString compilerError;
    const QString glslc = resolve_acmxvk_shader_compiler(compilerError);
    editorPreviewPath = pendingEditorPreviewPath;
    const QString source = pendingEditorPreviewSource;
    pendingEditorPreviewPath.clear();
    pendingEditorPreviewSource.clear();
    if (glslc.isEmpty()) {
        updateOpenEditorCompileStatus(editorPreviewPath, false, false, compilerError);
        editorPreviewPath.clear();
        return;
    }

    const QFileInfo sourceInfo(editorPreviewPath);
    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    const QString previewDirectory = QDir(acmxvk_build_directory(sourceRoot)).filePath(QStringLiteral(".editor-preview"));
    if (!QDir().mkpath(previewDirectory)) {
        const QString error = tr("Could not create the editor preview directory.");
        updateOpenEditorCompileStatus(editorPreviewPath, false, false, error);
        editorPreviewPath.clear();
        return;
    }
    const QString suffix = sourceInfo.suffix().toLower();
    const QString baseName = QStringLiteral("preview-%1-%2.%3").arg(QCoreApplication::applicationPid()).arg(++editorPreviewSequence).arg(suffix == QStringLiteral("comp") ? QStringLiteral("comp") : QStringLiteral("frag"));
    editorPreviewInput = QDir(previewDirectory).filePath(baseName);
    editorPreviewOutput = editorPreviewInput + QStringLiteral(".spv");
    QSaveFile inputFile(editorPreviewInput);
    const QByteArray sourceBytes = source.toUtf8();
    if (!inputFile.open(QIODevice::WriteOnly | QIODevice::Text) || inputFile.write(sourceBytes) != sourceBytes.size() || !inputFile.commit()) {
        const QString error = tr("Could not write the temporary preview source.");
        updateOpenEditorCompileStatus(editorPreviewPath, false, false, error);
        editorPreviewPath.clear();
        editorPreviewInput.clear();
        editorPreviewOutput.clear();
        return;
    }

    updateOpenEditorCompileStatus(editorPreviewPath, true);
    QStringList arguments{QStringLiteral("-I"), sourceInfo.absolutePath()};
    if (!sourceRoot.isEmpty() && sourceRoot != sourceInfo.absolutePath())
        arguments << QStringLiteral("-I") << sourceRoot;
    arguments << editorPreviewInput << QStringLiteral("-o") << editorPreviewOutput;
    editorPreviewStdout.clear();
    editorPreviewStderr.clear();
    editorPreviewProcess->start(glslc, arguments);
    if (!editorPreviewProcess->waitForStarted()) {
        const QString error = tr("Failed to start the shader compiler: %1").arg(editorPreviewProcess->errorString());
        updateOpenEditorCompileStatus(editorPreviewPath, false, false, error);
        QFile::remove(editorPreviewInput);
        editorPreviewPath.clear();
        editorPreviewInput.clear();
        editorPreviewOutput.clear();
    }
}

void MainWindow::publishAcmxvkCompiledShaderReload(const QString &sourcePath, const QString &runtimePath) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend != acmx2::Backend::Acmxvk || !shaderSelectionShm || !process || process->state() != QProcess::Running) {
        return;
    }

    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    const QString resolvedSourcePath = QFileInfo(sourcePath).canonicalFilePath();
    const QString sourceName = sourceRoot.isEmpty() || resolvedSourcePath.isEmpty() ? QString() : sanitizeShaderName(QDir(sourceRoot).relativeFilePath(resolvedSourcePath));
    if (sourceName.isEmpty()) {
        Log(tr("Compiled shader source cannot be resolved inside the active "
               "library: %1")
                .arg(sourcePath));
        return;
    }

    const int shaderIndex = items.indexOf(sourceName, 0, Qt::CaseInsensitive);
    if (shaderIndex < 0) {
        Log(tr("Compiled shader is not present in the active library "
               "manifest: %1")
                .arg(sourceName));
        return;
    }

    const QByteArray reloadPath = QFileInfo(runtimePath).canonicalFilePath().toUtf8();
    if (reloadPath.isEmpty()) {
        Log(tr("Compiled shader output cannot be resolved for live reload: %1").arg(runtimePath));
        return;
    }
    if (reloadPath.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxReloadPath)) {
        Log(tr("Compiled shader path is too long for live reload: %1").arg(runtimePath));
        return;
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log(tr("Could not lock the live shader reload channel."));
        return;
    }
    shaderSelectionShm->reload_shader_index = shaderIndex;
    std::fill(std::begin(shaderSelectionShm->reload_shader_path), std::end(shaderSelectionShm->reload_shader_path), '\0');
    std::copy(reloadPath.cbegin(), reloadPath.cend(), shaderSelectionShm->reload_shader_path);
    ++shaderSelectionShm->reload_sequence;
    ++shaderSelectionShm->sequence;
    Log(tr("Requested live ACMXVK pipeline reload: %1<br>").arg(sourceName));
#else
    Q_UNUSED(sourcePath);
    Q_UNUSED(runtimePath);
#endif
}

void MainWindow::publishMultipassShadersToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;

    std::array<qint32, acmx2::ipc::kShaderSelectionMaxPassCount> passIndices;
    passIndices.fill(-1);
    std::array<std::array<char, acmx2::ipc::kShaderSelectionMaxShaderName>, acmx2::ipc::kShaderSelectionMaxPassCount> passNames{};

    quint32 passCount = 0;
    if (shader_pass_enabled && !shader_pass_names.isEmpty()) {
        for (const QString &name : shader_pass_names) {
            if (passCount >= acmx2::ipc::kShaderSelectionMaxPassCount)
                break;
            const int idx = items.indexOf(name);
            if (idx < 0)
                continue;
            passIndices[passCount] = idx;
            const QByteArray shaderName = name.toUtf8();
            const qsizetype copyLength = std::min<qsizetype>(shaderName.size(), static_cast<qsizetype>(acmx2::ipc::kShaderSelectionMaxShaderName - 1));
            std::copy_n(shaderName.constData(), copyLength, passNames[passCount].begin());
            ++passCount;
        }
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->shader_pass_enabled = (shader_pass_enabled && passCount > 0) ? 1 : 0;
    shaderSelectionShm->shader_pass_count = passCount;
    std::copy(passIndices.begin(), passIndices.end(), std::begin(shaderSelectionShm->shader_pass_indices));
    for (std::size_t i = 0; i < passNames.size(); ++i) {
        std::copy(passNames[i].begin(), passNames[i].end(), shaderSelectionShm->shader_pass_names[i]);
    }
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishRepeatStateToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;
    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->repeat_enabled = (play_repeat && play_repeat->isChecked()) ? 1 : 0;
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishRuntimeSettingsToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->display_filter_enabled = display_filter_enabled ? 1 : 0;
    shaderSelectionShm->normalized_time_enabled = normalized_time ? 1 : 0;

    const bool watermarkActive = watermark_enabled && !watermark_text.isEmpty();
    shaderSelectionShm->watermark_enabled = watermarkActive ? 1 : 0;
    shaderSelectionShm->watermark_r = static_cast<uint8_t>(std::clamp(watermark_r, 0, 255));
    shaderSelectionShm->watermark_g = static_cast<uint8_t>(std::clamp(watermark_g, 0, 255));
    shaderSelectionShm->watermark_b = static_cast<uint8_t>(std::clamp(watermark_b, 0, 255));
    std::fill(std::begin(shaderSelectionShm->watermark_text), std::end(shaderSelectionShm->watermark_text), '\0');
    const QByteArray wmUtf8 = watermark_text.toUtf8();
    const std::size_t wmCap = static_cast<std::size_t>(acmx2::ipc::kShaderSelectionMaxWatermarkText - 1);
    const std::size_t wmLen = std::min<std::size_t>(wmCap, static_cast<std::size_t>(wmUtf8.size()));
    std::copy_n(wmUtf8.constData(), static_cast<int>(wmLen), shaderSelectionShm->watermark_text);

    std::array<qint32, acmx2::ipc::kShaderSelectionMaxGpuFilterCount> gpuIndices;
    gpuIndices.fill(-1);
    quint32 gpuCount = 0;
    if (cuda_available && gpu_filter_enabled && !gpu_filter_indices.isEmpty()) {
        const QStringList parts = gpu_filter_indices.split(',', Qt::SkipEmptyParts);
        for (const QString &part : parts) {
            if (gpuCount >= acmx2::ipc::kShaderSelectionMaxGpuFilterCount)
                break;
            bool ok = false;
            const int idx = part.trimmed().toInt(&ok);
            if (!ok || idx < 0)
                continue;
            gpuIndices[gpuCount++] = idx;
        }
    }

    shaderSelectionShm->gpu_filter_enabled = (gpuCount > 0) ? 1 : 0;
    shaderSelectionShm->gpu_filter_count = gpuCount;
    shaderSelectionShm->gpu_buffer_size = static_cast<uint8_t>(std::clamp(gpu_buffer_size, 4, 32));
    std::copy(gpuIndices.begin(), gpuIndices.end(), std::begin(shaderSelectionShm->gpu_filter_indices));

    const DeepDreamConfiguration dream = effectPackBrowser && effectPackBrowser->has_active_pack() ? active_effect_pack_dream : regular_deep_dream_configuration();
    if (dream_state_fits(dream)) {
        write_dream_state(shaderSelectionShm, dream, active_backend == acmx2::Backend::Acmxvk && deep_dream_available);
    } else if (dream.enabled) {
        Log(tr("Deep Dream settings were not published because the model path or layer is too long."));
    }

    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishCustomUniformsToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend == acmx2::Backend::Acmxvk && effectPackBrowser && effectPackBrowser->has_active_pack()) {
        return;
    }
    if (!shaderSelectionShm || !customUniformDialog)
        return;

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    std::fill(&shaderSelectionShm->custom_uniform_names[0][0], &shaderSelectionShm->custom_uniform_names[0][0] + acmx2::ipc::kShaderSelectionMaxCustomUniforms * acmx2::ipc::kShaderSelectionMaxUniformName, '\0');
    std::fill(std::begin(shaderSelectionShm->custom_uniform_values), std::end(shaderSelectionShm->custom_uniform_values), 0.0f);

    quint32 count = 0;
    for (const acmx2::CustomUniformDefinition &uniform : customUniformDialog->uniforms()) {
        if (count >= acmx2::ipc::kShaderSelectionMaxCustomUniforms)
            break;
        const QByteArray name = uniform.name.toUtf8();
        if (name.isEmpty() || name.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxUniformName)) {
            continue;
        }
        std::copy(name.cbegin(), name.cend(), shaderSelectionShm->custom_uniform_names[count]);
        shaderSelectionShm->custom_uniform_values[count] = static_cast<float>(uniform.value);
        ++count;
    }
    shaderSelectionShm->custom_uniform_count = count;
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::cleanupShaderSelectionSharedMemory() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (shaderSelectionShm) {
#if defined(__linux__) || defined(__APPLE__)
        ::munmap(shaderSelectionShm, sizeof(acmx2::ipc::ShaderSelectionShmData));
#else
        ::UnmapViewOfFile(shaderSelectionShm);
#endif
        shaderSelectionShm = nullptr;
    }
#if defined(__linux__) || defined(__APPLE__)
    if (shaderSelectionShmFd >= 0) {
        ::close(shaderSelectionShmFd);
        shaderSelectionShmFd = -1;
    }
#else
    if (shaderSelectionMapping != nullptr) {
        ::CloseHandle(shaderSelectionMapping);
        shaderSelectionMapping = nullptr;
    }
#endif
    cleanupShaderSelectionSemaphore();
#endif
}

void MainWindow::cleanupShaderSelectionSemaphore() {
#if defined(__linux__) || defined(__APPLE__)
    if (shaderSelectionSemaphore == SEM_FAILED)
        return;

    //::sem_unlink(acmx2::ipc::kShaderSelectionSemaphoreName);
    ::sem_close(shaderSelectionSemaphore);
    shaderSelectionSemaphore = SEM_FAILED;
#elif defined(_WIN32)
    if (shaderSelectionSemaphore == nullptr)
        return;

    ::CloseHandle(shaderSelectionSemaphore);
    shaderSelectionSemaphore = nullptr;
#endif
}

void MainWindow::selectShaderRow(int row) {
    if (!list_view || row < 0 || row >= list_view->topLevelItemCount())
        return;
    QTreeWidgetItem *it = list_view->topLevelItem(row);
    if (!it)
        return;
    list_view->setCurrentItem(it);
    list_view->scrollToItem(it, QAbstractItemView::PositionAtCenter);
}

void MainWindow::refreshShaderCacheStatus() {
    shaderCacheStatus.clear();
    shaderCacheMTime = QDateTime();
    if (active_backend != acmx2::Backend::Acmx2)
        return;
#ifdef Q_OS_MACOS
    // There is no persistent binary cache to inspect on macOS. Source saves
    // are handled by the live-reload IPC path instead.
    return;
#else
    if (shader_path.isEmpty())
        return;
    const QString cachePath = resolveShaderCachePath(shader_path, cache_size, cache_enabled && textureCacheArraySettingEnabled());
    QFileInfo cacheInfo(cachePath);
    if (!cacheInfo.exists() || !cacheInfo.isFile()) {
        Log("Shader cache not found at: " + cachePath);
        return;
    }
    shaderCacheMTime = cacheInfo.lastModified();
    shaderCacheStatus = parseShaderCacheStatus(cachePath);
    // Log("Shader cache: " + cachePath + " (" + QString::number(shaderCacheStatus.size()) + " entries)");
#endif
}

void MainWindow::populateShaderTree() {
    if (!list_view)
        return;
    refreshShaderCacheStatus();

    // Preserve the currently selected row so a refresh (e.g. after the
    // child process exits) does not lose the user's place in the list.
    const int previousRow = currentShaderRow();

    const QSignalBlocker blocker(list_view);
    list_view->clear();

    QProgressDialog *progress = shader_library_progress_dialog.data();
    if (progress) {
        progress->setRange(0, items.size());
        progress->setValue(0);
    }

    QString acmxvk_type_error;
    const bool acmxvk_source = active_backend == acmx2::Backend::Acmxvk && is_acmxvk_source_library(shader_path, acmxvk_type_error) && acmxvk_type_error.isEmpty();
    const int width = QString::number(items.size()).size();
    for (int i = 0; i < items.size(); ++i) {
        if (i % 32 == 0) {
            if (progress) {
                progress->setValue(i);
                progress->setLabelText(tr("Loading shader %1 of %2...").arg(i + 1).arg(items.size()));
            }
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
        const QString &name = items.at(i);
        QFileInfo fi(shader_path + "/" + name);
        const QString stem = QFileInfo(name).completeBaseName();
        const bool isCompute = name.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive) || name.endsWith(QStringLiteral(".comp.spv"), Qt::CaseInsensitive);
        const QString shaderType = isCompute ? tr("Compute") : tr("Fragment");

        QString health;
        QColor healthColor;
        if (active_backend == acmx2::Backend::Acmxvk) {
            const AcmxvkBuildState state = acmxvk_source ? acmxvk_shader_build_state(shader_path, name) : AcmxvkBuildState::UpToDate;
            if (state == AcmxvkBuildState::NotBuilt) {
                health = tr("Not Built");
                healthColor = QColor("#888888");
            } else if (state == AcmxvkBuildState::Stale) {
                health = tr("Stale");
                healthColor = QColor("#ffaa00");
            } else {
                health = tr("Up to Date");
                healthColor = QColor("#55ff55");
            }
        } else if (shaderCacheStatus.isEmpty()) {
            health = tr("No cache");
            healthColor = QColor("#888888");
        } else if (!shaderCacheStatus.contains(stem)) {
            health = tr("Uncached");
            healthColor = QColor("#cccc00");
        } else if (shaderCacheStatus.value(stem)) {
            health = tr("Failed");
            healthColor = QColor("#ff5555");
        } else if (fi.exists() && shaderCacheMTime.isValid() && fi.lastModified() > shaderCacheMTime) {
            health = tr("Stale");
            healthColor = QColor("#ffaa00");
        } else {
            health = tr("Cached");
            healthColor = QColor("#55ff55");
        }

        QStringList cols;
        cols << QString("%1").arg(i, width, 10, QLatin1Char(' ')) << name << (fi.exists() ? formatLastModified(fi.lastModified()) : tr("missing")) << health << shaderType;
        auto *item = new QTreeWidgetItem(list_view, cols);
        item->setTextAlignment(0, Qt::AlignRight | Qt::AlignVCenter);
        item->setForeground(3, QBrush(healthColor));
        if (!fi.exists())
            item->setForeground(2, QBrush(QColor("#ff5555")));
    }
    if (progress)
        progress->setValue(items.size());

    // Restore the previously selected row after the rebuild.
    if (previousRow >= 0 && previousRow < list_view->topLevelItemCount()) {
        QTreeWidgetItem *it = list_view->topLevelItem(previousRow);
        if (it) {
            list_view->setCurrentItem(it);
            list_view->scrollToItem(it, QAbstractItemView::PositionAtCenter);
        }
    }
}

void MainWindow::refreshShaderTreeMetadata() {
    if (!list_view || list_view->topLevelItemCount() != items.size())
        return;

    refreshShaderCacheStatus();

    const QSignalBlocker blocker(list_view);
    QString acmxvkTypeError;
    const bool acmxvkSource = active_backend == acmx2::Backend::Acmxvk && is_acmxvk_source_library(shader_path, acmxvkTypeError) && acmxvkTypeError.isEmpty();

    for (int i = 0; i < items.size(); ++i) {
        QTreeWidgetItem *item = list_view->topLevelItem(i);
        if (!item)
            continue;

        const QString &name = items.at(i);
        const QFileInfo fileInfo(QDir(shader_path).filePath(name));
        const QString stem = QFileInfo(name).completeBaseName();
        const bool isCompute = name.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive) || name.endsWith(QStringLiteral(".comp.spv"), Qt::CaseInsensitive);

        QString health;
        QColor healthColor;
        if (active_backend == acmx2::Backend::Acmxvk) {
            const AcmxvkBuildState state = acmxvkSource ? acmxvk_shader_build_state(shader_path, name) : AcmxvkBuildState::UpToDate;
            if (state == AcmxvkBuildState::NotBuilt) {
                health = tr("Not Built");
                healthColor = QColor("#888888");
            } else if (state == AcmxvkBuildState::Stale) {
                health = tr("Stale");
                healthColor = QColor("#ffaa00");
            } else {
                health = tr("Up to Date");
                healthColor = QColor("#55ff55");
            }
        } else if (shaderCacheStatus.isEmpty()) {
            health = tr("No cache");
            healthColor = QColor("#888888");
        } else if (!shaderCacheStatus.contains(stem)) {
            health = tr("Uncached");
            healthColor = QColor("#cccc00");
        } else if (shaderCacheStatus.value(stem)) {
            health = tr("Failed");
            healthColor = QColor("#ff5555");
        } else if (fileInfo.exists() && shaderCacheMTime.isValid() && fileInfo.lastModified() > shaderCacheMTime) {
            health = tr("Stale");
            healthColor = QColor("#ffaa00");
        } else {
            health = tr("Cached");
            healthColor = QColor("#55ff55");
        }

        item->setText(2, fileInfo.exists() ? formatLastModified(fileInfo.lastModified()) : tr("missing"));
        item->setText(3, health);
        item->setText(4, isCompute ? tr("Compute") : tr("Fragment"));
        item->setForeground(2, fileInfo.exists() ? QBrush() : QBrush(QColor("#ff5555")));
        item->setForeground(3, QBrush(healthColor));
    }
}

void MainWindow::Log(const QString &message) {
    QString normalized = message;
    while (normalized.endsWith('\n') || normalized.endsWith('\r')) {
        normalized.chop(1);
    }

    bottomTextBox->append(normalized);
    QTextCursor cursor = bottomTextBox->textCursor();
    cursor.movePosition(QTextCursor::End);
    bottomTextBox->setTextCursor(cursor);
}

void MainWindow::Write(const QString &message) {
    QTextCursor cursor = bottomTextBox->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertHtml(message);
    bottomTextBox->setTextCursor(cursor);
}

void MainWindow::queueProcessOutput(const QString &message) {
    pendingProcessOutput += message;
    if (!processOutputFlushTimer->isActive()) {
        processOutputFlushTimer->start();
    }
}

void MainWindow::flushProcessOutput() {
    if (pendingProcessOutput.isEmpty()) {
        return;
    }

    constexpr qsizetype MAX_OUTPUT_CHARS_PER_FLUSH = 48 * 1024;
    qsizetype length = std::min<qsizetype>(pendingProcessOutput.size(), MAX_OUTPUT_CHARS_PER_FLUSH);
    if (length < pendingProcessOutput.size()) {
        const qsizetype lineEnd = pendingProcessOutput.lastIndexOf(QStringLiteral("<br>"), length - 1);
        if (lineEnd >= 0) {
            length = lineEnd + 4;
        }
    }

    Write(pendingProcessOutput.left(length));
    pendingProcessOutput.remove(0, length);
    if (!pendingProcessOutput.isEmpty()) {
        processOutputFlushTimer->start();
    }
}

void MainWindow::beginOutputRunLog(const QString &command, const QString &run_output_file) {
    const QString effective_output_file = run_output_file.isEmpty() ? output_file : run_output_file;
    if (!save_output_log || effective_output_file.isEmpty()) {
        return;
    }

    if (output_run_log.isOpen()) {
        output_run_log.close();
    }
    const QFileInfo output_info(effective_output_file);
    const QString log_directory = project_output_directory.isEmpty() ? output_info.absoluteDir().filePath(QStringLiteral("logs")) : QDir(project_output_directory).filePath(QStringLiteral("logs"));
    if (!QDir().mkpath(log_directory)) {
        Log(tr("<b style='color:red;'>Unable to create render log directory: %1</b>").arg(log_directory));
        return;
    }
    output_run_log.setFileName(QDir(log_directory).filePath(output_info.completeBaseName() + QStringLiteral(".log")));
    if (!output_run_log.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        Log(tr("<b style='color:red;'>Unable to save render log: %1</b>").arg(output_run_log.errorString()));
        return;
    }

    appendOutputRunLog(tr("ACMX render log"));
    appendOutputRunLog(tr("Started: %1").arg(QDateTime::currentDateTime().toString(Qt::ISODate)));
    appendOutputRunLog(tr("Command: %1").arg(command));
    appendOutputRunLog(QString());
}

void MainWindow::appendOutputRunLog(const QString &message) {
    if (!output_run_log.isOpen()) {
        return;
    }

    output_run_log.write(message.toUtf8());
    if (!message.endsWith(QLatin1Char('\n'))) {
        output_run_log.write("\n");
    }
    output_run_log.flush();
}

void MainWindow::finishOutputRunLog(int exitCode, QProcess::ExitStatus exitStatus) {
    if (!output_run_log.isOpen()) {
        return;
    }

    appendOutputRunLog(QString());
    appendOutputRunLog(tr("Finished: %1").arg(QDateTime::currentDateTime().toString(Qt::ISODate)));
    appendOutputRunLog(tr("Exit: %1 (%2)").arg(exitCode).arg(exitStatus == QProcess::NormalExit ? tr("normal") : tr("crashed")));
    output_run_log.close();
}

void MainWindow::menuLoadLibrary() {
    QSettings settings("LostSideDead");
    QString startDirectory = settings.value("lastShaderDir").toString();
    if (startDirectory.isEmpty())
        startDirectory = shader_path;
    if (startDirectory.isEmpty())
        startDirectory = QDir::homePath();

    const QString directory = QFileDialog::getExistingDirectory(this, tr("Load Shader Library"), startDirectory, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (directory.isEmpty())
        return;

    settings.setValue("lastShaderDir", directory);
    loadLibraryPath(directory);
}

bool MainWindow::backend_launch_available() const { return true; }

void MainWindow::prompt_acmxvk_rebuild(const QString &reason, PendingAcmxvkAction resume_action) {
    QString type_error;
    if (!is_acmxvk_source_library(shader_path, type_error) || !type_error.isEmpty()) {
        QMessageBox::warning(this, tr("Build ACMXVK Library"), reason);
        return;
    }

    const QMessageBox::StandardButton answer = QMessageBox::question(this,
                                                                     tr("Build ACMXVK Library"),
                                                                     tr("The ACMXVK build is out of date or incomplete.\n\n%1\n\n"
                                                                        "Do you wish to rebuild it now?")
                                                                         .arg(reason),
                                                                     QMessageBox::Yes | QMessageBox::No,
                                                                     QMessageBox::Yes);
    if (answer != QMessageBox::Yes)
        return;

    pending_acmxvk_action = resume_action;
    Log(tr("ACMXVK rebuild requested before launch."));
    menuBuildShaderCache();
}

void MainWindow::update_backend_ui() {
    const QString name = acmx2::backend_name(active_backend);
    const QString project_name = current_project_path.isEmpty() ? QString() : QFileInfo(current_project_path).completeBaseName();
    setWindowTitle(project_name.isEmpty() ? tr("%1 - Interface").arg(name) : tr("%1 - Interface - %2").arg(name, project_name));
    if (backendAcmx2Action)
        backendAcmx2Action->setChecked(active_backend == acmx2::Backend::Acmx2);
    if (backendAcmxvkAction)
        backendAcmxvkAction->setChecked(active_backend == acmx2::Backend::Acmxvk);
    if (projectMenu)
        projectMenu->setEnabled(active_backend == acmx2::Backend::Acmxvk);
    if (effectPacksAction)
        effectPacksAction->setVisible(active_backend == acmx2::Backend::Acmxvk);

    const bool launchAvailable = backend_launch_available();
    const bool acmx2Tools = active_backend == acmx2::Backend::Acmx2;
    if (deepDreamAction) {
        deepDreamAction->setVisible(!acmx2Tools);
    }
    QString sourceTypeError;
    const bool acmxvkSource = active_backend == acmx2::Backend::Acmxvk && !shader_path.isEmpty() && is_acmxvk_source_library(shader_path, sourceTypeError) && sourceTypeError.isEmpty();
    if (runMenu_select)
        runMenu_select->setEnabled(launchAvailable);
    if (runMenu_all)
        runMenu_all->setEnabled(launchAvailable);
    if (runMenu_copyCommand)
        runMenu_copyCommand->setEnabled(launchAvailable);
    if (buildCacheAction) {
        buildCacheAction->setText(acmxvkSource ? tr("Build") : tr("Rebuild Shader Cache"));
        buildCacheAction->setVisible(acmx2Tools || acmxvkSource);
        buildCacheAction->setEnabled(acmx2Tools || acmxvkSource);
        buildCacheAction->setToolTip(acmxvkSource ? tr("Compile changed GLSL sources into %1").arg(acmxvk_build_directory(shader_path)) : QString());
    }
    if (fixBuildAction) {
        fixBuildAction->setVisible(acmxvkSource);
        fixBuildAction->setEnabled(acmxvkSource);
        fixBuildAction->setToolTip(acmxvkSource ? tr("Build into %1 and omit shaders that fail to compile").arg(acmxvk_build_directory(shader_path)) : QString());
    }
    if (cleanShaderCacheAction) {
        cleanShaderCacheAction->setVisible(acmx2Tools);
        cleanShaderCacheAction->setEnabled(acmx2Tools);
    }
    if (removeBrokenAction) {
        removeBrokenAction->setVisible(acmx2Tools || acmxvkSource);
        removeBrokenAction->setEnabled(acmx2Tools || acmxvkSource);
        removeBrokenAction->setToolTip(acmxvkSource ? tr("Permanently delete .frag and .comp sources that fail "
                                                         "the ACMXVK Fix Build")
                                                    : QString());
    }
    if (runFromCacheAction) {
        runFromCacheAction->setVisible(acmx2Tools);
        runFromCacheAction->setEnabled(acmx2Tools);
    }
#ifdef Q_OS_MACOS
    if (buildCacheAction && acmx2Tools) {
        buildCacheAction->setVisible(false);
        buildCacheAction->setEnabled(false);
    }
    if (cleanShaderCacheAction)
        cleanShaderCacheAction->setEnabled(false);
    if (runFromCacheAction)
        runFromCacheAction->setEnabled(false);
#endif
    const bool processIdle = !process || process->state() == QProcess::NotRunning;
    if (libraryBuilderAction)
        libraryBuilderAction->setEnabled(processIdle);
    if (listMenu_new)
        listMenu_new->setEnabled(processIdle);
    if (listMenu_shader)
        listMenu_shader->setEnabled(processIdle);
    if (list_view) {
#ifdef Q_OS_MACOS
        list_view->setColumnHidden(3, acmx2Tools);
#else
        list_view->setColumnHidden(3, false);
#endif
        list_view->headerItem()->setText(3, acmx2Tools ? tr("Compile Health") : tr("Build Status"));
    }

    if (runMenu) {
        runMenu_select->setToolTip({});
        runMenu_all->setToolTip({});
        runMenu_copyCommand->setToolTip({});
    }
    if (list_view) {
        list_view->setToolTip(tr("Right click while running to change the active shader."));
    }
}

void MainWindow::set_backend(acmx2::Backend backend, bool persist) {
    if (process && process->state() == QProcess::Running) {
        QMessageBox::information(this, tr("Process Running"), tr("Stop the running process before changing backends."));
        update_backend_ui();
        return;
    }

    if (libraryBuilderDialog) {
        libraryBuilderDialog->close();
        libraryBuilderDialog = nullptr;
    }
    if (effectPackBrowser && backend != acmx2::Backend::Acmxvk) {
        effectPackBrowser->hide();
    }
    if (deepDreamSettingsDialog) {
        deepDreamSettingsDialog->close();
        deepDreamSettingsDialog = nullptr;
    }
    if (gpuFilterDialog) {
        gpuFilterDialog->close();
        gpuFilterDialog = nullptr;
    }

    QSettings settings("LostSideDead");
    settings.setValue(acmx2::backend_settings_key(active_backend, "executable"), executable_path);
    settings.setValue(acmx2::backend_settings_key(active_backend, "library"), shader_path);

    active_backend = backend;
    if (persist)
        settings.setValue("interface/backend", acmx2::backend_id(active_backend));
    executable_path = settings.value(acmx2::backend_settings_key(active_backend, "executable"), acmx2::default_backend_executable(active_backend)).toString();
    const QString nextLibrary = settings.value(acmx2::backend_settings_key(active_backend, "library"), "").toString().trimmed();

    shader_path.clear();
    items.clear();
    indexTimestamp = QDateTime();
    activeShaderManifestPath.clear();
    if (list_view)
        list_view->clear();

    if (!nextLibrary.isEmpty() && QFileInfo(nextLibrary).isDir() && acmx2::shader_manifest_exists(nextLibrary)) {
        QString backendError;
        const std::optional<acmx2::Backend> libraryBackend = acmx2::shader_manifest_backend(nextLibrary, backendError);
        if (!backendError.isEmpty()) {
            Log(tr("Warning: Could not read backend metadata for %1: %2").arg(nextLibrary, backendError));
        } else if (libraryBackend && *libraryBackend != active_backend) {
            Log(tr("Warning: Saved %1 library belongs to %2: %3").arg(acmx2::backend_name(active_backend), acmx2::backend_name(*libraryBackend), nextLibrary));
        } else {
            shader_path = nextLibrary;
            loadShaders(shader_path, true);
        }
    }

    cuda_available = false;
    cuda_device_available = false;
    audio_available = false;
    midi_available = false;
    dnn_available = false;
    deep_dream_available = false;
    stable_diffusion_available = false;
    initShaderSelectionSharedMemory();
    detectFeatureSupport();
    updateRecentLibrariesMenu();
    update_backend_ui();
    settings.sync();
    Log(tr("Backend selected: %1").arg(acmx2::backend_name(active_backend)));
    if (active_backend == acmx2::Backend::Acmxvk)
        Log(tr("ACMXVK launching and live shader selection are enabled."));
}

bool MainWindow::loadLibraryPath(const QString &path) {
    const QString trimmedPath = path.trimmed();
    if (trimmedPath.isEmpty())
        return false;

    QProgressDialog progress(tr("Reading shader library metadata..."), QString(), 0, 0, this);
    if (show_shader_library_load_progress) {
        progress.setWindowTitle(tr("Loading Shader Library"));
        progress.setWindowModality(Qt::WindowModal);
        progress.setAutoClose(false);
        progress.setAutoReset(false);
        progress.setMinimumDuration(0);
        shader_library_progress_dialog = &progress;
        progress.show();
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    const QString libraryPath = QDir::cleanPath(trimmedPath);
    const QFileInfo libraryInfo(libraryPath);
    if (!libraryInfo.exists()) {
        QMessageBox::warning(this, tr("Invalid Shader Path"), tr("Shader directory does not exist:\n%1").arg(libraryPath));
        return false;
    }
    if (!libraryInfo.isDir()) {
        QMessageBox::warning(this, tr("Invalid Shader Path"), tr("Shader path is not a directory:\n%1").arg(libraryPath));
        return false;
    }
    if (!acmx2::shader_manifest_exists(libraryPath)) {
        QMessageBox::warning(this, tr("Missing Shader Manifest"), tr("Shader directory does not contain library.json or index.txt:\n%1").arg(libraryPath));
        return false;
    }
    QString backendError;
    const std::optional<acmx2::Backend> libraryBackend = acmx2::shader_manifest_backend(libraryPath, backendError);
    if (!backendError.isEmpty()) {
        QMessageBox::warning(this, tr("Invalid Backend Metadata"), backendError);
        return false;
    }
    if (libraryBackend && *libraryBackend != active_backend) {
        const QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                                        tr("Switch Backend"),
                                                                        tr("This library targets %1, but the active backend is %2.\n\n"
                                                                           "Switch to %1 and load it?")
                                                                            .arg(acmx2::backend_name(*libraryBackend), acmx2::backend_name(active_backend)),
                                                                        QMessageBox::Yes | QMessageBox::No,
                                                                        QMessageBox::Yes);
        if (reply != QMessageBox::Yes)
            return false;
        set_backend(*libraryBackend);
    }
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString libraryTypeError;
        acmx2::shader_manifest_library_type(libraryPath, libraryTypeError);
        if (!libraryTypeError.isEmpty()) {
            QMessageBox::warning(this, tr("Invalid Library Type"), libraryTypeError);
            return false;
        }
    }
    if (!loadShaders(libraryPath, true)) {
        Log(tr("Warning: Could not load shaders from directory: %1").arg(libraryPath));
        return false;
    }

    if (shader_library_progress_dialog) {
        progress.close();
        shader_library_progress_dialog.clear();
    }

    QSettings settings("LostSideDead");
    settings.setValue(acmx2::backend_settings_key(active_backend, "library"), libraryPath);
    if (active_backend == acmx2::Backend::Acmx2)
        settings.setValue("shaders", libraryPath);
    settings.sync();
    addRecentLibrary(libraryPath);
    Log(tr("Successfully loaded shader library: %1").arg(libraryPath));
    update_backend_ui();
    return true;
}

void MainWindow::addRecentLibrary(const QString &path) {
    const QString trimmedPath = path.trimmed();
    if (trimmedPath.isEmpty())
        return;
    const QString libraryPath = QDir::cleanPath(trimmedPath);

    QSettings settings("LostSideDead");
    const QString recentKey = acmx2::backend_settings_key(active_backend, "recentLibraries");
    const QStringList legacyRecent = active_backend == acmx2::Backend::Acmx2 ? settings.value("recentLibraries").toStringList() : QStringList();
    QStringList recentLibraries = settings.value(recentKey, legacyRecent).toStringList();
    for (auto it = recentLibraries.begin(); it != recentLibraries.end();) {
        if (QDir::cleanPath(*it).compare(libraryPath, Qt::CaseInsensitive) == 0)
            it = recentLibraries.erase(it);
        else
            ++it;
    }
    recentLibraries.prepend(libraryPath);
    while (recentLibraries.size() > RECENT_LIBRARY_LIMIT)
        recentLibraries.removeLast();
    settings.setValue(recentKey, recentLibraries);
    if (active_backend == acmx2::Backend::Acmx2)
        settings.setValue("recentLibraries", recentLibraries);
    settings.sync();
    updateRecentLibrariesMenu();
}

void MainWindow::updateRecentLibrariesMenu() {
    if (!loadRecentMenu)
        return;

    loadRecentMenu->clear();
    QSettings settings("LostSideDead");
    const QString recentKey = acmx2::backend_settings_key(active_backend, "recentLibraries");
    const QStringList legacyRecent = active_backend == acmx2::Backend::Acmx2 ? settings.value("recentLibraries").toStringList() : QStringList();
    const QStringList recentLibraries = settings.value(recentKey, legacyRecent).toStringList();
    if (recentLibraries.isEmpty()) {
        QAction *emptyAction = loadRecentMenu->addAction(tr("No Recent Libraries"));
        emptyAction->setEnabled(false);
        return;
    }

    for (const QString &path : recentLibraries) {
        QAction *action = loadRecentMenu->addAction(path);
        connect(action, &QAction::triggered, this, [this, path]() { loadLibraryPath(path); });
    }
}

void MainWindow::addRecentPreset(const QString &path) {
    const QString presetPath = QFileInfo(path).absoluteFilePath();
    if (presetPath.isEmpty())
        return;

    QSettings settings("LostSideDead");
    QStringList recentPresets = settings.value("projects/recent", settings.value("presets/recent")).toStringList();
    recentPresets.removeAll(presetPath);
    recentPresets.prepend(presetPath);
    while (recentPresets.size() > RECENT_PRESET_LIMIT)
        recentPresets.removeLast();
    settings.setValue("projects/recent", recentPresets);
    settings.sync();
    updateRecentPresetsMenu();
}

void MainWindow::set_current_project_path(const QString &path) {
    current_project_path = path.isEmpty() ? QString() : QFileInfo(path).absoluteFilePath();
    QSettings settings("LostSideDead");
    if (current_project_path.isEmpty()) {
        settings.remove("projects/last_open");
    } else {
        settings.setValue("projects/last_open", current_project_path);
    }
    settings.sync();
    update_backend_ui();
}

void MainWindow::updateRecentPresetsMenu() {
    if (!recentPresetsMenu)
        return;

    recentPresetsMenu->clear();
    const QSettings settings("LostSideDead");
    const QStringList recentPresets = settings.value("projects/recent", settings.value("presets/recent")).toStringList();
    bool added = false;
    for (const QString &path : recentPresets) {
        if (!QFileInfo::exists(path))
            continue;
        QAction *action = recentPresetsMenu->addAction(QFileInfo(path).fileName());
        action->setToolTip(path);
        connect(action, &QAction::triggered, this, [this, path]() { importPreset(path); });
        added = true;
    }
    if (!added) {
        QAction *emptyAction = recentPresetsMenu->addAction(tr("No Recent Projects"));
        emptyAction->setEnabled(false);
    }
}

bool MainWindow::savePreset(const QString &path, const QString &outputExtension, bool exportProject) {
    if (active_backend != acmx2::Backend::Acmxvk) {
        QMessageBox::information(this, tr("Save Project"), tr("Projects are available only with the ACMXVK backend."));
        return false;
    }
    if (effectPackBrowser && effectPackBrowser->has_project_pack() && !effectPackBrowser->has_active_pack()) {
        QMessageBox::warning(this, tr("Save Project"), tr("Rebuild and activate the project's effect pack before saving this project again."));
        return false;
    }
    const QFileInfo existing_output(output_file);
    const QString output_base_name = output_file.isEmpty() ? QFileInfo(path).completeBaseName() : untimestamped_output_base_name(existing_output.filePath());
    const QString project_output_path = QDir(QFileInfo(path).absolutePath()).filePath(QStringLiteral("output/%1.%2").arg(output_base_name, outputExtension));
    QStringList acmxvk_arguments;
    if (!buildRunArguments(acmxvk_arguments, PendingAcmxvkAction::CopyCommand, true, project_output_path))
        return false;

    ProjectSaveRequest request;
    request.path = path;
    request.interface_settings = preset_settings(QSettings("LostSideDead", "acmx2"));
    request.application_settings = preset_settings(QSettings("LostSideDead"));
    request.application_settings.remove("presets/recent");
    request.application_settings.remove("recentLibraries");
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmx2, "recentLibraries"));
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "recentLibraries"));
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmx2, "executable"));
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "executable"));
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "shader_compiler_path"));
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmx2, "library"));
    request.application_settings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "library"));
    request.shader_library = shader_path;
    request.selected_shader = currentShaderName();
    request.repeat = play_repeat && play_repeat->isChecked();
    request.acmxvk_arguments = acmxvk_arguments;
    request.video_file = video_file;
    request.graphics_file = graphics_file;
    request.audio_file = audio_file;
    request.model_file = model_file;
    request.onnx_model = onnx_model;
    request.deep_dream_model = deep_dream_model;
    request.stable_diffusion_model = stable_diffusion_model;
    request.stable_diffusion_upscale_model = stable_diffusion_upscale_model;
    request.stable_diffusion_loras = stable_diffusion_lora_files;
    request.midi_config_file = midi_config_file;
    request.playlist_file = playlist_file_path;
    request.output_file = project_output_path;
    request.output_extension = outputExtension;
    if (effectPackBrowser && effectPackBrowser->has_active_pack())
        request.effect_pack = effectPackBrowser->project_state();
    if (effectPackBrowser && effectPackBrowser->has_active_pack() && request.effect_pack.manifest_path.isEmpty()) {
        QMessageBox::warning(this, tr("Save Project"), tr("The active effect pack could not be captured for this project."));
        return false;
    }
    request.enable_3d = enable_3d;
    request.onnx_model_enabled = onnx_model_enabled;
    request.deep_dream_enabled = deep_dream_enabled;
    request.stable_diffusion_enabled = stable_diffusion_enabled;
    request.stable_diffusion_upscale_only = stable_diffusion_upscale_only;
    request.midi_enabled = midi_enabled;
    request.playlist_enabled = playlist_enabled;
    request.png_output = png_output;
    request.create_output_directories = !exportProject;
    request.runtime.insert("gpu_filter_enabled", gpu_filter_enabled);
    request.runtime.insert("gpu_filter_indices", gpu_filter_indices);
    request.runtime.insert("gpu_buffer_size", gpu_buffer_size);
    request.runtime.insert("shader_pass_enabled", shader_pass_enabled);
    request.runtime.insert("playlist_enabled", playlist_enabled);
    request.runtime.insert("autopilot_frames", autopilot_frames);
    request.runtime.insert("autopilot_random", autopilot_random);
    request.runtime.insert("stay_on_top", stayOnTopAction && stayOnTopAction->isChecked());
    QJsonArray shader_passes;
    for (const QString &name : shader_pass_names)
        shader_passes.append(name);
    request.runtime.insert("shader_passes", shader_passes);
    QJsonArray playlist_names_json;
    for (const QString &name : playlist_names)
        playlist_names_json.append(name);
    request.runtime.insert("playlist_names", playlist_names_json);
    QJsonArray playlist_tree;
    for (const auto &[node_name, node_shaders] : playlist_tree_data) {
        QJsonObject node;
        node.insert("name", node_name);
        QJsonArray shaders;
        for (const QString &shader : node_shaders)
            shaders.append(shader);
        node.insert("shaders", shaders);
        playlist_tree.append(node);
    }
    request.runtime.insert("playlist_tree", playlist_tree);

    const auto progress = std::make_shared<ProjectProgress>();
    request.progress = progress;
    auto *dialog = new QProgressDialog(exportProject ? tr("Preparing exported project resources...") : tr("Preparing project resources..."), tr("Cancel"), 0, 0, this);
    dialog->setWindowTitle(exportProject ? tr("Exporting Project") : tr("Saving Project"));
    dialog->setWindowModality(Qt::WindowModal);
    dialog->setAutoClose(false);
    dialog->setAutoReset(false);
    dialog->setMinimumDuration(0);
    dialog->show();

    auto *watcher = new QFutureWatcher<ProjectSaveResult>(this);
    auto *timer = new QTimer(watcher);
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, this, [dialog, progress]() {
        const qint64 total_mb = (progress->total_bytes.load() + 1048575) / 1048576;
        const qint64 copied_mb = (progress->copied_bytes.load() + 1048575) / 1048576;
        if (total_mb <= 0) {
            dialog->setRange(0, 0);
            dialog->setLabelText(QObject::tr("Preparing project resources..."));
            return;
        }
        const int maximum = static_cast<int>(std::min<qint64>(total_mb, std::numeric_limits<int>::max()));
        dialog->setRange(0, maximum);
        dialog->setValue(static_cast<int>(std::min<qint64>(copied_mb, maximum)));
        dialog->setLabelText(QObject::tr("Copying project resources: %1 MB of %2 MB").arg(copied_mb).arg(total_mb));
    });
    connect(dialog, &QProgressDialog::canceled, this, [progress]() { progress->cancelled.store(true); });
    connect(watcher, &QFutureWatcher<ProjectSaveResult>::finished, this, [this, watcher, timer, dialog, path, project_output_path, exportProject]() {
        timer->stop();
        const ProjectSaveResult result = watcher->result();
        dialog->close();
        dialog->deleteLater();
        watcher->deleteLater();
        if (!result.success) {
            QMessageBox::warning(this, exportProject ? tr("Export Project") : tr("Save Project"), result.message);
            Log(exportProject ? tr("Project export failed: %1").arg(result.message) : tr("Project save failed: %1").arg(result.message));
            return;
        }
        if (exportProject) {
            Log(tr("Exported portable project: %1").arg(path));
            return;
        }
        output_file = project_output_path;
        project_output_directory = QFileInfo(project_output_path).absolutePath();
        project_output_filename = QFileInfo(project_output_path).fileName();
        QSettings settings("LostSideDead", "acmx2");
        settings.setValue("interface/save_output", true);
        settings.setValue("interface/output_video", output_file);
        settings.sync();
        addRecentPreset(path);
        set_current_project_path(path);
        Log(tr("Saved portable project: %1").arg(path));
    });
    timer->start();
    watcher->setFuture(QtConcurrent::run([request = std::move(request)]() mutable { return save_project_bundle(std::move(request)); }));
    return true;
}

bool MainWindow::savePresetSynchronously(const QString &path) {
    const QString projectRoot = QFileInfo(path).absolutePath();
    if (!QDir().mkpath(projectRoot)) {
        QMessageBox::warning(this, tr("Save Project"), tr("Could not create project directory:\n%1").arg(projectRoot));
        return false;
    }
    ProjectResources resources(projectRoot, std::make_shared<ProjectProgress>());
    QString resourceError;
    QStringList acmxvkArguments;
    if (active_backend == acmx2::Backend::Acmxvk && !buildRunArguments(acmxvkArguments, PendingAcmxvkAction::CopyCommand, true))
        return false;

    QJsonObject interfaceSettings = preset_settings(QSettings("LostSideDead", "acmx2"));
    QJsonObject applicationSettings = preset_settings(QSettings("LostSideDead"));
    normalize_project_parallel_build_settings(interfaceSettings, applicationSettings);
    applicationSettings.remove("presets/recent");
    applicationSettings.remove("recentLibraries");
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmx2, "recentLibraries"));
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "recentLibraries"));
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmx2, "executable"));
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "executable"));
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "shader_compiler_path"));
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmx2, "library"));
    applicationSettings.remove(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "library"));

    const auto copyResource = [&resources, &resourceError](const QString &source, const QString &category) { return resources.copy_path(source, category, resourceError); };
    QString projectLibrary;
    if (!shader_path.isEmpty()) {
        projectLibrary = copyResource(shader_path, QStringLiteral("resources/shaders"));
        if (projectLibrary.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        applicationSettings.insert(acmx2::backend_settings_key(active_backend, "library"), projectLibrary);
    }
    if (!video_file.isEmpty()) {
        const QString video = copyResource(video_file, QStringLiteral("resources/media"));
        if (video.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        set_project_path(interfaceSettings, QStringLiteral("interface/input_video"), video);
    }
    if (!graphics_file.isEmpty()) {
        const QString graphic = copyResource(graphics_file, QStringLiteral("resources/media"));
        if (graphic.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        set_project_path(interfaceSettings, QStringLiteral("interface/graphics_file"), graphic);
    }
    if (!audio_file.isEmpty()) {
        const bool audioPlaylist = interfaceSettings.value("audio/playlist_enabled").toBool();
        const QString audio = audioPlaylist ? copy_audio_playlist(resources, audio_file, resourceError) : copyResource(audio_file, QStringLiteral("resources/audio"));
        if (audio.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        set_project_path(interfaceSettings, audioPlaylist ? QStringLiteral("audio/playlist_path") : QStringLiteral("audio/file_path"), audio);
    }
    if (enable_3d && !model_file.isEmpty()) {
        const QString model = copyResource(model_file, QStringLiteral("resources/models"));
        if (model.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        set_project_path(interfaceSettings, QStringLiteral("interface/model_file"), model);
    }
    if (onnx_model_enabled && !onnx_model.isEmpty()) {
        const QString model = copyResource(onnx_model, QStringLiteral("resources/models"));
        if (model.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        set_project_path(interfaceSettings, QStringLiteral("interface/onnx_model_file"), model);
    }
    if (deep_dream_enabled && !deep_dream_model.isEmpty()) {
        const QString model = copyResource(deep_dream_model, QStringLiteral("resources/models"));
        if (model.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        if (QFileInfo::exists(deep_dream_model + QStringLiteral(".json")) && copyResource(deep_dream_model + QStringLiteral(".json"), QStringLiteral("resources/models")).isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        set_project_path(interfaceSettings, QStringLiteral("deep_dream/model_file"), model);
    }
    if (stable_diffusion_enabled) {
        if (!stable_diffusion_upscale_only && !stable_diffusion_model.isEmpty()) {
            const QString model = copyResource(stable_diffusion_model, QStringLiteral("resources/models"));
            if (model.isEmpty()) {
                QMessageBox::warning(this, tr("Save Project"), resourceError);
                return false;
            }
            set_project_path(interfaceSettings, QStringLiteral("stable_diffusion/model_file"), model);
        }
        if (!stable_diffusion_upscale_model.isEmpty()) {
            const QString model = copyResource(stable_diffusion_upscale_model, QStringLiteral("resources/models"));
            if (model.isEmpty()) {
                QMessageBox::warning(this, tr("Save Project"), resourceError);
                return false;
            }
            set_project_path(interfaceSettings, QStringLiteral("stable_diffusion/upscale_model_file"), model);
        }
        QJsonArray loras;
        for (const QString &lora : stable_diffusion_lora_files) {
            const QString copiedLora = copyResource(lora, QStringLiteral("resources/models/loras"));
            if (copiedLora.isEmpty()) {
                QMessageBox::warning(this, tr("Save Project"), resourceError);
                return false;
            }
            loras.append(copiedLora);
        }
        interfaceSettings.insert("stable_diffusion/lora_files", loras);
    }
    if (midi_enabled && !midi_config_file.isEmpty()) {
        const QString midiMap = copyResource(midi_config_file, QStringLiteral("resources/midi"));
        if (midiMap.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
        applicationSettings.insert("midiConfigFile", midiMap);
    }
    QString projectPlaylist;
    if (playlist_enabled && !playlist_file_path.isEmpty()) {
        projectPlaylist = copyResource(playlist_file_path, QStringLiteral("resources/playlists"));
        if (projectPlaylist.isEmpty()) {
            QMessageBox::warning(this, tr("Save Project"), resourceError);
            return false;
        }
    }

    if (!output_file.isEmpty()) {
        const QString output = QDir(QStringLiteral("output")).filePath(QFileInfo(output_file).fileName());
        set_project_path(interfaceSettings, QStringLiteral("interface/output_video"), output);
    }
    if (png_output) {
        set_project_path(interfaceSettings, QStringLiteral("interface/png_output_directory"), QStringLiteral("output/png-sequence"));
    }
    applicationSettings.insert("prefix_path", QStringLiteral("output/snapshots"));

    QStringList portableArguments;
    const QSet<QString> resourceOptions = {"--input", "--graphic", "--audio-file", "--shaders", "--model", "--onnx", "--dream-model", "--sd-model", "--upscale-model", "--sd-lora", "--playlist", "--midi-map"};
    for (int index = 0; index < acmxvkArguments.size(); ++index) {
        const QString argument = acmxvkArguments.at(index);
        if (argument == QStringLiteral("--path")) {
            ++index;
            continue;
        }
        portableArguments.append(argument);
        if (index + 1 >= acmxvkArguments.size())
            continue;
        const QString value = acmxvkArguments.at(index + 1);
        if (resourceOptions.contains(argument)) {
            const QString category = argument == QStringLiteral("--shaders") ? QStringLiteral("resources/shaders") : QStringLiteral("resources/external");
            const QString resource = copyResource(value, category);
            if (resource.isEmpty()) {
                QMessageBox::warning(this, tr("Save Project"), resourceError);
                return false;
            }
            portableArguments.append(resource);
            ++index;
        } else if (argument == QStringLiteral("--output") || argument == QStringLiteral("--record-audio")) {
            portableArguments.append(QDir(QStringLiteral("output")).filePath(QFileInfo(value).fileName()));
            ++index;
        } else if (argument == QStringLiteral("--prefix")) {
            portableArguments.append(QStringLiteral("output/snapshots"));
            ++index;
        }
    }

    QJsonObject root;
    root.insert("format", "acmx-project");
    root.insert("version", 2);
    root.insert("interface_settings", interfaceSettings);
    root.insert("application_settings", applicationSettings);
    root.insert("shader_library", projectLibrary);
    root.insert("selected_shader", currentShaderName());
    root.insert("repeat", play_repeat && play_repeat->isChecked());

    QJsonObject runtime;
    runtime.insert("gpu_filter_enabled", gpu_filter_enabled);
    runtime.insert("gpu_filter_indices", gpu_filter_indices);
    runtime.insert("gpu_buffer_size", gpu_buffer_size);
    runtime.insert("shader_pass_enabled", shader_pass_enabled);
    runtime.insert("playlist_enabled", playlist_enabled);
    runtime.insert("playlist_file", projectPlaylist);
    runtime.insert("autopilot_frames", autopilot_frames);
    runtime.insert("autopilot_random", autopilot_random);
    runtime.insert("stay_on_top", stayOnTopAction && stayOnTopAction->isChecked());
    QJsonArray shaderPasses;
    for (const QString &name : shader_pass_names)
        shaderPasses.append(name);
    runtime.insert("shader_passes", shaderPasses);
    QJsonArray playlistNames;
    for (const QString &name : playlist_names)
        playlistNames.append(name);
    runtime.insert("playlist_names", playlistNames);
    QJsonArray playlistTree;
    for (const auto &[nodeName, nodeShaders] : playlist_tree_data) {
        QJsonObject node;
        node.insert("name", nodeName);
        QJsonArray shaders;
        for (const QString &shader : nodeShaders)
            shaders.append(shader);
        node.insert("shaders", shaders);
        playlistTree.append(node);
    }
    runtime.insert("playlist_tree", playlistTree);
    root.insert("runtime", runtime);

    QJsonArray arguments;
    for (const QString &argument : portableArguments)
        arguments.append(argument);
    root.insert("acmxvk_arguments", arguments);

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("Save Project"), tr("Could not write project:\n%1").arg(path));
        return false;
    }
    if (file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0 || !file.commit()) {
        QMessageBox::warning(this, tr("Save Project"), tr("Could not finish writing project:\n%1").arg(path));
        return false;
    }
    addRecentPreset(path);
    set_current_project_path(path);
    Log(tr("Saved portable project: %1").arg(path));
    return true;
}

bool MainWindow::importPreset(const QString &path) {
    auto *dialog = new QProgressDialog(tr("Reading project settings..."), QString(), 0, 0, this);
    dialog->setWindowTitle(tr("Loading Project"));
    dialog->setWindowModality(Qt::WindowModal);
    dialog->setAutoClose(false);
    dialog->setAutoReset(false);
    dialog->setMinimumDuration(0);
    dialog->show();

    auto *watcher = new QFutureWatcher<ProjectLoadResult>(this);
    connect(watcher, &QFutureWatcher<ProjectLoadResult>::finished, this, [this, watcher, dialog, path]() {
        const ProjectLoadResult result = watcher->result();
        dialog->close();
        dialog->deleteLater();
        watcher->deleteLater();
        if (!result.success) {
            QMessageBox::warning(this, tr("Load Project"), result.message);
            return;
        }
        applyProjectDocument(path, result.document);
    });
    watcher->setFuture(QtConcurrent::run([path]() { return load_project_document(path); }));
    return true;
}

bool MainWindow::applyProjectDocument(const QString &path, const QJsonDocument &document) {
    Log(tr("Loading project: %1").arg(path));
    const QJsonObject root = document.object();
    const int projectVersion = root.value("version").toInt();
    const bool legacyPreset = root.value("format").toString() == QStringLiteral("acmx-preset") && projectVersion == 1;
    const bool portableProject = root.value("format").toString() == QStringLiteral("acmx-project") && projectVersion == 2;
    if (!legacyPreset && !portableProject) {
        QMessageBox::warning(this, tr("Load Project"), tr("This is not a supported ACMX project file."));
        return false;
    }
    if (!root.value("interface_settings").isObject() || !root.value("application_settings").isObject()) {
        QMessageBox::warning(this, tr("Load Project"), tr("The project does not contain interface settings."));
        return false;
    }

    QJsonObject projectInterfaceSettings = root.value("interface_settings").toObject();
    QJsonObject projectApplicationSettings = root.value("application_settings").toObject();
    QString projectLibrary = root.value("shader_library").toString();
    QJsonObject runtime = root.value("runtime").toObject();
    if (portableProject) {
        const QString projectRoot = QFileInfo(path).absolutePath();
        if (!QDir().mkpath(QDir(projectRoot).filePath(QStringLiteral("output/snapshots"))) || !QDir().mkpath(QDir(projectRoot).filePath(QStringLiteral("output/logs")))) {
            QMessageBox::warning(this, tr("Load Project"), tr("Could not create the project output directories."));
            return false;
        }
        for (const QString &key : {QStringLiteral("interface/input_video"), QStringLiteral("interface/graphics_file"), QStringLiteral("interface/model_file"), QStringLiteral("interface/onnx_model_file"), QStringLiteral("deep_dream/model_file"), QStringLiteral("stable_diffusion/model_file"), QStringLiteral("stable_diffusion/upscale_model_file"), QStringLiteral("audio/file_path"), QStringLiteral("audio/playlist_path"), QStringLiteral("interface/output_video"), QStringLiteral("interface/png_output_directory")}) {
            resolve_project_path(projectInterfaceSettings, key, projectRoot);
        }
        resolve_project_path_list(projectInterfaceSettings, QStringLiteral("stable_diffusion/lora_files"), projectRoot);
        for (const QString &key : {QStringLiteral("midiConfigFile"), QStringLiteral("prefix_path"), acmx2::backend_settings_key(acmx2::Backend::Acmx2, "library"), acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "library")}) {
            resolve_project_path(projectApplicationSettings, key, projectRoot);
        }
        projectLibrary = project_path(projectRoot, projectLibrary);
        runtime.insert("playlist_file", project_path(projectRoot, runtime.value("playlist_file").toString()));
        const QString outputName = QFileInfo(projectInterfaceSettings.value("interface/output_video").toString()).fileName().isEmpty() ? QFileInfo(path).completeBaseName() + QStringLiteral(".mp4") : QFileInfo(projectInterfaceSettings.value("interface/output_video").toString()).fileName();
        projectInterfaceSettings.insert("interface/save_output", true);
        projectInterfaceSettings.insert("interface/output_video", QDir(projectRoot).filePath(QStringLiteral("output/") + outputName));
    }

    QSettings interfaceSettings("LostSideDead", "acmx2");
    QSettings applicationSettings("LostSideDead");
    apply_preset_settings(interfaceSettings, projectInterfaceSettings);
    apply_preset_settings(applicationSettings, projectApplicationSettings);
    if (portableProject) {
        set_backend(acmx2::Backend::Acmxvk, false);
    } else {
        const std::optional<acmx2::Backend> backend = acmx2::backend_from_id(root.value("backend").toString());
        if (backend)
            set_backend(*backend, false);
    }
    loadSessionSettings();

    if (portableProject) {
        project_output_directory = QDir(QFileInfo(path).absolutePath()).filePath(QStringLiteral("output"));
        project_output_filename = QFileInfo(output_file).fileName();
    } else {
        project_output_directory.clear();
        project_output_filename.clear();
    }

    prefix_path = applicationSettings.value("prefix_path", QDir(QFileInfo(path).absolutePath()).filePath(QStringLiteral("output/snapshots"))).toString();
    if (portableProject && prefix_path.isEmpty())
        prefix_path = QDir(QFileInfo(path).absolutePath()).filePath(QStringLiteral("output/snapshots"));

    audio_enabled = interfaceSettings.value("audio/enabled", false).toBool();
    audio_channels = static_cast<unsigned int>(std::max(1, interfaceSettings.value("audio/channels", 2).toInt()));
    audio_sense = static_cast<float>(interfaceSettings.value("audio/sensitivity", 10).toDouble() / 10.0);
    audio_passthrough = interfaceSettings.value("audio/passthrough", false).toBool();
    record_audio = interfaceSettings.value("audio/record", false).toBool();
    record_volume = interfaceSettings.value("audio/record_volume", 100).toDouble() / 100.0;
    audio_input = interfaceSettings.value("audio/input_device", -1).toInt();
    audio_output = interfaceSettings.value("audio/output_device", -1).toInt();
    const bool audioPlaylistEnabled = interfaceSettings.value("audio/playlist_enabled", false).toBool();
    const bool audioFileEnabled = interfaceSettings.value("audio/file_enabled", false).toBool();
    if (audioPlaylistEnabled) {
        audio_file = interfaceSettings.value("audio/playlist_path").toString();
    } else if (audioFileEnabled) {
        audio_file = interfaceSettings.value("audio/file_path").toString();
    } else {
        audio_file.clear();
    }
    audio_trunc = interfaceSettings.value("audio/file_trunc", false).toBool();
    audio_repeat = interfaceSettings.value("audio/file_repeat", false).toBool();
    audio_buffers_enabled = interfaceSettings.value("audio/buffers_enabled", false).toBool();
    audio_buffer_frames = std::max(1, interfaceSettings.value("audio/buffers_frames", 8).toInt());
    audio_warm_rate = std::max(0.0, interfaceSettings.value("audio/warm_rate", 0.5).toDouble());
    midi_enabled = applicationSettings.value("midiEnabled", false).toBool();
    midi_config_file = applicationSettings.value("midiConfigFile").toString();
    midi_device = applicationSettings.value("midiDevice", -1).toInt();
    watermark_enabled = applicationSettings.value("watermarkEnabled", false).toBool();
    watermark_text = applicationSettings.value("watermarkText").toString();
    watermark_r = applicationSettings.value("watermarkR", 255).toInt();
    watermark_g = applicationSettings.value("watermarkG", 0).toInt();
    watermark_b = applicationSettings.value("watermarkB", 150).toInt();
    display_filter_enabled = applicationSettings.value("displayFilter", false).toBool();
    if (displayFilterAction) {
        QSignalBlocker blocker(displayFilterAction);
        displayFilterAction->setChecked(display_filter_enabled);
    }
    customStyleSheet = applicationSettings.value("customStyleSheet", acmx2::defaultCustomStyleSheet()).toString();
    const bool useCustomStyle = applicationSettings.value("useCustomStyle", false).toBool();
    if (styleSheetAction) {
        QSignalBlocker blocker(styleSheetAction);
        styleSheetAction->setChecked(useCustomStyle);
    }
    applyCustomStyleSheet(useCustomStyle);

    gpu_filter_enabled = runtime.value("gpu_filter_enabled").toBool(gpu_filter_enabled);
    gpu_filter_indices = runtime.value("gpu_filter_indices").toString(gpu_filter_indices);
    gpu_buffer_size = runtime.value("gpu_buffer_size").toInt(gpu_buffer_size);
    shader_pass_enabled = runtime.value("shader_pass_enabled").toBool(shader_pass_enabled);
    playlist_enabled = runtime.value("playlist_enabled").toBool(playlist_enabled);
    playlist_file_path = runtime.value("playlist_file").toString(playlist_file_path);
    autopilot_frames = runtime.value("autopilot_frames").toInt(autopilot_frames);
    autopilot_random = runtime.value("autopilot_random").toBool(autopilot_random);
    if (stayOnTopAction)
        stayOnTopAction->setChecked(runtime.value("stay_on_top").toBool(false));
    shader_pass_names.clear();
    for (const QJsonValue &value : runtime.value("shader_passes").toArray())
        shader_pass_names.append(value.toString());
    playlist_names.clear();
    for (const QJsonValue &value : runtime.value("playlist_names").toArray())
        playlist_names.append(value.toString());
    playlist_tree_data.clear();
    for (const QJsonValue &value : runtime.value("playlist_tree").toArray()) {
        const QJsonObject node = value.toObject();
        const QString nodeName = node.value("name").toString();
        if (nodeName.isEmpty())
            continue;
        QStringList nodeShaders;
        for (const QJsonValue &shader : node.value("shaders").toArray())
            nodeShaders.append(shader.toString());
        playlist_tree_data.append({nodeName, nodeShaders});
    }
    if (play_repeat) {
        QSignalBlocker blocker(play_repeat);
        play_repeat->setChecked(root.value("repeat").toBool(false));
    }

    const QString library = projectLibrary;
    Log(tr("Loading project shader library..."));
    QCoreApplication::processEvents();
    show_shader_library_load_progress = true;
    const bool libraryLoaded = !library.isEmpty() && loadLibraryPath(library);
    show_shader_library_load_progress = false;
    if (libraryLoaded) {
        const int row = items.indexOf(root.value("selected_shader").toString());
        if (row >= 0)
            selectShaderRow(row);
    }
    if (effectPackBrowser)
        effectPackBrowser->clear_project_pack();
    if (portableProject && root.value(QStringLiteral("effect_pack")).isObject()) {
        EffectPackProjectState pack_state;
        QString pack_error;
        if (acmx2::resolve_effect_pack_project_state(root.value(QStringLiteral("effect_pack")).toObject(), QFileInfo(path).absolutePath(), pack_state, pack_error)) {
            ensureEffectPackBrowser();
            if (!effectPackBrowser->restore_project_pack(pack_state, pack_error)) {
                Log(tr("Project effect pack was not activated: %1").arg(pack_error));
                QString recovery_error;
                if (!effectPackBrowser->queue_project_pack_for_build(pack_state, recovery_error))
                    Log(tr("Project effect pack cannot be rebuilt: %1").arg(recovery_error));
            }
        } else {
            Log(tr("Project effect pack was not activated: %1").arg(pack_error));
        }
    }
    if (!effectPackBrowser || !effectPackBrowser->has_active_pack()) {
        publishEffectPackToRunningProcess({}, {}, regular_deep_dream_configuration());
    }
    publishRuntimeSettingsToRunningProcess();
    publishMultipassShadersToRunningProcess();
    addRecentPreset(path);
    set_current_project_path(path);
    Log(tr("Loaded project: %1").arg(path));
    return true;
}

void MainWindow::menuSaveProject() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    if (current_project_path.isEmpty()) {
        menuSaveProjectAs();
        return;
    }

    QString outputFormat = QFileInfo(output_file).suffix().toLower();
    if (outputFormat.isEmpty())
        outputFormat = QFileInfo(project_output_filename).suffix().toLower();
    if (outputFormat.isEmpty())
        outputFormat = QStringLiteral("mp4");
    savePreset(current_project_path, outputFormat);
}

void MainWindow::menuSaveProjectAs() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    const QString baseDirectory = QFileDialog::getExistingDirectory(this, tr("Choose Project Location"), QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    if (baseDirectory.isEmpty())
        return;
    bool accepted = false;
    QString projectName = QInputDialog::getText(this, tr("Project Name"), tr("Project name:"), QLineEdit::Normal, tr("acmx-project"), &accepted).trimmed();
    if (!accepted || projectName.isEmpty())
        return;
    projectName.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]+")), QStringLiteral("-"));
    if (projectName.isEmpty())
        projectName = QStringLiteral("acmx-project");
    const QString projectDirectory = QDir(baseDirectory).filePath(projectName);
    if (QDir(projectDirectory).exists() && !QDir(projectDirectory).entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty()) {
        QMessageBox::warning(this, tr("Save Project"), tr("Choose a new project name. The project directory already contains files:\n%1").arg(projectDirectory));
        return;
    }
    const QStringList formats = {QStringLiteral("mp4"), QStringLiteral("mkv"), QStringLiteral("mov"), QStringLiteral("webm"), QStringLiteral("avi")};
    const int defaultFormat = std::max(0, static_cast<int>(formats.indexOf(QFileInfo(output_file).suffix().toLower())));
    const QString outputFormat = QInputDialog::getItem(this, tr("Project Video Format"), tr("Project output format:"), formats, defaultFormat, false, &accepted);
    if (!accepted)
        return;
    savePreset(QDir(projectDirectory).filePath(projectName + QStringLiteral(".acmxproj")), outputFormat);
}

void MainWindow::menuExportProject() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    const QString initialDirectory = current_project_path.isEmpty() ? QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) : QFileInfo(current_project_path).absolutePath();
    const QString baseDirectory = QFileDialog::getExistingDirectory(this, tr("Choose Export Location"), initialDirectory);
    if (baseDirectory.isEmpty())
        return;
    bool accepted = false;
    const QString defaultName = current_project_path.isEmpty() ? QStringLiteral("acmx-project-export") : QFileInfo(current_project_path).completeBaseName() + QStringLiteral("-export");
    QString projectName = QInputDialog::getText(this, tr("Export Project"), tr("Export name:"), QLineEdit::Normal, defaultName, &accepted).trimmed();
    if (!accepted || projectName.isEmpty())
        return;
    projectName.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]+")), QStringLiteral("-"));
    if (projectName.isEmpty())
        projectName = QStringLiteral("acmx-project-export");
    const QString projectDirectory = QDir(baseDirectory).filePath(projectName);
    if (QDir(projectDirectory).exists() && !QDir(projectDirectory).entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty()) {
        QMessageBox::warning(this, tr("Export Project"), tr("Choose a new export name. The export directory already contains files:\n%1").arg(projectDirectory));
        return;
    }
    const QStringList formats = {QStringLiteral("mp4"), QStringLiteral("mkv"), QStringLiteral("mov"), QStringLiteral("webm"), QStringLiteral("avi")};
    const int defaultFormat = std::max(0, static_cast<int>(formats.indexOf(QFileInfo(output_file).suffix().toLower())));
    const QString outputFormat = QInputDialog::getItem(this, tr("Project Video Format"), tr("Project output format:"), formats, defaultFormat, false, &accepted);
    if (!accepted)
        return;
    savePreset(QDir(projectDirectory).filePath(projectName + QStringLiteral(".acmxproj")), outputFormat, true);
}

void MainWindow::menuNewProject() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    const auto answer = QMessageBox::warning(this, tr("New Project"), tr("Clear all ACMX interface settings and start a new project?\n\nThis keeps existing project files and source media, but clears the current project configuration and shader list."), QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer != QMessageBox::Yes)
        return;

    if (effectPackBrowser)
        effectPackBrowser->clear_project_pack();
    publishEffectPackToRunningProcess({}, {}, regular_deep_dream_configuration());

    QSettings interfaceSettings("LostSideDead", "acmx2");
    QSettings applicationSettings("LostSideDead");
    interfaceSettings.clear();
    applicationSettings.clear();
    applicationSettings.setValue("interface/backend", acmx2::backend_id(acmx2::Backend::Acmxvk));
    interfaceSettings.sync();
    applicationSettings.sync();

    active_backend = acmx2::Backend::Acmxvk;
    if (backendAcmxvkAction)
        backendAcmxvkAction->setChecked(true);
    executable_path = acmx2::default_backend_executable(active_backend);
    shader_path.clear();
    activeShaderManifestPath.clear();
    indexTimestamp = QDateTime();
    items.clear();
    if (list_view)
        list_view->clear();

    project_output_directory.clear();
    project_output_filename.clear();
    set_current_project_path(QString());
    prefix_path = QStringLiteral(".");
    output_file.clear();
    save_output_log = false;
    video_file.clear();
    graphics_file.clear();
    loadSessionSettings();
    updateRecentLibrariesMenu();
    updateRecentPresetsMenu();
    update_backend_ui();
    publishRuntimeSettingsToRunningProcess();
    publishSelectedShaderIndexToRunningProcess();
    Log(tr("Started a new project. Load a shader library to continue."));
}

void MainWindow::menuImportPreset() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    const QString path = QFileDialog::getOpenFileName(this, tr("Load Project"), QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation), tr("ACMX Project (*.acmxproj);;Legacy ACMX Project (*.json)"));
    if (!path.isEmpty())
        importPreset(path);
}

void MainWindow::fileOpenProp() {
    const acmx2::Backend propertiesBackend = active_backend;
    PropWindow propWindow(propertiesBackend, this);
    if (propWindow.exec() == QDialog::Accepted) {
        QString exePath = propWindow.exePathLineEdit->text();
        QString shaderDir = propWindow.shaderDirLineEdit->text();
        QString prefix = propWindow.screenshotDirLineEdit->text();
        QString compilerMode;
        QString compilerPath;
        if (propertiesBackend == acmx2::Backend::Acmxvk && propWindow.shaderCompilerComboBox) {
            compilerMode = propWindow.shaderCompilerComboBox->currentData().toString();
            compilerPath = propWindow.shaderCompilerPathLineEdit->text().trimmed();
            if (compilerMode == QStringLiteral("custom") && compilerPath.isEmpty()) {
                QMessageBox::information(this, tr("Shader Compiler"), tr("Select a custom glslc-compatible compiler path."));
                return;
            }
        }
        if (exePath.length() == 0) {
            QMessageBox::information(this, "No Path", "Requires Executable path");
            return;
        }
        if (shaderDir.length() == 0) {
            QMessageBox::information(this, "Shader Path", "Requires Shader Path");
            return;
        }

        if (!loadLibraryPath(shaderDir))
            return;

        QSettings appSettings("LostSideDead");
        if (active_backend == propertiesBackend) {
            appSettings.setValue(acmx2::backend_settings_key(active_backend, "executable"), exePath);
            if (active_backend == acmx2::Backend::Acmx2)
                appSettings.setValue("exePath", exePath);
            executable_path = exePath;
            if (propertiesBackend == acmx2::Backend::Acmxvk) {
                appSettings.setValue(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "shader_compiler_mode"), compilerMode.isEmpty() ? QStringLiteral("auto") : compilerMode);
                appSettings.setValue(acmx2::backend_settings_key(acmx2::Backend::Acmxvk, "shader_compiler_path"), compilerPath);
            }
        } else {
            Log(tr("Backend changed while loading the library; retained the "
                   "%1 executable setting.")
                    .arg(acmx2::backend_name(active_backend)));
        }
        appSettings.setValue("prefix_path", prefix);
        appSettings.sync();

        prefix_path = prefix;

        Log("Executable Path: " + executable_path);
        Log("Prefix Path: " + prefix);
        Log("Shader Directory: " + shaderDir);

    } else {
        Log("Canceled");
    }
}

void MainWindow::menuCustomUniforms() {
    if (!customUniformDialog || shader_path.isEmpty()) {
        QMessageBox::information(this, tr("Custom Uniforms"), tr("Load a shader library first."));
        return;
    }
    const QString jsonPath = QDir(shader_path).filePath("library.json");
    if (!QFileInfo(jsonPath).isFile()) {
        QMessageBox::warning(this, tr("Custom Uniforms"), tr("Custom uniforms require a library.json manifest."));
        return;
    }

    QString error;
    if (!customUniformDialog->loadLibrary(shader_path, active_backend, &error)) {
        QMessageBox::warning(this, tr("Could Not Load Custom Uniforms"), error);
        return;
    }
    customUniformDialog->show();
    customUniformDialog->raise();
    customUniformDialog->activateWindow();
}

void MainWindow::menuUniformReference() {
    if (!uniformReferenceDialog) {
        uniformReferenceDialog = new UniformReferenceDialog(active_backend, this);
        uniformReferenceDialog->setAttribute(Qt::WA_DeleteOnClose);
    } else
        uniformReferenceDialog->setBackend(active_backend);
    uniformReferenceDialog->show();
    uniformReferenceDialog->raise();
    uniformReferenceDialog->activateWindow();
}

bool MainWindow::loadShaders(const QString &path, bool force) {
    QString manifestPath = acmx2::shader_manifest_path(path);
    if (manifestPath.isEmpty()) {
        QMessageBox::warning(this, "Could not open shader manifest", "No library.json or index.txt found in: " + path);
        return false;
    }

    if (active_backend == acmx2::Backend::Acmx2 && QFileInfo(manifestPath).fileName().compare("index.txt", Qt::CaseInsensitive) == 0) {
        bool generated = false;
        QString migrationError;
        if (!acmx2::migrate_index_manifest_to_json(path, generated, migrationError)) {
            Log("Could not generate library.json from index.txt: " + migrationError);
        } else if (generated) {
            manifestPath = acmx2::shader_manifest_path(path);
            Log("Generated library.json from index.txt");
        }
    }

    QDateTime modified = QFileInfo(manifestPath).lastModified();
    if (!force && path == shader_path && manifestPath == activeShaderManifestPath && !indexTimestamp.isNull() && modified <= indexTimestamp) {
        return true;
    }
    QStringList manifestEntries;
    QString manifestError;
    if (!acmx2::load_shader_manifest(path, manifestEntries, manifestError)) {
        QMessageBox::warning(this, "Could not open shader manifest", manifestError);
        return false;
    }

    shader_path = path;
    activeShaderManifestPath = manifestPath;
    indexTimestamp = modified;
    if (customUniformDialog && QFileInfo(manifestPath).fileName().compare("library.json", Qt::CaseInsensitive) == 0) {
        QString uniformError;
        if (!customUniformDialog->loadLibrary(path, active_backend, &uniformError))
            Log("Could not load custom uniforms: " + uniformError);
    }
    updateOpenEditorShaderContexts();
    const int previousRow = currentShaderRow();
    const QString previouslySelected = currentShaderName();
    items.clear();
    QStringList uniqueItems;
    QSet<QString> uniqueKeys;
    if (QProgressDialog *progress = shader_library_progress_dialog.data()) {
        progress->setRange(0, manifestEntries.size());
        progress->setLabelText(tr("Validating shader library..."));
    }
    for (int index = 0; index < manifestEntries.size(); ++index) {
        if (index % 32 == 0) {
            if (QProgressDialog *progress = shader_library_progress_dialog.data())
                progress->setValue(index);
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
        const QString &rawEntry = manifestEntries.at(index);
        const QString line = rawEntry.trimmed();

        if (line.isEmpty()) {
            continue;
        }
        const QString shaderEntry = sanitizeShaderName(line);
        if (shaderEntry.isEmpty()) {
            Log("Skipping invalid shader path in " + QFileInfo(manifestPath).fileName() + ": " + line);
            continue;
        }
        QString fullPath = path + "/" + shaderEntry;
        QFileInfo fileInfo(fullPath);
        if (!fileInfo.exists() || !fileInfo.isFile()) {
            Log("Skipping non-existent file: " + shaderEntry);
            continue;
        }
        const QString uniqueKey = shaderEntry.toCaseFolded();
        if (!uniqueKeys.contains(uniqueKey)) {
            uniqueKeys.insert(uniqueKey);
            uniqueItems.append(shaderEntry);
        } else {
            Log("Skipping duplicate shader: " + shaderEntry);
        }
    }
    items = uniqueItems;

    Log("Loaded " + QString::number(items.size()) + " unique shader files");
    items.sort(Qt::CaseInsensitive);
    populateShaderTree();
    Log("Shaders sorted alphabetically");

    if (!items.isEmpty()) {
        int restoredRow = previousRow;
        if (restoredRow < 0 || restoredRow >= items.size()) {
            if (!previouslySelected.isEmpty() && items.contains(previouslySelected)) {
                restoredRow = items.indexOf(previouslySelected);
            } else {
                restoredRow = 0;
            }
        }
        selectShaderRow(restoredRow);
    }

    return true;
}

void MainWindow::fileExit() { QApplication::quit(); }

void MainWindow::menuAudioSettings() {
    if (!audio_available) {
        QMessageBox::information(this, tr("Audio Settings"), tr("Audio support is unavailable: acmx2 was built without audio support."));
        return;
    }
    const QString previousAudioFile = audio_file;
    const int previousAudioOutput = audio_output;
    const bool previousAudioPassThrough = audio_passthrough;
    const bool previousAudioTrunc = audio_trunc;
    const bool previousAudioRepeat = audio_repeat;
    AudioSettings audio_set(this);
    if (audio_set.exec() == QDialog::Accepted) {
        audio_enabled = audio_set.isAudioReactivityEnabled();
        audio_channels = audio_set.getNumberOfChannels();
        audio_sense = audio_set.getSensitivity();
        audio_passthrough = audio_set.isAudioPassThroughEnabled();
        record_audio = audio_set.isRecordAudioEnabled();
        record_volume = audio_set.getRecordVolume();
        audio_input = audio_set.getInputDeviceIndex();
        audio_output = audio_set.getOutputDeviceIndex();
        if (audio_set.isAudioFileEnabled()) {
            audio_file = audio_set.getAudioFilePath();
        } else {
            audio_file = "";
        }
        audio_trunc = audio_set.isAudioTruncEnabled();
        audio_repeat = audio_set.isAudioRepeatEnabled();
        audio_buffers_enabled = audio_set.isAudioBuffersEnabled();
        audio_buffer_frames = audio_set.getAudioBufferFrames();
        audio_warm_rate = audio_set.getAudioWarmRate();
        Log("Audio Settings Saved");
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
        const bool liveAudioSettingsChanged = QFileInfo(audio_file).absoluteFilePath() != QFileInfo(previousAudioFile).absoluteFilePath() || audio_output != previousAudioOutput || audio_passthrough != previousAudioPassThrough || audio_trunc != previousAudioTrunc || audio_repeat != previousAudioRepeat;
        if (shaderSelectionShm && process && process->state() == QProcess::Running && !audio_file.isEmpty() && liveAudioSettingsChanged) {
            const QByteArray path = QFileInfo(audio_file).absoluteFilePath().toUtf8();
            if (path.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxAudioFilePath)) {
                Log("Audio file path is too long for live playback: " + audio_file);
            } else {
                acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
                if (!lock) {
                    Log("Could not lock the live playback control channel");
                    return;
                }
                std::fill(std::begin(shaderSelectionShm->audio_file_path), std::end(shaderSelectionShm->audio_file_path), '\0');
                std::copy(path.cbegin(), path.cend(), shaderSelectionShm->audio_file_path);
                shaderSelectionShm->audio_output_device = audio_output;
                shaderSelectionShm->audio_pass_through = audio_passthrough ? 1 : 0;
                shaderSelectionShm->audio_trunc = audio_trunc ? 1 : 0;
                shaderSelectionShm->audio_repeat = audio_repeat ? 1 : 0;
                ++shaderSelectionShm->audio_file_sequence;
                ++shaderSelectionShm->sequence;
                Log("Requested live audio-file change: " + audio_file + "<br>");
            }
        }
#endif
    }
}

void MainWindow::menuGPUFilterSettings() {
    if (!cuda_available) {
        QMessageBox::information(this, tr("GPU Filter Settings"), tr("GPU filters are unavailable: acmx2 was built without CUDA support."));
        return;
    }

    if (gpuFilterDialog) {
        gpuFilterDialog->show();
        gpuFilterDialog->raise();
        gpuFilterDialog->activateWindow();
        return;
    }

    gpuFilterDialog = new GPUFilterDialog(executable_path, this);
    gpuFilterDialog->setAttribute(Qt::WA_DeleteOnClose);
    GPUFilterDialog *dialog = gpuFilterDialog;

    auto applyGpuDialogSettings = [this](bool enabled, const QString &filters, int bufferSize) {
        gpu_filter_enabled = enabled;
        gpu_filter_indices = filters;
        gpu_buffer_size = bufferSize;
        if ((!gpu_filter_enabled || gpu_filter_indices.isEmpty()) && deep_dream_gpu_filter_first) {
            deep_dream_gpu_filter_first = false;
            QSettings("LostSideDead", "acmx2").setValue("deep_dream/gpu_filter_first", false);
            Log("Deep Dream pipeline order reset because GPU filtering was disabled");
        }
        if (gpu_filter_enabled) {
            Log("GPU Filter Settings Saved: Filters=" + gpu_filter_indices + ", Buffer=" + QString::number(gpu_buffer_size));
        } else {
            Log("GPU Filtering Disabled");
        }
        publishRuntimeSettingsToRunningProcess();
    };

    connect(dialog, &GPUFilterDialog::settingsApplied, this, [applyGpuDialogSettings](bool enabled, const QString &filterArgument, int bufferSize) { applyGpuDialogSettings(enabled, filterArgument, bufferSize); });
    connect(dialog, &QDialog::accepted, this, [dialog, applyGpuDialogSettings]() { applyGpuDialogSettings(dialog->isGPUFilterEnabled(), dialog->getFilterArgument(), dialog->getBufferSize()); });

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

void MainWindow::menuDeepDreamSettings() {
    if (active_backend != acmx2::Backend::Acmxvk || !deep_dream_available) {
        QMessageBox::information(this,
                                 tr("Deep Dream Settings"),
                                 tr("Deep Dream is unavailable: ACMXVK must be built with "
                                    "-DWITH_DEEP_DREAM=ON and CUDA-enabled LibTorch."));
        return;
    }

    if (deepDreamSettingsDialog) {
        deepDreamSettingsDialog->show();
        deepDreamSettingsDialog->raise();
        deepDreamSettingsDialog->activateWindow();
        return;
    }

    const bool gpu_filter_configured = cuda_available && gpu_filter_enabled && !gpu_filter_indices.trimmed().isEmpty();
    deepDreamSettingsDialog = new DeepDreamSettingsDialog(gpu_filter_configured, this);
    deepDreamSettingsDialog->setAttribute(Qt::WA_DeleteOnClose);
    DeepDreamSettingsDialog *dialog = deepDreamSettingsDialog;
    connect(dialog, &DeepDreamSettingsDialog::settingsApplied, this, [this, dialog]() {
        const DeepDreamConfiguration config = dialog->configuration();
        deep_dream_enabled = config.enabled;
        deep_dream_model = config.model_file;
        deep_dream_layer = config.layer;
        deep_dream_iterations = config.iterations;
        deep_dream_strength = config.strength;
        deep_dream_feedback = config.feedback;
        deep_dream_zoom = config.zoom;
        deep_dream_rotation = config.rotation;
        deep_dream_maximum_dimension = config.maximum_dimension;
        deep_dream_fp16 = config.fp16;
        deep_dream_channel = config.channel;
        deep_dream_octaves = config.octaves;
        deep_dream_octave_scale = config.octave_scale;
        deep_dream_jitter = config.jitter;
        deep_dream_smoothing = config.smoothing;
        deep_dream_gpu_filter_first = config.gpu_filter_first;
        deep_dream_original = config.deep_original;
        publishRuntimeSettingsToRunningProcess();

        if (deep_dream_enabled) {
            Log(tr("Deep Dream Settings Applied: %1/%2, %3 "
                   "iteration(s), rotation %4 degrees, %5, %6")
                    .arg(QFileInfo(deep_dream_model).fileName(), deep_dream_layer)
                    .arg(deep_dream_iterations)
                    .arg(deep_dream_rotation, 0, 'f', 3)
                    .arg(deep_dream_gpu_filter_first ? tr("acidcam-gpu first") : tr("Deep Dream first"))
                    .arg(deep_dream_original ? tr("independent-frame preview") : tr("temporal feedback")));
        } else {
            Log("Deep Dream Disabled");
        }
    });
    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

bool MainWindow::validateDeepDreamLaunch(QString &error) const {
    error.clear();
    if (!deep_dream_enabled || active_backend != acmx2::Backend::Acmxvk) {
        return true;
    }
    if (!deep_dream_available) {
        error = tr("The selected ACMXVK executable does not provide Deep "
                   "Dream support.");
        return false;
    }
    if (!QFileInfo(deep_dream_model).isFile()) {
        error = tr("The configured Deep Dream model does not exist:\n%1").arg(deep_dream_model);
        return false;
    }
    if (deep_dream_layer.trimmed().isEmpty()) {
        error = tr("Select a Deep Dream feature layer.");
        return false;
    }
    if (deep_dream_original && (video_file.isEmpty() || output_file.isEmpty())) {
        error = tr("Independent-frame preview requires video input and an "
                   "enabled video output file.");
        return false;
    }
    if (!deep_dream_gpu_filter_first) {
        return true;
    }
    if (!cuda_available || !gpu_filter_enabled || gpu_filter_indices.trimmed().isEmpty()) {
        error = tr("Running acidcam-gpu before Deep Dream requires an enabled "
                   "GPU filter chain.");
        return false;
    }
    if (!graphics_file.isEmpty()) {
        error = tr("Running acidcam-gpu before Deep Dream currently supports "
                   "camera and video input, not still graphics.");
        return false;
    }
    if (maximize_fps) {
        error = tr("Running acidcam-gpu before Deep Dream cannot be combined "
                   "with Maximize FPS.");
        return false;
    }
    if (onnx_model_enabled && !onnx_model.isEmpty()) {
        error = tr("Running acidcam-gpu before Deep Dream cannot be combined "
                   "with an ONNX input effect.");
        return false;
    }
    return true;
}

void MainWindow::appendDeepDreamArguments(QStringList &arguments) const {
    if (!deep_dream_enabled || active_backend != acmx2::Backend::Acmxvk) {
        return;
    }

    arguments << "--dream-model" << deep_dream_model;
    arguments << "--dream-layer" << deep_dream_layer;
    arguments << "--dream-iterations" << QString::number(deep_dream_iterations);
    arguments << "--dream-strength" << QString::number(deep_dream_strength, 'g', 12);
    arguments << "--dream-feedback" << QString::number(deep_dream_feedback, 'g', 12);
    arguments << "--dream-zoom" << QString::number(deep_dream_zoom, 'g', 12);
    arguments << "--dream-rotation" << QString::number(deep_dream_rotation, 'g', 12);
    arguments << "--dream-size" << QString::number(deep_dream_maximum_dimension);
    if (deep_dream_fp16) {
        arguments << "--dream-fp16";
    }
    arguments << "--dream-channel" << (deep_dream_channel < 0 ? QString("all") : QString::number(deep_dream_channel));
    arguments << "--dream-octaves" << QString::number(deep_dream_octaves);
    arguments << "--dream-octave-scale" << QString::number(deep_dream_octave_scale, 'g', 12);
    arguments << "--dream-jitter" << QString::number(deep_dream_jitter);
    arguments << "--dream-smoothing" << QString::number(deep_dream_smoothing);
    if (deep_dream_gpu_filter_first) {
        arguments << "--gpu-filter-before-dream";
    }
    if (deep_dream_original) {
        arguments << "--deep-orig";
    }
}

void MainWindow::menuStableDiffusionSettings() {
    if (active_backend != acmx2::Backend::Acmxvk || !stable_diffusion_available) {
        QMessageBox::information(this,
                                 tr("Stable Diffusion Settings"),
                                 tr("Stable Diffusion is unavailable: ACMXVK must be built with "
                                    "-DWITH_STABLE_DIFFUSION=ON."));
        return;
    }

    if (stableDiffusionSettingsDialog) {
        stableDiffusionSettingsDialog->show();
        stableDiffusionSettingsDialog->raise();
        stableDiffusionSettingsDialog->activateWindow();
        return;
    }

    stableDiffusionSettingsDialog = new StableDiffusionSettingsDialog(this);
    stableDiffusionSettingsDialog->setAttribute(Qt::WA_DeleteOnClose);
    StableDiffusionSettingsDialog *dialog = stableDiffusionSettingsDialog;
    connect(dialog, &StableDiffusionSettingsDialog::settingsApplied, this, [this, dialog]() {
        const StableDiffusionConfiguration config = dialog->configuration();
        stable_diffusion_enabled = config.enabled;
        stable_diffusion_model = config.model_file;
        stable_diffusion_upscale_model = config.upscale_model_file;
        stable_diffusion_lora_files = config.lora_files;
        stable_diffusion_lora_multipliers = config.lora_multipliers;
        stable_diffusion_prompt = config.prompt;
        stable_diffusion_negative_prompt = config.negative_prompt;
        stable_diffusion_server = config.server_executable;
        stable_diffusion_server_arguments = config.server_arguments;
        stable_diffusion_server_port = config.server_port;
        stable_diffusion_width = config.width;
        stable_diffusion_height = config.height;
        stable_diffusion_upscale_working_width = config.upscale_working_width;
        stable_diffusion_upscale_working_height = config.upscale_working_height;
        stable_diffusion_steps = config.steps;
        stable_diffusion_strength = config.strength;
        stable_diffusion_cfg_scale = config.cfg_scale;
        stable_diffusion_seed = config.seed;
        stable_diffusion_sampler = config.sampler;
        stable_diffusion_scheduler = config.scheduler;
        stable_diffusion_upscale = config.upscale;
        stable_diffusion_upscale_only = config.upscale_only;

        if (stable_diffusion_enabled) {
            Log(tr("Stable Diffusion Settings Applied: %1, %2x%3, "
                   "%4 step(s)%5%6; changes apply on the next launch")
                    .arg(stable_diffusion_upscale_only ? tr("ESRGAN upscale only") : QFileInfo(stable_diffusion_model).fileName())
                    .arg(stable_diffusion_width)
                    .arg(stable_diffusion_height)
                    .arg(stable_diffusion_steps)
                    .arg(stable_diffusion_upscale_only               ? tr(", before shader chain")
                         : stable_diffusion_upscale                  ? tr(", compute upscale")
                         : !stable_diffusion_upscale_model.isEmpty() ? tr(", ESRGAN upscale")
                                                                     : QString())
                    .arg(stable_diffusion_upscale_only || stable_diffusion_lora_files.isEmpty() ? QString() : tr(", %1 LoRA(s)").arg(stable_diffusion_lora_files.size())));
        } else {
            Log("Stable Diffusion Disabled");
        }
    });
    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

bool MainWindow::validateStableDiffusionLaunch(QString &error) const {
    error.clear();
    if (!stable_diffusion_enabled || active_backend != acmx2::Backend::Acmxvk) {
        return true;
    }
    if (!stable_diffusion_available) {
        error = tr("The selected ACMXVK executable does not provide Stable "
                   "Diffusion support.");
        return false;
    }
    if (video_file.trimmed().isEmpty() || !QFileInfo(video_file).isFile()) {
        error = tr("Stable Diffusion requires an existing video input file.");
        return false;
    }
    if (!graphics_file.trimmed().isEmpty()) {
        error = tr("Stable Diffusion cannot be used with still-image input.");
        return false;
    }
    if (!stable_diffusion_upscale_only && !QFileInfo(stable_diffusion_model).isFile()) {
        error = tr("The configured Stable Diffusion model does not exist:\n%1").arg(stable_diffusion_model);
        return false;
    }
    if (!stable_diffusion_upscale_only && stable_diffusion_lora_files.size() != stable_diffusion_lora_multipliers.size()) {
        error = tr("The saved Stable Diffusion LoRA settings are incomplete. "
                   "Open Stable Diffusion Settings and apply them again.");
        return false;
    }
    QString lora_directory;
    for (int index = 0; !stable_diffusion_upscale_only && index < stable_diffusion_lora_files.size(); ++index) {
        const QString filename = stable_diffusion_lora_files.at(index);
        const QFileInfo file_info(filename);
        if (!file_info.isFile()) {
            error = tr("The configured LoRA model does not exist:\n%1").arg(filename);
            return false;
        }
        if (lora_directory.isEmpty()) {
            lora_directory = file_info.absolutePath();
        } else if (lora_directory != file_info.absolutePath()) {
            error = tr("All configured LoRA models must be in the same folder. "
                       "sd-server scans one LoRA model directory per launch.");
            return false;
        }
        const double multiplier = stable_diffusion_lora_multipliers.at(index);
        if (multiplier < -10.0 || multiplier > 10.0) {
            error = tr("LoRA multipliers must be between -10 and 10.");
            return false;
        }
    }
    if (!stable_diffusion_upscale_model.isEmpty() && !QFileInfo(stable_diffusion_upscale_model).isFile()) {
        error = tr("The configured Stable Diffusion upscale model does not "
                   "exist:\n%1")
                    .arg(stable_diffusion_upscale_model);
        return false;
    }
    if (stable_diffusion_upscale_only && stable_diffusion_upscale_model.isEmpty()) {
        error = tr("ESRGAN upscale-only mode requires an existing upscale model.");
        return false;
    }
    if (!stable_diffusion_upscale_only && stable_diffusion_prompt.trimmed().isEmpty()) {
        error = tr("Enter a Stable Diffusion image-to-image prompt.");
        return false;
    }
    if (stable_diffusion_server.trimmed().isEmpty()) {
        error = tr("Enter the sd-server executable name or path.");
        return false;
    }
    if (!stable_diffusion_upscale_only && ((stable_diffusion_width % 64) != 0 || (stable_diffusion_height % 64) != 0)) {
        error = tr("Stable Diffusion dimensions must be multiples of 64.");
        return false;
    }
    if (encode_fill_pts_gaps) {
        error = tr("Stable Diffusion cannot be combined with Fill PTS Gaps.");
        return false;
    }
    return true;
}

void MainWindow::appendStableDiffusionArguments(QStringList &arguments) const {
    if (!stable_diffusion_enabled || active_backend != acmx2::Backend::Acmxvk) {
        return;
    }
    if (stable_diffusion_upscale_only) {
        arguments << "--sd-upscale-only";
    } else {
        arguments << "--sd-model" << stable_diffusion_model;
        arguments << "--sd-prompt" << stable_diffusion_prompt;
        if (!stable_diffusion_negative_prompt.trimmed().isEmpty()) {
            arguments << "--sd-negative-prompt" << stable_diffusion_negative_prompt;
        }
        for (int index = 0; index < stable_diffusion_lora_files.size(); ++index) {
            arguments << "--sd-lora" << stable_diffusion_lora_files.at(index);
            arguments << "--sd-lora-strength" << QString::number(stable_diffusion_lora_multipliers.at(index), 'g', 12);
        }
    }
    arguments << "--sd-server" << stable_diffusion_server;
    const QStringList server_arguments = QProcess::splitCommand(stable_diffusion_server_arguments);
    for (const QString &argument : server_arguments) {
        arguments << "--sd-server-arg" << argument;
    }
    arguments << "--sd-server-port" << QString::number(stable_diffusion_server_port);
    if (stable_diffusion_upscale_working_width > 0 && stable_diffusion_upscale_working_height > 0) {
        arguments << "--sd-upscale-size" << QString("%1x%2").arg(stable_diffusion_upscale_working_width).arg(stable_diffusion_upscale_working_height);
    }
    if (!stable_diffusion_upscale_only) {
        arguments << "--sd-size" << QString("%1x%2").arg(stable_diffusion_width).arg(stable_diffusion_height);
        arguments << "--sd-steps" << QString::number(stable_diffusion_steps);
        arguments << "--sd-strength" << QString::number(stable_diffusion_strength, 'g', 12);
        arguments << "--sd-cfg-scale" << QString::number(stable_diffusion_cfg_scale, 'g', 12);
        arguments << "--sd-seed" << QString::number(stable_diffusion_seed);
        arguments << "--sd-sampler" << stable_diffusion_sampler;
        arguments << "--sd-scheduler" << stable_diffusion_scheduler;
    }
    arguments << "--sd-quiet";
    if (stable_diffusion_upscale) {
        arguments << "--sd-upscale";
    } else if (!stable_diffusion_upscale_model.isEmpty()) {
        arguments << "--upscale-model" << stable_diffusion_upscale_model;
    }
}

void MainWindow::menuMidiSettings() {
    if (!midi_available) {
        QMessageBox::information(this, tr("MIDI Settings"), tr("MIDI support is unavailable: acmx2 was built without MIDI support."));
        return;
    }
    MidiSettings midiDialog(executable_path, this);
    if (midiDialog.exec() == QDialog::Accepted) {
        midi_enabled = midiDialog.isMidiEnabled();
        midi_config_file = midiDialog.getMidiConfigFile();
        midi_device = midiDialog.getMidiDeviceIndex();
        QSettings appSettings("LostSideDead");
        appSettings.setValue("midiEnabled", midi_enabled);
        appSettings.setValue("midiConfigFile", midi_config_file);
        appSettings.setValue("midiDevice", midi_device);
        if (midi_enabled) {
            Log("MIDI Settings Saved: Config=" + midi_config_file + ", Device=" + QString::number(midi_device));
        } else {
            Log("MIDI Disabled");
        }
    }
}

void MainWindow::menuToggleDisplayFilter(bool checked) {
    display_filter_enabled = checked;
    QSettings appSettings("LostSideDead");
    appSettings.setValue("displayFilter", display_filter_enabled);
    Log(QString("Display Filter Overlay: %1").arg(display_filter_enabled ? "Enabled" : "Disabled"));
    publishRuntimeSettingsToRunningProcess();
}

void MainWindow::menuWatermarkSettings() {
    QDialog dlg(this);
    dlg.setWindowTitle(tr("Watermark Settings"));
    acmx2::applyCustomStyleIfEnabled(&dlg);

    auto *enableCheck = new QCheckBox(tr("Enable watermark in recorded video"), &dlg);
    enableCheck->setChecked(watermark_enabled);

    auto *textEdit = new QLineEdit(watermark_text, &dlg);
    textEdit->setPlaceholderText(tr("Watermark text (shown upper-left of recorded video)"));

    auto *colorPreview = new QLabel(&dlg);
    colorPreview->setAutoFillBackground(true);
    colorPreview->setMinimumSize(80, 24);
    colorPreview->setFrameStyle(QFrame::Box | QFrame::Plain);
    colorPreview->setAlignment(Qt::AlignCenter);

    int curR = watermark_r, curG = watermark_g, curB = watermark_b;
    auto applyPreview = [colorPreview, &curR, &curG, &curB]() {
        QPalette pal = colorPreview->palette();
        pal.setColor(QPalette::Window, QColor(curR, curG, curB));
        QColor fg = (curR * 0.299 + curG * 0.587 + curB * 0.114) > 140 ? Qt::black : Qt::white;
        pal.setColor(QPalette::WindowText, fg);
        colorPreview->setPalette(pal);
        colorPreview->setText(QString(" %1, %2, %3 ").arg(curR).arg(curG).arg(curB));
    };
    applyPreview();

    auto *colorBtn = new QPushButton(tr("Choose Color..."), &dlg);
    QObject::connect(colorBtn, &QPushButton::clicked, &dlg, [&]() {
        QColor chosen = QColorDialog::getColor(QColor(curR, curG, curB), &dlg, tr("Watermark Color"));
        if (chosen.isValid()) {
            curR = chosen.red();
            curG = chosen.green();
            curB = chosen.blue();
            applyPreview();
        }
    });

    auto *form = new QFormLayout();
    form->addRow(enableCheck);
    form->addRow(tr("Text:"), textEdit);
    auto *colorRow = new QHBoxLayout();
    colorRow->addWidget(colorPreview, 1);
    colorRow->addWidget(colorBtn);
    form->addRow(tr("Color:"), colorRow);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);

    auto *layout = new QVBoxLayout(&dlg);
    layout->addLayout(form);
    layout->addWidget(buttons);

    if (dlg.exec() != QDialog::Accepted) {
        return;
    }

    watermark_enabled = enableCheck->isChecked();
    watermark_text = textEdit->text();
    watermark_r = curR;
    watermark_g = curG;
    watermark_b = curB;

    QSettings appSettings("LostSideDead");
    appSettings.setValue("watermarkEnabled", watermark_enabled);
    appSettings.setValue("watermarkText", watermark_text);
    appSettings.setValue("watermarkR", watermark_r);
    appSettings.setValue("watermarkG", watermark_g);
    appSettings.setValue("watermarkB", watermark_b);

    Log(QString("Watermark %1: \"%2\" color=%3,%4,%5").arg(watermark_enabled ? "Enabled" : "Disabled").arg(watermark_text).arg(watermark_r).arg(watermark_g).arg(watermark_b));
    publishRuntimeSettingsToRunningProcess();
}

void MainWindow::menuShaderPassSettings() {
    if (shader_path.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First", "Please load a shader library before configuring multi-pass shaders.");
        return;
    }

    if (items.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First", "Please load a shader library before configuring multi-pass shaders.");
        return;
    }

    if (shaderPassDialog) {
        shaderPassDialog->updateShaderList(items);
        shaderPassDialog->show();
        shaderPassDialog->raise();
        shaderPassDialog->activateWindow();
        return;
    }

    shaderPassDialog = new ShaderPassDialog(items, this);
    shaderPassDialog->setAttribute(Qt::WA_DeleteOnClose);
    shaderPassDialog->setEnabled(shader_pass_enabled);
    if (!shader_pass_names.isEmpty()) {
        shaderPassDialog->setSelectedShaderNames(shader_pass_names);
    }

    ShaderPassDialog *dialog = shaderPassDialog;
    auto applyMultipassSettings = [this, dialog]() {
        shader_pass_enabled = dialog->isShaderPassEnabled();
        shader_pass_names = dialog->getSelectedShaderNames();
        publishMultipassShadersToRunningProcess();
        if (shader_pass_enabled) {
            Log("Multi-Pass Shader Settings Saved: " + QString::number(shader_pass_names.size()) + " passes");
        } else {
            Log("Multi-Pass Shader Disabled");
        }
    };

    connect(dialog, &ShaderPassDialog::settingsApplied, this, [this](bool enabled, const QStringList &selectedShaderNames) {
        shader_pass_enabled = enabled;
        shader_pass_names = selectedShaderNames;
        publishMultipassShadersToRunningProcess();
        if (shader_pass_enabled) {
            Log("Multi-Pass Shader Settings Saved: " + QString::number(shader_pass_names.size()) + " passes");
        } else {
            Log("Multi-Pass Shader Disabled");
        }
    });
    connect(dialog, &ShaderPassDialog::shaderEditRequested, this, [this](const QString &shaderName) {
        const QString safeName = sanitizeShaderName(shaderName);
        if (!safeName.isEmpty())
            openShaderEditor(QDir(shader_path).filePath(safeName));
    });
    connect(dialog, &QDialog::accepted, this, applyMultipassSettings);

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

void MainWindow::menuPlaylistSettings() {
    if (shader_path.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First", "Please load a shader library before configuring playlist.");
        return;
    }

    if (items.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First", "Please load a shader library before configuring playlist.");
        return;
    }

    if (playlistDialog) {
        playlistDialog->updateShaderList(items);
        playlistDialog->show();
        playlistDialog->raise();
        playlistDialog->activateWindow();
        return;
    }

    playlistDialog = new PlaylistDialog(items, active_backend, this);
    playlistDialog->setAttribute(Qt::WA_DeleteOnClose);
    playlistDialog->setEnabled(playlist_enabled);
    if (!playlist_tree_data.isEmpty()) {
        playlistDialog->setPlaylistTree(playlist_tree_data);
    } else if (!playlist_names.isEmpty()) {
        playlistDialog->setSelectedShaderNames(playlist_names);
    }
    if (!playlist_file_path.isEmpty()) {
        playlistDialog->setPlaylistFile(playlist_file_path);
    }
    playlistDialog->setAutopilotFrames(autopilot_frames);
    playlistDialog->setAutopilotRandom(autopilot_random);

    PlaylistDialog *dialog = playlistDialog;
    connect(dialog, &QDialog::accepted, this, [this, dialog]() {
        playlist_enabled = dialog->isPlaylistEnabled();
        playlist_names = dialog->getSelectedShaderNames();
        playlist_tree_data = dialog->getPlaylistTree();
        playlist_file_path = dialog->getPlaylistFile();
        autopilot_frames = dialog->getAutopilotFrames();
        autopilot_random = dialog->isAutopilotRandom();
        QSettings appSettings("LostSideDead");
        appSettings.setValue("playlistAutopilotFrames", autopilot_frames);
        appSettings.setValue("playlistAutopilotRandom", autopilot_random);
        if (playlist_enabled) {
            Log("Playlist Settings Saved: " + QString::number(playlist_names.size()) + " shaders");
            if (!playlist_file_path.isEmpty()) {
                Log("Playlist file: " + playlist_file_path);
            }
            if (autopilot_frames > 0) {
                Log(QString("Autopilot timeout mode: %1 (%2 frames)").arg(autopilot_random ? "random" : "fixed").arg(autopilot_frames));
            }
        } else {
            Log("Playlist Disabled");
        }
    });

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

void MainWindow::cameraSettings() {
    SettingsWindow settingsWindow(executable_path, active_backend, this);
    settingsWindow.setCudaAvailable(cuda_device_available);
    settingsWindow.setDnnAvailable(dnn_available);
    if (settingsWindow.exec() == QDialog::Accepted) {
        full_screen_value = settingsWindow.isFullscreen();
        enable_vsync = settingsWindow.isVsyncEnabled();
        monitor_index = settingsWindow.getMonitorIndex();
        if (settingsWindow.isUsingInputVideoFile()) {
            QString videoFile = settingsWindow.getInputVideoFile();
            QSize screenResolution = settingsWindow.getSelectedScreenResolution();
            screen_res = screenResolution;
            video_file = videoFile;
            graphics_file = "";
            cache_enabled = settingsWindow.isTextureCacheEnabled();
            cache_delay = settingsWindow.getCacheDelay();
            cache_size = settingsWindow.getCacheSize();
            copy_audio = settingsWindow.isCopyAudioEnabled();
        } else if (settingsWindow.isUsingGraphicsFile()) {
            QString graphicsFile = settingsWindow.getGraphicsFile();
            QSize screenResolution = settingsWindow.getSelectedScreenResolution();
            screen_res = screenResolution;
            graphics_file = graphicsFile;
            video_file = "";
            output_fps = settingsWindow.getCameraFPS();
            cache_enabled = false;
            cache_delay = 1;
            cache_size = 8;
            copy_audio = false;
        } else {
            int cameraIndex = settingsWindow.getSelectedCameraIndex();
            QSize cameraResolution = settingsWindow.getSelectedCameraResolution();
            QSize screenResolution = settingsWindow.getSelectedScreenResolution();
            screen_res = screenResolution;
            camera_index = cameraIndex;
            video_file = "";
            graphics_file = "";
            camera_res = cameraResolution;
            output_fps = settingsWindow.getCameraFPS();
            cache_enabled = settingsWindow.isTextureCacheEnabled();
            cache_delay = settingsWindow.getCacheDelay();
            cache_size = settingsWindow.getCacheSize();
            use_yuv = settingsWindow.isUseYuvEnabled();
        }
        if (settingsWindow.isSavingToOutputVideoFile()) {
            output_file = settingsWindow.getOutputVideoFile();
            if (!project_output_directory.isEmpty()) {
                QString output_filename = QFileInfo(output_file).fileName();
                if (output_filename.isEmpty())
                    output_filename = project_output_filename;
                if (output_filename.isEmpty())
                    output_filename = QStringLiteral("output.mp4");
                project_output_filename = output_filename;
                output_file = QDir(project_output_directory).filePath(output_filename);
            }
            save_output_log = settingsWindow.isSavingOutputLog();

            QSettings settings("LostSideDead", "acmx2");
            settings.setValue("interface/save_output", true);
            settings.setValue("interface/output_video", output_file);
            settings.setValue("interface/save_output_log", save_output_log);
            settings.sync();
        } else {
            output_file = "";
            save_output_log = false;
        }
        // Only meaningful in input-video mode + with an output file. The
        // settings dialog already gates this on HDR detection, but we re-check
        // here so it stays consistent if other modes are selected.
        convert_to_hdr10 = settingsWindow.isConvertToHdr10Enabled() && settingsWindow.isUsingInputVideoFile() && settingsWindow.isSavingToOutputVideoFile();
        maximize_fps = settingsWindow.isMaximizeFpsEnabled();
        use_source_fps = settingsWindow.isUseSourceFpsEnabled();
        use_source_audio = settingsWindow.isUseSourceAudioEnabled();
    }
    enable_3d = settingsWindow.is3dEnabled();
    model_file = settingsWindow.getModelFile();
    onnx_model_enabled = settingsWindow.isOnnxModelEnabled();
    onnx_model = settingsWindow.getOnnxModelFile();
    cuda_device = settingsWindow.getSelectedCudaDevice();
    time_speed = settingsWindow.getTimeSpeed();
    duration_limit_enabled = settingsWindow.isDurationLimitEnabled();
    max_duration = settingsWindow.getDurationLimit();
    max_size_limit_enabled = settingsWindow.isMaxSizeLimitEnabled();
    max_size_mb = settingsWindow.getMaxSizeLimit();
    cross_fade_duration = settingsWindow.getCrossFadeDuration();
    flip_enabled = settingsWindow.isFlipEnabled();
    rotate_enabled = settingsWindow.is_rotate_enabled();
    rotation_mode = settingsWindow.get_rotation_mode();
    png_output = active_backend == acmx2::Backend::Acmxvk && settingsWindow.isPngOutputEnabled();
    png_output_directory = settingsWindow.getPngOutputDirectory();
    png_level = settingsWindow.getPngLevel();
    generate_enabled = settingsWindow.isGenerateEnabled();
    generate_interval = settingsWindow.getGenerateInterval();
    encode_preset = settingsWindow.getEncodePreset();
    encode_tune = settingsWindow.getEncodeTune();
    encode_crf = settingsWindow.getEncodeCrf();
    encode_rate_control = settingsWindow.getEncodeRateControl();
    encode_bitrate = settingsWindow.getEncodeBitrate();
    encode_codec = settingsWindow.getEncodeCodec();
    encode_parameters = settingsWindow.getEncodeParameters();
    encode_realtime = settingsWindow.isEncodeRealtime();
    encode_no_drop = settingsWindow.isEncodeNoDrop();
    encode_constant_frame_rate = settingsWindow.isEncodeConstantFrameRate();
    encode_fill_pts_gaps = settingsWindow.isEncodeFillPtsGaps();
}

void MainWindow::runSelected() {
    if (process->state() == QProcess::Running) {
        QMessageBox::information(this, "Process Running", "A process is already running. Please stop it first.");
        return;
    }

    QString deep_dream_error;
    if (!validateDeepDreamLaunch(deep_dream_error)) {
        QMessageBox::warning(this, tr("Deep Dream Settings"), deep_dream_error);
        return;
    }
    QString stable_diffusion_error;
    if (!validateStableDiffusionLaunch(stable_diffusion_error)) {
        QMessageBox::warning(this, tr("Stable Diffusion Settings"), stable_diffusion_error);
        return;
    }

#ifdef __linux__
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    for (const QString &entry : defaultLinuxRunEnvAssignments()) {
        const int equals = entry.indexOf('=');
        if (equals > 0) {
            env.insert(entry.left(equals), entry.mid(equals + 1));
        }
    }
    process->setProcessEnvironment(env);
#endif

    if (shader_path.length() == 0) {
        QMessageBox::information(this, "Select Shaders", "Select Shader Path");
        return;
    }
    initShaderSelectionSharedMemory();
    publishSelectedShaderIndexToRunningProcess();
    publishMultipassShadersToRunningProcess();
    publishRuntimeSettingsToRunningProcess();
    const QString data = currentShaderName();
    if (data.isEmpty()) {
        Log("<b>No item selected.</b>");
        return;
    }
    QString launchShaderPath = shader_path;
    QString launchShaderName = data;
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString runtimeError;
        if (!resolve_acmxvk_runtime_library(shader_path, launchShaderPath, runtimeError)) {
            prompt_acmxvk_rebuild(runtimeError, PendingAcmxvkAction::RunSelected);
            return;
        }
        if (launchShaderPath != shader_path)
            launchShaderName = acmxvk_runtime_shader_name(data);
        if (!QFileInfo(QDir(launchShaderPath).filePath(launchShaderName)).isFile()) {
            QMessageBox::warning(this,
                                 tr("Build ACMXVK Library"),
                                 tr("The compiled shader is missing. Choose Playback > Build "
                                    "and try again.\n\n%1")
                                     .arg(QDir(launchShaderPath).filePath(launchShaderName)));
            return;
        }
    }
    QStringList arguments;
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/" + acmx2::default_backend_executable(active_backend);
#endif
    dirPath = resolve_backend_assets_path(active_backend, executable_path, shader_path);
    const int selectedIndex = currentShaderRow();
    if (selectedIndex < 0 || selectedIndex >= items.size()) {
        Log("<b>No valid shader selection.</b>");
        return;
    }
    if (active_backend == acmx2::Backend::Acmxvk)
        arguments << "--unbuffered";
    arguments << "--path" << dirPath;
    if (active_backend == acmx2::Backend::Acmxvk) {
        // ACMXVK needs its manifest to resolve fragment/compute types and
        // custom-uniform metadata for the selected SPIR-V shader.
        arguments << "--shaders" << launchShaderPath << "--shader-file" << launchShaderName << "--interface-shm";
    } else {
        // ACMX2 can compile a selected source directly without loading the
        // complete shader library and its binary cache.
        arguments << "--fragment" << (shader_path + "/" + data) << "--interface-shm";
    }
    // Pass texture cache size so the SIZE macro injected into the fragment
    // matches whatever the user has configured for cache shaders.
    arguments << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        arguments << "--texture-cache-array";
    const QSize effectiveCameraResolution = hasPositiveResolution(camera_res) ? camera_res : QSize(1280, 720);
    QString res;
    QTextStream stream(&res);
    stream << effectiveCameraResolution.width() << "x" << effectiveCameraResolution.height();

    QString scr_res;
    QTextStream stream_r(&scr_res);
    stream_r << screen_res.width() << "x" << screen_res.height();

    if (full_screen_value)
        arguments << "--fullscreen";
    if (active_backend == acmx2::Backend::Acmxvk && enable_vsync)
        arguments << "--enable-vsync";
    if (active_backend == acmx2::Backend::Acmxvk && monitor_index > 0)
        arguments << "--monitor" << QString::number(monitor_index);

    if (!graphics_file.isEmpty()) {
        arguments << "--graphic" << graphics_file;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--fps" << QString::number(output_fps);
    } else if (video_file.isEmpty()) {
        arguments << "--camera-res" << res;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--device" << QString::number(camera_index);
        arguments << "--fps" << QString::number(output_fps);
        if (active_backend == acmx2::Backend::Acmxvk && maximize_fps)
            arguments << "--maximize-fps";
        if (use_yuv)
            arguments << "--use-yuv";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
    } else {
        arguments << "--input" << video_file;
        if (active_backend == acmx2::Backend::Acmxvk && use_source_fps) {
            arguments << "--use-source-fps";
            if (use_source_audio)
                arguments << "--use-source-audio";
        }
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        if (play_repeat->isChecked())
            arguments << "--repeat";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
        if (copy_audio)
            arguments << "--copy-audio";
    }
    arguments << "--prefix" << prefix_path;
    if (active_backend == acmx2::Backend::Acmxvk) {
        arguments << "--png-level" << QString::number(png_level);
    }

    if (!output_file.isEmpty()) {
        const QString launch_output_file = active_backend == acmx2::Backend::Acmxvk ? timestamped_output_path(output_file) : output_file;
        arguments << "--output" << launch_output_file;
        if (active_backend == acmx2::Backend::Acmxvk && encode_rate_control == "bitrate")
            arguments << "--video-bitrate" << encode_bitrate;
        else
            arguments << "--encode-crf" << QString::number(encode_crf);
        if (!encode_preset.isEmpty())
            arguments << "--encode-preset" << encode_preset;
        if (!encode_tune.isEmpty())
            arguments << "--encode-tune" << encode_tune;
        if (!encode_codec.isEmpty() && encode_codec != "auto")
            arguments << "--encode-codec" << encode_codec;
        if (!encode_parameters.isEmpty())
            arguments << "--encode-params" << encode_parameters;
        if (encode_realtime)
            arguments << "--encode-realtime";
        if (encode_no_drop && (!video_file.isEmpty() || !graphics_file.isEmpty()))
            arguments << "--no-drop";
        if (active_backend == acmx2::Backend::Acmxvk && encode_constant_frame_rate && !video_file.isEmpty() && !png_output)
            arguments << "--constant-frame-rate";
        if (active_backend == acmx2::Backend::Acmxvk && encode_fill_pts_gaps && !png_output)
            arguments << "--fill-pts-gaps";
    }
    const bool sourceAudioActive = active_backend == acmx2::Backend::Acmxvk && !video_file.isEmpty() && use_source_fps && use_source_audio;
    if (audio_available && audio_enabled && !sourceAudioActive) {
        arguments << "--enable-audio";
        arguments << "--channels" << QString::number(audio_channels);

        if (audio_input == -1)
            arguments << "--audio-input" << "default";
        else
            arguments << "--audio-input" << QString::number(audio_input);

        if (record_audio) {
            QString wavPath;
            if (!output_file.isEmpty()) {
                QFileInfo fi(output_file);
                wavPath = fi.absolutePath() + "/" + fi.completeBaseName() + ".wav";
            } else {
                wavPath = prefix_path + "/recorded_audio.wav";
            }
            arguments << "--record-audio" << wavPath;
            arguments << "--record-gain" << QString::number(record_volume, 'f', 2);
        }
    }

    if (active_backend == acmx2::Backend::Acmxvk && !record_audio && (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--mute-output";
    }

    if (audio_available && (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--sense" << QString::number(audio_sense);
        if (audio_passthrough) {
            arguments << "--pass-through";
            if (audio_output == -1)
                arguments << "--audio-output" << "default";
            else
                arguments << "--audio-output" << QString::number(audio_output);
        }
    }

    if (audio_available && !audio_file.isEmpty() && !sourceAudioActive) {
        arguments << "--audio-file" << audio_file;
        if (audio_trunc) {
            arguments << "--audio-trunc";
        }
        if (audio_repeat) {
            arguments << "--audio-repeat";
        }
    }

    if (audio_available && audio_buffers_enabled) {
        arguments << "--enable-audio-buffers" << QString::number(audio_buffer_frames);
    }

    if (audio_available && (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--audio-warm-rate" << QString::number(audio_warm_rate, 'f', 2);
    }

    if (enable_3d) {
        arguments << "--enable-3d";
        arguments << "--model" << model_file;
    }

    if (onnx_model_enabled && !onnx_model.isEmpty()) {
        arguments << "--onnx" << onnx_model;
    }

    if (cuda_available && gpu_filter_enabled && !gpu_filter_indices.isEmpty()) {
        arguments << "--gpu-filter" << gpu_filter_indices;
        arguments << "--gpu-buffer" << QString::number(gpu_buffer_size);
    }

    appendDeepDreamArguments(arguments);
    appendStableDiffusionArguments(arguments);

    if (cuda_device_available) {
        arguments << "--cuda-device" << QString::number(cuda_device);
    }

    arguments << "--time-speed" << QString::number(static_cast<double>(time_speed), 'f', 2);
    if (normalized_time) {
        arguments << "--normalized";
    }

    if (!use_shader_cache && active_backend == acmx2::Backend::Acmx2) {
        arguments << "--no-cache";
    }

    if (midi_available && midi_enabled && !midi_config_file.isEmpty()) {
        arguments << "--midi-map" << midi_config_file;
        if (midi_device >= 0)
            arguments << "--midi-device" << QString::number(midi_device);
    }

    if ((!output_file.isEmpty() || (active_backend == acmx2::Backend::Acmxvk && png_output)) && duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (!output_file.isEmpty() && max_size_limit_enabled && max_size_mb > 0.0) {
        arguments << "--max-size" << QString::number(max_size_mb, 'f', 2);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    if (flip_enabled) {
        arguments << "--flip";
    }

    if (rotate_enabled) {
        arguments << "--rotate" << rotation_mode;
    }

    if (active_backend == acmx2::Backend::Acmxvk && png_output && !png_output_directory.isEmpty()) {
        arguments << "--png" << png_output_directory;
    }

    const int effective_generate_interval = generate_enabled ? generate_interval : (active_backend == acmx2::Backend::Acmxvk && png_output ? 1 : 0);
    if (effective_generate_interval > 0) {
        arguments << "--generate" << QString::number(effective_generate_interval);
    }

    if (watermark_enabled && !watermark_text.isEmpty()) {
        arguments << "--use-watermark" << watermark_text;
        arguments << "--use-watermark-color" << QString("%1,%2,%3").arg(watermark_r).arg(watermark_g).arg(watermark_b);
    }

    if (display_filter_enabled) {
        arguments << "--display-filter";
    }

    arguments.append(QProcess::splitCommand(extra_arguments));

    // ACMX2 single-source mode bypasses its binary cache. ACMXVK uses the
    // selected runtime library and does not accept ACMX2 cache controls.
    Log("shell: " + executable_path + " " + concatList(arguments) + "<br>");
    beginOutputRunLog(executable_path + " " + concatList(arguments), output_path_from_arguments(arguments));
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        appendOutputRunLog(tr("Failed to start: %1").arg(process->errorString()));
        finishOutputRunLog(-1, QProcess::CrashExit);
        Log("<b style='color:red;'>Failed to start the program.</b>");
        QMessageBox::critical(this, "Error", "Failed to start the program.");
    } else {
        play_stop->setEnabled(true);
    }
}

bool MainWindow::buildRunArguments(QStringList &arguments, PendingAcmxvkAction resume_action, bool include_extra_arguments, const QString &output_override) {
    QString deep_dream_error;
    if (!validateDeepDreamLaunch(deep_dream_error)) {
        QMessageBox::warning(this, tr("Deep Dream Settings"), deep_dream_error);
        return false;
    }
    QString stable_diffusion_error;
    if (!validateStableDiffusionLaunch(stable_diffusion_error)) {
        QMessageBox::warning(this, tr("Stable Diffusion Settings"), stable_diffusion_error);
        return false;
    }
    if (shader_path.length() == 0) {
        QMessageBox::information(this, "Select Shaders", "Select Shader Path");
        return false;
    }
    int index = 0;
    const int row = currentShaderRow();
    if (row < 0) {
        index = 0;
        Log("No selection, defaulting to index 0");
    } else {
        index = row;
        const QString selectedData = currentShaderName();
        Log("Selected shader: " + selectedData + " at index: " + QString::number(index));
    }
    if (items.isEmpty()) {
        QMessageBox::warning(this, tr("Empty Shader Library"), tr("The selected shader library contains no shaders."));
        return false;
    }
    if (index < 0 || index >= items.size()) {
        QMessageBox::warning(this, tr("Invalid Shader Selection"), tr("Select a shader from the active library."));
        return false;
    }
    QString launchShaderPath = shader_path;
    QString launchShaderName = items.at(index);
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString runtimeError;
        if (resume_action == PendingAcmxvkAction::CopyCommand) {
            if (is_acmxvk_source_library(shader_path, runtimeError)) {
                launchShaderPath = acmxvk_build_directory(shader_path);
                launchShaderName = acmxvk_runtime_shader_name(launchShaderName);
            } else if (!runtimeError.isEmpty()) {
                QMessageBox::warning(this, tr("ACMXVK Library"), runtimeError);
                return false;
            }
        } else {
            if (!resolve_acmxvk_runtime_library(shader_path, launchShaderPath, runtimeError)) {
                prompt_acmxvk_rebuild(runtimeError, resume_action);
                return false;
            }
            if (launchShaderPath != shader_path)
                launchShaderName = acmxvk_runtime_shader_name(launchShaderName);
            if (!QFileInfo(QDir(launchShaderPath).filePath(launchShaderName)).isFile()) {
                QMessageBox::warning(this,
                                     tr("Build ACMXVK Library"),
                                     tr("The compiled shader is missing. Choose Playback > "
                                        "Build and try again.\n\n%1")
                                         .arg(QDir(launchShaderPath).filePath(launchShaderName)));
                return false;
            }
        }
    }
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/" + acmx2::default_backend_executable(active_backend);
#endif
    dirPath = resolve_backend_assets_path(active_backend, executable_path, shader_path);

    QString shader_file = launchShaderPath;
    if (active_backend == acmx2::Backend::Acmxvk)
        arguments << "--unbuffered";
    arguments << "--path" << dirPath << "--shaders" << shader_file;
    arguments << "--interface-shm";
    // Always pass texture cache size so runtime SIZE matches the cache file.
    arguments << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        arguments << "--texture-cache-array";
    const QSize effectiveCameraResolution = hasPositiveResolution(camera_res) ? camera_res : QSize(1280, 720);
    QString res;
    QTextStream stream(&res);
    stream << effectiveCameraResolution.width() << "x" << effectiveCameraResolution.height();
    QString scr_res;
    QTextStream stream_r(&scr_res);
    stream_r << screen_res.width() << "x" << screen_res.height();

    if (full_screen_value)
        arguments << "--fullscreen";
    if (active_backend == acmx2::Backend::Acmxvk && enable_vsync)
        arguments << "--enable-vsync";
    if (active_backend == acmx2::Backend::Acmxvk && monitor_index > 0)
        arguments << "--monitor" << QString::number(monitor_index);

    if (!graphics_file.isEmpty()) {
        arguments << "--graphic" << graphics_file;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--fps" << QString::number(output_fps);
    } else if (video_file.isEmpty()) {
        arguments << "--camera-res" << res;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--device" << QString::number(camera_index);
        arguments << "--fps" << QString::number(output_fps);
        if (active_backend == acmx2::Backend::Acmxvk && maximize_fps)
            arguments << "--maximize-fps";
        if (use_yuv)
            arguments << "--use-yuv";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
    } else {
        arguments << "--input" << video_file;
        if (active_backend == acmx2::Backend::Acmxvk && use_source_fps) {
            arguments << "--use-source-fps";
            if (use_source_audio)
                arguments << "--use-source-audio";
        }
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        if (play_repeat->isChecked())
            arguments << "--repeat";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
        if (copy_audio)
            arguments << "--copy-audio";
    }
    arguments << "--prefix" << prefix_path;
    if (active_backend == acmx2::Backend::Acmxvk) {
        arguments << "--png-level" << QString::number(png_level);
    }
    const QString configured_output_file = output_override.isEmpty() ? output_file : output_override;
    if (!configured_output_file.isEmpty()) {
        const QString launch_output_file = output_override.isEmpty() && active_backend == acmx2::Backend::Acmxvk ? timestamped_output_path(configured_output_file) : configured_output_file;
        arguments << "--output" << launch_output_file;
        if (active_backend == acmx2::Backend::Acmxvk && encode_rate_control == "bitrate")
            arguments << "--video-bitrate" << encode_bitrate;
        else
            arguments << "--encode-crf" << QString::number(encode_crf);
        if (!encode_preset.isEmpty())
            arguments << "--encode-preset" << encode_preset;
        if (!encode_tune.isEmpty())
            arguments << "--encode-tune" << encode_tune;
        if (!encode_codec.isEmpty() && encode_codec != "auto")
            arguments << "--encode-codec" << encode_codec;
        if (!encode_parameters.isEmpty())
            arguments << "--encode-params" << encode_parameters;
        if (encode_realtime)
            arguments << "--encode-realtime";
        if (encode_no_drop && (!video_file.isEmpty() || !graphics_file.isEmpty()))
            arguments << "--no-drop";
        if (active_backend == acmx2::Backend::Acmxvk && encode_constant_frame_rate && !video_file.isEmpty() && !png_output)
            arguments << "--constant-frame-rate";
        if (active_backend == acmx2::Backend::Acmxvk && encode_fill_pts_gaps && !png_output)
            arguments << "--fill-pts-gaps";
    }
    arguments << "--shader-file" << launchShaderName;

    const bool sourceAudioActive = active_backend == acmx2::Backend::Acmxvk && !video_file.isEmpty() && use_source_fps && use_source_audio;
    if (audio_available && audio_enabled && !sourceAudioActive) {
        arguments << "--enable-audio";
        arguments << "--channels" << QString::number(audio_channels);

        if (audio_input == -1)
            arguments << "--audio-input" << "default";
        else
            arguments << "--audio-input" << QString::number(audio_input);

        if (record_audio) {
            QString wavPath;
            if (!configured_output_file.isEmpty()) {
                QFileInfo fi(configured_output_file);
                wavPath = fi.absolutePath() + "/" + fi.completeBaseName() + ".wav";
            } else {
                wavPath = prefix_path + "/recorded_audio.wav";
            }
            arguments << "--record-audio" << wavPath;
            arguments << "--record-gain" << QString::number(record_volume, 'f', 2);
        }
    }

    if (active_backend == acmx2::Backend::Acmxvk && !record_audio && (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--mute-output";
    }

    if (audio_available && (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--sense" << QString::number(audio_sense);
        if (audio_passthrough) {
            arguments << "--pass-through";
            if (audio_output == -1)
                arguments << "--audio-output" << "default";
            else
                arguments << "--audio-output" << QString::number(audio_output);
        }
    }

    if (audio_available && !audio_file.isEmpty() && !sourceAudioActive) {
        arguments << "--audio-file" << audio_file;
        if (audio_trunc) {
            arguments << "--audio-trunc";
        }
        if (audio_repeat) {
            arguments << "--audio-repeat";
        }
    }

    if (audio_available && audio_buffers_enabled) {
        arguments << "--enable-audio-buffers" << QString::number(audio_buffer_frames);
    }

    if (audio_available && (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--audio-warm-rate" << QString::number(audio_warm_rate, 'f', 2);
    }

    if (enable_3d) {
        arguments << "--enable-3d";
        arguments << "--model" << model_file;
    }

    if (onnx_model_enabled && !onnx_model.isEmpty()) {
        arguments << "--onnx" << onnx_model;
    }

    if (cuda_available && gpu_filter_enabled && !gpu_filter_indices.isEmpty()) {
        arguments << "--gpu-filter" << gpu_filter_indices;
        arguments << "--gpu-buffer" << QString::number(gpu_buffer_size);
    }

    appendDeepDreamArguments(arguments);
    appendStableDiffusionArguments(arguments);

    if (shader_pass_enabled && !shader_pass_names.isEmpty()) {
        QString passIndices = getShaderPassIndicesFromNames();
        if (!passIndices.isEmpty()) {
            QStringList passFiles;
            const QStringList indexValues = passIndices.split(',');
            for (const QString &indexValue : indexValues) {
                bool ok = false;
                const int passIndex = indexValue.toInt(&ok);
                if (ok && passIndex >= 0 && passIndex < items.size()) {
                    const QString passFile = items.at(passIndex);
                    passFiles.append(launchShaderPath == shader_path ? passFile : acmxvk_runtime_shader_name(passFile));
                }
            }
            QByteArray passFilePayload;
            for (const QString &passFile : passFiles) {
                const QByteArray encodedName = passFile.toUtf8();
                passFilePayload.append(QByteArray::number(encodedName.size()));
                passFilePayload.append(':');
                passFilePayload.append(encodedName);
            }
            arguments << "--shader-pass-files" << QString::fromUtf8(passFilePayload);
        }
    }

    if (cuda_device_available) {
        arguments << "--cuda-device" << QString::number(cuda_device);
    }

    arguments << "--time-speed" << QString::number(static_cast<double>(time_speed), 'f', 2);
    if (normalized_time) {
        arguments << "--normalized";
    }

    if (!use_shader_cache && active_backend == acmx2::Backend::Acmx2) {
        arguments << "--no-cache";
    }

    if (midi_available && midi_enabled && !midi_config_file.isEmpty()) {
        arguments << "--midi-map" << midi_config_file;
        if (midi_device >= 0)
            arguments << "--midi-device" << QString::number(midi_device);
    }

    const bool playlistActive = playlist_enabled && !playlist_names.isEmpty();
    if (playlistActive) {
        QString plFile = playlist_file_path;
        if (plFile.isEmpty()) {
            plFile = prefix_path + "/playlist.txt";
        }
        QFile f(plFile);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&f);
            if (!playlist_tree_data.isEmpty()) {
                for (const auto &[nodeName, shaders] : playlist_tree_data) {
                    out << "[" << nodeName << "]\n";
                    for (const QString &name : shaders) {
                        out << (launchShaderPath == shader_path ? name : acmxvk_runtime_shader_name(name)) << "\n";
                    }
                }
            } else {
                for (const QString &name : playlist_names) {
                    out << (launchShaderPath == shader_path ? name : acmxvk_runtime_shader_name(name)) << "\n";
                }
            }
            f.close();
            playlist_file_path = plFile;
        }
        arguments << "--playlist" << plFile;
    }

    if (playlistActive && autopilot_frames > 0) {
        arguments << (autopilot_random ? "--autopilot-random" : "--autopilot-frames") << QString::number(autopilot_frames);
    }

    if ((!output_file.isEmpty() || (active_backend == acmx2::Backend::Acmxvk && png_output)) && duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (!output_file.isEmpty() && max_size_limit_enabled && max_size_mb > 0.0) {
        arguments << "--max-size" << QString::number(max_size_mb, 'f', 2);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    if (flip_enabled) {
        arguments << "--flip";
    }

    if (rotate_enabled) {
        arguments << "--rotate" << rotation_mode;
    }

    if (active_backend == acmx2::Backend::Acmxvk && png_output && !png_output_directory.isEmpty()) {
        arguments << "--png" << png_output_directory;
    }

    const int effective_generate_interval = generate_enabled ? generate_interval : (active_backend == acmx2::Backend::Acmxvk && png_output ? 1 : 0);
    if (effective_generate_interval > 0) {
        arguments << "--generate" << QString::number(effective_generate_interval);
    }

    if (watermark_enabled && !watermark_text.isEmpty()) {
        arguments << "--use-watermark" << watermark_text;
        arguments << "--use-watermark-color" << QString("%1,%2,%3").arg(watermark_r).arg(watermark_g).arg(watermark_b);
    }

    if (display_filter_enabled) {
        arguments << "--display-filter";
    }

    if (include_extra_arguments) {
        arguments.append(QProcess::splitCommand(extra_arguments));
    }

    return true;
}

void MainWindow::runHdr10Conversion() {
    if (!hdr10Process) {
        return;
    }
    if (hdr10Process->state() == QProcess::Running) {
        Log("<b style='color:red;'>HDR10 conversion already running; skipping.</b>");
        return;
    }
    if (output_file.isEmpty() || !QFileInfo::exists(output_file)) {
        Log("<b style='color:red;'>HDR10 conversion: source file missing.</b>");
        return;
    }

    QFileInfo fi(output_file);
    const QString suffix = fi.suffix();
    const QString hdr10Path = fi.absolutePath() + "/" + fi.completeBaseName() + ".HDR10" + (suffix.isEmpty() ? QString() : "." + suffix);

    QStringList args;
    args << "-y"
         << "-i" << output_file;

    // Honor the user's codec selection from the recording settings dialog.
    // Values come from the encodeCodecComboBox: "auto", "software", "nvenc".
    // "auto" picks NVENC if CUDA is available, otherwise libx265.
    const QString codecChoice = encode_codec.toLower();
    bool useNvenc;
    if (codecChoice == "software" || codecChoice == "libx265" || codecChoice == "x265") {
        useNvenc = false;
    } else if (codecChoice == "nvenc" || codecChoice == "hevc_nvenc") {
        useNvenc = true;
    } else {
        useNvenc = cuda_available; // "auto" or empty
    }

    if (useNvenc) {
        // NVENC HEVC HDR10 path. p010le = 10-bit 4:2:0 semi-planar, required
        // by hevc_nvenc Main10. NVENC's preset namespace is p1..p7 (fastest
        // -> slowest); map the x264-style names from the UI combo onto it.
        QString nvencPreset;
        const QString p = encode_preset.toLower();
        if (p == "ultrafast")
            nvencPreset = "p1";
        else if (p == "superfast")
            nvencPreset = "p2";
        else if (p == "veryfast")
            nvencPreset = "p3";
        else if (p == "faster")
            nvencPreset = "p4";
        else if (p == "fast")
            nvencPreset = "p5";
        else if (p == "medium")
            nvencPreset = "p6";
        else if (p == "slow")
            nvencPreset = "p6";
        else if (p == "slower")
            nvencPreset = "p7";
        else if (p == "veryslow")
            nvencPreset = "p7";
        else if (p.startsWith("p") && p.size() == 2 && p[1].isDigit())
            nvencPreset = p; // already an NVENC preset
        else
            nvencPreset = "p6";

        args << "-vf" << "zscale=p=bt2020:t=smpte2084:m=bt2020nc,format=p010le"
             << "-c:v" << "hevc_nvenc"
             << "-preset" << nvencPreset << "-tune" << "hq"
             << "-b:v" << "56M"
             << "-maxrate" << "60M"
             << "-bufsize" << "60M"
             << "-color_primaries" << "bt2020"
             << "-colorspace" << "bt2020nc"
             << "-color_trc" << "smpte2084";
        Log("HDR10 codec: hevc_nvenc (CUDA detected, preset=" + nvencPreset + ")<br>");
    } else {
        args << "-vf" << "zscale=p=bt2020:t=smpte2084:m=bt2020nc,format=yuv420p10le"
             << "-c:v" << "libx265"
             << "-preset" << (encode_preset.isEmpty() ? QStringLiteral("medium") : encode_preset) << "-b:v" << "56M"
             << "-maxrate" << "60M"
             << "-bufsize" << "60M"
             << "-pix_fmt" << "yuv420p10le"
             << "-x265-params"
             << "hdr10=1:hdr10-opt=1:repeat-headers=1:"
                "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:"
                "master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1):"
                "max-cll=1000,400"
             << "-color_primaries" << "bt2020"
             << "-colorspace" << "bt2020nc"
             << "-color_trc" << "smpte2084";
        Log("HDR10 codec: libx265 (codec=" + (codecChoice.isEmpty() ? QStringLiteral("auto") : codecChoice) + ")<br>");
    }

    args << "-c:a" << "copy" << hdr10Path;

    Log("shell: ffmpeg " + concatList(args) + "<br>");
    Log("HDR10 output: " + hdr10Path + "<br>");

    // ffmpeg writes most of its progress to stderr; merge channels so the
    // log keeps messages in source order.
    hdr10Process->setProcessChannelMode(QProcess::MergedChannels);
    hdr10Process->start("ffmpeg", args);
    if (!hdr10Process->waitForStarted(5000)) {
        Log("<b style='color:red;'>Failed to start ffmpeg for HDR10 conversion.</b>");
        return;
    }
    play_stop->setEnabled(true);
}

void MainWindow::runAll() {
    if (process->state() == QProcess::Running) {
        QMessageBox::information(this, "Process Running", "A process is already running. Please stop it first.");
        return;
    }

#ifdef __linux__
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    for (const QString &entry : defaultLinuxRunEnvAssignments()) {
        int eq = entry.indexOf('=');
        if (eq <= 0) {
            continue;
        }
        env.insert(entry.left(eq), entry.mid(eq + 1));
    }
    process->setProcessEnvironment(env);
#endif

    QStringList arguments;
    if (!buildRunArguments(arguments, PendingAcmxvkAction::RunAll))
        return;
    initShaderSelectionSharedMemory();
    publishSelectedShaderIndexToRunningProcess();
    publishMultipassShadersToRunningProcess();
    publishRuntimeSettingsToRunningProcess();

    Log("shell: " + executable_path + " " + concatList(arguments) + "<br>");
    beginOutputRunLog(executable_path + " " + concatList(arguments), output_path_from_arguments(arguments));
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        appendOutputRunLog(tr("Failed to start: %1").arg(process->errorString()));
        finishOutputRunLog(-1, QProcess::CrashExit);
        Log("<b style='color:red;'>Failed to start the program.</b>");
        QMessageBox::critical(this, "Error", "Failed to start the program.");
    } else {
        play_stop->setEnabled(true);
    }
}

void MainWindow::copyCommand() {
    QStringList arguments;
    if (!buildRunArguments(arguments, PendingAcmxvkAction::CopyCommand, false))
        return;

    QString exe = executable_path;
    if (exe.isEmpty())
        exe = acmx2::default_backend_executable(active_backend);
    QStringList envAssignments;
#ifdef __linux__
    envAssignments = defaultLinuxRunEnvAssignments();
#endif
    QString commandText = buildShellCommand(envAssignments, exe, arguments).trimmed();

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Edit Command"));
    dialog.resize(720, active_backend == acmx2::Backend::Acmxvk ? 360 : 320);
    acmx2::applyCustomStyleIfEnabled(&dialog);

    QVBoxLayout *layout = new QVBoxLayout(&dialog);
    QPlainTextEdit *textBox = new QPlainTextEdit(&dialog);
    textBox->setPlainText(commandText);
    textBox->setReadOnly(false);
    textBox->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    if (!acmx2::isCustomStyleEnabled()) {
        textBox->setStyleSheet("QPlainTextEdit { background-color: black; color: lime; "
                               "font-size: 14px; font-family: 'Courier New', Courier, monospace; "
                               "border: 1px solid red; }");
    } else {
        QFont commandFont("Courier New");
        commandFont.setStyleHint(QFont::Monospace);
        commandFont.setPointSize(14);
        textBox->setFont(commandFont);
    }
    layout->addWidget(textBox);

    auto *extraArgumentsEdit = new QLineEdit(&dialog);
    extraArgumentsEdit->setText(extra_arguments);
    extraArgumentsEdit->setPlaceholderText(tr("--option value --another-option \"value with spaces\""));
    extraArgumentsEdit->setToolTip(tr("These arguments are appended after the generated arguments for "
                                      "Run Selected, Run All, and this dialog's command. Use double "
                                      "quotes around values containing spaces."));
    auto *extraArgumentsLayout = new QFormLayout();
    extraArgumentsLayout->addRow(tr("Extra arguments:"), extraArgumentsEdit);
    layout->addLayout(extraArgumentsLayout);

    if (active_backend == acmx2::Backend::Acmxvk) {
        QSettings settings("LostSideDead");
        const QString jobsKey = parallel_build_jobs_key();
        const int configuredJobs = parallel_build_jobs(settings);
        auto *parallelBuildCheckBox = new QCheckBox(tr("Enable parallel build"), &dialog);
        auto *parallelBuildJobsSpinBox = new QSpinBox(&dialog);
        parallelBuildJobsSpinBox->setRange(1, 256);
        parallelBuildJobsSpinBox->setValue(configuredJobs > 0 ? configuredJobs : 2);
        parallelBuildCheckBox->setChecked(configuredJobs > 0);
        parallelBuildJobsSpinBox->setEnabled(parallelBuildCheckBox->isChecked());
        parallelBuildJobsSpinBox->setToolTip(tr("Number of concurrent ACMXVK shader compiler jobs (1-256)."));
        auto *parallelBuildLayout = new QHBoxLayout();
        parallelBuildLayout->addWidget(parallelBuildCheckBox);
        parallelBuildLayout->addWidget(new QLabel(tr("Jobs:"), &dialog));
        parallelBuildLayout->addWidget(parallelBuildJobsSpinBox);
        parallelBuildLayout->addStretch(1);
        layout->addLayout(parallelBuildLayout);
        connect(parallelBuildCheckBox, &QCheckBox::toggled, &dialog, [parallelBuildJobsSpinBox, jobsKey](bool enabled) {
            parallelBuildJobsSpinBox->setEnabled(enabled);
            QSettings settings("LostSideDead");
            settings.setValue(jobsKey, enabled ? parallelBuildJobsSpinBox->value() : 0);
        });
        connect(parallelBuildJobsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), &dialog, [parallelBuildCheckBox, jobsKey](int jobs) {
            if (parallelBuildCheckBox->isChecked())
                QSettings("LostSideDead").setValue(jobsKey, jobs);
        });
    }

    QDialogButtonBox *buttonBox = new QDialogButtonBox(&dialog);
    QPushButton *copyButton = buttonBox->addButton(tr("Copy to Clipboard"), QDialogButtonBox::ActionRole);
    QPushButton *runButton = buttonBox->addButton(tr("Run"), QDialogButtonBox::ActionRole);
    QPushButton *okButton = buttonBox->addButton(QDialogButtonBox::Ok);
    layout->addWidget(buttonBox);

    const auto saveExtraArguments = [this, extraArgumentsEdit]() {
        extra_arguments = extraArgumentsEdit->text().trimmed();
        QSettings settings("LostSideDead", "acmx2");
        settings.setValue("interface/extra_arguments", extra_arguments);
    };
    const auto editedCommand = [textBox, extraArgumentsEdit]() {
        QString command = textBox->toPlainText().trimmed();
        const QString extras = extraArgumentsEdit->text().trimmed();
        if (!extras.isEmpty()) {
            command += QLatin1Char(' ');
            command += extras;
        }
        return command;
    };

    connect(copyButton, &QPushButton::clicked, &dialog, [saveExtraArguments, editedCommand, &dialog]() {
        saveExtraArguments();
        const QString copiedText = editedCommand();
        QClipboard *clipboard = QGuiApplication::clipboard();
        clipboard->setText(copiedText, QClipboard::Clipboard);
#ifdef __linux__
        if (clipboard->supportsSelection()) {
            clipboard->setText(copiedText, QClipboard::Selection);
        }
#endif
        QCoreApplication::processEvents();
        QMessageBox::information(&dialog, tr("Copied"), tr("Command copied to clipboard."));
    });
    connect(runButton, &QPushButton::clicked, &dialog, [this, saveExtraArguments, editedCommand, &dialog]() {
        if (process->state() != QProcess::NotRunning) {
            QMessageBox::information(&dialog, tr("Process Running"), tr("A process is already running. Please stop it first."));
            return;
        }
        saveExtraArguments();
        const QString cmdText = editedCommand();
        if (cmdText.isEmpty()) {
            QMessageBox::warning(&dialog, tr("Empty Command"), tr("The command is empty."));
            return;
        }

        process->setProcessEnvironment(QProcessEnvironment::systemEnvironment());
#ifdef Q_OS_WIN
        QStringList command_arguments = QProcess::splitCommand(cmdText);
        if (command_arguments.isEmpty()) {
            QMessageBox::warning(&dialog, tr("Invalid Command"), tr("The command does not contain an executable."));
            return;
        }
        const QString command_program = command_arguments.takeFirst();
#else
        const QString command_program = QStringLiteral("/bin/sh");
        const QStringList command_arguments{QStringLiteral("-c"), cmdText};
#endif
        Log("shell: " + cmdText + "<br>");
        initShaderSelectionSharedMemory();
        beginOutputRunLog(cmdText, output_path_from_arguments(QProcess::splitCommand(cmdText)));
        process->start(command_program, command_arguments);
        if (!process->waitForStarted()) {
            appendOutputRunLog(tr("Failed to start: %1").arg(process->errorString()));
            finishOutputRunLog(-1, QProcess::CrashExit);
            Log("<b style='color:red;'>Failed to start the program.</b>");
            QMessageBox::critical(&dialog, tr("Error"), tr("Failed to start the program."));
            return;
        }
        play_stop->setEnabled(true);
        dialog.accept();
    });
    connect(okButton, &QPushButton::clicked, &dialog, [saveExtraArguments, &dialog]() {
        saveExtraArguments();
        dialog.accept();
    });

    dialog.exec();
}

QString MainWindow::concatList(const QStringList lst) {
    QString text;
    QTextStream stream(&text);
    for (auto &i : lst) {
        stream << i << " ";
    }
    return text;
}

QString MainWindow::getShaderPassIndicesFromNames() {
    QStringList indices;
    for (const QString &name : shader_pass_names) {
        int idx = items.indexOf(name);
        if (idx >= 0) {
            indices.append(QString::number(idx));
        }
    }
    return indices.join(",");
}

QString MainWindow::sanitizeShaderName(const QString &name) {
    QString sanitized = name.trimmed();
    sanitized.replace('\\', '/');
    sanitized = QDir::cleanPath(sanitized);

    while (sanitized.startsWith("./")) {
        sanitized = sanitized.mid(2);
    }

    if (sanitized.isEmpty() || sanitized == "." || sanitized == "..") {
        Log("Warning: Invalid shader name detected: " + name);
        return QString();
    }

    if (QDir::isAbsolutePath(sanitized) || sanitized.startsWith("../") || sanitized.contains("/../") || sanitized.endsWith("/..")) {
        Log("Warning: Invalid shader name detected (path traversal attempt): " + name);
        return QString();
    }

    return sanitized;
}

void MainWindow::cleanupClosedEditors() {
    open_files.erase(std::remove_if(open_files.begin(), open_files.end(), [](const QPointer<TextEditor> &ptr) { return ptr.isNull(); }), open_files.end());
}

void MainWindow::menuShuffle() {
    if (items.isEmpty()) {
        return;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(items.begin(), items.end(), g);
    populateShaderTree();
    updateIndex();
    Log("Shaders shuffled");
}

void MainWindow::menuSort() {
    if (items.isEmpty()) {
        return;
    }
    items.sort(Qt::CaseInsensitive);
    populateShaderTree();
    updateIndex();
    Log("Shaders sorted alphabetically");
}

void MainWindow::menuBuildShaderCache() {
    QString build_path = shader_path;
    if (build_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        build_path = appSettings.value(acmx2::backend_settings_key(active_backend, "library"), active_backend == acmx2::Backend::Acmx2 ? appSettings.value("shaders", "").toString() : QString()).toString();
    }

    if (build_path.isEmpty()) {
        cacheBuildInProgress = false;
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }

    if (process->state() == QProcess::Running) {
        cacheBuildInProgress = false;
        QMessageBox::warning(this, "Error", "A process is already running. Please wait for it to finish.");
        return;
    }

    if (active_backend == acmx2::Backend::Acmxvk) {
        start_acmxvk_build(build_path, AcmxvkBuildMode::Strict);
        return;
    }

#ifdef Q_OS_MACOS
    // ACMX2 does not support its persistent OpenGL binary cache on macOS.
    Log("Rebuild Shader Cache is not available on macOS.");
    cacheBuildInProgress = false;
    return;
#else

    const QString assets_path = resolveAssetsPath();

    QStringList args;
    args << "--build" << build_path;
    args << "-p" << assets_path;
    args << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        args << "--texture-cache-array";

    if (enable_3d) {
        args << "--enable-3d";
    }

    Log("Building shader cache for: " + build_path);
    Log("Command: " + executable_path + " " + args.join(" ") + "<br>");

    play_stop->setEnabled(true);
    cacheBuildInProgress = true;
    initShaderSelectionSharedMemory();
    process->start(executable_path, args);

    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start shader cache build process");
        cacheBuildInProgress = false;
        play_stop->setEnabled(false);
    }
#endif
}

void MainWindow::menuFixBuild() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    if (process->state() == QProcess::Running) {
        QMessageBox::warning(this, tr("Fix Build"), tr("A process is already running. Please wait for it to finish."));
        return;
    }
    if (shader_path.isEmpty()) {
        QMessageBox::warning(this, tr("Fix Build"), tr("No shader library is loaded."));
        return;
    }
    start_acmxvk_build(shader_path, AcmxvkBuildMode::Fix);
}

void MainWindow::start_acmxvk_build(const QString &build_path, AcmxvkBuildMode mode) {
    const bool fix = mode != AcmxvkBuildMode::Strict;
    const bool prune = mode == AcmxvkBuildMode::Prune;
    const QString dialogTitle = prune ? tr("Remove Broken Shaders") : (fix ? tr("Fix Build") : tr("Build ACMXVK Library"));
    QString type_error;
    if (!is_acmxvk_source_library(build_path, type_error)) {
        pending_acmxvk_action = PendingAcmxvkAction::None;
        QMessageBox::warning(this,
                             dialogTitle,
                             type_error.isEmpty() ? tr("The selected ACMXVK library is already a compiled "
                                                       "runtime library.")
                                                  : type_error);
        return;
    }

    const QString manifest_path = QDir(build_path).filePath(QStringLiteral("library.json"));
    if (!QFileInfo(manifest_path).isFile()) {
        pending_acmxvk_action = PendingAcmxvkAction::None;
        QMessageBox::warning(this, dialogTitle, tr("ACMXVK source builds require library.json:\n%1").arg(manifest_path));
        return;
    }

    const QString output_path = acmxvk_build_directory(build_path);
    QString compilerError;
    const QString compiler = resolve_acmxvk_shader_compiler(compilerError);
    if (compiler.isEmpty()) {
        pending_acmxvk_action = PendingAcmxvkAction::None;
        Log(tr("<b style='color:red;'>Cannot build ACMXVK library: %1</b>").arg(compilerError.toHtmlEscaped()));
        QMessageBox::warning(this, tr("ACMXVK Shader Compiler"), compilerError);
        return;
    }
    QStringList arguments{"--unbuffered", "--build", manifest_path};
    arguments << (fix ? QStringLiteral("--fix") : QStringLiteral("--builddir")) << output_path;
    arguments << QStringLiteral("--glslc") << compiler;
    QSettings settings("LostSideDead");
    const int parallelBuildJobs = parallel_build_jobs(settings);
    if (parallelBuildJobs > 0)
        arguments << QStringLiteral("--parallel") << QString::number(parallelBuildJobs);
    if (prune)
        arguments << QStringLiteral("--prune") << QStringLiteral("--force");
    Log(prune ? tr("Removing broken ACMXVK shader sources from: %1").arg(build_path) : (fix ? tr("Fix building ACMXVK SPIR-V library: %1").arg(build_path) : tr("Building ACMXVK SPIR-V library: %1").arg(build_path)));
    Log("Command: " + executable_path + " " + concatList(arguments) + "<br>");
    play_stop->setEnabled(true);
    cacheBuildInProgress = true;
    acmxvkPruneLibraryPath = prune ? build_path : QString();
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start ACMXVK build process");
        cacheBuildInProgress = false;
        acmxvkPruneLibraryPath.clear();
        pending_acmxvk_action = PendingAcmxvkAction::None;
        play_stop->setEnabled(false);
    }
}

void MainWindow::menuRunFromCache() {}

void MainWindow::menuMetadataViewer() {
    MetadataViewer dlg(this);
    dlg.exec();
}

void MainWindow::menuRemoveBroken() {
    QString scan_path = shader_path;
    if (scan_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        scan_path = appSettings.value(acmx2::backend_settings_key(active_backend, "library"), active_backend == acmx2::Backend::Acmx2 ? appSettings.value("shaders", "").toString() : QString()).toString();
    }
    if (scan_path.isEmpty()) {
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }
    if (process->state() == QProcess::Running) {
        QMessageBox::warning(this, "Error", "A process is already running. Please wait for it to finish.");
        return;
    }

    const QString manifestPath = acmx2::shader_manifest_path(scan_path);
    if (manifestPath.isEmpty()) {
        QMessageBox::warning(this, "Missing Shader Manifest", "No library.json or index.txt found in: " + scan_path);
        return;
    }
    const QString manifestName = QFileInfo(manifestPath).fileName();

    if (active_backend == acmx2::Backend::Acmxvk) {
        QString typeError;
        if (!is_acmxvk_source_library(scan_path, typeError)) {
            QMessageBox::warning(this,
                                 tr("Remove Broken Shaders"),
                                 typeError.isEmpty() ? tr("Remove Broken requires an ACMXVK source library, "
                                                          "not a compiled runtime library.")
                                                     : typeError);
            return;
        }

        QMessageBox confirmation(this);
        confirmation.setIcon(QMessageBox::Warning);
        confirmation.setWindowTitle(tr("Permanently Remove Broken Shaders"));
        confirmation.setText(tr("This operation permanently deletes source shader files."));
        confirmation.setInformativeText(tr("ACMXVK will compile every shader listed in:\n\n%1\n\n"
                                           "Any .frag or .comp source for which glslc reports a compilation "
                                           "failure will be deleted. The generated runtime library and the "
                                           "source library manifest will then omit those shaders.\n\n"
                                           "No backup is created and this operation cannot be undone. "
                                           "Commit or archive the library before continuing.\n\n"
                                           "Do you want to permanently remove the broken sources?")
                                            .arg(scan_path));
        confirmation.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        confirmation.setDefaultButton(QMessageBox::No);
        confirmation.setEscapeButton(QMessageBox::No);
        if (confirmation.exec() != QMessageBox::Yes)
            return;

        start_acmxvk_build(scan_path, AcmxvkBuildMode::Prune);
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                              tr("Remove Broken Shaders"),
                                                              tr("This will compile every shader in:\n\n%1\n\n"
                                                                 "Any shader that fails to compile will be removed from %2 "
                                                                 "(the original will be backed up as %2.bak).\n\nContinue?")
                                                                  .arg(scan_path, manifestName),
                                                              QMessageBox::Yes | QMessageBox::No);
    if (reply != QMessageBox::Yes)
        return;

    const QString assets_path = resolveAssetsPath();

    QStringList args;
    args << "--remove-broken" << scan_path;
    args << "-p" << assets_path;
    args << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        args << "--texture-cache-array";
    if (enable_3d)
        args << "--enable-3d";

    Log("Scanning for broken shaders in: " + scan_path);
    Log("Command: " + executable_path + " " + args.join(" ") + "<br>");

    // Use a dedicated QProcess so we can reload the list when it finishes
    // without interfering with the main playback process.
    QProcess *scan = new QProcess(this);
    scan->setProcessChannelMode(QProcess::SeparateChannels);
    connect(scan, &QProcess::readyReadStandardOutput, this, [this, scan]() {
        QString output = scan->readAllStandardOutput();
        output.replace("\n", "<br>");
        this->Write(output);
    });
    connect(scan, &QProcess::readyReadStandardError, this, [this, scan]() {
        QString output = scan->readAllStandardError();
        output.replace("\n", "<br>");
        this->Write("<b style='color:red;'>" + output + "</b>");
    });
    connect(scan, static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, [this, scan, scan_path, manifestName](int exitCode, QProcess::ExitStatus) {
        Log(QString("Remove-broken finished with exit code: %1<br>").arg(exitCode));
        if (exitCode == 0) {
            // Reload the list view from the updated manifest.
            loadShaders(scan_path, true);
            QMessageBox::information(this,
                                     tr("Remove Broken"),
                                     tr("Finished scanning shader library.\n\n"
                                        "%1 has been updated and the shader list reloaded.\n"
                                        "A backup of the original is at:\n%2/%1.bak")
                                         .arg(manifestName, scan_path));
        } else {
            QMessageBox::warning(this,
                                 tr("Remove Broken"),
                                 tr("Remove-broken failed with exit code %1. "
                                    "%2 was not changed.")
                                     .arg(exitCode)
                                     .arg(manifestName));
        }
        scan->deleteLater();
    });

    scan->start(executable_path, args);
    if (!scan->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start remove-broken process");
        scan->deleteLater();
    }
}

void MainWindow::menuCleanShaderCache() {
#ifdef Q_OS_MACOS
    Log("Clean Shader Cache is not available on macOS.");
    return;
#else
    QString libraryPath = shader_path;
    if (libraryPath.isEmpty()) {
        QSettings appSettings("LostSideDead");
        libraryPath = appSettings.value("shaders", "").toString();
    }

    if (libraryPath.isEmpty()) {
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }
    if (process->state() == QProcess::Running || cacheBuildInProgress) {
        QMessageBox::warning(this, "Error", "A process is running. Stop it before cleaning the shader cache.");
        return;
    }

    const QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                                    tr("Clean Shader Cache"),
                                                                    tr("Delete all cached shader binaries for:\n\n%1\n\n"
                                                                       "This will not rebuild the cache. Continue?")
                                                                        .arg(libraryPath),
                                                                    QMessageBox::Yes | QMessageBox::No);
    if (reply != QMessageBox::Yes)
        return;

    const QString assetsPath = resolveAssetsPath();
    QStringList cacheFiles;
    const auto addCacheFile = [&cacheFiles](const QString &path) {
        if (!cacheFiles.contains(path))
            cacheFiles.append(path);
    };

    // Current cache files are keyed by texture-cache size and array mode.
    // Enumerate every valid combination so cleaning is independent of the
    // currently selected Session Settings.
    for (int size = 1; size <= 64; ++size) {
        for (const bool useArray : {false, true}) {
            const QString filename = shaderCacheFilename(libraryPath, size, useArray);
            addCacheFile(assetsPath + "/" + filename);
            addCacheFile(libraryPath + "/" + filename);
        }
    }

    // Remove the pre-size-key hashed cache and the original fixed-name cache.
    std::error_code ec;
    const std::filesystem::path libraryFsPath(libraryPath.toStdString());
    const std::filesystem::path absoluteLibrary = std::filesystem::absolute(libraryFsPath, ec);
    const std::string legacyKey = ec ? libraryPath.toStdString() : absoluteLibrary.lexically_normal().string();
    std::ostringstream legacyName;
    legacyName << ".shader_cache_" << std::hex << std::hash<std::string>{}(legacyKey);
    const QString legacyHashedName = QString::fromStdString(legacyName.str());
    addCacheFile(assetsPath + "/" + legacyHashedName);
    addCacheFile(libraryPath + "/" + legacyHashedName);
    addCacheFile(libraryPath + "/.shader_cache");

    int removedCount = 0;
    int failedCount = 0;
    for (const QString &cacheFile : cacheFiles) {
        if (!QFileInfo::exists(cacheFile))
            continue;
        if (QFile::remove(cacheFile)) {
            Log("Deleted shader cache: " + cacheFile);
            ++removedCount;
        } else {
            Log("<b style='color:red;'>Warning:</b> Could not delete cache file: " + cacheFile);
            ++failedCount;
        }
    }

    if (removedCount == 0 && failedCount == 0) {
        Log("No existing shader cache found");
    } else {
        Log(QString("Shader cache clean complete: removed %1 file(s), %2 failed").arg(removedCount).arg(failedCount));
    }
    refreshShaderTreeMetadata();
#endif
}

void MainWindow::detectCudaSupport() { detectFeatureSupport(); }

static QString probe_feature_output(const QString &exe, const QString &flag) {
    QProcess probe;
    probe.start(exe, QStringList() << flag);
    if (!probe.waitForFinished(5000)) {
        probe.kill();
        return {};
    }
    return QString::fromLocal8Bit(probe.readAllStandardOutput()).trimmed();
}

static bool probeFeature(const QString &exe, const QString &flag, const QString &token) { return probe_feature_output(exe, flag).contains(token, Qt::CaseInsensitive); }

void MainWindow::detectFeatureSupport() {
    const bool isAcmxvk = active_backend == acmx2::Backend::Acmxvk;
    const QString backendName = acmx2::backend_name(active_backend);
    const QString cudaOutput = probe_feature_output(executable_path, "--check-cuda");
    cuda_available = cudaOutput.contains(isAcmxvk ? "acidcam-gpu filters: enabled" : "CUDA: enabled", Qt::CaseInsensitive);
    cuda_device_available = isAcmxvk ? cudaOutput.contains("MXVK CUDA interop: enabled", Qt::CaseInsensitive) : cuda_available;
    audio_available = probeFeature(executable_path, "--check-audio", "AUDIO: enabled");
    midi_available = probeFeature(executable_path, "--check-midi", "MIDI: enabled");
    dnn_available = probeFeature(executable_path, "--check-dnn", isAcmxvk ? "OpenCV DNN effects: enabled" : "OpenCV DNN: enabled");
    deep_dream_available = isAcmxvk && probeFeature(executable_path, "--check-deep-dream", "Deep Dream: enabled");
    stable_diffusion_available = isAcmxvk && probeFeature(executable_path, "--check-stable-diffusion", "Stable Diffusion: enabled");

    Log(QString("CUDA filters: %1 (%2)").arg(cuda_available ? "enabled" : "disabled", backendName));
    if (isAcmxvk) {
        Log(QString("CUDA device interop: %1 (%2)").arg(cuda_device_available ? "enabled" : "disabled", backendName));
    }
    Log(QString("AUDIO: %1 (%2)").arg(audio_available ? "enabled" : "disabled", backendName));
    Log(QString("MIDI: %1 (%2)").arg(midi_available ? "enabled" : "disabled", backendName));
    Log(QString("OpenCV DNN: %1 (%2)").arg(dnn_available ? "enabled" : "disabled", backendName));
    if (isAcmxvk) {
        Log(QString("Deep Dream: %1 (%2)").arg(deep_dream_available ? "enabled" : "disabled", backendName));
        Log(QString("Stable Diffusion: %1 (%2)").arg(stable_diffusion_available ? "enabled" : "disabled", backendName));
    }

    if (!dnn_available) {
        onnx_model_enabled = false;
        onnx_model.clear();
    }

    if (deepDreamAction) {
        deepDreamAction->setVisible(isAcmxvk);
        deepDreamAction->setEnabled(deep_dream_available);
        deepDreamAction->setToolTip(deep_dream_available ? QString() : tr("Disabled: ACMXVK was built without Deep Dream support."));
    }
    if (isAcmxvk && !deep_dream_available) {
        deep_dream_enabled = false;
        deep_dream_gpu_filter_first = false;
    }
    if (stableDiffusionAction) {
        stableDiffusionAction->setVisible(isAcmxvk);
        stableDiffusionAction->setEnabled(stable_diffusion_available);
        stableDiffusionAction->setToolTip(stable_diffusion_available ? QString()
                                                                     : tr("Disabled: ACMXVK was built without Stable Diffusion "
                                                                          "support."));
    }
    if (isAcmxvk && !stable_diffusion_available) {
        stable_diffusion_enabled = false;
    }
    if (gpuFilterAction) {
        gpuFilterAction->setEnabled(cuda_available);
        gpuFilterAction->setToolTip(cuda_available ? QString() : tr("Disabled: %1 was built without acidcam-gpu filter support.").arg(backendName));
    }
    if (!cuda_available) {
        gpu_filter_enabled = false;
        gpu_filter_indices.clear();
        if (!cuda_device_available)
            cuda_device = 0;
    }

    if (audioSet) {
        audioSet->setEnabled(audio_available);
        audioSet->setToolTip(audio_available ? QString() : tr("Disabled: %1 was built without audio support.").arg(backendName));
    }
    if (!audio_available) {
        audio_enabled = false;
        record_audio = false;
        audio_passthrough = false;
        audio_file.clear();
        audio_trunc = false;
        audio_repeat = false;
    }

    if (midiSettingsAction) {
        midiSettingsAction->setEnabled(midi_available);
        midiSettingsAction->setToolTip(midi_available ? QString() : tr("Disabled: %1 was built without MIDI support.").arg(backendName));
    }
    if (!midi_available) {
        midi_enabled = false;
        midi_config_file.clear();
        midi_device = -1;
    }
}
