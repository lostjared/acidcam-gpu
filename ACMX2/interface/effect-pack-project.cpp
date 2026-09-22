#include "effect-pack-project.hpp"
#include "effect-pack-transfer.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace acmx2 {
    namespace {
        bool safe_relative_path(const QString &path) {
            if (path.isEmpty() || path.contains(QLatin1Char('\\')) || path.contains(QLatin1Char(':')) || QDir::isAbsolutePath(path) || QDir::cleanPath(path) != path) {
                return false;
            }
            for (const QString &part : path.split(QLatin1Char('/'))) {
                if (part.isEmpty() || part == QStringLiteral(".") || part == QStringLiteral("..")) {
                    return false;
                }
            }
            return true;
        }

        bool below_root(const QString &path, const QString &root) {
            const QString relative = QDir(root).relativeFilePath(path);
            return relative != QStringLiteral("..") && !relative.startsWith(QStringLiteral("../")) && !QDir::isAbsolutePath(relative);
        }

        QString checked_file(const QString &root, const QString &relative, QString &error) {
            if (!safe_relative_path(relative)) {
                error = QStringLiteral("Unsafe effect-pack path: %1").arg(relative);
                return {};
            }
            const QFileInfo requested(QDir(root).filePath(relative));
            const QString canonical = requested.canonicalFilePath();
            if (requested.isSymLink() || canonical.isEmpty() || !below_root(canonical, root) || !QFileInfo(canonical).isFile()) {
                error = QStringLiteral("Missing or external effect-pack file: %1").arg(relative);
                return {};
            }
            return canonical;
        }

        bool collect_dependencies(const QString &root, const QString &relative, QSet<QString> &seen, QDateTime &latest, QString &error, int depth) {
            if (depth > 32 || seen.size() >= 256) {
                error = QStringLiteral("Effect-pack include graph exceeds its limits.");
                return false;
            }
            const QString path = checked_file(root, relative, error);
            if (path.isEmpty()) {
                return false;
            }
            if (seen.contains(path)) {
                return true;
            }
            seen.insert(path);
            const QFileInfo info(path);
            if (info.size() > 8 * 1024 * 1024) {
                error = QStringLiteral("Effect-pack shader source is too large: %1").arg(relative);
                return false;
            }
            latest = std::max(latest, info.lastModified());
            QFile source(path);
            if (!source.open(QIODevice::ReadOnly)) {
                error = QStringLiteral("Could not read effect-pack source: %1").arg(relative);
                return false;
            }
            const QRegularExpression include_expression(QStringLiteral(R"(^\s*#\s*include\s*(["<])([^">]+)[">])"), QRegularExpression::MultilineOption);
            auto matches = include_expression.globalMatch(QString::fromUtf8(source.readAll()));
            while (matches.hasNext()) {
                const QRegularExpressionMatch match = matches.next();
                const QString include = match.captured(2);
                const QString local = QDir::cleanPath(QFileInfo(relative).dir().filePath(include));
                const QString include_path = match.captured(1) == QStringLiteral("\"") && QFileInfo(QDir(root).filePath(local)).isFile() ? local : include;
                if (!collect_dependencies(root, include_path, seen, latest, error, depth + 1)) {
                    return false;
                }
            }
            return true;
        }

        QString fingerprint_files(const QString &root, QStringList files, QString &error) {
            std::sort(files.begin(), files.end(), [](const QString &left, const QString &right) { return left.toUtf8() < right.toUtf8(); });
            quint64 hash = 14695981039346656037ULL;
            for (const QString &relative : files) {
                const QString path = checked_file(root, relative, error);
                QFile file(path);
                if (path.isEmpty() || !file.open(QIODevice::ReadOnly) || file.size() > 8 * 1024 * 1024) {
                    error = QStringLiteral("Could not hash effect-pack source: %1").arg(relative);
                    return {};
                }
                const auto append = [&hash](const QByteArray &bytes) {
                    for (const char value : bytes)
                        hash = (hash ^ static_cast<unsigned char>(value)) * 1099511628211ULL;
                    hash = (hash ^ 0U) * 1099511628211ULL;
                };
                append(relative.toUtf8());
                append(file.readAll());
            }
            return QString::number(hash, 16).rightJustified(16, QLatin1Char('0'));
        }

        bool copy_with_timestamp(const QString &source, const QString &destination, QString &error) {
            if (!QDir().mkpath(QFileInfo(destination).absolutePath()) || !QFile::copy(source, destination)) {
                error = QStringLiteral("Could not copy effect-pack cache file: %1").arg(source);
                return false;
            }
            QFile output(destination);
            if (!output.open(QIODevice::ReadWrite) || !output.setFileTime(QFileInfo(source).lastModified(), QFileDevice::FileModificationTime)) {
                error = QStringLiteral("Could not preserve effect-pack cache timestamp: %1").arg(source);
                return false;
            }
            return true;
        }

        QString project_file(const QString &root, const QString &relative, QString &error) {
            const QString canonical_root = QFileInfo(root).canonicalFilePath();
            if (canonical_root.isEmpty()) {
                error = QStringLiteral("Project directory is unavailable.");
                return {};
            }
            return checked_file(canonical_root, relative, error);
        }
    } // namespace

    bool validate_effect_pack_project_cache(const QString &manifest_path, const QString &expected_id, QString &error) {
        error.clear();
        const QFileInfo manifest_info(manifest_path);
        const QString root = manifest_info.dir().canonicalPath();
        if (manifest_info.fileName() != QStringLiteral("effect.json") || root.isEmpty() || manifest_info.isSymLink() || manifest_info.canonicalFilePath() != QDir(root).filePath(QStringLiteral("effect.json")) || manifest_info.size() > 1024 * 1024) {
            error = QStringLiteral("Effect-pack manifest is missing, external, or too large.");
            return false;
        }
        QFile manifest(manifest_path);
        if (!manifest.open(QIODevice::ReadOnly)) {
            error = QStringLiteral("Could not read effect-pack manifest.");
            return false;
        }
        QJsonParseError parse_error;
        const QJsonDocument document = QJsonDocument::fromJson(manifest.readAll(), &parse_error);
        const QJsonObject object = document.object();
        const QJsonArray passes = object.value(QStringLiteral("passes")).toArray();
        if (parse_error.error != QJsonParseError::NoError || !document.isObject() || object.value(QStringLiteral("format")).toString() != QStringLiteral("acmxvk-effect-pack") || object.value(QStringLiteral("version")).toInt() != 1 || object.value(QStringLiteral("id")).toString() != expected_id || expected_id.isEmpty() || passes.isEmpty() || passes.size() > 64) {
            error = QStringLiteral("Effect-pack manifest has an invalid format, ID, or pass list.");
            return false;
        }
        const QString cache_root = QDir(root).filePath(QStringLiteral(".acmxvk-build"));
        if (QFileInfo(cache_root).isSymLink() || !QFileInfo(cache_root).isDir()) {
            error = QStringLiteral("Effect-pack compiled cache is missing or unsafe.");
            return false;
        }
        const QString cache_manifest_path = checked_file(cache_root, QStringLiteral("effect-cache.json"), error);
        QFile cache_manifest(cache_manifest_path);
        if (cache_manifest_path.isEmpty() || !cache_manifest.open(QIODevice::ReadOnly) || cache_manifest.size() > 1024 * 1024) {
            error = QStringLiteral("Effect-pack cache manifest is missing or unreadable.");
            return false;
        }
        const QJsonDocument cache_document = QJsonDocument::fromJson(cache_manifest.readAll());
        const QJsonObject cache = cache_document.object();
        const QJsonArray cached_passes = cache.value(QStringLiteral("passes")).toArray();
        if (!cache_document.isObject() || cache.value(QStringLiteral("format")).toString() != QStringLiteral("acmxvk-effect-cache") || cache.value(QStringLiteral("version")).toInt() != 2 || cache.value(QStringLiteral("pack_id")).toString() != expected_id || cache.value(QStringLiteral("shader_abi")).toString() != QStringLiteral("acmxvk-effect-abi-1") || cache.value(QStringLiteral("vulkan_target")).toString() != QStringLiteral("vulkan1.0") || cached_passes.size() != passes.size() || cache.value(QStringLiteral("pass_hashes")).toArray().size() != passes.size()) {
            error = QStringLiteral("Effect-pack compiled cache does not match the manifest.");
            return false;
        }
        QDateTime latest = manifest_info.lastModified();
        QSet<QString> seen;
        QSet<QString> source_paths;
        for (int index = 0; index < passes.size(); ++index) {
            if (!passes[index].isString()) {
                error = QStringLiteral("Effect-pack pass is not a filename.");
                return false;
            }
            const QString source = passes[index].toString();
            source_paths.insert(source);
            QSet<QString> pass_files;
            QDateTime pass_latest;
            if (!collect_dependencies(root, source, pass_files, pass_latest, error, 0))
                return false;
            QStringList relative_files;
            for (const QString &file : pass_files)
                relative_files.append(QDir(root).relativeFilePath(file));
            const QString pass_hash = fingerprint_files(root, relative_files, error);
            if (pass_hash.isEmpty() || cache.value(QStringLiteral("pass_hashes")).toArray()[index].toString() != pass_hash) {
                if (error.isEmpty())
                    error = QStringLiteral("Effect-pack pass source or includes differ from the compiled cache.");
                return false;
            }
            if ((!source.endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) && !source.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive)) || !collect_dependencies(root, source, seen, latest, error, 0)) {
                if (error.isEmpty()) {
                    error = QStringLiteral("Effect-pack pass must be a readable GLSL source: %1").arg(source);
                }
                return false;
            }
            const QString output_name = source + QStringLiteral(".spv");
            if (cached_passes[index].toString() != output_name) {
                error = QStringLiteral("Effect-pack cached pass order does not match the manifest.");
                return false;
            }
            const QString output = checked_file(cache_root, output_name, error);
            if (output.isEmpty() || QFileInfo(output).lastModified() < latest || QFileInfo(output).size() < 20 || QFileInfo(output).size() > 64 * 1024 * 1024 || QFileInfo(output).size() % 4 != 0) {
                if (error.isEmpty()) {
                    error = QStringLiteral("Effect-pack compiled pass is missing, stale, or invalid: %1").arg(output_name);
                }
                return false;
            }
            QFile spirv(output);
            if (!spirv.open(QIODevice::ReadOnly) || spirv.read(4) != QByteArray::fromHex("03022307")) {
                error = QStringLiteral("Effect-pack compiled pass is not SPIR-V: %1").arg(output_name);
                return false;
            }
        }
        QStringList source_files = source_paths.values();
        QStringList include_files;
        for (const QString &file : seen) {
            const QString relative = QDir(root).relativeFilePath(file);
            if (!source_paths.contains(relative))
                include_files.append(relative);
        }
        if (cache.value(QStringLiteral("source_hash")).toString() != fingerprint_files(root, source_files, error) || cache.value(QStringLiteral("include_hash")).toString() != fingerprint_files(root, include_files, error)) {
            if (error.isEmpty())
                error = QStringLiteral("Effect-pack source or include hash differs from the compiled cache.");
            return false;
        }
        for (const QJsonValue &pass : passes) {
            if (QFileInfo(checked_file(cache_root, pass.toString() + QStringLiteral(".spv"), error)).lastModified() < latest) {
                error = QStringLiteral("Effect-pack compiled cache is stale after an include change.");
                return false;
            }
        }
        return true;
    }

    bool bundle_effect_pack_project(const EffectPackProjectState &state, const QString &project_root, QString &relative_manifest, QString &error) {
        relative_manifest.clear();
        if (!validate_effect_pack_project_cache(state.manifest_path, state.id, error)) {
            return false;
        }
        const QString source_root = QFileInfo(state.manifest_path).canonicalPath();
        const QString project_directory = QFileInfo(project_root).canonicalFilePath();
        const QString existing_relative = QDir(project_directory).relativeFilePath(source_root);
        if (existing_relative.startsWith(QStringLiteral("resources/effect-packs/")) && below_root(source_root, project_directory)) {
            relative_manifest = QDir(project_directory).relativeFilePath(state.manifest_path);
            return true;
        }
        const QString destination_parent = QDir(project_root).filePath(QStringLiteral("resources/effect-packs"));
        if (!QDir().mkpath(destination_parent)) {
            error = QStringLiteral("Could not create the project effect-pack directory.");
            return false;
        }
        QFile source_manifest(state.manifest_path);
        if (!source_manifest.open(QIODevice::ReadOnly)) {
            error = QStringLiteral("Could not read effect-pack manifest.");
            return false;
        }
        const QJsonObject manifest = QJsonDocument::fromJson(source_manifest.readAll()).object();
        QString folder = QFileInfo(source_root).fileName().toLower();
        folder.replace(QRegularExpression(QStringLiteral("[^a-z0-9_-]+")), QStringLiteral("-"));
        folder = folder.left(80);
        if (folder.isEmpty() || !folder.front().isLetterOrNumber()) {
            folder = QStringLiteral("effect-pack");
        }
        const EffectPackTransferResult transferred = transfer_effect_pack({source_root, destination_parent, folder, manifest});
        if (!transferred.success) {
            error = transferred.error;
            return false;
        }
        const QString destination_manifest = QDir(transferred.destination).filePath(QStringLiteral("effect.json"));
        if (!QFile::remove(destination_manifest) || !copy_with_timestamp(state.manifest_path, destination_manifest, error)) {
            return false;
        }
        const QJsonArray passes = manifest.value(QStringLiteral("passes")).toArray();
        QSet<QString> outputs;
        for (const QJsonValue &pass : passes) {
            outputs.insert(pass.toString() + QStringLiteral(".spv"));
        }
        outputs.insert(QStringLiteral("effect-cache.json"));
        for (const QString &name : outputs) {
            if (!copy_with_timestamp(QDir(source_root).filePath(QStringLiteral(".acmxvk-build/") + name), QDir(transferred.destination).filePath(QStringLiteral(".acmxvk-build/") + name), error)) {
                return false;
            }
        }
        if (!validate_effect_pack_project_cache(destination_manifest, state.id, error)) {
            return false;
        }
        relative_manifest = QDir(project_root).relativeFilePath(destination_manifest);
        return true;
    }

    bool resolve_effect_pack_project_state(const QJsonObject &object, const QString &project_root, EffectPackProjectState &state, QString &error) {
        error.clear();
        state = {};
        state.id = object.value(QStringLiteral("id")).toString();
        const QString manifest = object.value(QStringLiteral("manifest")).toString();
        if (state.id.isEmpty() || !manifest.startsWith(QStringLiteral("resources/effect-packs/")) || !manifest.endsWith(QStringLiteral("/effect.json")) || !object.value(QStringLiteral("values")).isObject()) {
            error = QStringLiteral("Project effect-pack identity, path, or control values are invalid.");
            return false;
        }
        state.manifest_path = project_file(project_root, manifest, error);
        if (state.manifest_path.isEmpty()) {
            return false;
        }
        state.values = object.value(QStringLiteral("values")).toObject();
        if (state.values.size() > 64) {
            error = QStringLiteral("Project effect pack has too many control overrides.");
            return false;
        }
        for (auto it = state.values.constBegin(); it != state.values.constEnd(); ++it) {
            if (it.key().isEmpty() || !it.value().isDouble() || !std::isfinite(it.value().toDouble())) {
                error = QStringLiteral("Project effect-pack control override is invalid: %1").arg(it.key());
                return false;
            }
        }
        const QString model = object.value(QStringLiteral("model")).toString();
        if (!model.isEmpty()) {
            if (!model.startsWith(QStringLiteral("resources/models/"))) {
                error = QStringLiteral("Project effect-pack model path is unsafe.");
                return false;
            }
            state.dream_model_file = project_file(project_root, model, error);
            if (state.dream_model_file.isEmpty()) {
                return false;
            }
        }
        return true;
    }
} // namespace acmx2
