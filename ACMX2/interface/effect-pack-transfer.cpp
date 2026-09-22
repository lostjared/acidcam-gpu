#include "effect-pack-transfer.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QTemporaryDir>
#include <QUuid>

namespace acmx2 {
    namespace {
        bool safe_relative_path(const QString &path) {
            if (path.isEmpty() || path.contains('\\') || path.contains(':') || QDir::isAbsolutePath(path) || path.contains(QRegularExpression(QStringLiteral("[<>\"|?*]")))) {
                return false;
            }
            for (const QString &part : path.split('/')) {
                if (part.isEmpty() || part == QStringLiteral(".") || part == QStringLiteral("..") || part.endsWith('.') || part.endsWith(' ')) {
                    return false;
                }
                const QString base = part.section('.', 0, 0).toUpper();
                if (base == QStringLiteral("CON") || base == QStringLiteral("PRN") || base == QStringLiteral("AUX") || base == QStringLiteral("NUL") || QRegularExpression(QStringLiteral("^(COM|LPT)[1-9]$")).match(base).hasMatch()) {
                    return false;
                }
            }
            return true;
        }

        bool below_root(const QString &path, const QString &root) {
            const QString relative = QDir(root).relativeFilePath(path);
            return relative != QStringLiteral("..") && !relative.startsWith(QStringLiteral("../")) && !QDir::isAbsolutePath(relative);
        }

        bool collect_file(const QString &relative, const QString &root, QSet<QString> &files, QString &error, int depth) {
            if (depth > 32 || files.size() >= 256 || !safe_relative_path(relative)) {
                error = QStringLiteral("Unsafe or excessive pack resource: %1").arg(relative);
                return false;
            }
            const QFileInfo requested(QDir(root).filePath(relative));
            if (requested.isSymLink()) {
                error = QStringLiteral("Pack resources must not be symbolic links: %1").arg(relative);
                return false;
            }
            const QString absolute = requested.canonicalFilePath();
            const QFileInfo info(absolute);
            if (absolute.isEmpty() || !below_root(absolute, root) || !info.isFile() || info.size() > 64 * 1024 * 1024) {
                error = QStringLiteral("Missing, oversized, or external pack resource: %1").arg(relative);
                return false;
            }
            const QString canonical_relative = QDir::fromNativeSeparators(QDir(root).relativeFilePath(absolute));
            if (!safe_relative_path(canonical_relative) || canonical_relative != QDir::cleanPath(relative)) {
                error = QStringLiteral("Pack resource is not portable: %1").arg(relative);
                return false;
            }
            if (files.contains(canonical_relative)) {
                return true;
            }
            files.insert(canonical_relative);
            if (!relative.endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".glsl"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".inc"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".h"), Qt::CaseInsensitive)) {
                return true;
            }
            QFile source(absolute);
            if (!source.open(QIODevice::ReadOnly) || source.size() > 8 * 1024 * 1024) {
                error = QStringLiteral("Cannot read shader source: %1").arg(relative);
                return false;
            }
            const QString contents = QString::fromUtf8(source.readAll());
            const QRegularExpression include_expression(QStringLiteral(R"(^\s*#\s*include\s*(["<])([^">]+)[">])"), QRegularExpression::MultilineOption);
            auto matches = include_expression.globalMatch(contents);
            while (matches.hasNext()) {
                const QRegularExpressionMatch match = matches.next();
                const QString include = match.captured(2);
                const QString local = QDir::cleanPath(QFileInfo(canonical_relative).dir().filePath(include));
                const QString include_path = match.captured(1) == QStringLiteral("\"") && QFileInfo(QDir(root).filePath(local)).isFile() ? local : include;
                if (!collect_file(include_path, root, files, error, depth + 1)) {
                    return false;
                }
            }
            return true;
        }

        QString unique_folder(const QString &root, const QString &requested) {
            QString name = requested;
            for (int suffix = 2; QFileInfo::exists(QDir(root).filePath(name)); ++suffix) {
                name = requested + QStringLiteral("-%1").arg(suffix);
            }
            return name;
        }
    } // namespace

    EffectPackTransferResult transfer_effect_pack(const EffectPackTransferRequest &request, const EffectPackTransferProgress &progress) {
        EffectPackTransferResult result;
        const QString root = QFileInfo(request.source_root).canonicalFilePath();
        const QString destination_root = QFileInfo(request.destination_root).canonicalFilePath();
        if (root.isEmpty() || !QFileInfo(root).isDir() || destination_root.isEmpty() || !QFileInfo(destination_root).isDir()) {
            result.error = QStringLiteral("Source or destination folder is unavailable.");
            return result;
        }
        if (!QRegularExpression(QStringLiteral("^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")).match(request.folder_name).hasMatch() || !safe_relative_path(request.folder_name)) {
            result.error = QStringLiteral("Use an ASCII folder name with letters, digits, hyphens, or underscores.");
            return result;
        }
        QJsonObject manifest = request.manifest;
        if (manifest.value(QStringLiteral("format")).toString() != QStringLiteral("acmxvk-effect-pack") || manifest.value(QStringLiteral("version")).toInt() != 1 || !manifest.value(QStringLiteral("passes")).isArray() || manifest.value(QStringLiteral("passes")).toArray().isEmpty()) {
            result.error = QStringLiteral("The effect-pack manifest is invalid.");
            return result;
        }
        if (request.assign_new_id) {
            const QString suffix = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
            manifest.insert(QStringLiteral("id"), manifest.value(QStringLiteral("id")).toString() + QStringLiteral(".") + suffix);
        }
        QSet<QString> files;
        for (const QJsonValue &pass : manifest.value(QStringLiteral("passes")).toArray()) {
            if (!pass.isString() || (!pass.toString().endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) && !pass.toString().endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive))) {
                result.error = QStringLiteral("Portable pack transfer requires GLSL .frag or .comp source passes.");
                return result;
            }
            if (!collect_file(pass.toString(), root, files, result.error, 0)) {
                return result;
            }
        }
        if (manifest.contains(QStringLiteral("icon")) && !collect_file(manifest.value(QStringLiteral("icon")).toString(), root, files, result.error, 0)) {
            return result;
        }
        QString icon_name;
        if (!request.external_icon.isEmpty()) {
            const QFileInfo icon(request.external_icon);
            const QString suffix = icon.suffix().toLower();
            if (!icon.isFile() || icon.size() > 4 * 1024 * 1024 || (suffix != QStringLiteral("png") && suffix != QStringLiteral("webp") && suffix != QStringLiteral("jpg") && suffix != QStringLiteral("jpeg"))) {
                result.error = QStringLiteral("The selected icon must be a PNG, WebP, or JPEG under 4 MiB.");
                return result;
            }
            icon_name = QStringLiteral("icon.") + suffix;
            manifest.insert(QStringLiteral("icon"), icon_name);
        }
        if (request.infer_requirements) {
            QJsonObject requirements;
            const QRegularExpression binding(QStringLiteral(R"(\bbinding\s*=\s*([2346])\b)"));
            for (const QString &relative : files) {
                if (!relative.endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".glsl"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".inc"), Qt::CaseInsensitive) && !relative.endsWith(QStringLiteral(".h"), Qt::CaseInsensitive)) {
                    continue;
                }
                QFile source(QDir(root).filePath(relative));
                if (!source.open(QIODevice::ReadOnly)) {
                    result.error = QStringLiteral("Cannot inspect shader: %1").arg(relative);
                    return result;
                }
                auto matches = binding.globalMatch(QString::fromUtf8(source.readAll()));
                while (matches.hasNext()) {
                    const int index = matches.next().captured(1).toInt();
                    const QString key = index == 2 ? QStringLiteral("history") : index == 3 ? QStringLiteral("spectrum") : index == 4 ? QStringLiteral("spectrum_history") : QStringLiteral("original_frame");
                    requirements.insert(key, true);
                }
            }
            manifest.insert(QStringLiteral("requires"), requirements);
        }
        const QString name = unique_folder(destination_root, request.folder_name);
        QTemporaryDir staging(QDir(destination_root).filePath(QStringLiteral(".acmx-pack-XXXXXX")));
        if (!staging.isValid()) {
            result.error = QStringLiteral("Cannot create a temporary pack folder in the destination.");
            return result;
        }
        const QStringList ordered_files = QStringList(files.values());
        const int total = ordered_files.size() + 1 + (icon_name.isEmpty() ? 0 : 1);
        int current = 0;
        for (const QString &relative : ordered_files) {
            const QString destination = QDir(staging.path()).filePath(relative);
            if (!QDir().mkpath(QFileInfo(destination).absolutePath()) || !QFile::copy(QDir(root).filePath(relative), destination)) {
                result.error = QStringLiteral("Could not copy pack resource: %1").arg(relative);
                return result;
            }
            QFile copied(destination);
            if (!copied.open(QIODevice::ReadWrite) || !copied.setFileTime(QFileInfo(QDir(root).filePath(relative)).lastModified(), QFileDevice::FileModificationTime)) {
                result.error = QStringLiteral("Could not preserve pack resource timestamp: %1").arg(relative);
                return result;
            }
            if (progress) {
                progress(++current, total);
            }
        }
        if (!icon_name.isEmpty()) {
            if (!QFile::copy(request.external_icon, QDir(staging.path()).filePath(icon_name))) {
                result.error = QStringLiteral("Could not copy the selected icon.");
                return result;
            }
            if (progress) {
                progress(++current, total);
            }
        }
        QSaveFile output(QDir(staging.path()).filePath(QStringLiteral("effect.json")));
        if (!output.open(QIODevice::WriteOnly) || output.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented)) < 0 || !output.commit()) {
            result.error = QStringLiteral("Could not write effect.json.");
            return result;
        }
        if (progress) {
            progress(++current, total);
        }
        const QString final_path = QDir(destination_root).filePath(name);
        if (!QDir().rename(staging.path(), final_path)) {
            result.error = QStringLiteral("Could not finalize effect pack in %1.").arg(destination_root);
            return result;
        }
        staging.setAutoRemove(false);
        result.destination = final_path;
        result.success = true;
        return result;
    }
} // namespace acmx2
