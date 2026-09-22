#include "effect-pack-models.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QStandardPaths>

namespace acmx2 {
    QStringList effect_pack_model_roots(const QString &configured_model) {
        QSettings settings("LostSideDead", "acmx2");
        QStringList roots = settings.value(QStringLiteral("effect_packs/model_roots")).toStringList();
        if (QFileInfo(configured_model).isFile()) {
            roots.prepend(QFileInfo(configured_model).absolutePath());
        }
        roots << QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/models");
        roots << QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../share/acmxvk/models"));
        roots << QDir::current().absoluteFilePath(QStringLiteral("models"));
        roots.removeDuplicates();
        return roots;
    }

    QString resolve_effect_pack_model(const QString &model_id, const QStringList &roots, const QString &override_path) {
        if (!override_path.isEmpty()) {
            const QFileInfo override_file(override_path);
            return override_file.isFile() ? override_file.canonicalFilePath() : QString();
        }
        if (model_id.isEmpty() || model_id == QStringLiteral(".") || model_id == QStringLiteral("..") || model_id.contains('/') || model_id.contains('\\')) {
            return {};
        }
        QStringList names{model_id};
        if (!model_id.endsWith(QStringLiteral(".pt"), Qt::CaseInsensitive)) {
            names << model_id + QStringLiteral(".pt");
            names << QStringLiteral("deep-dream-") + model_id + QStringLiteral(".pt");
        }
        if (!model_id.endsWith(QStringLiteral(".torchscript"), Qt::CaseInsensitive)) {
            names << model_id + QStringLiteral(".torchscript");
        }
        for (const QString &root : roots) {
            const QDir directory(root);
            for (const QString &name : names) {
                const QFileInfo candidate(directory.filePath(name));
                if (candidate.isFile()) {
                    return candidate.canonicalFilePath();
                }
            }
        }
        return {};
    }
} // namespace acmx2
