#include "acmxvk-source-manifest.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <iostream>

namespace {
    bool write_file(const QString &path, const QByteArray &contents) {
        if (!QDir().mkpath(QFileInfo(path).absolutePath()))
            return false;
        QFile file(path);
        return file.open(QIODevice::WriteOnly) &&
               file.write(contents) == contents.size();
    }

    QJsonObject read_manifest(const QString &path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
            return {};
        return QJsonDocument::fromJson(file.readAll()).object();
    }
} // namespace

int main() {
    QTemporaryDir temporary;
    if (!temporary.isValid())
        return 1;
    const QString root = temporary.path();
    if (!write_file(QDir(root).filePath(QStringLiteral("z.frag")),
                    QByteArrayLiteral("#version 450\n")) ||
        !write_file(QDir(root).filePath(QStringLiteral("a.frag")),
                    QByteArrayLiteral(
                        "#define slider1 ext.custom_uniforms[5].x\n")) ||
        !write_file(QDir(root).filePath(QStringLiteral("compute/fx.comp")),
                    QByteArrayLiteral("#version 450\n"))) {
        return 2;
    }

    acmx2::AcmxvkSourceManifestResult result;
    QString error;
    if (!acmx2::create_acmxvk_source_manifest(root, QString(), result, error)) {
        std::cerr << error.toStdString() << '\n';
        return 3;
    }
    if (result.fragmentCount != 2 || result.computeCount != 1 ||
        result.customUniformCount != 27) {
        return 4;
    }

    const QString manifestPath = QDir(root).filePath(QStringLiteral("library.json"));
    QJsonObject manifest = read_manifest(manifestPath);
    const QJsonArray shaders = manifest.value(QStringLiteral("shaders")).toArray();
    if (manifest.value(QStringLiteral("backend")).toString() !=
            QStringLiteral("acmxvk") ||
        manifest.value(QStringLiteral("library_type")).toString() !=
            QStringLiteral("source") ||
        shaders.size() != 3 || shaders.at(0).toString() != QStringLiteral("a.frag") ||
        shaders.at(2).toString() != QStringLiteral("compute/fx.comp")) {
        return 5;
    }

    QJsonObject uniforms =
        manifest.value(QStringLiteral("custom_uniforms")).toObject();
    QJsonObject slider = uniforms.value(QStringLiteral("slider1")).toObject();
    slider.insert(QStringLiteral("value"), 0.75);
    slider.insert(QStringLiteral("maximum"), 2.0);
    uniforms.insert(QStringLiteral("slider1"), slider);
    manifest.insert(QStringLiteral("custom_uniforms"), uniforms);
    if (!write_file(manifestPath,
                    QJsonDocument(manifest).toJson(QJsonDocument::Indented))) {
        return 6;
    }
    if (!acmx2::create_acmxvk_source_manifest(root, QString(), result, error)) {
        std::cerr << error.toStdString() << '\n';
        return 7;
    }
    slider = read_manifest(manifestPath)
                 .value(QStringLiteral("custom_uniforms"))
                 .toObject()
                 .value(QStringLiteral("slider1"))
                 .toObject();
    if (slider.value(QStringLiteral("slot")).toInt() != 20 ||
        slider.value(QStringLiteral("value")).toDouble() != 0.75 ||
        slider.value(QStringLiteral("maximum")).toDouble() != 2.0) {
        return 8;
    }
    return 0;
}
