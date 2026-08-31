#include "acmxvk-source-manifest.hpp"
#include "shader-manifest.hpp"

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

    if (!acmx2::remove_shader_manifest_entry(root, QStringLiteral("z.frag"),
                                             error)) {
        std::cerr << error.toStdString() << '\n';
        return 9;
    }
    QStringList remainingShaders;
    if (!acmx2::load_shader_manifest(root, remainingShaders, error) ||
        remainingShaders.contains(QStringLiteral("z.frag"),
                                  Qt::CaseInsensitive) ||
        !QFileInfo::exists(QDir(root).filePath(QStringLiteral("z.frag")))) {
        return 10;
    }
    if (!write_file(QDir(root).filePath(QStringLiteral("new.frag")),
                    QByteArrayLiteral("#version 450\n"))) {
        return 11;
    }
    remainingShaders.append(QStringLiteral("new.frag"));
    if (!acmx2::create_acmxvk_source_manifest_for_shaders(
            root, remainingShaders, QString(), result, error)) {
        std::cerr << error.toStdString() << '\n';
        return 12;
    }
    remainingShaders.clear();
    if (!acmx2::load_shader_manifest(root, remainingShaders, error) ||
        remainingShaders.contains(QStringLiteral("z.frag"),
                                  Qt::CaseInsensitive) ||
        !remainingShaders.contains(QStringLiteral("new.frag"),
                                   Qt::CaseInsensitive)) {
        return 13;
    }

    QList<acmx2::CustomUniformDefinition> customUniforms;
    if (!acmx2::load_custom_uniforms(root, customUniforms, error) ||
        customUniforms.size() != 27) {
        return 14;
    }
    customUniforms.append({QStringLiteral("custom_knob"), 0.0, 2.0, 0.1,
                           1.0, 27});
    if (!acmx2::write_custom_uniforms(root, customUniforms, error)) {
        std::cerr << error.toStdString() << '\n';
        return 15;
    }
    QJsonObject writtenUniforms =
        read_manifest(manifestPath)
            .value(QStringLiteral("custom_uniforms"))
            .toObject();
    if (writtenUniforms.value(QStringLiteral("custom_knob"))
            .toObject()
            .value(QStringLiteral("slot"))
            .toInt(-1) != 27) {
        return 16;
    }

    customUniforms.removeLast();
    if (!acmx2::write_custom_uniforms(root, customUniforms, error) ||
        read_manifest(manifestPath)
            .value(QStringLiteral("custom_uniforms"))
            .toObject()
            .contains(QStringLiteral("custom_knob"))) {
        return 17;
    }

    while (customUniforms.size() < 65) {
        const int slot = customUniforms.size();
        customUniforms.append(
            {QStringLiteral("extra_%1").arg(slot), 0.0, 1.0, 0.01, 0.0,
             slot});
    }
    if (acmx2::write_custom_uniforms(root, customUniforms, error) ||
        !error.contains(QStringLiteral("at most 64"))) {
        return 18;
    }

    QTemporaryDir runtimeDirectory;
    if (!runtimeDirectory.isValid())
        return 19;
    QJsonObject runtimeManifest = read_manifest(manifestPath);
    QJsonObject runtimeUniforms =
        runtimeManifest.value(QStringLiteral("custom_uniforms")).toObject();
    QJsonObject runtimeSlider =
        runtimeUniforms.value(QStringLiteral("slider1")).toObject();
    runtimeSlider.insert(QStringLiteral("value"), 0.75000000000001);
    runtimeUniforms.insert(QStringLiteral("slider1"), runtimeSlider);
    runtimeManifest.insert(QStringLiteral("custom_uniforms"), runtimeUniforms);
    if (!write_file(QDir(runtimeDirectory.path())
                        .filePath(QStringLiteral("library.json")),
                    QJsonDocument(runtimeManifest)
                        .toJson(QJsonDocument::Indented))) {
        return 20;
    }
    bool metadataMatches = false;
    if (!acmx2::custom_uniform_metadata_matches(
            root, runtimeDirectory.path(), metadataMatches, error) ||
        !metadataMatches) {
        return 21;
    }
    runtimeSlider.insert(QStringLiteral("value"), 0.8);
    runtimeUniforms.insert(QStringLiteral("slider1"), runtimeSlider);
    runtimeManifest.insert(QStringLiteral("custom_uniforms"), runtimeUniforms);
    if (!write_file(QDir(runtimeDirectory.path())
                        .filePath(QStringLiteral("library.json")),
                    QJsonDocument(runtimeManifest)
                        .toJson(QJsonDocument::Indented)) ||
        !acmx2::custom_uniform_metadata_matches(
            root, runtimeDirectory.path(), metadataMatches, error) ||
        metadataMatches) {
        return 22;
    }
    return 0;
}
