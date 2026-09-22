#include "effect-pack-project.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <algorithm>

namespace {
    bool write_file(const QString &path, const QByteArray &contents, const QDateTime &modified) {
        if (!QDir().mkpath(QFileInfo(path).absolutePath()))
            return false;
        QFile file(path);
        return file.open(QIODevice::WriteOnly) && file.write(contents) == contents.size() && file.setFileTime(modified, QFileDevice::FileModificationTime);
    }
    QString fingerprint(QList<QPair<QString, QByteArray>> files) {
        std::sort(files.begin(), files.end(), [](const auto &left, const auto &right) { return left.first < right.first; });
        quint64 hash = 14695981039346656037ULL;
        for (const auto &file : files) {
            for (const QByteArray &piece : {file.first.toUtf8(), file.second}) {
                for (const char value : piece)
                    hash = (hash ^ static_cast<unsigned char>(value)) * 1099511628211ULL;
                hash = (hash ^ 0U) * 1099511628211ULL;
            }
        }
        return QString::number(hash, 16).rightJustified(16, QLatin1Char('0'));
    }
} // namespace

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir source;
    QTemporaryDir project;
    if (!source.isValid() || !project.isValid())
        return 1;
    const QDateTime source_time = QDateTime::currentDateTimeUtc().addSecs(-20);
    const QDateTime compiled_time = QDateTime::currentDateTimeUtc().addSecs(-10);
    const QString manifest = QDir(source.path()).filePath(QStringLiteral("effect.json"));
    const QString shader = QDir(source.path()).filePath(QStringLiteral("shaders/main.frag"));
    const QString include = QDir(source.path()).filePath(QStringLiteral("include/colors.glsl"));
    const QString cache = QDir(source.path()).filePath(QStringLiteral(".acmxvk-build"));
    const QJsonObject pack{{QStringLiteral("format"), QStringLiteral("acmxvk-effect-pack")}, {QStringLiteral("version"), 1}, {QStringLiteral("id"), QStringLiteral("test.project")}, {QStringLiteral("name"), QStringLiteral("Project")}, {QStringLiteral("passes"), QJsonArray{QStringLiteral("shaders/main.frag"), QStringLiteral("shaders/main.frag")}}};
    const QByteArray shader_source = "#version 450\n#include \"../include/colors.glsl\"\n";
    const QByteArray include_source = "vec3 color() { return vec3(1); }\n";
    const QString pass_hash = fingerprint({{QStringLiteral("shaders/main.frag"), shader_source}, {QStringLiteral("include/colors.glsl"), include_source}});
    const QJsonObject cache_manifest{{QStringLiteral("format"), QStringLiteral("acmxvk-effect-cache")}, {QStringLiteral("version"), 2}, {QStringLiteral("pack_id"), QStringLiteral("test.project")}, {QStringLiteral("shader_abi"), QStringLiteral("acmxvk-effect-abi-1")}, {QStringLiteral("vulkan_target"), QStringLiteral("vulkan1.0")}, {QStringLiteral("source_hash"), fingerprint({{QStringLiteral("shaders/main.frag"), shader_source}})}, {QStringLiteral("include_hash"), fingerprint({{QStringLiteral("include/colors.glsl"), include_source}})}, {QStringLiteral("passes"), QJsonArray{QStringLiteral("shaders/main.frag.spv"), QStringLiteral("shaders/main.frag.spv")}}, {QStringLiteral("pass_hashes"), QJsonArray{pass_hash, pass_hash}}};
    if (!write_file(manifest, QJsonDocument(pack).toJson(), source_time) || !write_file(shader, shader_source, source_time) || !write_file(include, include_source, source_time) || !write_file(QDir(cache).filePath(QStringLiteral("effect-cache.json")), QJsonDocument(cache_manifest).toJson(), compiled_time) || !write_file(QDir(cache).filePath(QStringLiteral("shaders/main.frag.spv")), QByteArray::fromHex("0302230700000100000000000100000000000000"), compiled_time))
        return 2;

    QString error;
    if (!acmx2::validate_effect_pack_project_cache(manifest, QStringLiteral("test.project"), error))
        return 3;
    EffectPackProjectState state{manifest, QStringLiteral("test.project"), QJsonObject{{QStringLiteral("symmetry"), 6.0}}, {}};
    QString relative;
    if (!acmx2::bundle_effect_pack_project(state, project.path(), relative, error))
        return 4;
    const QJsonObject entry{{QStringLiteral("id"), state.id}, {QStringLiteral("manifest"), relative}, {QStringLiteral("values"), state.values}};
    EffectPackProjectState restored;
    if (!acmx2::resolve_effect_pack_project_state(entry, project.path(), restored, error) || restored.values.value(QStringLiteral("symmetry")).toDouble() != 6.0 || !acmx2::validate_effect_pack_project_cache(restored.manifest_path, restored.id, error))
        return 5;
    QString same_relative;
    if (!acmx2::bundle_effect_pack_project(restored, project.path(), same_relative, error) || same_relative != relative)
        return 6;
    const QString copied_cache_path = QDir(QFileInfo(restored.manifest_path).absolutePath()).filePath(QStringLiteral(".acmxvk-build/effect-cache.json"));
    QJsonObject incompatible_cache = cache_manifest;
    incompatible_cache.insert(QStringLiteral("vulkan_target"), QStringLiteral("vulkan9.9"));
    if (!write_file(copied_cache_path, QJsonDocument(incompatible_cache).toJson(), compiled_time) || acmx2::validate_effect_pack_project_cache(restored.manifest_path, restored.id, error))
        return 9;
    QJsonObject unsafe = entry;
    unsafe.insert(QStringLiteral("manifest"), QStringLiteral("resources/effect-packs/../other/effect.json"));
    if (acmx2::resolve_effect_pack_project_state(unsafe, project.path(), restored, error))
        return 7;
    if (!write_file(include, "vec3 color() { return vec3(0); }\n", QDateTime::currentDateTimeUtc()) || acmx2::validate_effect_pack_project_cache(manifest, state.id, error))
        return 8;
    return 0;
}
