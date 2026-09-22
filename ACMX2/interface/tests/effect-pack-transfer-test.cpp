#include "effect-pack-transfer.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QTemporaryDir>

namespace {
    bool write_file(const QString &path, const QByteArray &data) {
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile file(path);
        return file.open(QIODevice::WriteOnly) && file.write(data) == data.size();
    }
} // namespace

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir source;
    QTemporaryDir output;
    if (!source.isValid() || !output.isValid() || !write_file(QDir(source.path()).filePath(QStringLiteral("shaders/main.frag")), "#version 450\n#extension GL_GOOGLE_include_directive : require\n#include \"../include/colors.glsl\"\nlayout(binding = 2) uniform sampler2D historyTex;\n") || !write_file(QDir(source.path()).filePath(QStringLiteral("include/colors.glsl")), "vec3 tint() { return vec3(1); }\n") || !write_file(QDir(source.path()).filePath(QStringLiteral(".acmxvk-build/old.frag.spv")), "stale")) {
        return 1;
    }
    QJsonObject manifest{{QStringLiteral("format"), QStringLiteral("acmxvk-effect-pack")}, {QStringLiteral("version"), 1}, {QStringLiteral("id"), QStringLiteral("test.transfer")}, {QStringLiteral("name"), QStringLiteral("Transfer")}, {QStringLiteral("passes"), QJsonArray{QStringLiteral("shaders/main.frag")}}};
    acmx2::EffectPackTransferRequest request{source.path(), output.path(), QStringLiteral("transfer"), manifest};
    request.infer_requirements = true;
    const acmx2::EffectPackTransferResult copied = acmx2::transfer_effect_pack(request);
    if (!copied.success || !QFileInfo::exists(QDir(copied.destination).filePath(QStringLiteral("include/colors.glsl"))) || QFileInfo::exists(QDir(copied.destination).filePath(QStringLiteral(".acmxvk-build/old.frag.spv")))) {
        return 2;
    }
    QFile result(QDir(copied.destination).filePath(QStringLiteral("effect.json")));
    if (!result.open(QIODevice::ReadOnly) || !result.readAll().contains("\"history\": true")) {
        return 3;
    }
    const acmx2::EffectPackTransferResult again = acmx2::transfer_effect_pack(request);
    if (!again.success || again.destination == copied.destination) {
        return 4;
    }
    manifest.insert(QStringLiteral("passes"), QJsonArray{QStringLiteral("../outside.frag")});
    request.manifest = manifest;
    if (acmx2::transfer_effect_pack(request).success) {
        return 5;
    }
    manifest.insert(QStringLiteral("passes"), QJsonArray{QStringLiteral(".acmxvk-build/old.frag.spv")});
    request.manifest = manifest;
    if (acmx2::transfer_effect_pack(request).success) {
        return 6;
    }
    return 0;
}
