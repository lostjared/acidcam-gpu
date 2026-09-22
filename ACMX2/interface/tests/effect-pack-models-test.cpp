#include "effect-pack-models.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <iostream>

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir temporary;
    if (!temporary.isValid()) {
        return 1;
    }
    const QString model = QDir(temporary.path()).filePath(QStringLiteral("deep-dream-vgg16.pt"));
    QFile file(model);
    if (!file.open(QIODevice::WriteOnly) || file.write("model-test") < 0) {
        return 2;
    }
    file.close();
    if (acmx2::resolve_effect_pack_model(QStringLiteral("vgg16"), {temporary.path()}, {}) != QFileInfo(model).canonicalFilePath()) {
        std::cerr << "logical model ID did not resolve\n";
        return 3;
    }
    if (!acmx2::resolve_effect_pack_model(QStringLiteral("../vgg16"), {temporary.path()}, {}).isEmpty()) {
        std::cerr << "unsafe model ID resolved\n";
        return 4;
    }
    if (acmx2::resolve_effect_pack_model(QStringLiteral("vgg16"), {}, model) != QFileInfo(model).canonicalFilePath()) {
        std::cerr << "explicit model override was ignored\n";
        return 5;
    }
    if (!acmx2::resolve_effect_pack_model(QStringLiteral("vgg16"), {temporary.path()}, model + QStringLiteral(".missing")).isEmpty()) {
        std::cerr << "missing explicit override fell back silently\n";
        return 6;
    }
    return 0;
}
