#include "acmxvk-source-manifest.hpp"

#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QTextStream>

int main(int argc, char **argv) {
    QCoreApplication application(argc, argv);
    QCoreApplication::setApplicationName(
        QStringLiteral("create_acmxvk_source_manifest"));
    QCoreApplication::setApplicationVersion(QStringLiteral("1.0"));

    QCommandLineParser parser;
    parser.setApplicationDescription(
        QStringLiteral("Create an ACMXVK source library.json from Vulkan GLSL "
                       "fragment and compute sources."));
    parser.addHelpOption();
    parser.addVersionOption();
    const QCommandLineOption rootOption(
        QStringList{QStringLiteral("r"), QStringLiteral("root")},
        QStringLiteral("Shader source directory to scan recursively."),
        QStringLiteral("directory"));
    const QCommandLineOption outputOption(
        QStringList{QStringLiteral("o"), QStringLiteral("output")},
        QStringLiteral("Manifest path (defaults to ROOT/library.json)."),
        QStringLiteral("file"));
    parser.addOption(rootOption);
    parser.addOption(outputOption);
    parser.process(application);

    if (!parser.isSet(rootOption))
        parser.showHelp(1);

    acmx2::AcmxvkSourceManifestResult result;
    QString error;
    if (!acmx2::create_acmxvk_source_manifest(
            parser.value(rootOption), parser.value(outputOption), result, error)) {
        QTextStream(stderr) << "Error: " << error << '\n';
        return 1;
    }

    QTextStream(stdout)
        << "Wrote " << result.outputPath << '\n'
        << result.fragmentCount << " fragment shader(s), "
        << result.computeCount << " compute shader(s), "
        << result.customUniformCount << " custom-uniform slot(s)\n";
    return 0;
}
