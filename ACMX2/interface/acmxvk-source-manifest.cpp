#include "acmxvk-source-manifest.hpp"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSaveFile>
#include <QStringList>
#include <algorithm>

namespace {
    struct UniformMetadata {
        double minimum = 0.0;
        double maximum = 1.0;
        double step = 0.01;
        double value = 0.0;
    };

    QHash<int, QString> default_slot_names() {
        return {{0, QStringLiteral("square_size")},
                {1, QStringLiteral("alpha_value")},
                {2, QStringLiteral("alpha_r")},
                {3, QStringLiteral("alpha_g")},
                {4, QStringLiteral("alpha_b")},
                {5, QStringLiteral("value_alpha_r")},
                {6, QStringLiteral("value_alpha_g")},
                {7, QStringLiteral("value_alpha_b")},
                {8, QStringLiteral("index_value")},
                {9, QStringLiteral("restore_black")},
                {10, QStringLiteral("seed")},
                {11, QStringLiteral("value1")},
                {12, QStringLiteral("iChannelTime")},
                {13, QStringLiteral("time_speed")},
                {14, QStringLiteral("blendAmt")},
                {15, QStringLiteral("uDistortion")},
                {16, QStringLiteral("uRotateSpeed")},
                {17, QStringLiteral("uWarpSpeed")},
                {18, QStringLiteral("uRandRate")},
                {19, QStringLiteral("uPhaseRate")},
                {20, QStringLiteral("slider1")},
                {21, QStringLiteral("slider2")},
                {22, QStringLiteral("slider3")},
                {23, QStringLiteral("slider4")},
                {24, QStringLiteral("frequency")},
                {25, QStringLiteral("strength")},
                {26, QStringLiteral("random_seed")}};
    }

    QHash<QString, UniformMetadata> default_metadata() {
        QHash<QString, UniformMetadata> metadata;
        metadata.insert(QStringLiteral("square_size"), {1.0, 128.0, 1.0, 55.0});
        metadata.insert(QStringLiteral("slider1"), {0.0, 1.0, 0.01, 0.5});
        metadata.insert(QStringLiteral("slider2"), {0.0, 1.0, 0.01, 0.6});
        metadata.insert(QStringLiteral("slider3"), {0.0, 1.0, 0.01, 0.35});
        metadata.insert(QStringLiteral("slider4"), {0.0, 1.0, 0.01, 0.8});
        return metadata;
    }

    bool read_existing_metadata(const QString &path,
                                QHash<QString, UniformMetadata> &metadata,
                                QString &error) {
        if (!QFileInfo::exists(path))
            return true;

        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            error = QObject::tr("Cannot read existing manifest %1: %2")
                        .arg(path, file.errorString());
            return false;
        }
        QJsonParseError parseError;
        const QJsonDocument document =
            QJsonDocument::fromJson(file.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            error = QObject::tr("Cannot parse existing manifest %1 at offset %2: %3")
                        .arg(path)
                        .arg(parseError.offset)
                        .arg(parseError.errorString());
            return false;
        }

        const QJsonObject uniforms =
            document.object().value(QStringLiteral("custom_uniforms")).toObject();
        for (auto it = uniforms.constBegin(); it != uniforms.constEnd(); ++it) {
            if (!it.value().isObject())
                continue;
            const UniformMetadata fallback = metadata.value(it.key());
            const QJsonObject value = it.value().toObject();
            metadata.insert(it.key(),
                            {value.value(QStringLiteral("minimum"))
                                 .toDouble(fallback.minimum),
                             value.value(QStringLiteral("maximum"))
                                 .toDouble(fallback.maximum),
                             value.value(QStringLiteral("step"))
                                 .toDouble(fallback.step),
                             value.value(QStringLiteral("value"))
                                 .toDouble(fallback.value)});
        }
        return true;
    }

    bool scan_slots(const QString &path, QHash<int, QString> &slotNames,
                    QHash<QString, int> &nameSlots, QString &error) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            error = QObject::tr("Cannot read %1: %2").arg(path, file.errorString());
            return false;
        }

        static const QRegularExpression aliasExpression(
            QStringLiteral("^\\s*#define\\s+([A-Za-z_][A-Za-z0-9_]*)\\s+"
                           "ext\\.custom_uniforms\\[(\\d+)\\]\\.([xyzw])"));
        const QStringList lines =
            QString::fromUtf8(file.readAll()).split(QLatin1Char('\n'));
        for (const QString &line : lines) {
            const QRegularExpressionMatch match = aliasExpression.match(line);
            if (!match.hasMatch())
                continue;
            const QString name = match.captured(1);
            const int vector = match.captured(2).toInt();
            const int component = QStringLiteral("xyzw").indexOf(match.captured(3));
            const int slot = vector * 4 + component;
            if (slotNames.contains(slot) && slotNames.value(slot) != name) {
                error = QObject::tr("Custom-uniform slot %1 is both %2 and %3 (%4)")
                            .arg(slot)
                            .arg(slotNames.value(slot), name, path);
                return false;
            }
            if (nameSlots.contains(name) && nameSlots.value(name) != slot) {
                error = QObject::tr("Custom uniform %1 uses slots %2 and %3 (%4)")
                            .arg(name)
                            .arg(nameSlots.value(name))
                            .arg(slot)
                            .arg(path);
                return false;
            }
            slotNames.insert(slot, name);
            nameSlots.insert(name, slot);
        }
        return true;
    }

    bool shader_name_less(const QString &left, const QString &right) {
        const int insensitive = QString::compare(left, right, Qt::CaseInsensitive);
        return insensitive == 0 ? left < right : insensitive < 0;
    }
} // namespace

namespace acmx2 {
    bool create_acmxvk_source_manifest(
        const QString &rootDirectory, const QString &requestedOutputPath,
        AcmxvkSourceManifestResult &result, QString &error) {
        result = {};
        error.clear();

        const QFileInfo rootInfo(rootDirectory);
        if (!rootInfo.exists() || !rootInfo.isDir()) {
            error = QObject::tr("Shader source directory does not exist: %1")
                        .arg(rootDirectory);
            return false;
        }
        const QString root = QDir(rootInfo.absoluteFilePath()).canonicalPath();
        const QString outputPath = requestedOutputPath.trimmed().isEmpty()
                                       ? QDir(root).filePath(QStringLiteral("library.json"))
                                       : QFileInfo(requestedOutputPath).absoluteFilePath();
        if (!QDir().mkpath(QFileInfo(outputPath).absolutePath())) {
            error = QObject::tr("Cannot create manifest directory: %1")
                        .arg(QFileInfo(outputPath).absolutePath());
            return false;
        }

        QHash<int, QString> slotNames = default_slot_names();
        QHash<QString, int> nameSlots;
        for (auto it = slotNames.constBegin(); it != slotNames.constEnd(); ++it)
            nameSlots.insert(it.value(), it.key());

        QStringList fragmentFiles;
        QStringList computeFiles;
        QDirIterator iterator(root, QDir::Files | QDir::Hidden,
                              QDirIterator::Subdirectories);
        while (iterator.hasNext()) {
            const QString absolutePath = iterator.next();
            QString relative = QDir(root).relativeFilePath(absolutePath);
            relative.replace(QLatin1Char('\\'), QLatin1Char('/'));
            const QStringList parts = relative.split(QLatin1Char('/'));
            const int directoryPartCount =
                std::max(0, static_cast<int>(parts.size()) - 1);
            const bool underCompute =
                parts.mid(0, directoryPartCount)
                    .contains(QStringLiteral("compute"));
            if (underCompute && relative.endsWith(QStringLiteral(".comp"))) {
                computeFiles.append(relative);
            } else if (!underCompute &&
                       relative.endsWith(QStringLiteral(".frag"))) {
                fragmentFiles.append(relative);
            } else {
                continue;
            }
            if (!scan_slots(absolutePath, slotNames, nameSlots, error))
                return false;
        }

        std::sort(fragmentFiles.begin(), fragmentFiles.end(), shader_name_less);
        std::sort(computeFiles.begin(), computeFiles.end(), shader_name_less);
        QStringList shaders = fragmentFiles;
        shaders.append(computeFiles);
        if (shaders.isEmpty()) {
            error = QObject::tr("No .frag or compute/*.comp sources found in %1")
                        .arg(root);
            return false;
        }
        if (shaders.size() > 16384) {
            error = QObject::tr("Shader library exceeds ACMXVK's 16384-entry limit");
            return false;
        }

        QHash<QString, QString> outputNames;
        for (const QString &shader : shaders) {
            const QString compiled = (shader + QStringLiteral(".spv")).toLower();
            if (outputNames.contains(compiled)) {
                error = QObject::tr("Case-insensitive duplicate shader output: %1.spv")
                            .arg(shader);
                return false;
            }
            outputNames.insert(compiled, shader);
        }

        int maximumSlot = 0;
        for (auto it = slotNames.constBegin(); it != slotNames.constEnd(); ++it)
            maximumSlot = std::max(maximumSlot, it.key());
        for (int slot = 0; slot <= maximumSlot; ++slot) {
            if (!slotNames.contains(slot)) {
                error = QObject::tr("No custom-uniform name was found for slot %1")
                            .arg(slot);
                return false;
            }
        }

        QHash<QString, UniformMetadata> metadata = default_metadata();
        if (!read_existing_metadata(outputPath, metadata, error))
            return false;

        QJsonObject customUniforms;
        for (int slot = 0; slot <= maximumSlot; ++slot) {
            const QString name = slotNames.value(slot);
            const UniformMetadata values = metadata.value(name);
            QJsonObject uniform;
            uniform.insert(QStringLiteral("slot"), slot);
            uniform.insert(QStringLiteral("minimum"), values.minimum);
            uniform.insert(QStringLiteral("maximum"), values.maximum);
            uniform.insert(QStringLiteral("step"), values.step);
            uniform.insert(QStringLiteral("value"), values.value);
            customUniforms.insert(name, uniform);
        }
        QJsonArray shaderArray;
        for (const QString &shader : shaders)
            shaderArray.append(shader);

        QJsonObject manifest;
        manifest.insert(QStringLiteral("version"), 1);
        manifest.insert(QStringLiteral("backend"), QStringLiteral("acmxvk"));
        manifest.insert(QStringLiteral("library_type"), QStringLiteral("source"));
        manifest.insert(QStringLiteral("custom_uniforms"), customUniforms);
        manifest.insert(QStringLiteral("shaders"), shaderArray);

        QSaveFile output(outputPath);
        if (!output.open(QIODevice::WriteOnly)) {
            error = QObject::tr("Cannot write %1: %2")
                        .arg(outputPath, output.errorString());
            return false;
        }
        const QByteArray json = QJsonDocument(manifest).toJson(QJsonDocument::Indented);
        if (output.write(json) != json.size() || !output.commit()) {
            error = QObject::tr("Cannot replace %1: %2")
                        .arg(outputPath, output.errorString());
            return false;
        }

        result.fragmentCount = fragmentFiles.size();
        result.computeCount = computeFiles.size();
        result.customUniformCount = maximumSlot + 1;
        result.outputPath = outputPath;
        return true;
    }
} // namespace acmx2
