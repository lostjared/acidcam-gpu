#include "shader-manifest.hpp"
#include "../shader_selection_shm.hpp"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTextStream>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
    constexpr auto JSON_MANIFEST_NAME = "library.json";
    constexpr auto TEXT_MANIFEST_NAME = "index.txt";

    bool valid_custom_uniform_name(const QString &name) {
        static const QRegularExpression identifier(
            QStringLiteral("^[A-Za-z_][A-Za-z0-9_]*$"));
        return identifier.match(name).hasMatch() && !name.startsWith("gl_") &&
               name.toUtf8().size() <
                   static_cast<int>(acmx2::ipc::kShaderSelectionMaxUniformName);
    }

    QString json_entry_file(const QJsonValue &value) {
        if (value.isString())
            return value.toString().trimmed();
        if (value.isObject())
            return value.toObject().value("file").toString().trimmed();
        return {};
    }

    bool load_json_document(const QString &path, QJsonDocument &document,
                            QString &error) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            error = QObject::tr("Could not open %1: %2")
                        .arg(path, file.errorString());
            return false;
        }

        QJsonParseError parseError;
        document = QJsonDocument::fromJson(file.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            error = QObject::tr("Could not parse %1 at offset %2: %3")
                        .arg(path)
                        .arg(parseError.offset)
                        .arg(parseError.errorString());
            return false;
        }
        if (!document.object().value("shaders").isArray()) {
            error = QObject::tr("%1 must contain a 'shaders' array.").arg(path);
            return false;
        }
        return true;
    }

    bool write_json_document(const QString &path, const QJsonDocument &document,
                             QString &error) {
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly)) {
            error = QObject::tr("Could not write %1: %2")
                        .arg(path, file.errorString());
            return false;
        }
        const QByteArray json = document.toJson(QJsonDocument::Indented);
        if (file.write(json) != json.size() ||
            !file.commit()) {
            error = QObject::tr("Could not finish writing %1: %2")
                        .arg(path, file.errorString());
            return false;
        }
        return true;
    }
} // namespace

namespace acmx2 {
    QString shader_manifest_path(const QString &directory) {
        const QString jsonPath = directory + "/" + JSON_MANIFEST_NAME;
        if (QFileInfo(jsonPath).isFile())
            return jsonPath;
        const QString textPath = directory + "/" + TEXT_MANIFEST_NAME;
        if (QFileInfo(textPath).isFile())
            return textPath;
        return {};
    }

    bool shader_manifest_exists(const QString &directory) {
        return !shader_manifest_path(directory).isEmpty();
    }

    QDateTime shader_manifest_last_modified(const QString &directory) {
        return QFileInfo(shader_manifest_path(directory)).lastModified();
    }

    std::optional<Backend> shader_manifest_backend(const QString &directory,
                                                   QString &error) {
        error.clear();
        const QString path = shader_manifest_path(directory);
        if (path.isEmpty()) {
            error = QObject::tr("No library.json or index.txt found in %1.")
                        .arg(directory);
            return std::nullopt;
        }
        if (!path.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive))
            return std::nullopt;

        QJsonDocument document;
        if (!load_json_document(path, document, error))
            return std::nullopt;
        const QJsonValue value = document.object().value(QStringLiteral("backend"));
        if (value.isUndefined() || value.isNull())
            return std::nullopt;
        if (!value.isString()) {
            error = QObject::tr("%1 has a non-string 'backend' value.").arg(path);
            return std::nullopt;
        }
        const std::optional<Backend> backend = backend_from_id(value.toString());
        if (!backend) {
            error = QObject::tr("%1 has an unsupported backend '%2'.")
                        .arg(path, value.toString());
        }
        return backend;
    }

    std::optional<ShaderLibraryType>
    shader_manifest_library_type(const QString &directory, QString &error) {
        error.clear();
        const QString path = shader_manifest_path(directory);
        if (path.isEmpty()) {
            error = QObject::tr("No library.json or index.txt found in %1.")
                        .arg(directory);
            return std::nullopt;
        }
        if (!path.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive))
            return std::nullopt;

        QJsonDocument document;
        if (!load_json_document(path, document, error))
            return std::nullopt;
        const QJsonValue value =
            document.object().value(QStringLiteral("library_type"));
        if (value.isUndefined() || value.isNull())
            return std::nullopt;
        if (!value.isString()) {
            error = QObject::tr("%1 has a non-string 'library_type' value.")
                        .arg(path);
            return std::nullopt;
        }
        const QString type = value.toString().trimmed().toLower();
        if (type == QStringLiteral("source"))
            return ShaderLibraryType::Source;
        if (type == QStringLiteral("runtime"))
            return ShaderLibraryType::Runtime;
        error = QObject::tr("%1 has an unsupported library_type '%2'.")
                    .arg(path, value.toString());
        return std::nullopt;
    }

    bool load_shader_manifest(const QString &directory, QStringList &shaders,
                              QString &error) {
        shaders.clear();
        error.clear();
        const QString path = shader_manifest_path(directory);
        if (path.isEmpty()) {
            error = QObject::tr("No library.json or index.txt found in %1.")
                        .arg(directory);
            return false;
        }

        if (path.endsWith(".json", Qt::CaseInsensitive)) {
            QJsonDocument document;
            if (!load_json_document(path, document, error))
                return false;
            const QJsonArray entries = document.object().value("shaders").toArray();
            for (const QJsonValue &entry : entries) {
                const QString file = json_entry_file(entry);
                if (file.isEmpty()) {
                    error = QObject::tr("%1 contains a shader entry without a file name.")
                                .arg(path);
                    return false;
                }
                shaders.append(file);
            }
            return true;
        }

        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            error = QObject::tr("Could not open %1: %2")
                        .arg(path, file.errorString());
            return false;
        }
        QTextStream input(&file);
        while (!input.atEnd())
            shaders.append(input.readLine().trimmed());
        return true;
    }

    bool write_shader_manifest(const QString &directory, const QStringList &shaders,
                               QString &error) {
        error.clear();
        const QString path = shader_manifest_path(directory);
        if (path.isEmpty()) {
            error = QObject::tr("No library.json or index.txt found in %1.")
                        .arg(directory);
            return false;
        }

        if (path.endsWith(".json", Qt::CaseInsensitive)) {
            QJsonDocument document;
            if (!load_json_document(path, document, error))
                return false;

            QJsonObject root = document.object();
            const QJsonArray previousEntries = root.value("shaders").toArray();
            QJsonArray updatedEntries;
            for (const QString &shader : shaders) {
                QJsonValue preservedEntry;
                bool found = false;
                for (const QJsonValue &entry : previousEntries) {
                    if (json_entry_file(entry).compare(shader, Qt::CaseInsensitive) == 0) {
                        preservedEntry = entry;
                        found = true;
                        break;
                    }
                }
                updatedEntries.append(found ? preservedEntry : QJsonValue(shader));
            }
            if (!root.contains("version"))
                root.insert("version", 1);
            root.insert("shaders", updatedEntries);
            return write_json_document(path, QJsonDocument(root), error);
        }

        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            error = QObject::tr("Could not write %1: %2")
                        .arg(path, file.errorString());
            return false;
        }
        QTextStream output(&file);
        for (const QString &shader : shaders)
            output << shader << "\n";
        output.flush();
        if (output.status() != QTextStream::Ok || !file.commit()) {
            error = QObject::tr("Could not finish writing %1: %2")
                        .arg(path, file.errorString());
            return false;
        }
        return true;
    }

    bool append_shader_manifest(const QString &directory, const QString &shader,
                                QString &error) {
        QStringList shaders;
        if (!load_shader_manifest(directory, shaders, error))
            return false;
        if (!shaders.contains(shader, Qt::CaseInsensitive))
            shaders.append(shader);
        return write_shader_manifest(directory, shaders, error);
    }

    bool migrate_index_manifest_to_json(const QString &directory, bool &created,
                                        QString &error) {
        created = false;
        error.clear();

        const QString jsonPath = directory + "/" + JSON_MANIFEST_NAME;
        if (QFileInfo(jsonPath).isFile())
            return true;

        const QString textPath = directory + "/" + TEXT_MANIFEST_NAME;
        QFile file(textPath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            error = QObject::tr("Could not open %1: %2")
                        .arg(textPath, file.errorString());
            return false;
        }

        QStringList shaders;
        QTextStream input(&file);
        while (!input.atEnd()) {
            const QString shader = input.readLine().trimmed();
            if (!shader.isEmpty())
                shaders.append(shader);
        }

        if (!create_shader_manifest(directory, ShaderManifestFormat::Json,
                                    shaders, error))
            return false;
        created = true;
        return true;
    }

    bool create_shader_manifest(const QString &directory, ShaderManifestFormat format,
                                const QStringList &shaders, QString &error) {
        error.clear();
        if (format == ShaderManifestFormat::Json) {
            QJsonArray entries;
            for (const QString &shader : shaders)
                entries.append(shader);
            QJsonObject root;
            root.insert("version", 1);
            root.insert("backend", QStringLiteral("acmx2"));
            root.insert("library_type", QStringLiteral("source"));
            root.insert("shaders", entries);
            return write_json_document(directory + "/" + JSON_MANIFEST_NAME,
                                       QJsonDocument(root), error);
        }

        QSaveFile file(directory + "/" + TEXT_MANIFEST_NAME);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            error = QObject::tr("Could not create index.txt: %1")
                        .arg(file.errorString());
            return false;
        }
        QTextStream output(&file);
        for (const QString &shader : shaders)
            output << shader << "\n";
        output.flush();
        if (output.status() != QTextStream::Ok || !file.commit()) {
            error = QObject::tr("Could not finish creating index.txt: %1")
                        .arg(file.errorString());
            return false;
        }
        return true;
    }

    bool load_custom_uniforms(const QString &directory,
                              QList<CustomUniformDefinition> &uniforms,
                              QString &error) {
        uniforms.clear();
        error.clear();
        const QString path = directory + "/" + JSON_MANIFEST_NAME;
        if (!QFileInfo(path).isFile()) {
            error = QObject::tr("Custom uniforms require %1.").arg(path);
            return false;
        }

        QJsonDocument document;
        if (!load_json_document(path, document, error))
            return false;

        const QJsonValue value = document.object().value("custom_uniforms");
        if (value.isUndefined() || value.isNull())
            return true;
        if (!value.isObject()) {
            error = QObject::tr("%1 field 'custom_uniforms' must be an object.")
                        .arg(path);
            return false;
        }

        const QJsonObject entries = value.toObject();
        bool hasExplicitSlots = false;
        bool hasImplicitSlots = false;
        std::vector<int> occupiedSlots;
        for (auto it = entries.constBegin(); it != entries.constEnd(); ++it) {
            if (!it.value().isObject()) {
                error = QObject::tr("Custom uniform '%1' must be an object.")
                            .arg(it.key());
                return false;
            }
            const QJsonObject entry = it.value().toObject();
            CustomUniformDefinition uniform;
            uniform.name = it.key();
            if (!valid_custom_uniform_name(uniform.name)) {
                error = QObject::tr("Custom uniform '%1' is not a valid GLSL identifier.")
                            .arg(uniform.name);
                return false;
            }
            uniform.minimum = entry.value("minimum").toDouble(0.0);
            uniform.maximum = entry.value("maximum").toDouble(1.0);
            uniform.step = entry.value("step").toDouble(0.01);
            uniform.value = entry.value("value").toDouble(uniform.minimum);
            const QJsonValue slotValue = entry.value("slot");
            if (!slotValue.isUndefined() && !slotValue.isNull()) {
                const double numericSlot = slotValue.toDouble(-1.0);
                uniform.slot = static_cast<int>(numericSlot);
                if (!slotValue.isDouble() || numericSlot != uniform.slot ||
                    uniform.slot < 0 ||
                    uniform.slot >= static_cast<int>(
                                        ipc::kShaderSelectionMaxCustomUniforms) ||
                    std::find(occupiedSlots.begin(), occupiedSlots.end(),
                              uniform.slot) != occupiedSlots.end()) {
                    error = QObject::tr(
                                "Custom uniform '%1' has an invalid or duplicate slot.")
                                .arg(uniform.name);
                    return false;
                }
                occupiedSlots.push_back(uniform.slot);
                hasExplicitSlots = true;
            } else {
                hasImplicitSlots = true;
            }
            if (!std::isfinite(uniform.minimum) ||
                !std::isfinite(uniform.maximum) ||
                !std::isfinite(uniform.step) ||
                !std::isfinite(uniform.value) ||
                uniform.maximum <= uniform.minimum || uniform.step <= 0.0) {
                error = QObject::tr("Custom uniform '%1' has an invalid range or value.")
                            .arg(uniform.name);
                return false;
            }
            uniform.value = std::clamp(uniform.value, uniform.minimum,
                                       uniform.maximum);
            uniforms.append(uniform);
        }
        if (hasExplicitSlots && hasImplicitSlots) {
            error = QObject::tr(
                "Custom uniforms must either all specify slots or all omit them.");
            return false;
        }
        if (hasExplicitSlots) {
            std::sort(uniforms.begin(), uniforms.end(),
                      [](const CustomUniformDefinition &left,
                         const CustomUniformDefinition &right) {
                          return left.slot < right.slot;
                      });
            for (int index = 0; index < uniforms.size(); ++index) {
                if (uniforms.at(index).slot != index) {
                    error = QObject::tr(
                        "Custom-uniform slots must be contiguous from zero.");
                    return false;
                }
            }
        }
        return true;
    }

    bool write_custom_uniforms(const QString &directory,
                               const QList<CustomUniformDefinition> &uniforms,
                               QString &error) {
        error.clear();
        const QString path = directory + "/" + JSON_MANIFEST_NAME;
        QJsonDocument document;
        if (!load_json_document(path, document, error))
            return false;

        QJsonObject entries;
        const bool hasExplicitSlots = std::any_of(
            uniforms.cbegin(), uniforms.cend(),
            [](const CustomUniformDefinition &uniform) {
                return uniform.slot >= 0;
            });
        std::vector<int> writtenSlots;
        for (const CustomUniformDefinition &uniform : uniforms) {
            if ((hasExplicitSlots && uniform.slot < 0) ||
                (!hasExplicitSlots && uniform.slot >= 0)) {
                error = QObject::tr(
                    "Custom uniforms must either all specify slots or all omit them.");
                return false;
            }
            if (hasExplicitSlots &&
                (uniform.slot >= static_cast<int>(
                                     ipc::kShaderSelectionMaxCustomUniforms) ||
                 std::find(writtenSlots.begin(), writtenSlots.end(),
                           uniform.slot) != writtenSlots.end())) {
                error = QObject::tr(
                            "Custom uniform '%1' has an invalid or duplicate slot.")
                            .arg(uniform.name);
                return false;
            }
            if (hasExplicitSlots)
                writtenSlots.push_back(uniform.slot);
            QJsonObject entry;
            if (hasExplicitSlots)
                entry.insert("slot", uniform.slot);
            entry.insert("minimum", uniform.minimum);
            entry.insert("maximum", uniform.maximum);
            entry.insert("step", uniform.step);
            entry.insert("value", std::clamp(uniform.value, uniform.minimum,
                                             uniform.maximum));
            entries.insert(uniform.name, entry);
        }
        if (hasExplicitSlots) {
            std::sort(writtenSlots.begin(), writtenSlots.end());
            for (std::size_t index = 0; index < writtenSlots.size(); ++index) {
                if (writtenSlots[index] != static_cast<int>(index)) {
                    error = QObject::tr(
                        "Custom-uniform slots must be contiguous from zero.");
                    return false;
                }
            }
        }

        QJsonObject root = document.object();
        if (entries.isEmpty())
            root.remove("custom_uniforms");
        else
            root.insert("custom_uniforms", entries);
        return write_json_document(path, QJsonDocument(root), error);
    }
} // namespace acmx2
