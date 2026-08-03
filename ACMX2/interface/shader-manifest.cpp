#include "shader-manifest.hpp"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>
#include <QTextStream>

namespace {
    constexpr auto JSON_MANIFEST_NAME = "library.json";
    constexpr auto TEXT_MANIFEST_NAME = "index.txt";

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
} // namespace acmx2
