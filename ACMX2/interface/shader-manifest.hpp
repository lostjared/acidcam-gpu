#ifndef SHADER_MANIFEST_HPP
#define SHADER_MANIFEST_HPP

/**
 * @file shader-manifest.hpp
 * @brief Optional JSON and legacy text shader-library manifests.
 */

#include "backend.hpp"

#include <QDateTime>
#include <QList>
#include <QString>
#include <QStringList>
#include <optional>

namespace acmx2 {
    struct CustomUniformDefinition {
        QString name;
        double minimum = 0.0;
        double maximum = 1.0;
        double step = 0.01;
        double value = 0.0;
    };

    enum class ShaderManifestFormat { Json,
                                      Text };
    enum class ShaderLibraryType { Source,
                                   Runtime };

    /// @brief Resolve library.json first, then fall back to index.txt.
    QString shader_manifest_path(const QString &directory);
    /// @brief Return true when either supported manifest exists.
    bool shader_manifest_exists(const QString &directory);
    /// @brief Return the selected manifest's modification time.
    QDateTime shader_manifest_last_modified(const QString &directory);
    /**
     * Read an optional top-level `backend` hint from library.json.
     *
     * A missing hint and a legacy index.txt both return std::nullopt with an
     * empty error. Invalid metadata returns std::nullopt with a non-empty
     * error.
     */
    std::optional<Backend> shader_manifest_backend(const QString &directory,
                                                   QString &error);
    /** Read and validate an optional top-level `library_type` value. */
    std::optional<ShaderLibraryType>
    shader_manifest_library_type(const QString &directory, QString &error);

    /** Load shader filenames from the preferred manifest in a directory. */
    bool load_shader_manifest(const QString &directory, QStringList &shaders,
                              QString &error);
    /** Rewrite the preferred manifest, preserving JSON fields when possible. */
    bool write_shader_manifest(const QString &directory, const QStringList &shaders,
                               QString &error);
    /** Add one shader to the preferred manifest if it is not already present. */
    bool append_shader_manifest(const QString &directory, const QString &shader,
                                QString &error);
    /** Create library.json from index.txt when a JSON manifest is absent. */
    bool migrate_index_manifest_to_json(const QString &directory, bool &created,
                                        QString &error);
    /** Create a new manifest in the requested format. */
    bool create_shader_manifest(const QString &directory, ShaderManifestFormat format,
                                const QStringList &shaders, QString &error);
    /** Load custom float-uniform controls from library.json. */
    bool load_custom_uniforms(const QString &directory,
                              QList<CustomUniformDefinition> &uniforms,
                              QString &error);
    /** Rewrite custom float-uniform controls while preserving all other JSON fields. */
    bool write_custom_uniforms(const QString &directory,
                               const QList<CustomUniformDefinition> &uniforms,
                               QString &error);
} // namespace acmx2

#endif
