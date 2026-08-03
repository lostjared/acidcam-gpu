#ifndef SHADER_MANIFEST_HPP
#define SHADER_MANIFEST_HPP

/**
 * @file shader-manifest.hpp
 * @brief Optional JSON and legacy text shader-library manifests.
 */

#include <QDateTime>
#include <QString>
#include <QStringList>

namespace acmx2 {
    enum class ShaderManifestFormat { Json,
                                      Text };

    /// @brief Resolve library.json first, then fall back to index.txt.
    QString shader_manifest_path(const QString &directory);
    /// @brief Return true when either supported manifest exists.
    bool shader_manifest_exists(const QString &directory);
    /// @brief Return the selected manifest's modification time.
    QDateTime shader_manifest_last_modified(const QString &directory);

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
} // namespace acmx2

#endif
