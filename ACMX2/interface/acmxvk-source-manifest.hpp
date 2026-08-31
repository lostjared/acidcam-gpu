#ifndef ACMXVK_SOURCE_MANIFEST_HPP
#define ACMXVK_SOURCE_MANIFEST_HPP

#include <QString>
#include <QStringList>

namespace acmx2 {
    struct AcmxvkSourceManifestResult {
        int fragmentCount = 0;
        int computeCount = 0;
        int customUniformCount = 0;
        QString outputPath;
    };

    /**
     * Scan an ACMXVK source tree and atomically generate its library.json.
     *
     * Fragment shaders are discovered recursively outside directories named
     * `compute`; compute shaders are discovered beneath a `compute` directory.
     * Existing custom-uniform ranges and values are preserved.
     */
    bool create_acmxvk_source_manifest(
        const QString &rootDirectory, const QString &outputPath,
        AcmxvkSourceManifestResult &result, QString &error);

    /**
     * Generate a source manifest from an explicit subset of files below root.
     * This preserves intentional manifest removals while adding a new shader.
     */
    bool create_acmxvk_source_manifest_for_shaders(
        const QString &rootDirectory, const QStringList &shaderFiles,
        const QString &outputPath, AcmxvkSourceManifestResult &result,
        QString &error);
} // namespace acmx2

#endif
