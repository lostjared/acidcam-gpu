#ifndef ACMX2_EFFECT_PACK_PROJECT_HPP
#define ACMX2_EFFECT_PACK_PROJECT_HPP

#include <QJsonObject>
#include <QString>

struct EffectPackProjectState {
    QString manifest_path;
    QString id;
    QJsonObject values;
    QString dream_model_file;
};

namespace acmx2 {
    bool validate_effect_pack_project_cache(const QString &manifest_path, const QString &expected_id, QString &error);
    bool bundle_effect_pack_project(const EffectPackProjectState &state, const QString &project_root, QString &relative_manifest, QString &error);
    bool resolve_effect_pack_project_state(const QJsonObject &object, const QString &project_root, EffectPackProjectState &state, QString &error);
} // namespace acmx2

#endif
