#ifndef ACMX2_INTERFACE_EFFECT_PACK_MODELS_HPP
#define ACMX2_INTERFACE_EFFECT_PACK_MODELS_HPP

#include <QString>
#include <QStringList>

namespace acmx2 {
    QStringList effect_pack_model_roots(const QString &configured_model);
    QString resolve_effect_pack_model(const QString &model_id, const QStringList &roots, const QString &override_path);
} // namespace acmx2

#endif
