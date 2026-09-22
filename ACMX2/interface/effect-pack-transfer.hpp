#ifndef ACMX2_EFFECT_PACK_TRANSFER_HPP
#define ACMX2_EFFECT_PACK_TRANSFER_HPP

#include <QJsonObject>
#include <QString>
#include <functional>

namespace acmx2 {
    struct EffectPackTransferRequest {
        QString source_root;
        QString destination_root;
        QString folder_name;
        QJsonObject manifest;
        QString external_icon;
        bool infer_requirements = false;
        bool assign_new_id = false;
    };

    struct EffectPackTransferResult {
        QString destination;
        QString error;
        bool success = false;
    };

    using EffectPackTransferProgress = std::function<void(int current, int total)>;
    EffectPackTransferResult transfer_effect_pack(const EffectPackTransferRequest &request, const EffectPackTransferProgress &progress = {});
} // namespace acmx2

#endif
