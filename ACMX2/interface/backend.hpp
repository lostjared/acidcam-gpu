#ifndef ACMX2_INTERFACE_BACKEND_HPP
#define ACMX2_INTERFACE_BACKEND_HPP

#include <QString>
#include <optional>

namespace acmx2 {
    enum class Backend { Acmx2,
                         Acmxvk };

    inline QString backend_id(Backend backend) {
        return backend == Backend::Acmxvk ? QStringLiteral("acmxvk")
                                          : QStringLiteral("acmx2");
    }

    inline QString backend_name(Backend backend) {
        return backend == Backend::Acmxvk ? QStringLiteral("ACMXVK")
                                          : QStringLiteral("ACMX2");
    }

    inline QString backend_settings_key(Backend backend,
                                        const QString &setting) {
        return QStringLiteral("backend/%1/%2")
            .arg(backend_id(backend), setting);
    }

    inline QString default_backend_executable(Backend backend) {
#ifdef _WIN32
        return backend == Backend::Acmxvk ? QStringLiteral("acmxvk.exe")
                                          : QStringLiteral("acmx2.exe");
#else
        return backend == Backend::Acmxvk ? QStringLiteral("acmxvk")
                                          : QStringLiteral("acmx2");
#endif
    }

    inline std::optional<Backend> backend_from_id(const QString &value) {
        const QString normalized = value.trimmed().toLower();
        if (normalized == QStringLiteral("acmx2"))
            return Backend::Acmx2;
        if (normalized == QStringLiteral("acmxvk"))
            return Backend::Acmxvk;
        return std::nullopt;
    }
} // namespace acmx2

#endif
