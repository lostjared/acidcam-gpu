#pragma once

#include <QSettings>
#include <QString>
#include <QWidget>

namespace acmx2 {

inline QString defaultCustomStyleSheet() {
    return QStringLiteral(
        "QMainWindow, QDialog { background-color: black; border: 3px solid red; }"
        "* { color: red; font-weight: bold; } "
        "QPushButton { border: 1px solid red; background-color: #110000; padding: 5px; }"
        "QPushButton:hover { background-color: red; color: black; }");
}

inline bool isCustomStyleEnabled() {
    QSettings appSettings("LostSideDead");
    return appSettings.value("useCustomStyle", false).toBool();
}

inline QString resolvedCustomStyleSheet() {
    QSettings appSettings("LostSideDead");
    const QString stored = appSettings.value("customStyleSheet", defaultCustomStyleSheet()).toString();
    if (stored.trimmed().isEmpty()) {
        return defaultCustomStyleSheet();
    }
    return stored;
}

inline void applyCustomStyleIfEnabled(QWidget *widget) {
    if (!widget || !isCustomStyleEnabled()) {
        return;
    }
    widget->setStyleSheet(resolvedCustomStyleSheet());
}

} // namespace acmx2
