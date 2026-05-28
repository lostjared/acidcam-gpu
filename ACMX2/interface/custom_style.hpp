#pragma once

#include <QSettings>
#include <QString>
#include <QWidget>

namespace acmx2 {

struct CustomStylePalette {
    QString windowBg;
    QString windowFg;
    QString accent;
    QString fieldBg;
    QString fieldFg;
    QString fieldBorder;
    QString buttonBg;
    QString buttonHover;
    QString buttonFg;
    QString menuBg;
    QString menuFg;
    QString menuSelBg;
    QString menuSelFg;
    QString selectionBg;
    QString border; // full CSS value, e.g. "3px solid red" or "none"
};

inline QString buildStyleSheet(const CustomStylePalette &p) {
    const QString border = p.border.isEmpty() ? QStringLiteral("none") : p.border;
    return QString(
               "QMainWindow, QDialog { background-color: %1; color: %2; border: %15; }"
               "QWidget { background-color: %1; color: %2; }"
               "QLabel, QCheckBox, QRadioButton, QGroupBox, QStatusBar { color: %2; background: transparent; }"
               "QGroupBox { border: 1px solid %3; margin-top: 12px; padding-top: 10px; }"
               "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: %2; }"
               "QMenuBar { background-color: %9; color: %10; }"
               "QMenuBar::item:selected { background-color: %11; color: %12; }"
               "QMenu { background-color: %9; color: %10; border: 1px solid %3; }"
               "QMenu::item:selected { background-color: %11; color: %12; }"
               "QToolTip { background-color: %4; color: %5; border: 1px solid %3; }"
               "QPushButton {"
               " background-color: %6;"
               " color: %8;"
               " border: 1px solid %13;"
               " border-radius: 4px;"
               " padding: 4px 10px;"
               " font-weight: 600;"
               " font-size: 10px;"
               " min-width: 60px;"
               " min-height: 20px;"
               " max-height: 30px;"
               " }"
               "QPushButton:hover { background-color: %7; color: %8; border: 1px solid %3; }"
               "QPushButton:disabled { color: %5; border-color: %13; background-color: %4; }"
               "QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QTreeWidget, QTableWidget,"
               " QComboBox, QSpinBox, QDoubleSpinBox {"
               " background-color: %4; color: %5; border: 1px solid %13;"
               " selection-background-color: %14; selection-color: %5; }"
               "QComboBox QAbstractItemView { background-color: %4; color: %5; border: 1px solid %13;"
               " selection-background-color: %14; selection-color: %5; }"
               "QHeaderView::section { background-color: %6; color: %8; border: 1px solid %13; padding: 4px; }"
               "QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {"
               " background-color: %14; color: %5; }"
               "QTabWidget::pane { border: 1px solid %3; }"
               "QTabBar::tab { background-color: %6; color: %8; border: 1px solid %3; padding: 4px 10px; }"
               "QTabBar::tab:selected { background-color: %7; color: %8; }"
               "QCheckBox::indicator, QRadioButton::indicator { width: 14px; height: 14px; }"
               "QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {"
               " background-color: %4; border: 1px solid %3; }"
               "QCheckBox::indicator:checked, QRadioButton::indicator:checked {"
               " background-color: %3; border: 1px solid %3; }"
               "QSlider::groove:horizontal { background: %4; height: 6px; border: 1px solid %3; }"
               "QSlider::handle:horizontal { background: %3; width: 14px; margin: -5px 0; border: 1px solid %3; }"
               "QScrollBar:vertical { background: %4; width: 12px; border: 1px solid %3; }"
               "QScrollBar::handle:vertical { background: %6; min-height: 24px; border: 1px solid %3; }"
               "QScrollBar:horizontal { background: %4; height: 12px; border: 1px solid %3; }"
               "QScrollBar::handle:horizontal { background: %6; min-width: 24px; border: 1px solid %3; }"
               "QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; background: none; border: none; }"
               "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }")
        .arg(p.windowBg,    // 1
             p.windowFg,    // 2
             p.accent,      // 3
             p.fieldBg,     // 4
             p.fieldFg,     // 5
             p.buttonBg,    // 6
             p.buttonHover, // 7
             p.buttonFg,    // 8
             p.menuBg)      // 9
        .arg(p.menuFg)      // 10
        .arg(p.menuSelBg)   // 11
        .arg(p.menuSelFg)   // 12
        .arg(p.fieldBorder) // 13
        .arg(p.selectionBg) // 14
        .arg(border);       // 15
}

inline QString defaultCustomStyleSheet() {
    CustomStylePalette p;
    p.windowBg = "#000000";
    p.windowFg = "#ff3030";
    p.accent = "#ff0000";
    p.fieldBg = "#110000";
    p.fieldFg = "#ff5555";
    p.fieldBorder = "#ff0000";
    p.buttonBg = "#110000";
    p.buttonHover = "#ff0000";
    p.buttonFg = "#ff5555";
    p.menuBg = "#110000";
    p.menuFg = "#ff5555";
    p.menuSelBg = "#ff0000";
    p.menuSelFg = "#000000";
    p.selectionBg = "#400000";
    p.border = "3px solid #ff0000";
    return buildStyleSheet(p);
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
