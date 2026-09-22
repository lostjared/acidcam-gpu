#include "effect-pack-controls.hpp"

#include <QApplication>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>
#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QTemporaryDir temporary;
    if (!temporary.isValid()) {
        return 1;
    }
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
    QApplication app(argc, argv);

    EffectPackControls dialog;
    const QVector<EffectPackControlDefinition> first = {
        {"symmetry", "Symmetry", "u_symmetry", 1.0, 16.0, 1.0, 6.0},
        {"depth", "Depth", "u_depth", 0.0, 1.0, 0.01, 0.2},
    };
    const QVector<EffectPackControlDefinition> second = {{"speed", "Speed", "u_speed", 0.0, 2.0, 0.1, 1.0}};
    int updates = 0;
    QObject::connect(&dialog, &EffectPackControls::values_changed, &app, [&updates](const QVector<EffectPackUniformValue> &) { ++updates; });
    dialog.set_pack("org.acmxvk.test.first", "First", first);
    auto spins = dialog.findChildren<QDoubleSpinBox *>();
    if (spins.size() != 2) {
        return 2;
    }
    spins[0]->setValue(9.0);
    spins[1]->setValue(0.75);
    dialog.set_pack("org.acmxvk.test.second", "Second", second);
    dialog.set_pack("org.acmxvk.test.first", "First", first);
    const auto restored = dialog.values();
    if (restored.size() != 2 || restored[0].name != "u_symmetry" || restored[0].value != 9.0 || std::abs(restored[1].value - 0.75) > 0.000001) {
        std::cerr << "per-pack controls were not restored\n";
        return 3;
    }
    for (QPushButton *button : dialog.findChildren<QPushButton *>()) {
        if (button->text() == "Reset") {
            button->click();
            break;
        }
    }
    const auto single_reset = dialog.values();
    if (single_reset[0].value != 6.0 || std::abs(single_reset[1].value - 0.75) > 0.000001) {
        std::cerr << "single-control reset changed the wrong values\n";
        return 4;
    }
    for (QPushButton *button : dialog.findChildren<QPushButton *>()) {
        if (button->text() == "Reset Pack to Defaults") {
            button->click();
            break;
        }
    }
    const auto reset = dialog.values();
    if (reset.size() != 2 || reset[0].value != 6.0 || std::abs(reset[1].value - 0.2) > 0.000001 || updates < 3) {
        std::cerr << "reset did not restore defaults and publish values\n";
        return 5;
    }
    dialog.set_pack("org.acmxvk.test.second", "Second", second);
    dialog.set_pack("org.acmxvk.test.first", "First", first);
    const auto persisted_reset = dialog.values();
    if (persisted_reset[0].value != 6.0 || std::abs(persisted_reset[1].value - 0.2) > 0.000001) {
        std::cerr << "reset defaults were not persisted\n";
        return 6;
    }
    dialog.set_pack("org.acmxvk.test.first", "First", first, false);
    dialog.set_project_values(QJsonObject{{QStringLiteral("symmetry"), 12.0}});
    if (dialog.values()[0].value != 12.0 || dialog.values()[1].value != 0.2) {
        std::cerr << "project override did not take precedence over defaults\n";
        return 7;
    }
    dialog.findChildren<QDoubleSpinBox *>()[0]->setValue(14.0);
    dialog.set_pack("org.acmxvk.test.second", "Second", second);
    dialog.set_pack("org.acmxvk.test.first", "First", first);
    if (dialog.values()[0].value != 6.0) {
        std::cerr << "project override leaked into the user's saved pack state\n";
        return 8;
    }
    return 0;
}
