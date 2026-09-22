#include "effect-pack-controls.hpp"

#include <QCryptographicHash>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

namespace {
    QString settings_key(const QString &id) { return QStringLiteral("effect_packs/values/") + QString::fromLatin1(QCryptographicHash::hash(id.toUtf8(), QCryptographicHash::Sha256).toHex()); }

    int slider_steps(const EffectPackControlDefinition &control) { return static_cast<int>(std::clamp(std::ceil((control.maximum - control.minimum) / control.step), 1.0, 1000000.0)); }

    double slider_value(const EffectPackControlDefinition &control, int position) {
        const int steps = slider_steps(control);
        if (position >= steps) {
            return control.maximum;
        }
        const double increment = std::ceil((control.maximum - control.minimum) / control.step) <= 1000000.0 ? control.step : (control.maximum - control.minimum) / steps;
        return std::clamp(control.minimum + increment * position, control.minimum, control.maximum);
    }

    int slider_position(const EffectPackControlDefinition &control, double value) {
        const int steps = slider_steps(control);
        const double increment = std::ceil((control.maximum - control.minimum) / control.step) <= 1000000.0 ? control.step : (control.maximum - control.minimum) / steps;
        return static_cast<int>(std::clamp(std::round((value - control.minimum) / increment), 0.0, static_cast<double>(steps)));
    }
} // namespace

EffectPackControls::EffectPackControls(QWidget *parent) : QDialog(parent) {
    setWindowTitle(tr("Effect Pack Controls"));
    setModal(false);
    resize(680, 520);
    save_timer = new QTimer(this);
    save_timer->setSingleShot(true);
    save_timer->setInterval(150);
    connect(save_timer, &QTimer::timeout, this, &EffectPackControls::save_values);
    auto *layout = new QVBoxLayout(this);
    auto *hint = new QLabel(tr("Control values are saved separately for each effect pack. Changes are sent live to ACMXVK."), this);
    hint->setWordWrap(true);
    layout->addWidget(hint);
    auto *scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    rows_widget = new QWidget(scroll);
    rows_layout = new QVBoxLayout(rows_widget);
    rows_layout->setAlignment(Qt::AlignTop);
    scroll->setWidget(rows_widget);
    layout->addWidget(scroll, 1);
    auto *reset_all = new QPushButton(tr("Reset Pack to Defaults"), this);
    layout->addWidget(reset_all);
    connect(reset_all, &QPushButton::clicked, this, [this]() {
        for (int index = 0; index < controls.size(); ++index) {
            control_values[index] = controls[index].default_value;
        }
        rebuild();
        save_timer->start();
        emit values_changed(values());
    });
}

EffectPackControls::~EffectPackControls() { save_values(); }

void EffectPackControls::set_pack(const QString &id, const QString &name, const QVector<EffectPackControlDefinition> &definitions, bool persist_user_values) {
    if (save_timer->isActive()) {
        save_timer->stop();
        save_values();
    }
    pack_id = id;
    pack_name = name;
    controls = definitions;
    this->persist_user_values = persist_user_values;
    control_values.clear();
    QSettings settings("LostSideDead", "acmx2");
    const QVariantMap saved = persist_user_values ? settings.value(settings_key(pack_id)).toMap() : QVariantMap{};
    for (const EffectPackControlDefinition &control : controls) {
        bool valid = false;
        const double restored = saved.value(control.id).toDouble(&valid);
        control_values.push_back(valid && std::isfinite(restored) ? std::clamp(restored, control.minimum, control.maximum) : control.default_value);
    }
    setWindowTitle(tr("%1 — Effect Pack Controls").arg(pack_name));
    rebuild();
}

void EffectPackControls::set_project_values(const QJsonObject &values) {
    for (int index = 0; index < controls.size(); ++index) {
        const QJsonValue value = values.value(controls[index].id);
        if (value.isDouble() && std::isfinite(value.toDouble())) {
            control_values[index] = std::clamp(value.toDouble(), controls[index].minimum, controls[index].maximum);
        }
    }
    rebuild();
}

QJsonObject EffectPackControls::project_values() const {
    QJsonObject result;
    for (int index = 0; index < controls.size(); ++index) {
        result.insert(controls[index].id, control_values[index]);
    }
    return result;
}

QVector<EffectPackUniformValue> EffectPackControls::values() const {
    QVector<EffectPackUniformValue> result;
    result.reserve(controls.size());
    for (int index = 0; index < controls.size(); ++index) {
        result.push_back({controls[index].uniform, control_values[index]});
    }
    return result;
}

void EffectPackControls::set_value(int index, double value) {
    if (index < 0 || index >= controls.size()) {
        return;
    }
    const double bounded = std::clamp(value, controls[index].minimum, controls[index].maximum);
    if (control_values[index] == bounded) {
        return;
    }
    control_values[index] = bounded;
    save_timer->start();
    emit values_changed(values());
}

void EffectPackControls::save_values() const {
    if (pack_id.isEmpty() || !persist_user_values) {
        return;
    }
    QVariantMap saved;
    for (int index = 0; index < controls.size(); ++index) {
        saved.insert(controls[index].id, control_values[index]);
    }
    QSettings settings("LostSideDead", "acmx2");
    settings.setValue(settings_key(pack_id), saved);
}

void EffectPackControls::rebuild() {
    while (QLayoutItem *item = rows_layout->takeAt(0)) {
        if (QWidget *widget = item->widget()) {
            delete widget;
        }
        delete item;
    }
    if (controls.isEmpty()) {
        rows_layout->addWidget(new QLabel(tr("This pack has no custom controls."), rows_widget));
    }
    for (int index = 0; index < controls.size(); ++index) {
        const EffectPackControlDefinition control = controls[index];
        auto *row = new QWidget(rows_widget);
        auto *layout = new QGridLayout(row);
        auto *label = new QLabel(control.label, row);
        label->setToolTip(tr("GLSL uniform: %1").arg(control.uniform));
        auto *slider = new QSlider(Qt::Horizontal, row);
        slider->setRange(0, slider_steps(control));
        slider->setValue(slider_position(control, control_values[index]));
        auto *spin = new QDoubleSpinBox(row);
        spin->setDecimals(8);
        spin->setRange(control.minimum, control.maximum);
        spin->setSingleStep(control.step);
        spin->setKeyboardTracking(false);
        spin->setValue(control_values[index]);
        auto *reset = new QPushButton(tr("Reset"), row);
        reset->setAutoDefault(false);
        auto *description = new QLabel(tr("%1 · %2 to %3").arg(control.uniform).arg(control.minimum, 0, 'g', 8).arg(control.maximum, 0, 'g', 8), row);
        description->setTextInteractionFlags(Qt::TextSelectableByMouse);
        layout->addWidget(label, 0, 0);
        layout->addWidget(slider, 0, 1);
        layout->addWidget(spin, 0, 2);
        layout->addWidget(reset, 0, 3);
        layout->addWidget(description, 1, 1, 1, 2);
        layout->setColumnStretch(1, 1);
        rows_layout->addWidget(row);
        connect(slider, &QSlider::valueChanged, this, [this, control, index, spin](int position) {
            const double next = slider_value(control, position);
            const QSignalBlocker blocker(spin);
            spin->setValue(next);
            set_value(index, next);
        });
        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, control, index, slider](double value) {
            const QSignalBlocker blocker(slider);
            slider->setValue(slider_position(control, value));
            set_value(index, value);
        });
        connect(reset, &QPushButton::clicked, this, [this, index]() {
            set_value(index, controls[index].default_value);
            QMetaObject::invokeMethod(this, [this]() { rebuild(); }, Qt::QueuedConnection);
        });
    }
    rows_layout->addStretch();
}
