#include "custom-uniforms.hpp"
#include "../shader_selection_shm.hpp"

#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

namespace {
    constexpr int MAX_SLIDER_STEPS = 10000000;

    void configureDoubleSpinBox(QDoubleSpinBox *spin) {
        spin->setDecimals(8);
        spin->setRange(-1000000000.0, 1000000000.0);
    }

    int sliderStepCount(const acmx2::CustomUniformDefinition &uniform) {
        const double count = std::ceil(
            (uniform.maximum - uniform.minimum) / uniform.step);
        if (count >= MAX_SLIDER_STEPS)
            return MAX_SLIDER_STEPS;
        return std::max(static_cast<int>(count), 1);
    }

    double sliderValue(const acmx2::CustomUniformDefinition &uniform, int position) {
        return std::min(uniform.maximum,
                        uniform.minimum + uniform.step * position);
    }

    int sliderPosition(const acmx2::CustomUniformDefinition &uniform, double value) {
        const double position =
            std::round((value - uniform.minimum) / uniform.step);
        return std::clamp(static_cast<int>(position), 0,
                          sliderStepCount(uniform));
    }

    int decimalsForStep(double step) {
        if (step >= 1.0)
            return 6;
        return std::clamp(static_cast<int>(std::ceil(-std::log10(step))) + 2,
                          2, 10);
    }
} // namespace

CustomUniformDialog::CustomUniformDialog(QWidget *parent)
    : QDialog(parent) {
    setWindowTitle(tr("Custom Uniforms"));
    resize(850, 520);
    setModal(false);

    auto *mainLayout = new QVBoxLayout(this);
    auto *addLayout = new QHBoxLayout();

    nameEdit = new QLineEdit(this);
    nameEdit->setPlaceholderText(tr("GLSL uniform name"));
    minimumSpin = new QDoubleSpinBox(this);
    maximumSpin = new QDoubleSpinBox(this);
    stepSpin = new QDoubleSpinBox(this);
    configureDoubleSpinBox(minimumSpin);
    configureDoubleSpinBox(maximumSpin);
    configureDoubleSpinBox(stepSpin);
    minimumSpin->setValue(0.0);
    maximumSpin->setValue(1.0);
    stepSpin->setMinimum(0.00000001);
    stepSpin->setValue(0.01);

    addLayout->addWidget(new QLabel(tr("Name"), this));
    addLayout->addWidget(nameEdit, 2);
    addLayout->addWidget(new QLabel(tr("Minimum"), this));
    addLayout->addWidget(minimumSpin);
    addLayout->addWidget(new QLabel(tr("Maximum"), this));
    addLayout->addWidget(maximumSpin);
    addLayout->addWidget(new QLabel(tr("Step"), this));
    addLayout->addWidget(stepSpin);
    auto *addButton = new QPushButton(tr("Add Slider"), this);
    addLayout->addWidget(addButton);
    mainLayout->addLayout(addLayout);

    auto *hint = new QLabel(
        tr("Names become GLSL float uniforms. Values are saved in library.json and sent live to acmx2."),
        this);
    hint->setWordWrap(true);
    mainLayout->addWidget(hint);

    scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    rowsWidget = new QWidget(scrollArea);
    rowsLayout = new QVBoxLayout(rowsWidget);
    rowsLayout->setAlignment(Qt::AlignTop);
    scrollArea->setWidget(rowsWidget);
    mainLayout->addWidget(scrollArea, 1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    mainLayout->addWidget(buttons);

    saveTimer = new QTimer(this);
    saveTimer->setSingleShot(true);
    saveTimer->setInterval(150);
    connect(saveTimer, &QTimer::timeout, this,
            &CustomUniformDialog::savePendingChanges);
    connect(addButton, &QPushButton::clicked, this,
            &CustomUniformDialog::addUniform);
    connect(nameEdit, &QLineEdit::returnPressed, this,
            &CustomUniformDialog::addUniform);
    connect(buttons, &QDialogButtonBox::rejected, this,
            &CustomUniformDialog::hide);

    rebuildUniformRows();
}

bool CustomUniformDialog::loadLibrary(const QString &directory, QString *error) {
    if (saveTimer->isActive()) {
        saveTimer->stop();
        saveUniforms(false);
    }

    QList<acmx2::CustomUniformDefinition> loadedUniforms;
    QString loadError;
    const auto clearFailedLoad = [this, &directory]() {
        libraryDirectory = directory;
        uniformDefinitions.clear();
        rebuildUniformRows();
        emit uniformsChanged();
    };
    if (!acmx2::load_custom_uniforms(directory, loadedUniforms, loadError)) {
        clearFailedLoad();
        if (error)
            *error = loadError;
        return false;
    }
    if (loadedUniforms.size() >
        static_cast<int>(acmx2::ipc::kShaderSelectionMaxCustomUniforms)) {
        clearFailedLoad();
        if (error) {
            *error = tr("library.json contains more than %1 custom uniforms.")
                         .arg(acmx2::ipc::kShaderSelectionMaxCustomUniforms);
        }
        return false;
    }
    for (const auto &uniform : loadedUniforms) {
        const double stepCount =
            std::ceil((uniform.maximum - uniform.minimum) / uniform.step);
        if (stepCount > MAX_SLIDER_STEPS) {
            clearFailedLoad();
            if (error) {
                *error = tr("Custom uniform '%1' has more than %2 slider positions. Increase its step size.")
                             .arg(uniform.name)
                             .arg(MAX_SLIDER_STEPS);
            }
            return false;
        }
    }

    libraryDirectory = directory;
    uniformDefinitions = loadedUniforms;
    rebuildUniformRows();
    emit uniformsChanged();
    return true;
}

const QList<acmx2::CustomUniformDefinition> &CustomUniformDialog::uniforms() const {
    return uniformDefinitions;
}

void CustomUniformDialog::addUniform() {
    if (libraryDirectory.isEmpty()) {
        QMessageBox::information(this, tr("Custom Uniforms"),
                                 tr("Load a shader library first."));
        return;
    }
    if (uniformDefinitions.size() >=
        static_cast<int>(acmx2::ipc::kShaderSelectionMaxCustomUniforms)) {
        QMessageBox::warning(
            this, tr("Custom Uniform Limit"),
            tr("A library can contain at most %1 custom uniforms.")
                .arg(acmx2::ipc::kShaderSelectionMaxCustomUniforms));
        return;
    }

    const QString name = nameEdit->text().trimmed();
    static const QRegularExpression identifier(
        QStringLiteral("^[A-Za-z_][A-Za-z0-9_]*$"));
    if (!identifier.match(name).hasMatch() || name.startsWith("gl_") ||
        name.toUtf8().size() >=
            static_cast<int>(acmx2::ipc::kShaderSelectionMaxUniformName)) {
        QMessageBox::warning(
            this, tr("Invalid Uniform Name"),
            tr("Use a GLSL identifier that does not begin with gl_."));
        return;
    }
    if (uniformIndex(name) >= 0) {
        QMessageBox::warning(this, tr("Duplicate Uniform"),
                             tr("A custom uniform named '%1' already exists.")
                                 .arg(name));
        return;
    }

    const double minimum = minimumSpin->value();
    const double maximum = maximumSpin->value();
    const double step = stepSpin->value();
    if (maximum <= minimum || step <= 0.0) {
        QMessageBox::warning(this, tr("Invalid Range"),
                             tr("Maximum must be greater than minimum and step must be positive."));
        return;
    }
    const double steps = std::ceil((maximum - minimum) / step);
    if (steps > MAX_SLIDER_STEPS) {
        QMessageBox::warning(
            this, tr("Too Many Slider Steps"),
            tr("Increase the step size so the slider has no more than %1 positions.")
                .arg(MAX_SLIDER_STEPS));
        return;
    }

    uniformDefinitions.append({name, minimum, maximum, step, minimum});
    if (!saveUniforms(true)) {
        uniformDefinitions.removeLast();
        return;
    }
    nameEdit->clear();
    rebuildUniformRows();
    emit uniformsChanged();
    emit uniformDefinitionsChanged();
}

void CustomUniformDialog::savePendingChanges() {
    saveUniforms(true);
}

void CustomUniformDialog::rebuildUniformRows() {
    while (QLayoutItem *item = rowsLayout->takeAt(0)) {
        if (QWidget *widget = item->widget())
            widget->deleteLater();
        delete item;
    }

    if (uniformDefinitions.isEmpty()) {
        auto *emptyLabel = new QLabel(
            tr("No custom uniforms are defined for this library."), rowsWidget);
        emptyLabel->setAlignment(Qt::AlignCenter);
        rowsLayout->addWidget(emptyLabel);
        rowsLayout->addStretch();
        return;
    }

    for (const acmx2::CustomUniformDefinition &uniform : uniformDefinitions) {
        auto *row = new QWidget(rowsWidget);
        auto *layout = new QGridLayout(row);
        layout->setContentsMargins(4, 4, 4, 4);

        auto *nameLabel = new QLabel(uniform.name, row);
        nameLabel->setMinimumWidth(140);
        auto *slider = new QSlider(Qt::Horizontal, row);
        slider->setRange(0, sliderStepCount(uniform));
        slider->setValue(sliderPosition(uniform, uniform.value));
        auto *valueSpin = new QDoubleSpinBox(row);
        valueSpin->setDecimals(decimalsForStep(uniform.step));
        valueSpin->setRange(uniform.minimum, uniform.maximum);
        valueSpin->setSingleStep(uniform.step);
        valueSpin->setValue(uniform.value);
        valueSpin->setKeyboardTracking(false);
        auto *rangeLabel = new QLabel(
            tr("%1 to %2, step %3")
                .arg(uniform.minimum, 0, 'g', 8)
                .arg(uniform.maximum, 0, 'g', 8)
                .arg(uniform.step, 0, 'g', 8),
            row);
        auto *deleteButton = new QPushButton(tr("Delete"), row);

        layout->addWidget(nameLabel, 0, 0);
        layout->addWidget(slider, 0, 1);
        layout->addWidget(valueSpin, 0, 2);
        layout->addWidget(deleteButton, 0, 3);
        layout->addWidget(rangeLabel, 1, 1, 1, 2);
        layout->setColumnStretch(1, 1);
        rowsLayout->addWidget(row);

        connect(slider, &QSlider::valueChanged, this,
                [this, name = uniform.name, valueSpin](int position) {
                    const int index = uniformIndex(name);
                    if (index < 0)
                        return;
                    auto &definition = uniformDefinitions[index];
                    definition.value = sliderValue(definition, position);
                    const QSignalBlocker blocker(valueSpin);
                    valueSpin->setValue(definition.value);
                    saveTimer->start();
                    emit uniformsChanged();
                });
        connect(valueSpin,
                static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
                this, [this, name = uniform.name, slider, valueSpin](double value) {
                    const int index = uniformIndex(name);
                    if (index < 0)
                        return;
                    auto &definition = uniformDefinitions[index];
                    const int position = sliderPosition(definition, value);
                    definition.value = sliderValue(definition, position);
                    const QSignalBlocker sliderBlocker(slider);
                    const QSignalBlocker spinBlocker(valueSpin);
                    slider->setValue(position);
                    valueSpin->setValue(definition.value);
                    saveTimer->start();
                    emit uniformsChanged();
                });
        connect(deleteButton, &QPushButton::clicked, this,
                [this, name = uniform.name]() { removeUniform(name); });
    }
    rowsLayout->addStretch();
}

void CustomUniformDialog::removeUniform(const QString &name) {
    const int index = uniformIndex(name);
    if (index < 0)
        return;
    const acmx2::CustomUniformDefinition removed = uniformDefinitions.takeAt(index);
    if (!saveUniforms(true)) {
        uniformDefinitions.insert(index, removed);
        return;
    }
    rebuildUniformRows();
    emit uniformsChanged();
    emit uniformDefinitionsChanged();
}

int CustomUniformDialog::uniformIndex(const QString &name) const {
    for (int i = 0; i < uniformDefinitions.size(); ++i) {
        if (uniformDefinitions.at(i).name == name)
            return i;
    }
    return -1;
}

bool CustomUniformDialog::saveUniforms(bool showError) {
    if (libraryDirectory.isEmpty())
        return false;
    QString error;
    if (acmx2::write_custom_uniforms(libraryDirectory, uniformDefinitions,
                                     error)) {
        return true;
    }
    if (showError)
        QMessageBox::warning(this, tr("Could Not Save Custom Uniforms"), error);
    return false;
}
