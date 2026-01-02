#include "gpufilter.hpp"
#include <QMessageBox>
#include <QRegularExpression>
#include <QTextStream>

GPUFilterDialog::GPUFilterDialog(const QString &executablePath, QWidget *parent)
    : QDialog(parent), execPath(executablePath)
{
    setWindowTitle("GPU Filter Settings");
    setMinimumSize(500, 450);
    setupUI();
    loadFiltersFromExecutable();
}

void GPUFilterDialog::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    enableCheckBox = new QCheckBox("Enable GPU Filtering", this);
    mainLayout->addWidget(enableCheckBox);
    QGroupBox *bufferGroup = new QGroupBox("Frame Buffer Size", this);
    QHBoxLayout *bufferLayout = new QHBoxLayout(bufferGroup);
    QLabel *bufferLabel = new QLabel("Buffer Size (4-32):", this);
    bufferSizeSpinBox = new QSpinBox(this);
    bufferSizeSpinBox->setRange(4, 32);
    bufferSizeSpinBox->setValue(8);
    bufferLayout->addWidget(bufferLabel);
    bufferLayout->addWidget(bufferSizeSpinBox);
    bufferLayout->addStretch();
    mainLayout->addWidget(bufferGroup);
    QGroupBox *filterGroup = new QGroupBox("Filter Selection", this);
    QVBoxLayout *filterMainLayout = new QVBoxLayout(filterGroup);
    QHBoxLayout *comboLayout = new QHBoxLayout();
    QLabel *availableLabel = new QLabel("Available Filters:", this);
    filterComboBox = new QComboBox(this);
    filterComboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    comboLayout->addWidget(availableLabel);
    comboLayout->addWidget(filterComboBox, 1);
    filterMainLayout->addLayout(comboLayout);
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    addButton = new QPushButton("Add →", this);
    removeButton = new QPushButton("← Remove", this);
    upButton = new QPushButton("↑ Up", this);
    downButton = new QPushButton("↓ Down", this);
    clearButton = new QPushButton("Clear All", this);
    buttonLayout->addWidget(addButton);
    buttonLayout->addWidget(removeButton);
    buttonLayout->addWidget(upButton);
    buttonLayout->addWidget(downButton);
    buttonLayout->addWidget(clearButton);
    filterMainLayout->addLayout(buttonLayout);
    QLabel *selectedLabel = new QLabel("Selected Filters (processing order):", this);
    filterMainLayout->addWidget(selectedLabel);
    selectedFiltersList = new QListWidget(this);
    selectedFiltersList->setMinimumHeight(150);
    filterMainLayout->addWidget(selectedFiltersList);
    mainLayout->addWidget(filterGroup);
    QHBoxLayout *dialogButtonLayout = new QHBoxLayout();
    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);
    dialogButtonLayout->addStretch();
    dialogButtonLayout->addWidget(okButton);
    dialogButtonLayout->addWidget(cancelButton);
    mainLayout->addLayout(dialogButtonLayout);
    connect(addButton, &QPushButton::clicked, this, &GPUFilterDialog::addFilter);
    connect(removeButton, &QPushButton::clicked, this, &GPUFilterDialog::removeFilter);
    connect(upButton, &QPushButton::clicked, this, &GPUFilterDialog::moveUp);
    connect(downButton, &QPushButton::clicked, this, &GPUFilterDialog::moveDown);
    connect(clearButton, &QPushButton::clicked, this, &GPUFilterDialog::clearAll);
    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    connect(enableCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        filterComboBox->setEnabled(checked);
        selectedFiltersList->setEnabled(checked);
        bufferSizeSpinBox->setEnabled(checked);
        addButton->setEnabled(checked);
        removeButton->setEnabled(checked);
        upButton->setEnabled(checked);
        downButton->setEnabled(checked);
        clearButton->setEnabled(checked);
    });

    enableCheckBox->setChecked(false);
    filterComboBox->setEnabled(false);
    selectedFiltersList->setEnabled(false);
    bufferSizeSpinBox->setEnabled(false);
    addButton->setEnabled(false);
    removeButton->setEnabled(false);
    upButton->setEnabled(false);
    downButton->setEnabled(false);
    clearButton->setEnabled(false);
    
    // Style
    QString style = "QDialog { background-color: black; }"
                    "QGroupBox { color: red; border: 1px solid red; margin-top: 10px; padding-top: 10px; }"
                    "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
                    "QLabel { color: red; }"
                    "QCheckBox { color: red; }"
                    "QComboBox { background-color: #110000; color: red; border: 1px solid red; }"
                    "QSpinBox { background-color: #110000; color: red; border: 1px solid red; }"
                    "QListWidget { background-color: #110000; color: lime; border: 1px solid red; }"
                    "QPushButton { border: 1px solid red; background-color: #110000; color: red; padding: 5px; }"
                    "QPushButton:hover { background-color: red; color: black; }";
    setStyleSheet(style);
}

void GPUFilterDialog::loadFiltersFromExecutable()
{
    QProcess process;
    QStringList args;
    args << "--list-filters";
    
    process.start(execPath, args);
    if (!process.waitForFinished(10000)) {
        QMessageBox::warning(this, "Error", 
            "Failed to get filter list from acmx2.\nMake sure the executable path is correct.");
        return;
    }
    
    QString output = process.readAllStandardOutput();
    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    QRegularExpression re("^\\s*(\\d+):\\s*(.+)$");
    
    filterNames.clear();
    filterNameToIndex.clear();
    filterComboBox->clear();
    
    for (const QString &line : lines) {
        QRegularExpressionMatch match = re.match(line);
        if (match.hasMatch()) {
            int index = match.captured(1).toInt();
            QString name = match.captured(2).trimmed();
            QString displayName = QString("%1: %2").arg(index).arg(name);
            
            filterNames.append(displayName);
            filterNameToIndex[displayName] = index;
            filterComboBox->addItem(displayName);
        }
    }
    
    if (filterNames.isEmpty()) {
        QMessageBox::warning(this, "Warning", 
            "No GPU filters found. Make sure acmx2 is compiled with GPU filter support.");
    }
}

void GPUFilterDialog::addFilter()
{
    if (filterComboBox->currentIndex() < 0) {
        return;
    }
    
    QString filterName = filterComboBox->currentText();
    selectedFiltersList->addItem(filterName);
}

void GPUFilterDialog::removeFilter()
{
    QListWidgetItem *item = selectedFiltersList->currentItem();
    if (item) {
        delete selectedFiltersList->takeItem(selectedFiltersList->row(item));
    }
}

void GPUFilterDialog::moveUp()
{
    int currentRow = selectedFiltersList->currentRow();
    if (currentRow > 0) {
        QListWidgetItem *item = selectedFiltersList->takeItem(currentRow);
        selectedFiltersList->insertItem(currentRow - 1, item);
        selectedFiltersList->setCurrentRow(currentRow - 1);
    }
}

void GPUFilterDialog::moveDown()
{
    int currentRow = selectedFiltersList->currentRow();
    if (currentRow >= 0 && currentRow < selectedFiltersList->count() - 1) {
        QListWidgetItem *item = selectedFiltersList->takeItem(currentRow);
        selectedFiltersList->insertItem(currentRow + 1, item);
        selectedFiltersList->setCurrentRow(currentRow + 1);
    }
}

void GPUFilterDialog::clearAll()
{
    selectedFiltersList->clear();
}

bool GPUFilterDialog::isGPUFilterEnabled() const
{
    return enableCheckBox->isChecked() && selectedFiltersList->count() > 0;
}

QStringList GPUFilterDialog::getSelectedFilterIndices() const
{
    QStringList indices;
    for (int i = 0; i < selectedFiltersList->count(); ++i) {
        QString itemText = selectedFiltersList->item(i)->text();
        if (filterNameToIndex.contains(itemText)) {
            indices.append(QString::number(filterNameToIndex[itemText]));
        }
    }
    return indices;
}

int GPUFilterDialog::getBufferSize() const
{
    return bufferSizeSpinBox->value();
}

QString GPUFilterDialog::getFilterArgument() const
{
    QStringList indices = getSelectedFilterIndices();
    return indices.join(",");
}
