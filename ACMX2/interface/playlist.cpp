#include "playlist.hpp"
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QSettings>
#include <QTextStream>
#include <algorithm>

PlaylistDialog::PlaylistDialog(const QStringList &shaderNames, QWidget *parent)
    : QDialog(parent) {
    setWindowTitle("Shader Playlist Settings");
    setMinimumSize(500, 550);
    setupUI();
    loadShaders(shaderNames);
}

void PlaylistDialog::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    enableCheckBox = new QCheckBox("Enable Shader Playlist", this);
    mainLayout->addWidget(enableCheckBox);

    QLabel *infoLabel = new QLabel(
        "Build a playlist of shaders. When enabled, the playlist file is passed\n"
        "to acmx2. Press P to toggle playlist mode; Up/Down to navigate the list.",
        this);
    infoLabel->setWordWrap(true);
    mainLayout->addWidget(infoLabel);

    QGroupBox *shaderGroup = new QGroupBox("Playlist Shader Selection", this);
    QVBoxLayout *shaderMainLayout = new QVBoxLayout(shaderGroup);

    QHBoxLayout *searchLayout = new QHBoxLayout();
    QLabel *searchLabel = new QLabel("Search:", this);
    searchLineEdit = new QLineEdit(this);
    searchLineEdit->setPlaceholderText("Type to search shaders...");
    searchLineEdit->setClearButtonEnabled(true);
    searchLayout->addWidget(searchLabel);
    searchLayout->addWidget(searchLineEdit, 1);
    shaderMainLayout->addLayout(searchLayout);

    QHBoxLayout *comboLayout = new QHBoxLayout();
    QLabel *availableLabel = new QLabel("Available Shaders:", this);
    shaderComboBox = new QComboBox(this);
    shaderComboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    shaderComboBox->setMaxVisibleItems(20);

    shaderModel = new QStandardItemModel(this);
    proxyModel = new QSortFilterProxyModel(this);
    proxyModel->setSourceModel(shaderModel);
    proxyModel->setFilterCaseSensitivity(Qt::CaseInsensitive);
    proxyModel->setSortCaseSensitivity(Qt::CaseInsensitive);
    shaderComboBox->setModel(proxyModel);

    comboLayout->addWidget(availableLabel);
    comboLayout->addWidget(shaderComboBox, 1);
    shaderMainLayout->addLayout(comboLayout);

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
    shaderMainLayout->addLayout(buttonLayout);

    QLabel *selectedLabel = new QLabel("Playlist Order (press P in acmx2 to toggle, Up/Down to navigate):", this);
    shaderMainLayout->addWidget(selectedLabel);
    selectedShadersList = new QListWidget(this);
    selectedShadersList->setMinimumHeight(200);
    shaderMainLayout->addWidget(selectedShadersList);

    QHBoxLayout *fileButtonLayout = new QHBoxLayout();
    saveButton = new QPushButton("Save Playlist...", this);
    loadButton = new QPushButton("Load Playlist...", this);
    fileButtonLayout->addWidget(saveButton);
    fileButtonLayout->addWidget(loadButton);
    fileButtonLayout->addStretch();
    shaderMainLayout->addLayout(fileButtonLayout);

    mainLayout->addWidget(shaderGroup);

    QHBoxLayout *dialogButtonLayout = new QHBoxLayout();
    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);
    dialogButtonLayout->addStretch();
    dialogButtonLayout->addWidget(okButton);
    dialogButtonLayout->addWidget(cancelButton);
    mainLayout->addLayout(dialogButtonLayout);

    connect(addButton, &QPushButton::clicked, this, &PlaylistDialog::addShader);
    connect(removeButton, &QPushButton::clicked, this, &PlaylistDialog::removeShader);
    connect(upButton, &QPushButton::clicked, this, &PlaylistDialog::moveUp);
    connect(downButton, &QPushButton::clicked, this, &PlaylistDialog::moveDown);
    connect(clearButton, &QPushButton::clicked, this, &PlaylistDialog::clearAll);
    connect(saveButton, &QPushButton::clicked, this, &PlaylistDialog::savePlaylist);
    connect(loadButton, &QPushButton::clicked, this, &PlaylistDialog::loadPlaylist);
    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    connect(searchLineEdit, &QLineEdit::textChanged, this, &PlaylistDialog::filterSearchChanged);

    connect(enableCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        shaderComboBox->setEnabled(checked);
        selectedShadersList->setEnabled(checked);
        searchLineEdit->setEnabled(checked);
        addButton->setEnabled(checked);
        removeButton->setEnabled(checked);
        upButton->setEnabled(checked);
        downButton->setEnabled(checked);
        clearButton->setEnabled(checked);
        saveButton->setEnabled(checked);
        loadButton->setEnabled(checked);
    });

    enableCheckBox->setChecked(false);
    shaderComboBox->setEnabled(false);
    selectedShadersList->setEnabled(false);
    searchLineEdit->setEnabled(false);
    addButton->setEnabled(false);
    removeButton->setEnabled(false);
    upButton->setEnabled(false);
    downButton->setEnabled(false);
    clearButton->setEnabled(false);
    saveButton->setEnabled(false);
    loadButton->setEnabled(false);

    QString style = "QDialog { background-color: black; }"
                    "QGroupBox { color: cyan; border: 1px solid cyan; margin-top: 10px; padding-top: 10px; }"
                    "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
                    "QLabel { color: cyan; }"
                    "QCheckBox { color: cyan; }"
                    "QLineEdit { background-color: #001111; color: cyan; border: 1px solid cyan; padding: 3px; }"
                    "QComboBox { background-color: #001111; color: cyan; border: 1px solid cyan; }"
                    "QListWidget { background-color: #001111; color: lime; border: 1px solid cyan; }"
                    "QPushButton { border: 1px solid cyan; background-color: #001111; color: cyan; padding: 5px; }"
                    "QPushButton:hover { background-color: cyan; color: black; }";
    QSettings appSettings("LostSideDead");
    if (appSettings.value("useCustomStyle", false).toBool()) {
        setStyleSheet(style);
    }
}

void PlaylistDialog::loadShaders(const QStringList &shaderNames) {
    QStringList selectedNames = getSelectedShaderNames();

    shaderNamesList.clear();
    shaderNameToIndex.clear();
    shaderModel->clear();

    for (int i = 0; i < shaderNames.size(); ++i) {
        QString name = shaderNames[i];
        shaderNamesList.append(name);
        shaderNameToIndex[name] = i;
        QStandardItem *item = new QStandardItem(name);
        item->setData(i, Qt::UserRole);
        shaderModel->appendRow(item);
    }

    if (!selectedNames.isEmpty()) {
        setSelectedShaderNames(selectedNames);
    }
}

void PlaylistDialog::filterSearchChanged(const QString &text) {
    proxyModel->setFilterFixedString(text);
    if (proxyModel->rowCount() > 0) {
        shaderComboBox->setCurrentIndex(0);
    }
}

void PlaylistDialog::addShader() {
    if (shaderComboBox->currentIndex() < 0)
        return;

    QString shaderName = shaderComboBox->currentText();
    QListWidgetItem *item = new QListWidgetItem(shaderName);
    if (shaderNameToIndex.contains(shaderName)) {
        item->setData(Qt::UserRole, shaderNameToIndex[shaderName]);
    }
    selectedShadersList->addItem(item);
}

void PlaylistDialog::removeShader() {
    QListWidgetItem *item = selectedShadersList->currentItem();
    if (item) {
        delete selectedShadersList->takeItem(selectedShadersList->row(item));
    }
}

void PlaylistDialog::moveUp() {
    int currentRow = selectedShadersList->currentRow();
    if (currentRow > 0) {
        QListWidgetItem *item = selectedShadersList->takeItem(currentRow);
        selectedShadersList->insertItem(currentRow - 1, item);
        selectedShadersList->setCurrentRow(currentRow - 1);
    }
}

void PlaylistDialog::moveDown() {
    int currentRow = selectedShadersList->currentRow();
    if (currentRow >= 0 && currentRow < selectedShadersList->count() - 1) {
        QListWidgetItem *item = selectedShadersList->takeItem(currentRow);
        selectedShadersList->insertItem(currentRow + 1, item);
        selectedShadersList->setCurrentRow(currentRow + 1);
    }
}

void PlaylistDialog::clearAll() {
    selectedShadersList->clear();
}

void PlaylistDialog::savePlaylist() {
    if (selectedShadersList->count() == 0) {
        QMessageBox::information(this, "Empty Playlist", "Add shaders to the playlist before saving.");
        return;
    }

    QString filePath = QFileDialog::getSaveFileName(this, "Save Playlist", QString(), "Text Files (*.txt);;All Files (*)");
    if (filePath.isEmpty())
        return;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Could not save playlist file: " + filePath);
        return;
    }

    QTextStream out(&file);
    for (int i = 0; i < selectedShadersList->count(); ++i) {
        out << selectedShadersList->item(i)->text() << "\n";
    }
    file.close();
    playlistFilePath = filePath;
    QMessageBox::information(this, "Saved", "Playlist saved to: " + filePath);
}

void PlaylistDialog::loadPlaylist() {
    QString filePath = QFileDialog::getOpenFileName(this, "Load Playlist", QString(), "Text Files (*.txt);;All Files (*)");
    if (filePath.isEmpty())
        return;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Could not open playlist file: " + filePath);
        return;
    }

    selectedShadersList->clear();
    QTextStream in(&file);
    int loadedCount = 0;
    int skippedCount = 0;
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty())
            continue;
        if (shaderNameToIndex.contains(line)) {
            QListWidgetItem *item = new QListWidgetItem(line);
            item->setData(Qt::UserRole, shaderNameToIndex[line]);
            selectedShadersList->addItem(item);
            ++loadedCount;
        } else {
            ++skippedCount;
        }
    }
    file.close();
    playlistFilePath = filePath;

    QString msg = "Loaded " + QString::number(loadedCount) + " shader(s).";
    if (skippedCount > 0)
        msg += "\n" + QString::number(skippedCount) + " shader(s) not found and skipped.";
    QMessageBox::information(this, "Playlist Loaded", msg);
}

bool PlaylistDialog::isPlaylistEnabled() const {
    return enableCheckBox->isChecked() && selectedShadersList->count() > 0;
}

QStringList PlaylistDialog::getSelectedShaderNames() const {
    QStringList names;
    for (int i = 0; i < selectedShadersList->count(); ++i) {
        names.append(selectedShadersList->item(i)->text());
    }
    return names;
}

QString PlaylistDialog::getPlaylistFile() const {
    return playlistFilePath;
}

void PlaylistDialog::setEnabled(bool enabled) {
    enableCheckBox->setChecked(enabled);
}

void PlaylistDialog::setSelectedShaderNames(const QStringList &names) {
    selectedShadersList->clear();
    for (const QString &name : names) {
        if (shaderNameToIndex.contains(name)) {
            int idx = shaderNameToIndex[name];
            QListWidgetItem *item = new QListWidgetItem(name);
            item->setData(Qt::UserRole, idx);
            selectedShadersList->addItem(item);
        }
    }
}

void PlaylistDialog::setPlaylistFile(const QString &path) {
    playlistFilePath = path;
}

void PlaylistDialog::updateShaderList(const QStringList &shaderNames) {
    loadShaders(shaderNames);
}
