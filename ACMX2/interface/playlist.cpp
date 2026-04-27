#include "playlist.hpp"
#include <QFileDialog>
#include <QFileInfo>
#include <QInputDialog>
#include <QMessageBox>
#include <QSettings>
#include <QTextStream>
#include <algorithm>
#include <random>

PlaylistDialog::PlaylistDialog(const QStringList &shaderNames, QWidget *parent)
    : QDialog(parent) {
    setWindowTitle("Shader Playlist Settings");
    setMinimumSize(600, 600);
    setupUI();
    loadShaders(shaderNames);
}

void PlaylistDialog::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    enableCheckBox = new QCheckBox("Enable Shader Playlist", this);
    mainLayout->addWidget(enableCheckBox);

    QLabel *infoLabel = new QLabel(
        "Build a playlist tree of shaders. Create named nodes, then add shaders to each node.\n"
        "When enabled, the playlist file is passed to acmx2. Press P to toggle; Up/Down to navigate.",
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

    QHBoxLayout *nodeButtonLayout = new QHBoxLayout();
    addNodeButton = new QPushButton("+ Node", this);
    renameNodeButton = new QPushButton("Rename Node", this);
    removeNodeButton = new QPushButton("- Node", this);
    nodeButtonLayout->addWidget(addNodeButton);
    nodeButtonLayout->addWidget(renameNodeButton);
    nodeButtonLayout->addWidget(removeNodeButton);
    nodeButtonLayout->addStretch();
    shaderMainLayout->addLayout(nodeButtonLayout);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    addButton = new QPushButton("Add Shader", this);
    removeButton = new QPushButton("Remove", this);
    upButton = new QPushButton("↑ Up", this);
    downButton = new QPushButton("↓ Down", this);
    clearButton = new QPushButton("Clear All", this);
    shuffleButton = new QPushButton("Shuffle", this);
    buttonLayout->addWidget(addButton);
    buttonLayout->addWidget(removeButton);
    buttonLayout->addWidget(upButton);
    buttonLayout->addWidget(downButton);
    buttonLayout->addWidget(clearButton);
    buttonLayout->addWidget(shuffleButton);
    shaderMainLayout->addLayout(buttonLayout);

    QLabel *selectedLabel = new QLabel("Playlist Tree (press P in acmx2 to toggle, Up/Down to navigate):", this);
    shaderMainLayout->addWidget(selectedLabel);
    playlistTree = new QTreeWidget(this);
    playlistTree->setHeaderLabels({"Shader / Node"});
    playlistTree->setMinimumHeight(250);
    playlistTree->setDragDropMode(QAbstractItemView::InternalMove);
    playlistTree->setSelectionMode(QAbstractItemView::SingleSelection);
    shaderMainLayout->addWidget(playlistTree);

    QHBoxLayout *fileButtonLayout = new QHBoxLayout();
    saveButton = new QPushButton("Save Playlist...", this);
    loadButton = new QPushButton("Load Playlist...", this);
    concatButton = new QPushButton("Concat Playlist...", this);
    fileButtonLayout->addWidget(saveButton);
    fileButtonLayout->addWidget(loadButton);
    fileButtonLayout->addWidget(concatButton);
    fileButtonLayout->addStretch();
    shaderMainLayout->addLayout(fileButtonLayout);

    QHBoxLayout *autopilotLayout = new QHBoxLayout();
    QLabel *autopilotLabel = new QLabel(
        "Autopilot frames (0 = off; switch to a random shader after this many frames, toggle with J):",
        this);
    autopilotLabel->setWordWrap(true);
    autopilotFramesSpinBox = new QSpinBox(this);
    autopilotFramesSpinBox->setRange(0, 1000000);
    autopilotFramesSpinBox->setSingleStep(30);
    autopilotFramesSpinBox->setValue(0);
    autopilotFramesSpinBox->setSuffix(" frames");
    autopilotLayout->addWidget(autopilotLabel, 1);
    autopilotLayout->addWidget(autopilotFramesSpinBox);
    shaderMainLayout->addLayout(autopilotLayout);

    mainLayout->addWidget(shaderGroup);

    QHBoxLayout *dialogButtonLayout = new QHBoxLayout();
    okButton = new QPushButton("OK", this);
    cancelButton = new QPushButton("Cancel", this);
    dialogButtonLayout->addStretch();
    dialogButtonLayout->addWidget(okButton);
    dialogButtonLayout->addWidget(cancelButton);
    mainLayout->addLayout(dialogButtonLayout);

    connect(addNodeButton, &QPushButton::clicked, this, &PlaylistDialog::addNode);
    connect(renameNodeButton, &QPushButton::clicked, this, &PlaylistDialog::renameNode);
    connect(removeNodeButton, &QPushButton::clicked, this, &PlaylistDialog::removeNode);
    connect(addButton, &QPushButton::clicked, this, &PlaylistDialog::addShader);
    connect(removeButton, &QPushButton::clicked, this, &PlaylistDialog::removeShader);
    connect(upButton, &QPushButton::clicked, this, &PlaylistDialog::moveUp);
    connect(downButton, &QPushButton::clicked, this, &PlaylistDialog::moveDown);
    connect(clearButton, &QPushButton::clicked, this, &PlaylistDialog::clearAll);
    connect(shuffleButton, &QPushButton::clicked, this, &PlaylistDialog::shufflePlaylist);
    connect(concatButton, &QPushButton::clicked, this, &PlaylistDialog::concatPlaylist);
    connect(saveButton, &QPushButton::clicked, this, &PlaylistDialog::savePlaylist);
    connect(loadButton, &QPushButton::clicked, this, &PlaylistDialog::loadPlaylist);
    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    connect(searchLineEdit, &QLineEdit::textChanged, this, &PlaylistDialog::filterSearchChanged);

    connect(enableCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        shaderComboBox->setEnabled(checked);
        playlistTree->setEnabled(checked);
        searchLineEdit->setEnabled(checked);
        addNodeButton->setEnabled(checked);
        renameNodeButton->setEnabled(checked);
        removeNodeButton->setEnabled(checked);
        addButton->setEnabled(checked);
        removeButton->setEnabled(checked);
        upButton->setEnabled(checked);
        downButton->setEnabled(checked);
        clearButton->setEnabled(checked);
        shuffleButton->setEnabled(checked);
        concatButton->setEnabled(checked);
        saveButton->setEnabled(checked);
        loadButton->setEnabled(checked);
        if (autopilotFramesSpinBox)
            autopilotFramesSpinBox->setEnabled(checked);
    });

    enableCheckBox->setChecked(false);
    shaderComboBox->setEnabled(false);
    playlistTree->setEnabled(false);
    searchLineEdit->setEnabled(false);
    addNodeButton->setEnabled(false);
    renameNodeButton->setEnabled(false);
    removeNodeButton->setEnabled(false);
    addButton->setEnabled(false);
    removeButton->setEnabled(false);
    upButton->setEnabled(false);
    downButton->setEnabled(false);
    clearButton->setEnabled(false);
    shuffleButton->setEnabled(false);
    concatButton->setEnabled(false);
    saveButton->setEnabled(false);
    loadButton->setEnabled(false);
    autopilotFramesSpinBox->setEnabled(false);

    QString style = "QDialog { background-color: black; }"
                    "QGroupBox { color: cyan; border: 1px solid cyan; margin-top: 10px; padding-top: 10px; }"
                    "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
                    "QLabel { color: cyan; }"
                    "QCheckBox { color: cyan; }"
                    "QLineEdit { background-color: #001111; color: cyan; border: 1px solid cyan; padding: 3px; }"
                    "QComboBox { background-color: #001111; color: cyan; border: 1px solid cyan; }"
                    "QTreeWidget { background-color: #001111; color: lime; border: 1px solid cyan; }"
                    "QTreeWidget::item { padding: 4px; }"
                    "QTreeWidget::item:hover { background-color: #002222; }"
                    "QTreeWidget::item:selected { background-color: #003333; color: lime; }"
                    "QTreeWidget::branch { background-color: #001111; }"
                    "QTreeWidget::branch:has-children:closed { image: none; }"
                    "QTreeWidget::branch:has-children:open { image: none; }"
                    "QHeaderView::section { background-color: #001111; color: cyan; border: 1px solid cyan; padding: 4px; }"
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

QTreeWidgetItem *PlaylistDialog::currentNodeItem() const {
    QTreeWidgetItem *current = playlistTree->currentItem();
    if (!current)
        return nullptr;
    if (!current->parent())
        return current;
    return current->parent();
}

void PlaylistDialog::addNode() {
    bool ok = false;
    QString name = QInputDialog::getText(this, "New Playlist Node", "Node name:", QLineEdit::Normal, QString(), &ok);
    if (!ok || name.trimmed().isEmpty())
        return;

    auto *nodeItem = new QTreeWidgetItem(playlistTree);
    nodeItem->setText(0, name.trimmed());
    nodeItem->setFlags(nodeItem->flags() | Qt::ItemIsEditable);
    nodeItem->setExpanded(true);
    playlistTree->setCurrentItem(nodeItem);
}

void PlaylistDialog::renameNode() {
    QTreeWidgetItem *node = currentNodeItem();
    if (!node) {
        QMessageBox::information(this, "No Node Selected", "Select a playlist node to rename.");
        return;
    }

    bool ok = false;
    QString name = QInputDialog::getText(this, "Rename Node", "New name:", QLineEdit::Normal, node->text(0), &ok);
    if (ok && !name.trimmed().isEmpty()) {
        node->setText(0, name.trimmed());
    }
}

void PlaylistDialog::removeNode() {
    QTreeWidgetItem *node = currentNodeItem();
    if (!node) {
        QMessageBox::information(this, "No Node Selected", "Select a playlist node to remove.");
        return;
    }

    if (node->childCount() > 0) {
        auto reply = QMessageBox::question(this, "Remove Node",
                                           "Node \"" + node->text(0) + "\" has " + QString::number(node->childCount()) +
                                               " shader(s). Remove it and all its shaders?",
                                           QMessageBox::Yes | QMessageBox::No);
        if (reply != QMessageBox::Yes)
            return;
    }

    delete node;
}

void PlaylistDialog::addShader() {
    if (shaderComboBox->currentIndex() < 0)
        return;

    QTreeWidgetItem *node = currentNodeItem();
    if (!node) {
        if (playlistTree->topLevelItemCount() == 0) {
            auto *nodeItem = new QTreeWidgetItem(playlistTree);
            nodeItem->setText(0, "Default");
            nodeItem->setFlags(nodeItem->flags() | Qt::ItemIsEditable);
            nodeItem->setExpanded(true);
            node = nodeItem;
        } else {
            QMessageBox::information(this, "No Node Selected", "Select a playlist node to add the shader to.");
            return;
        }
    }

    QString shaderName = shaderComboBox->currentText();
    auto *item = new QTreeWidgetItem(node);
    item->setText(0, shaderName);
    if (shaderNameToIndex.contains(shaderName)) {
        item->setData(0, Qt::UserRole, shaderNameToIndex[shaderName]);
    }
    node->setExpanded(true);
}

void PlaylistDialog::removeShader() {
    QTreeWidgetItem *current = playlistTree->currentItem();
    if (!current)
        return;
    if (!current->parent()) {
        removeNode();
        return;
    }
    delete current;
}

void PlaylistDialog::moveUp() {
    QTreeWidgetItem *current = playlistTree->currentItem();
    if (!current)
        return;

    QTreeWidgetItem *parent = current->parent();
    if (parent) {
        int idx = parent->indexOfChild(current);
        if (idx > 0) {
            parent->takeChild(idx);
            parent->insertChild(idx - 1, current);
            playlistTree->setCurrentItem(current);
        }
    } else {
        int idx = playlistTree->indexOfTopLevelItem(current);
        if (idx > 0) {
            playlistTree->takeTopLevelItem(idx);
            playlistTree->insertTopLevelItem(idx - 1, current);
            playlistTree->setCurrentItem(current);
        }
    }
}

void PlaylistDialog::moveDown() {
    QTreeWidgetItem *current = playlistTree->currentItem();
    if (!current)
        return;

    QTreeWidgetItem *parent = current->parent();
    if (parent) {
        int idx = parent->indexOfChild(current);
        if (idx < parent->childCount() - 1) {
            parent->takeChild(idx);
            parent->insertChild(idx + 1, current);
            playlistTree->setCurrentItem(current);
        }
    } else {
        int idx = playlistTree->indexOfTopLevelItem(current);
        if (idx < playlistTree->topLevelItemCount() - 1) {
            playlistTree->takeTopLevelItem(idx);
            playlistTree->insertTopLevelItem(idx + 1, current);
            playlistTree->setCurrentItem(current);
        }
    }
}

void PlaylistDialog::clearAll() {
    playlistTree->clear();
}

void PlaylistDialog::shufflePlaylist() {
    int totalShaders = 0;
    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        totalShaders += playlistTree->topLevelItem(i)->childCount();
    }
    if (totalShaders == 0) {
        QMessageBox::information(this, "Empty Playlist", "Add shaders to the playlist before shuffling.");
        return;
    }

    static thread_local std::mt19937 rng{std::random_device{}()};

    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem *node = playlistTree->topLevelItem(i);
        const int n = node->childCount();
        if (n < 2)
            continue;

        QList<QTreeWidgetItem *> children;
        children.reserve(n);
        while (node->childCount() > 0) {
            children.append(node->takeChild(0));
        }
        std::shuffle(children.begin(), children.end(), rng);
        for (QTreeWidgetItem *child : children) {
            node->addChild(child);
        }
        node->setExpanded(true);
    }
}

void PlaylistDialog::concatPlaylist() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastPlaylistDir", "").toString();
    QString filePath = QFileDialog::getOpenFileName(this, "Concat Playlist", lastDir,
                                                    "Text Files (*.txt);;All Files (*)");
    if (filePath.isEmpty())
        return;

    appSettings.setValue("lastPlaylistDir", QFileInfo(filePath).absolutePath());

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Could not open playlist file: " + filePath);
        return;
    }

    QTextStream in(&file);
    int loadedCount = 0;
    int skippedCount = 0;
    int nodesAdded = 0;
    QTreeWidgetItem *currentNode = nullptr;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty())
            continue;

        if (line.startsWith('[') && line.endsWith(']')) {
            QString nodeName = line.mid(1, line.length() - 2);
            currentNode = new QTreeWidgetItem(playlistTree);
            currentNode->setText(0, nodeName);
            currentNode->setFlags(currentNode->flags() | Qt::ItemIsEditable);
            currentNode->setExpanded(true);
            ++nodesAdded;
        } else {
            if (!currentNode) {
                QString nodeName = QFileInfo(filePath).baseName();
                if (nodeName.isEmpty())
                    nodeName = "Concat";
                currentNode = new QTreeWidgetItem(playlistTree);
                currentNode->setText(0, nodeName);
                currentNode->setFlags(currentNode->flags() | Qt::ItemIsEditable);
                currentNode->setExpanded(true);
                ++nodesAdded;
            }
            if (shaderNameToIndex.contains(line)) {
                auto *item = new QTreeWidgetItem(currentNode);
                item->setText(0, line);
                item->setData(0, Qt::UserRole, shaderNameToIndex[line]);
                ++loadedCount;
            } else {
                ++skippedCount;
            }
        }
    }
    file.close();

    QString msg = "Concatenated " + QString::number(loadedCount) + " shader(s) into " +
                  QString::number(nodesAdded) + " node(s).";
    if (skippedCount > 0)
        msg += "\n" + QString::number(skippedCount) + " shader(s) not found and skipped.";
    QMessageBox::information(this, "Playlist Concatenated", msg);
}

void PlaylistDialog::savePlaylist() {
    int shaderCount = 0;
    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        shaderCount += playlistTree->topLevelItem(i)->childCount();
    }
    if (shaderCount == 0) {
        QMessageBox::information(this, "Empty Playlist", "Add shaders to the playlist before saving.");
        return;
    }

    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastPlaylistDir", "").toString();
    QString filePath = QFileDialog::getSaveFileName(this, "Save Playlist", lastDir, "Text Files (*.txt);;All Files (*)");
    if (filePath.isEmpty())
        return;

    appSettings.setValue("lastPlaylistDir", QFileInfo(filePath).absolutePath());

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Could not save playlist file: " + filePath);
        return;
    }

    QTextStream out(&file);
    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem *node = playlistTree->topLevelItem(i);
        out << "[" << node->text(0) << "]\n";
        for (int j = 0; j < node->childCount(); ++j) {
            out << node->child(j)->text(0) << "\n";
        }
    }
    file.close();
    playlistFilePath = filePath;
    QMessageBox::information(this, "Saved", "Playlist saved to: " + filePath);
}

void PlaylistDialog::loadPlaylist() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastPlaylistDir", "").toString();
    QString filePath = QFileDialog::getOpenFileName(this, "Load Playlist", lastDir, "Text Files (*.txt);;All Files (*)");
    if (filePath.isEmpty())
        return;

    appSettings.setValue("lastPlaylistDir", QFileInfo(filePath).absolutePath());

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Could not open playlist file: " + filePath);
        return;
    }

    playlistTree->clear();
    QTextStream in(&file);
    int loadedCount = 0;
    int skippedCount = 0;
    QTreeWidgetItem *currentNode = nullptr;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty())
            continue;

        if (line.startsWith('[') && line.endsWith(']')) {
            QString nodeName = line.mid(1, line.length() - 2);
            currentNode = new QTreeWidgetItem(playlistTree);
            currentNode->setText(0, nodeName);
            currentNode->setFlags(currentNode->flags() | Qt::ItemIsEditable);
            currentNode->setExpanded(true);
        } else {
            if (!currentNode) {
                currentNode = new QTreeWidgetItem(playlistTree);
                currentNode->setText(0, "Default");
                currentNode->setFlags(currentNode->flags() | Qt::ItemIsEditable);
                currentNode->setExpanded(true);
            }
            if (shaderNameToIndex.contains(line)) {
                auto *item = new QTreeWidgetItem(currentNode);
                item->setText(0, line);
                item->setData(0, Qt::UserRole, shaderNameToIndex[line]);
                ++loadedCount;
            } else {
                ++skippedCount;
            }
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
    if (!enableCheckBox->isChecked())
        return false;
    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        if (playlistTree->topLevelItem(i)->childCount() > 0)
            return true;
    }
    return false;
}

QStringList PlaylistDialog::getSelectedShaderNames() const {
    QStringList names;
    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem *node = playlistTree->topLevelItem(i);
        for (int j = 0; j < node->childCount(); ++j) {
            names.append(node->child(j)->text(0));
        }
    }
    return names;
}

QList<QPair<QString, QStringList>> PlaylistDialog::getPlaylistTree() const {
    QList<QPair<QString, QStringList>> tree;
    for (int i = 0; i < playlistTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem *node = playlistTree->topLevelItem(i);
        QStringList shaders;
        for (int j = 0; j < node->childCount(); ++j) {
            shaders.append(node->child(j)->text(0));
        }
        tree.append({node->text(0), shaders});
    }
    return tree;
}

QString PlaylistDialog::getPlaylistFile() const {
    return playlistFilePath;
}

int PlaylistDialog::getAutopilotFrames() const {
    return autopilotFramesSpinBox ? autopilotFramesSpinBox->value() : 0;
}

void PlaylistDialog::setAutopilotFrames(int frames) {
    if (autopilotFramesSpinBox) {
        if (frames < 0) frames = 0;
        autopilotFramesSpinBox->setValue(frames);
    }
}

void PlaylistDialog::setEnabled(bool enabled) {
    enableCheckBox->setChecked(enabled);
}

void PlaylistDialog::setSelectedShaderNames(const QStringList &names) {
    playlistTree->clear();
    if (names.isEmpty())
        return;

    auto *node = new QTreeWidgetItem(playlistTree);
    node->setText(0, "Default");
    node->setFlags(node->flags() | Qt::ItemIsEditable);
    node->setExpanded(true);

    for (const QString &name : names) {
        if (shaderNameToIndex.contains(name)) {
            auto *item = new QTreeWidgetItem(node);
            item->setText(0, name);
            item->setData(0, Qt::UserRole, shaderNameToIndex[name]);
        }
    }
}

void PlaylistDialog::setPlaylistTree(const QList<QPair<QString, QStringList>> &tree) {
    playlistTree->clear();
    for (const auto &[nodeName, shaders] : tree) {
        auto *node = new QTreeWidgetItem(playlistTree);
        node->setText(0, nodeName);
        node->setFlags(node->flags() | Qt::ItemIsEditable);
        node->setExpanded(true);
        for (const QString &name : shaders) {
            if (shaderNameToIndex.contains(name)) {
                auto *item = new QTreeWidgetItem(node);
                item->setText(0, name);
                item->setData(0, Qt::UserRole, shaderNameToIndex[name]);
            }
        }
    }
}

void PlaylistDialog::setPlaylistFile(const QString &path) {
    playlistFilePath = path;
    if (!path.isEmpty()) {
        QFile file(path);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&file);
            bool hasNodes = false;
            while (!in.atEnd()) {
                QString line = in.readLine().trimmed();
                if (line.startsWith('[') && line.endsWith(']')) {
                    hasNodes = true;
                    break;
                }
            }
            file.close();

            if (hasNodes && file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QTextStream in2(&file);
                playlistTree->clear();
                QTreeWidgetItem *currentNode = nullptr;
                while (!in2.atEnd()) {
                    QString line = in2.readLine().trimmed();
                    if (line.isEmpty())
                        continue;
                    if (line.startsWith('[') && line.endsWith(']')) {
                        QString nodeName = line.mid(1, line.length() - 2);
                        currentNode = new QTreeWidgetItem(playlistTree);
                        currentNode->setText(0, nodeName);
                        currentNode->setFlags(currentNode->flags() | Qt::ItemIsEditable);
                        currentNode->setExpanded(true);
                    } else if (currentNode && shaderNameToIndex.contains(line)) {
                        auto *item = new QTreeWidgetItem(currentNode);
                        item->setText(0, line);
                        item->setData(0, Qt::UserRole, shaderNameToIndex[line]);
                    }
                }
                file.close();
            }
        }
    }
}

void PlaylistDialog::updateShaderList(const QStringList &shaderNames) {
    loadShaders(shaderNames);
}
