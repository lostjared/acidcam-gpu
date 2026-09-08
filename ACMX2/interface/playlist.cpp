#include "playlist.hpp"
#include "custom_style.hpp"
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QInputDialog>
#include <QMessageBox>
#include <QSet>
#include <QSettings>
#include <QTextStream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

PlaylistDialog::PlaylistDialog(const QStringList &shaderNames,
                               acmx2::Backend backend, QWidget *parent)
    : QDialog(parent), backend(backend) {
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
    if (backend == acmx2::Backend::Acmxvk) {
        generateRandomButton = new QPushButton("Generate Random...", this);
        generateRandomButton->setToolTip(
            "Create randomized multipass nodes from the active ACMXVK shader "
            "library.");
    }
    fileButtonLayout->addWidget(saveButton);
    fileButtonLayout->addWidget(loadButton);
    fileButtonLayout->addWidget(concatButton);
    if (generateRandomButton)
        fileButtonLayout->addWidget(generateRandomButton);
    fileButtonLayout->addStretch();
    shaderMainLayout->addLayout(fileButtonLayout);

    QHBoxLayout *autopilotLayout = new QHBoxLayout();
    QLabel *autopilotLabel = new QLabel(
        "Autopilot frames (minimum 4; switch to a random shader after this many frames, toggle with J):",
        this);
    autopilotLabel->setWordWrap(true);
    autopilotFramesSpinBox = new QSpinBox(this);
    autopilotFramesSpinBox->setRange(4, 1000000);
    autopilotFramesSpinBox->setSingleStep(30);
    autopilotFramesSpinBox->setValue(4);
    autopilotFramesSpinBox->setSuffix(" frames");
    autopilotRandomCheckBox = new QCheckBox("Random", this);
    autopilotRandomCheckBox->setToolTip("Use --autopilot-random instead of --autopilot-frames");
    autopilotLayout->addWidget(autopilotLabel, 1);
    autopilotLayout->addWidget(autopilotFramesSpinBox);
    autopilotLayout->addWidget(autopilotRandomCheckBox);
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
    if (generateRandomButton) {
        connect(generateRandomButton, &QPushButton::clicked, this,
                &PlaylistDialog::generateRandomPlaylist);
    }
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
        if (generateRandomButton)
            generateRandomButton->setEnabled(checked);
        concatButton->setEnabled(checked);
        saveButton->setEnabled(checked);
        loadButton->setEnabled(checked);
        if (autopilotFramesSpinBox)
            autopilotFramesSpinBox->setEnabled(checked);
        if (autopilotRandomCheckBox)
            autopilotRandomCheckBox->setEnabled(checked);
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
    if (generateRandomButton)
        generateRandomButton->setEnabled(false);
    concatButton->setEnabled(false);
    saveButton->setEnabled(false);
    loadButton->setEnabled(false);
    autopilotFramesSpinBox->setEnabled(false);
    autopilotRandomCheckBox->setEnabled(false);

    acmx2::applyCustomStyleIfEnabled(this);
}

void PlaylistDialog::loadShaders(const QStringList &shaderNames) {
    QStringList selectedNames = getSelectedShaderNames();

    shaderNamesList.clear();
    shaderNameToIndex.clear();
    shaderAliasToName.clear();
    shaderModel->clear();

    for (int i = 0; i < shaderNames.size(); ++i) {
        QString name = shaderNames[i];
        shaderNamesList.append(name);
        shaderNameToIndex[name] = i;
        const QString normalizedName = QDir::fromNativeSeparators(name);
        shaderAliasToName.insert(normalizedName.toLower(), name);
        if (backend == acmx2::Backend::Acmxvk) {
            if (normalizedName.endsWith(QStringLiteral(".spv"),
                                        Qt::CaseInsensitive)) {
                shaderAliasToName.insert(
                    normalizedName.chopped(4).toLower(), name);
            } else {
                shaderAliasToName.insert(
                    (normalizedName + QStringLiteral(".spv")).toLower(), name);
            }
        }
        QStandardItem *item = new QStandardItem(name);
        item->setData(i, Qt::UserRole);
        shaderModel->appendRow(item);
    }

    if (!selectedNames.isEmpty()) {
        setSelectedShaderNames(selectedNames);
    }
}

QString
PlaylistDialog::resolvePlaylistShaderName(const QString &name) const {
    const QString key = QDir::fromNativeSeparators(name.trimmed()).toLower();
    return shaderAliasToName.value(key);
}

QString
PlaylistDialog::runtimePlaylistShaderName(const QString &name) const {
    if (backend == acmx2::Backend::Acmxvk &&
        !name.endsWith(QStringLiteral(".spv"), Qt::CaseInsensitive)) {
        return name + QStringLiteral(".spv");
    }
    return name;
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

void PlaylistDialog::generateRandomPlaylist() {
    constexpr int MAX_PLAYLIST_NODES = 10000;
    constexpr int MAX_PLAYLIST_ENTRIES = 65536;

    QStringList availableShaders;
    QSet<QString> seenShaders;
    for (const QString &shaderName : shaderNamesList) {
        const QString normalized = QDir::fromNativeSeparators(shaderName);
        if (normalized.trimmed().isEmpty() ||
            seenShaders.contains(normalized)) {
            continue;
        }
        seenShaders.insert(normalized);
        availableShaders.append(shaderName);
    }
    if (availableShaders.isEmpty()) {
        QMessageBox::information(
            this, "No Shaders",
            "Load an ACMXVK shader library before generating a playlist.");
        return;
    }

    QSettings appSettings("LostSideDead");
    QDialog optionsDialog(this);
    optionsDialog.setWindowTitle("Generate Random ACMXVK Playlist");
    auto *layout = new QVBoxLayout(&optionsDialog);
    auto *description = new QLabel(
        "Each node receives a random selection of unique shaders from the "
        "active library.",
        &optionsDialog);
    description->setWordWrap(true);
    layout->addWidget(description);

    auto *form = new QFormLayout();
    auto *nodeCount = new QSpinBox(&optionsDialog);
    nodeCount->setRange(1, MAX_PLAYLIST_NODES);
    nodeCount->setValue(
        appSettings.value("playlist/random_nodes", 100).toInt());
    nodeCount->setSuffix(" nodes");
    form->addRow("Playlist nodes:", nodeCount);

    auto *maximumShaders = new QSpinBox(&optionsDialog);
    maximumShaders->setRange(
        1, std::min(static_cast<int>(availableShaders.size()),
                    MAX_PLAYLIST_ENTRIES));
    maximumShaders->setValue(std::min(
        appSettings.value("playlist/random_max_shaders", 4).toInt(),
        maximumShaders->maximum()));
    maximumShaders->setSuffix(" shaders");
    form->addRow("Maximum per node:", maximumShaders);

    auto *fixedSeed = new QCheckBox("Use repeatable seed", &optionsDialog);
    fixedSeed->setChecked(
        appSettings.value("playlist/random_fixed_seed", false).toBool());
    form->addRow(QString(), fixedSeed);

    auto *seedValue = new QSpinBox(&optionsDialog);
    seedValue->setRange(0, std::numeric_limits<int>::max());
    seedValue->setValue(
        appSettings.value("playlist/random_seed", 1).toInt());
    seedValue->setEnabled(fixedSeed->isChecked());
    form->addRow("Seed:", seedValue);
    layout->addLayout(form);

    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &optionsDialog);
    buttons->button(QDialogButtonBox::Ok)->setText("Generate");
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &optionsDialog,
            &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &optionsDialog,
            &QDialog::reject);
    connect(fixedSeed, &QCheckBox::toggled, seedValue,
            &QWidget::setEnabled);
    acmx2::applyCustomStyleIfEnabled(&optionsDialog);

    if (optionsDialog.exec() != QDialog::Accepted)
        return;
    if (playlistTree->topLevelItemCount() > 0 &&
        QMessageBox::question(
            this, "Replace Playlist",
            "Generating a random playlist will replace the current playlist "
            "tree. Continue?",
            QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes) {
        return;
    }

    appSettings.setValue("playlist/random_nodes", nodeCount->value());
    appSettings.setValue("playlist/random_max_shaders",
                         maximumShaders->value());
    appSettings.setValue("playlist/random_fixed_seed",
                         fixedSeed->isChecked());
    appSettings.setValue("playlist/random_seed", seedValue->value());

    std::random_device seedSource;
    const unsigned int seed =
        fixedSeed->isChecked()
            ? static_cast<unsigned int>(seedValue->value())
            : std::uniform_int_distribution<unsigned int>(
                  0U, static_cast<unsigned int>(
                          std::numeric_limits<int>::max()))(seedSource);
    std::mt19937 randomGenerator(seed);
    std::vector<int> shaderIndices(
        static_cast<std::size_t>(availableShaders.size()));
    std::iota(shaderIndices.begin(), shaderIndices.end(), 0);

    playlistTree->setUpdatesEnabled(false);
    playlistTree->clear();
    int totalEntries = 0;
    for (int nodeIndex = 1; nodeIndex <= nodeCount->value(); ++nodeIndex) {
        const int remainingNodes = nodeCount->value() - nodeIndex;
        const int remainingCapacity = MAX_PLAYLIST_ENTRIES - totalEntries;
        const int nodeMaximum =
            std::min(maximumShaders->value(),
                     remainingCapacity - remainingNodes);
        const int shaderCount =
            std::uniform_int_distribution<int>(1, nodeMaximum)(
                randomGenerator);

        for (int selectionIndex = 0; selectionIndex < shaderCount;
             ++selectionIndex) {
            const int swapIndex = std::uniform_int_distribution<int>(
                selectionIndex,
                static_cast<int>(shaderIndices.size()) - 1)(randomGenerator);
            std::swap(shaderIndices[static_cast<std::size_t>(selectionIndex)],
                      shaderIndices[static_cast<std::size_t>(swapIndex)]);
        }

        auto *node = new QTreeWidgetItem(playlistTree);
        node->setText(0, QStringLiteral("Random %1")
                             .arg(nodeIndex, 4, 10, QLatin1Char('0')));
        node->setFlags(node->flags() | Qt::ItemIsEditable);
        node->setExpanded(nodeCount->value() <= 100);
        for (int selectionIndex = 0; selectionIndex < shaderCount;
             ++selectionIndex) {
            const QString &shaderName = availableShaders.at(
                shaderIndices[static_cast<std::size_t>(selectionIndex)]);
            auto *item = new QTreeWidgetItem(node);
            item->setText(0, shaderName);
            if (shaderNameToIndex.contains(shaderName)) {
                item->setData(0, Qt::UserRole,
                              shaderNameToIndex.value(shaderName));
            }
        }
        totalEntries += shaderCount;
    }
    playlistTree->setUpdatesEnabled(true);
    playlistFilePath.clear();

    QMessageBox::information(
        this, "Random Playlist Generated",
        QStringLiteral("Created %1 nodes with %2 shader entries.\nSeed: %3")
            .arg(nodeCount->value())
            .arg(totalEntries)
            .arg(seed));
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
        if (line.isEmpty() || line.startsWith('#'))
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
            const QString shaderName = resolvePlaylistShaderName(line);
            if (!shaderName.isEmpty()) {
                auto *item = new QTreeWidgetItem(currentNode);
                item->setText(0, shaderName);
                item->setData(0, Qt::UserRole,
                              shaderNameToIndex[shaderName]);
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
            out << runtimePlaylistShaderName(node->child(j)->text(0))
                << "\n";
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
        if (line.isEmpty() || line.startsWith('#'))
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
            const QString shaderName = resolvePlaylistShaderName(line);
            if (!shaderName.isEmpty()) {
                auto *item = new QTreeWidgetItem(currentNode);
                item->setText(0, shaderName);
                item->setData(0, Qt::UserRole,
                              shaderNameToIndex[shaderName]);
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

bool PlaylistDialog::isAutopilotRandom() const {
    return autopilotRandomCheckBox ? autopilotRandomCheckBox->isChecked() : false;
}

void PlaylistDialog::setAutopilotFrames(int frames) {
    if (autopilotFramesSpinBox) {
        if (frames < 4)
            frames = 4;
        autopilotFramesSpinBox->setValue(frames);
    }
}

void PlaylistDialog::setAutopilotRandom(bool enabled) {
    if (autopilotRandomCheckBox) {
        autopilotRandomCheckBox->setChecked(enabled);
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
        const QString shaderName = resolvePlaylistShaderName(name);
        if (!shaderName.isEmpty()) {
            auto *item = new QTreeWidgetItem(node);
            item->setText(0, shaderName);
            item->setData(0, Qt::UserRole,
                          shaderNameToIndex[shaderName]);
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
            const QString shaderName = resolvePlaylistShaderName(name);
            if (!shaderName.isEmpty()) {
                auto *item = new QTreeWidgetItem(node);
                item->setText(0, shaderName);
                item->setData(0, Qt::UserRole,
                              shaderNameToIndex[shaderName]);
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
                    if (line.isEmpty() || line.startsWith('#'))
                        continue;
                    if (line.startsWith('[') && line.endsWith(']')) {
                        QString nodeName = line.mid(1, line.length() - 2);
                        currentNode = new QTreeWidgetItem(playlistTree);
                        currentNode->setText(0, nodeName);
                        currentNode->setFlags(currentNode->flags() | Qt::ItemIsEditable);
                        currentNode->setExpanded(true);
                    } else if (currentNode) {
                        const QString shaderName =
                            resolvePlaylistShaderName(line);
                        if (shaderName.isEmpty())
                            continue;
                        auto *item = new QTreeWidgetItem(currentNode);
                        item->setText(0, shaderName);
                        item->setData(0, Qt::UserRole,
                                      shaderNameToIndex[shaderName]);
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
