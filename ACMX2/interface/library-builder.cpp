#include "library-builder.hpp"

#include "custom_style.hpp"
#include "shader-manifest.hpp"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QSettings>
#include <QVBoxLayout>

namespace {
    constexpr int SOURCE_PATH_ROLE = Qt::UserRole;

    bool is_shader_file(const QFileInfo &fileInfo) {
        const QString suffix = fileInfo.suffix();
        return fileInfo.isFile() && fileInfo.isReadable() &&
               (suffix.compare("glsl", Qt::CaseInsensitive) == 0 ||
                suffix.compare("comp", Qt::CaseInsensitive) == 0);
    }

    QString normalized_path(const QString &path) {
        QFileInfo fileInfo(path);
        const QString canonicalPath = fileInfo.canonicalFilePath();
        return canonicalPath.isEmpty() ? fileInfo.absoluteFilePath() : canonicalPath;
    }
} // namespace

LibraryBuilderDialog::LibraryBuilderDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle(tr("Shader Library Builder"));
    setMinimumSize(720, 520);

    auto *layout = new QVBoxLayout(this);
    auto *intro = new QLabel(
        tr("Add fragment (.glsl) and compute (.comp) shaders. The list stays "
           "sorted automatically and exports as a portable library.json."),
        this);
    intro->setWordWrap(true);
    layout->addWidget(intro);

    auto *addRow = new QHBoxLayout();
    auto *addFilesButton = new QPushButton(tr("Add Files..."), this);
    auto *addFolderButton = new QPushButton(tr("Add Folder..."), this);
    auto *openButton = new QPushButton(tr("Open Library..."), this);
    recursiveCheck = new QCheckBox(tr("Include subfolders"), this);
    recursiveCheck->setChecked(true);
    addRow->addWidget(addFilesButton);
    addRow->addWidget(addFolderButton);
    addRow->addWidget(openButton);
    addRow->addStretch();
    addRow->addWidget(recursiveCheck);
    layout->addLayout(addRow);

    shaderList = new QListWidget(this);
    shaderList->setAlternatingRowColors(false);
    shaderList->setSortingEnabled(true);
    shaderList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    shaderList->setToolTip(tr("Shaders are kept in alphabetical order."));
    layout->addWidget(shaderList, 1);

    auto *editRow = new QHBoxLayout();
    removeButton = new QPushButton(tr("Remove"), this);
    auto *clearButton = new QPushButton(tr("Clear"), this);
    editRow->addWidget(removeButton);
    editRow->addStretch();
    editRow->addWidget(clearButton);
    layout->addLayout(editRow);

    summaryLabel = new QLabel(this);
    layout->addWidget(summaryLabel);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close, this);
    exportButton = buttonBox->addButton(tr("Export Library..."),
                                        QDialogButtonBox::ActionRole);
    layout->addWidget(buttonBox);

    connect(addFilesButton, &QPushButton::clicked, this,
            &LibraryBuilderDialog::addFiles);
    connect(addFolderButton, &QPushButton::clicked, this,
            &LibraryBuilderDialog::addFolder);
    connect(openButton, &QPushButton::clicked, this,
            &LibraryBuilderDialog::openLibrary);
    connect(removeButton, &QPushButton::clicked, this,
            &LibraryBuilderDialog::removeSelected);
    connect(clearButton, &QPushButton::clicked, this,
            &LibraryBuilderDialog::clearShaders);
    connect(exportButton, &QPushButton::clicked, this,
            &LibraryBuilderDialog::exportLibrary);
    connect(buttonBox, &QDialogButtonBox::rejected, this,
            &QDialog::reject);
    connect(shaderList, &QListWidget::itemSelectionChanged, this,
            &LibraryBuilderDialog::updateControls);

    updateControls();
    acmx2::applyCustomStyleIfEnabled(this);
}

QString LibraryBuilderDialog::shaderFilter() const {
    return tr("Shader files (*.glsl *.comp);;Fragment shaders (*.glsl);;Compute "
              "shaders (*.comp)");
}

void LibraryBuilderDialog::addFiles() {
    QSettings settings("LostSideDead");
    const QString startDir = settings.value("libraryBuilder/sourceDir").toString();
    const QStringList files = QFileDialog::getOpenFileNames(
        this, tr("Add Shader Files"), startDir, shaderFilter());
    if (files.isEmpty())
        return;

    settings.setValue("libraryBuilder/sourceDir", QFileInfo(files.first()).absolutePath());
    addShaderFiles(files);
}

void LibraryBuilderDialog::addFolder() {
    QSettings settings("LostSideDead");
    const QString startDir = settings.value("libraryBuilder/sourceDir").toString();
    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Add Shader Folder"), startDir);
    if (directory.isEmpty())
        return;

    settings.setValue("libraryBuilder/sourceDir", directory);
    const auto flags = recursiveCheck->isChecked() ? QDirIterator::Subdirectories
                                                   : QDirIterator::NoIteratorFlags;
    QDirIterator iterator(directory, {"*.glsl", "*.comp"}, QDir::Files, flags);
    QStringList files;
    while (iterator.hasNext())
        files.append(iterator.next());
    files.sort(Qt::CaseInsensitive);

    const int added = addShaderFiles(files);
    if (files.isEmpty()) {
        QMessageBox::information(this, tr("No Shaders Found"),
                                 tr("No .glsl or .comp files were found in %1.")
                                     .arg(QDir::toNativeSeparators(directory)));
    } else if (added == 0) {
        QMessageBox::information(this, tr("No Shaders Added"),
                                 tr("All shaders in that folder are already in the list."));
    }
}

void LibraryBuilderDialog::openLibrary() {
    QSettings settings("LostSideDead");
    const QString startDir = settings.value("libraryBuilder/sourceDir").toString();
    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Open Shader Library"), startDir);
    if (directory.isEmpty())
        return;

    QStringList shaderNames;
    QString error;
    if (!acmx2::load_shader_manifest(directory, shaderNames, error)) {
        QMessageBox::critical(this, tr("Could Not Open Library"), error);
        return;
    }

    if (shaderList->count() > 0 &&
        QMessageBox::question(this, tr("Replace Current List"),
                              tr("Replace the current shader list with this library?")) !=
            QMessageBox::Yes) {
        return;
    }

    shaderList->clear();
    QStringList missing;
    for (const QString &shaderName : shaderNames) {
        const QString path = QDir(directory).filePath(shaderName);
        if (!addShader(path, false))
            missing.append(shaderName);
    }
    settings.setValue("libraryBuilder/sourceDir", directory);
    updateControls();

    if (!missing.isEmpty()) {
        QMessageBox::warning(
            this, tr("Some Shaders Were Skipped"),
            tr("%1 entries were missing, unreadable, duplicated, or not .glsl/.comp files.")
                .arg(missing.size()));
    }
}

bool LibraryBuilderDialog::addShader(const QString &filePath, bool showErrors) {
    const QFileInfo fileInfo(filePath);
    if (!is_shader_file(fileInfo)) {
        if (showErrors) {
            QMessageBox::warning(this, tr("Invalid Shader"),
                                 tr("%1 is not a readable .glsl or .comp file.")
                                     .arg(QDir::toNativeSeparators(filePath)));
        }
        return false;
    }

    const QString sourcePath = normalized_path(filePath);
    for (int index = 0; index < shaderList->count(); ++index) {
        if (shaderList->item(index)->data(SOURCE_PATH_ROLE).toString().compare(sourcePath, Qt::CaseSensitive) == 0) {
            return false;
        }
    }

    const bool compute = fileInfo.suffix().compare("comp", Qt::CaseInsensitive) == 0;
    auto *item = new QListWidgetItem(
        QString("%1    [%2]").arg(fileInfo.fileName(), compute ? tr("Compute") : tr("Fragment")),
        shaderList);
    item->setData(SOURCE_PATH_ROLE, sourcePath);
    item->setToolTip(QDir::toNativeSeparators(sourcePath));
    return true;
}

int LibraryBuilderDialog::addShaderFiles(const QStringList &filePaths) {
    int added = 0;
    for (const QString &filePath : filePaths)
        added += addShader(filePath, filePaths.size() == 1) ? 1 : 0;
    updateControls();
    return added;
}

QStringList LibraryBuilderDialog::selectedSourcePaths() const {
    QStringList paths;
    for (QListWidgetItem *item : shaderList->selectedItems())
        paths.append(item->data(SOURCE_PATH_ROLE).toString());
    return paths;
}

void LibraryBuilderDialog::removeSelected() {
    const QStringList selectedPaths = selectedSourcePaths();
    for (int row = shaderList->count() - 1; row >= 0; --row) {
        if (selectedPaths.contains(
                shaderList->item(row)->data(SOURCE_PATH_ROLE).toString())) {
            delete shaderList->takeItem(row);
        }
    }
    updateControls();
}

void LibraryBuilderDialog::clearShaders() {
    if (shaderList->count() == 0)
        return;
    if (QMessageBox::question(this, tr("Clear Shader List"),
                              tr("Remove all shaders from the builder?")) ==
        QMessageBox::Yes) {
        shaderList->clear();
        updateControls();
    }
}

QString LibraryBuilderDialog::uniqueExportName(
    const QString &fileName, const QStringList &usedNames) const {
    if (!usedNames.contains(fileName, Qt::CaseInsensitive))
        return fileName;

    const QFileInfo fileInfo(fileName);
    const QString stem = fileInfo.completeBaseName();
    const QString suffix = fileInfo.suffix();
    for (int number = 2;; ++number) {
        const QString candidate = QString("%1-%2.%3").arg(stem).arg(number).arg(suffix);
        if (!usedNames.contains(candidate, Qt::CaseInsensitive))
            return candidate;
    }
}

bool LibraryBuilderDialog::copyShader(const QString &sourcePath,
                                      const QString &destinationPath,
                                      QString &error) const {
    if (normalized_path(sourcePath) == normalized_path(destinationPath))
        return true;

    QFile source(sourcePath);
    if (!source.open(QIODevice::ReadOnly)) {
        error = tr("Could not read %1: %2").arg(sourcePath, source.errorString());
        return false;
    }

    QSaveFile destination(destinationPath);
    if (!destination.open(QIODevice::WriteOnly)) {
        error = tr("Could not write %1: %2")
                    .arg(destinationPath, destination.errorString());
        return false;
    }

    while (!source.atEnd()) {
        const QByteArray block = source.read(64 * 1024);
        if (block.isEmpty() && source.error() != QFile::NoError) {
            error = tr("Could not finish reading %1: %2")
                        .arg(sourcePath, source.errorString());
            destination.cancelWriting();
            return false;
        }
        if (destination.write(block) != block.size()) {
            error = tr("Could not finish writing %1: %2")
                        .arg(destinationPath, destination.errorString());
            destination.cancelWriting();
            return false;
        }
    }

    if (!destination.commit()) {
        error = tr("Could not finish writing %1: %2")
                    .arg(destinationPath, destination.errorString());
        return false;
    }
    return true;
}

void LibraryBuilderDialog::exportLibrary() {
    if (shaderList->count() == 0)
        return;

    QSettings settings("LostSideDead");
    const QString startDir = settings.value("libraryBuilder/exportDir").toString();
    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Choose Library Export Folder"), startDir,
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (directory.isEmpty())
        return;

    const QString manifestPath = QDir(directory).filePath("library.json");
    const bool replacingLibrary = QFileInfo::exists(manifestPath);
    if (replacingLibrary &&
        QMessageBox::question(this, tr("Replace Existing Library"),
                              tr("This folder already contains library.json. Replace "
                                 "the manifest and matching shader files?")) !=
            QMessageBox::Yes) {
        return;
    }

    QStringList exportedNames;
    QStringList usedNames;
    if (!replacingLibrary) {
        QDir exportDir(directory);
        const QStringList existingFiles = exportDir.entryList(QDir::Files);
        QStringList sourcePaths;
        for (int row = 0; row < shaderList->count(); ++row) {
            sourcePaths.append(
                shaderList->item(row)->data(SOURCE_PATH_ROLE).toString());
        }
        for (const QString &existingFile : existingFiles) {
            const QString existingPath = exportDir.filePath(existingFile);
            bool isSelectedSource = false;
            for (const QString &sourcePath : sourcePaths) {
                if (normalized_path(existingPath) == normalized_path(sourcePath)) {
                    isSelectedSource = true;
                    break;
                }
            }
            if (!isSelectedSource)
                usedNames.append(existingFile);
        }
    }
    QString error;
    for (int row = 0; row < shaderList->count(); ++row) {
        const QString sourcePath =
            shaderList->item(row)->data(SOURCE_PATH_ROLE).toString();
        const QFileInfo sourceInfo(sourcePath);
        if (!is_shader_file(sourceInfo)) {
            QMessageBox::critical(
                this, tr("Export Failed"),
                tr("A source shader is now missing or unreadable:\n%1")
                    .arg(QDir::toNativeSeparators(sourcePath)));
            return;
        }

        const QString exportName =
            uniqueExportName(sourceInfo.fileName(), usedNames);
        const QString destinationPath = QDir(directory).filePath(exportName);
        if (!copyShader(sourcePath, destinationPath, error)) {
            QMessageBox::critical(this, tr("Export Failed"), error);
            return;
        }
        exportedNames.append(exportName);
        usedNames.append(exportName);
    }

    if (!acmx2::create_shader_manifest(directory,
                                       acmx2::ShaderManifestFormat::Json,
                                       exportedNames, error)) {
        QMessageBox::critical(this, tr("Export Failed"), error);
        return;
    }

    settings.setValue("libraryBuilder/exportDir", directory);
    emit libraryExported(directory);
    QMessageBox::information(
        this, tr("Library Exported"),
        tr("Exported %1 shaders and library.json to:\n%2")
            .arg(exportedNames.size())
            .arg(QDir::toNativeSeparators(directory)));
}

void LibraryBuilderDialog::updateControls() {
    int fragmentCount = 0;
    int computeCount = 0;
    for (int row = 0; row < shaderList->count(); ++row) {
        const QString path = shaderList->item(row)->data(SOURCE_PATH_ROLE).toString();
        if (QFileInfo(path).suffix().compare("comp", Qt::CaseInsensitive) == 0)
            ++computeCount;
        else
            ++fragmentCount;
    }

    summaryLabel->setText(
        tr("%1 shaders — %2 fragment, %3 compute")
            .arg(shaderList->count())
            .arg(fragmentCount)
            .arg(computeCount));
    const bool hasSelection = !shaderList->selectedItems().isEmpty();
    removeButton->setEnabled(hasSelection);
    exportButton->setEnabled(shaderList->count() > 0);
}
