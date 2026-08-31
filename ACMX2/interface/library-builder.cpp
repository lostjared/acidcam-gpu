#include "library-builder.hpp"

#include "acmxvk-source-manifest.hpp"
#include "custom_style.hpp"
#include "shader-manifest.hpp"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QSettings>
#include <QTemporaryDir>
#include <QVBoxLayout>
#include <QtConcurrent>

namespace {
    constexpr int SOURCE_PATH_ROLE = Qt::UserRole;

    struct ExportResult {
        bool success = false;
        QStringList exportedNames;
        QString error;
    };

    bool is_shader_file(const QFileInfo &fileInfo, acmx2::Backend backend) {
        const QString suffix = fileInfo.suffix();
        if (!fileInfo.isFile() || !fileInfo.isReadable())
            return false;
        if (suffix.compare("comp", Qt::CaseInsensitive) == 0)
            return true;
        return suffix.compare(backend == acmx2::Backend::Acmxvk ? "frag"
                                                                : "glsl",
                              Qt::CaseInsensitive) == 0;
    }

    QString normalized_path(const QString &path) {
        QFileInfo fileInfo(path);
        const QString canonicalPath = fileInfo.canonicalFilePath();
        return canonicalPath.isEmpty() ? fileInfo.absoluteFilePath() : canonicalPath;
    }

    QString unique_export_name(const QString &fileName,
                               const QStringList &usedNames) {
        if (!usedNames.contains(fileName, Qt::CaseInsensitive))
            return fileName;

        const QFileInfo fileInfo(fileName);
        const QString stem = fileInfo.completeBaseName();
        const QString suffix = fileInfo.suffix();
        for (int number = 2;; ++number) {
            const QString candidate =
                QString("%1-%2.%3").arg(stem).arg(number).arg(suffix);
            if (!usedNames.contains(candidate, Qt::CaseInsensitive))
                return candidate;
        }
    }

    bool copy_shader(const QString &sourcePath, const QString &destinationPath,
                     QString &error) {
        if (normalized_path(sourcePath) == normalized_path(destinationPath))
            return true;

        QFile source(sourcePath);
        if (!source.open(QIODevice::ReadOnly)) {
            error = QObject::tr("Could not read %1: %2")
                        .arg(sourcePath, source.errorString());
            return false;
        }

        QSaveFile destination(destinationPath);
        if (!destination.open(QIODevice::WriteOnly)) {
            error = QObject::tr("Could not write %1: %2")
                        .arg(destinationPath, destination.errorString());
            return false;
        }

        while (!source.atEnd()) {
            const QByteArray block = source.read(64 * 1024);
            if (block.isEmpty() && source.error() != QFile::NoError) {
                error = QObject::tr("Could not finish reading %1: %2")
                            .arg(sourcePath, source.errorString());
                destination.cancelWriting();
                return false;
            }
            if (destination.write(block) != block.size()) {
                error = QObject::tr("Could not finish writing %1: %2")
                            .arg(destinationPath, destination.errorString());
                destination.cancelWriting();
                return false;
            }
        }

        if (!destination.commit()) {
            error = QObject::tr("Could not finish writing %1: %2")
                        .arg(destinationPath, destination.errorString());
            return false;
        }
        return true;
    }

    ExportResult export_shader_library(const QString &directory,
                                       bool replacingLibrary,
                                       const QStringList &sourcePaths,
                                       acmx2::Backend backend) {
        ExportResult result;
        QStringList usedFragmentNames;
        QStringList usedComputeNames;
        if (!replacingLibrary) {
            auto reserveExisting = [&](const QDir &outputDirectory,
                                       QStringList &usedNames,
                                       bool computeDirectory) {
                const QStringList existingFiles =
                    outputDirectory.entryList(QDir::Files);
                for (const QString &existingFile : existingFiles) {
                    const QFileInfo existingInfo(
                        outputDirectory.filePath(existingFile));
                    if (!is_shader_file(existingInfo, backend))
                        continue;
                    if (backend == acmx2::Backend::Acmxvk) {
                        const bool compute = existingInfo.suffix().compare(
                                                 "comp", Qt::CaseInsensitive) == 0;
                        if (compute != computeDirectory)
                            continue;
                    }
                    bool isSelectedSource = false;
                    for (const QString &sourcePath : sourcePaths) {
                        if (normalized_path(existingInfo.absoluteFilePath()) ==
                            normalized_path(sourcePath)) {
                            isSelectedSource = true;
                            break;
                        }
                    }
                    if (!isSelectedSource)
                        usedNames.append(existingFile);
                }
            };
            reserveExisting(QDir(directory), usedFragmentNames, false);
            if (backend == acmx2::Backend::Acmxvk)
                reserveExisting(QDir(QDir(directory).filePath("compute")),
                                usedComputeNames, true);
        }

        QTemporaryDir stagingDirectory;
        if (backend == acmx2::Backend::Acmxvk &&
            !stagingDirectory.isValid()) {
            result.error = QObject::tr(
                "Could not create a temporary ACMXVK manifest directory.");
            return result;
        }

        for (const QString &sourcePath : sourcePaths) {
            const QFileInfo sourceInfo(sourcePath);
            if (!is_shader_file(sourceInfo, backend)) {
                result.error =
                    QObject::tr("A source shader is now missing or unreadable:\n%1")
                        .arg(QDir::toNativeSeparators(sourcePath));
                return result;
            }

            const bool compute =
                sourceInfo.suffix().compare("comp", Qt::CaseInsensitive) == 0;
            QStringList &usedNames =
                backend == acmx2::Backend::Acmxvk && compute
                    ? usedComputeNames
                    : usedFragmentNames;
            const QString preferredName =
                backend == acmx2::Backend::Acmxvk
                    ? sourceInfo.completeBaseName() +
                          (compute ? QStringLiteral(".comp")
                                   : QStringLiteral(".frag"))
                    : sourceInfo.fileName();
            const QString exportName =
                unique_export_name(preferredName, usedNames);
            const QString relativeName =
                backend == acmx2::Backend::Acmxvk && compute
                    ? QStringLiteral("compute/") + exportName
                    : exportName;
            const QString destinationPath = QDir(directory).filePath(relativeName);
            if (!QDir().mkpath(QFileInfo(destinationPath).absolutePath())) {
                result.error = QObject::tr("Could not create shader directory: %1")
                                   .arg(QFileInfo(destinationPath).absolutePath());
                return result;
            }
            if (!copy_shader(sourcePath, destinationPath, result.error))
                return result;
            result.exportedNames.append(relativeName);
            usedNames.append(exportName);

            if (backend == acmx2::Backend::Acmxvk) {
                const QString stagingPath =
                    QDir(stagingDirectory.path()).filePath(relativeName);
                if (!QDir().mkpath(QFileInfo(stagingPath).absolutePath())) {
                    result.error = QObject::tr(
                                       "Could not create temporary shader directory: %1")
                                       .arg(QFileInfo(stagingPath).absolutePath());
                    return result;
                }
                if (!copy_shader(sourcePath, stagingPath, result.error))
                    return result;
            }
        }

        if (backend == acmx2::Backend::Acmxvk) {
            acmx2::AcmxvkSourceManifestResult manifestResult;
            if (!acmx2::create_acmxvk_source_manifest(
                    stagingDirectory.path(),
                    QDir(directory).filePath(QStringLiteral("library.json")),
                    manifestResult, result.error)) {
                return result;
            }
        } else {
            if (!acmx2::create_shader_manifest(
                    directory, acmx2::ShaderManifestFormat::Json,
                    result.exportedNames, result.error)) {
                return result;
            }
        }

        result.success = true;
        return result;
    }
} // namespace

LibraryBuilderDialog::LibraryBuilderDialog(acmx2::Backend selectedBackend,
                                           QWidget *parent)
    : QDialog(parent), backend(selectedBackend) {
    setWindowTitle(tr("%1 Shader Library Builder")
                       .arg(acmx2::backend_name(backend)));
    setMinimumSize(720, 520);

    auto *layout = new QVBoxLayout(this);
    auto *intro = new QLabel(this);
    intro->setText(
        backend == acmx2::Backend::Acmxvk
            ? tr("Add Vulkan fragment (.frag) and compute (.comp) sources. "
                 "Compute files export beneath compute/, and the native ACMXVK "
                 "generator creates a source library.json with custom-uniform "
                 "metadata.")
            : tr("Add fragment (.glsl) and compute (.comp) shaders. The list "
                 "stays sorted automatically and exports as a portable "
                 "library.json."));
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

acmx2::Backend LibraryBuilderDialog::selectedBackend() const {
    return backend;
}

QString LibraryBuilderDialog::shaderFilter() const {
    return backend == acmx2::Backend::Acmxvk
               ? tr("Shader files (*.frag *.comp);;Fragment shaders "
                    "(*.frag);;Compute shaders (*.comp)")
               : tr("Shader files (*.glsl *.comp);;Fragment shaders "
                    "(*.glsl);;Compute shaders (*.comp)");
}

void LibraryBuilderDialog::addFiles() {
    QSettings settings("LostSideDead");
    const QString sourceKey =
        acmx2::backend_settings_key(backend, "library_builder_source_dir");
    const QString startDir = settings.value(sourceKey).toString();
    const QStringList files = QFileDialog::getOpenFileNames(
        this, tr("Add Shader Files"), startDir, shaderFilter());
    if (files.isEmpty())
        return;

    settings.setValue(sourceKey, QFileInfo(files.first()).absolutePath());
    addShaderFiles(files);
}

void LibraryBuilderDialog::addFolder() {
    QSettings settings("LostSideDead");
    const QString sourceKey =
        acmx2::backend_settings_key(backend, "library_builder_source_dir");
    const QString startDir = settings.value(sourceKey).toString();
    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Add Shader Folder"), startDir);
    if (directory.isEmpty())
        return;

    settings.setValue(sourceKey, directory);
    const auto flags = recursiveCheck->isChecked() ? QDirIterator::Subdirectories
                                                   : QDirIterator::NoIteratorFlags;
    const QStringList nameFilters =
        backend == acmx2::Backend::Acmxvk
            ? QStringList{QStringLiteral("*.frag"), QStringLiteral("*.comp")}
            : QStringList{QStringLiteral("*.glsl"), QStringLiteral("*.comp")};
    QDirIterator iterator(directory, nameFilters, QDir::Files, flags);
    QStringList files;
    while (iterator.hasNext())
        files.append(iterator.next());
    files.sort(Qt::CaseInsensitive);

    const int added = addShaderFiles(files);
    if (files.isEmpty()) {
        QMessageBox::information(this, tr("No Shaders Found"),
                                 (backend == acmx2::Backend::Acmxvk
                                      ? tr("No .frag or .comp files were found in %1.")
                                      : tr("No .glsl or .comp files were found in %1."))
                                     .arg(QDir::toNativeSeparators(directory)));
    } else if (added == 0) {
        QMessageBox::information(this, tr("No Shaders Added"),
                                 tr("All shaders in that folder are already in the list."));
    }
}

void LibraryBuilderDialog::openLibrary() {
    QSettings settings("LostSideDead");
    const QString sourceKey =
        acmx2::backend_settings_key(backend, "library_builder_source_dir");
    const QString startDir = settings.value(sourceKey).toString();
    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Open Shader Library"), startDir);
    if (directory.isEmpty())
        return;

    QString metadataError;
    const std::optional<acmx2::Backend> manifestBackend =
        acmx2::shader_manifest_backend(directory, metadataError);
    if (!metadataError.isEmpty()) {
        QMessageBox::critical(this, tr("Could Not Open Library"), metadataError);
        return;
    }
    if (manifestBackend && *manifestBackend != backend) {
        QMessageBox::warning(
            this, tr("Wrong Library Backend"),
            tr("This library targets %1, but the builder is in %2 mode.")
                .arg(acmx2::backend_name(*manifestBackend),
                     acmx2::backend_name(backend)));
        return;
    }
    if (backend == acmx2::Backend::Acmxvk) {
        const std::optional<acmx2::ShaderLibraryType> libraryType =
            acmx2::shader_manifest_library_type(directory, metadataError);
        if (!metadataError.isEmpty()) {
            QMessageBox::critical(this, tr("Could Not Open Library"),
                                  metadataError);
            return;
        }
        if (libraryType &&
            *libraryType == acmx2::ShaderLibraryType::Runtime) {
            QMessageBox::warning(
                this, tr("Runtime Library Not Supported"),
                tr("The ACMXVK library builder accepts source libraries with "
                   ".frag and .comp files, not compiled SPIR-V runtime "
                   "libraries."));
            return;
        }
    }

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
    settings.setValue(sourceKey, directory);
    updateControls();

    if (!missing.isEmpty()) {
        QMessageBox::warning(
            this, tr("Some Shaders Were Skipped"),
            (backend == acmx2::Backend::Acmxvk
                 ? tr("%1 entries were missing, unreadable, duplicated, or not "
                      ".frag/.comp files.")
                 : tr("%1 entries were missing, unreadable, duplicated, or not "
                      ".glsl/.comp files."))
                .arg(missing.size()));
    }
}

bool LibraryBuilderDialog::addShader(const QString &filePath, bool showErrors) {
    const QFileInfo fileInfo(filePath);
    if (!is_shader_file(fileInfo, backend)) {
        if (showErrors) {
            QMessageBox::warning(this, tr("Invalid Shader"),
                                 (backend == acmx2::Backend::Acmxvk
                                      ? tr("%1 is not a readable .frag or .comp file.")
                                      : tr("%1 is not a readable .glsl or .comp file."))
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

void LibraryBuilderDialog::exportLibrary() {
    if (exportInProgress || shaderList->count() == 0)
        return;

    QSettings settings("LostSideDead");
    const QString exportKey =
        acmx2::backend_settings_key(backend, "library_builder_export_dir");
    const QString startDir = settings.value(exportKey).toString();
    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Choose Library Export Folder"), startDir,
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (directory.isEmpty())
        return;

    const QString manifestPath = QDir(directory).filePath("library.json");
    const bool replacingLibrary = QFileInfo::exists(manifestPath);
    if (replacingLibrary) {
        QString backendError;
        const std::optional<acmx2::Backend> existingBackend =
            acmx2::shader_manifest_backend(directory, backendError);
        if (!backendError.isEmpty()) {
            QMessageBox::critical(this, tr("Invalid Existing Library"),
                                  backendError);
            return;
        }
        if (existingBackend && *existingBackend != backend) {
            QMessageBox::warning(
                this, tr("Wrong Export Backend"),
                tr("The selected folder contains a %1 library. Choose a "
                   "different folder for the %2 export.")
                    .arg(acmx2::backend_name(*existingBackend),
                         acmx2::backend_name(backend)));
            return;
        }
    }
    if (replacingLibrary &&
        QMessageBox::question(this, tr("Replace Existing Library"),
                              tr("This folder already contains library.json. Replace "
                                 "the manifest and matching shader files?")) !=
            QMessageBox::Yes) {
        return;
    }

    QStringList sourcePaths;
    for (int row = 0; row < shaderList->count(); ++row) {
        sourcePaths.append(
            shaderList->item(row)->data(SOURCE_PATH_ROLE).toString());
    }

    exportInProgress = true;
    updateControls();
    exportButton->setText(tr("Exporting..."));

    auto *watcher = new QFutureWatcher<ExportResult>(this);
    connect(watcher, &QFutureWatcher<ExportResult>::finished, this,
            [this, watcher, directory, exportKey]() {
                const ExportResult result = watcher->result();
                watcher->deleteLater();
                exportInProgress = false;
                exportButton->setText(tr("Export Library..."));
                updateControls();

                if (!result.success) {
                    QMessageBox::critical(this, tr("Export Failed"), result.error);
                    return;
                }

                QSettings settings("LostSideDead");
                settings.setValue(exportKey, directory);
                emit libraryExported(directory);
                QMessageBox::information(
                    this, tr("Library Exported"),
                    tr("Exported %1 shaders and library.json to:\n%2")
                        .arg(result.exportedNames.size())
                        .arg(QDir::toNativeSeparators(directory)));
            });
    const acmx2::Backend exportBackend = backend;
    watcher->setFuture(QtConcurrent::run([directory, replacingLibrary,
                                          sourcePaths, exportBackend]() {
        return export_shader_library(directory, replacingLibrary, sourcePaths,
                                     exportBackend);
    }));
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
    exportButton->setEnabled(!exportInProgress && shaderList->count() > 0);
}
