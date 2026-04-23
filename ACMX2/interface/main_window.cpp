#include "main_window.hpp"
#include "audio-window.hpp"
#include "settings.hpp"
#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QIcon>
#include <QInputDialog>
#include <QLayout>
#include <QMessageBox>
#include <QProcess>
#include <QTextStream>
#include <algorithm>
#include <random>
#ifdef __linux__
#include <sys/types.h>
#include <unistd.h>
#endif

void MainWindow::initControls() {
    lastFoundIndex = -1;
    lastSearchText = QString();
    process = new QProcess(this);
    connect(process, &QProcess::readyReadStandardOutput, this, [this]() {
        QString output = process->readAllStandardOutput();
        output.replace("\n", "<br>");
        this->Write(output);
    });

    connect(process, &QProcess::readyReadStandardError, this, [this]() {
        stderrBuffer += process->readAllStandardError();
        int idx;
        while ((idx = stderrBuffer.indexOf('\n')) != -1) {
            QString line = stderrBuffer.left(idx);
            stderrBuffer.remove(0, idx + 1);
            if (!line.contains("GStreamer")) {
                this->Write("<b style='color:red;'>Error:</b> " + line + "<br>");
            }
        }
        if (stderrBuffer.size() > 4096) {
            if (!stderrBuffer.contains("GStreamer")) {
                this->Write("<b style='color:red;'>Error:</b> " + stderrBuffer + "<br>");
            }
            stderrBuffer.clear();
        }
    });

    connect(process,
            static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus) {
                if (!stderrBuffer.isEmpty() && !stderrBuffer.contains("GStreamer")) {
                    this->Write("<b style='color:red;'>Error:</b> " + stderrBuffer + "<br>");
                    stderrBuffer.clear();
                }
                QString text;
                QTextStream stream(&text);
                stream << "acmx2: Exited with Code: " << exitCode;
                Log(text + "<br>");
                play_stop->setEnabled(false);
            });

    setStyleSheet(" QMainWindow { background-color: rgb(0,0,0); }");
    camera_index = 0;
    camera_res = QSize(1280, 720);
    screen_res = QSize(0, 0);
    setGeometry(150, 150, 1280, 720);
    setWindowTitle("ACMX2 - Interface");
    QMenuBar *menuBarPtr = menuBar();

    menuBar()->setNativeMenuBar(false);
    fileMenu = menuBarPtr->addMenu(tr("File"));
    cameraMenu = menuBarPtr->addMenu(tr("Session"));
    playbackMenu = menuBarPtr->addMenu(tr("Playback"));
    runMenu = menuBarPtr->addMenu(tr("Run"));
    listMenu = menuBarPtr->addMenu(tr("List"));
    viewMenu = menuBarPtr->addMenu(tr("View"));
    helpMenu = menuBarPtr->addMenu(tr("Help"));
    stayOnTopAction = new QAction(tr("Stay on Top"), this);
    stayOnTopAction->setCheckable(true);
    stayOnTopAction->setChecked(false);
    connect(stayOnTopAction, &QAction::toggled, this, [this](bool checked) {
        if (checked) {
            setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);
        } else {
            setWindowFlags(windowFlags() & ~Qt::WindowStaysOnTopHint);
        }
        show();
        if (checked && QGuiApplication::platformName() == "wayland") {
            Log("Stay on Top may not work on Wayland. Launch with QT_QPA_PLATFORM=xcb for X11 support.");
        }
    });
    viewMenu->addAction(stayOnTopAction);
    fileMenu_prop = new QAction(tr("Properties"), this);
    fileMenu->addAction(fileMenu_prop);
    connect(fileMenu_prop, &QAction::triggered, this, &MainWindow::fileOpenProp);
    fileMenu->addSeparator();
    fileMenu_exit = new QAction(tr("Exit"), this);
    connect(fileMenu_exit, &QAction::triggered, this, &MainWindow::fileExit);
    fileMenu->addAction(fileMenu_exit);
    cameraSet = new QAction(tr("Session Properties"), this);
    connect(cameraSet, &QAction::triggered, this, &MainWindow::cameraSettings);
    cameraMenu->addAction(cameraSet);
    audioSet = new QAction(tr("Audio Settings"), this);
    connect(audioSet, &QAction::triggered, this, &MainWindow::menuAudioSettings);
    cameraMenu->addAction(audioSet);
    gpuFilterAction = new QAction(tr("GPU Filter Settings"), this);
    connect(gpuFilterAction, &QAction::triggered, this, &MainWindow::menuGPUFilterSettings);
    cameraMenu->addAction(gpuFilterAction);
    cameraMenu->addSeparator();
    styleSheetAction = new QAction(tr("Use Custom Style"), this);
    styleSheetAction->setCheckable(true);
    styleSheetAction->setChecked(false);
    connect(styleSheetAction, &QAction::toggled, this, &MainWindow::applyCustomStyleSheet);
    cameraMenu->addAction(styleSheetAction);
    runMenu_select = new QAction(tr("Run Selected"), this);
    runMenu_select->setShortcut(QKeySequence("F5"));
    connect(runMenu_select, &QAction::triggered, this, &MainWindow::runSelected);
    runMenu->addAction(runMenu_select);
    runMenu->addSeparator();
    runMenu_all = new QAction(tr("Run All"), this);
    runMenu_all->setShortcut(QKeySequence("Ctrl+E"));
    connect(runMenu_all, &QAction::triggered, this, &MainWindow::runAll);
    runMenu->addAction(runMenu_all);
    play_repeat = new QAction(tr("Repeat"), this);
    play_repeat->setCheckable(true);
    play_repeat->setChecked(false);
    playbackMenu->addAction(play_repeat);
    play_stop = new QAction(tr("Stop"), this);
    play_stop->setEnabled(false);
    connect(play_stop, &QAction::triggered, this, [=]() {
        process->terminate();
    });
    playbackMenu->addAction(play_stop);
    playbackMenu->addSeparator();
    shaderPassAction = new QAction(tr("Multi-Pass Shader Settings..."), this);
    connect(shaderPassAction, &QAction::triggered, this, &MainWindow::menuShaderPassSettings);
    playbackMenu->addAction(shaderPassAction);
    playbackMenu->addSeparator();
    playlistAction = new QAction(tr("Shader Playlist Settings..."), this);
    connect(playlistAction, &QAction::triggered, this, &MainWindow::menuPlaylistSettings);
    playbackMenu->addAction(playlistAction);
    playbackMenu->addSeparator();
    buildCacheAction = new QAction(tr("Rebuild Shader Cache"), this);
    connect(buildCacheAction, &QAction::triggered, this, &MainWindow::menuBuildShaderCache);
    playbackMenu->addAction(buildCacheAction);

    removeBrokenAction = new QAction(tr("Remove Broken"), this);
    connect(removeBrokenAction, &QAction::triggered, this, &MainWindow::menuRemoveBroken);
    playbackMenu->addAction(removeBrokenAction);

    runFromCacheAction = new QAction(tr("Run from Cache"), this);
    runFromCacheAction->setCheckable(true);
    runFromCacheAction->setChecked(true);
    connect(runFromCacheAction, &QAction::toggled, this, [this](bool checked) {
        use_shader_cache = checked;
        if (checked) {
            Log("Shader cache enabled - will use cached shaders if available");
        } else {
            Log("Shader cache disabled - shaders will be recompiled each run");
        }
    });
    playbackMenu->addAction(runFromCacheAction);

    playbackMenu->addSeparator();
    midiSettingsAction = new QAction(tr("MIDI Settings..."), this);
    connect(midiSettingsAction, &QAction::triggered, this, &MainWindow::menuMidiSettings);
    playbackMenu->addAction(midiSettingsAction);

    // recompileShadersAction = new QAction(tr("Recompile All Shaders"), this);
    // connect(recompileShadersAction, &QAction::triggered, this, &MainWindow::menuRecompileShaders);
    // playbackMenu->addAction(recompileShadersAction);

    listMenu_new = new QAction(tr("New Shader Library"), this);
    connect(listMenu_new, &QAction::triggered, this, &MainWindow::newList);
    listMenu->addAction(listMenu_new);
    listMenu_shader = new QAction(tr("New Shader GLSL File"), this);
    connect(listMenu_shader, &QAction::triggered, this, &MainWindow::newShader);
    listMenu->addAction(listMenu_shader);
    listMenu->addSeparator();
    listMenu_remove = new QAction(tr("Remove Shader"), this);
    connect(listMenu_remove, &QAction::triggered, this, &MainWindow::menuRemove);
    listMenu->addAction(listMenu_remove);
    listMenu->addSeparator();
    listMenu_up = new QAction(tr("Shift Shader Up"), this);
    connect(listMenu_up, &QAction::triggered, this, &MainWindow::menuUp);
    listMenu->addAction(listMenu_up);
    listMenu_down = new QAction(tr("Shift Shader Down"), this);
    connect(listMenu_down, &QAction::triggered, this, &MainWindow::menuDown);
    listMenu->addAction(listMenu_down);
    listMenu_shuffle = new QAction(tr("Shuffle Shaders"), this);
    connect(listMenu_shuffle, &QAction::triggered, this, &MainWindow::menuShuffle);
    listMenu->addAction(listMenu_shuffle);

    listMenu_sort = new QAction(tr("Sort Shaders"), this);
    connect(listMenu_sort, &QAction::triggered, this, &MainWindow::menuSort);
    listMenu->addAction(listMenu_sort);
    listMenu->addSeparator();
    listMenu_search = new QAction(tr("Search Shaders"), this);
    listMenu_search->setShortcut(QKeySequence("Ctrl+F"));
    connect(listMenu_search, &QAction::triggered, this, &MainWindow::menuSearch);
    listMenu->addAction(listMenu_search);
    listMenu_findNext = new QAction(tr("Find Next"), this);
    listMenu_findNext->setShortcut(QKeySequence("F3"));
    connect(listMenu_findNext, &QAction::triggered, this, &MainWindow::menuFindNext);
    listMenu->addAction(listMenu_findNext);
    helpMenu_about = new QAction("About", this);

    connect(helpMenu_about, &QAction::triggered, this, [=]() {
        QMessageBox box(this);
        box.setWindowTitle("About ACMX2");
        box.setWindowIcon(QIcon(":/win-icon.png"));
        QString info;
        QTextStream stream(&info);
        stream << "ACMX2 " << VERSION_INFO << "\n(C) 2026 " << VERSION_AUTHOR << " Software\nhttps://lostsidedead.biz\nThis software is dedicated to all that have experienced mental illness.\n";
        box.setText(info);
        QPixmap bigIcon(":/win-icon.png");
        if (!bigIcon.isNull()) {
            QPixmap resizedIcon = bigIcon.scaled(64, 64, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            box.setIconPixmap(resizedIcon);
        }
        Log(info);
        box.exec();
    });
    helpMenu->addAction(helpMenu_about);
    model = new ReadOnlyStringListModel(this);
    model->setStringList(items);
    list_view = new QListView(this);
    list_view->setStyleSheet("QListView { background-color: black; color: white; font-size: 24px; font-family: 'Courier New', Courier, monospace; }");
    list_view->setModel(model);
    bottomTextBox = new QTextEdit(this);
    bottomTextBox->setHtml("<b style='color:red;'>ACMX2</b> - Interface: Loaded.");
    bottomTextBox->setStyleSheet("QTextEdit { background-color: black; color: lime; font-size: 24px; font-family: 'Courier New', Courier, monospace;; }");
    bottomTextBox->setReadOnly(true);
    connect(list_view, &QListView::doubleClicked,
            this, &MainWindow::listClicked);
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);
    layout->addWidget(list_view, 3);
    layout->addWidget(bottomTextBox, 1);
    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);
    QSettings appSettings("LostSideDead");
    QString path = appSettings.value("shaders", "").toString();
    path = path.trimmed();
    while (path.endsWith("/") || path.endsWith("\\")) {
        path.chop(1);
    }
#ifdef _WIN32
    executable_path = appSettings.value("exePath", "acmx2.exe").toString();
#else
    executable_path = appSettings.value("exePath", "acmx2").toString();
#endif
    prefix_path = appSettings.value("prefix_path", ".").toString();
    detectCudaSupport();
    bool useCustomStyle = appSettings.value("useCustomStyle", false).toBool();
    styleSheetAction->setChecked(useCustomStyle);
    midi_enabled = appSettings.value("midiEnabled", false).toBool();
    midi_config_file = appSettings.value("midiConfigFile", "").toString();
    midi_device = appSettings.value("midiDevice", -1).toInt();
    if (!path.isEmpty()) {
        QFileInfo pathInfo(path);
        QFileInfo indexInfo(path + "/index.txt");
        if (pathInfo.exists() && pathInfo.isDir() && indexInfo.exists()) {
            shader_path = path;
            loadShaders(path);
            Log("Successfully loaded saved shader path");
        } else {
            QString errorMsg = "Warning: Saved shader path is invalid: " + path + " - ";
            if (!pathInfo.exists()) {
                errorMsg += "directory does not exist";
            } else if (!pathInfo.isDir()) {
                errorMsg += "path is not a directory";
            } else if (!indexInfo.exists()) {
                errorMsg += "index.txt not found in directory";
            }
            Log(errorMsg);
        }
    }
    customStyleSheet = "QMainWindow, QDialog { background-color: black; border: 3px solid red; }"
                       "* { color: red; font-weight: bold; } "
                       "QPushButton { border: 1px solid red; background-color: #110000; padding: 5px; }"
                       "QPushButton:hover { background-color: red; color: black; }";

    applyCustomStyleSheet(useCustomStyle);
}

void MainWindow::applyCustomStyleSheet(bool enable) {
    QSettings appSettings("LostSideDead");
    appSettings.setValue("useCustomStyle", enable);
    if (enable) {
        setStyleSheet(customStyleSheet);
    } else {
        setStyleSheet("");
    }
}

void MainWindow::newList() {
    LibraryWindow library(this);

    if (library.exec() == QDialog::Accepted) {
        shader_path = library.getShaderPath();
        loadShaders(shader_path);
        QSettings appSettings("LostSideDead");
        appSettings.setValue("shaders", shader_path);
        appSettings.sync();
    }
}

void MainWindow::menuSearch() {
    bool ok;
    QString searchText = QInputDialog::getText(this,
                                               tr("Search Shaders"),
                                               tr("Enter shader name to search:"),
                                               QLineEdit::Normal,
                                               lastSearchText,
                                               &ok);

    if (!ok || searchText.isEmpty()) {
        return;
    }

    lastSearchText = searchText;
    lastFoundIndex = -1;
    QStringListModel *model = qobject_cast<QStringListModel *>(list_view->model());
    if (!model) {
        QMessageBox::warning(this, "Error", "The model is not a QStringListModel.");
        return;
    }
    QStringList shaderList = model->stringList();
    int foundIndex = -1;

    for (int i = 0; i < shaderList.size(); ++i) {
        if (shaderList[i].compare(searchText, Qt::CaseInsensitive) == 0) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex == -1) {
        for (int i = 0; i < shaderList.size(); ++i) {
            if (shaderList[i].contains(searchText, Qt::CaseInsensitive)) {
                foundIndex = i;
                break;
            }
        }
    }

    if (foundIndex != -1) {
        lastFoundIndex = foundIndex;
        QModelIndex matchIndex = model->index(foundIndex, 0);
        list_view->setCurrentIndex(matchIndex);
        list_view->selectionModel()->select(matchIndex, QItemSelectionModel::ClearAndSelect);
        list_view->scrollTo(matchIndex, QAbstractItemView::PositionAtCenter);

        Log("Found shader: " + shaderList[foundIndex] + " at index " + QString::number(foundIndex));
    } else {
        QMessageBox::information(this,
                                 tr("Not Found"),
                                 tr("Shader \"") + searchText + tr("\" not found in the list."));
        Log("Shader not found: " + searchText);
    }
}

void MainWindow::menuFindNext() {
    if (lastSearchText.isEmpty()) {
        QMessageBox::information(this,
                                 tr("No Search"),
                                 tr("Please perform a search first (Ctrl+F)."));
        return;
    }

    QStringListModel *model = qobject_cast<QStringListModel *>(list_view->model());
    if (!model) {
        QMessageBox::warning(this, "Error", "The model is not a QStringListModel.");
        return;
    }

    QStringList shaderList = model->stringList();
    if (shaderList.isEmpty()) {
        return;
    }

    int foundIndex = -1;
    int startIndex = (lastFoundIndex + 1) % shaderList.size();

    for (int i = startIndex; i < shaderList.size(); ++i) {
        if (shaderList[i].contains(lastSearchText, Qt::CaseInsensitive)) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex == -1 && startIndex > 0) {
        for (int i = 0; i < startIndex; ++i) {
            if (shaderList[i].contains(lastSearchText, Qt::CaseInsensitive)) {
                foundIndex = i;
                break;
            }
        }
    }

    if (foundIndex != -1) {
        lastFoundIndex = foundIndex;
        QModelIndex matchIndex = model->index(foundIndex, 0);
        list_view->setCurrentIndex(matchIndex);
        list_view->selectionModel()->select(matchIndex, QItemSelectionModel::ClearAndSelect);
        list_view->scrollTo(matchIndex, QAbstractItemView::PositionAtCenter);

        Log("Found next: " + shaderList[foundIndex] + " at index " + QString::number(foundIndex));
    } else {
        QMessageBox::information(this,
                                 tr("No More Results"),
                                 tr("No more matches for \"") + lastSearchText + tr("\"."));
        Log("No more matches for: " + lastSearchText);
    }
}

void MainWindow::newShader() {
    ShaderDialog new_shader(this);
    new_shader.setShaderPath(shader_path);
    if (new_shader.exec() == QDialog::Accepted) {
        QSettings appSettings("LostSideDead");
        appSettings.setValue("shaders", shader_path);
        appSettings.sync();
        loadShaders(shader_path, true);
    }
}

void MainWindow::menuRemove() {
    QItemSelectionModel *selModel = list_view->selectionModel();
    if (!selModel) {
        return;
    }
    QModelIndex currentIndex = selModel->currentIndex();
    if (!currentIndex.isValid()) {
        return;
    }
    model->removeRow(currentIndex.row());
    updateIndex();
    loadShaders(shader_path, true);
}

void MainWindow::updateIndex() {
    QFile file(shader_path + "/index.txt");
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        QStringListModel *stringModel = qobject_cast<QStringListModel *>(list_view->model());
        if (!stringModel) {
            return;
        }

        QStringList writtenItems;
        int rowCount = stringModel->rowCount();

        for (int row = 0; row < rowCount; ++row) {
            QModelIndex index = stringModel->index(row, 0);
            QVariant data = stringModel->data(index, Qt::DisplayRole);
            QString shaderName = data.toString().trimmed();

            if (shaderName.isEmpty() || writtenItems.contains(shaderName, Qt::CaseInsensitive)) {
                continue;
            }

            QString fullPath = shader_path + "/" + shaderName;
            QFileInfo fileInfo(fullPath);
            if (fileInfo.exists() && fileInfo.isFile()) {
                out << shaderName << "\n";
                writtenItems.append(shaderName);
            } else {
                Log("Warning: File no longer exists, removing from list: " + shaderName);
            }
        }
        file.close();

        indexTimestamp = QFileInfo(shader_path + "/index.txt").lastModified();

        if (writtenItems.size() != rowCount) {
            items = writtenItems;
            stringModel->setStringList(items);
            Log("Updated shader list, removed " + QString::number(rowCount - writtenItems.size()) + " non-existent files");
        }
    }
}

void MainWindow::menuUp() {
    QStringListModel *model = qobject_cast<QStringListModel *>(list_view->model());
    if (!model) {
        QMessageBox::warning(list_view, "Error", "The model is not a QStringListModel.");
        return;
    }
    QItemSelectionModel *selModel = list_view->selectionModel();
    if (!selModel) {
        return;
    }
    QModelIndex currentIndex = selModel->currentIndex();
    if (!currentIndex.isValid()) {
        return;
    }
    int currentRow = currentIndex.row();
    if (currentRow == 0) {
        return;
    }
    QStringList shaderList = model->stringList();
    shaderList.swapItemsAt(currentRow, currentRow - 1);
    model->setStringList(shaderList);
    QModelIndex newIndex = model->index(currentRow - 1);
    list_view->selectionModel()->setCurrentIndex(newIndex, QItemSelectionModel::Select);
    updateIndex();
}

void MainWindow::menuDown() {
    QStringListModel *model = qobject_cast<QStringListModel *>(list_view->model());
    if (!model) {
        return;
    }
    QItemSelectionModel *selModel = list_view->selectionModel();
    if (!selModel) {
        return;
    }
    QModelIndex currentIndex = selModel->currentIndex();
    if (!currentIndex.isValid()) {
        return;
    }
    int currentRow = currentIndex.row();
    int rowCount = model->rowCount();

    if (currentRow >= rowCount - 1) {
        return;
    }

    QStringList shaderList = model->stringList();
    shaderList.swapItemsAt(currentRow, currentRow + 1);
    model->setStringList(shaderList);
    QModelIndex newIndex = model->index(currentRow + 1);
    list_view->selectionModel()->setCurrentIndex(newIndex, QItemSelectionModel::Select);
    updateIndex();
}

QString MainWindow::readFileContents(const QString &filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        Log("Failed to open file: " + filePath);
        return QString();
    }

    QTextStream in(&file);
    QString contents = in.readAll();
    file.close();
    return contents;
}

void MainWindow::listClicked(const QModelIndex &i) {
    if (!i.isValid())
        return;
    QString itemText = sanitizeShaderName(i.data(Qt::DisplayRole).toString());
    if (itemText.isEmpty()) {
        Log("Invalid shader name");
        return;
    }
    cleanupClosedEditors();
    TextEditor *editor = new TextEditor(this);
    QString filePath = shader_path + "/" + itemText;
    editor->setText(readFileContents(filePath));
    editor->setFileName(filePath);
    open_files.append(editor);
    editor->show();
}

void MainWindow::Log(const QString &message) {
    bottomTextBox->append(message);
    QTextCursor cursor = bottomTextBox->textCursor();
    cursor.movePosition(QTextCursor::End);
    bottomTextBox->setTextCursor(cursor);
}

void MainWindow::Write(const QString &message) {
    QTextCursor cursor = bottomTextBox->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertHtml(message);
    bottomTextBox->setTextCursor(cursor);

    constexpr int MAX_BLOCKS = 5000;
    QTextDocument *doc = bottomTextBox->document();
    int excess = doc->blockCount() - MAX_BLOCKS;
    if (excess > 0) {
        QTextCursor trim(doc);
        trim.movePosition(QTextCursor::Start);
        trim.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor, excess);
        trim.movePosition(QTextCursor::StartOfBlock, QTextCursor::KeepAnchor);
        trim.removeSelectedText();
        trim.deleteChar();
    }
}

void MainWindow::fileOpenProp() {
    PropWindow propWindow(this);
    if (propWindow.exec() == QDialog::Accepted) {
        QString exePath = propWindow.exePathLineEdit->text();
        QString shaderDir = propWindow.shaderDirLineEdit->text();
        QString prefix = propWindow.screenshotDirLineEdit->text();
        if (exePath.length() == 0) {
            QMessageBox::information(this, "No Path", "Requires Executable path");
            return;
        }
        if (shaderDir.length() == 0) {
            QMessageBox::information(this, "Shader Path", "Requires Shader Path");
            return;
        }

        QFileInfo shaderDirInfo(shaderDir);
        QFileInfo indexFileInfo(shaderDir + "/index.txt");
        if (!shaderDirInfo.exists()) {
            QMessageBox::warning(this, "Invalid Shader Path", "Shader directory does not exist:\n" + shaderDir);
            return;
        }

        if (!shaderDirInfo.isDir()) {
            QMessageBox::warning(this, "Invalid Shader Path", "Shader path is not a directory:\n" + shaderDir);
            return;
        }

        if (!indexFileInfo.exists()) {
            QMessageBox::warning(this, "Missing index.txt", "Shader directory does not contain index.txt:\n" + shaderDir + "/index.txt");
            return;
        }

        QSettings appSettings("LostSideDead");
        appSettings.setValue("exePath", exePath);
        appSettings.setValue("prefix_path", prefix);
        appSettings.setValue("shaders", shaderDir);
        appSettings.sync();

        executable_path = exePath;
        prefix_path = prefix;
        shader_path = shaderDir;

        Log("Executable Path: " + exePath);
        Log("Prefix Path: " + prefix);
        Log("Shader Directory: " + shaderDir);

        if (loadShaders(shaderDir)) {
            Log("Successfully loaded shaders from new directory<br>");
        } else {
            Log("Warning: Could not load shaders from new directory<br>");
        }
    } else {
        Log("Canceled");
    }
}

bool MainWindow::loadShaders(const QString &path, bool force) {
    QFileInfo info(path + "/index.txt");
    if (!info.exists() || !info.isFile()) {
        QMessageBox::warning(this, "Could not open index file", "Failed to open file: " + path + "/index.txt");
        return false;
    }

    QDateTime modified = info.lastModified();
    if (force == false && path == shader_path && !indexTimestamp.isNull() && modified <= indexTimestamp) {
        return true;
    }
    QFile file(path + "/index.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Could not open index file", "Failed to open file: " + file.errorString());
        return false;
    }
    shader_path = path;
    indexTimestamp = modified;
    int previousRow = -1;
    QModelIndex currentIndex = list_view->currentIndex();
    if (currentIndex.isValid()) {
        previousRow = currentIndex.row();
    }
    QString previouslySelected;
    if (currentIndex.isValid()) {
        previouslySelected = model->data(currentIndex, Qt::DisplayRole).toString();
    }
    items.clear();
    QStringList uniqueItems;
    QTextStream in(&file);

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();

        if (line.isEmpty()) {
            continue;
        }
        QString fullPath = path + "/" + line;
        QFileInfo fileInfo(fullPath);
        if (!fileInfo.exists() || !fileInfo.isFile()) {
            Log("Skipping non-existent file: " + line);
            continue;
        }
        if (!uniqueItems.contains(line, Qt::CaseInsensitive)) {
            uniqueItems.append(line);
            // Log("Added shader: " + line);
        } else {
            Log("Skipping duplicate shader: " + line);
        }
    }
    file.close();
    items = uniqueItems;
    model->setStringList(items);

    Log("Loaded " + QString::number(items.size()) + " unique shader files");
    menuSort();

    if (!items.isEmpty()) {
        int restoredRow = previousRow;
        if (restoredRow < 0 || restoredRow >= items.size()) {
            if (!previouslySelected.isEmpty() && items.contains(previouslySelected)) {
                restoredRow = items.indexOf(previouslySelected);
            } else {
                restoredRow = 0;
            }
        }
        QModelIndex restoredIndex = model->index(restoredRow, 0);
        list_view->setCurrentIndex(restoredIndex);
        list_view->selectionModel()->select(restoredIndex, QItemSelectionModel::ClearAndSelect);
        list_view->scrollTo(restoredIndex, QAbstractItemView::PositionAtCenter);
    }

    return true;
}

void MainWindow::fileExit() {
    QApplication::quit();
}

void MainWindow::menuAudioSettings() {
    if (!audio_available) {
        QMessageBox::information(this, tr("Audio Settings"),
            tr("Audio support is unavailable: acmx2 was built without audio support."));
        return;
    }
    AudioSettings audio_set(this);
    if (audio_set.exec() == QDialog::Accepted) {
        audio_enabled = audio_set.isAudioReactivityEnabled();
        audio_channels = audio_set.getNumberOfChannels();
        audio_sense = audio_set.getSensitivity();
        audio_passthrough = audio_set.isAudioPassThroughEnabled();
        record_audio = audio_set.isRecordAudioEnabled();
        record_volume = audio_set.getRecordVolume();
        audio_input = audio_set.getInputDeviceIndex();
        audio_output = audio_set.getOutputDeviceIndex();
        if (audio_set.isAudioFileEnabled()) {
            audio_file = audio_set.getAudioFilePath();
        } else {
            audio_file = "";
        }
        audio_trunc = audio_set.isAudioTruncEnabled();
        Log("Audio Settings Saved");
    }
}

void MainWindow::menuGPUFilterSettings() {
    if (!cuda_available) {
        QMessageBox::information(this, tr("GPU Filter Settings"),
            tr("GPU filters are unavailable: acmx2 was built without CUDA support."));
        return;
    }
    GPUFilterDialog gpuDialog(executable_path, this);
    if (gpuDialog.exec() == QDialog::Accepted) {
        gpu_filter_enabled = gpuDialog.isGPUFilterEnabled();
        gpu_filter_indices = gpuDialog.getFilterArgument();
        gpu_buffer_size = gpuDialog.getBufferSize();
        if (gpu_filter_enabled) {
            Log("GPU Filter Settings Saved: Filters=" + gpu_filter_indices + ", Buffer=" + QString::number(gpu_buffer_size));
        } else {
            Log("GPU Filtering Disabled");
        }
    }
}

void MainWindow::menuMidiSettings() {
    if (!midi_available) {
        QMessageBox::information(this, tr("MIDI Settings"),
            tr("MIDI support is unavailable: acmx2 was built without MIDI support."));
        return;
    }
    MidiSettings midiDialog(executable_path, this);
    if (midiDialog.exec() == QDialog::Accepted) {
        midi_enabled = midiDialog.isMidiEnabled();
        midi_config_file = midiDialog.getMidiConfigFile();
        midi_device = midiDialog.getMidiDeviceIndex();
        QSettings appSettings("LostSideDead");
        appSettings.setValue("midiEnabled", midi_enabled);
        appSettings.setValue("midiConfigFile", midi_config_file);
        appSettings.setValue("midiDevice", midi_device);
        if (midi_enabled) {
            Log("MIDI Settings Saved: Config=" + midi_config_file + ", Device=" + QString::number(midi_device));
        } else {
            Log("MIDI Disabled");
        }
    }
}

void MainWindow::menuShaderPassSettings() {
    if (shader_path.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First",
                                 "Please load a shader library before configuring multi-pass shaders.");
        return;
    }

    loadShaders(shader_path, true);

    if (items.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First",
                                 "Please load a shader library before configuring multi-pass shaders.");
        return;
    }

    ShaderPassDialog passDialog(items, this);
    passDialog.setEnabled(shader_pass_enabled);
    if (!shader_pass_names.isEmpty()) {
        passDialog.setSelectedShaderNames(shader_pass_names);
    }

    if (passDialog.exec() == QDialog::Accepted) {
        shader_pass_enabled = passDialog.isShaderPassEnabled();
        shader_pass_names = passDialog.getSelectedShaderNames();
        if (shader_pass_enabled) {
            Log("Multi-Pass Shader Settings Saved: " + QString::number(shader_pass_names.size()) + " passes");
        } else {
            Log("Multi-Pass Shader Disabled");
        }
    }
}

void MainWindow::menuPlaylistSettings() {
    if (shader_path.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First",
                                 "Please load a shader library before configuring playlist.");
        return;
    }

    loadShaders(shader_path, true);

    if (items.isEmpty()) {
        QMessageBox::information(this, "Load Shaders First",
                                 "Please load a shader library before configuring playlist.");
        return;
    }

    PlaylistDialog playlistDialog(items, this);
    playlistDialog.setEnabled(playlist_enabled);
    if (!playlist_tree_data.isEmpty()) {
        playlistDialog.setPlaylistTree(playlist_tree_data);
    } else if (!playlist_names.isEmpty()) {
        playlistDialog.setSelectedShaderNames(playlist_names);
    }
    if (!playlist_file_path.isEmpty()) {
        playlistDialog.setPlaylistFile(playlist_file_path);
    }

    if (playlistDialog.exec() == QDialog::Accepted) {
        playlist_enabled = playlistDialog.isPlaylistEnabled();
        playlist_names = playlistDialog.getSelectedShaderNames();
        playlist_tree_data = playlistDialog.getPlaylistTree();
        playlist_file_path = playlistDialog.getPlaylistFile();
        if (playlist_enabled) {
            Log("Playlist Settings Saved: " + QString::number(playlist_names.size()) + " shaders");
            if (!playlist_file_path.isEmpty()) {
                Log("Playlist file: " + playlist_file_path);
            }
        } else {
            Log("Playlist Disabled");
        }
    }
}

void MainWindow::cameraSettings() {
    SettingsWindow settingsWindow(executable_path, this);
    settingsWindow.setCudaAvailable(cuda_available);
    if (settingsWindow.exec() == QDialog::Accepted) {
        full_screen_value = settingsWindow.isFullscreen();
        if (settingsWindow.isUsingInputVideoFile()) {
            QString videoFile = settingsWindow.getInputVideoFile();
            QSize screenResolution = settingsWindow.getSelectedScreenResolution();
            screen_res = screenResolution;
            video_file = videoFile;
            graphics_file = "";
            cache_enabled = settingsWindow.isTextureCacheEnabled();
            cache_delay = settingsWindow.getCacheDelay();
            copy_audio = settingsWindow.isCopyAudioEnabled();
        } else if (settingsWindow.isUsingGraphicsFile()) {
            QString graphicsFile = settingsWindow.getGraphicsFile();
            QSize screenResolution = settingsWindow.getSelectedScreenResolution();
            screen_res = screenResolution;
            graphics_file = graphicsFile;
            video_file = "";
            output_fps = settingsWindow.getCameraFPS();
            cache_enabled = false;
            cache_delay = 1;
            copy_audio = false;
        } else {
            int cameraIndex = settingsWindow.getSelectedCameraIndex();
            QSize cameraResolution = settingsWindow.getSelectedCameraResolution();
            QSize screenResolution = settingsWindow.getSelectedScreenResolution();
            screen_res = screenResolution;
            camera_index = cameraIndex;
            video_file = "";
            graphics_file = "";
            camera_res = cameraResolution;
            output_fps = settingsWindow.getCameraFPS();
            cache_enabled = false;
            cache_delay = 1;
            use_yuv = settingsWindow.isUseYuvEnabled();
        }
        if (settingsWindow.isSavingToOutputVideoFile()) {
            output_file = settingsWindow.getOutputVideoFile();
            output_kbps = settingsWindow.getSaveFileKbps();
        } else {
            output_file = "";
            output_kbps = 23;
        }
    }
    enable_3d = settingsWindow.is3dEnabled();
    model_file = settingsWindow.getModelFile();
    cuda_device = settingsWindow.getSelectedCudaDevice();
    time_speed = settingsWindow.getTimeSpeed();
    duration_limit_enabled = settingsWindow.isDurationLimitEnabled();
    max_duration = settingsWindow.getDurationLimit();
    cross_fade_duration = settingsWindow.getCrossFadeDuration();
    encode_preset = settingsWindow.getEncodePreset();
    encode_tune = settingsWindow.getEncodeTune();
    encode_crf = settingsWindow.getEncodeCrf();
    encode_codec = settingsWindow.getEncodeCodec();
    encode_realtime = settingsWindow.isEncodeRealtime();
}

void MainWindow::runSelected() {
    if (process->state() == QProcess::Running) {
        QMessageBox::information(this, "Process Running", "A process is already running. Please stop it first.");
        return;
    }

#ifdef __linux__
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString uid = QString::number(getuid());
    QString user_run_path = "/run/user/" + uid;
    env.insert("SDL_VIDEODRIVER", "x11");
    if (QDir(user_run_path).exists()) {
        env.insert("XDG_RUNTIME_DIR", user_run_path);
        env.insert("PULSE_SERVER", "unix:" + user_run_path + "/pulse/native");
    }
    env.insert("CUDA_VISIBLE_DEVICES", "0");
    env.insert("vblank_mode", "0");
    process->setProcessEnvironment(env);
#endif

    if (shader_path.length() == 0) {
        QMessageBox::information(this, "Select Shaders", "Select Shader Path");
        return;
    }
    QItemSelectionModel *selectionModel = list_view->selectionModel();
    if (!selectionModel->hasSelection()) {
        Log("<b>No item selected.</b>");
        return;
    }
    QModelIndex selectedIndex = selectionModel->currentIndex();
    QString data = selectedIndex.data(Qt::DisplayRole).toString();
    QStringList arguments;
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/acmx2";
#endif
    if (!QFileInfo::exists(dirPath + "/data/win-icon.png"))
        dirPath = "/usr/local/share/acmx2";
    QString shader_file = shader_path + "/" + data;
    arguments << "--path" << dirPath << "--fragment" << shader_file;
    QString res;
    QTextStream stream(&res);
    stream << camera_res.width() << "x" << camera_res.height();

    QString scr_res;
    QTextStream stream_r(&scr_res);
    stream_r << screen_res.width() << "x" << screen_res.height();

    if (full_screen_value)
        arguments << "--fullscreen";

    if (!graphics_file.isEmpty()) {
        arguments << "--graphic" << graphics_file;
        if (screen_res.width() != 0)
            arguments << "--resolution" << scr_res;
        arguments << "--fps" << QString::number(output_fps);
    } else if (video_file.isEmpty()) {
        arguments << "--camera-res" << res;
        if (screen_res.width() != 0)
            arguments << "--resolution" << scr_res;
        arguments << "--device" << QString::number(camera_index);
        arguments << "--fps" << QString::number(output_fps);
        if (use_yuv)
            arguments << "--use-yuv";
    } else {
        arguments << "--input" << video_file;
        if (screen_res.width() != 0)
            arguments << "--resolution" << scr_res;
        if (play_repeat->isChecked())
            arguments << "--repeat";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
        if (copy_audio)
            arguments << "--copy-audio";
    }
    arguments << "--prefix" << prefix_path;

    if (!output_file.isEmpty()) {
        arguments << "--output" << output_file;
        arguments << "--encode-crf" << QString::number(encode_crf);
        if (!encode_preset.isEmpty())
            arguments << "--encode-preset" << encode_preset;
        if (!encode_tune.isEmpty())
            arguments << "--encode-tune" << encode_tune;
        if (!encode_codec.isEmpty() && encode_codec != "auto")
            arguments << "--encode-codec" << encode_codec;
        if (encode_realtime)
            arguments << "--encode-realtime";
    }
    if (audio_available && audio_enabled) {
        arguments << "--enable-audio";
        arguments << "--channels" << QString::number(audio_channels);
        arguments << "--sense" << QString::number(audio_sense);
        if (audio_passthrough)
            arguments << "--pass-through";

        if (audio_input == -1)
            arguments << "--audio-input" << "default";
        else
            arguments << "--audio-input" << QString::number(audio_input);

        if (audio_output == -1)
            arguments << "--audio-output" << "default";
        else
            arguments << "--audio-output" << QString::number(audio_output);

        if (record_audio) {
            QString wavPath;
            if (!output_file.isEmpty()) {
                QFileInfo fi(output_file);
                wavPath = fi.absolutePath() + "/" + fi.completeBaseName() + ".wav";
            } else {
                wavPath = prefix_path + "/recorded_audio.wav";
            }
            arguments << "--record-audio" << wavPath;
            arguments << "--record-gain" << QString::number(record_volume, 'f', 2);
        }
    }

    if (audio_available && !audio_file.isEmpty()) {
        arguments << "--audio-file" << audio_file;
        if (audio_trunc) {
            arguments << "--audio-trunc";
        }
    }

    if (enable_3d) {
        arguments << "--enable-3d";
        arguments << "--model" << model_file;
    }

    if (cuda_available && gpu_filter_enabled && !gpu_filter_indices.isEmpty()) {
        arguments << "--gpu-filter" << gpu_filter_indices;
        arguments << "--gpu-buffer" << QString::number(gpu_buffer_size);
    }

    if (cuda_available) {
        arguments << "--cuda-device" << QString::number(cuda_device);
    }

    if (time_speed != 1.0f) {
        arguments << "--time-speed" << QString::number(static_cast<double>(time_speed), 'f', 2);
    }

    if (!use_shader_cache) {
        arguments << "--no-cache";
    }

    if (midi_available && midi_enabled && !midi_config_file.isEmpty()) {
        arguments << "--midi-map" << midi_config_file;
        if (midi_device >= 0)
            arguments << "--midi-device" << QString::number(midi_device);
    }

    if (duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    Log("shell: acmx2 " + concatList(arguments) + "<br>");
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Failed to start the program.</b>");
        QMessageBox::critical(this, "Error", "Failed to start the program.");
    } else {
        play_stop->setEnabled(true);
    }
}

void MainWindow::runAll() {
    if (process->state() == QProcess::Running) {
        QMessageBox::information(this, "Process Running", "A process is already running. Please stop it first.");
        return;
    }

#ifdef __linux__
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString uid = QString::number(getuid());
    QString user_run_path = "/run/user/" + uid;
    env.insert("SDL_VIDEODRIVER", "x11");
    if (QDir(user_run_path).exists()) {
        env.insert("XDG_RUNTIME_DIR", user_run_path);
        env.insert("PULSE_SERVER", "unix:" + user_run_path + "/pulse/native");
    }
    env.insert("CUDA_VISIBLE_DEVICES", "0");
    env.insert("vblank_mode", "0");
    process->setProcessEnvironment(env);
#endif
    if (shader_path.length() == 0) {
        QMessageBox::information(this, "Select Shaders", "Select Shader Path");
        return;
    }
    int index = 0;
    QItemSelectionModel *selectionModel = list_view->selectionModel();
    if (!selectionModel->hasSelection()) {
        index = 0;
        Log("No selection, defaulting to index 0");
    } else {
        QModelIndex selectedIndex = selectionModel->currentIndex();
        index = selectedIndex.row();
        QString selectedData = selectedIndex.data(Qt::DisplayRole).toString();
        Log("Selected shader: " + selectedData + " at index: " + QString::number(index));
    }
    QStringList arguments;
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/acmx2";
#endif
    if (!QFileInfo::exists(dirPath + "/data/win-icon.png"))
        dirPath = "/usr/local/share/acmx2";

    QString shader_file = shader_path;
    arguments << "--path" << dirPath << "--shaders" << shader_file;
    QString res;
    QTextStream stream(&res);
    stream << camera_res.width() << "x" << camera_res.height();
    QString scr_res;
    QTextStream stream_r(&scr_res);
    stream_r << screen_res.width() << "x" << screen_res.height();

    if (full_screen_value)
        arguments << "--fullscreen";

    if (!graphics_file.isEmpty()) {
        arguments << "--graphic" << graphics_file;
        if (screen_res.width() != 0)
            arguments << "--resolution" << scr_res;
        arguments << "--fps" << QString::number(output_fps);
    } else if (video_file.isEmpty()) {
        arguments << "--camera-res" << res;
        if (screen_res.width() != 0)
            arguments << "--resolution" << scr_res;
        arguments << "--device" << QString::number(camera_index);
        arguments << "--fps" << QString::number(output_fps);
        if (use_yuv)
            arguments << "--use-yuv";
    } else {
        arguments << "--input" << video_file;
        if (screen_res.width() != 0)
            arguments << "--resolution" << scr_res;
        if (play_repeat->isChecked())
            arguments << "--repeat";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
        if (copy_audio)
            arguments << "--copy-audio";
    }
    arguments << "--prefix" << prefix_path;
    if (!output_file.isEmpty()) {
        arguments << "--output" << output_file;
        arguments << "--encode-crf" << QString::number(encode_crf);
        if (!encode_preset.isEmpty())
            arguments << "--encode-preset" << encode_preset;
        if (!encode_tune.isEmpty())
            arguments << "--encode-tune" << encode_tune;
        if (!encode_codec.isEmpty() && encode_codec != "auto")
            arguments << "--encode-codec" << encode_codec;
        if (encode_realtime)
            arguments << "--encode-realtime";
    }
    arguments << "--shader" << QString::number(index);

    if (audio_available && audio_enabled) {
        arguments << "--enable-audio";
        arguments << "--channels" << QString::number(audio_channels);
        arguments << "--sense" << QString::number(audio_sense);
        if (audio_passthrough)
            arguments << "--pass-through";

        if (audio_input == -1)
            arguments << "--audio-input" << "default";
        else
            arguments << "--audio-input" << QString::number(audio_input);

        if (audio_output == -1)
            arguments << "--audio-output" << "default";
        else
            arguments << "--audio-output" << QString::number(audio_output);

        if (record_audio) {
            QString wavPath;
            if (!output_file.isEmpty()) {
                QFileInfo fi(output_file);
                wavPath = fi.absolutePath() + "/" + fi.completeBaseName() + ".wav";
            } else {
                wavPath = prefix_path + "/recorded_audio.wav";
            }
            arguments << "--record-audio" << wavPath;
            arguments << "--record-gain" << QString::number(record_volume, 'f', 2);
        }
    }

    if (audio_available && !audio_file.isEmpty()) {
        arguments << "--audio-file" << audio_file;
        if (audio_trunc) {
            arguments << "--audio-trunc";
        }
    }

    if (enable_3d) {
        arguments << "--enable-3d";
        arguments << "--model" << model_file;
    }

    if (cuda_available && gpu_filter_enabled && !gpu_filter_indices.isEmpty()) {
        arguments << "--gpu-filter" << gpu_filter_indices;
        arguments << "--gpu-buffer" << QString::number(gpu_buffer_size);
    }

    if (shader_pass_enabled && !shader_pass_names.isEmpty()) {
        QString passIndices = getShaderPassIndicesFromNames();
        if (!passIndices.isEmpty()) {
            arguments << "--shader-pass" << passIndices;
        }
    }

    if (cuda_available) {
        arguments << "--cuda-device" << QString::number(cuda_device);
    }

    if (time_speed != 1.0f) {
        arguments << "--time-speed" << QString::number(static_cast<double>(time_speed), 'f', 2);
    }

    if (!use_shader_cache) {
        arguments << "--no-cache";
    }

    if (midi_available && midi_enabled && !midi_config_file.isEmpty()) {
        arguments << "--midi-map" << midi_config_file;
        if (midi_device >= 0)
            arguments << "--midi-device" << QString::number(midi_device);
    }

    if (playlist_enabled && !playlist_names.isEmpty()) {
        QString plFile = playlist_file_path;
        if (plFile.isEmpty()) {
            plFile = prefix_path + "/playlist.txt";
        }
        QFile f(plFile);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&f);
            if (!playlist_tree_data.isEmpty()) {
                for (const auto &[nodeName, shaders] : playlist_tree_data) {
                    out << "[" << nodeName << "]\n";
                    for (const QString &name : shaders)
                        out << name << "\n";
                }
            } else {
                for (const QString &name : playlist_names)
                    out << name << "\n";
            }
            f.close();
            playlist_file_path = plFile;
        }
        arguments << "--playlist" << plFile;
    }

    if (duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    Log("shell: acmx2 " + concatList(arguments) + "<br>");
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Failed to start the program.</b>");
        QMessageBox::critical(this, "Error", "Failed to start the program.");
    } else {
        play_stop->setEnabled(true);
    }
}

QString MainWindow::concatList(const QStringList lst) {
    QString text;
    QTextStream stream(&text);
    for (auto &i : lst) {
        stream << i << " ";
    }
    return text;
}

QString MainWindow::getShaderPassIndicesFromNames() {
    QStringList indices;
    loadShaders(shader_path, true);
    for (const QString &name : shader_pass_names) {
        int idx = items.indexOf(name);
        if (idx >= 0) {
            indices.append(QString::number(idx));
        }
    }
    return indices.join(",");
}

QString MainWindow::sanitizeShaderName(const QString &name) {
    QString sanitized = name.trimmed();
    if (sanitized.contains("..") || sanitized.contains("/") || sanitized.contains("\\")) {
        Log("Warning: Invalid shader name detected (path traversal attempt): " + name);
        return QString();
    }
    while (sanitized.startsWith('.')) {
        sanitized = sanitized.mid(1);
    }
    return sanitized;
}

void MainWindow::cleanupClosedEditors() {
    open_files.erase(
        std::remove_if(open_files.begin(), open_files.end(),
                       [](const QPointer<TextEditor> &ptr) { return ptr.isNull(); }),
        open_files.end());
}

void MainWindow::menuShuffle() {
    QStringListModel *model = qobject_cast<QStringListModel *>(list_view->model());
    if (!model) {
        QMessageBox::warning(this, "Error", "The model is not a QStringListModel.");
        return;
    }
    QStringList shaderList = model->stringList();
    if (shaderList.isEmpty()) {
        return;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shaderList.begin(), shaderList.end(), g);

    model->setStringList(shaderList);
    updateIndex();
    Log("Shaders shuffled");
}

void MainWindow::menuSort() {
    QStringListModel *model = qobject_cast<QStringListModel *>(list_view->model());
    if (!model) {
        QMessageBox::warning(this, "Error", "The model is not a QStringListModel.");
        return;
    }
    QStringList shaderList = model->stringList();
    if (shaderList.isEmpty()) {
        return;
    }
    shaderList.sort(Qt::CaseInsensitive);
    model->setStringList(shaderList);
    updateIndex();
    Log("Shaders sorted alphabetically");
}

void MainWindow::menuBuildShaderCache() {
    QString build_path = shader_path;
    if (build_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        build_path = appSettings.value("shaders", "").toString();
    }

    if (build_path.isEmpty()) {
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }

    if (process->state() == QProcess::Running) {
        QMessageBox::warning(this, "Error", "A process is already running. Please wait for it to finish.");
        return;
    }

    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    QString assets_path = dirPath + "/../Helpers";
#else
    QString assets_path = QFileInfo::exists(dirPath + "/data/win-icon.png") ? dirPath : QStringLiteral("/usr/local/share/acmx2");
#endif

    QStringList args;
    args << "--build" << build_path;
    args << "-p" << assets_path;

    if (enable_3d) {
        args << "--enable-3d";
    }

    Log("Building shader cache for: " + build_path);
    Log("Command: " + executable_path + " " + args.join(" "));

    play_stop->setEnabled(true);
    process->start(executable_path, args);

    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start shader cache build process");
        play_stop->setEnabled(false);
    }
}

void MainWindow::menuRunFromCache() {
}

void MainWindow::menuRemoveBroken() {
    QString scan_path = shader_path;
    if (scan_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        scan_path = appSettings.value("shaders", "").toString();
    }
    if (scan_path.isEmpty()) {
        QMessageBox::warning(this, "Error",
            "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }
    if (process->state() == QProcess::Running) {
        QMessageBox::warning(this, "Error",
            "A process is already running. Please wait for it to finish.");
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(this,
        tr("Remove Broken Shaders"),
        tr("This will compile every shader in:\n\n%1\n\n"
           "Any shader that fails to compile will be removed from index.txt "
           "(the original will be backed up as index.txt.bak).\n\nContinue?")
            .arg(scan_path),
        QMessageBox::Yes | QMessageBox::No);
    if (reply != QMessageBox::Yes) return;

    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    QString assets_path = dirPath + "/../Helpers";
#else
    QString assets_path = QFileInfo::exists(dirPath + "/data/win-icon.png")
                              ? dirPath
                              : QStringLiteral("/usr/local/share/acmx2");
#endif

    QStringList args;
    args << "--remove-broken" << scan_path;
    args << "-p" << assets_path;
    if (enable_3d) args << "--enable-3d";

    Log("Scanning for broken shaders in: " + scan_path);
    Log("Command: " + executable_path + " " + args.join(" "));

    // Use a dedicated QProcess so we can reload the list when it finishes
    // without interfering with the main playback process.
    QProcess *scan = new QProcess(this);
    scan->setProcessChannelMode(QProcess::SeparateChannels);
    connect(scan, &QProcess::readyReadStandardOutput, this, [this, scan]() {
        QString output = scan->readAllStandardOutput();
        output.replace("\n", "<br>");
        this->Write(output);
    });
    connect(scan, &QProcess::readyReadStandardError, this, [this, scan]() {
        QString output = scan->readAllStandardError();
        output.replace("\n", "<br>");
        this->Write("<b style='color:red;'>" + output + "</b>");
    });
    connect(scan,
            static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
            this,
            [this, scan, scan_path](int exitCode, QProcess::ExitStatus) {
                Log(QString("Remove-broken finished with exit code: %1<br>").arg(exitCode));
                if (exitCode == 0) {
                    // Reload the list view from the updated index.txt.
                    loadShaders(scan_path, true);
                    QMessageBox::information(this,
                        tr("Remove Broken"),
                        tr("Finished scanning shader library.\n\n"
                           "index.txt has been updated and the shader list reloaded.\n"
                           "A backup of the original is at:\n%1/index.txt.bak")
                            .arg(scan_path));
                } else {
                    QMessageBox::warning(this,
                        tr("Remove Broken"),
                        tr("Remove-broken failed with exit code %1. "
                           "index.txt was not changed.").arg(exitCode));
                }
                scan->deleteLater();
            });

    scan->start(executable_path, args);
    if (!scan->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start remove-broken process");
        scan->deleteLater();
    }
}

void MainWindow::menuRecompileShaders() {
    QString recompile_path = shader_path;

    if (recompile_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        recompile_path = appSettings.value("shaders", "").toString();
    }

    if (recompile_path.isEmpty()) {
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }

    QString cacheFile = recompile_path + "/.shader_cache";
    QFile cache(cacheFile);
    if (cache.exists()) {
        if (cache.remove()) {
            Log("Deleted shader cache: " + cacheFile);
        } else {
            Log("<b style='color:red;'>Warning:</b> Could not delete cache file: " + cacheFile);
        }
    } else {
        Log("No existing shader cache found");
    }

    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        "Rebuild Cache?",
        "Shader cache has been cleared. Would you like to rebuild the cache now?",
        QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        QString old_path = shader_path;
        shader_path = recompile_path;
        menuBuildShaderCache();
        shader_path = old_path;
    } else {
        Log("Shaders will be recompiled on next run");
    }
}

void MainWindow::detectCudaSupport() {
    detectFeatureSupport();
}

static bool probeFeature(const QString &exe, const QString &flag, const QString &token) {
    QProcess probe;
    probe.start(exe, QStringList() << flag);
    if (!probe.waitForFinished(5000)) {
        probe.kill();
        return false;
    }
    const QString out = QString::fromLocal8Bit(probe.readAllStandardOutput()).trimmed();
    return out.contains(token, Qt::CaseInsensitive);
}

void MainWindow::detectFeatureSupport() {
    cuda_available = probeFeature(executable_path, "--check-cuda", "CUDA: enabled");
    audio_available = probeFeature(executable_path, "--check-audio", "AUDIO: enabled");
    midi_available = probeFeature(executable_path, "--check-midi", "MIDI: enabled");

    Log(cuda_available ? "CUDA: enabled (acmx2 built with CUDA support)"
                       : "CUDA: disabled (acmx2 built without CUDA support)");
    Log(audio_available ? "AUDIO: enabled (acmx2 built with audio support)"
                        : "AUDIO: disabled (acmx2 built without audio support)");
    Log(midi_available ? "MIDI: enabled (acmx2 built with MIDI support)"
                       : "MIDI: disabled (acmx2 built without MIDI support)");

    if (gpuFilterAction) {
        gpuFilterAction->setEnabled(cuda_available);
        if (!cuda_available) {
            gpuFilterAction->setToolTip(tr("Disabled: acmx2 was built without CUDA support."));
        }
    }
    if (!cuda_available) {
        gpu_filter_enabled = false;
        gpu_filter_indices.clear();
        cuda_device = 0;
    }

    if (audioSet) {
        audioSet->setEnabled(audio_available);
        if (!audio_available) {
            audioSet->setToolTip(tr("Disabled: acmx2 was built without audio support."));
        }
    }
    if (!audio_available) {
        audio_enabled = false;
        record_audio = false;
        audio_passthrough = false;
        audio_file.clear();
        audio_trunc = false;
    }

    if (midiSettingsAction) {
        midiSettingsAction->setEnabled(midi_available);
        if (!midi_available) {
            midiSettingsAction->setToolTip(tr("Disabled: acmx2 was built without MIDI support."));
        }
    }
    if (!midi_available) {
        midi_enabled = false;
        midi_config_file.clear();
        midi_device = -1;
    }
}
