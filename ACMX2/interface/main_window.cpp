#include "main_window.hpp"
#include "audio-window.hpp"
#include "metadata-viewer.hpp"
#include "settings.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QColorDialog>
#include <QDateTime>
#include <QFrame>
#include <QHBoxLayout>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDataStream>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHeaderView>
#include <QIcon>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QFormLayout>
#include <QSpinBox>
#include <QLayout>
#include <QLocale>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QTextStream>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <random>
#include <sstream>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {
QString shellQuote(const QString &value) {
    if (value.isEmpty()) {
        return "''";
    }
    QString out = value;
    out.replace("'", "'\\''");
    return "'" + out + "'";
}

QString buildShellCommand(const QStringList &envAssignments, const QString &program,
                          const QStringList &arguments) {
    QStringList parts;
    parts.reserve(envAssignments.size() + 1 + arguments.size());
    for (const QString &entry : envAssignments) {
        int eq = entry.indexOf('=');
        if (eq <= 0) {
            continue;
        }
        QString key = entry.left(eq);
        QString value = entry.mid(eq + 1);
        parts << (key + "=" + shellQuote(value));
    }
    parts << shellQuote(program);
    for (const QString &arg : arguments) {
        parts << shellQuote(arg);
    }
    return parts.join(' ');
}

#ifdef __linux__
QStringList defaultLinuxRunEnvAssignments() {
    QStringList envAssignments;
    QString uid = QString::number(getuid());
    QString userRunPath = "/run/user/" + uid;
    // Only force the X11 backend when an X server is actually reachable.
    // On Wayland-only sessions (no XWayland) forcing x11 makes SDL fail with
    // "'x11' not available". Leave SDL to auto-detect in that case.
    QByteArray display = qgetenv("DISPLAY");
    QByteArray waylandDisplay = qgetenv("WAYLAND_DISPLAY");
    QByteArray sessionType = qgetenv("XDG_SESSION_TYPE");
    if (!display.isEmpty()) {
        envAssignments << "SDL_VIDEODRIVER=x11";
    } else if (!waylandDisplay.isEmpty() || sessionType == "wayland") {
        envAssignments << "SDL_VIDEODRIVER=wayland";
    }
    if (QDir(userRunPath).exists()) {
        envAssignments << ("XDG_RUNTIME_DIR=" + userRunPath);
        envAssignments << ("PULSE_SERVER=unix:" + userRunPath + "/pulse/native");
    }
    envAssignments << "vblank_mode=0";
    return envAssignments;
}
#endif

QString resolveAssetsPath() {
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    return dirPath + "/../Helpers";
#else
    if (QFileInfo::exists(dirPath + "/data/win-icon.png"))
        return dirPath;
    return QStringLiteral("/usr/local/share/acmx2");
#endif
}

QString resolveShaderCachePath(const QString &libraryPath, int cacheSize) {
    const QString assets = resolveAssetsPath();
    std::error_code ec;
    std::filesystem::path libFsPath(libraryPath.toStdString());
    std::filesystem::path absLib = std::filesystem::absolute(libFsPath, ec);
    std::string key = ec ? libraryPath.toStdString() : absLib.lexically_normal().string();
    key += "|s=" + std::to_string(cacheSize);
    std::ostringstream nameStream;
    nameStream << ".shader_cache_" << std::hex << std::hash<std::string>{}(key);
    const QString filename = QString::fromStdString(nameStream.str());

    // Mirror ShaderLibrary::shaderCacheFilePath: prefer cache in assets dir,
    // then fall back to the library directory itself (acmx2 writes there when
    // assets isn't writable).
    const QString assetsCache = assets + "/" + filename;
    const QString libCache = libraryPath + "/" + filename;
    if (QFileInfo::exists(assetsCache))
        return assetsCache;
    if (QFileInfo::exists(libCache))
        return libCache;
    return assetsCache;
}

// Parse the shader cache file produced by ShaderLibrary::buildShaderCache().
// Returns a map of shader stem -> failed flag. Empty on missing/invalid cache.
QHash<QString, bool> parseShaderCacheStatus(const QString &cachePath) {
    QHash<QString, bool> result;
    QFile f(cachePath);
    if (!f.open(QIODevice::ReadOnly))
        return result;

    auto readU32 = [&](quint32 &v) -> bool {
        return f.read(reinterpret_cast<char *>(&v), sizeof(v)) == qint64(sizeof(v));
    };
    auto readU64 = [&](quint64 &v) -> bool {
        return f.read(reinterpret_cast<char *>(&v), sizeof(v)) == qint64(sizeof(v));
    };
    auto readU8 = [&](quint8 &v) -> bool {
        return f.read(reinterpret_cast<char *>(&v), sizeof(v)) == qint64(sizeof(v));
    };
    auto readStr = [&](QString &out) -> bool {
        quint32 len = 0;
        if (!readU32(len))
            return false;
        QByteArray buf = f.read(len);
        if (quint32(buf.size()) != len)
            return false;
        out = QString::fromUtf8(buf);
        return true;
    };
    auto skipBytes = [&](quint32 n) -> bool { return f.skip(n) == qint64(n); };

    constexpr quint32 CACHE_MAGIC = 0x53484452;
    constexpr quint32 CACHE_VERSION = 3;

    quint32 magic = 0, version = 0;
    if (!readU32(magic) || !readU32(version))
        return result;
    if (magic != CACHE_MAGIC || version != CACHE_VERSION)
        return result;

    QString tmp;
    if (!readStr(tmp))
        return result; // gl_renderer
    if (!readStr(tmp))
        return result; // gl_version

    quint8 dual_mode = 0;
    if (!readU8(dual_mode))
        return result;

    quint32 count = 0;
    if (!readU32(count))
        return result;

    for (quint32 i = 0; i < count; ++i) {
        QString name;
        if (!readStr(name))
            return result;
        quint8 failed_flag = 0;
        if (!readU8(failed_flag))
            return result;
        quint64 source_hash = 0;
        if (!readU64(source_hash))
            return result;
        quint32 fmt2d = 0, sz2d = 0, fmt3d = 0, sz3d = 0;
        if (!readU32(fmt2d) || !readU32(sz2d) || !skipBytes(sz2d))
            return result;
        if (!readU32(fmt3d) || !readU32(sz3d) || !skipBytes(sz3d))
            return result;
        result.insert(name, failed_flag != 0);
    }
    return result;
}

QString formatLastModified(const QDateTime &dt) {
    if (!dt.isValid())
        return QStringLiteral("-");
    return dt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm"));
}
} // namespace

void MainWindow::initControls() {
    lastFoundIndex = -1;
    lastSearchText = QString();
    process = new QProcess(this);
    initShaderSelectionSharedMemory();
    connect(process, &QProcess::stateChanged, this, [this](QProcess::ProcessState state) {
        if (listMenu_set_current) {
            listMenu_set_current->setEnabled(state == QProcess::Running);
        }
    });
    connect(process, &QProcess::readyReadStandardOutput, this, [this]() {
        QString output = process->readAllStandardOutput();
        output.replace("\n", "<br>");
        this->Write(output);
    });

    connect(process, &QProcess::readyReadStandardError, this, [this]() {
        auto writeStderrLine = [this](const QString &line) {
            if (line.contains("GStreamer")) return;
            if (line.contains("[ WARN:"))
                this->Write("<b style='color:#ccaa00;'>Warning:</b> " + line + "<br>");
            else
                this->Write("<b style='color:red;'>Error:</b> " + line + "<br>");
        };

        stderrBuffer += process->readAllStandardError();
        int idx;
        while ((idx = stderrBuffer.indexOf('\n')) != -1) {
            QString line = stderrBuffer.left(idx);
            stderrBuffer.remove(0, idx + 1);
            writeStderrLine(line);
        }
        if (stderrBuffer.size() > 4096) {
            writeStderrLine(stderrBuffer);
            stderrBuffer.clear();
        }
    });

    connect(process,
            static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus) {
                if (!stderrBuffer.isEmpty() && !stderrBuffer.contains("GStreamer")) {
                    if (stderrBuffer.contains("[ WARN:"))
                        this->Write("<b style='color:#ccaa00;'>Warning:</b> " + stderrBuffer + "<br>");
                    else
                        this->Write("<b style='color:red;'>Error:</b> " + stderrBuffer + "<br>");
                    stderrBuffer.clear();
                }
                QString text;
                QTextStream stream(&text);
                stream << "acmx2: Exited with Code: " << exitCode;
                Log(text + "<br>");
                play_stop->setEnabled(false);

                // Refresh the shader tree's compile-health column now that
                // the child process has (re)written the binary shader cache.
                populateShaderTree();

                if (cacheBuildInProgress) {
                    if (exitCode == 0) {
                        shaderCacheMarkedStaleBySave = false;
                    }
                    cacheBuildInProgress = false;
                }

                // If this exit was a shader cache rebuild triggered by a
                // pending Run, launch the actual session now.
                if (pendingLaunchAfterBuild) {
                    const QStringList queuedArgs = pendingLaunchArguments;
                    pendingLaunchAfterBuild = false;
                    pendingLaunchArguments.clear();
                    if (exitCode == 0) {
                        Log("Cache rebuild finished; launching acmx2...");
                        Log("shell: acmx2 " + concatList(queuedArgs) + "<br>");
                        process->start(executable_path, queuedArgs);
                        if (!process->waitForStarted()) {
                            Log("<b style='color:red;'>Failed to start the program.</b>");
                        } else {
                            play_stop->setEnabled(true);
                        }
                    } else {
                        Log("<b style='color:red;'>Cache rebuild failed; "
                            "aborting queued launch.</b>");
                    }
                    return;
                }

                // Optional post-process: convert the produced HLG HDR file
                // to HDR10 via ffmpeg and stream its output to the log.
                if (convert_to_hdr10 && exitCode == 0 && !output_file.isEmpty() &&
                    QFileInfo::exists(output_file)) {
                    runHdr10Conversion();
                }
            });

    hdr10Process = new QProcess(this);
    connect(hdr10Process, &QProcess::readyReadStandardOutput, this, [this]() {
        QString output = QString::fromUtf8(hdr10Process->readAllStandardOutput());
        output.replace("\n", "<br>");
        this->Write(output);
    });
    connect(hdr10Process, &QProcess::readyReadStandardError, this, [this]() {
        QString output = QString::fromUtf8(hdr10Process->readAllStandardError());
        output.replace("\n", "<br>");
        // ffmpeg writes progress to stderr; render in a neutral colour rather
        // than the alarming red used for acmx2 errors.
        this->Write("<span style='color:#88aaff;'>" + output + "</span>");
    });
    connect(hdr10Process,
            static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus) {
                QString text;
                QTextStream stream(&text);
                stream << "ffmpeg (HDR10): Exited with Code: " << exitCode;
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
    QAction *metadataAction = new QAction(tr("Media Metadata Viewer..."), this);
    connect(metadataAction, &QAction::triggered, this, &MainWindow::menuMetadataViewer);
    viewMenu->addSeparator();
    viewMenu->addAction(metadataAction);
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
    runMenu->addSeparator();
    runMenu_copyCommand = new QAction(tr("Edit Command"), this);
    connect(runMenu_copyCommand, &QAction::triggered, this, &MainWindow::copyCommand);
    runMenu->addAction(runMenu_copyCommand);
    runMenu->addSeparator();
    QAction *runMenu_clearLog = new QAction(tr("Clear Log"), this);
    connect(runMenu_clearLog, &QAction::triggered, this, [this]() {
        bottomTextBox->clear();
    });
    runMenu->addAction(runMenu_clearLog);
    play_repeat = new QAction(tr("Repeat"), this);
    play_repeat->setCheckable(true);
    play_repeat->setChecked(false);
    playbackMenu->addAction(play_repeat);
    play_stop = new QAction(tr("Stop"), this);
    play_stop->setEnabled(false);
    connect(play_stop, &QAction::triggered, this, [=]() {
        if (process->state() == QProcess::Running) {
            // If the user stops during a pending cache rebuild, cancel the
            // queued launch so we don't auto-run after termination.
            pendingLaunchAfterBuild = false;
            pendingLaunchArguments.clear();
            process->terminate();
        }
        if (hdr10Process && hdr10Process->state() == QProcess::Running) {
            hdr10Process->terminate();
        }
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
#ifdef Q_OS_MACOS
    // macOS does not support the persistent binary shader cache.
    buildCacheAction->setVisible(false);
    buildCacheAction->setEnabled(false);
#endif

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

    playbackMenu->addSeparator();
    watermarkAction = new QAction(tr("Watermark..."), this);
    connect(watermarkAction, &QAction::triggered, this, &MainWindow::menuWatermarkSettings);
    playbackMenu->addAction(watermarkAction);

    displayFilterAction = new QAction(tr("Display"), this);
    displayFilterAction->setCheckable(true);
    displayFilterAction->setChecked(false);
    connect(displayFilterAction, &QAction::toggled, this, &MainWindow::menuToggleDisplayFilter);
    playbackMenu->addAction(displayFilterAction);

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
    listMenu_set_current = new QAction(tr("Set Current Shader"), this);
    listMenu_set_current->setEnabled(false);
    connect(listMenu_set_current, &QAction::triggered, this, &MainWindow::menuSetCurrentShader);
    listMenu->addAction(listMenu_set_current);
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
    list_view = new QTreeWidget(this);
    list_view->setColumnCount(4);
    list_view->setHeaderLabels({tr("#"), tr("Name"), tr("Last Modified"), tr("Compile Health")});
    list_view->setRootIsDecorated(false);
    list_view->setUniformRowHeights(true);
    list_view->setAlternatingRowColors(false);
    list_view->setSelectionMode(QAbstractItemView::SingleSelection);
    list_view->setSelectionBehavior(QAbstractItemView::SelectRows);
    list_view->setContextMenuPolicy(Qt::CustomContextMenu);
    list_view->setSortingEnabled(false);
    list_view->setAllColumnsShowFocus(true);
    list_view->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    list_view->header()->setSectionResizeMode(1, QHeaderView::Stretch);
    list_view->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    list_view->header()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
#ifdef Q_OS_MACOS
    // macOS does not support the persistent shader cache; hide the column.
    list_view->setColumnHidden(3, true);
#endif
    list_view->setStyleSheet(
        "QTreeWidget { background-color: black; color: white; font-size: 18px;"
        " font-family: 'Courier New', Courier, monospace; }"
        "QHeaderView::section { background-color: #110000; color: lime;"
        " font-family: 'Courier New', Courier, monospace; padding: 4px;"
        " border: 1px solid #330000; }");
    list_view->setToolTip(tr("Right click while running to change the active shader."));
    bottomTextBox = new QTextEdit(this);
    bottomTextBox->setHtml("<b style='color:red;'>ACMX2</b> - Interface: Loaded.");
    bottomTextBox->setStyleSheet("QTextEdit { background-color: black; color: lime; font-size: 24px; font-family: 'Courier New', Courier, monospace;; }");
    bottomTextBox->setReadOnly(true);
    connect(list_view, &QTreeWidget::doubleClicked,
            this, &MainWindow::listClicked);
    connect(list_view, &QTreeWidget::customContextMenuRequested,
            this, [this](const QPoint &pos) {
                if (!list_view)
                    return;
                if (QTreeWidgetItem *item = list_view->itemAt(pos)) {
                    list_view->setCurrentItem(item);
                    publishSelectedShaderIndexToRunningProcess();
                    if (process && process->state() == QProcess::Running) {
                        return;
                    }
                }
                if (listMenu) {
                    listMenu->exec(list_view->viewport()->mapToGlobal(pos));
                }
            });
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
    watermark_enabled = appSettings.value("watermarkEnabled", false).toBool();
    watermark_text = appSettings.value("watermarkText", "").toString();
    watermark_r = appSettings.value("watermarkR", 255).toInt();
    watermark_g = appSettings.value("watermarkG", 0).toInt();
    watermark_b = appSettings.value("watermarkB", 150).toInt();
    display_filter_enabled = appSettings.value("displayFilter", false).toBool();
    autopilot_frames = appSettings.value("playlistAutopilotFrames", 4).toInt();
    if (autopilot_frames < 4) {
        autopilot_frames = 4;
    }
    autopilot_random = appSettings.value("playlistAutopilotRandom", false).toBool();
    if (displayFilterAction) {
        QSignalBlocker blocker(displayFilterAction);
        displayFilterAction->setChecked(display_filter_enabled);
    }
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
    if (items.isEmpty()) {
        QMessageBox::information(this, tr("Search Shaders"),
                                 tr("No shaders are loaded."));
        return;
    }
    int foundIndex = -1;

    for (int i = 0; i < items.size(); ++i) {
        if (items[i].compare(searchText, Qt::CaseInsensitive) == 0) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex == -1) {
        for (int i = 0; i < items.size(); ++i) {
            if (items[i].contains(searchText, Qt::CaseInsensitive)) {
                foundIndex = i;
                break;
            }
        }
    }

    if (foundIndex != -1) {
        lastFoundIndex = foundIndex;
        selectShaderRow(foundIndex);
        Log("Found shader: " + items[foundIndex] + " at index " + QString::number(foundIndex));
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

    if (items.isEmpty()) {
        return;
    }

    int foundIndex = -1;
    int startIndex = (lastFoundIndex + 1) % items.size();

    for (int i = startIndex; i < items.size(); ++i) {
        if (items[i].contains(lastSearchText, Qt::CaseInsensitive)) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex == -1 && startIndex > 0) {
        for (int i = 0; i < startIndex; ++i) {
            if (items[i].contains(lastSearchText, Qt::CaseInsensitive)) {
                foundIndex = i;
                break;
            }
        }
    }

    if (foundIndex != -1) {
        lastFoundIndex = foundIndex;
        selectShaderRow(foundIndex);
        Log("Found next: " + items[foundIndex] + " at index " + QString::number(foundIndex));
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
    int row = currentShaderRow();
    if (row < 0 || row >= items.size())
        return;
    items.removeAt(row);
    populateShaderTree();
    updateIndex();
    loadShaders(shader_path, true);
}

void MainWindow::menuSetCurrentShader() {
    if (!process || process->state() != QProcess::Running)
        return;
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size()) {
        Log("No shader selected.");
        return;
    }
    publishSelectedShaderIndexToRunningProcess();
    Log("Set current shader to index " + QString::number(row) + ".");
}

void MainWindow::updateIndex() {
    QFile file(shader_path + "/index.txt");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return;

    QTextStream out(&file);
    QStringList writtenItems;
    const int rowCount = items.size();

    for (int row = 0; row < rowCount; ++row) {
        const QString shaderName = items.at(row).trimmed();
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
        populateShaderTree();
        Log("Updated shader list, removed " + QString::number(rowCount - writtenItems.size()) +
            " non-existent files");
    }
}

void MainWindow::menuUp() {
    const int row = currentShaderRow();
    if (row <= 0 || row >= items.size())
        return;
    items.swapItemsAt(row, row - 1);
    populateShaderTree();
    selectShaderRow(row - 1);
    updateIndex();
}

void MainWindow::menuDown() {
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size() - 1)
        return;
    items.swapItemsAt(row, row + 1);
    populateShaderTree();
    selectShaderRow(row + 1);
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
    const int row = i.row();
    if (row < 0 || row >= items.size())
        return;
    QString itemText = sanitizeShaderName(items.at(row));
    if (itemText.isEmpty()) {
        Log("Invalid shader name");
        return;
    }
    cleanupClosedEditors();
    TextEditor *editor = new TextEditor(this);
    QString filePath = shader_path + "/" + itemText;
    editor->setText(readFileContents(filePath));
    editor->setFileName(filePath);
    connect(editor, &TextEditor::fileSaved, this, [this](const QString &) {
        shaderCacheMarkedStaleBySave = true;
        populateShaderTree();
    });
    open_files.append(editor);
    editor->show();
}

QString MainWindow::currentShaderName() const {
    QTreeWidgetItem *it = list_view ? list_view->currentItem() : nullptr;
    if (!it)
        return QString();
    const int row = list_view->indexOfTopLevelItem(it);
    if (row < 0 || row >= items.size())
        return it->text(1);
    return items.at(row);
}

int MainWindow::currentShaderRow() const {
    if (!list_view)
        return -1;
    QTreeWidgetItem *it = list_view->currentItem();
    if (!it)
        return -1;
    return list_view->indexOfTopLevelItem(it);
}

void MainWindow::initShaderSelectionSharedMemory() {
#ifdef __linux__
    if (shaderSelectionShm)
        return;

    shaderSelectionShmFd = ::shm_open(acmx2::ipc::kShaderSelectionShmName,
                                      O_CREAT | O_RDWR,
                                      0666);
    if (shaderSelectionShmFd < 0) {
        return;
    }

    if (::ftruncate(shaderSelectionShmFd, sizeof(acmx2::ipc::ShaderSelectionShmData)) != 0) {
        ::close(shaderSelectionShmFd);
        shaderSelectionShmFd = -1;
        return;
    }

    void *mapped = ::mmap(nullptr,
                          sizeof(acmx2::ipc::ShaderSelectionShmData),
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED,
                          shaderSelectionShmFd,
                          0);
    if (mapped == MAP_FAILED) {
        ::close(shaderSelectionShmFd);
        shaderSelectionShmFd = -1;
        return;
    }

    shaderSelectionShm = static_cast<acmx2::ipc::ShaderSelectionShmData *>(mapped);
    if (shaderSelectionShm->magic != acmx2::ipc::kShaderSelectionMagic ||
        shaderSelectionShm->version != acmx2::ipc::kShaderSelectionVersion) {
        shaderSelectionShm->magic = acmx2::ipc::kShaderSelectionMagic;
        shaderSelectionShm->version = acmx2::ipc::kShaderSelectionVersion;
        shaderSelectionShm->selected_index = -1;
        shaderSelectionShm->sequence = 0;
    }
    shaderSelectionSequence = shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishSelectedShaderIndexToRunningProcess() {
#ifdef __linux__
    if (!shaderSelectionShm)
        return;
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size())
        return;
    shaderSelectionShm->selected_index = row;
    shaderSelectionShm->sequence = ++shaderSelectionSequence;
#endif
}

void MainWindow::cleanupShaderSelectionSharedMemory() {
#ifdef __linux__
    if (shaderSelectionShm) {
        ::munmap(shaderSelectionShm, sizeof(acmx2::ipc::ShaderSelectionShmData));
        shaderSelectionShm = nullptr;
    }
    if (shaderSelectionShmFd >= 0) {
        ::close(shaderSelectionShmFd);
        shaderSelectionShmFd = -1;
    }
#endif
}

void MainWindow::selectShaderRow(int row) {
    if (!list_view || row < 0 || row >= list_view->topLevelItemCount())
        return;
    QTreeWidgetItem *it = list_view->topLevelItem(row);
    if (!it)
        return;
    list_view->setCurrentItem(it);
    list_view->scrollToItem(it, QAbstractItemView::PositionAtCenter);
}

void MainWindow::refreshShaderCacheStatus() {
    shaderCacheStatus.clear();
    shaderCacheMTime = QDateTime();
    if (shader_path.isEmpty())
        return;
    const QString cachePath = resolveShaderCachePath(shader_path, cache_size);
    QFileInfo cacheInfo(cachePath);
    if (!cacheInfo.exists() || !cacheInfo.isFile()) {
        Log("Shader cache not found at: " + cachePath);
        return;
    }
    shaderCacheMTime = cacheInfo.lastModified();
    shaderCacheStatus = parseShaderCacheStatus(cachePath);
    //Log("Shader cache: " + cachePath + " (" + QString::number(shaderCacheStatus.size()) + " entries)");
}

bool MainWindow::isShaderCacheStale() const {
#ifdef Q_OS_MACOS
    // macOS does not support the persistent binary shader cache; never
    // report staleness so we never trigger an auto-rebuild.
    return false;
#else
    if (!use_shader_cache || shader_path.isEmpty() || items.isEmpty())
           return false;
    const QString cachePath = resolveShaderCachePath(shader_path, cache_size);
    QFileInfo cacheInfo(cachePath);
    if (!cacheInfo.exists() || !cacheInfo.isFile())
        return true;
    const QDateTime cacheMTime = cacheInfo.lastModified();
    for (const QString &name : items) {
        QFileInfo src(shader_path + "/" + name);
        if (src.exists() && src.lastModified() > cacheMTime)
            return true;
    }
    return false;
#endif
}

void MainWindow::populateShaderTree() {
    if (!list_view)
        return;
    refreshShaderCacheStatus();

    // Preserve the currently selected row so a refresh (e.g. after the
    // child process exits) does not lose the user's place in the list.
    const int previousRow = currentShaderRow();

    const QSignalBlocker blocker(list_view);
    list_view->clear();

    const int width = QString::number(items.size()).size();
    for (int i = 0; i < items.size(); ++i) {
        const QString &name = items.at(i);
        QFileInfo fi(shader_path + "/" + name);
        const QString stem = QFileInfo(name).completeBaseName();

        QString health;
        QColor healthColor;
        if (shaderCacheStatus.isEmpty()) {
            health = tr("No cache");
            healthColor = QColor("#888888");
        } else if (!shaderCacheStatus.contains(stem)) {
            health = tr("Uncached");
            healthColor = QColor("#cccc00");
        } else if (shaderCacheStatus.value(stem)) {
            health = tr("Failed");
            healthColor = QColor("#ff5555");
        } else if (fi.exists() && shaderCacheMTime.isValid() &&
                   fi.lastModified() > shaderCacheMTime) {
            health = tr("Stale");
            healthColor = QColor("#ffaa00");
        } else {
            health = tr("Cached");
            healthColor = QColor("#55ff55");
        }

        QStringList cols;
        cols << QString("%1").arg(i, width, 10, QLatin1Char(' '))
             << name
             << (fi.exists() ? formatLastModified(fi.lastModified()) : tr("missing"))
             << health;
        auto *item = new QTreeWidgetItem(list_view, cols);
        item->setTextAlignment(0, Qt::AlignRight | Qt::AlignVCenter);
        item->setForeground(3, QBrush(healthColor));
        if (!fi.exists())
            item->setForeground(2, QBrush(QColor("#ff5555")));
    }

    // Restore the previously selected row after the rebuild.
    if (previousRow >= 0 && previousRow < list_view->topLevelItemCount()) {
        QTreeWidgetItem *it = list_view->topLevelItem(previousRow);
        if (it) {
            list_view->setCurrentItem(it);
            list_view->scrollToItem(it, QAbstractItemView::PositionAtCenter);
        }
    }
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

        Log("Executable Path: " + exePath);
        Log("Prefix Path: " + prefix);
        Log("Shader Directory: " + shaderDir);

        // Force a reload so the list reflects the newly selected library
        // even if index timestamps happen to be unchanged.
        if (loadShaders(shaderDir, true)) {
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
    const int previousRow = currentShaderRow();
    const QString previouslySelected = currentShaderName();
    items.clear();
    QStringList uniqueItems;
    QTextStream in(&file);

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();

        if (line.isEmpty()) {
            continue;
        }
        const QString shaderEntry = sanitizeShaderName(line);
        if (shaderEntry.isEmpty()) {
            Log("Skipping invalid shader path in index.txt: " + line);
            continue;
        }
        QString fullPath = path + "/" + shaderEntry;
        QFileInfo fileInfo(fullPath);
        if (!fileInfo.exists() || !fileInfo.isFile()) {
            Log("Skipping non-existent file: " + shaderEntry);
            continue;
        }
        if (!uniqueItems.contains(shaderEntry, Qt::CaseInsensitive)) {
            uniqueItems.append(shaderEntry);
        } else {
            Log("Skipping duplicate shader: " + shaderEntry);
        }
    }
    file.close();
    items = uniqueItems;

    Log("Loaded " + QString::number(items.size()) + " unique shader files");
    populateShaderTree();
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
        selectShaderRow(restoredRow);
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
        audio_buffers_enabled = audio_set.isAudioBuffersEnabled();
        audio_buffer_frames = audio_set.getAudioBufferFrames();
        audio_warm_rate = audio_set.getAudioWarmRate();
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

void MainWindow::menuToggleDisplayFilter(bool checked) {
    display_filter_enabled = checked;
    QSettings appSettings("LostSideDead");
    appSettings.setValue("displayFilter", display_filter_enabled);
    Log(QString("Display Filter Overlay: %1").arg(display_filter_enabled ? "Enabled" : "Disabled"));
}

void MainWindow::menuWatermarkSettings() {
    QDialog dlg(this);
    dlg.setWindowTitle(tr("Watermark Settings"));

    auto *enableCheck = new QCheckBox(tr("Enable watermark in recorded video"), &dlg);
    enableCheck->setChecked(watermark_enabled);

    auto *textEdit = new QLineEdit(watermark_text, &dlg);
    textEdit->setPlaceholderText(tr("Watermark text (shown upper-left of recorded video)"));

    auto *colorPreview = new QLabel(&dlg);
    colorPreview->setAutoFillBackground(true);
    colorPreview->setMinimumSize(80, 24);
    colorPreview->setFrameStyle(QFrame::Box | QFrame::Plain);
    colorPreview->setAlignment(Qt::AlignCenter);

    int curR = watermark_r, curG = watermark_g, curB = watermark_b;
    auto applyPreview = [colorPreview, &curR, &curG, &curB]() {
        QPalette pal = colorPreview->palette();
        pal.setColor(QPalette::Window, QColor(curR, curG, curB));
        QColor fg = (curR * 0.299 + curG * 0.587 + curB * 0.114) > 140 ? Qt::black : Qt::white;
        pal.setColor(QPalette::WindowText, fg);
        colorPreview->setPalette(pal);
        colorPreview->setText(QString(" %1, %2, %3 ").arg(curR).arg(curG).arg(curB));
    };
    applyPreview();

    auto *colorBtn = new QPushButton(tr("Choose Color..."), &dlg);
    QObject::connect(colorBtn, &QPushButton::clicked, &dlg, [&]() {
        QColor chosen = QColorDialog::getColor(QColor(curR, curG, curB), &dlg, tr("Watermark Color"));
        if (chosen.isValid()) {
            curR = chosen.red();
            curG = chosen.green();
            curB = chosen.blue();
            applyPreview();
        }
    });

    auto *form = new QFormLayout();
    form->addRow(enableCheck);
    form->addRow(tr("Text:"), textEdit);
    auto *colorRow = new QHBoxLayout();
    colorRow->addWidget(colorPreview, 1);
    colorRow->addWidget(colorBtn);
    form->addRow(tr("Color:"), colorRow);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);

    auto *layout = new QVBoxLayout(&dlg);
    layout->addLayout(form);
    layout->addWidget(buttons);

    if (dlg.exec() != QDialog::Accepted) {
        return;
    }

    watermark_enabled = enableCheck->isChecked();
    watermark_text = textEdit->text();
    watermark_r = curR;
    watermark_g = curG;
    watermark_b = curB;

    QSettings appSettings("LostSideDead");
    appSettings.setValue("watermarkEnabled", watermark_enabled);
    appSettings.setValue("watermarkText", watermark_text);
    appSettings.setValue("watermarkR", watermark_r);
    appSettings.setValue("watermarkG", watermark_g);
    appSettings.setValue("watermarkB", watermark_b);

    Log(QString("Watermark %1: \"%2\" color=%3,%4,%5")
            .arg(watermark_enabled ? "Enabled" : "Disabled")
            .arg(watermark_text)
            .arg(watermark_r).arg(watermark_g).arg(watermark_b));
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
    playlistDialog.setAutopilotFrames(autopilot_frames);
    playlistDialog.setAutopilotRandom(autopilot_random);

    if (playlistDialog.exec() == QDialog::Accepted) {
        playlist_enabled = playlistDialog.isPlaylistEnabled();
        playlist_names = playlistDialog.getSelectedShaderNames();
        playlist_tree_data = playlistDialog.getPlaylistTree();
        playlist_file_path = playlistDialog.getPlaylistFile();
        autopilot_frames = playlistDialog.getAutopilotFrames();
        autopilot_random = playlistDialog.isAutopilotRandom();
        QSettings appSettings("LostSideDead");
        appSettings.setValue("playlistAutopilotFrames", autopilot_frames);
        appSettings.setValue("playlistAutopilotRandom", autopilot_random);
        if (playlist_enabled) {
            Log("Playlist Settings Saved: " + QString::number(playlist_names.size()) + " shaders");
            if (!playlist_file_path.isEmpty()) {
                Log("Playlist file: " + playlist_file_path);
            }
            if (autopilot_frames > 0) {
                Log(QString("Autopilot timeout mode: %1 (%2 frames)")
                        .arg(autopilot_random ? "random" : "fixed")
                        .arg(autopilot_frames));
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
            cache_size = settingsWindow.getCacheSize();
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
            cache_size = 8;
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
            cache_enabled = settingsWindow.isTextureCacheEnabled();
            cache_delay = settingsWindow.getCacheDelay();
            cache_size = settingsWindow.getCacheSize();
            use_yuv = settingsWindow.isUseYuvEnabled();
        }
        if (settingsWindow.isSavingToOutputVideoFile()) {
            output_file = settingsWindow.getOutputVideoFile();
        } else {
            output_file = "";
        }
        // Only meaningful in input-video mode + with an output file. The
        // settings dialog already gates this on HDR detection, but we re-check
        // here so it stays consistent if other modes are selected.
        convert_to_hdr10 = settingsWindow.isConvertToHdr10Enabled() &&
                           settingsWindow.isUsingInputVideoFile() &&
                           settingsWindow.isSavingToOutputVideoFile();
    }
    enable_3d = settingsWindow.is3dEnabled();
    model_file = settingsWindow.getModelFile();
    onnx_model_enabled = settingsWindow.isOnnxModelEnabled();
    onnx_model = settingsWindow.getOnnxModelFile();
    cuda_device = settingsWindow.getSelectedCudaDevice();
    time_speed = settingsWindow.getTimeSpeed();
    duration_limit_enabled = settingsWindow.isDurationLimitEnabled();
    max_duration = settingsWindow.getDurationLimit();
    max_size_limit_enabled = settingsWindow.isMaxSizeLimitEnabled();
    max_size_mb = settingsWindow.getMaxSizeLimit();
    cross_fade_duration = settingsWindow.getCrossFadeDuration();
    flip_enabled = settingsWindow.isFlipEnabled();
    png_output = settingsWindow.isPngOutputEnabled();
    generate_enabled = settingsWindow.isGenerateEnabled();
    generate_interval = settingsWindow.getGenerateInterval();
    encode_preset = settingsWindow.getEncodePreset();
    encode_tune = settingsWindow.getEncodeTune();
    encode_crf = settingsWindow.getEncodeCrf();
    encode_codec = settingsWindow.getEncodeCodec();
    encode_realtime = settingsWindow.isEncodeRealtime();
    encode_no_drop = settingsWindow.isEncodeNoDrop();
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
    // Only force x11 when DISPLAY is set; on Wayland-only sessions the x11
    // SDL backend is not available and SDL would error out.
    QByteArray display = qgetenv("DISPLAY");
    QByteArray waylandDisplay = qgetenv("WAYLAND_DISPLAY");
    QByteArray sessionType = qgetenv("XDG_SESSION_TYPE");
    if (!display.isEmpty()) {
        env.insert("SDL_VIDEODRIVER", "x11");
    } else if (!waylandDisplay.isEmpty() || sessionType == "wayland") {
        env.insert("SDL_VIDEODRIVER", "wayland");
    }
    if (QDir(user_run_path).exists()) {
        env.insert("XDG_RUNTIME_DIR", user_run_path);
        env.insert("PULSE_SERVER", "unix:" + user_run_path + "/pulse/native");
    }
    env.insert("vblank_mode", "0");
    process->setProcessEnvironment(env);
#endif

    if (shader_path.length() == 0) {
        QMessageBox::information(this, "Select Shaders", "Select Shader Path");
        return;
    }
    publishSelectedShaderIndexToRunningProcess();
    const QString data = currentShaderName();
    if (data.isEmpty()) {
        Log("<b>No item selected.</b>");
        return;
    }
    QStringList arguments;
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/acmx2";
#endif
    if (!QFileInfo::exists(dirPath + "/data/win-icon.png"))
        dirPath = "/usr/local/share/acmx2";
    const int selectedIndex = currentShaderRow();
    if (selectedIndex < 0 || selectedIndex >= items.size()) {
        Log("<b>No valid shader selection.</b>");
        return;
    }
    // Single-shader run: use --fragment to bypass library load/compile entirely.
    // Only the selected shader gets compiled; no binary cache lookup, no 1700+
    // shader compile pass.
    const QString fragmentPath = shader_path + "/" + data;
    arguments << "--path" << dirPath
              << "--fragment" << fragmentPath;
    // Pass texture cache size so the SIZE macro injected into the fragment
    // matches whatever the user has configured for cache shaders.
    arguments << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
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
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
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
        if (encode_no_drop)
            arguments << "--no-drop";
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

    if (audio_available && audio_buffers_enabled) {
        arguments << "--enable-audio-buffers" << QString::number(audio_buffer_frames);
    }

    if (audio_available && (audio_enabled || !audio_file.isEmpty())) {
        arguments << "--audio-warm-rate" << QString::number(audio_warm_rate, 'f', 2);
    }

    if (enable_3d) {
        arguments << "--enable-3d";
        arguments << "--model" << model_file;
    }

    if (onnx_model_enabled && !onnx_model.isEmpty()) {
        arguments << "--onnx" << onnx_model;
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

    if (max_size_limit_enabled && max_size_mb > 0.0 && !output_file.isEmpty()) {
        arguments << "--max-size" << QString::number(max_size_mb, 'f', 2);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    if (flip_enabled) {
        arguments << "--flip";
    }

    if (png_output && !output_file.isEmpty()) {
        arguments << "--png";
    }

    if (generate_enabled && generate_interval > 0) {
        arguments << "--generate" << QString::number(generate_interval);
    }

    if (watermark_enabled && !watermark_text.isEmpty()) {
        arguments << "--use-watermark" << watermark_text;
        arguments << "--use-watermark-color"
                  << QString("%1,%2,%3").arg(watermark_r).arg(watermark_g).arg(watermark_b);
    }

    if (display_filter_enabled) {
        arguments << "--display-filter";
    }

    // Single-shader (--fragment) mode does not use the library binary cache,
    // so skip the auto-rebuild gate that's needed for full library runs.
    Log("shell: acmx2 " + concatList(arguments) + "<br>");
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Failed to start the program.</b>");
        QMessageBox::critical(this, "Error", "Failed to start the program.");
    } else {
        play_stop->setEnabled(true);
    }
}

bool MainWindow::buildRunArguments(QStringList &arguments) {
    if (shader_path.length() == 0) {
        QMessageBox::information(this, "Select Shaders", "Select Shader Path");
        return false;
    }
    int index = 0;
    const int row = currentShaderRow();
    if (row < 0) {
        index = 0;
        Log("No selection, defaulting to index 0");
    } else {
        index = row;
        const QString selectedData = currentShaderName();
        Log("Selected shader: " + selectedData + " at index: " + QString::number(index));
    }
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/acmx2";
#endif
    if (!QFileInfo::exists(dirPath + "/data/win-icon.png"))
        dirPath = "/usr/local/share/acmx2";

    QString shader_file = shader_path;
    arguments << "--path" << dirPath << "--shaders" << shader_file;
    // Always pass texture cache size so runtime SIZE matches the cache file.
    arguments << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
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
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
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
        if (encode_no_drop)
            arguments << "--no-drop";
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

    if (audio_available && audio_buffers_enabled) {
        arguments << "--enable-audio-buffers" << QString::number(audio_buffer_frames);
    }

    if (audio_available && (audio_enabled || !audio_file.isEmpty())) {
        arguments << "--audio-warm-rate" << QString::number(audio_warm_rate, 'f', 2);
    }

    if (enable_3d) {
        arguments << "--enable-3d";
        arguments << "--model" << model_file;
    }

    if (onnx_model_enabled && !onnx_model.isEmpty()) {
        arguments << "--onnx" << onnx_model;
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

    if (autopilot_frames > 0) {
        arguments << (autopilot_random ? "--autopilot-random" : "--autopilot-frames")
                  << QString::number(autopilot_frames);
    }

    if (duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (max_size_limit_enabled && max_size_mb > 0.0 && !output_file.isEmpty()) {
        arguments << "--max-size" << QString::number(max_size_mb, 'f', 2);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    if (flip_enabled) {
        arguments << "--flip";
    }

    if (png_output && !output_file.isEmpty()) {
        arguments << "--png";
    }

    if (generate_enabled && generate_interval > 0) {
        arguments << "--generate" << QString::number(generate_interval);
    }

    if (watermark_enabled && !watermark_text.isEmpty()) {
        arguments << "--use-watermark" << watermark_text;
        arguments << "--use-watermark-color"
                  << QString("%1,%2,%3").arg(watermark_r).arg(watermark_g).arg(watermark_b);
    }

    if (display_filter_enabled) {
        arguments << "--display-filter";
    }

    return true;
}

void MainWindow::runHdr10Conversion() {
    if (!hdr10Process) {
        return;
    }
    if (hdr10Process->state() == QProcess::Running) {
        Log("<b style='color:red;'>HDR10 conversion already running; skipping.</b>");
        return;
    }
    if (output_file.isEmpty() || !QFileInfo::exists(output_file)) {
        Log("<b style='color:red;'>HDR10 conversion: source file missing.</b>");
        return;
    }

    QFileInfo fi(output_file);
    const QString suffix = fi.suffix();
    const QString hdr10Path = fi.absolutePath() + "/" + fi.completeBaseName() +
                              ".HDR10" + (suffix.isEmpty() ? QString() : "." + suffix);

    QStringList args;
    args << "-y"
         << "-i" << output_file;

    // Honor the user's codec selection from the recording settings dialog.
    // Values come from the encodeCodecComboBox: "auto", "software", "nvenc".
    // "auto" picks NVENC if CUDA is available, otherwise libx265.
    const QString codecChoice = encode_codec.toLower();
    bool useNvenc;
    if (codecChoice == "software" || codecChoice == "libx265" || codecChoice == "x265") {
        useNvenc = false;
    } else if (codecChoice == "nvenc" || codecChoice == "hevc_nvenc") {
        useNvenc = true;
    } else {
        useNvenc = cuda_available; // "auto" or empty
    }

    if (useNvenc) {
        // NVENC HEVC HDR10 path. p010le = 10-bit 4:2:0 semi-planar, required
        // by hevc_nvenc Main10. NVENC's preset namespace is p1..p7 (fastest
        // -> slowest); map the x264-style names from the UI combo onto it.
        QString nvencPreset;
        const QString p = encode_preset.toLower();
        if (p == "ultrafast")      nvencPreset = "p1";
        else if (p == "superfast") nvencPreset = "p2";
        else if (p == "veryfast")  nvencPreset = "p3";
        else if (p == "faster")    nvencPreset = "p4";
        else if (p == "fast")      nvencPreset = "p5";
        else if (p == "medium")    nvencPreset = "p6";
        else if (p == "slow")      nvencPreset = "p6";
        else if (p == "slower")    nvencPreset = "p7";
        else if (p == "veryslow")  nvencPreset = "p7";
        else if (p.startsWith("p") && p.size() == 2 && p[1].isDigit())
            nvencPreset = p; // already an NVENC preset
        else
            nvencPreset = "p6";

        args << "-vf" << "zscale=p=bt2020:t=smpte2084:m=bt2020nc,format=p010le"
             << "-c:v" << "hevc_nvenc"
             << "-preset" << nvencPreset
             << "-tune" << "hq"
             << "-b:v" << "56M"
             << "-maxrate" << "60M"
             << "-bufsize" << "60M"
             << "-color_primaries" << "bt2020"
             << "-colorspace" << "bt2020nc"
             << "-color_trc" << "smpte2084";
         Log("HDR10 codec: hevc_nvenc (CUDA detected, preset=" + nvencPreset + ")<br>");
    } else {
         args << "-vf" << "zscale=p=bt2020:t=smpte2084:m=bt2020nc,format=yuv420p10le"
             << "-c:v" << "libx265"
             << "-preset" << (encode_preset.isEmpty() ? QStringLiteral("medium") : encode_preset)
             << "-b:v" << "56M"
             << "-maxrate" << "60M"
             << "-bufsize" << "60M"
             << "-pix_fmt" << "yuv420p10le"
             << "-x265-params"
             << "hdr10=1:hdr10-opt=1:repeat-headers=1:"
                "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:"
                "master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1):"
                "max-cll=1000,400"
             << "-color_primaries" << "bt2020"
             << "-colorspace" << "bt2020nc"
             << "-color_trc" << "smpte2084";
        Log("HDR10 codec: libx265 (codec=" + (codecChoice.isEmpty() ? QStringLiteral("auto") : codecChoice) + ")<br>");
    }

    args << "-c:a" << "copy"
         << hdr10Path;

    Log("shell: ffmpeg " + concatList(args) + "<br>");
    Log("HDR10 output: " + hdr10Path + "<br>");

    // ffmpeg writes most of its progress to stderr; merge channels so the
    // log keeps messages in source order.
    hdr10Process->setProcessChannelMode(QProcess::MergedChannels);
    hdr10Process->start("ffmpeg", args);
    if (!hdr10Process->waitForStarted(5000)) {
        Log("<b style='color:red;'>Failed to start ffmpeg for HDR10 conversion.</b>");
        return;
    }
    play_stop->setEnabled(true);
}

void MainWindow::runAll() {
    if (process->state() == QProcess::Running) {
        QMessageBox::information(this, "Process Running", "A process is already running. Please stop it first.");
        return;
    }

#ifdef __linux__
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    for (const QString &entry : defaultLinuxRunEnvAssignments()) {
        int eq = entry.indexOf('=');
        if (eq <= 0) {
            continue;
        }
        env.insert(entry.left(eq), entry.mid(eq + 1));
    }
    process->setProcessEnvironment(env);
#endif

    QStringList arguments;
    if (!buildRunArguments(arguments))
        return;
    publishSelectedShaderIndexToRunningProcess();

    const bool firstRunAllInvocation = firstRunAllPendingRebuild;
    firstRunAllPendingRebuild = false;
    const bool shouldForceInitialRebuild = firstRunAllInvocation && use_shader_cache;

    if (shouldForceInitialRebuild || shaderCacheMarkedStaleBySave || isShaderCacheStale()) {
        if (shouldForceInitialRebuild) {
            Log("First Run All after startup: rebuilding shader cache before launch.");
        } else {
            Log("Shader cache is out of date; rebuilding then launching automatically.");
        }
        pendingLaunchArguments = arguments;
        pendingLaunchAfterBuild = true;
        menuBuildShaderCache();
        return;
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

void MainWindow::copyCommand() {
    QStringList arguments;
    if (!buildRunArguments(arguments))
        return;

    QString exe = executable_path;
    if (exe.isEmpty())
        exe = "acmx2";
    QStringList envAssignments;
#ifdef __linux__
    envAssignments = defaultLinuxRunEnvAssignments();
#endif
    QString commandText = buildShellCommand(envAssignments, exe, arguments).trimmed();

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Copy Command"));
    dialog.resize(720, 320);

    QVBoxLayout *layout = new QVBoxLayout(&dialog);
    QPlainTextEdit *textBox = new QPlainTextEdit(&dialog);
    textBox->setPlainText(commandText);
    textBox->setReadOnly(false);
    textBox->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    textBox->setStyleSheet("QPlainTextEdit { background-color: black; color: lime; "
                           "font-size: 14px; font-family: 'Courier New', Courier, monospace; "
                           "border: 1px solid red; }");
    layout->addWidget(textBox);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(&dialog);
    QPushButton *copyButton = buttonBox->addButton(tr("Copy to Clipboard"), QDialogButtonBox::ActionRole);
    QPushButton *runButton = buttonBox->addButton(tr("Run"), QDialogButtonBox::ActionRole);
    QPushButton *okButton = buttonBox->addButton(QDialogButtonBox::Ok);
    layout->addWidget(buttonBox);

    QString style = "QDialog { background-color: black; border: 3px solid red; }"
                    "QLabel { color: red; }"
                    "QPushButton { border: 1px solid red; background-color: #110000; color: red; padding: 5px; }"
                    "QPushButton:hover { background-color: red; color: black; }";
    QSettings appSettings("LostSideDead");
    if (appSettings.value("useCustomStyle", false).toBool()) {
        dialog.setStyleSheet(style);
    }

    connect(copyButton, &QPushButton::clicked, &dialog, [textBox, &dialog]() {
        const QString copiedText = textBox->toPlainText();
        QClipboard *clipboard = QGuiApplication::clipboard();
        clipboard->setText(copiedText, QClipboard::Clipboard);
#ifdef __linux__
        if (clipboard->supportsSelection()) {
            clipboard->setText(copiedText, QClipboard::Selection);
        }
#endif
        QCoreApplication::processEvents();
        QMessageBox::information(&dialog, tr("Copied"),
                                 tr("Command copied to clipboard."));
    });
    connect(runButton, &QPushButton::clicked, &dialog, [this, textBox, &dialog]() {
        if (process->state() != QProcess::NotRunning) {
            QMessageBox::information(&dialog, tr("Process Running"),
                                     tr("A process is already running. Please stop it first."));
            return;
        }
        QString cmdText = textBox->toPlainText().trimmed();
        if (cmdText.isEmpty()) {
            QMessageBox::warning(&dialog, tr("Empty Command"), tr("The command is empty."));
            return;
        }

        // Run the command verbatim through a shell so that env-var prefixes,
        // quoting, and PATH lookup behave exactly like pasting it into a
        // terminal. This avoids any ambiguity from re-parsing the line into
        // tokens and re-applying environment via QProcessEnvironment.
        process->setProcessEnvironment(QProcessEnvironment::systemEnvironment());
#ifdef Q_OS_WIN
        QString shell = qEnvironmentVariable("COMSPEC");
        if (shell.isEmpty()) shell = "cmd.exe";
        QStringList shellArgs{"/C", cmdText};
#else
        QString shell = "/bin/sh";
        QStringList shellArgs{"-c", cmdText};
#endif
        Log("shell: " + cmdText + "<br>");
        process->start(shell, shellArgs);
        if (!process->waitForStarted()) {
            Log("<b style='color:red;'>Failed to start the program.</b>");
            QMessageBox::critical(&dialog, tr("Error"), tr("Failed to start the program."));
            return;
        }
        play_stop->setEnabled(true);
        dialog.accept();
    });
    connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);

    dialog.exec();
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
    sanitized.replace('\\', '/');
    sanitized = QDir::cleanPath(sanitized);

    while (sanitized.startsWith("./")) {
        sanitized = sanitized.mid(2);
    }

    if (sanitized.isEmpty() || sanitized == "." || sanitized == "..") {
        Log("Warning: Invalid shader name detected: " + name);
        return QString();
    }

    if (QDir::isAbsolutePath(sanitized) ||
        sanitized.startsWith("../") ||
        sanitized.contains("/../") ||
        sanitized.endsWith("/..")) {
        Log("Warning: Invalid shader name detected (path traversal attempt): " + name);
        return QString();
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
    if (items.isEmpty()) {
        return;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(items.begin(), items.end(), g);
    populateShaderTree();
    updateIndex();
    Log("Shaders shuffled");
}

void MainWindow::menuSort() {
    if (items.isEmpty()) {
        return;
    }
    items.sort(Qt::CaseInsensitive);
    populateShaderTree();
    updateIndex();
    Log("Shaders sorted alphabetically");
}

void MainWindow::menuBuildShaderCache() {
#ifdef Q_OS_MACOS
    // macOS does not support the persistent binary shader cache.
    Log("Rebuild Shader Cache is not available on macOS.");
    cacheBuildInProgress = false;
    pendingLaunchAfterBuild = false;
    pendingLaunchArguments.clear();
    return;
#else
    QString build_path = shader_path;
    if (build_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        build_path = appSettings.value("shaders", "").toString();
    }

    if (build_path.isEmpty()) {
        cacheBuildInProgress = false;
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }

    if (process->state() == QProcess::Running) {
        cacheBuildInProgress = false;
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
    args << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);

    if (enable_3d) {
        args << "--enable-3d";
    }

    Log("Building shader cache for: " + build_path);
    Log("Command: " + executable_path + " " + args.join(" "));

    play_stop->setEnabled(true);
    cacheBuildInProgress = true;
    process->start(executable_path, args);

    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start shader cache build process");
        cacheBuildInProgress = false;
        play_stop->setEnabled(false);
    }
#endif
}

void MainWindow::menuRunFromCache() {
}

void MainWindow::menuMetadataViewer() {
    MetadataViewer dlg(this);
    dlg.exec();
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
    if (reply != QMessageBox::Yes)
        return;

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
    if (enable_3d)
        args << "--enable-3d";

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
                                            "index.txt was not changed.")
                                             .arg(exitCode));
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

    // Determine assets path (same logic acmx2 uses for --path resolution).
    QString assetsPath = QCoreApplication::applicationDirPath();
    if (!QFileInfo::exists(assetsPath + "/data/win-icon.png"))
        assetsPath = "/usr/local/share/acmx2";

    // Compute hashed cache filename matching ShaderLibrary::shaderCacheFilePath.
    std::error_code ec;
    std::filesystem::path libFsPath(recompile_path.toStdString());
    std::filesystem::path absLib = std::filesystem::absolute(libFsPath, ec);
    std::string key = ec ? recompile_path.toStdString()
                         : absLib.lexically_normal().string();
    std::ostringstream nameStream;
    nameStream << ".shader_cache_" << std::hex
               << std::hash<std::string>{}(key);
    QString cacheFile = assetsPath + "/" + QString::fromStdString(nameStream.str());

    bool removedAny = false;
    QFile cache(cacheFile);
    if (cache.exists()) {
        if (cache.remove()) {
            Log("Deleted shader cache: " + cacheFile);
            removedAny = true;
        } else {
            Log("<b style='color:red;'>Warning:</b> Could not delete cache file: " + cacheFile);
        }
    }
    // Also clean up any legacy cache file stored alongside the library.
    QString legacyCache = recompile_path + "/.shader_cache";
    QFile legacy(legacyCache);
    if (legacy.exists()) {
        if (legacy.remove()) {
            Log("Deleted legacy shader cache: " + legacyCache);
            removedAny = true;
        }
    }
    if (!removedAny) {
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
