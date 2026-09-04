#include "main_window.hpp"
#include "audio-window.hpp"
#include "custom-uniforms.hpp"
#include "custom_style.hpp"
#include "find-shader.hpp"
#include "library-builder.hpp"
#include "metadata-viewer.hpp"
#include "settings.hpp"
#include "shader-manifest.hpp"
#include "uniform-reference.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QColorDialog>
#include <QComboBox>
#include <QDataStream>
#include <QDateTime>
#include <QDebug>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QIcon>
#include <QInputDialog>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QLocale>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTextStream>
#include <QTimer>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {
    constexpr int RECENT_LIBRARY_LIMIT = 10;

    QString shellQuote(const QString &value) {
#ifdef _WIN32
        if (value.isEmpty()) {
            return QStringLiteral("\"\"");
        }

        QString quoted = QStringLiteral("\"");
        qsizetype backslash_count = 0;
        for (const QChar character : value) {
            if (character == QLatin1Char('\\')) {
                ++backslash_count;
                continue;
            }
            if (character == QLatin1Char('"')) {
                quoted += QString(backslash_count * 2 + 1, QLatin1Char('\\'));
                quoted += character;
                backslash_count = 0;
                continue;
            }
            quoted += QString(backslash_count, QLatin1Char('\\'));
            backslash_count = 0;
            quoted += character;
        }
        quoted += QString(backslash_count * 2, QLatin1Char('\\'));
        quoted += QLatin1Char('"');
        return quoted;
#else
        if (value.isEmpty()) {
            return "''";
        }
        QString out = value;
        out.replace("'", "'\\''");
        return "'" + out + "'";
#endif
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
#ifdef _WIN32
            parts << (QStringLiteral("set \"") + key + QLatin1Char('=') +
                      value + QStringLiteral("\" &&"));
#else
            parts << (key + "=" + shellQuote(value));
#endif
        }
        parts << shellQuote(program);
        for (const QString &arg : arguments) {
            parts << shellQuote(arg);
        }
        return parts.join(' ');
    }

    void replace_file(const std::filesystem::path &source,
                      const std::filesystem::path &destination,
                      std::error_code &error) {
#ifdef _WIN32
        if (MoveFileExW(source.c_str(), destination.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) !=
            FALSE) {
            error.clear();
            return;
        }
        error = std::error_code(static_cast<int>(GetLastError()),
                                std::system_category());
#else
        std::filesystem::rename(source, destination, error);
#endif
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
        const QString installedPath = QDir::cleanPath(dirPath + "/../share/acmx2");
        if (QFileInfo::exists(installedPath + "/data/win-icon.png"))
            return installedPath;
        return dirPath;
#endif
    }

    QString resolve_acmxvk_shader_compiler(QString &error) {
        error.clear();
        QSettings settings("LostSideDead");
        const QString mode =
            settings
                .value(acmx2::backend_settings_key(
                           acmx2::Backend::Acmxvk,
                           "shader_compiler_mode"),
                       "auto")
                .toString();
        if (mode == QStringLiteral("custom")) {
            QString compiler =
                settings
                    .value(acmx2::backend_settings_key(
                        acmx2::Backend::Acmxvk,
                        "shader_compiler_path"))
                    .toString()
                    .trimmed();
            if (compiler.isEmpty()) {
                error = QStringLiteral(
                    "The custom ACMXVK shader compiler path is empty. Select "
                    "one in Properties (Ctrl+,).");
                return {};
            }
            if (QFileInfo(compiler).isRelative()) {
                const QString resolved =
                    QStandardPaths::findExecutable(compiler);
                if (!resolved.isEmpty())
                    compiler = resolved;
            }
            const QFileInfo compilerInfo(compiler);
            if (!compilerInfo.isFile() || !compilerInfo.isExecutable()) {
                error = QStringLiteral(
                            "The configured ACMXVK shader compiler is not an "
                            "executable file: %1")
                            .arg(compiler);
                return {};
            }
            return compilerInfo.absoluteFilePath();
        }

        QString compiler =
            QStandardPaths::findExecutable(QStringLiteral("glslc"));
        if (compiler.isEmpty()) {
            const QString sdk =
                QString::fromLocal8Bit(qgetenv("VULKAN_SDK"));
            const QString sdkCompiler =
                QDir(sdk).filePath(QStringLiteral("bin/glslc"));
            if (!sdk.isEmpty() && QFileInfo(sdkCompiler).isExecutable())
                compiler = sdkCompiler;
        }
        if (compiler.isEmpty()) {
            error = QStringLiteral(
                "glslc was not found in PATH or VULKAN_SDK. Select a custom "
                "compiler in Properties (Ctrl+,).");
        }
        return compiler;
    }

    QString resolve_backend_assets_path(acmx2::Backend backend,
                                        const QString &executable,
                                        const QString &libraryPath) {
        if (backend == acmx2::Backend::Acmx2)
            return resolveAssetsPath();

        const QString applicationDir = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
        const QString bundleResources =
            QDir::cleanPath(applicationDir + "/../Resources/acmxvk");
        if (QFileInfo(bundleResources).isDir())
            return bundleResources;
#endif
        QStringList candidates;
        QString resolvedExecutable = executable;
        if (QFileInfo(resolvedExecutable).isRelative()) {
            const QString pathExecutable =
                QStandardPaths::findExecutable(resolvedExecutable);
            if (!pathExecutable.isEmpty())
                resolvedExecutable = pathExecutable;
        }
        const QFileInfo executableInfo(resolvedExecutable);
        if (!executableInfo.absolutePath().isEmpty()) {
            candidates << QDir::cleanPath(executableInfo.absolutePath() +
                                          "/../share/acmxvk");
        }
        if (!libraryPath.isEmpty())
            candidates << QDir::cleanPath(QFileInfo(libraryPath).absolutePath());
        candidates << QDir::cleanPath(applicationDir + "/../share/acmxvk")
                   << QStringLiteral("/usr/local/share/acmxvk")
                   << QStringLiteral("/opt/homebrew/share/acmxvk")
                   << QStringLiteral("/usr/share/acmxvk");
        for (const QString &candidate : candidates) {
            if (QFileInfo(candidate + "/data").isDir())
                return candidate;
        }

        // --path only requires a readable directory. ACMXVK can still use the
        // explicitly selected shader library if no installed data tree exists.
        return applicationDir;
    }

    bool is_acmxvk_source_library(const QString &libraryPath, QString &error) {
        error.clear();
        const std::optional<acmx2::ShaderLibraryType> type =
            acmx2::shader_manifest_library_type(libraryPath, error);
        if (!error.isEmpty())
            return false;
        if (type)
            return *type == acmx2::ShaderLibraryType::Source;

        // Legacy manifests may not carry library_type. Infer source libraries
        // from GLSL entries while continuing to accept old SPIR-V manifests.
        QStringList entries;
        if (!acmx2::load_shader_manifest(libraryPath, entries, error))
            return false;
        return std::any_of(entries.cbegin(), entries.cend(), [](const QString &entry) {
            return entry.endsWith(".frag", Qt::CaseInsensitive) ||
                   entry.endsWith(".comp", Qt::CaseInsensitive);
        });
    }

    QString acmxvk_build_directory(const QString &sourceLibrary) {
        return QDir(sourceLibrary).filePath(QStringLiteral(".acmxvk-build"));
    }

    QString acmxvk_runtime_shader_name(const QString &sourceName) {
        return sourceName.endsWith(".spv", Qt::CaseInsensitive)
                   ? sourceName
                   : sourceName + QStringLiteral(".spv");
    }

    enum class AcmxvkBuildState { UpToDate,
                                  Stale,
                                  NotBuilt };

    AcmxvkBuildState acmxvk_shader_build_state(
        const QString &source_library, const QString &source_name) {
        const QString runtime_library =
            acmxvk_build_directory(source_library);
        if (!acmx2::shader_manifest_exists(runtime_library))
            return AcmxvkBuildState::NotBuilt;

        const QFileInfo source_file(
            QDir(source_library).filePath(source_name));
        const QFileInfo runtime_file(
            QDir(runtime_library)
                .filePath(acmxvk_runtime_shader_name(source_name)));
        if (!runtime_file.isFile())
            return AcmxvkBuildState::NotBuilt;
        if (source_file.isFile() &&
            runtime_file.lastModified() < source_file.lastModified()) {
            return AcmxvkBuildState::Stale;
        }
        return AcmxvkBuildState::UpToDate;
    }

    bool acmxvk_runtime_manifest_matches(const QString &source_library,
                                         const QString &runtime_library,
                                         QString &error) {
        QStringList source_entries;
        QStringList runtime_entries;
        if (!acmx2::load_shader_manifest(source_library, source_entries,
                                         error) ||
            !acmx2::load_shader_manifest(runtime_library, runtime_entries,
                                         error)) {
            return false;
        }

        QStringList expected_entries;
        expected_entries.reserve(source_entries.size());
        for (const QString &entry : source_entries)
            expected_entries.append(acmxvk_runtime_shader_name(entry));
        if (runtime_entries != expected_entries) {
            error = QObject::tr(
                "The ACMXVK runtime manifest does not match the source "
                "shader list. Choose Playback > Build before running.");
            return false;
        }

        bool uniformMetadataMatches = false;
        if (!acmx2::custom_uniform_metadata_matches(
                source_library, runtime_library, uniformMetadataMatches,
                error)) {
            return false;
        }
        if (!uniformMetadataMatches) {
            error = QObject::tr(
                "The ACMXVK runtime custom-uniform metadata is out of date. "
                "Choose Playback > Build before running.");
            return false;
        }
        return true;
    }

    bool resolve_acmxvk_runtime_library(const QString &selectedLibrary,
                                        QString &runtimeLibrary,
                                        QString &error) {
        error.clear();
        runtimeLibrary = selectedLibrary;
        if (!is_acmxvk_source_library(selectedLibrary, error))
            return error.isEmpty();

        runtimeLibrary = acmxvk_build_directory(selectedLibrary);
        if (!acmx2::shader_manifest_exists(runtimeLibrary)) {
            error = QObject::tr(
                        "The ACMXVK source library has not been built yet. "
                        "Choose Playback > Build first.\n\nExpected output: %1")
                        .arg(runtimeLibrary);
            return false;
        }
        const std::optional<acmx2::ShaderLibraryType> type =
            acmx2::shader_manifest_library_type(runtimeLibrary, error);
        if (!error.isEmpty())
            return false;
        if (type && *type != acmx2::ShaderLibraryType::Runtime) {
            error = QObject::tr("Compiled output is not an ACMXVK runtime library: %1")
                        .arg(runtimeLibrary);
            return false;
        }
        QStringList sourceEntries;
        if (!acmx2::load_shader_manifest(selectedLibrary, sourceEntries, error))
            return false;
        if (!acmxvk_runtime_manifest_matches(selectedLibrary, runtimeLibrary,
                                             error)) {
            return false;
        }
        for (const QString &sourceEntry : sourceEntries) {
            const QFileInfo sourceFile(
                QDir(selectedLibrary).filePath(sourceEntry));
            const QFileInfo runtimeFile(QDir(runtimeLibrary)
                                            .filePath(acmxvk_runtime_shader_name(
                                                sourceEntry)));
            if (!runtimeFile.isFile() ||
                runtimeFile.lastModified() < sourceFile.lastModified()) {
                error = QObject::tr(
                            "The ACMXVK build is missing or older than %1. "
                            "Choose Playback > Build before running.")
                            .arg(sourceEntry);
                return false;
            }
        }
        return true;
    }

    bool textureCacheArraySettingEnabled() {
        QSettings settings("LostSideDead", "acmx2");
        return settings.value("interface/texture_cache_array", false).toBool();
    }

    QSize storedResolution(QSettings &settings, const QString &key,
                           const QSize &fallback, bool defaultIsEmpty) {
        const QString text = settings.value(key).toString().trimmed();
        if (text.compare("Default", Qt::CaseInsensitive) == 0) {
            return defaultIsEmpty ? QSize(0, 0) : fallback;
        }

        static const QRegularExpression resolutionPattern(
            R"(^\s*(\d+)\s*[xX]\s*(\d+)\s*$)");
        const QRegularExpressionMatch match = resolutionPattern.match(text);
        if (!match.hasMatch()) {
            return fallback;
        }

        const int width = match.captured(1).toInt();
        const int height = match.captured(2).toInt();
        return width > 0 && height > 0 ? QSize(width, height) : fallback;
    }

    bool hasPositiveResolution(const QSize &resolution) {
        return resolution.width() > 0 && resolution.height() > 0;
    }

    QString shaderCacheFilename(const QString &libraryPath, int cacheSize,
                                bool useArray) {
        std::error_code ec;
        const std::filesystem::path libraryFsPath(libraryPath.toStdString());
        const std::filesystem::path absoluteLibrary =
            std::filesystem::absolute(libraryFsPath, ec);
        std::string key = ec ? libraryPath.toStdString()
                             : absoluteLibrary.lexically_normal().string();
        key += "|s=" + std::to_string(cacheSize);
        key += "|a=" + std::to_string(useArray ? 1 : 0);
        std::ostringstream nameStream;
        nameStream << ".shader_cache_" << std::hex
                   << std::hash<std::string>{}(key);
        return QString::fromStdString(nameStream.str());
    }

    QString resolveShaderCachePath(const QString &libraryPath, int cacheSize,
                                   bool useArray) {
        const QString assets = resolveAssetsPath();
        const QString filename =
            shaderCacheFilename(libraryPath, cacheSize, useArray);

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
        constexpr quint32 CACHE_VERSION = 4;

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
            quint8 shader_kind = 0;
            if (!readU8(shader_kind) || shader_kind > 2)
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
    auto updateShaderMenuState = [this](QProcess::ProcessState state) {
        const bool running = (state == QProcess::Running);
        if (backendMenu)
            backendMenu->setEnabled(!running);
        if (listMenu_new) {
            listMenu_new->setEnabled(!running);
        }
        if (listMenu_shader) {
            listMenu_shader->setEnabled(!running);
        }
        if (libraryBuilderAction) {
            libraryBuilderAction->setEnabled(!running);
        }
        if (listMenu_remove) {
            listMenu_remove->setEnabled(!running);
        }
        if (listMenu_up) {
            listMenu_up->setEnabled(!running);
        }
        if (listMenu_down) {
            listMenu_down->setEnabled(!running);
        }
        if (listMenu_shuffle) {
            listMenu_shuffle->setEnabled(!running);
        }
        if (listMenu_sort) {
            listMenu_sort->setEnabled(!running);
        }
        if (listMenu_set_current) {
            listMenu_set_current->setEnabled(running);
        }
    };
    connect(process, &QProcess::stateChanged, this, updateShaderMenuState);
    updateShaderMenuState(process->state());
    connect(process, &QProcess::readyReadStandardOutput, this, [this]() {
        QString output = process->readAllStandardOutput();
        output.replace("\n", "<br>");
        this->Write(output);
    });

    connect(process, &QProcess::readyReadStandardError, this, [this]() {
        auto writeStderrLine = [this](const QString &line) {
            if (line.contains("GStreamer"))
                return;
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
            [this](int exitCode, QProcess::ExitStatus exitStatus) {
                if (!stderrBuffer.isEmpty() && !stderrBuffer.contains("GStreamer")) {
                    if (stderrBuffer.contains("[ WARN:"))
                        this->Write("<b style='color:#ccaa00;'>Warning:</b> " + stderrBuffer + "<br>");
                    else
                        this->Write("<b style='color:red;'>Error:</b> " + stderrBuffer + "<br>");
                    stderrBuffer.clear();
                }
                QString text;
                QTextStream stream(&text);
                stream << acmx2::backend_name(active_backend)
                       << ": Exited with Code: " << exitCode;
                Log(text + "<br>");
                play_stop->setEnabled(false);

                if (exitStatus == QProcess::CrashExit) {
                    qDebug() << acmx2::backend_name(active_backend)
                             << "engine crashed.";
                    Log("<b style='color:red;'>" +
                        acmx2::backend_name(active_backend) +
                        " engine crashed.</b><br>");
                }

                // Refresh the shader tree's compile-health column now that
                // the child process has (re)written the binary shader cache.
                populateShaderTree();

                const bool finishedBuildProcess = cacheBuildInProgress;
                if (cacheBuildInProgress) {
                    const PendingAcmxvkAction resume_action =
                        pending_acmxvk_action;
                    const QString pruneLibraryPath =
                        acmxvkPruneLibraryPath;
                    pending_acmxvk_action = PendingAcmxvkAction::None;
                    acmxvkPruneLibraryPath.clear();
                    if (active_backend == acmx2::Backend::Acmxvk) {
                        if (exitCode == 0) {
                            Log(tr("ACMXVK build ready: %1")
                                    .arg(acmxvk_build_directory(shader_path)));
                        } else {
                            Log(tr("<b style='color:red;'>ACMXVK build failed "
                                   "with exit code %1.</b>")
                                    .arg(exitCode));
                        }
                    }
                    cacheBuildInProgress = false;
                    update_backend_ui();
                    if (!pruneLibraryPath.isEmpty() && exitCode == 0 &&
                        exitStatus == QProcess::NormalExit) {
                        QStringList sourceShaders;
                        QString manifestError;
                        if (!acmx2::load_shader_manifest(
                                pruneLibraryPath, sourceShaders,
                                manifestError)) {
                            Log(tr("<b style='color:red;'>Broken sources were "
                                   "pruned, but the source manifest could not "
                                   "be read: %1</b>")
                                    .arg(manifestError.toHtmlEscaped()));
                            QMessageBox::warning(
                                this, tr("Remove Broken Shaders"),
                                tr("Broken source files were deleted, but the "
                                   "source manifest could not be updated.\n\n%1")
                                    .arg(manifestError));
                        } else {
                            QStringList retainedShaders;
                            int removedCount = 0;
                            const QDir sourceDirectory(pruneLibraryPath);
                            for (const QString &shader : sourceShaders) {
                                const QString suffix =
                                    QFileInfo(shader).suffix().toLower();
                                const bool sourceEntry =
                                    suffix == QStringLiteral("frag") ||
                                    suffix == QStringLiteral("comp");
                                if (sourceEntry &&
                                    !QFileInfo(sourceDirectory.filePath(shader))
                                         .isFile()) {
                                    ++removedCount;
                                } else {
                                    retainedShaders.append(shader);
                                }
                            }

                            if (removedCount > 0 &&
                                !acmx2::write_shader_manifest(
                                    pruneLibraryPath, retainedShaders,
                                    manifestError)) {
                                Log(tr("<b style='color:red;'>Broken sources "
                                       "were pruned, but the source manifest "
                                       "could not be updated: %1</b>")
                                        .arg(manifestError.toHtmlEscaped()));
                                QMessageBox::warning(
                                    this, tr("Remove Broken Shaders"),
                                    tr("%1 source file(s) were permanently "
                                       "deleted, but library.json could not be "
                                       "updated.\n\n%2")
                                        .arg(removedCount)
                                        .arg(manifestError));
                            } else {
                                if (shader_path == pruneLibraryPath)
                                    loadShaders(pruneLibraryPath, true);
                                Log(tr("Remove Broken completed: %1 source "
                                       "shader(s) permanently deleted.")
                                        .arg(removedCount));
                                QMessageBox::information(
                                    this, tr("Remove Broken Shaders"),
                                    removedCount > 0
                                        ? tr("Removed %1 broken source shader(s) "
                                             "and updated library.json.\n\n"
                                             "This deletion cannot be undone.")
                                              .arg(removedCount)
                                        : tr("The build completed and no broken "
                                             "source shaders were found."));
                            }
                        }
                    }
                    if (active_backend == acmx2::Backend::Acmxvk &&
                        exitCode == 0 &&
                        exitStatus == QProcess::NormalExit &&
                        resume_action != PendingAcmxvkAction::None) {
                        Log(tr("ACMXVK build succeeded; resuming the requested "
                               "action."));
                        QTimer::singleShot(0, this, [this, resume_action]() {
                            if (resume_action ==
                                PendingAcmxvkAction::RunSelected) {
                                runSelected();
                            } else if (resume_action ==
                                       PendingAcmxvkAction::RunAll) {
                                runAll();
                            } else if (resume_action ==
                                       PendingAcmxvkAction::CopyCommand) {
                                copyCommand();
                            }
                        });
                    }
                }

                // Optional post-process: convert the produced HLG HDR file
                // to HDR10 via ffmpeg and stream its output to the log.
                if (!finishedBuildProcess && convert_to_hdr10 && exitCode == 0 &&
                    !output_file.isEmpty() &&
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
    backendMenu = menuBarPtr->addMenu(tr("Backend"));
    playbackMenu = menuBarPtr->addMenu(tr("Playback"));
    runMenu = menuBarPtr->addMenu(tr("Run"));
    listMenu = menuBarPtr->addMenu(tr("List"));
    viewMenu = menuBarPtr->addMenu(tr("View"));
    helpMenu = menuBarPtr->addMenu(tr("Help"));
    backendActionGroup = new QActionGroup(this);
    backendActionGroup->setExclusive(true);
    backendAcmx2Action = backendMenu->addAction(tr("ACMX2"));
    backendAcmx2Action->setCheckable(true);
    backendAcmx2Action->setChecked(true);
    backendActionGroup->addAction(backendAcmx2Action);
    backendAcmxvkAction = backendMenu->addAction(tr("ACMXVK"));
    backendAcmxvkAction->setCheckable(true);
    backendActionGroup->addAction(backendAcmxvkAction);
    connect(backendAcmx2Action, &QAction::triggered, this,
            [this]() { set_backend(acmx2::Backend::Acmx2); });
    connect(backendAcmxvkAction, &QAction::triggered, this,
            [this]() { set_backend(acmx2::Backend::Acmxvk); });
    stayOnTopAction = new QAction(tr("Stay on Top"), this);
    stayOnTopAction->setShortcut(QKeySequence("Ctrl+Alt+T"));
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
    metadataAction->setShortcut(QKeySequence("Ctrl+Alt+V"));
    connect(metadataAction, &QAction::triggered, this, &MainWindow::menuMetadataViewer);
    viewMenu->addSeparator();
    viewMenu->addAction(metadataAction);
    fileMenu_loadLibrary = new QAction(tr("Load Library..."), this);
    fileMenu_loadLibrary->setShortcut(QKeySequence::Open);
    connect(fileMenu_loadLibrary, &QAction::triggered, this,
            &MainWindow::menuLoadLibrary);
    fileMenu->addAction(fileMenu_loadLibrary);
    loadRecentMenu = fileMenu->addMenu(tr("Load Recent"));
    loadRecentMenu->menuAction()->setShortcut(QKeySequence("Ctrl+Shift+O"));
    connect(loadRecentMenu, &QMenu::aboutToShow, this,
            &MainWindow::updateRecentLibrariesMenu);
    updateRecentLibrariesMenu();
    fileMenu->addSeparator();
    fileMenu_prop = new QAction(tr("Properties"), this);
    fileMenu_prop->setShortcut(QKeySequence("Ctrl+,"));
    fileMenu->addAction(fileMenu_prop);
    connect(fileMenu_prop, &QAction::triggered, this, &MainWindow::fileOpenProp);
    fileMenu->addSeparator();
    fileMenu_exit = new QAction(tr("Exit"), this);
    fileMenu_exit->setShortcut(QKeySequence::Quit);
    connect(fileMenu_exit, &QAction::triggered, this, &MainWindow::fileExit);
    fileMenu->addAction(fileMenu_exit);
    cameraSet = new QAction(tr("Session Properties"), this);
    cameraSet->setShortcut(QKeySequence("Ctrl+Shift+P"));
    connect(cameraSet, &QAction::triggered, this, &MainWindow::cameraSettings);
    cameraMenu->addAction(cameraSet);
    audioSet = new QAction(tr("Audio Settings"), this);
    audioSet->setShortcut(QKeySequence("Ctrl+Shift+A"));
    connect(audioSet, &QAction::triggered, this, &MainWindow::menuAudioSettings);
    cameraMenu->addAction(audioSet);
    gpuFilterAction = new QAction(tr("GPU Filter Settings"), this);
    gpuFilterAction->setShortcut(QKeySequence("Ctrl+Shift+G"));
    connect(gpuFilterAction, &QAction::triggered, this, &MainWindow::menuGPUFilterSettings);
    cameraMenu->addAction(gpuFilterAction);
    cameraMenu->addSeparator();
    styleSheetAction = new QAction(tr("Use Custom Style"), this);
    styleSheetAction->setShortcut(QKeySequence("Ctrl+Shift+T"));
    styleSheetAction->setCheckable(true);
    styleSheetAction->setChecked(false);
    connect(styleSheetAction, &QAction::triggered, this, &MainWindow::openCustomStyleEditor);
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
    runMenu_copyCommand->setShortcut(QKeySequence("Ctrl+Shift+C"));
    connect(runMenu_copyCommand, &QAction::triggered, this, &MainWindow::copyCommand);
    runMenu->addAction(runMenu_copyCommand);
    runMenu->addSeparator();
    QAction *runMenu_clearLog = new QAction(tr("Clear Log"), this);
    runMenu_clearLog->setShortcut(QKeySequence("Ctrl+L"));
    connect(runMenu_clearLog, &QAction::triggered, this, [this]() {
        bottomTextBox->clear();
    });
    runMenu->addAction(runMenu_clearLog);
    play_repeat = new QAction(tr("Repeat"), this);
    play_repeat->setShortcut(QKeySequence("Ctrl+R"));
    play_repeat->setCheckable(true);
    play_repeat->setChecked(false);
    connect(play_repeat, &QAction::toggled, this, [this](bool) {
        publishRepeatStateToRunningProcess();
    });
    playbackMenu->addAction(play_repeat);
    normalizedTimeAction = new QAction(tr("Normalized Time"), this);
    normalizedTimeAction->setShortcut(QKeySequence("Ctrl+Alt+N"));
    normalizedTimeAction->setCheckable(true);
    normalizedTimeAction->setChecked(false);
    normalizedTimeAction->setToolTip(
        tr("Advance shader time by a fixed amount per output frame."));
    connect(normalizedTimeAction, &QAction::toggled, this, [this](bool checked) {
        normalized_time = checked;
        QSettings settings("LostSideDead", "acmx2");
        settings.setValue("interface/normalized_time", checked);
        publishRuntimeSettingsToRunningProcess();
    });
    playbackMenu->addAction(normalizedTimeAction);
    play_stop = new QAction(tr("Stop"), this);
    play_stop->setShortcut(QKeySequence("Shift+F5"));
    play_stop->setEnabled(false);
    connect(play_stop, &QAction::triggered, this, [=]() {
        if (process->state() == QProcess::Running) {
            process->terminate();
        }
        if (hdr10Process && hdr10Process->state() == QProcess::Running) {
            hdr10Process->terminate();
        }
    });
    playbackMenu->addAction(play_stop);
    playbackMenu->addSeparator();
    shaderPassAction = new QAction(tr("Multi-Pass Shader Settings..."), this);
    shaderPassAction->setShortcut(QKeySequence("Ctrl+Alt+M"));
    connect(shaderPassAction, &QAction::triggered, this, &MainWindow::menuShaderPassSettings);
    playbackMenu->addAction(shaderPassAction);
    playbackMenu->addSeparator();
    playlistAction = new QAction(tr("Shader Playlist Settings..."), this);
    playlistAction->setShortcut(QKeySequence("Ctrl+Alt+P"));
    connect(playlistAction, &QAction::triggered, this, &MainWindow::menuPlaylistSettings);
    playbackMenu->addAction(playlistAction);
    playbackMenu->addSeparator();
    buildCacheAction = new QAction(tr("Rebuild Shader Cache"), this);
    buildCacheAction->setShortcut(QKeySequence("Ctrl+Alt+B"));
    connect(buildCacheAction, &QAction::triggered, this, &MainWindow::menuBuildShaderCache);
    playbackMenu->addAction(buildCacheAction);
    fixBuildAction = new QAction(tr("Fix Build"), this);
    fixBuildAction->setShortcut(QKeySequence("Ctrl+Alt+F"));
    fixBuildAction->setToolTip(
        tr("Build ACMXVK while omitting shaders that fail to compile."));
    connect(fixBuildAction, &QAction::triggered, this,
            &MainWindow::menuFixBuild);
    playbackMenu->addAction(fixBuildAction);
    cleanShaderCacheAction = new QAction(tr("Clean Shader Cache"), this);
    cleanShaderCacheAction->setShortcut(QKeySequence("Ctrl+Alt+C"));
    connect(cleanShaderCacheAction, &QAction::triggered,
            this, &MainWindow::menuCleanShaderCache);
    playbackMenu->addAction(cleanShaderCacheAction);
#ifdef Q_OS_MACOS
    // macOS does not support the persistent binary shader cache.
    buildCacheAction->setVisible(false);
    buildCacheAction->setEnabled(false);
    fixBuildAction->setVisible(false);
    fixBuildAction->setEnabled(false);
    cleanShaderCacheAction->setVisible(false);
    cleanShaderCacheAction->setEnabled(false);
#endif

    removeBrokenAction = new QAction(tr("Remove Broken"), this);
    removeBrokenAction->setShortcut(QKeySequence("Ctrl+Alt+R"));
    connect(removeBrokenAction, &QAction::triggered, this, &MainWindow::menuRemoveBroken);
    playbackMenu->addAction(removeBrokenAction);

    runFromCacheAction = new QAction(tr("Run from Cache"), this);
    runFromCacheAction->setShortcut(QKeySequence("Ctrl+Alt+K"));
    runFromCacheAction->setCheckable(true);
#ifdef Q_OS_MACOS
    use_shader_cache = false;
    runFromCacheAction->setChecked(false);
    runFromCacheAction->setEnabled(false);
    runFromCacheAction->setToolTip(
        tr("Shader binary caching is not supported on macOS."));
#else
    runFromCacheAction->setChecked(true);
#endif
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
    midiSettingsAction->setShortcut(QKeySequence("Ctrl+Alt+I"));
    connect(midiSettingsAction, &QAction::triggered, this, &MainWindow::menuMidiSettings);
    playbackMenu->addAction(midiSettingsAction);

    playbackMenu->addSeparator();
    watermarkAction = new QAction(tr("Watermark..."), this);
    watermarkAction->setShortcut(QKeySequence("Ctrl+Alt+W"));
    connect(watermarkAction, &QAction::triggered, this, &MainWindow::menuWatermarkSettings);
    playbackMenu->addAction(watermarkAction);

    displayFilterAction = new QAction(tr("Display"), this);
    displayFilterAction->setShortcut(QKeySequence("Ctrl+Alt+D"));
    displayFilterAction->setCheckable(true);
    displayFilterAction->setChecked(false);
    connect(displayFilterAction, &QAction::toggled, this, &MainWindow::menuToggleDisplayFilter);
    playbackMenu->addAction(displayFilterAction);

    listMenu_new = new QAction(tr("New Shader Library"), this);
    listMenu_new->setShortcut(QKeySequence("Ctrl+Shift+N"));
    connect(listMenu_new, &QAction::triggered, this, &MainWindow::newList);
    listMenu->addAction(listMenu_new);
    libraryBuilderAction = new QAction(tr("Shader Library Builder..."), this);
    libraryBuilderAction->setShortcut(QKeySequence("Ctrl+Shift+B"));
    connect(libraryBuilderAction, &QAction::triggered, this,
            &MainWindow::menuLibraryBuilder);
    listMenu->addAction(libraryBuilderAction);
    listMenu_shader = new QAction(tr("New Shader File..."), this);
    listMenu_shader->setShortcut(QKeySequence::New);
    connect(listMenu_shader, &QAction::triggered, this, &MainWindow::newShader);
    listMenu->addAction(listMenu_shader);
    customUniformsAction = new QAction(tr("Custom Uniforms..."), this);
    customUniformsAction->setShortcut(QKeySequence("Ctrl+U"));
    connect(customUniformsAction, &QAction::triggered, this,
            &MainWindow::menuCustomUniforms);
    listMenu->addAction(customUniformsAction);
    listMenu->addSeparator();
    listMenu_remove = new QAction(tr("Remove Shader"), this);
    listMenu_remove->setShortcut(QKeySequence::Delete);
    connect(listMenu_remove, &QAction::triggered, this, &MainWindow::menuRemove);
    listMenu->addAction(listMenu_remove);
    listMenu_set_current = new QAction(tr("Set Current Shader"), this);
    listMenu_set_current->setShortcut(QKeySequence("Ctrl+Return"));
    listMenu_set_current->setEnabled(false);
    connect(listMenu_set_current, &QAction::triggered, this, &MainWindow::menuSetCurrentShader);
    listMenu->addAction(listMenu_set_current);
    listMenu->addSeparator();
    listMenu_up = new QAction(tr("Shift Shader Up"), this);
    listMenu_up->setShortcut(QKeySequence("Alt+Up"));
    connect(listMenu_up, &QAction::triggered, this, &MainWindow::menuUp);
    listMenu->addAction(listMenu_up);
    listMenu_down = new QAction(tr("Shift Shader Down"), this);
    listMenu_down->setShortcut(QKeySequence("Alt+Down"));
    connect(listMenu_down, &QAction::triggered, this, &MainWindow::menuDown);
    listMenu->addAction(listMenu_down);
    listMenu_shuffle = new QAction(tr("Shuffle Shaders"), this);
    listMenu_shuffle->setShortcut(QKeySequence("Ctrl+Shift+H"));
    connect(listMenu_shuffle, &QAction::triggered, this, &MainWindow::menuShuffle);
    listMenu->addAction(listMenu_shuffle);

    listMenu_sort = new QAction(tr("Sort Shaders"), this);
    listMenu_sort->setShortcut(QKeySequence("Ctrl+Shift+S"));
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
    listMenu_findInFiles = new QAction(tr("Find in Files..."), this);
    listMenu_findInFiles->setShortcut(QKeySequence("Ctrl+Shift+F"));
    connect(listMenu_findInFiles, &QAction::triggered, this, [this]() {
        if (shader_path.isEmpty() || !QDir(shader_path).exists()) {
            QMessageBox::information(
                this, tr("Find in Files"),
                tr("Load a shader library before searching its files."));
            return;
        }

        auto *dialog = new FindShaderDialog(shader_path, this);
        connect(dialog, &FindShaderDialog::resultActivated, this,
                [this](const QString &filePath, int lineNumber,
                       int columnNumber, int matchLength) {
                    openShaderEditor(filePath, lineNumber, columnNumber, matchLength);
                });
        dialog->show();
        dialog->raise();
        dialog->activateWindow();
    });
    listMenu->addAction(listMenu_findInFiles);
    helpMenu_uniformReference = new QAction(tr("Built-in Uniform Reference..."), this);
    helpMenu_uniformReference->setShortcut(QKeySequence::HelpContents);
    connect(helpMenu_uniformReference, &QAction::triggered, this,
            &MainWindow::menuUniformReference);
    helpMenu->addAction(helpMenu_uniformReference);
    helpMenu->addSeparator();

    helpMenu_about = new QAction("About", this);
    helpMenu_about->setShortcut(QKeySequence("Shift+F1"));

    connect(helpMenu_about, &QAction::triggered, this, [=]() {
        QMessageBox box(this);
        box.setWindowTitle("About ACMX2");
        box.setWindowIcon(QIcon(":/win-icon.png"));
        const QString info =
            QStringLiteral("<p><b>ACMX %1</b><br>"
                           "(C) 2026 %2 Software<br>"
                           "<a href=\"https://lostsidedead.biz\">"
                           "http://lostsidedead.biz</a><br>"
                           "This software is dedicated to all that have "
                           "experienced mental health issues.</p>")
                .arg(QStringLiteral(VERSION_INFO),
                     QStringLiteral(VERSION_AUTHOR));
        box.setTextFormat(Qt::RichText);
        box.setTextInteractionFlags(Qt::TextBrowserInteraction);
        box.setText(info);
        for (QLabel *label : box.findChildren<QLabel *>())
            label->setOpenExternalLinks(true);
        QPixmap bigIcon(":/win-icon.png");
        if (!bigIcon.isNull()) {
            QPixmap resizedIcon = bigIcon.scaled(64, 64, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            box.setIconPixmap(resizedIcon);
        }
        box.exec();
    });
    helpMenu->addAction(helpMenu_about);
    customUniformDialog = new CustomUniformDialog(this);
    connect(customUniformDialog, &CustomUniformDialog::uniformsChanged, this,
            &MainWindow::publishCustomUniformsToRunningProcess);
    connect(customUniformDialog, &CustomUniformDialog::uniformDefinitionsChanged,
            this, [this]() {
                const QString shaderName = currentShaderName();
                if (!shaderName.isEmpty())
                    publishShaderReloadToRunningProcess(
                        QDir(shader_path).filePath(shaderName));
            });
    list_view = new QTreeWidget(this);
    list_view->setColumnCount(5);
    list_view->setHeaderLabels(
        {tr("#"), tr("Name"), tr("Last Modified"), tr("Compile Health"), tr("Type")});
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
    list_view->header()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
#ifdef Q_OS_MACOS
    // macOS does not support the persistent shader cache; hide the column.
    list_view->setColumnHidden(3, true);
#endif
    list_view->setToolTip(tr("Right click while running to change the active shader."));
    bottomTextBox = new QTextEdit(this);
    bottomTextBox->setHtml(
        "<b style='color:red;'>ACMX</b> - Interface: Loaded.");
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
    active_backend = acmx2::backend_from_id(
                         appSettings.value("interface/backend", "acmx2")
                             .toString())
                         .value_or(acmx2::Backend::Acmx2);
    backendAcmx2Action->setChecked(active_backend == acmx2::Backend::Acmx2);
    backendAcmxvkAction->setChecked(active_backend == acmx2::Backend::Acmxvk);
    loadSessionSettings();
    baseAppStyleSheet = qApp->styleSheet();
    const QString legacyLibrary =
        active_backend == acmx2::Backend::Acmx2
            ? appSettings.value("shaders", "").toString()
            : QString();
    QString path = appSettings
                       .value(acmx2::backend_settings_key(active_backend,
                                                          "library"),
                              legacyLibrary)
                       .toString();
    path = path.trimmed();
    while (path.endsWith("/") || path.endsWith("\\")) {
        path.chop(1);
    }
    const QString legacyExecutable =
        active_backend == acmx2::Backend::Acmx2
            ? appSettings.value("exePath", acmx2::default_backend_executable(
                                               acmx2::Backend::Acmx2))
                  .toString()
            : acmx2::default_backend_executable(active_backend);
    executable_path =
        appSettings
            .value(acmx2::backend_settings_key(active_backend, "executable"),
                   legacyExecutable)
            .toString();
    prefix_path = appSettings.value("prefix_path", ".").toString();
    initShaderSelectionSharedMemory();
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
    publishRuntimeSettingsToRunningProcess();
    if (!path.isEmpty()) {
        QFileInfo pathInfo(path);
        if (pathInfo.exists() && pathInfo.isDir() &&
            acmx2::shader_manifest_exists(path)) {
            QString backendError;
            const std::optional<acmx2::Backend> libraryBackend =
                acmx2::shader_manifest_backend(path, backendError);
            if (!backendError.isEmpty()) {
                Log("Warning: Saved shader library backend metadata is invalid: " +
                    backendError);
            } else if (libraryBackend && *libraryBackend != active_backend) {
                Log(tr("Warning: Saved shader library targets %1 while the "
                       "active backend is %2: %3")
                        .arg(acmx2::backend_name(*libraryBackend),
                             acmx2::backend_name(active_backend), path));
            } else {
                shader_path = path;
                loadShaders(path);
                addRecentLibrary(path);
                Log("Successfully loaded saved shader path");
            }
        } else {
            QString errorMsg = "Warning: Saved shader path is invalid: " + path + " - ";
            if (!pathInfo.exists()) {
                errorMsg += "directory does not exist";
            } else if (!pathInfo.isDir()) {
                errorMsg += "path is not a directory";
            } else if (!acmx2::shader_manifest_exists(path)) {
                errorMsg += "library.json or index.txt not found in directory";
            }
            Log(errorMsg);
        }
    }
    update_backend_ui();
    const QString defaultCustomStyleSheet = acmx2::defaultCustomStyleSheet();
    customStyleSheet = appSettings.value("customStyleSheet", defaultCustomStyleSheet).toString();

    applyCustomStyleSheet(useCustomStyle);
}

void MainWindow::loadSessionSettings() {
    QSettings settings("LostSideDead", "acmx2");

    const QString inputMode =
        settings.value("interface/input_mode", "camera").toString();
    const bool videoMode = inputMode == "video";
    const bool graphicsMode = inputMode == "graphic";
    const bool cameraMode = !videoMode && !graphicsMode;

    camera_index = static_cast<unsigned int>(
        std::max(0, settings.value("interface/camera_device", 0).toInt()));
    camera_res = storedResolution(settings, "interface/camera_resolution",
                                  QSize(1280, 720), false);
    screen_res = storedResolution(settings, "interface/screen_resolution",
                                  QSize(0, 0), true);

    output_fps = settings.value("interface/camera_fps", 30.0).toDouble();
    if (output_fps <= 0.0)
        output_fps = 30.0;

    video_file = videoMode
                     ? settings.value("interface/input_video", "").toString()
                     : QString();
    graphics_file = graphicsMode
                        ? settings.value("interface/graphics_file", "").toString()
                        : QString();

    const bool saveOutput =
        settings.value("interface/save_output", false).toBool();
    output_file = saveOutput
                      ? settings.value("interface/output_video", "").toString()
                      : QString();
    full_screen_value =
        settings.value("interface/fullscreen", false).toBool();
    copy_audio = videoMode && saveOutput &&
                 settings.value("interface/copy_audio", false).toBool();

    cache_enabled = !graphicsMode &&
                    settings.value("interface/texture_cache", false).toBool();
    cache_delay = settings.value("interface/cache_delay", 1).toInt();
    cache_size = std::clamp(
        settings.value("interface/cache_size", 8).toInt(), 1, 64);
    use_yuv = cameraMode &&
              settings.value("interface/use_yuv", false).toBool();

    convert_to_hdr10 = videoMode && saveOutput &&
                       settings.value("interface/convert_to_hdr10", false).toBool();
    enable_3d = settings.value("interface/enable_3d", false).toBool();
    model_file = settings.value("interface/model_file", "cube.mxmod.z").toString();
    onnx_model_enabled = settings.value("interface/use_onnx_model", false).toBool();
    onnx_model = settings.value("interface/onnx_model_file", "").toString();
    cuda_device = settings.value("interface/cuda_device", 0).toInt();
    time_speed = settings.value("interface/time_speed", 1.0).toFloat();
    normalized_time =
        settings.value("interface/normalized_time", false).toBool();
    if (normalizedTimeAction) {
        QSignalBlocker blocker(normalizedTimeAction);
        normalizedTimeAction->setChecked(normalized_time);
    }
    duration_limit_enabled =
        settings.value("interface/duration_enabled", false).toBool();
    max_duration = settings.value("interface/duration_seconds", 60.0).toDouble();
    max_size_limit_enabled =
        settings.value("interface/max_size_enabled", false).toBool();
    max_size_mb = settings.value("interface/max_size_mb", 500.0).toDouble();
    cross_fade_duration = settings.value("interface/crossfade", 0.5).toFloat();
    flip_enabled = settings.value("interface/flip", false).toBool();
    rotate_enabled = settings.value("interface/rotate", false).toBool();
    rotation_mode = settings.value("interface/rotation_mode", "clockwise").toString();
    png_output = settings.value("interface/write_png", false).toBool();
    generate_enabled = settings.value("interface/generate_enabled", false).toBool();
    generate_interval = settings.value("interface/generate_interval", 30).toInt();

    encode_preset = settings.value("recording/preset", "medium").toString();
    encode_tune = settings.value("recording/tune", "").toString();
    encode_crf = settings.value("recording/crf", 18).toInt();
    encode_rate_control =
        settings.value("recording/rate_control", "quality").toString();
    encode_bitrate =
        settings.value("recording/bitrate", "10M").toString();
    encode_codec = settings.value("recording/codec", "auto").toString();
    encode_parameters = settings.value("recording/parameters", "").toString();
    encode_realtime = settings.value("recording/realtime", false).toBool();
    encode_no_drop = !cameraMode &&
                     settings.value("recording/no_drop", false).toBool();
    maximize_fps =
        settings.value("interface/acmxvk_maximize_fps", false).toBool();
    use_source_fps =
        settings.value("interface/acmxvk_use_source_fps", false).toBool();
    use_source_audio = use_source_fps &&
                       settings.value("interface/acmxvk_use_source_audio", false)
                           .toBool();
}

void MainWindow::applyMainViewStyles(bool customStyleEnabled) {
    if (list_view) {
        QFont listFont("Courier New");
        listFont.setStyleHint(QFont::Monospace);
        listFont.setPointSize(12);
        list_view->setFont(listFont);

        if (customStyleEnabled) {
            list_view->setStyleSheet("");
        } else {
            list_view->setStyleSheet(
                "QTreeWidget { background-color: black; color: white; font-size: 13px;"
                " font-family: 'Courier New', Courier, monospace; }"
                "QHeaderView::section { background-color: #110000; color: lime;"
                " font-family: 'Courier New', Courier, monospace; padding: 4px;"
                " border: 1px solid #330000; }");
        }
    }

    if (bottomTextBox) {
        QFont logFont("Courier New");
        logFont.setStyleHint(QFont::Monospace);
        logFont.setPointSize(11);
        bottomTextBox->setFont(logFont);

        if (customStyleEnabled) {
            bottomTextBox->setStyleSheet("");
        } else {
            bottomTextBox->setStyleSheet(
                "QTextEdit { background-color: black; color: lime; font-size: 13px;"
                " font-family: 'Courier New', Courier, monospace; }");
        }
    }
}

void MainWindow::applyCustomStyleSheet(bool enable) {
    QSettings appSettings("LostSideDead");
    appSettings.setValue("useCustomStyle", enable);

    if (baseAppStyleSheet.isEmpty()) {
        baseAppStyleSheet = qApp->styleSheet();
    }

    if (enable) {
        qApp->setStyleSheet(customStyleSheet);
    } else {
        qApp->setStyleSheet(baseAppStyleSheet);
    }

    // Keep this window clean so it follows the global app style consistently.
    setStyleSheet("");
    applyMainViewStyles(enable);
}

void MainWindow::openCustomStyleEditor() {
    QSettings appSettings("LostSideDead");
    const bool currentlyEnabled = appSettings.value("useCustomStyle", false).toBool();
    const QString lastPresetName = appSettings.value("customStylePreset", "Current Style").toString();

    auto makePalette = [](const char *winBg, const char *winFg, const char *accent,
                          const char *fieldBg, const char *fieldFg, const char *fieldBorder,
                          const char *btnBg, const char *btnHover, const char *btnFg,
                          const char *menuBg, const char *menuFg,
                          const char *menuSelBg, const char *menuSelFg,
                          const char *selBg, const char *border) {
        acmx2::CustomStylePalette p;
        p.windowBg = winBg;
        p.windowFg = winFg;
        p.accent = accent;
        p.fieldBg = fieldBg;
        p.fieldFg = fieldFg;
        p.fieldBorder = fieldBorder;
        p.buttonBg = btnBg;
        p.buttonHover = btnHover;
        p.buttonFg = btnFg;
        p.menuBg = menuBg;
        p.menuFg = menuFg;
        p.menuSelBg = menuSelBg;
        p.menuSelFg = menuSelFg;
        p.selectionBg = selBg;
        p.border = border;
        return acmx2::buildStyleSheet(p);
    };

    const std::array<QPair<QString, QString>, 26> presetStyles = {{{"Current Style", customStyleSheet},
                                                                   {"Light: Blue & White",
                                                                    makePalette("#f6fbff", "#143a5c", "#2d7cc4",
                                                                                "#ffffff", "#123b61", "#9cc6ea",
                                                                                "#2d7cc4", "#2368a6", "#ffffff",
                                                                                "#eaf5ff", "#143a5c", "#cfe6ff", "#0b2e4d",
                                                                                "#bcdcff", "1px solid #9cc6ea")},
                                                                   {"Light: Slate",
                                                                    makePalette("#f5f7fa", "#1f2a37", "#4b5563",
                                                                                "#ffffff", "#1f2937", "#b6c3d4",
                                                                                "#4b5563", "#374151", "#ffffff",
                                                                                "#e8edf4", "#1f2a37", "#d2dbe7", "#111827",
                                                                                "#cdd5e0", "1px solid #b6c3d4")},
                                                                   {"Light: White & Red",
                                                                    makePalette("#fffdfd", "#5b1515", "#d63b3b",
                                                                                "#ffffff", "#5a1a1a", "#e8bcbc",
                                                                                "#d63b3b", "#bc2f2f", "#ffffff",
                                                                                "#fff4f4", "#5b1515", "#ffdede", "#4b0f0f",
                                                                                "#ffd1d1", "1px solid #e8bcbc")},
                                                                   {"Light: White & Green",
                                                                    makePalette("#fcfffc", "#164529", "#2e9d57",
                                                                                "#ffffff", "#1a4f2f", "#b8dfc7",
                                                                                "#2e9d57", "#25824a", "#ffffff",
                                                                                "#f1fbf4", "#164529", "#d6f3df", "#11361f",
                                                                                "#c9eecf", "1px solid #b8dfc7")},
                                                                   {"Light: White & Blue",
                                                                    makePalette("#fcfdff", "#16395f", "#2f6ed7",
                                                                                "#ffffff", "#1b446f", "#b7d0f0",
                                                                                "#2f6ed7", "#285db7", "#ffffff",
                                                                                "#f1f6ff", "#16395f", "#d9e8ff", "#102b49",
                                                                                "#cddfff", "1px solid #b7d0f0")},
                                                                   {"Light: White & Cyan",
                                                                    makePalette("#fbfeff", "#12404a", "#1ea9bf",
                                                                                "#ffffff", "#14505d", "#b8e2ea",
                                                                                "#1ea9bf", "#198da0", "#ffffff",
                                                                                "#effbfe", "#12404a", "#d5f3f8", "#0e3138",
                                                                                "#c7edf4", "1px solid #b8e2ea")},
                                                                   {"Light: White & Amber",
                                                                    makePalette("#fffefb", "#5a3a12", "#d18b1f",
                                                                                "#ffffff", "#644317", "#ead7b6",
                                                                                "#d18b1f", "#b37518", "#ffffff",
                                                                                "#fff9ed", "#5a3a12", "#ffebcb", "#4a2f0f",
                                                                                "#ffe2b5", "1px solid #ead7b6")},
                                                                   {"Dark: Crimson",
                                                                    makePalette("#0f0608", "#ff637d", "#a02949",
                                                                                "#1b0b10", "#ff8fa3", "#7f2036",
                                                                                "#6f1630", "#8a1f3d", "#ffdfe6",
                                                                                "#16090d", "#ff637d", "#52111f", "#ffd5dc",
                                                                                "#52111f", "2px solid #a02949")},
                                                                   {"Dark: Emerald",
                                                                    makePalette("#06110c", "#7af7c2", "#2c8e68",
                                                                                "#0d1e16", "#95ffd0", "#2c8e68",
                                                                                "#1c6a4d", "#258961", "#dcfff2",
                                                                                "#08160f", "#7af7c2", "#12402d", "#d9fff0",
                                                                                "#12402d", "2px solid #2c8e68")},
                                                                   {"Dark: Indigo",
                                                                    makePalette("#070713", "#c6c8ff", "#5362ba",
                                                                                "#121634", "#d8daff", "#4956a5",
                                                                                "#36439a", "#4453b4", "#eef0ff",
                                                                                "#0d1022", "#c6c8ff", "#232a5a", "#eef0ff",
                                                                                "#232a5a", "2px solid #5362ba")},
                                                                   {"Dark: Black & Red",
                                                                    makePalette("#050505", "#ff4d4d", "#d90000",
                                                                                "#120808", "#ff7b7b", "#b50000",
                                                                                "#2a0c0c", "#3a1010", "#ffd6d6",
                                                                                "#0b0707", "#ff5a5a", "#6b1111", "#ffe9e9",
                                                                                "#5a0c0c", "2px solid #d90000")},
                                                                   {"Dark: Black & Green",
                                                                    makePalette("#040704", "#6dfb88", "#22b44a",
                                                                                "#0a140b", "#a8ffbe", "#1d9a3e",
                                                                                "#12331b", "#164425", "#e1ffe8",
                                                                                "#08100a", "#74ff95", "#12331b", "#e7ffed",
                                                                                "#10381d", "2px solid #22b44a")},
                                                                   {"Dark: Black & Blue",
                                                                    makePalette("#04060a", "#81b9ff", "#2f6ed7",
                                                                                "#0a1222", "#b4d4ff", "#2a5eb7",
                                                                                "#132749", "#1a3260", "#e7f1ff",
                                                                                "#070d1a", "#8cc0ff", "#1a3260", "#eef5ff",
                                                                                "#17335f", "2px solid #2f6ed7")},
                                                                   {"Dark: Black & Cyan",
                                                                    makePalette("#030809", "#7defff", "#1ba8c3",
                                                                                "#09161a", "#b8f7ff", "#1990a7",
                                                                                "#10323a", "#14414b", "#e7fbff",
                                                                                "#071015", "#89f3ff", "#0f3943", "#e8fcff",
                                                                                "#0f3943", "2px solid #1ba8c3")},
                                                                   {"Dark: Black & Amber",
                                                                    makePalette("#090704", "#ffd77a", "#d88c1d",
                                                                                "#1a1308", "#ffe7b4", "#bf7a19",
                                                                                "#3d2810", "#523618", "#fff3db",
                                                                                "#130e07", "#ffdf8a", "#5a3a16", "#fff4df",
                                                                                "#5a3a16", "2px solid #d88c1d")},
                                                                   {"Light: Lavender Mist",
                                                                    makePalette("#f8f6ff", "#302653", "#7157c8",
                                                                                "#ffffff", "#34295b", "#c9bdea",
                                                                                "#7157c8", "#5d45ae", "#ffffff",
                                                                                "#eee9ff", "#302653", "#ded5ff", "#241a48",
                                                                                "#d9d0ff", "1px solid #c9bdea")},
                                                                   {"Light: Rose Quartz",
                                                                    makePalette("#fff8fa", "#532535", "#c25578",
                                                                                "#ffffff", "#5b293c", "#e8c1cf",
                                                                                "#c25578", "#a94465", "#ffffff",
                                                                                "#fff0f4", "#532535", "#f6d7e1", "#411a28",
                                                                                "#f1ccd8", "1px solid #e8c1cf")},
                                                                   {"Light: Sandstone",
                                                                    makePalette("#fbf7ef", "#493728", "#a66a3f",
                                                                                "#fffdf8", "#4f3929", "#d9c3aa",
                                                                                "#a66a3f", "#895431", "#ffffff",
                                                                                "#f3eadc", "#493728", "#ead8c1", "#35251a",
                                                                                "#e5d1b7", "1px solid #d9c3aa")},
                                                                   {"Light: Mint & Navy",
                                                                    makePalette("#f3fbf8", "#173a3c", "#2b8c7f",
                                                                                "#ffffff", "#173a3c", "#addbd2",
                                                                                "#1d5962", "#287681", "#ffffff",
                                                                                "#e5f6f1", "#173a3c", "#c8eee5", "#102f34",
                                                                                "#bde5dc", "1px solid #addbd2")},
                                                                   {"Light: High Contrast",
                                                                    makePalette("#ffffff", "#111111", "#005fcc",
                                                                                "#ffffff", "#000000", "#4d4d4d",
                                                                                "#111111", "#005fcc", "#ffffff",
                                                                                "#f0f0f0", "#000000", "#005fcc", "#ffffff",
                                                                                "#9dccff", "2px solid #111111")},
                                                                   {"Dark: Cyberpunk Neon",
                                                                    makePalette("#070513", "#f3e7ff", "#ff2bd6",
                                                                                "#100c24", "#5ffbf1", "#6e4cff",
                                                                                "#2a145c", "#ff2bd6", "#ffffff",
                                                                                "#0c081c", "#5ffbf1", "#381b72", "#ffffff",
                                                                                "#381b72", "2px solid #ff2bd6")},
                                                                   {"Dark: Dracula",
                                                                    makePalette("#282a36", "#f8f8f2", "#bd93f9",
                                                                                "#21222c", "#f8f8f2", "#6272a4",
                                                                                "#44475a", "#6272a4", "#f8f8f2",
                                                                                "#21222c", "#f8f8f2", "#44475a", "#f8f8f2",
                                                                                "#44475a", "1px solid #6272a4")},
                                                                   {"Dark: Nord Frost",
                                                                    makePalette("#2e3440", "#eceff4", "#88c0d0",
                                                                                "#3b4252", "#eceff4", "#4c566a",
                                                                                "#4c566a", "#5e81ac", "#eceff4",
                                                                                "#242933", "#d8dee9", "#434c5e", "#eceff4",
                                                                                "#434c5e", "1px solid #88c0d0")},
                                                                   {"Dark: Solarized",
                                                                    makePalette("#002b36", "#93a1a1", "#b58900",
                                                                                "#073642", "#eee8d5", "#586e75",
                                                                                "#07576b", "#268bd2", "#fdf6e3",
                                                                                "#00242d", "#93a1a1", "#07576b", "#fdf6e3",
                                                                                "#07576b", "1px solid #586e75")},
                                                                   {"Dark: Graphite Orange",
                                                                    makePalette("#171717", "#f2f2f2", "#ff8a3d",
                                                                                "#242424", "#f7f7f7", "#5f5f5f",
                                                                                "#3a3a3a", "#ff8a3d", "#ffffff",
                                                                                "#202020", "#f2f2f2", "#59311c", "#ffffff",
                                                                                "#59311c", "2px solid #ff8a3d")}}};

    if (styleSheetAction) {
        QSignalBlocker blocker(styleSheetAction);
        styleSheetAction->setChecked(currentlyEnabled);
    }

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Custom Style Editor"));
    dialog.resize(900, 640);
    // Keep the editor dialog on the application stylesheet so Apply updates it live.
    dialog.setStyleSheet("");

    auto *layout = new QVBoxLayout(&dialog);
    auto *topRow = new QHBoxLayout();
    auto *enableCheck = new QCheckBox(tr("Use custom style"), &dialog);
    enableCheck->setChecked(currentlyEnabled);
    auto *presetLabel = new QLabel(tr("Preset:"), &dialog);
    auto *presetCombo = new QComboBox(&dialog);
    for (const auto &preset : presetStyles) {
        presetCombo->addItem(preset.first);
    }
    int presetIndex = 0;
    for (int i = 0; i < static_cast<int>(presetStyles.size()); ++i) {
        if (presetStyles[static_cast<std::size_t>(i)].first == lastPresetName) {
            presetIndex = i;
            break;
        }
    }
    presetCombo->setCurrentIndex(presetIndex);

    auto *editor = new QPlainTextEdit(&dialog);
    editor->setPlainText(customStyleSheet);
    editor->setLineWrapMode(QPlainTextEdit::NoWrap);
    editor->setPlaceholderText(tr("Enter a Qt stylesheet (QSS) for ACMX2 interface..."));
    {
        QFont qssFont("Courier New");
        qssFont.setStyleHint(QFont::Monospace);
        qssFont.setPointSize(10);
        editor->setFont(qssFont);
    }

    auto *buttonBox = new QDialogButtonBox(&dialog);
    auto *applyButton = buttonBox->addButton(tr("Apply"), QDialogButtonBox::ApplyRole);
    auto *saveButton = buttonBox->addButton(tr("Save"), QDialogButtonBox::ActionRole);
    auto *closeButton = buttonBox->addButton(QDialogButtonBox::Close);

    topRow->addWidget(enableCheck);
    topRow->addSpacing(12);
    topRow->addWidget(presetLabel);
    topRow->addWidget(presetCombo, 1);
    layout->addLayout(topRow);
    layout->addWidget(editor, 1);
    layout->addWidget(buttonBox);

    connect(presetCombo, &QComboBox::currentTextChanged, &dialog,
            [editor, &presetStyles, &appSettings](const QString &name) {
                for (const auto &preset : presetStyles) {
                    if (preset.first == name) {
                        editor->setPlainText(preset.second);
                        appSettings.setValue("customStylePreset", name);
                        break;
                    }
                }
            });

    auto applyEditorStyle = [this, &dialog, enableCheck, editor, presetCombo]() {
        customStyleSheet = editor->toPlainText();
        QSettings styleSettings("LostSideDead");
        styleSettings.setValue("customStyleSheet", customStyleSheet);
        styleSettings.setValue("customStylePreset", presetCombo->currentText());
        styleSettings.setValue("useCustomStyle", enableCheck->isChecked());
        applyCustomStyleSheet(enableCheck->isChecked());
        // Ensure no local override remains so the dialog always follows qApp style.
        dialog.setStyleSheet("");
        if (styleSheetAction) {
            QSignalBlocker blocker(styleSheetAction);
            styleSheetAction->setChecked(enableCheck->isChecked());
        }
    };

    connect(applyButton, &QPushButton::clicked, &dialog, applyEditorStyle);
    connect(saveButton, &QPushButton::clicked, &dialog, applyEditorStyle);
    connect(closeButton, &QPushButton::clicked, &dialog, &QDialog::accept);

    dialog.exec();
}

void MainWindow::newList() {
    LibraryWindow library(active_backend, this);

    if (library.exec() == QDialog::Accepted) {
        loadLibraryPath(library.getShaderPath());
    }
}

void MainWindow::menuLibraryBuilder() {
    if (libraryBuilderDialog &&
        libraryBuilderDialog->selectedBackend() != active_backend) {
        libraryBuilderDialog->close();
        libraryBuilderDialog = nullptr;
    }
    if (libraryBuilderDialog) {
        libraryBuilderDialog->show();
        libraryBuilderDialog->raise();
        libraryBuilderDialog->activateWindow();
        return;
    }

    libraryBuilderDialog = new LibraryBuilderDialog(active_backend, this);
    libraryBuilderDialog->setAttribute(Qt::WA_DeleteOnClose);
    connect(libraryBuilderDialog, &LibraryBuilderDialog::libraryExported, this,
            [this](const QString &directory) {
                if (loadLibraryPath(directory))
                    Log(tr("Loaded exported shader library: %1").arg(shader_path));
            });
    libraryBuilderDialog->show();
    libraryBuilderDialog->raise();
    libraryBuilderDialog->activateWindow();
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
    if (shader_path.isEmpty() || !acmx2::shader_manifest_exists(shader_path)) {
        QMessageBox::information(this, tr("New Shader File"),
                                 tr("Create or load a shader library first."));
        return;
    }
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString typeError;
        const auto libraryType =
            acmx2::shader_manifest_library_type(shader_path, typeError);
        if (!typeError.isEmpty()) {
            QMessageBox::warning(this, tr("New Shader File"), typeError);
            return;
        }
        if (libraryType &&
            *libraryType == acmx2::ShaderLibraryType::Runtime) {
            QMessageBox::information(
                this, tr("New Shader File"),
                tr("New ACMXVK shaders must be added to a source library, not "
                   "a compiled SPIR-V runtime library."));
            return;
        }
    }
    ShaderDialog new_shader(active_backend, this);
    new_shader.setShaderPath(shader_path);
    if (new_shader.exec() == QDialog::Accepted) {
        QSettings appSettings("LostSideDead");
        appSettings.setValue(
            acmx2::backend_settings_key(active_backend, "library"), shader_path);
        if (active_backend == acmx2::Backend::Acmx2)
            appSettings.setValue("shaders", shader_path);
        appSettings.sync();
        loadShaders(shader_path, true);
    }
}

void MainWindow::menuRemove() {
    int row = currentShaderRow();
    if (row < 0 || row >= items.size())
        return;
    const QString shaderName = items.at(row);
    QString manifestError;
    if (!acmx2::remove_shader_manifest_entry(shader_path, shaderName,
                                             manifestError)) {
        QMessageBox::warning(this, tr("Could Not Remove Shader"),
                             manifestError);
        Log(tr("Could not remove %1 from the library manifest: %2")
                .arg(shaderName, manifestError));
        return;
    }
    items.removeAt(row);
    populateShaderTree();
    indexTimestamp = acmx2::shader_manifest_last_modified(shader_path);
    activeShaderManifestPath = acmx2::shader_manifest_path(shader_path);
    Log(tr("Removed shader from library manifest: %1").arg(shaderName));
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
}

void MainWindow::updateIndex() {
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
            writtenItems.append(shaderName);
        } else {
            Log("Warning: File no longer exists, removing from list: " + shaderName);
        }
    }
    QString manifestError;
    QStringList existingItems;
    if (acmx2::load_shader_manifest(shader_path, existingItems,
                                    manifestError) &&
        existingItems == writtenItems) {
        indexTimestamp = acmx2::shader_manifest_last_modified(shader_path);
        activeShaderManifestPath =
            acmx2::shader_manifest_path(shader_path);
        return;
    }
    manifestError.clear();
    if (!acmx2::write_shader_manifest(shader_path, writtenItems, manifestError)) {
        Log("Failed to update shader manifest: " + manifestError);
        return;
    }
    indexTimestamp = acmx2::shader_manifest_last_modified(shader_path);
    activeShaderManifestPath = acmx2::shader_manifest_path(shader_path);

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
    QString filePath = shader_path + "/" + itemText;
    openShaderEditor(filePath);
}

void MainWindow::openShaderEditor(const QString &filePath, int lineNumber,
                                  int columnNumber, int matchLength) {
    const QFileInfo requestedFile(filePath);
    if (!requestedFile.exists() || !requestedFile.isFile()) {
        QMessageBox::warning(this, tr("Open Shader"),
                             tr("Shader file no longer exists:\n%1").arg(filePath));
        return;
    }

    cleanupClosedEditors();
    const QString canonicalPath = requestedFile.canonicalFilePath();
    for (const QPointer<TextEditor> &openEditor : open_files) {
        if (!openEditor)
            continue;
        const QString openPath = QFileInfo(openEditor->fileName()).canonicalFilePath();
        if (!canonicalPath.isEmpty() && openPath == canonicalPath) {
            openEditor->show();
            openEditor->raise();
            openEditor->activateWindow();
            openEditor->revealLocation(lineNumber, columnNumber, matchLength);
            return;
        }
    }

    TextEditor *editor = new TextEditor(this);
    editor->setText(readFileContents(filePath));
    editor->setFileName(filePath);
    connect(editor, &TextEditor::fileSaved, this, [this](const QString &filePath) {
        handleSavedShader(filePath);
    });
    open_files.append(editor);
    editor->show();
    editor->revealLocation(lineNumber, columnNumber, matchLength);
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
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
#if defined(__linux__) || defined(__APPLE__)
    // A named semaphore can be unlinked while an existing process still owns
    // a usable handle. Verify that new child processes can still discover the
    // name before every launch and recreate it when necessary.
    if (shaderSelectionSemaphore != SEM_FAILED) {
        sem_t *publishedSemaphore = ::sem_open(
            acmx2::ipc::kShaderSelectionSemaphoreName, 0);
        if (publishedSemaphore != SEM_FAILED) {
            ::sem_close(publishedSemaphore);
        } else {
            ::sem_close(shaderSelectionSemaphore);
            shaderSelectionSemaphore = SEM_FAILED;
        }
    }
    if (shaderSelectionSemaphore == SEM_FAILED) {
        shaderSelectionSemaphore = ::sem_open(
            acmx2::ipc::kShaderSelectionSemaphoreName, O_CREAT, 0666, 1);
    }
    if (shaderSelectionSemaphore == SEM_FAILED) {
        Log(tr("Shared interface control unavailable: sem_open(%1) failed: %2")
                .arg(acmx2::ipc::kShaderSelectionSemaphoreName,
                     QString::fromLocal8Bit(std::strerror(errno))));
        return;
    }

    if (shaderSelectionShm)
        return;

    shaderSelectionShmFd = ::shm_open(acmx2::ipc::kShaderSelectionShmName,
                                      O_CREAT | O_RDWR,
                                      0666);
    if (shaderSelectionShmFd < 0) {
        const int openError = errno;
        Log(tr("Shared interface control unavailable: shm_open(%1) failed: "
               "%2")
                .arg(acmx2::ipc::kShaderSelectionShmName,
                     QString::fromLocal8Bit(std::strerror(openError))));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    constexpr std::size_t SHARED_MEMORY_SIZE =
        sizeof(acmx2::ipc::ShaderSelectionShmData);
    struct stat shmStat{};
    if (::fstat(shaderSelectionShmFd, &shmStat) != 0) {
        const int statError = errno;
        Log(tr("Shared interface control unavailable: fstat(%1) failed: %2")
                .arg(acmx2::ipc::kShaderSelectionShmName,
                     QString::fromLocal8Bit(std::strerror(statError))));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    if (shmStat.st_size == 0) {
        if (::ftruncate(shaderSelectionShmFd,
                        static_cast<off_t>(SHARED_MEMORY_SIZE)) != 0) {
            const int truncateError = errno;
            Log(tr("Shared interface control unavailable: ftruncate(%1, %2) "
                   "failed: %3")
                    .arg(acmx2::ipc::kShaderSelectionShmName)
                    .arg(static_cast<qulonglong>(SHARED_MEMORY_SIZE))
                    .arg(QString::fromLocal8Bit(
                        std::strerror(truncateError))));
            cleanupShaderSelectionSharedMemory();
            return;
        }
    } else if (shmStat.st_size < static_cast<off_t>(SHARED_MEMORY_SIZE)) {
        Log(tr("Shared interface control unavailable: %1 has size %2 bytes; "
               "expected %3. Refusing to resize an active or stale shared "
               "memory object.")
                .arg(acmx2::ipc::kShaderSelectionShmName)
                .arg(static_cast<qlonglong>(shmStat.st_size))
                .arg(static_cast<qulonglong>(SHARED_MEMORY_SIZE)));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    void *mapped = ::mmap(nullptr,
                          SHARED_MEMORY_SIZE,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED,
                          shaderSelectionShmFd,
                          0);
    if (mapped == MAP_FAILED) {
        const int mapError = errno;
        Log(tr("Shared interface control unavailable: mmap(%1, %2) failed: "
               "%3")
                .arg(acmx2::ipc::kShaderSelectionShmName)
                .arg(static_cast<qulonglong>(SHARED_MEMORY_SIZE))
                .arg(QString::fromLocal8Bit(std::strerror(mapError))));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    shaderSelectionShm = static_cast<acmx2::ipc::ShaderSelectionShmData *>(mapped);
#else
    if (shaderSelectionSemaphore == nullptr) {
        shaderSelectionSemaphore = ::CreateMutexW(
            nullptr, FALSE, acmx2::ipc::kShaderSelectionMutexNameWindows);
    }
    if (shaderSelectionSemaphore == nullptr) {
        Log(tr("Shared interface control unavailable: CreateMutexW failed "
               "with Windows error %1")
                .arg(static_cast<qulonglong>(::GetLastError())));
        return;
    }

    if (shaderSelectionShm)
        return;

    constexpr std::size_t SHARED_MEMORY_SIZE =
        sizeof(acmx2::ipc::ShaderSelectionShmData);
    shaderSelectionMapping = ::CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        static_cast<DWORD>(SHARED_MEMORY_SIZE),
        acmx2::ipc::kShaderSelectionMappingNameWindows);
    if (shaderSelectionMapping == nullptr) {
        Log(tr("Shared interface control unavailable: CreateFileMappingW "
               "failed with Windows error %1")
                .arg(static_cast<qulonglong>(::GetLastError())));
        cleanupShaderSelectionSharedMemory();
        return;
    }

    void *mapped = ::MapViewOfFile(shaderSelectionMapping, FILE_MAP_ALL_ACCESS,
                                   0, 0, SHARED_MEMORY_SIZE);
    if (mapped == nullptr) {
        Log(tr("Shared interface control unavailable: MapViewOfFile failed "
               "with Windows error %1")
                .arg(static_cast<qulonglong>(::GetLastError())));
        cleanupShaderSelectionSharedMemory();
        return;
    }
    shaderSelectionShm =
        static_cast<acmx2::ipc::ShaderSelectionShmData *>(mapped);
#endif

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
#if defined(__linux__) || defined(__APPLE__)
        Log(tr("Shared interface control unavailable: could not lock %1: %2")
                .arg(acmx2::ipc::kShaderSelectionSemaphoreName,
                     QString::fromLocal8Bit(std::strerror(errno))));
#else
        Log(tr("Shared interface control unavailable: could not lock the "
               "Windows control mutex (error %1)")
                .arg(static_cast<qulonglong>(::GetLastError())));
#endif
        cleanupShaderSelectionSharedMemory();
        return;
    }

    if (shaderSelectionShm->magic != acmx2::ipc::kShaderSelectionMagic ||
        shaderSelectionShm->version != acmx2::ipc::kShaderSelectionVersion) {
        shaderSelectionShm->magic = acmx2::ipc::kShaderSelectionMagic;
        shaderSelectionShm->version = acmx2::ipc::kShaderSelectionVersion;
        shaderSelectionShm->selected_index = -1;
        shaderSelectionShm->shader_pass_count = 0;
        shaderSelectionShm->shader_pass_enabled = 0;
        shaderSelectionShm->repeat_enabled = 0;
        shaderSelectionShm->display_filter_enabled = 0;
        shaderSelectionShm->watermark_enabled = 0;
        shaderSelectionShm->normalized_time_enabled = 0;
        std::fill(std::begin(shaderSelectionShm->reserved_flags),
                  std::end(shaderSelectionShm->reserved_flags), 0);
        std::fill(std::begin(shaderSelectionShm->shader_pass_indices), std::end(shaderSelectionShm->shader_pass_indices), -1);
        std::fill(&shaderSelectionShm->shader_pass_names[0][0],
                  &shaderSelectionShm->shader_pass_names[0][0] +
                      acmx2::ipc::kShaderSelectionMaxPassCount *
                          acmx2::ipc::kShaderSelectionMaxShaderName,
                  '\0');
        shaderSelectionShm->gpu_filter_count = 0;
        shaderSelectionShm->gpu_filter_enabled = 0;
        shaderSelectionShm->gpu_buffer_size = 8;
        shaderSelectionShm->watermark_r = 255;
        shaderSelectionShm->watermark_g = 0;
        shaderSelectionShm->watermark_b = 150;
        std::fill(std::begin(shaderSelectionShm->reserved), std::end(shaderSelectionShm->reserved), 0);
        std::fill(std::begin(shaderSelectionShm->gpu_filter_indices), std::end(shaderSelectionShm->gpu_filter_indices), -1);
        std::fill(std::begin(shaderSelectionShm->watermark_text), std::end(shaderSelectionShm->watermark_text), '\0');
        shaderSelectionShm->reload_shader_index = -1;
        std::fill(std::begin(shaderSelectionShm->reload_shader_path), std::end(shaderSelectionShm->reload_shader_path), '\0');
        shaderSelectionShm->reload_sequence = 0;
        shaderSelectionShm->custom_uniform_count = 0;
        std::fill(&shaderSelectionShm->custom_uniform_names[0][0],
                  &shaderSelectionShm->custom_uniform_names[0][0] +
                      acmx2::ipc::kShaderSelectionMaxCustomUniforms *
                          acmx2::ipc::kShaderSelectionMaxUniformName,
                  '\0');
        std::fill(std::begin(shaderSelectionShm->custom_uniform_values),
                  std::end(shaderSelectionShm->custom_uniform_values), 0.0f);
        std::fill(std::begin(shaderSelectionShm->audio_file_path),
                  std::end(shaderSelectionShm->audio_file_path), '\0');
        shaderSelectionShm->audio_output_device = -1;
        shaderSelectionShm->audio_pass_through = 0;
        shaderSelectionShm->audio_trunc = 0;
        shaderSelectionShm->audio_repeat = 0;
        shaderSelectionShm->audio_reserved = 0;
        shaderSelectionShm->audio_file_sequence = 0;
        std::fill(std::begin(shaderSelectionShm->selected_shader_name),
                  std::end(shaderSelectionShm->selected_shader_name), '\0');
        shaderSelectionShm->sequence = 0;
    }
#endif
}

void MainWindow::publishSelectedShaderIndexToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;
    const int row = currentShaderRow();
    if (row < 0 || row >= items.size())
        return;
    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->selected_index = row;
    const QByteArray shaderName = items.at(row).toUtf8();
    const qsizetype copyLength = std::min<qsizetype>(
        shaderName.size(),
        static_cast<qsizetype>(acmx2::ipc::kShaderSelectionMaxShaderName - 1));
    std::fill(std::begin(shaderSelectionShm->selected_shader_name),
              std::end(shaderSelectionShm->selected_shader_name), '\0');
    std::copy_n(shaderName.constData(), copyLength,
                shaderSelectionShm->selected_shader_name);
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishShaderReloadToRunningProcess(const QString &filePath) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend != acmx2::Backend::Acmx2 || !shaderSelectionShm || !process ||
        process->state() != QProcess::Running || cacheBuildInProgress) {
        return;
    }

    const QFileInfo savedFile(filePath);
    const QString shaderName = QDir(shader_path).relativeFilePath(savedFile.absoluteFilePath());
    const int shaderIndex = items.indexOf(shaderName, 0, Qt::CaseInsensitive);
    if (shaderIndex < 0) {
        Log("Saved shader is not in the active library; live reload was skipped: " + filePath);
        return;
    }

    const QByteArray reloadPath = savedFile.canonicalFilePath().toUtf8();
    if (reloadPath.isEmpty() ||
        reloadPath.size() >= static_cast<int>(acmx2::ipc::kShaderSelectionMaxReloadPath)) {
        Log("Shader path is too long for live reload: " + filePath);
        return;
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->reload_shader_index = shaderIndex;
    std::fill(std::begin(shaderSelectionShm->reload_shader_path), std::end(shaderSelectionShm->reload_shader_path), '\0');
    std::copy(reloadPath.cbegin(), reloadPath.cend(), shaderSelectionShm->reload_shader_path);
    ++shaderSelectionShm->reload_sequence;
    ++shaderSelectionShm->sequence;
    Log("Requested live shader reload: " + shaderName + "<br>");
#else
    Q_UNUSED(filePath);
#endif
}

void MainWindow::handleSavedShader(const QString &filePath) {
    populateShaderTree();
    if (active_backend == acmx2::Backend::Acmxvk && process &&
        process->state() == QProcess::Running) {
        queueAcmxvkLiveCompile(filePath);
        return;
    }
    publishShaderReloadToRunningProcess(filePath);
}

void MainWindow::queueAcmxvkLiveCompile(const QString &filePath) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    QString typeError;
    if (!is_acmxvk_source_library(shader_path, typeError)) {
        Log(typeError.isEmpty()
                ? tr("ACMXVK live compile requires a source library.")
                : typeError);
        return;
    }

    const QFileInfo sourceInfo(filePath);
    const QString sourcePath = sourceInfo.canonicalFilePath();
    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    if (sourcePath.isEmpty() || sourceRoot.isEmpty()) {
        Log(tr("Could not resolve saved ACMXVK shader: %1").arg(filePath));
        return;
    }
    const QString sourceName =
        sanitizeShaderName(QDir(sourceRoot).relativeFilePath(sourcePath));
    if (sourceName.isEmpty() ||
        (!sourceName.endsWith(QStringLiteral(".frag"), Qt::CaseInsensitive) &&
         !sourceName.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive)) ||
        !items.contains(sourceName, Qt::CaseInsensitive)) {
        Log(tr("Saved file is not a fragment or compute source in the active "
               "ACMXVK library: %1")
                .arg(filePath));
        return;
    }

    liveShaderCompileQueue.removeAll(sourcePath);
    liveShaderCompileQueue.append(sourcePath);
    startNextAcmxvkLiveCompile();
#else
    Q_UNUSED(filePath);
#endif
}

void MainWindow::startNextAcmxvkLiveCompile() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (liveShaderCompileProcess &&
        liveShaderCompileProcess->state() != QProcess::NotRunning) {
        return;
    }
    if (liveShaderCompileQueue.isEmpty()) {
        return;
    }

    if (!liveShaderCompileProcess) {
        liveShaderCompileProcess = new QProcess(this);
        liveShaderCompileProcess->setProcessChannelMode(
            QProcess::SeparateChannels);
        connect(liveShaderCompileProcess, &QProcess::readyReadStandardOutput,
                this, [this]() {
                    QString output = QString::fromUtf8(
                        liveShaderCompileProcess->readAllStandardOutput());
                    liveShaderCompileStdout += output;
                    if (liveShaderCompileStdout.size() > 262144)
                        liveShaderCompileStdout =
                            liveShaderCompileStdout.right(262144);
                    Write(output.toHtmlEscaped().replace(
                        '\n', QStringLiteral("<br>")));
                });
        connect(liveShaderCompileProcess, &QProcess::readyReadStandardError,
                this, [this]() {
                    QString output = QString::fromUtf8(
                        liveShaderCompileProcess->readAllStandardError());
                    liveShaderCompileStderr += output;
                    if (liveShaderCompileStderr.size() > 262144)
                        liveShaderCompileStderr =
                            liveShaderCompileStderr.right(262144);
                    Write(QStringLiteral("<b style='color:red;'>") +
                          output.toHtmlEscaped().replace(
                              '\n', QStringLiteral("<br>")) +
                          QStringLiteral("</b>"));
                });
        connect(
            liveShaderCompileProcess,
            static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(
                &QProcess::finished),
            this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
                liveShaderCompileStdout += QString::fromUtf8(
                    liveShaderCompileProcess->readAllStandardOutput());
                liveShaderCompileStderr += QString::fromUtf8(
                    liveShaderCompileProcess->readAllStandardError());
                bool installed = false;
                if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                    QFile compiled(liveShaderCompileTemporary);
                    quint32 magic = 0;
                    if (compiled.open(QIODevice::ReadOnly)) {
                        QDataStream stream(&compiled);
                        stream.setByteOrder(QDataStream::LittleEndian);
                        stream >> magic;
                    }
                    constexpr quint32 SPIRV_MAGIC = 0x07230203U;
                    if (magic != SPIRV_MAGIC) {
                        Log(tr("<b style='color:red;'>Live ACMXVK compile did "
                               "not produce valid SPIR-V for %1.</b>")
                                .arg(liveShaderCompileSource));
                    } else if (QFileInfo(liveShaderCompileOutput).isSymLink()) {
                        Log(tr("<b style='color:red;'>Refusing to replace "
                               "symbolic-link shader output: %1</b>")
                                .arg(liveShaderCompileOutput));
                    } else {
                        std::error_code error;
                        replace_file(
                            std::filesystem::u8path(
                                liveShaderCompileTemporary.toUtf8().constData()),
                            std::filesystem::u8path(
                                liveShaderCompileOutput.toUtf8().constData()),
                            error);
                        if (error) {
                            Log(tr("<b style='color:red;'>Could not install "
                                   "live ACMXVK shader: %1</b>")
                                    .arg(QString::fromStdString(
                                        error.message())));
                        } else {
                            installed = true;
                        }
                    }
                } else {
                    Log(tr("<b style='color:red;'>Live ACMXVK compile failed "
                           "for %1 (%2, exit code %3).</b>")
                            .arg(liveShaderCompileSource)
                            .arg(exitStatus == QProcess::CrashExit
                                     ? tr("compiler crashed")
                                     : tr("compiler error"))
                            .arg(exitCode));
                    QString diagnostics =
                        liveShaderCompileStderr.trimmed();
                    if (diagnostics.isEmpty())
                        diagnostics = liveShaderCompileStdout.trimmed();
                    if (diagnostics.isEmpty())
                        diagnostics = liveShaderCompileProcess->errorString();
                    Log(tr("<b style='color:red;'>Compiler message:</b>"
                           "<pre style='white-space:pre-wrap;'>%1</pre>")
                            .arg(diagnostics.toHtmlEscaped()));
                }

                if (!installed) {
                    QFile::remove(liveShaderCompileTemporary);
                } else {
                    Log(tr("Compiled and installed ACMXVK shader: %1")
                            .arg(liveShaderCompileOutput));
                    populateShaderTree();
                    publishAcmxvkCompiledShaderReload(
                        liveShaderCompileSource, liveShaderCompileOutput);
                }

                liveShaderCompileSource.clear();
                liveShaderCompileOutput.clear();
                liveShaderCompileTemporary.clear();
                liveShaderCompileStdout.clear();
                liveShaderCompileStderr.clear();
                QTimer::singleShot(
                    0, this, &MainWindow::startNextAcmxvkLiveCompile);
            });
    }

    QString compilerError;
    const QString glslc = resolve_acmxvk_shader_compiler(compilerError);
    if (glslc.isEmpty()) {
        Log(tr("<b style='color:red;'>Cannot live compile ACMXVK shader: "
               "%1</b>")
                .arg(compilerError.toHtmlEscaped()));
        liveShaderCompileQueue.clear();
        return;
    }

    liveShaderCompileSource = liveShaderCompileQueue.takeFirst();
    const QString sourceRoot = QFileInfo(shader_path).canonicalFilePath();
    const QString sourceName =
        QDir(sourceRoot).relativeFilePath(liveShaderCompileSource);
    liveShaderCompileOutput =
        QDir(acmxvk_build_directory(sourceRoot))
            .filePath(acmxvk_runtime_shader_name(sourceName));
    if (!QDir().mkpath(QFileInfo(liveShaderCompileOutput).absolutePath())) {
        Log(tr("<b style='color:red;'>Could not create live shader output "
               "directory for %1.</b>")
                .arg(liveShaderCompileOutput));
        liveShaderCompileSource.clear();
        liveShaderCompileOutput.clear();
        QTimer::singleShot(0, this,
                           &MainWindow::startNextAcmxvkLiveCompile);
        return;
    }

    liveShaderCompileTemporary =
        liveShaderCompileOutput + QStringLiteral(".live-tmp-%1-%2")
                                      .arg(QCoreApplication::applicationPid())
                                      .arg(++liveShaderCompileSequence);
    const QStringList arguments{
        QStringLiteral("-I"), sourceRoot, liveShaderCompileSource,
        QStringLiteral("-o"), liveShaderCompileTemporary};
    Log(tr("Live compiling ACMXVK shader: %1").arg(sourceName));
    Log(tr("Command: %1 %2<br>")
            .arg(glslc, concatList(arguments)));
    liveShaderCompileStdout.clear();
    liveShaderCompileStderr.clear();
    liveShaderCompileProcess->start(glslc, arguments);
    if (!liveShaderCompileProcess->waitForStarted()) {
        Log(tr("<b style='color:red;'>Failed to start the ACMXVK shader "
               "compiler: %1</b>")
                .arg(liveShaderCompileProcess->errorString().toHtmlEscaped()));
        QFile::remove(liveShaderCompileTemporary);
        liveShaderCompileSource.clear();
        liveShaderCompileOutput.clear();
        liveShaderCompileTemporary.clear();
        liveShaderCompileStdout.clear();
        liveShaderCompileStderr.clear();
        QTimer::singleShot(0, this,
                           &MainWindow::startNextAcmxvkLiveCompile);
    }
#endif
}

void MainWindow::publishAcmxvkCompiledShaderReload(
    const QString &sourcePath, const QString &runtimePath) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (active_backend != acmx2::Backend::Acmxvk || !shaderSelectionShm ||
        !process || process->state() != QProcess::Running) {
        return;
    }

    const QString sourceName = QDir(shader_path).relativeFilePath(sourcePath);
    const int shaderIndex =
        items.indexOf(sourceName, 0, Qt::CaseInsensitive);
    const QByteArray reloadPath =
        QFileInfo(runtimePath).canonicalFilePath().toUtf8();
    if (shaderIndex < 0 || reloadPath.isEmpty() ||
        reloadPath.size() >=
            static_cast<int>(acmx2::ipc::kShaderSelectionMaxReloadPath)) {
        Log(tr("Compiled shader cannot be published for live reload: %1")
                .arg(runtimePath));
        return;
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log(tr("Could not lock the live shader reload channel."));
        return;
    }
    shaderSelectionShm->reload_shader_index = shaderIndex;
    std::fill(std::begin(shaderSelectionShm->reload_shader_path),
              std::end(shaderSelectionShm->reload_shader_path), '\0');
    std::copy(reloadPath.cbegin(), reloadPath.cend(),
              shaderSelectionShm->reload_shader_path);
    ++shaderSelectionShm->reload_sequence;
    ++shaderSelectionShm->sequence;
    Log(tr("Requested live ACMXVK pipeline reload: %1<br>")
            .arg(sourceName));
#else
    Q_UNUSED(sourcePath);
    Q_UNUSED(runtimePath);
#endif
}

void MainWindow::publishMultipassShadersToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;

    std::array<qint32, acmx2::ipc::kShaderSelectionMaxPassCount> passIndices;
    passIndices.fill(-1);
    std::array<std::array<char, acmx2::ipc::kShaderSelectionMaxShaderName>,
               acmx2::ipc::kShaderSelectionMaxPassCount>
        passNames{};

    quint32 passCount = 0;
    if (shader_pass_enabled && !shader_pass_names.isEmpty()) {
        loadShaders(shader_path, true);
        for (const QString &name : shader_pass_names) {
            if (passCount >= acmx2::ipc::kShaderSelectionMaxPassCount)
                break;
            const int idx = items.indexOf(name);
            if (idx < 0)
                continue;
            passIndices[passCount] = idx;
            const QByteArray shaderName = name.toUtf8();
            const qsizetype copyLength = std::min<qsizetype>(
                shaderName.size(),
                static_cast<qsizetype>(acmx2::ipc::kShaderSelectionMaxShaderName - 1));
            std::copy_n(shaderName.constData(), copyLength,
                        passNames[passCount].begin());
            ++passCount;
        }
    }

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->shader_pass_enabled = (shader_pass_enabled && passCount > 0) ? 1 : 0;
    shaderSelectionShm->shader_pass_count = passCount;
    std::copy(passIndices.begin(), passIndices.end(), std::begin(shaderSelectionShm->shader_pass_indices));
    for (std::size_t i = 0; i < passNames.size(); ++i) {
        std::copy(passNames[i].begin(), passNames[i].end(),
                  shaderSelectionShm->shader_pass_names[i]);
    }
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishRepeatStateToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;
    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->repeat_enabled = (play_repeat && play_repeat->isChecked()) ? 1 : 0;
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishRuntimeSettingsToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm)
        return;

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    shaderSelectionShm->display_filter_enabled = display_filter_enabled ? 1 : 0;
    shaderSelectionShm->normalized_time_enabled = normalized_time ? 1 : 0;

    const bool watermarkActive = watermark_enabled && !watermark_text.isEmpty();
    shaderSelectionShm->watermark_enabled = watermarkActive ? 1 : 0;
    shaderSelectionShm->watermark_r = static_cast<uint8_t>(std::clamp(watermark_r, 0, 255));
    shaderSelectionShm->watermark_g = static_cast<uint8_t>(std::clamp(watermark_g, 0, 255));
    shaderSelectionShm->watermark_b = static_cast<uint8_t>(std::clamp(watermark_b, 0, 255));
    std::fill(std::begin(shaderSelectionShm->watermark_text), std::end(shaderSelectionShm->watermark_text), '\0');
    const QByteArray wmUtf8 = watermark_text.toUtf8();
    const std::size_t wmCap = static_cast<std::size_t>(acmx2::ipc::kShaderSelectionMaxWatermarkText - 1);
    const std::size_t wmLen = std::min<std::size_t>(wmCap, static_cast<std::size_t>(wmUtf8.size()));
    std::copy_n(wmUtf8.constData(), static_cast<int>(wmLen), shaderSelectionShm->watermark_text);

    std::array<qint32, acmx2::ipc::kShaderSelectionMaxGpuFilterCount> gpuIndices;
    gpuIndices.fill(-1);
    quint32 gpuCount = 0;
    if (cuda_available && gpu_filter_enabled && !gpu_filter_indices.isEmpty()) {
        const QStringList parts = gpu_filter_indices.split(',', Qt::SkipEmptyParts);
        for (const QString &part : parts) {
            if (gpuCount >= acmx2::ipc::kShaderSelectionMaxGpuFilterCount)
                break;
            bool ok = false;
            const int idx = part.trimmed().toInt(&ok);
            if (!ok || idx < 0)
                continue;
            gpuIndices[gpuCount++] = idx;
        }
    }

    shaderSelectionShm->gpu_filter_enabled = (gpuCount > 0) ? 1 : 0;
    shaderSelectionShm->gpu_filter_count = gpuCount;
    shaderSelectionShm->gpu_buffer_size = static_cast<uint8_t>(std::clamp(gpu_buffer_size, 4, 32));
    std::copy(gpuIndices.begin(), gpuIndices.end(), std::begin(shaderSelectionShm->gpu_filter_indices));

    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::publishCustomUniformsToRunningProcess() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (!shaderSelectionShm || !customUniformDialog)
        return;

    acmx2::ipc::ShaderSelectionLock lock(shaderSelectionSemaphore);
    if (!lock) {
        Log("<br><style color=\"red\">Error lock failed</style><br>");
        return;
    }
    std::fill(&shaderSelectionShm->custom_uniform_names[0][0],
              &shaderSelectionShm->custom_uniform_names[0][0] +
                  acmx2::ipc::kShaderSelectionMaxCustomUniforms *
                      acmx2::ipc::kShaderSelectionMaxUniformName,
              '\0');
    std::fill(std::begin(shaderSelectionShm->custom_uniform_values),
              std::end(shaderSelectionShm->custom_uniform_values), 0.0f);

    quint32 count = 0;
    for (const acmx2::CustomUniformDefinition &uniform :
         customUniformDialog->uniforms()) {
        if (count >= acmx2::ipc::kShaderSelectionMaxCustomUniforms)
            break;
        const QByteArray name = uniform.name.toUtf8();
        if (name.isEmpty() ||
            name.size() >=
                static_cast<int>(acmx2::ipc::kShaderSelectionMaxUniformName)) {
            continue;
        }
        std::copy(name.cbegin(), name.cend(),
                  shaderSelectionShm->custom_uniform_names[count]);
        shaderSelectionShm->custom_uniform_values[count] =
            static_cast<float>(uniform.value);
        ++count;
    }
    shaderSelectionShm->custom_uniform_count = count;
    ++shaderSelectionShm->sequence;
#endif
}

void MainWindow::cleanupShaderSelectionSharedMemory() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    if (shaderSelectionShm) {
#if defined(__linux__) || defined(__APPLE__)
        ::munmap(shaderSelectionShm, sizeof(acmx2::ipc::ShaderSelectionShmData));
#else
        ::UnmapViewOfFile(shaderSelectionShm);
#endif
        shaderSelectionShm = nullptr;
    }
#if defined(__linux__) || defined(__APPLE__)
    if (shaderSelectionShmFd >= 0) {
        ::close(shaderSelectionShmFd);
        shaderSelectionShmFd = -1;
    }
#else
    if (shaderSelectionMapping != nullptr) {
        ::CloseHandle(shaderSelectionMapping);
        shaderSelectionMapping = nullptr;
    }
#endif
    cleanupShaderSelectionSemaphore();
#endif
}

void MainWindow::cleanupShaderSelectionSemaphore() {
#if defined(__linux__) || defined(__APPLE__)
    if (shaderSelectionSemaphore == SEM_FAILED)
        return;

    //::sem_unlink(acmx2::ipc::kShaderSelectionSemaphoreName);
    ::sem_close(shaderSelectionSemaphore);
    shaderSelectionSemaphore = SEM_FAILED;
#elif defined(_WIN32)
    if (shaderSelectionSemaphore == nullptr)
        return;

    ::CloseHandle(shaderSelectionSemaphore);
    shaderSelectionSemaphore = nullptr;
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
    if (active_backend != acmx2::Backend::Acmx2)
        return;
#ifdef Q_OS_MACOS
    // There is no persistent binary cache to inspect on macOS. Source saves
    // are handled by the live-reload IPC path instead.
    return;
#else
    if (shader_path.isEmpty())
        return;
    const QString cachePath = resolveShaderCachePath(
        shader_path, cache_size,
        cache_enabled && textureCacheArraySettingEnabled());
    QFileInfo cacheInfo(cachePath);
    if (!cacheInfo.exists() || !cacheInfo.isFile()) {
        Log("Shader cache not found at: " + cachePath);
        return;
    }
    shaderCacheMTime = cacheInfo.lastModified();
    shaderCacheStatus = parseShaderCacheStatus(cachePath);
    // Log("Shader cache: " + cachePath + " (" + QString::number(shaderCacheStatus.size()) + " entries)");
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

    QString acmxvk_type_error;
    const bool acmxvk_source =
        active_backend == acmx2::Backend::Acmxvk &&
        is_acmxvk_source_library(shader_path, acmxvk_type_error) &&
        acmxvk_type_error.isEmpty();
    const int width = QString::number(items.size()).size();
    for (int i = 0; i < items.size(); ++i) {
        const QString &name = items.at(i);
        QFileInfo fi(shader_path + "/" + name);
        const QString stem = QFileInfo(name).completeBaseName();
        const bool isCompute =
            name.endsWith(QStringLiteral(".comp"), Qt::CaseInsensitive) ||
            name.endsWith(QStringLiteral(".comp.spv"), Qt::CaseInsensitive);
        const QString shaderType = isCompute ? tr("Compute") : tr("Fragment");

        QString health;
        QColor healthColor;
        if (active_backend == acmx2::Backend::Acmxvk) {
            const AcmxvkBuildState state =
                acmxvk_source
                    ? acmxvk_shader_build_state(shader_path, name)
                    : AcmxvkBuildState::UpToDate;
            if (state == AcmxvkBuildState::NotBuilt) {
                health = tr("Not Built");
                healthColor = QColor("#888888");
            } else if (state == AcmxvkBuildState::Stale) {
                health = tr("Stale");
                healthColor = QColor("#ffaa00");
            } else {
                health = tr("Up to Date");
                healthColor = QColor("#55ff55");
            }
        } else if (shaderCacheStatus.isEmpty()) {
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
             << health
             << shaderType;
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
    QString normalized = message;
    while (normalized.endsWith('\n') || normalized.endsWith('\r')) {
        normalized.chop(1);
    }

    bottomTextBox->append(normalized);
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

void MainWindow::menuLoadLibrary() {
    QSettings settings("LostSideDead");
    QString startDirectory = settings.value("lastShaderDir").toString();
    if (startDirectory.isEmpty())
        startDirectory = shader_path;
    if (startDirectory.isEmpty())
        startDirectory = QDir::homePath();

    const QString directory = QFileDialog::getExistingDirectory(
        this, tr("Load Shader Library"), startDirectory,
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (directory.isEmpty())
        return;

    settings.setValue("lastShaderDir", directory);
    loadLibraryPath(directory);
}

bool MainWindow::backend_launch_available() const {
    return true;
}

void MainWindow::prompt_acmxvk_rebuild(
    const QString &reason, PendingAcmxvkAction resume_action) {
    QString type_error;
    if (!is_acmxvk_source_library(shader_path, type_error) ||
        !type_error.isEmpty()) {
        QMessageBox::warning(this, tr("Build ACMXVK Library"), reason);
        return;
    }

    const QMessageBox::StandardButton answer = QMessageBox::question(
        this, tr("Build ACMXVK Library"),
        tr("The ACMXVK build is out of date or incomplete.\n\n%1\n\n"
           "Do you wish to rebuild it now?")
            .arg(reason),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    if (answer != QMessageBox::Yes)
        return;

    pending_acmxvk_action = resume_action;
    Log(tr("ACMXVK rebuild requested before launch."));
    menuBuildShaderCache();
}

void MainWindow::update_backend_ui() {
    const QString name = acmx2::backend_name(active_backend);
    setWindowTitle(tr("%1 - Interface").arg(name));
    if (backendAcmx2Action)
        backendAcmx2Action->setChecked(active_backend == acmx2::Backend::Acmx2);
    if (backendAcmxvkAction)
        backendAcmxvkAction->setChecked(active_backend == acmx2::Backend::Acmxvk);

    const bool launchAvailable = backend_launch_available();
    const bool acmx2Tools = active_backend == acmx2::Backend::Acmx2;
    QString sourceTypeError;
    const bool acmxvkSource =
        active_backend == acmx2::Backend::Acmxvk && !shader_path.isEmpty() &&
        is_acmxvk_source_library(shader_path, sourceTypeError) &&
        sourceTypeError.isEmpty();
    if (runMenu_select)
        runMenu_select->setEnabled(launchAvailable);
    if (runMenu_all)
        runMenu_all->setEnabled(launchAvailable);
    if (runMenu_copyCommand)
        runMenu_copyCommand->setEnabled(launchAvailable);
    if (buildCacheAction) {
        buildCacheAction->setText(acmxvkSource ? tr("Build")
                                               : tr("Rebuild Shader Cache"));
        buildCacheAction->setVisible(acmx2Tools || acmxvkSource);
        buildCacheAction->setEnabled(acmx2Tools || acmxvkSource);
        buildCacheAction->setToolTip(
            acmxvkSource
                ? tr("Compile changed GLSL sources into %1")
                      .arg(acmxvk_build_directory(shader_path))
                : QString());
    }
    if (fixBuildAction) {
        fixBuildAction->setVisible(acmxvkSource);
        fixBuildAction->setEnabled(acmxvkSource);
        fixBuildAction->setToolTip(
            acmxvkSource
                ? tr("Build into %1 and omit shaders that fail to compile")
                      .arg(acmxvk_build_directory(shader_path))
                : QString());
    }
    if (cleanShaderCacheAction) {
        cleanShaderCacheAction->setVisible(acmx2Tools);
        cleanShaderCacheAction->setEnabled(acmx2Tools);
    }
    if (removeBrokenAction) {
        removeBrokenAction->setVisible(acmx2Tools || acmxvkSource);
        removeBrokenAction->setEnabled(acmx2Tools || acmxvkSource);
        removeBrokenAction->setToolTip(
            acmxvkSource
                ? tr("Permanently delete .frag and .comp sources that fail "
                     "the ACMXVK Fix Build")
                : QString());
    }
    if (runFromCacheAction) {
        runFromCacheAction->setVisible(acmx2Tools);
        runFromCacheAction->setEnabled(acmx2Tools);
    }
#ifdef Q_OS_MACOS
    if (buildCacheAction && acmx2Tools) {
        buildCacheAction->setVisible(false);
        buildCacheAction->setEnabled(false);
    }
    if (cleanShaderCacheAction)
        cleanShaderCacheAction->setEnabled(false);
    if (runFromCacheAction)
        runFromCacheAction->setEnabled(false);
#endif
    const bool processIdle =
        !process || process->state() == QProcess::NotRunning;
    if (libraryBuilderAction)
        libraryBuilderAction->setEnabled(processIdle);
    if (listMenu_new)
        listMenu_new->setEnabled(processIdle);
    if (listMenu_shader)
        listMenu_shader->setEnabled(processIdle);
    if (list_view) {
#ifdef Q_OS_MACOS
        list_view->setColumnHidden(3, acmx2Tools);
#else
        list_view->setColumnHidden(3, false);
#endif
        list_view->headerItem()->setText(
            3, acmx2Tools ? tr("Compile Health") : tr("Build Status"));
    }

    if (runMenu) {
        runMenu_select->setToolTip({});
        runMenu_all->setToolTip({});
        runMenu_copyCommand->setToolTip({});
    }
    if (list_view) {
        list_view->setToolTip(
            tr("Right click while running to change the active shader."));
    }
}

void MainWindow::set_backend(acmx2::Backend backend, bool persist) {
    if (process && process->state() == QProcess::Running) {
        QMessageBox::information(
            this, tr("Process Running"),
            tr("Stop the running process before changing backends."));
        update_backend_ui();
        return;
    }

    if (libraryBuilderDialog) {
        libraryBuilderDialog->close();
        libraryBuilderDialog = nullptr;
    }

    QSettings settings("LostSideDead");
    settings.setValue(
        acmx2::backend_settings_key(active_backend, "executable"),
        executable_path);
    settings.setValue(acmx2::backend_settings_key(active_backend, "library"),
                      shader_path);

    active_backend = backend;
    if (persist)
        settings.setValue("interface/backend", acmx2::backend_id(active_backend));
    executable_path =
        settings
            .value(acmx2::backend_settings_key(active_backend, "executable"),
                   acmx2::default_backend_executable(active_backend))
            .toString();
    const QString nextLibrary =
        settings
            .value(acmx2::backend_settings_key(active_backend, "library"), "")
            .toString()
            .trimmed();

    shader_path.clear();
    items.clear();
    indexTimestamp = QDateTime();
    activeShaderManifestPath.clear();
    if (list_view)
        list_view->clear();

    if (!nextLibrary.isEmpty() && QFileInfo(nextLibrary).isDir() &&
        acmx2::shader_manifest_exists(nextLibrary)) {
        QString backendError;
        const std::optional<acmx2::Backend> libraryBackend =
            acmx2::shader_manifest_backend(nextLibrary, backendError);
        if (!backendError.isEmpty()) {
            Log(tr("Warning: Could not read backend metadata for %1: %2")
                    .arg(nextLibrary, backendError));
        } else if (libraryBackend && *libraryBackend != active_backend) {
            Log(tr("Warning: Saved %1 library belongs to %2: %3")
                    .arg(acmx2::backend_name(active_backend),
                         acmx2::backend_name(*libraryBackend), nextLibrary));
        } else {
            shader_path = nextLibrary;
            loadShaders(shader_path, true);
        }
    }

    cuda_available = false;
    cuda_device_available = false;
    audio_available = false;
    midi_available = false;
    dnn_available = false;
    initShaderSelectionSharedMemory();
    detectFeatureSupport();
    updateRecentLibrariesMenu();
    update_backend_ui();
    settings.sync();
    Log(tr("Backend selected: %1").arg(acmx2::backend_name(active_backend)));
    if (active_backend == acmx2::Backend::Acmxvk)
        Log(tr("ACMXVK launching and live shader selection are enabled."));
}

bool MainWindow::loadLibraryPath(const QString &path) {
    const QString trimmedPath = path.trimmed();
    if (trimmedPath.isEmpty())
        return false;
    const QString libraryPath = QDir::cleanPath(trimmedPath);
    const QFileInfo libraryInfo(libraryPath);
    if (!libraryInfo.exists()) {
        QMessageBox::warning(this, tr("Invalid Shader Path"),
                             tr("Shader directory does not exist:\n%1")
                                 .arg(libraryPath));
        return false;
    }
    if (!libraryInfo.isDir()) {
        QMessageBox::warning(this, tr("Invalid Shader Path"),
                             tr("Shader path is not a directory:\n%1")
                                 .arg(libraryPath));
        return false;
    }
    if (!acmx2::shader_manifest_exists(libraryPath)) {
        QMessageBox::warning(
            this, tr("Missing Shader Manifest"),
            tr("Shader directory does not contain library.json or index.txt:\n%1")
                .arg(libraryPath));
        return false;
    }
    QString backendError;
    const std::optional<acmx2::Backend> libraryBackend =
        acmx2::shader_manifest_backend(libraryPath, backendError);
    if (!backendError.isEmpty()) {
        QMessageBox::warning(this, tr("Invalid Backend Metadata"), backendError);
        return false;
    }
    if (libraryBackend && *libraryBackend != active_backend) {
        const QMessageBox::StandardButton reply = QMessageBox::question(
            this, tr("Switch Backend"),
            tr("This library targets %1, but the active backend is %2.\n\n"
               "Switch to %1 and load it?")
                .arg(acmx2::backend_name(*libraryBackend),
                     acmx2::backend_name(active_backend)),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        if (reply != QMessageBox::Yes)
            return false;
        set_backend(*libraryBackend);
    }
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString libraryTypeError;
        acmx2::shader_manifest_library_type(libraryPath, libraryTypeError);
        if (!libraryTypeError.isEmpty()) {
            QMessageBox::warning(this, tr("Invalid Library Type"),
                                 libraryTypeError);
            return false;
        }
    }
    if (!loadShaders(libraryPath, true)) {
        Log(tr("Warning: Could not load shaders from directory: %1")
                .arg(libraryPath));
        return false;
    }

    QSettings settings("LostSideDead");
    settings.setValue(acmx2::backend_settings_key(active_backend, "library"),
                      libraryPath);
    if (active_backend == acmx2::Backend::Acmx2)
        settings.setValue("shaders", libraryPath);
    settings.sync();
    addRecentLibrary(libraryPath);
    Log(tr("Successfully loaded shader library: %1").arg(libraryPath));
    update_backend_ui();
    return true;
}

void MainWindow::addRecentLibrary(const QString &path) {
    const QString trimmedPath = path.trimmed();
    if (trimmedPath.isEmpty())
        return;
    const QString libraryPath = QDir::cleanPath(trimmedPath);

    QSettings settings("LostSideDead");
    const QString recentKey =
        acmx2::backend_settings_key(active_backend, "recentLibraries");
    const QStringList legacyRecent = active_backend == acmx2::Backend::Acmx2
                                         ? settings.value("recentLibraries")
                                               .toStringList()
                                         : QStringList();
    QStringList recentLibraries =
        settings.value(recentKey, legacyRecent).toStringList();
    for (auto it = recentLibraries.begin(); it != recentLibraries.end();) {
        if (QDir::cleanPath(*it).compare(libraryPath, Qt::CaseInsensitive) == 0)
            it = recentLibraries.erase(it);
        else
            ++it;
    }
    recentLibraries.prepend(libraryPath);
    while (recentLibraries.size() > RECENT_LIBRARY_LIMIT)
        recentLibraries.removeLast();
    settings.setValue(recentKey, recentLibraries);
    if (active_backend == acmx2::Backend::Acmx2)
        settings.setValue("recentLibraries", recentLibraries);
    settings.sync();
    updateRecentLibrariesMenu();
}

void MainWindow::updateRecentLibrariesMenu() {
    if (!loadRecentMenu)
        return;

    loadRecentMenu->clear();
    QSettings settings("LostSideDead");
    const QString recentKey =
        acmx2::backend_settings_key(active_backend, "recentLibraries");
    const QStringList legacyRecent = active_backend == acmx2::Backend::Acmx2
                                         ? settings.value("recentLibraries")
                                               .toStringList()
                                         : QStringList();
    const QStringList recentLibraries =
        settings.value(recentKey, legacyRecent).toStringList();
    if (recentLibraries.isEmpty()) {
        QAction *emptyAction = loadRecentMenu->addAction(tr("No Recent Libraries"));
        emptyAction->setEnabled(false);
        return;
    }

    for (const QString &path : recentLibraries) {
        QAction *action = loadRecentMenu->addAction(path);
        connect(action, &QAction::triggered, this,
                [this, path]() { loadLibraryPath(path); });
    }
}

void MainWindow::fileOpenProp() {
    const acmx2::Backend propertiesBackend = active_backend;
    PropWindow propWindow(propertiesBackend, this);
    if (propWindow.exec() == QDialog::Accepted) {
        QString exePath = propWindow.exePathLineEdit->text();
        QString shaderDir = propWindow.shaderDirLineEdit->text();
        QString prefix = propWindow.screenshotDirLineEdit->text();
        QString compilerMode;
        QString compilerPath;
        if (propertiesBackend == acmx2::Backend::Acmxvk &&
            propWindow.shaderCompilerComboBox) {
            compilerMode =
                propWindow.shaderCompilerComboBox->currentData().toString();
            compilerPath =
                propWindow.shaderCompilerPathLineEdit->text().trimmed();
            if (compilerMode == QStringLiteral("custom") &&
                compilerPath.isEmpty()) {
                QMessageBox::information(
                    this, tr("Shader Compiler"),
                    tr("Select a custom glslc-compatible compiler path."));
                return;
            }
        }
        if (exePath.length() == 0) {
            QMessageBox::information(this, "No Path", "Requires Executable path");
            return;
        }
        if (shaderDir.length() == 0) {
            QMessageBox::information(this, "Shader Path", "Requires Shader Path");
            return;
        }

        if (!loadLibraryPath(shaderDir))
            return;

        QSettings appSettings("LostSideDead");
        if (active_backend == propertiesBackend) {
            appSettings.setValue(
                acmx2::backend_settings_key(active_backend, "executable"),
                exePath);
            if (active_backend == acmx2::Backend::Acmx2)
                appSettings.setValue("exePath", exePath);
            executable_path = exePath;
            if (propertiesBackend == acmx2::Backend::Acmxvk) {
                appSettings.setValue(
                    acmx2::backend_settings_key(
                        acmx2::Backend::Acmxvk,
                        "shader_compiler_mode"),
                    compilerMode.isEmpty() ? QStringLiteral("auto")
                                           : compilerMode);
                appSettings.setValue(
                    acmx2::backend_settings_key(
                        acmx2::Backend::Acmxvk,
                        "shader_compiler_path"),
                    compilerPath);
            }
        } else {
            Log(tr("Backend changed while loading the library; retained the "
                   "%1 executable setting.")
                    .arg(acmx2::backend_name(active_backend)));
        }
        appSettings.setValue("prefix_path", prefix);
        appSettings.sync();

        prefix_path = prefix;

        Log("Executable Path: " + executable_path);
        Log("Prefix Path: " + prefix);
        Log("Shader Directory: " + shaderDir);

    } else {
        Log("Canceled");
    }
}

void MainWindow::menuCustomUniforms() {
    if (!customUniformDialog || shader_path.isEmpty()) {
        QMessageBox::information(this, tr("Custom Uniforms"),
                                 tr("Load a shader library first."));
        return;
    }
    const QString jsonPath = QDir(shader_path).filePath("library.json");
    if (!QFileInfo(jsonPath).isFile()) {
        QMessageBox::warning(
            this, tr("Custom Uniforms"),
            tr("Custom uniforms require a library.json manifest."));
        return;
    }

    QString error;
    if (!customUniformDialog->loadLibrary(shader_path, active_backend, &error)) {
        QMessageBox::warning(this, tr("Could Not Load Custom Uniforms"), error);
        return;
    }
    customUniformDialog->show();
    customUniformDialog->raise();
    customUniformDialog->activateWindow();
}

void MainWindow::menuUniformReference() {
    if (!uniformReferenceDialog) {
        uniformReferenceDialog =
            new UniformReferenceDialog(active_backend, this);
        uniformReferenceDialog->setAttribute(Qt::WA_DeleteOnClose);
    } else
        uniformReferenceDialog->setBackend(active_backend);
    uniformReferenceDialog->show();
    uniformReferenceDialog->raise();
    uniformReferenceDialog->activateWindow();
}

bool MainWindow::loadShaders(const QString &path, bool force) {
    QString manifestPath = acmx2::shader_manifest_path(path);
    if (manifestPath.isEmpty()) {
        QMessageBox::warning(this, "Could not open shader manifest",
                             "No library.json or index.txt found in: " + path);
        return false;
    }

    if (active_backend == acmx2::Backend::Acmx2 &&
        QFileInfo(manifestPath).fileName().compare("index.txt", Qt::CaseInsensitive) == 0) {
        bool generated = false;
        QString migrationError;
        if (!acmx2::migrate_index_manifest_to_json(path, generated,
                                                   migrationError)) {
            Log("Could not generate library.json from index.txt: " + migrationError);
        } else if (generated) {
            manifestPath = acmx2::shader_manifest_path(path);
            Log("Generated library.json from index.txt");
        }
    }

    QDateTime modified = QFileInfo(manifestPath).lastModified();
    if (!force && path == shader_path && manifestPath == activeShaderManifestPath &&
        !indexTimestamp.isNull() && modified <= indexTimestamp) {
        return true;
    }
    QStringList manifestEntries;
    QString manifestError;
    if (!acmx2::load_shader_manifest(path, manifestEntries, manifestError)) {
        QMessageBox::warning(this, "Could not open shader manifest", manifestError);
        return false;
    }

    shader_path = path;
    activeShaderManifestPath = manifestPath;
    indexTimestamp = modified;
    if (customUniformDialog &&
        QFileInfo(manifestPath).fileName().compare("library.json", Qt::CaseInsensitive) == 0) {
        QString uniformError;
        if (!customUniformDialog->loadLibrary(path, active_backend, &uniformError))
            Log("Could not load custom uniforms: " + uniformError);
    }
    const int previousRow = currentShaderRow();
    const QString previouslySelected = currentShaderName();
    items.clear();
    QStringList uniqueItems;
    for (const QString &rawEntry : manifestEntries) {
        const QString line = rawEntry.trimmed();

        if (line.isEmpty()) {
            continue;
        }
        const QString shaderEntry = sanitizeShaderName(line);
        if (shaderEntry.isEmpty()) {
            Log("Skipping invalid shader path in " + QFileInfo(manifestPath).fileName() + ": " + line);
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
    const QString previousAudioFile = audio_file;
    const int previousAudioOutput = audio_output;
    const bool previousAudioPassThrough = audio_passthrough;
    const bool previousAudioTrunc = audio_trunc;
    const bool previousAudioRepeat = audio_repeat;
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
        audio_repeat = audio_set.isAudioRepeatEnabled();
        audio_buffers_enabled = audio_set.isAudioBuffersEnabled();
        audio_buffer_frames = audio_set.getAudioBufferFrames();
        audio_warm_rate = audio_set.getAudioWarmRate();
        Log("Audio Settings Saved");
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
        const bool liveAudioSettingsChanged =
            QFileInfo(audio_file).absoluteFilePath() !=
                QFileInfo(previousAudioFile).absoluteFilePath() ||
            audio_output != previousAudioOutput ||
            audio_passthrough != previousAudioPassThrough ||
            audio_trunc != previousAudioTrunc ||
            audio_repeat != previousAudioRepeat;
        if (shaderSelectionShm && process &&
            process->state() == QProcess::Running && !audio_file.isEmpty() &&
            liveAudioSettingsChanged) {
            const QByteArray path =
                QFileInfo(audio_file).absoluteFilePath().toUtf8();
            if (path.size() >=
                static_cast<int>(
                    acmx2::ipc::kShaderSelectionMaxAudioFilePath)) {
                Log("Audio file path is too long for live playback: " +
                    audio_file);
            } else {
                acmx2::ipc::ShaderSelectionLock lock(
                    shaderSelectionSemaphore);
                if (!lock) {
                    Log("Could not lock the live playback control channel");
                    return;
                }
                std::fill(std::begin(shaderSelectionShm->audio_file_path),
                          std::end(shaderSelectionShm->audio_file_path), '\0');
                std::copy(path.cbegin(), path.cend(),
                          shaderSelectionShm->audio_file_path);
                shaderSelectionShm->audio_output_device = audio_output;
                shaderSelectionShm->audio_pass_through =
                    audio_passthrough ? 1 : 0;
                shaderSelectionShm->audio_trunc = audio_trunc ? 1 : 0;
                shaderSelectionShm->audio_repeat = audio_repeat ? 1 : 0;
                ++shaderSelectionShm->audio_file_sequence;
                ++shaderSelectionShm->sequence;
                Log("Requested live audio-file change: " + audio_file +
                    "<br>");
            }
        }
#endif
    }
}

void MainWindow::menuGPUFilterSettings() {
    if (!cuda_available) {
        QMessageBox::information(this, tr("GPU Filter Settings"),
                                 tr("GPU filters are unavailable: acmx2 was built without CUDA support."));
        return;
    }
    GPUFilterDialog gpuDialog(executable_path, this);

    auto applyGpuDialogSettings = [&](bool enabled, const QString &filters, int bufferSize) {
        gpu_filter_enabled = enabled;
        gpu_filter_indices = filters;
        gpu_buffer_size = bufferSize;
        if (gpu_filter_enabled) {
            Log("GPU Filter Settings Saved: Filters=" + gpu_filter_indices + ", Buffer=" + QString::number(gpu_buffer_size));
        } else {
            Log("GPU Filtering Disabled");
        }
        publishRuntimeSettingsToRunningProcess();
    };

    connect(&gpuDialog, &GPUFilterDialog::settingsApplied, this,
            [&](bool enabled, const QString &filterArgument, int bufferSize) {
                applyGpuDialogSettings(enabled, filterArgument, bufferSize);
            });

    if (gpuDialog.exec() == QDialog::Accepted) {
        applyGpuDialogSettings(gpuDialog.isGPUFilterEnabled(),
                               gpuDialog.getFilterArgument(),
                               gpuDialog.getBufferSize());
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
    publishRuntimeSettingsToRunningProcess();
}

void MainWindow::menuWatermarkSettings() {
    QDialog dlg(this);
    dlg.setWindowTitle(tr("Watermark Settings"));
    acmx2::applyCustomStyleIfEnabled(&dlg);

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
            .arg(watermark_r)
            .arg(watermark_g)
            .arg(watermark_b));
    publishRuntimeSettingsToRunningProcess();
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

    if (shaderPassDialog) {
        shaderPassDialog->updateShaderList(items);
        shaderPassDialog->show();
        shaderPassDialog->raise();
        shaderPassDialog->activateWindow();
        return;
    }

    shaderPassDialog = new ShaderPassDialog(items, this);
    shaderPassDialog->setAttribute(Qt::WA_DeleteOnClose);
    shaderPassDialog->setEnabled(shader_pass_enabled);
    if (!shader_pass_names.isEmpty()) {
        shaderPassDialog->setSelectedShaderNames(shader_pass_names);
    }

    ShaderPassDialog *dialog = shaderPassDialog;
    auto applyMultipassSettings = [this, dialog]() {
        shader_pass_enabled = dialog->isShaderPassEnabled();
        shader_pass_names = dialog->getSelectedShaderNames();
        publishMultipassShadersToRunningProcess();
        if (shader_pass_enabled) {
            Log("Multi-Pass Shader Settings Saved: " + QString::number(shader_pass_names.size()) + " passes");
        } else {
            Log("Multi-Pass Shader Disabled");
        }
    };

    connect(dialog, &ShaderPassDialog::settingsApplied, this,
            [this](bool enabled, const QStringList &selectedShaderNames) {
                shader_pass_enabled = enabled;
                shader_pass_names = selectedShaderNames;
                publishMultipassShadersToRunningProcess();
                if (shader_pass_enabled) {
                    Log("Multi-Pass Shader Settings Saved: " + QString::number(shader_pass_names.size()) + " passes");
                } else {
                    Log("Multi-Pass Shader Disabled");
                }
            });
    connect(dialog, &ShaderPassDialog::shaderEditRequested, this,
            [this](const QString &shaderName) {
                const QString safeName = sanitizeShaderName(shaderName);
                if (!safeName.isEmpty())
                    openShaderEditor(QDir(shader_path).filePath(safeName));
            });
    connect(dialog, &QDialog::accepted, this, applyMultipassSettings);

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
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

    if (playlistDialog) {
        playlistDialog->updateShaderList(items);
        playlistDialog->show();
        playlistDialog->raise();
        playlistDialog->activateWindow();
        return;
    }

    playlistDialog = new PlaylistDialog(items, this);
    playlistDialog->setAttribute(Qt::WA_DeleteOnClose);
    playlistDialog->setEnabled(playlist_enabled);
    if (!playlist_tree_data.isEmpty()) {
        playlistDialog->setPlaylistTree(playlist_tree_data);
    } else if (!playlist_names.isEmpty()) {
        playlistDialog->setSelectedShaderNames(playlist_names);
    }
    if (!playlist_file_path.isEmpty()) {
        playlistDialog->setPlaylistFile(playlist_file_path);
    }
    playlistDialog->setAutopilotFrames(autopilot_frames);
    playlistDialog->setAutopilotRandom(autopilot_random);

    PlaylistDialog *dialog = playlistDialog;
    connect(dialog, &QDialog::accepted, this, [this, dialog]() {
        playlist_enabled = dialog->isPlaylistEnabled();
        playlist_names = dialog->getSelectedShaderNames();
        playlist_tree_data = dialog->getPlaylistTree();
        playlist_file_path = dialog->getPlaylistFile();
        autopilot_frames = dialog->getAutopilotFrames();
        autopilot_random = dialog->isAutopilotRandom();
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
    });

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

void MainWindow::cameraSettings() {
    SettingsWindow settingsWindow(executable_path, active_backend, this);
    settingsWindow.setCudaAvailable(cuda_device_available);
    settingsWindow.setDnnAvailable(dnn_available);
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
        maximize_fps = settingsWindow.isMaximizeFpsEnabled();
        use_source_fps = settingsWindow.isUseSourceFpsEnabled();
        use_source_audio = settingsWindow.isUseSourceAudioEnabled();
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
    rotate_enabled = settingsWindow.is_rotate_enabled();
    rotation_mode = settingsWindow.get_rotation_mode();
    png_output = settingsWindow.isPngOutputEnabled();
    generate_enabled = settingsWindow.isGenerateEnabled();
    generate_interval = settingsWindow.getGenerateInterval();
    encode_preset = settingsWindow.getEncodePreset();
    encode_tune = settingsWindow.getEncodeTune();
    encode_crf = settingsWindow.getEncodeCrf();
    encode_rate_control = settingsWindow.getEncodeRateControl();
    encode_bitrate = settingsWindow.getEncodeBitrate();
    encode_codec = settingsWindow.getEncodeCodec();
    encode_parameters = settingsWindow.getEncodeParameters();
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
    initShaderSelectionSharedMemory();
    publishSelectedShaderIndexToRunningProcess();
    publishMultipassShadersToRunningProcess();
    publishRuntimeSettingsToRunningProcess();
    const QString data = currentShaderName();
    if (data.isEmpty()) {
        Log("<b>No item selected.</b>");
        return;
    }
    QString launchShaderPath = shader_path;
    QString launchShaderName = data;
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString runtimeError;
        if (!resolve_acmxvk_runtime_library(shader_path, launchShaderPath,
                                            runtimeError)) {
            prompt_acmxvk_rebuild(runtimeError,
                                  PendingAcmxvkAction::RunSelected);
            return;
        }
        if (launchShaderPath != shader_path)
            launchShaderName = acmxvk_runtime_shader_name(data);
        if (!QFileInfo(QDir(launchShaderPath).filePath(launchShaderName)).isFile()) {
            QMessageBox::warning(
                this, tr("Build ACMXVK Library"),
                tr("The compiled shader is missing. Choose Playback > Build "
                   "and try again.\n\n%1")
                    .arg(QDir(launchShaderPath).filePath(launchShaderName)));
            return;
        }
    }
    QStringList arguments;
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/" +
                      acmx2::default_backend_executable(active_backend);
#endif
    dirPath = resolve_backend_assets_path(active_backend, executable_path,
                                          shader_path);
    const int selectedIndex = currentShaderRow();
    if (selectedIndex < 0 || selectedIndex >= items.size()) {
        Log("<b>No valid shader selection.</b>");
        return;
    }
    if (active_backend == acmx2::Backend::Acmxvk)
        arguments << "--unbuffered";
    arguments << "--path" << dirPath;
    if (active_backend == acmx2::Backend::Acmxvk) {
        // ACMXVK needs its manifest to resolve fragment/compute types and
        // custom-uniform metadata for the selected SPIR-V shader.
        arguments << "--shaders" << launchShaderPath << "--shader-file"
                  << launchShaderName << "--interface-shm";
    } else {
        // ACMX2 can compile a selected source directly without loading the
        // complete shader library and its binary cache.
        arguments << "--fragment" << (shader_path + "/" + data)
                  << "--interface-shm";
    }
    // Pass texture cache size so the SIZE macro injected into the fragment
    // matches whatever the user has configured for cache shaders.
    arguments << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        arguments << "--texture-cache-array";
    const QSize effectiveCameraResolution =
        hasPositiveResolution(camera_res) ? camera_res : QSize(1280, 720);
    QString res;
    QTextStream stream(&res);
    stream << effectiveCameraResolution.width() << "x"
           << effectiveCameraResolution.height();

    QString scr_res;
    QTextStream stream_r(&scr_res);
    stream_r << screen_res.width() << "x" << screen_res.height();

    if (full_screen_value)
        arguments << "--fullscreen";

    if (!graphics_file.isEmpty()) {
        arguments << "--graphic" << graphics_file;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--fps" << QString::number(output_fps);
    } else if (video_file.isEmpty()) {
        arguments << "--camera-res" << res;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--device" << QString::number(camera_index);
        arguments << "--fps" << QString::number(output_fps);
        if (active_backend == acmx2::Backend::Acmxvk && maximize_fps)
            arguments << "--maximize-fps";
        if (use_yuv)
            arguments << "--use-yuv";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
    } else {
        arguments << "--input" << video_file;
        if (active_backend == acmx2::Backend::Acmxvk && use_source_fps) {
            arguments << "--use-source-fps";
            if (use_source_audio)
                arguments << "--use-source-audio";
        }
        if (hasPositiveResolution(screen_res))
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
        if (active_backend == acmx2::Backend::Acmxvk &&
            encode_rate_control == "bitrate")
            arguments << "--video-bitrate" << encode_bitrate;
        else
            arguments << "--encode-crf" << QString::number(encode_crf);
        if (!encode_preset.isEmpty())
            arguments << "--encode-preset" << encode_preset;
        if (!encode_tune.isEmpty())
            arguments << "--encode-tune" << encode_tune;
        if (!encode_codec.isEmpty() && encode_codec != "auto")
            arguments << "--encode-codec" << encode_codec;
        if (!encode_parameters.isEmpty())
            arguments << "--encode-params" << encode_parameters;
        if (encode_realtime)
            arguments << "--encode-realtime";
        if (encode_no_drop &&
            (!video_file.isEmpty() || !graphics_file.isEmpty()))
            arguments << "--no-drop";
    }
    const bool sourceAudioActive =
        active_backend == acmx2::Backend::Acmxvk && !video_file.isEmpty() &&
        use_source_fps && use_source_audio;
    if (audio_available && audio_enabled && !sourceAudioActive) {
        arguments << "--enable-audio";
        arguments << "--channels" << QString::number(audio_channels);

        if (audio_input == -1)
            arguments << "--audio-input" << "default";
        else
            arguments << "--audio-input" << QString::number(audio_input);

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

    if (active_backend == acmx2::Backend::Acmxvk && !record_audio &&
        (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--mute-output";
    }

    if (audio_available &&
        (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--sense" << QString::number(audio_sense);
        if (audio_passthrough) {
            arguments << "--pass-through";
            if (audio_output == -1)
                arguments << "--audio-output" << "default";
            else
                arguments << "--audio-output" << QString::number(audio_output);
        }
    }

    if (audio_available && !audio_file.isEmpty() && !sourceAudioActive) {
        arguments << "--audio-file" << audio_file;
        if (audio_trunc) {
            arguments << "--audio-trunc";
        }
        if (audio_repeat) {
            arguments << "--audio-repeat";
        }
    }

    if (audio_available && audio_buffers_enabled) {
        arguments << "--enable-audio-buffers" << QString::number(audio_buffer_frames);
    }

    if (audio_available &&
        (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
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

    if (cuda_device_available) {
        arguments << "--cuda-device" << QString::number(cuda_device);
    }

    arguments << "--time-speed"
              << QString::number(static_cast<double>(time_speed), 'f', 2);
    if (normalized_time) {
        arguments << "--normalized";
    }

    if (!use_shader_cache && active_backend == acmx2::Backend::Acmx2) {
        arguments << "--no-cache";
    }

    if (midi_available && midi_enabled && !midi_config_file.isEmpty()) {
        arguments << "--midi-map" << midi_config_file;
        if (midi_device >= 0)
            arguments << "--midi-device" << QString::number(midi_device);
    }

    if (!output_file.isEmpty() && duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (!output_file.isEmpty() && max_size_limit_enabled && max_size_mb > 0.0) {
        arguments << "--max-size" << QString::number(max_size_mb, 'f', 2);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    if (flip_enabled) {
        arguments << "--flip";
    }

    if (rotate_enabled) {
        arguments << "--rotate" << rotation_mode;
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

    // ACMX2 single-source mode bypasses its binary cache. ACMXVK uses the
    // selected runtime library and does not accept ACMX2 cache controls.
    Log("shell: " + executable_path + " " + concatList(arguments) + "<br>");
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Failed to start the program.</b>");
        QMessageBox::critical(this, "Error", "Failed to start the program.");
    } else {
        play_stop->setEnabled(true);
    }
}

bool MainWindow::buildRunArguments(QStringList &arguments,
                                   PendingAcmxvkAction resume_action) {
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
    if (items.isEmpty()) {
        QMessageBox::warning(this, tr("Empty Shader Library"),
                             tr("The selected shader library contains no shaders."));
        return false;
    }
    if (index < 0 || index >= items.size()) {
        QMessageBox::warning(this, tr("Invalid Shader Selection"),
                             tr("Select a shader from the active library."));
        return false;
    }
    QString launchShaderPath = shader_path;
    QString launchShaderName = items.at(index);
    if (active_backend == acmx2::Backend::Acmxvk) {
        QString runtimeError;
        if (!resolve_acmxvk_runtime_library(shader_path, launchShaderPath,
                                            runtimeError)) {
            prompt_acmxvk_rebuild(runtimeError, resume_action);
            return false;
        }
        if (launchShaderPath != shader_path)
            launchShaderName = acmxvk_runtime_shader_name(launchShaderName);
        if (!QFileInfo(QDir(launchShaderPath).filePath(launchShaderName)).isFile()) {
            QMessageBox::warning(
                this, tr("Build ACMXVK Library"),
                tr("The compiled shader is missing. Choose Playback > Build "
                   "and try again.\n\n%1")
                    .arg(QDir(launchShaderPath).filePath(launchShaderName)));
            return false;
        }
    }
    QString dirPath = QCoreApplication::applicationDirPath();
#ifdef BUILD_BUNDLE
    executable_path = dirPath + "/../Helpers/" +
                      acmx2::default_backend_executable(active_backend);
#endif
    dirPath = resolve_backend_assets_path(active_backend, executable_path,
                                          shader_path);

    QString shader_file = launchShaderPath;
    if (active_backend == acmx2::Backend::Acmxvk)
        arguments << "--unbuffered";
    arguments << "--path" << dirPath << "--shaders" << shader_file;
    arguments << "--interface-shm";
    // Always pass texture cache size so runtime SIZE matches the cache file.
    arguments << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        arguments << "--texture-cache-array";
    const QSize effectiveCameraResolution =
        hasPositiveResolution(camera_res) ? camera_res : QSize(1280, 720);
    QString res;
    QTextStream stream(&res);
    stream << effectiveCameraResolution.width() << "x"
           << effectiveCameraResolution.height();
    QString scr_res;
    QTextStream stream_r(&scr_res);
    stream_r << screen_res.width() << "x" << screen_res.height();

    if (full_screen_value)
        arguments << "--fullscreen";

    if (!graphics_file.isEmpty()) {
        arguments << "--graphic" << graphics_file;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--fps" << QString::number(output_fps);
    } else if (video_file.isEmpty()) {
        arguments << "--camera-res" << res;
        if (hasPositiveResolution(screen_res))
            arguments << "--resolution" << scr_res;
        arguments << "--device" << QString::number(camera_index);
        arguments << "--fps" << QString::number(output_fps);
        if (active_backend == acmx2::Backend::Acmxvk && maximize_fps)
            arguments << "--maximize-fps";
        if (use_yuv)
            arguments << "--use-yuv";
        if (cache_enabled) {
            arguments << "--texture-cache";
            arguments << "--cache-delay" << QString::number(cache_delay);
        }
    } else {
        arguments << "--input" << video_file;
        if (active_backend == acmx2::Backend::Acmxvk && use_source_fps) {
            arguments << "--use-source-fps";
            if (use_source_audio)
                arguments << "--use-source-audio";
        }
        if (hasPositiveResolution(screen_res))
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
        if (active_backend == acmx2::Backend::Acmxvk &&
            encode_rate_control == "bitrate")
            arguments << "--video-bitrate" << encode_bitrate;
        else
            arguments << "--encode-crf" << QString::number(encode_crf);
        if (!encode_preset.isEmpty())
            arguments << "--encode-preset" << encode_preset;
        if (!encode_tune.isEmpty())
            arguments << "--encode-tune" << encode_tune;
        if (!encode_codec.isEmpty() && encode_codec != "auto")
            arguments << "--encode-codec" << encode_codec;
        if (!encode_parameters.isEmpty())
            arguments << "--encode-params" << encode_parameters;
        if (encode_realtime)
            arguments << "--encode-realtime";
        if (encode_no_drop &&
            (!video_file.isEmpty() || !graphics_file.isEmpty()))
            arguments << "--no-drop";
    }
    arguments << "--shader-file" << launchShaderName;

    const bool sourceAudioActive =
        active_backend == acmx2::Backend::Acmxvk && !video_file.isEmpty() &&
        use_source_fps && use_source_audio;
    if (audio_available && audio_enabled && !sourceAudioActive) {
        arguments << "--enable-audio";
        arguments << "--channels" << QString::number(audio_channels);

        if (audio_input == -1)
            arguments << "--audio-input" << "default";
        else
            arguments << "--audio-input" << QString::number(audio_input);

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

    if (active_backend == acmx2::Backend::Acmxvk && !record_audio &&
        (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--mute-output";
    }

    if (audio_available &&
        (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
        arguments << "--sense" << QString::number(audio_sense);
        if (audio_passthrough) {
            arguments << "--pass-through";
            if (audio_output == -1)
                arguments << "--audio-output" << "default";
            else
                arguments << "--audio-output" << QString::number(audio_output);
        }
    }

    if (audio_available && !audio_file.isEmpty() && !sourceAudioActive) {
        arguments << "--audio-file" << audio_file;
        if (audio_trunc) {
            arguments << "--audio-trunc";
        }
        if (audio_repeat) {
            arguments << "--audio-repeat";
        }
    }

    if (audio_available && audio_buffers_enabled) {
        arguments << "--enable-audio-buffers" << QString::number(audio_buffer_frames);
    }

    if (audio_available &&
        (audio_enabled || !audio_file.isEmpty() || sourceAudioActive)) {
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
            QStringList passFiles;
            const QStringList indexValues = passIndices.split(',');
            for (const QString &indexValue : indexValues) {
                bool ok = false;
                const int passIndex = indexValue.toInt(&ok);
                if (ok && passIndex >= 0 && passIndex < items.size()) {
                    const QString passFile = items.at(passIndex);
                    passFiles.append(launchShaderPath == shader_path
                                         ? passFile
                                         : acmxvk_runtime_shader_name(passFile));
                }
            }
            QByteArray passFilePayload;
            for (const QString &passFile : passFiles) {
                const QByteArray encodedName = passFile.toUtf8();
                passFilePayload.append(QByteArray::number(encodedName.size()));
                passFilePayload.append(':');
                passFilePayload.append(encodedName);
            }
            arguments << "--shader-pass-files"
                      << QString::fromUtf8(passFilePayload);
        }
    }

    if (cuda_device_available) {
        arguments << "--cuda-device" << QString::number(cuda_device);
    }

    arguments << "--time-speed"
              << QString::number(static_cast<double>(time_speed), 'f', 2);
    if (normalized_time) {
        arguments << "--normalized";
    }

    if (!use_shader_cache && active_backend == acmx2::Backend::Acmx2) {
        arguments << "--no-cache";
    }

    if (midi_available && midi_enabled && !midi_config_file.isEmpty()) {
        arguments << "--midi-map" << midi_config_file;
        if (midi_device >= 0)
            arguments << "--midi-device" << QString::number(midi_device);
    }

    const bool playlistActive = playlist_enabled && !playlist_names.isEmpty();
    if (playlistActive) {
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
                    for (const QString &name : shaders) {
                        out << (launchShaderPath == shader_path
                                    ? name
                                    : acmxvk_runtime_shader_name(name))
                            << "\n";
                    }
                }
            } else {
                for (const QString &name : playlist_names) {
                    out << (launchShaderPath == shader_path
                                ? name
                                : acmxvk_runtime_shader_name(name))
                        << "\n";
                }
            }
            f.close();
            playlist_file_path = plFile;
        }
        arguments << "--playlist" << plFile;
    }

    if (playlistActive && autopilot_frames > 0) {
        arguments << (autopilot_random ? "--autopilot-random" : "--autopilot-frames")
                  << QString::number(autopilot_frames);
    }

    if (!output_file.isEmpty() && duration_limit_enabled && max_duration > 0.0) {
        arguments << "--duration" << QString::number(max_duration, 'f', 1);
    }

    if (!output_file.isEmpty() && max_size_limit_enabled && max_size_mb > 0.0) {
        arguments << "--max-size" << QString::number(max_size_mb, 'f', 2);
    }

    if (cross_fade_duration != 0.5f) {
        arguments << "--cross-fade" << QString::number(static_cast<double>(cross_fade_duration), 'f', 2);
    }

    if (flip_enabled) {
        arguments << "--flip";
    }

    if (rotate_enabled) {
        arguments << "--rotate" << rotation_mode;
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
        if (p == "ultrafast")
            nvencPreset = "p1";
        else if (p == "superfast")
            nvencPreset = "p2";
        else if (p == "veryfast")
            nvencPreset = "p3";
        else if (p == "faster")
            nvencPreset = "p4";
        else if (p == "fast")
            nvencPreset = "p5";
        else if (p == "medium")
            nvencPreset = "p6";
        else if (p == "slow")
            nvencPreset = "p6";
        else if (p == "slower")
            nvencPreset = "p7";
        else if (p == "veryslow")
            nvencPreset = "p7";
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
    if (!buildRunArguments(arguments, PendingAcmxvkAction::RunAll))
        return;
    initShaderSelectionSharedMemory();
    publishSelectedShaderIndexToRunningProcess();
    publishMultipassShadersToRunningProcess();
    publishRuntimeSettingsToRunningProcess();

    Log("shell: " + executable_path + " " + concatList(arguments) + "<br>");
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
    if (!buildRunArguments(arguments, PendingAcmxvkAction::CopyCommand))
        return;

    QString exe = executable_path;
    if (exe.isEmpty())
        exe = acmx2::default_backend_executable(active_backend);
    QStringList envAssignments;
#ifdef __linux__
    envAssignments = defaultLinuxRunEnvAssignments();
#endif
    QString commandText = buildShellCommand(envAssignments, exe, arguments).trimmed();

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Copy Command"));
    dialog.resize(720, 320);
    acmx2::applyCustomStyleIfEnabled(&dialog);

    QVBoxLayout *layout = new QVBoxLayout(&dialog);
    QPlainTextEdit *textBox = new QPlainTextEdit(&dialog);
    textBox->setPlainText(commandText);
    textBox->setReadOnly(false);
    textBox->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    if (!acmx2::isCustomStyleEnabled()) {
        textBox->setStyleSheet("QPlainTextEdit { background-color: black; color: lime; "
                               "font-size: 14px; font-family: 'Courier New', Courier, monospace; "
                               "border: 1px solid red; }");
    } else {
        QFont commandFont("Courier New");
        commandFont.setStyleHint(QFont::Monospace);
        commandFont.setPointSize(14);
        textBox->setFont(commandFont);
    }
    layout->addWidget(textBox);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(&dialog);
    QPushButton *copyButton = buttonBox->addButton(tr("Copy to Clipboard"), QDialogButtonBox::ActionRole);
    QPushButton *runButton = buttonBox->addButton(tr("Run"), QDialogButtonBox::ActionRole);
    QPushButton *okButton = buttonBox->addButton(QDialogButtonBox::Ok);
    layout->addWidget(buttonBox);

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
        if (shell.isEmpty())
            shell = "cmd.exe";
        QStringList shellArgs{"/C", cmdText};
#else
        QString shell = "/bin/sh";
        QStringList shellArgs{"-c", cmdText};
#endif
        Log("shell: " + cmdText + "<br>");
        initShaderSelectionSharedMemory();
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
    QString build_path = shader_path;
    if (build_path.isEmpty()) {
        QSettings appSettings("LostSideDead");
        build_path =
            appSettings
                .value(acmx2::backend_settings_key(active_backend, "library"),
                       active_backend == acmx2::Backend::Acmx2
                           ? appSettings.value("shaders", "").toString()
                           : QString())
                .toString();
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

    if (active_backend == acmx2::Backend::Acmxvk) {
        start_acmxvk_build(build_path, AcmxvkBuildMode::Strict);
        return;
    }

#ifdef Q_OS_MACOS
    // ACMX2 does not support its persistent OpenGL binary cache on macOS.
    Log("Rebuild Shader Cache is not available on macOS.");
    cacheBuildInProgress = false;
    return;
#else

    const QString assets_path = resolveAssetsPath();

    QStringList args;
    args << "--build" << build_path;
    args << "-p" << assets_path;
    args << "--texture-cache-size" << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        args << "--texture-cache-array";

    if (enable_3d) {
        args << "--enable-3d";
    }

    Log("Building shader cache for: " + build_path);
    Log("Command: " + executable_path + " " + args.join(" ") + "<br>");

    play_stop->setEnabled(true);
    cacheBuildInProgress = true;
    initShaderSelectionSharedMemory();
    process->start(executable_path, args);

    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start shader cache build process");
        cacheBuildInProgress = false;
        play_stop->setEnabled(false);
    }
#endif
}

void MainWindow::menuFixBuild() {
    if (active_backend != acmx2::Backend::Acmxvk)
        return;
    if (process->state() == QProcess::Running) {
        QMessageBox::warning(
            this, tr("Fix Build"),
            tr("A process is already running. Please wait for it to finish."));
        return;
    }
    if (shader_path.isEmpty()) {
        QMessageBox::warning(
            this, tr("Fix Build"),
            tr("No shader library is loaded."));
        return;
    }
    start_acmxvk_build(shader_path, AcmxvkBuildMode::Fix);
}

void MainWindow::start_acmxvk_build(const QString &build_path,
                                    AcmxvkBuildMode mode) {
    const bool fix = mode != AcmxvkBuildMode::Strict;
    const bool prune = mode == AcmxvkBuildMode::Prune;
    const QString dialogTitle =
        prune ? tr("Remove Broken Shaders")
              : (fix ? tr("Fix Build") : tr("Build ACMXVK Library"));
    QString type_error;
    if (!is_acmxvk_source_library(build_path, type_error)) {
        pending_acmxvk_action = PendingAcmxvkAction::None;
        QMessageBox::warning(
            this, dialogTitle,
            type_error.isEmpty()
                ? tr("The selected ACMXVK library is already a compiled "
                     "runtime library.")
                : type_error);
        return;
    }

    const QString manifest_path =
        QDir(build_path).filePath(QStringLiteral("library.json"));
    if (!QFileInfo(manifest_path).isFile()) {
        pending_acmxvk_action = PendingAcmxvkAction::None;
        QMessageBox::warning(
            this, dialogTitle,
            tr("ACMXVK source builds require library.json:\n%1")
                .arg(manifest_path));
        return;
    }

    const QString output_path = acmxvk_build_directory(build_path);
    QString compilerError;
    const QString compiler =
        resolve_acmxvk_shader_compiler(compilerError);
    if (compiler.isEmpty()) {
        pending_acmxvk_action = PendingAcmxvkAction::None;
        Log(tr("<b style='color:red;'>Cannot build ACMXVK library: %1</b>")
                .arg(compilerError.toHtmlEscaped()));
        QMessageBox::warning(this, tr("ACMXVK Shader Compiler"),
                             compilerError);
        return;
    }
    QStringList arguments{"--unbuffered", "--build", manifest_path};
    arguments << (fix ? QStringLiteral("--fix")
                      : QStringLiteral("--builddir"))
              << output_path;
    arguments << QStringLiteral("--glslc") << compiler;
    if (prune)
        arguments << QStringLiteral("--prune") << QStringLiteral("--force");
    Log(prune ? tr("Removing broken ACMXVK shader sources from: %1")
                    .arg(build_path)
              : (fix ? tr("Fix building ACMXVK SPIR-V library: %1")
                           .arg(build_path)
                     : tr("Building ACMXVK SPIR-V library: %1")
                           .arg(build_path)));
    Log("Command: " + executable_path + " " + concatList(arguments) + "<br>");
    play_stop->setEnabled(true);
    cacheBuildInProgress = true;
    acmxvkPruneLibraryPath = prune ? build_path : QString();
    process->start(executable_path, arguments);
    if (!process->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start ACMXVK build process");
        cacheBuildInProgress = false;
        acmxvkPruneLibraryPath.clear();
        pending_acmxvk_action = PendingAcmxvkAction::None;
        play_stop->setEnabled(false);
    }
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
        scan_path =
            appSettings
                .value(acmx2::backend_settings_key(active_backend, "library"),
                       active_backend == acmx2::Backend::Acmx2
                           ? appSettings.value("shaders", "").toString()
                           : QString())
                .toString();
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

    const QString manifestPath = acmx2::shader_manifest_path(scan_path);
    if (manifestPath.isEmpty()) {
        QMessageBox::warning(this, "Missing Shader Manifest",
                             "No library.json or index.txt found in: " + scan_path);
        return;
    }
    const QString manifestName = QFileInfo(manifestPath).fileName();

    if (active_backend == acmx2::Backend::Acmxvk) {
        QString typeError;
        if (!is_acmxvk_source_library(scan_path, typeError)) {
            QMessageBox::warning(
                this, tr("Remove Broken Shaders"),
                typeError.isEmpty()
                    ? tr("Remove Broken requires an ACMXVK source library, "
                         "not a compiled runtime library.")
                    : typeError);
            return;
        }

        QMessageBox confirmation(this);
        confirmation.setIcon(QMessageBox::Warning);
        confirmation.setWindowTitle(tr("Permanently Remove Broken Shaders"));
        confirmation.setText(
            tr("This operation permanently deletes source shader files."));
        confirmation.setInformativeText(
            tr("ACMXVK will compile every shader listed in:\n\n%1\n\n"
               "Any .frag or .comp source for which glslc reports a compilation "
               "failure will be deleted. The generated runtime library and the "
               "source library manifest will then omit those shaders.\n\n"
               "No backup is created and this operation cannot be undone. "
               "Commit or archive the library before continuing.\n\n"
               "Do you want to permanently remove the broken sources?")
                .arg(scan_path));
        confirmation.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        confirmation.setDefaultButton(QMessageBox::No);
        confirmation.setEscapeButton(QMessageBox::No);
        if (confirmation.exec() != QMessageBox::Yes)
            return;

        start_acmxvk_build(scan_path, AcmxvkBuildMode::Prune);
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                              tr("Remove Broken Shaders"),
                                                              tr("This will compile every shader in:\n\n%1\n\n"
                                                                 "Any shader that fails to compile will be removed from %2 "
                                                                 "(the original will be backed up as %2.bak).\n\nContinue?")
                                                                  .arg(scan_path, manifestName),
                                                              QMessageBox::Yes | QMessageBox::No);
    if (reply != QMessageBox::Yes)
        return;

    const QString assets_path = resolveAssetsPath();

    QStringList args;
    args << "--remove-broken" << scan_path;
    args << "-p" << assets_path;
    args << "--texture-cache-size"
         << QString::number(cache_size > 0 ? cache_size : 8);
    if (cache_enabled && textureCacheArraySettingEnabled())
        args << "--texture-cache-array";
    if (enable_3d)
        args << "--enable-3d";

    Log("Scanning for broken shaders in: " + scan_path);
    Log("Command: " + executable_path + " " + args.join(" ") + "<br>");

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
            [this, scan, scan_path, manifestName](int exitCode, QProcess::ExitStatus) {
                Log(QString("Remove-broken finished with exit code: %1<br>").arg(exitCode));
                if (exitCode == 0) {
                    // Reload the list view from the updated manifest.
                    loadShaders(scan_path, true);
                    QMessageBox::information(this,
                                             tr("Remove Broken"),
                                             tr("Finished scanning shader library.\n\n"
                                                "%1 has been updated and the shader list reloaded.\n"
                                                "A backup of the original is at:\n%2/%1.bak")
                                                 .arg(manifestName, scan_path));
                } else {
                    QMessageBox::warning(this,
                                         tr("Remove Broken"),
                                         tr("Remove-broken failed with exit code %1. "
                                            "%2 was not changed.")
                                             .arg(exitCode)
                                             .arg(manifestName));
                }
                scan->deleteLater();
            });

    scan->start(executable_path, args);
    if (!scan->waitForStarted()) {
        Log("<b style='color:red;'>Error:</b> Failed to start remove-broken process");
        scan->deleteLater();
    }
}

void MainWindow::menuCleanShaderCache() {
#ifdef Q_OS_MACOS
    Log("Clean Shader Cache is not available on macOS.");
    return;
#else
    QString libraryPath = shader_path;
    if (libraryPath.isEmpty()) {
        QSettings appSettings("LostSideDead");
        libraryPath = appSettings.value("shaders", "").toString();
    }

    if (libraryPath.isEmpty()) {
        QMessageBox::warning(this, "Error", "No shader library loaded. Please set a shader directory in Properties or load a shader library first.");
        return;
    }
    if (process->state() == QProcess::Running || cacheBuildInProgress) {
        QMessageBox::warning(this, "Error", "A process is running. Stop it before cleaning the shader cache.");
        return;
    }

    const QMessageBox::StandardButton reply = QMessageBox::question(
        this, tr("Clean Shader Cache"),
        tr("Delete all cached shader binaries for:\n\n%1\n\n"
           "This will not rebuild the cache. Continue?")
            .arg(libraryPath),
        QMessageBox::Yes | QMessageBox::No);
    if (reply != QMessageBox::Yes)
        return;

    const QString assetsPath = resolveAssetsPath();
    QStringList cacheFiles;
    const auto addCacheFile = [&cacheFiles](const QString &path) {
        if (!cacheFiles.contains(path))
            cacheFiles.append(path);
    };

    // Current cache files are keyed by texture-cache size and array mode.
    // Enumerate every valid combination so cleaning is independent of the
    // currently selected Session Settings.
    for (int size = 1; size <= 64; ++size) {
        for (const bool useArray : {false, true}) {
            const QString filename =
                shaderCacheFilename(libraryPath, size, useArray);
            addCacheFile(assetsPath + "/" + filename);
            addCacheFile(libraryPath + "/" + filename);
        }
    }

    // Remove the pre-size-key hashed cache and the original fixed-name cache.
    std::error_code ec;
    const std::filesystem::path libraryFsPath(libraryPath.toStdString());
    const std::filesystem::path absoluteLibrary =
        std::filesystem::absolute(libraryFsPath, ec);
    const std::string legacyKey =
        ec ? libraryPath.toStdString()
           : absoluteLibrary.lexically_normal().string();
    std::ostringstream legacyName;
    legacyName << ".shader_cache_" << std::hex
               << std::hash<std::string>{}(legacyKey);
    const QString legacyHashedName =
        QString::fromStdString(legacyName.str());
    addCacheFile(assetsPath + "/" + legacyHashedName);
    addCacheFile(libraryPath + "/" + legacyHashedName);
    addCacheFile(libraryPath + "/.shader_cache");

    int removedCount = 0;
    int failedCount = 0;
    for (const QString &cacheFile : cacheFiles) {
        if (!QFileInfo::exists(cacheFile))
            continue;
        if (QFile::remove(cacheFile)) {
            Log("Deleted shader cache: " + cacheFile);
            ++removedCount;
        } else {
            Log("<b style='color:red;'>Warning:</b> Could not delete cache file: " + cacheFile);
            ++failedCount;
        }
    }

    if (removedCount == 0 && failedCount == 0) {
        Log("No existing shader cache found");
    } else {
        Log(QString("Shader cache clean complete: removed %1 file(s), %2 failed")
                .arg(removedCount)
                .arg(failedCount));
    }
    populateShaderTree();
#endif
}

void MainWindow::detectCudaSupport() {
    detectFeatureSupport();
}

static QString probe_feature_output(const QString &exe, const QString &flag) {
    QProcess probe;
    probe.start(exe, QStringList() << flag);
    if (!probe.waitForFinished(5000)) {
        probe.kill();
        return {};
    }
    return QString::fromLocal8Bit(probe.readAllStandardOutput()).trimmed();
}

static bool probeFeature(const QString &exe, const QString &flag,
                         const QString &token) {
    return probe_feature_output(exe, flag).contains(token, Qt::CaseInsensitive);
}

void MainWindow::detectFeatureSupport() {
    const bool isAcmxvk = active_backend == acmx2::Backend::Acmxvk;
    const QString backendName = acmx2::backend_name(active_backend);
    const QString cudaOutput =
        probe_feature_output(executable_path, "--check-cuda");
    cuda_available = cudaOutput.contains(
        isAcmxvk ? "acidcam-gpu filters: enabled" : "CUDA: enabled",
        Qt::CaseInsensitive);
    cuda_device_available =
        isAcmxvk
            ? cudaOutput.contains("MXVK CUDA interop: enabled",
                                  Qt::CaseInsensitive)
            : cuda_available;
    audio_available = probeFeature(executable_path, "--check-audio", "AUDIO: enabled");
    midi_available = probeFeature(executable_path, "--check-midi", "MIDI: enabled");
    dnn_available = probeFeature(
        executable_path, "--check-dnn",
        isAcmxvk ? "OpenCV DNN effects: enabled" : "OpenCV DNN: enabled");

    Log(QString("CUDA filters: %1 (%2)")
            .arg(cuda_available ? "enabled" : "disabled", backendName));
    if (isAcmxvk) {
        Log(QString("CUDA device interop: %1 (%2)")
                .arg(cuda_device_available ? "enabled" : "disabled",
                     backendName));
    }
    Log(QString("AUDIO: %1 (%2)")
            .arg(audio_available ? "enabled" : "disabled", backendName));
    Log(QString("MIDI: %1 (%2)")
            .arg(midi_available ? "enabled" : "disabled", backendName));
    Log(QString("OpenCV DNN: %1 (%2)")
            .arg(dnn_available ? "enabled" : "disabled", backendName));

    if (!dnn_available) {
        onnx_model_enabled = false;
        onnx_model.clear();
    }

    if (gpuFilterAction) {
        gpuFilterAction->setEnabled(cuda_available);
        gpuFilterAction->setToolTip(
            cuda_available
                ? QString()
                : tr("Disabled: %1 was built without acidcam-gpu filter support.")
                      .arg(backendName));
    }
    if (!cuda_available) {
        gpu_filter_enabled = false;
        gpu_filter_indices.clear();
        if (!cuda_device_available)
            cuda_device = 0;
    }

    if (audioSet) {
        audioSet->setEnabled(audio_available);
        audioSet->setToolTip(
            audio_available
                ? QString()
                : tr("Disabled: %1 was built without audio support.")
                      .arg(backendName));
    }
    if (!audio_available) {
        audio_enabled = false;
        record_audio = false;
        audio_passthrough = false;
        audio_file.clear();
        audio_trunc = false;
        audio_repeat = false;
    }

    if (midiSettingsAction) {
        midiSettingsAction->setEnabled(midi_available);
        midiSettingsAction->setToolTip(
            midi_available
                ? QString()
                : tr("Disabled: %1 was built without MIDI support.")
                      .arg(backendName));
    }
    if (!midi_available) {
        midi_enabled = false;
        midi_config_file.clear();
        midi_device = -1;
    }
}
