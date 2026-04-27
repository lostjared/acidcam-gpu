// #define BUILD_BUNDLE
// uncomment above if building BUNDLE

#ifndef __APP_WINDOW_H_
#define __APP_WINDOW_H_

/**
 * @file main_window.hpp
 * @brief Main launcher window for ACMX2 shader selection and execution.
 */
#include "editor.hpp"
#include "gpufilter.hpp"
#include "midi-settings.hpp"
#include "playlist.hpp"
#include "prop.hpp"
#include "shader.hpp"
#include "shaderlibrary.hpp"
#include "shaderpass.hpp"
#include "version_info.hpp" //defines VERSION_INFO
#include <QListView>
#include <QMainWindow>
#include <QMenuBar>
#include <QPointer>
#include <QProcess>
#include <QSettings>
#include <QStringListModel>
#include <QTextEdit>
#include <random>

/**
 * @brief Read-only list model used for shader list display.
 */
class ReadOnlyStringListModel : public QStringListModel {
    Q_OBJECT
  public:
    using QStringListModel::QStringListModel;
    Qt::ItemFlags flags(const QModelIndex &index) const override {
        return QStringListModel::flags(index) & ~Qt::ItemIsEditable;
    }
};

/**
 * @brief Primary ACMX2 desktop UI.
 *
 * Manages shader discovery, process launch arguments, and related option dialogs.
 */
class MainWindow : public QMainWindow {
    Q_OBJECT
  public:
    MainWindow(QWidget *parent = 0) : QMainWindow(parent) {
        initControls();
    }
    /// @brief Build menus, actions, widgets, and signal wiring.
    void initControls();
    /// @brief Append timestamped text to the UI log output.
    /// @param message Text to append to the launcher log.
    void Log(const QString &message);
    /// @brief Write raw text to the lower output pane.
    /// @param message Text block to display.
    void Write(const QString &message);
    /// @brief Load shader names from index/cache for the provided path.
    /// @param path Shader directory path to scan.
    /// @param force When true, bypass index timestamp checks and reload.
    /// @return true if shader list was loaded successfully.
    bool loadShaders(const QString &path, bool force = false);
    /// @brief Refresh shader index metadata timestamp.
    void updateIndex();
    QDateTime indexTimestamp;
  public slots:
    void fileOpenProp();
    void fileExit();
    void runSelected();
    void runAll();
    void copyCommand();
    void cameraSettings();
    /// @brief Handle shader list selection changes.
    /// @param i Selected model index.
    void listClicked(const QModelIndex &i);
    void newList();
    void newShader();
    void menuUp();
    void menuDown();
    void menuRemove();
    void menuAudioSettings();
    void menuSort();
    void menuShuffle();
    void menuSearch();
    void menuFindNext();
    void menuGPUFilterSettings();
    void menuShaderPassSettings();
    void menuPlaylistSettings();
    void menuBuildShaderCache();
    void menuRunFromCache();
    void menuRecompileShaders();
    void menuRemoveBroken();
    void menuMidiSettings();

  protected:
    /// @brief Add shader to list if it is valid and not already present.
    /// @param shaderName Candidate shader identifier.
    /// @return true if the shader was added.
    bool addShaderToList(const QString &shaderName);

    void closeEvent(QCloseEvent *event) override {
        if (process->state() == QProcess::Running) {
            process->terminate();
            if (!process->waitForFinished(10000)) {
                process->kill();
            }
        }
        if (hdr10Process && hdr10Process->state() == QProcess::Running) {
            hdr10Process->terminate();
            if (!hdr10Process->waitForFinished(10000)) {
                hdr10Process->kill();
            }
        }
        QMainWindow::closeEvent(event);
    }

  private:
    QListView *list_view;
    QStringList items;
    ReadOnlyStringListModel *model;
    QTextEdit *bottomTextBox;
    QMenu *fileMenu;
    QMenu *cameraMenu;
    QMenu *playbackMenu;
    QMenu *runMenu;
    QMenu *listMenu;
    QMenu *viewMenu;
    QMenu *helpMenu;
    QAction *fileMenu_prop, *fileMenu_exit;
    QAction *cameraSet, *audioSet;
    QAction *runMenu_select, *runMenu_all;
    QAction *runMenu_copyCommand = nullptr;
    QAction *play_repeat, *play_stop;
    QAction *listMenu_new, *listMenu_shader, *listMenu_remove, *listMenu_up, *listMenu_down, *listMenu_shuffle, *listMenu_sort;
    QAction *helpMenu_about;
    QAction *listMenu_findNext;
    QString lastSearchText;
    int lastFoundIndex = -1;
    QString executable_path;
    bool cuda_available = false;
    bool audio_available = false;
    bool midi_available = false;
    void detectCudaSupport();
    void detectFeatureSupport();
    QAction *listMenu_search;
    QString shader_path;
    QProcess *process;
    QProcess *hdr10Process = nullptr;
    bool convert_to_hdr10 = false;
    QSize camera_res, screen_res;
    unsigned int camera_index;
    QString video_file;
    QString graphics_file;
    QString prefix_path;
    QString output_file;
    double output_fps = 24.0f;
    QString encode_preset = "medium";
    QString encode_tune; // empty => "none"
    int encode_crf = 18;
    QString encode_codec = "auto";
    bool encode_realtime = false;
    bool encode_no_drop = false;
    /// @brief Join list items into a comma-separated argument string.
    /// @param lst Input list of values.
    /// @return Concatenated string for command-line usage.
    QString concatList(const QStringList lst);
    /// @brief Build acmx2 command-line arguments from current UI state.
    /// @param arguments Output list to populate with command-line tokens.
    /// @return true if arguments were built, false on user-facing error.
    bool buildRunArguments(QStringList &arguments);
    /// @brief Run ffmpeg to convert the just-produced acmx2 output (assumed
    ///        HLG HDR) into HDR10 (HEVC NVENC, BT.2020 / SMPTE2084) and pipe
    ///        ffmpeg's stdout/stderr to the main log window.
    void runHdr10Conversion();
    QVector<QPointer<TextEditor>> open_files;
    /// @brief Read an entire text file into memory.
    /// @param filePath Path to source file.
    /// @return File contents, or empty string on failure.
    QString readFileContents(const QString &filePath);
    /// @brief Normalize shader names for filesystem/process safety.
    /// @param name Raw shader name.
    /// @return Sanitized shader name.
    QString sanitizeShaderName(const QString &name);
    void cleanupClosedEditors();
    bool audio_enabled = false;
    unsigned int audio_channels = 2;
    float audio_sense = 0.25f;
    bool audio_passthrough = false;
    bool record_audio = false;
    double record_volume = 1.0;
    bool cache_enabled = false;
    int cache_delay = 1;
    bool full_screen_value = false;
    bool copy_audio = false;
    bool enable_3d = false;
    int audio_input = -1;
    int audio_output = -1;
    QString audio_file;
    bool audio_trunc = false;
    QString model_file;
    bool gpu_filter_enabled = false;
    QString gpu_filter_indices;
    int gpu_buffer_size = 8;
    QAction *gpuFilterAction;
    QAction *shaderPassAction;
    QAction *styleSheetAction;
    QAction *buildCacheAction;
    QAction *runFromCacheAction;
    QAction *recompileShadersAction;
    QAction *removeBrokenAction;
    QString customStyleSheet;
    /// @brief Apply or remove the custom stylesheet override.
    /// @param enable True to apply, false to revert.
    void applyCustomStyleSheet(bool enable);
    bool shader_pass_enabled = false;
    QStringList shader_pass_names;
    /// @brief Map selected shader-pass names back to numeric indices.
    /// @return Comma-separated list of indices used by CLI args.
    QString getShaderPassIndicesFromNames();
    int cuda_device = 0;
    float time_speed = 1.0f;
    bool use_shader_cache = true;
    bool use_yuv = false;
    bool duration_limit_enabled = false;
    double max_duration = 0.0;
    float cross_fade_duration = 0.5f;
    bool flip_enabled = false;
    bool midi_enabled = false;
    QString midi_config_file;
    int midi_device = -1;
    QAction *midiSettingsAction;
    QAction *stayOnTopAction;
    bool playlist_enabled = false;
    QStringList playlist_names;
    QList<QPair<QString, QStringList>> playlist_tree_data;
    QString playlist_file_path;
    int autopilot_frames = 0;
    QAction *playlistAction;
    QString stderrBuffer;
};

#endif
