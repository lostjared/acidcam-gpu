// #define BUILD_BUNDLE
// uncomment above if building BUNDLE

#ifndef __APP_WINDOW_H_
#define __APP_WINDOW_H_

/**
 * @file main_window.hpp
 * @brief Main launcher window for ACMX2 shader selection and execution.
 */
#include "../shader_selection_shm.hpp"
#include "editor.hpp"
#include "gpufilter.hpp"
#include "midi-settings.hpp"
#include "playlist.hpp"
#include "prop.hpp"
#include "shader.hpp"
#include "shaderlibrary.hpp"
#include "shaderpass.hpp"
#include "version_info.hpp" //defines VERSION_INFO
#include <QDateTime>
#include <QHash>
#include <QMainWindow>
#include <QMenuBar>
#include <QPointer>
#include <QProcess>
#include <QSettings>
#include <QTextEdit>
#include <QTreeWidget>
#include <random>

class CustomUniformDialog;

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
    QString activeShaderManifestPath;
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
    void menuSetCurrentShader();
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
    void menuMetadataViewer();
    void menuWatermarkSettings();
    void menuCustomUniforms();
    void menuToggleDisplayFilter(bool checked);
    void openCustomStyleEditor();

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
        cleanupShaderSelectionSharedMemory();
        QMainWindow::closeEvent(event);
    }

  private:
    QTreeWidget *list_view;
    QStringList items;
    QTextEdit *bottomTextBox;
    /// @brief Repopulate the shader tree widget from the current `items` list,
    ///        recomputing Last Modified and Compile Health columns.
    void populateShaderTree();
    /// @brief Compile-health status for a single shader.
    enum class CompileHealth { Unknown,
                               Cached,
                               Failed,
                               Stale };
    /// @brief Cached map of shader stem -> failed flag for the current library.
    QHash<QString, bool> shaderCacheStatus;
    /// @brief Modification time of the shader cache file when last read.
    QDateTime shaderCacheMTime;
    /// @brief Refresh `shaderCacheStatus` from the on-disk shader cache.
    void refreshShaderCacheStatus();
    /// @brief Return the filename in the Name column for the current selection.
    QString currentShaderName() const;
    /// @brief Return the row index of the current selection, or -1.
    int currentShaderRow() const;
    /// @brief Select the row at @p row and scroll it into view.
    void selectShaderRow(int row);
    /// @brief Open or focus an editor for a shader source location.
    void openShaderEditor(const QString &filePath, int lineNumber = 1,
                          int columnNumber = 0, int matchLength = 0);
    QMenu *fileMenu = nullptr;
    QMenu *cameraMenu = nullptr;
    QMenu *playbackMenu = nullptr;
    QMenu *runMenu = nullptr;
    QMenu *listMenu = nullptr;
    QMenu *viewMenu = nullptr;
    QMenu *helpMenu = nullptr;
    QAction *fileMenu_prop = nullptr, *fileMenu_exit = nullptr;
    QAction *cameraSet = nullptr, *audioSet = nullptr;
    QAction *runMenu_select = nullptr, *runMenu_all = nullptr;
    QAction *runMenu_copyCommand = nullptr;
    QAction *play_repeat = nullptr, *play_stop = nullptr;
    QAction *listMenu_new = nullptr, *listMenu_shader = nullptr, *listMenu_remove = nullptr, *listMenu_set_current = nullptr, *listMenu_up = nullptr, *listMenu_down = nullptr, *listMenu_shuffle = nullptr, *listMenu_sort = nullptr;
    QAction *helpMenu_about = nullptr;
    QAction *listMenu_findNext = nullptr;
    QAction *listMenu_findInFiles = nullptr;
    QString lastSearchText;
    int lastFoundIndex = -1;
    QString executable_path;
    bool cuda_available = false;
    bool audio_available = false;
    bool midi_available = false;
    bool dnn_available = false;
    void detectCudaSupport();
    void detectFeatureSupport();
    QAction *listMenu_search = nullptr;
    QString shader_path;
    QProcess *process = nullptr;
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
    QString encode_parameters;
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
    int cache_size = 8;
    bool full_screen_value = false;
    bool copy_audio = false;
    bool enable_3d = false;
    bool onnx_model_enabled = false;
    QString onnx_model;
    int audio_input = -1;
    int audio_output = -1;
    QString audio_file;
    bool audio_trunc = false;
    bool audio_buffers_enabled = false;
    int audio_buffer_frames = 8;
    double audio_warm_rate = 0.5;
    QString model_file;
    bool gpu_filter_enabled = false;
    QString gpu_filter_indices;
    int gpu_buffer_size = 8;
    QAction *gpuFilterAction;
    QAction *shaderPassAction;
    QPointer<ShaderPassDialog> shaderPassDialog;
    QPointer<PlaylistDialog> playlistDialog;
    QAction *styleSheetAction;
    QAction *buildCacheAction;
    QAction *runFromCacheAction;
    QAction *recompileShadersAction;
    QAction *removeBrokenAction;
    QString baseAppStyleSheet;
    QString customStyleSheet;
    void applyMainViewStyles(bool customStyleEnabled);
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
    bool max_size_limit_enabled = false;
    double max_size_mb = 0.0;
    float cross_fade_duration = 0.5f;
    bool flip_enabled = false;
    bool rotate_enabled = false;
    QString rotation_mode = "clockwise";
    bool png_output = false;
    bool generate_enabled = false;
    int generate_interval = 30;
    bool watermark_enabled = false;
    QString watermark_text;
    int watermark_r = 255;
    int watermark_g = 0;
    int watermark_b = 150;
    bool display_filter_enabled = false;
    QAction *watermarkAction = nullptr;
    QAction *displayFilterAction = nullptr;
    bool midi_enabled = false;
    QString midi_config_file;
    int midi_device = -1;
    QAction *midiSettingsAction;
    QAction *stayOnTopAction;
    QAction *customUniformsAction = nullptr;
    CustomUniformDialog *customUniformDialog = nullptr;
    bool playlist_enabled = false;
    QStringList playlist_names;
    QList<QPair<QString, QStringList>> playlist_tree_data;
    QString playlist_file_path;
    int autopilot_frames = 0;
    bool autopilot_random = false;
    QAction *playlistAction;
    QString stderrBuffer;
    /// @brief True while a cache rebuild process started by menuBuildShaderCache
    ///        is running.
    bool cacheBuildInProgress = false;

    void initShaderSelectionSharedMemory();
    void publishSelectedShaderIndexToRunningProcess();
    void publishShaderReloadToRunningProcess(const QString &filePath);
    void publishMultipassShadersToRunningProcess();
    void publishRepeatStateToRunningProcess();
    void publishRuntimeSettingsToRunningProcess();
    void publishCustomUniformsToRunningProcess();
    void cleanupShaderSelectionSharedMemory();
#if defined(__linux__) || defined(__APPLE__)
    int shaderSelectionShmFd = -1;
    acmx2::ipc::ShaderSelectionShmData *shaderSelectionShm = nullptr;
    quint32 shaderSelectionSequence = 0;
    quint32 shaderReloadSequence = 0;
#endif
};

#endif
