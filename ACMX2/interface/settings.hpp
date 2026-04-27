#ifndef SETTINGS_HPP
#define SETTINGS_HPP

/**
 * @file settings.hpp
 * @brief Main capture/playback settings dialog for ACMX2 execution.
 */

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMap>
#include <QPushButton>
#include <QRadioButton>
#include <QSet>
#include <QSpinBox>
#include <QVBoxLayout>

/**
 * @brief Dialog that collects camera, input source, output, and runtime options.
 */
class SettingsWindow : public QDialog {
    Q_OBJECT
  public:
    /// @brief Construct the settings dialog.
    /// @param execPath Path to ACMX2 executable used for capability discovery.
    /// @param parent Parent widget.
    explicit SettingsWindow(const QString &execPath, QWidget *parent = nullptr);
    /// @brief Show or hide CUDA-specific controls based on availability.
    /// @param available True when CUDA runtime/device support is available.
    void setCudaAvailable(bool available);
    /// @return Selected camera device index.
    int getSelectedCameraIndex() const;
    /// @return Selected input camera resolution.
    QSize getSelectedCameraResolution() const;
    /// @return Selected output screen resolution.
    QSize getSelectedScreenResolution() const;
    /// @return Selected camera capture FPS.
    int getCameraFPS() const;
    /// @return Input video file path.
    QString getInputVideoFile() const;
    /// @return Output video file path.
    QString getOutputVideoFile() const;
    /// @return Graphics/video overlay file path.
    QString getGraphicsFile() const;
    /// @return True if input-video mode is selected.
    bool isUsingInputVideoFile() const;
    /// @return True if graphics-file mode is selected.
    bool isUsingGraphicsFile() const;
    /// @return True if save-to-file is enabled.
    bool isSavingToOutputVideoFile() const;
    /// @return True if HDR was detected in the currently selected input video.
    bool isInputHdrDetected() const;
    /// @return True if the user enabled the post-process HDR10 conversion.
    bool isConvertToHdr10Enabled() const;
    /// @return True if texture cache is enabled.
    bool isTextureCacheEnabled() const;
    /// @return Frame delay used by texture cache.
    int getCacheDelay() const;
    /// @return True if fullscreen mode is enabled.
    bool isFullscreen() const;
    /// @return True if input audio should be copied to output.
    bool isCopyAudioEnabled() const;
    /// @return True if 3D rendering mode is enabled.
    bool is3dEnabled() const;
    /// @return True if YUV mode is enabled for selected resolution.
    bool isUseYuvEnabled() const;
    /// @return Selected 3D model file path.
    QString getModelFile() const;
    /// @return Selected CUDA device index.
    int getSelectedCudaDevice() const;
    /// @return Time speed multiplier applied at runtime.
    float getTimeSpeed() const;
    /// @brief Return whether maximum duration limiting is enabled.
    bool isDurationLimitEnabled() const;
    /// @brief Return configured max run duration in seconds.
    double getDurationLimit() const;
    /// @brief Return crossfade duration between shader transitions.
    float getCrossFadeDuration() const;
    /// @brief Return whether flip mode is enabled.
    bool isFlipEnabled() const;
    /// @brief Resolve display name for a camera device index.
    /// @param device_index Camera device index.
    /// @return Human-readable camera name.
    QString getCameraName(int device_index);

    /// @return Selected x264-style encoder preset name (ultrafast..veryslow).
    QString getEncodePreset() const;
    /// @return Selected encoder tune (empty for "none").
    QString getEncodeTune() const;
    /// @return Selected encoder CRF value (0..51).
    int getEncodeCrf() const;
    /// @return Selected encoder codec preference (auto/software/nvenc).
    QString getEncodeCodec() const;
    /// @return True if realtime low-latency encoding is enabled.
    bool isEncodeRealtime() const;
    /// @return True if no-drop encoder backpressure mode is enabled.
    bool isEncodeNoDrop() const;
  private slots:
    void acceptSettings();
    void rejectSettings();
    void browseInputVideoFile();
    void browseOutputVideoFile();
    void browseGraphicsFile();
    void browseModelFile();
    /// @brief Probe the currently selected input video file with ffprobe and
    ///        update HDR-related UI state (status label, HDR10 checkbox).
    void detectInputHdr();
    /// @brief React to camera-device combo selection changes.
    /// @param comboIndex Selected combo-box index.
    void onCameraDeviceChanged(int comboIndex);
    /// @brief React to camera-resolution selection changes.
    /// @param comboIndex Selected combo-box index.
    void onCameraResolutionChanged(int comboIndex);

  private:
    void init();
    void loadUiState();
    void saveUiState();
    void populateCameraDevices();
    void populateCudaDevices();
    /// @brief Query resolutions and FPS capabilities for one camera device.
    /// @param deviceIndex Camera device index to inspect.
    void enumerateDevice(int deviceIndex);
    void populateResolutions();
    void populateFPS();

    QString executablePath;
    QComboBox *cameraFPSComboBox;
    // resolution -> list of FPS values
    QMap<QString, QList<double>> deviceCapabilities;
    QSet<QString> yuvResolutions;

    QComboBox *cameraIndexComboBox;
    QComboBox *cameraResolutionComboBox;
    QComboBox *screenResolutionComboBox;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QLineEdit *inputVideoFileLineEdit;
    QPushButton *browseInputVideoButton;
    QLabel *hdrStatusLabel = nullptr;
    QCheckBox *convertHdr10CheckBox = nullptr;
    QLineEdit *outputVideoFileLineEdit;
    QPushButton *browseOutputVideoButton;
    QLineEdit *graphicsFileLineEdit;
    QPushButton *browseGraphicsButton;
    QRadioButton *cameraOptionRadioButton;
    QRadioButton *inputVideoOptionRadioButton;
    QRadioButton *graphicsFileOptionRadioButton;
    QCheckBox *saveOutputVideoCheckBox;
    QCheckBox *textureCacheCheckBox;
    QSpinBox *cacheDelaySpinBox;
    QCheckBox *fullscreenCheckBox;
    QCheckBox *copyAudioCheckBox;
    QCheckBox *enable3dCheckBox;
    QCheckBox *useYuvCheckBox;
    QLineEdit *modelFileLineEdit;
    QPushButton *browseModelButton;
    QComboBox *cudaDeviceComboBox;
    QLabel *cudaDeviceLabel = nullptr;
    QDoubleSpinBox *timeSpeedSpinBox;
    QCheckBox *durationLimitCheckBox;
    QDoubleSpinBox *durationLimitSpinBox;
    QDoubleSpinBox *crossFadeSpinBox;
    QCheckBox *flipCheckBox;

    QComboBox *encodePresetComboBox = nullptr;
    QComboBox *encodeTuneComboBox = nullptr;
    QSpinBox *encodeCrfSpinBox = nullptr;
    QComboBox *encodeCodecComboBox = nullptr;
    QCheckBox *encodeRealtimeCheckBox = nullptr;
    QCheckBox *encodeNoDropCheckBox = nullptr;

    // Responsive layout: groups are reflowed between 1 and 2 columns
    // depending on the dialog width when the user resizes the window.
    QHBoxLayout *groupsRow = nullptr;
    QVBoxLayout *leftColumn = nullptr;
    QVBoxLayout *rightColumn = nullptr;
    QList<QGroupBox *> reflowGroups;
    int currentColumnCount = 0;
    void reflowGroupColumns(int columns);

  protected:
    void resizeEvent(QResizeEvent *event) override;

  private:

    int selectedCameraIndex;
    QSize selectedCameraResolution;
    QSize selectedScreenResolution;
    int cameraFPS;
    QString inputVideoFile;
    QString outputVideoFile;
    bool useInputVideoFile;
    bool useGraphicsFile;
    bool saveOutputVideoFile;
    QString graphicsFile;
    int graphicsDuration;
    QString modelFile;
    int selectedCudaDevice;
    double maxDuration;
    bool inputHdrDetected = false;
};

#endif
