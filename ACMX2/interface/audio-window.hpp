/**
 * @file audio-window.hpp
 * @brief Qt6 dialog for configuring audio reactivity settings.
 *
 * Provides device selection (input / output), sensitivity, channel count,
 * pass-through/file playback, recording options, and an optional file-based
 * audio source that replaces live microphone input for audio-reactive shaders.
 */

#ifndef AUDIOSETTINGS_HPP
#define AUDIOSETTINGS_HPP

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

/**
 * @brief Dialog for configuring live and file-based audio reactivity options.
 */
class AudioSettings : public QDialog {
    Q_OBJECT

  public:
    explicit AudioSettings(QWidget *parent = nullptr);

    bool isAudioReactivityEnabled() const;
    bool isAudioPassThroughEnabled() const;
    bool isRecordAudioEnabled() const;
    double getRecordVolume() const;
    int getNumberOfChannels() const;
    double getSensitivity() const;
    int getInputDeviceIndex() const;
    int getOutputDeviceIndex() const;
    /// @brief Check whether a file or M3U audio source is selected and has a path.
    /// @return @c true when either mutually exclusive source is ready.
    bool isAudioFileEnabled() const;
    /// @brief Get the effective audio file or M3U playlist path.
    /// @return The selected source path, or an empty string if none is selected.
    QString getAudioFilePath() const;
    /// @brief Check whether the M3U playlist overrides the single audio file.
    bool isAudioPlaylistEnabled() const;
    /// @brief Get the selected M3U playlist path.
    QString getAudioPlaylistPath() const;
    /// @brief Check whether "stop when audio ends" is selected.
    /// @return @c true if the audio-trunc checkbox is checked.
    bool isAudioTruncEnabled() const;
    /// @brief Check whether file audio should restart when it reaches the end.
    /// @return @c true if the audio-repeat checkbox is checked.
    bool isAudioRepeatEnabled() const;
    /// @brief Check whether spectrum history buffers are enabled.
    /// @return @c true when --enable-audio-buffers should be emitted.
    bool isAudioBuffersEnabled() const;
    /// @brief Number of spectrum history frames requested.
    /// @return Frame count for --enable-audio-buffers.
    int getAudioBufferFrames() const;
    /// @brief Audio startup warmup rate in 1/sec.
    /// @return Warmup slope where 0.5 ~= 2 seconds to full intensity.
    double getAudioWarmRate() const;

  private:
    void populateAudioDevices();
    void loadUiState();
    void saveUiState();
    void updateAudioSourceControls();

    QCheckBox *audioReactivityCheckBox;
    QCheckBox *audioPassThroughCheckBox;
    QCheckBox *recordAudioCheckBox;
    QSlider *recordVolumeSlider;
    QSpinBox *channelSpinBox;
    QSlider *sensitivitySlider;
    QComboBox *inputDeviceComboBox;
    QComboBox *outputDeviceComboBox;
    QCheckBox *audioFileCheckBox;           ///< Toggle file-based audio input.
    QLineEdit *audioFileLineEdit;           ///< Displays the selected audio file path.
    QPushButton *audioFileBrowseButton;     ///< Opens a file dialog to choose an audio file.
    QCheckBox *audioPlaylistCheckBox;       ///< Use an M3U playlist instead of the audio file.
    QLineEdit *audioPlaylistLineEdit;       ///< Displays the selected M3U playlist path.
    QPushButton *audioPlaylistBrowseButton; ///< Opens a file dialog to choose an M3U playlist.
    QPushButton *audioPlaylistEditButton;   ///< Opens the M3U playlist editor.
    QCheckBox *audioTruncCheckBox;          ///< Stop playback when the audio file ends.
    QCheckBox *audioRepeatCheckBox;         ///< Restart playback when the audio file ends.
    QCheckBox *audioBuffersCheckBox;        ///< Enable spectrum history buffer CLI option.
    QSpinBox *audioBuffersSpinBox;          ///< Number of spectrum history frames.
    QDoubleSpinBox *audioWarmRateSpinBox;   ///< Startup warmup rate for audio intensity ramp.
    QPushButton *okButton;
    QPushButton *cancelButton;
};

#endif
