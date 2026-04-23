/**
 * @file audio-window.hpp
 * @brief Qt6 dialog for configuring audio reactivity settings.
 *
 * Provides device selection (input / output), sensitivity, channel count,
 * pass-through, recording options, and an optional file-based audio source
 * that replaces live microphone input for audio-reactive shaders.
 */

#ifndef AUDIOSETTINGS_HPP
#define AUDIOSETTINGS_HPP

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
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
    /// @brief Check whether file-based audio input is selected and a path is set.
    /// @return @c true if the "Use Audio File" checkbox is checked and the path is non-empty.
    bool isAudioFileEnabled() const;
    /// @brief Get the path to the selected audio file.
    /// @return Absolute file path, or an empty string if none selected.
    QString getAudioFilePath() const;
    /// @brief Check whether "stop when audio ends" is selected.
    /// @return @c true if the audio-trunc checkbox is checked.
    bool isAudioTruncEnabled() const;

  private:
    void populateAudioDevices();

    QCheckBox *audioReactivityCheckBox;
    QCheckBox *audioPassThroughCheckBox;
    QCheckBox *recordAudioCheckBox;
    QSlider *recordVolumeSlider;
    QSpinBox *channelSpinBox;
    QSlider *sensitivitySlider;
    QComboBox *inputDeviceComboBox;
    QComboBox *outputDeviceComboBox;
    QCheckBox *audioFileCheckBox;          ///< Toggle file-based audio input.
    QLineEdit *audioFileLineEdit;          ///< Displays the selected audio file path.
    QPushButton *audioFileBrowseButton;    ///< Opens a file dialog to choose an audio file.
    QCheckBox *audioTruncCheckBox;         ///< Stop playback when the audio file ends.
    QPushButton *okButton;
    QPushButton *cancelButton;
};

#endif
