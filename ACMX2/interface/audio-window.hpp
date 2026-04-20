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
    bool isAudioFileEnabled() const;
    QString getAudioFilePath() const;

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
    QCheckBox *audioFileCheckBox;
    QLineEdit *audioFileLineEdit;
    QPushButton *audioFileBrowseButton;
    QPushButton *okButton;
    QPushButton *cancelButton;
};

#endif
