#ifndef __MIDI_SETTINGS_H__
#define __MIDI_SETTINGS_H__

/**
 * @file midi-settings.hpp
 * @brief Dialog for enabling MIDI control and selecting mapping/device settings.
 */

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

/**
 * @brief MIDI settings dialog used by the main launcher UI.
 */
class MidiSettings : public QDialog {
    Q_OBJECT

  public:
    explicit MidiSettings(const QString &executablePath, QWidget *parent = nullptr);

    /// @brief Return whether MIDI support is enabled.
    bool isMidiEnabled() const;
    /// @brief Return path to the active MIDI mapping config file.
    QString getMidiConfigFile() const;
    /// @brief Return selected MIDI input device index.
    int getMidiDeviceIndex() const;

  private slots:
    void browseConfigFile();
    void launchMidiMapTool();
    void refreshDevices();

  private:
    void populateMidiDevices();
    void loadUiState();
    void saveUiState();

    QCheckBox *enableCheckBox;
    QLineEdit *configFileEdit;
    QPushButton *browseButton;
    QPushButton *launchMapToolButton;
    QComboBox *deviceComboBox;
    QPushButton *refreshButton;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QString execPath;
};

#endif
