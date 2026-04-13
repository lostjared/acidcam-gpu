#ifndef __MIDI_SETTINGS_H__
#define __MIDI_SETTINGS_H__

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

class MidiSettings : public QDialog {
    Q_OBJECT

  public:
    explicit MidiSettings(const QString &executablePath, QWidget *parent = nullptr);

    bool isMidiEnabled() const;
    QString getMidiConfigFile() const;
    int getMidiDeviceIndex() const;

  private slots:
    void browseConfigFile();
    void launchMidiMapTool();
    void refreshDevices();

  private:
    void populateMidiDevices();

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
