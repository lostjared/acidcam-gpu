#ifndef __MIDI_WINDOW_H__
#define __MIDI_WINDOW_H__

#include <QMainWindow>
#include <QComboBox>
#include <QTableWidget>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QStatusBar>
#include <QTimer>
#include <rtmidi/RtMidi.h>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>

struct MidiMapping {
    std::string actionName;
    std::string description;
    int key1{};
    int key2{};
    bool captured{false};
    unsigned char byte0{};
    unsigned char byte1{};
    unsigned char byte2{};
};

class MidiMapWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MidiMapWindow(QWidget *parent = nullptr);
    ~MidiMapWindow() override;

private slots:
    void refreshDevices();
    void openDevice(int index);
    void captureMapping();
    void clearMapping();
    void saveConfig();
    void loadConfig();
    void pollMidi();

private:
    void setupUi();
    void applyStyleSheet();
    void populateActions();
    void updateTable();
    void setStatus(const QString &msg);

    QComboBox *deviceCombo{};
    QPushButton *refreshButton{};
    QPushButton *captureButton{};
    QPushButton *clearButton{};
    QPushButton *saveButton{};
    QPushButton *loadButton{};
    QLineEdit *fileEdit{};
    QTableWidget *table{};
    QLabel *midiMonitorLabel{};

    RtMidiIn *midiIn{};
    bool deviceOpen{false};
    bool capturing{false};
    int captureRow{-1};
    QTimer *pollTimer{};

    std::vector<MidiMapping> mappings;
};

#endif
