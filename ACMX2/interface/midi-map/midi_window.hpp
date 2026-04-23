#ifndef __MIDI_WINDOW_H__
#define __MIDI_WINDOW_H__

/**
 * @file midi_window.hpp
 * @brief MIDI mapping editor window for action-to-message binding.
 */

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

/**
 * @brief Single MIDI action binding entry.
 */
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

/**
 * @brief Main window for creating and testing MIDI action mappings.
 */
class MidiMapWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MidiMapWindow(QWidget *parent = nullptr);
    ~MidiMapWindow() override;

private slots:
    /// @brief Refresh available MIDI input devices.
    void refreshDevices();
    /// @brief Open selected MIDI input device.
    void openDevice(int index);
    /// @brief Start capture mode for the currently selected mapping row.
    void captureMapping();
    /// @brief Clear mapping bytes for the selected row.
    void clearMapping();
    /// @brief Save mappings to a config file.
    void saveConfig();
    /// @brief Load mappings from a config file.
    void loadConfig();
    /// @brief Poll RtMidi input and apply capture logic.
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
