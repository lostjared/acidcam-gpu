#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMap>
#include <QPushButton>
#include <QRadioButton>
#include <QSet>
#include <QSpinBox>
#include <QVBoxLayout>

class SettingsWindow : public QDialog {
    Q_OBJECT
  public:
    explicit SettingsWindow(const QString &execPath, QWidget *parent = nullptr);
    void setCudaAvailable(bool available);
    int getSelectedCameraIndex() const;
    QSize getSelectedCameraResolution() const;
    QSize getSelectedScreenResolution() const;
    int getCameraFPS() const;
    int getSaveFileKbps() const;
    QString getInputVideoFile() const;
    QString getOutputVideoFile() const;
    QString getGraphicsFile() const;
    bool isUsingInputVideoFile() const;
    bool isUsingGraphicsFile() const;
    bool isSavingToOutputVideoFile() const;
    bool isTextureCacheEnabled() const;
    int getCacheDelay() const;
    bool isFullscreen() const;
    bool isCopyAudioEnabled() const;
    bool is3dEnabled() const;
    bool isUseYuvEnabled() const;
    QString getModelFile() const;
    int getSelectedCudaDevice() const;
    float getTimeSpeed() const;
    bool isDurationLimitEnabled() const;
    double getDurationLimit() const;
    float getCrossFadeDuration() const;
    QString getCameraName(int device_index);
  private slots:
    void acceptSettings();
    void rejectSettings();
    void browseInputVideoFile();
    void browseOutputVideoFile();
    void browseGraphicsFile();
    void browseModelFile();
    void onCameraDeviceChanged(int comboIndex);
    void onCameraResolutionChanged(int comboIndex);

  private:
    void init();
    void populateCameraDevices();
    void populateCudaDevices();
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
    QSpinBox *saveFileKbpsSpinBox;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QLineEdit *inputVideoFileLineEdit;
    QPushButton *browseInputVideoButton;
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

    int selectedCameraIndex;
    QSize selectedCameraResolution;
    QSize selectedScreenResolution;
    int cameraFPS;
    int saveFileKbps;
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
};

#endif
