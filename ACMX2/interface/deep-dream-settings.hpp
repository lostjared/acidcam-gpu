#ifndef DEEP_DREAM_SETTINGS_HPP
#define DEEP_DREAM_SETTINGS_HPP

#include <QDialog>
#include <QString>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QPushButton;
class QSpinBox;

struct DeepDreamConfiguration {
    bool enabled = false;
    QString model_file;
    QString layer = "relu4_2";
    int iterations = 1;
    double strength = 0.05;
    double feedback = 0.9;
    double zoom = 1.01;
    double rotation = 0.1;
    int maximum_dimension = 512;
    bool fp16 = false;
    int channel = -1;
    int octaves = 1;
    double octave_scale = 1.4;
    int jitter = 0;
    int smoothing = 0;
    bool gpu_filter_first = false;
};

class DeepDreamSettingsDialog : public QDialog {
    Q_OBJECT

  public:
    explicit DeepDreamSettingsDialog(bool gpu_filter_enabled,
                                     QWidget *parent = nullptr);
    [[nodiscard]] DeepDreamConfiguration configuration() const;

  private slots:
    void browse_model();
    void accept_settings();

  private:
    void load_ui_state();
    void save_ui_state();
    void update_enabled_state();

    bool gpu_filter_available = false;
    QCheckBox *enable_check_box = nullptr;
    QLineEdit *model_file_edit = nullptr;
    QPushButton *browse_model_button = nullptr;
    QComboBox *layer_combo_box = nullptr;
    QSpinBox *iterations_spin_box = nullptr;
    QDoubleSpinBox *strength_spin_box = nullptr;
    QDoubleSpinBox *feedback_spin_box = nullptr;
    QDoubleSpinBox *zoom_spin_box = nullptr;
    QDoubleSpinBox *rotation_spin_box = nullptr;
    QCheckBox *native_size_check_box = nullptr;
    QSpinBox *maximum_dimension_spin_box = nullptr;
    QCheckBox *fp16_check_box = nullptr;
    QSpinBox *channel_spin_box = nullptr;
    QSpinBox *octaves_spin_box = nullptr;
    QDoubleSpinBox *octave_scale_spin_box = nullptr;
    QSpinBox *jitter_spin_box = nullptr;
    QSpinBox *smoothing_spin_box = nullptr;
    QCheckBox *gpu_filter_first_check_box = nullptr;
};

#endif
