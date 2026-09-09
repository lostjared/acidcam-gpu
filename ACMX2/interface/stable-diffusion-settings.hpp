#ifndef STABLE_DIFFUSION_SETTINGS_HPP
#define STABLE_DIFFUSION_SETTINGS_HPP

#include <QDialog>
#include <QString>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QPushButton;
class QSpinBox;

struct StableDiffusionConfiguration {
    bool enabled = false;
    QString model_file;
    QString upscale_model_file;
    QString prompt;
    QString negative_prompt;
    QString server_executable = "sd-server";
    int server_port = 1234;
    int width = 576;
    int height = 320;
    int steps = 12;
    double strength = 0.35;
    double cfg_scale = 5.0;
    int seed = 1234;
    QString sampler = "euler_a";
    QString scheduler = "discrete";
    bool upscale = false;
};

class StableDiffusionSettingsDialog : public QDialog {
    Q_OBJECT

  public:
    explicit StableDiffusionSettingsDialog(QWidget *parent = nullptr);
    [[nodiscard]] StableDiffusionConfiguration configuration() const;

  signals:
    void settingsApplied();

  private slots:
    void browse_model();
    void browse_upscale_model();
    void browse_server();
    void apply_settings();
    void accept_settings();

  private:
    [[nodiscard]] bool validate_settings();
    void load_ui_state();
    void save_ui_state();
    void update_enabled_state();

    QCheckBox *enable_check_box = nullptr;
    QLineEdit *model_file_edit = nullptr;
    QPushButton *browse_model_button = nullptr;
    QLineEdit *prompt_edit = nullptr;
    QLineEdit *negative_prompt_edit = nullptr;
    QLineEdit *server_edit = nullptr;
    QPushButton *browse_server_button = nullptr;
    QSpinBox *server_port_spin_box = nullptr;
    QComboBox *resolution_combo_box = nullptr;
    QSpinBox *steps_spin_box = nullptr;
    QDoubleSpinBox *strength_spin_box = nullptr;
    QDoubleSpinBox *cfg_scale_spin_box = nullptr;
    QSpinBox *seed_spin_box = nullptr;
    QComboBox *sampler_combo_box = nullptr;
    QComboBox *scheduler_combo_box = nullptr;
    QCheckBox *upscale_check_box = nullptr;
    QCheckBox *server_upscale_check_box = nullptr;
    QLineEdit *upscale_model_edit = nullptr;
    QPushButton *browse_upscale_model_button = nullptr;
};

#endif
