#ifndef ACMX2_INTERFACE_EFFECT_PACK_BROWSER_HPP
#define ACMX2_INTERFACE_EFFECT_PACK_BROWSER_HPP

#include "deep-dream-settings.hpp"
#include "effect-pack-controls.hpp"
#include "effect-pack-transfer.hpp"
#include <QByteArray>
#include <QDialog>
#include <QImage>
#include <QJsonObject>
#include <QStringList>
#include <QVector>

class QComboBox;
class QFutureWatcherBase;
class QLabel;
class QListWidget;
class QProcess;
class QPushButton;
class QProgressBar;

class EffectPackBrowser : public QDialog {
    Q_OBJECT

  public:
    explicit EffectPackBrowser(QWidget *parent = nullptr);
    void set_build_tools(const QString &executable, const QString &compiler, int parallel_jobs);
    void set_runtime_context(bool dream_supported, const QString &configured_model, bool audio_supported, bool midi_supported, bool midi_profile_selected);
    void set_session_snapshot(const QString &source_root, const QJsonObject &manifest);
    void refresh();
    bool has_active_pack() const;
    void clear_active_pack();

  signals:
    void activation_requested(const QString &manifest_path, const QVector<EffectPackUniformValue> &values, const DeepDreamConfiguration &dream);
    void uniform_values_changed(const QVector<EffectPackUniformValue> &values);

  private:
    struct PackEntry {
        QString id;
        QString name;
        QString description;
        QString manifest;
        QString status;
        QImage icon;
        QVector<EffectPackControlDefinition> controls;
        DeepDreamConfiguration dream;
        QString dream_model_id;
        bool dream_declared = false;
        bool valid = false;
    };

    static QVector<PackEntry> discover(const QStringList &roots, const QStringList &model_roots, bool dream_supported, bool audio_supported, bool midi_supported, bool midi_profile_selected);
    QStringList search_roots() const;
    void populate();
    void update_details(int row);
    void activate(int row);
    void start_build(int row);
    void set_busy(bool busy);
    void start_transfer(const acmx2::EffectPackTransferRequest &request);
    void create_from_session();
    void save_pack_as();
    void export_pack();
    void import_pack();

    QComboBox *root_combo = nullptr;
    QListWidget *pack_list = nullptr;
    QLabel *details = nullptr;
    QLabel *status = nullptr;
    QPushButton *build_button = nullptr;
    QPushButton *library_button = nullptr;
    QPushButton *controls_button = nullptr;
    QPushButton *choose_model_button = nullptr;
    QPushButton *refresh_button = nullptr;
    QPushButton *create_button = nullptr;
    QPushButton *save_button = nullptr;
    QPushButton *export_button = nullptr;
    QPushButton *import_button = nullptr;
    QProgressBar *progress = nullptr;
    QProcess *build_process = nullptr;
    EffectPackControls *control_dialog = nullptr;
    QVector<PackEntry> packs;
    QString executable_path;
    QString compiler_path;
    QString active_manifest;
    QString control_manifest;
    QString configured_dream_model;
    QString session_source_root;
    QJsonObject session_manifest;
    bool transferring = false;
    QString transfer_message;
    bool dream_supported = false;
    bool audio_supported = false;
    bool midi_supported = false;
    bool midi_profile_selected = false;
    int jobs = 2;
    int pending_row = -1;
    bool scanning = false;
    QByteArray build_output;
};

#endif
