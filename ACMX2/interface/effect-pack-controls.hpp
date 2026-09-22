#ifndef ACMX2_INTERFACE_EFFECT_PACK_CONTROLS_HPP
#define ACMX2_INTERFACE_EFFECT_PACK_CONTROLS_HPP

#include <QDialog>
#include <QJsonObject>
#include <QString>
#include <QVector>

class QVBoxLayout;
class QWidget;
class QTimer;

struct EffectPackControlDefinition {
    QString id;
    QString label;
    QString uniform;
    double minimum = 0.0;
    double maximum = 1.0;
    double step = 0.01;
    double default_value = 0.0;
};

struct EffectPackUniformValue {
    QString name;
    double value = 0.0;
};

class EffectPackControls : public QDialog {
    Q_OBJECT

  public:
    explicit EffectPackControls(QWidget *parent = nullptr);
    ~EffectPackControls() override;
    void set_pack(const QString &id, const QString &name, const QVector<EffectPackControlDefinition> &definitions, bool persist_user_values = true);
    void set_project_values(const QJsonObject &values);
    QJsonObject project_values() const;
    QVector<EffectPackUniformValue> values() const;

  signals:
    void values_changed(const QVector<EffectPackUniformValue> &values);

  private:
    void rebuild();
    void set_value(int index, double value);
    void save_values() const;

    QString pack_id;
    QString pack_name;
    QVector<EffectPackControlDefinition> controls;
    QVector<double> control_values;
    bool persist_user_values = true;
    QVBoxLayout *rows_layout = nullptr;
    QWidget *rows_widget = nullptr;
    QTimer *save_timer = nullptr;
};

#endif
