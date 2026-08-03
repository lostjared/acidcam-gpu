#ifndef CUSTOM_UNIFORMS_HPP
#define CUSTOM_UNIFORMS_HPP

#include "shader-manifest.hpp"
#include <QDialog>

class QDoubleSpinBox;
class QLineEdit;
class QScrollArea;
class QTimer;
class QVBoxLayout;

class CustomUniformDialog : public QDialog {
    Q_OBJECT

  public:
    explicit CustomUniformDialog(QWidget *parent = nullptr);

    bool loadLibrary(const QString &directory, QString *error = nullptr);
    const QList<acmx2::CustomUniformDefinition> &uniforms() const;

  signals:
    void uniformsChanged();
    void uniformDefinitionsChanged();

  private slots:
    void addUniform();
    void savePendingChanges();

  private:
    void rebuildUniformRows();
    void removeUniform(const QString &name);
    int uniformIndex(const QString &name) const;
    bool saveUniforms(bool showError);

    QString libraryDirectory;
    QList<acmx2::CustomUniformDefinition> uniformDefinitions;
    QLineEdit *nameEdit = nullptr;
    QDoubleSpinBox *minimumSpin = nullptr;
    QDoubleSpinBox *maximumSpin = nullptr;
    QDoubleSpinBox *stepSpin = nullptr;
    QScrollArea *scrollArea = nullptr;
    QWidget *rowsWidget = nullptr;
    QVBoxLayout *rowsLayout = nullptr;
    QTimer *saveTimer = nullptr;
};

#endif
