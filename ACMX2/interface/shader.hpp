#ifndef SHADER_H
#define SHADER_H

/**
 * @file shader.hpp
 * @brief Dialog for creating a new shader file.
 */

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTextStream>
#include <QVBoxLayout>

/**
 * @brief New-shader dialog with optional starter template content.
 */
class ShaderDialog : public QDialog {
    Q_OBJECT

  public:
    ShaderDialog(QWidget *parent = nullptr);
    /// @brief Set output directory used for generated shader files.
    void setShaderPath(const QString &path);

  private:
    QLineEdit *shaderNameEdit;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QCheckBox *defaultCodeCheckBox;
    QCheckBox *cacheShaderCheckBox;
    QComboBox *shaderTypeComboBox;
    QString shaderPath;

    void init();
    bool createShaderFile(const QString &shaderName, bool includeDefaultCode,
                          bool createCacheShader, bool createComputeShader);
  private slots:
    void onOkButtonClicked();
    void onCancelButtonClicked();
};

#endif
