#ifndef PROP_HPP
#define PROP_HPP

#include "backend.hpp"

/**
 * @file prop.hpp
 * @brief Dialog for selecting executable and resource directories.
 */
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>

/**
 * @brief Launcher properties dialog for executable/shader/screenshot paths.
 */
class PropWindow : public QDialog {
    Q_OBJECT
  public:
    explicit PropWindow(acmx2::Backend backend,
                        QWidget *parent = nullptr);

  private:
    void init();
    void selectExecutable();
    void selectShaderDirectory();
    void selectScreenshotDirectory();
    void selectShaderCompiler();
    void restoreDefaults();
    QString getDefaultPicturesDirectory();
    acmx2::Backend active_backend;

  public:
    QLineEdit *exePathLineEdit;
    QLineEdit *shaderDirLineEdit;
    QLineEdit *screenshotDirLineEdit;
    QComboBox *shaderCompilerComboBox = nullptr;
    QLineEdit *shaderCompilerPathLineEdit = nullptr;
};

#endif
#include "backend.hpp"
