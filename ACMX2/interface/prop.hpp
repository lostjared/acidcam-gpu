#ifndef PROP_HPP
#define PROP_HPP

/**
 * @file prop.hpp
 * @brief Dialog for selecting executable and resource directories.
 */
#include <QDialog>
#include <QDir>
#include <QFileDialog>
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
    explicit PropWindow(QWidget *parent = nullptr);

  private:
    void init();
    void selectExecutable();
    void selectShaderDirectory();
    void selectScreenshotDirectory();
    void restoreDefaults();
    QString getDefaultPicturesDirectory();

  public:
    QLineEdit *exePathLineEdit;
    QLineEdit *shaderDirLineEdit;
    QLineEdit *screenshotDirLineEdit;
};

#endif