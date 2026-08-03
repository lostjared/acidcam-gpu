#ifndef __SHADER_LIB_H__Y
#define __SHADER_LIB_H__Y

/**
 * @file shaderlibrary.hpp
 * @brief Dialog for generating a shader index from a selected folder.
 */

#include <QCheckBox>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTextStream>
#include <QVBoxLayout>

/**
 * @brief Shader-library utility dialog.
 */
class LibraryWindow : public QDialog {
    Q_OBJECT

  public:
    LibraryWindow(QWidget *parent = nullptr);

  private:
    QLineEdit *folderPathEdit;
    QPushButton *browseButton;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QCheckBox *createDefaultShaderCheckBox;
    QCheckBox *createJsonManifestCheckBox;
    void init();
    bool createShaderManifest(const QString &folderPath);

  public:
    /// @brief Return folder chosen for shader-library indexing.
    QString getShaderPath();

  private slots:
    void onBrowseButtonClicked();
    void onOkButtonClicked();
    void onCancelButtonClicked();

  private:
    QString path;
};

#endif
