#ifndef __SHADER_LIB_H__Y
#define __SHADER_LIB_H__Y

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
    void init();
    void createShaderIndexFile(const QString &folderPath);

  public:
    QString getShaderPath();

  private slots:
    void onBrowseButtonClicked();
    void onOkButtonClicked();
    void onCancelButtonClicked();

  private:
    QString path;
};

#endif
