#include "prop.hpp"
#include "custom_style.hpp"
#include <QFileInfo>
#include <QMainWindow>
#include <QSettings>
#include <QStandardPaths>

PropWindow::PropWindow(QWidget *parent) : QDialog(parent) {
    init();
}

void PropWindow::init() {
    setWindowTitle("Properties");
    setFixedSize(400, 300);

    QLabel *exeLabel = new QLabel("Program Executable:");
    exePathLineEdit = new QLineEdit(this);
#ifdef _WIN32
    exePathLineEdit->setText("acmx2.exe");
#else
    exePathLineEdit->setText("acmx2");
#endif
    exePathLineEdit->setReadOnly(true);
    QPushButton *exeBrowseButton = new QPushButton("Browse");

    QLabel *shaderDirLabel = new QLabel("Shader Directory:");
    shaderDirLineEdit = new QLineEdit(this);
    shaderDirLineEdit->setReadOnly(true);
    QPushButton *shaderDirBrowseButton = new QPushButton("Browse");

    QLabel *screenshotDirLabel = new QLabel("Screenshot Directory:");
    screenshotDirLineEdit = new QLineEdit(this);
    screenshotDirLineEdit->setReadOnly(true);
    QPushButton *screenshotDirBrowseButton = new QPushButton("Browse");

    QPushButton *okButton = new QPushButton("OK");
    QPushButton *cancelButton = new QPushButton("Cancel");
    QPushButton *restoreDefaultsButton = new QPushButton("Restore Defaults");

    QHBoxLayout *exeLayout = new QHBoxLayout();
    exeLayout->addWidget(exePathLineEdit, 1);
    exeLayout->addWidget(exeBrowseButton);

    QHBoxLayout *shaderDirLayout = new QHBoxLayout();
    shaderDirLayout->addWidget(shaderDirLineEdit, 1);
    shaderDirLayout->addWidget(shaderDirBrowseButton);

    QHBoxLayout *screenshotDirLayout = new QHBoxLayout();
    screenshotDirLayout->addWidget(screenshotDirLineEdit, 1);
    screenshotDirLayout->addWidget(screenshotDirBrowseButton);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(restoreDefaultsButton);
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(exeLabel);
    mainLayout->addLayout(exeLayout);
    mainLayout->addWidget(shaderDirLabel);
    mainLayout->addLayout(shaderDirLayout);
    mainLayout->addWidget(screenshotDirLabel);
    mainLayout->addLayout(screenshotDirLayout);
    mainLayout->addStretch(1);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);

    connect(exeBrowseButton, &QPushButton::clicked, this, &PropWindow::selectExecutable);
    connect(shaderDirBrowseButton, &QPushButton::clicked, this, &PropWindow::selectShaderDirectory);
    connect(screenshotDirBrowseButton, &QPushButton::clicked, this, &PropWindow::selectScreenshotDirectory);
    connect(restoreDefaultsButton, &QPushButton::clicked, this, &PropWindow::restoreDefaults);
    connect(okButton, &QPushButton::clicked, this, [this]() {
        QSettings appSettings("LostSideDead");
        appSettings.setValue("exePath", exePathLineEdit->text());
        appSettings.setValue("shaders", shaderDirLineEdit->text());
        appSettings.setValue("prefix_path", screenshotDirLineEdit->text());
        accept();
    });
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    QString defaultPicturesDir = getDefaultPicturesDirectory();
    QSettings appSettings("LostSideDead");
#ifndef _WIN32
    QString filePath = appSettings.value("exePath", "acmx2").toString();
#else
    QString filePath = appSettings.value("exePath", "acmx2.exe").toString();
#endif
    QString shader = appSettings.value("shaders", "").toString();
    QString screenshotDir = appSettings.value("prefix_path", defaultPicturesDir).toString();
    exePathLineEdit->setText(filePath);
    shaderDirLineEdit->setText(shader);
    screenshotDirLineEdit->setText(screenshotDir);
    exePathLineEdit->setMinimumHeight(30);
    shaderDirLineEdit->setMinimumHeight(30);
    screenshotDirLineEdit->setMinimumHeight(30);
    exeBrowseButton->setMinimumHeight(30);
    shaderDirBrowseButton->setMinimumHeight(30);
    screenshotDirBrowseButton->setMinimumHeight(30);
    okButton->setMinimumHeight(30);
    cancelButton->setMinimumHeight(30);
    restoreDefaultsButton->setMinimumHeight(30);
    acmx2::applyCustomStyleIfEnabled(this);
}

QString PropWindow::getDefaultPicturesDirectory() {
    QStringList picturePaths = QStandardPaths::standardLocations(QStandardPaths::PicturesLocation);
    if (!picturePaths.isEmpty()) {
        QString picturesDir = picturePaths.first();
        QDir dir(picturesDir);
        if (dir.exists()) {
            return picturesDir;
        }
    }
    return ".";
}

void PropWindow::selectExecutable() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastExeDir", "").toString();
    QString filePath = QFileDialog::getOpenFileName(
        this, "Select Program Executable", lastDir, "Executable Files (*.exe);;All Files (*)");
    if (!filePath.isEmpty()) {
        appSettings.setValue("lastExeDir", QFileInfo(filePath).absolutePath());
        exePathLineEdit->setText(filePath);
    }
}

void PropWindow::selectShaderDirectory() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastShaderDir", "").toString();
    QString dirPath = QFileDialog::getExistingDirectory(
        this, "Select Shader Directory", lastDir, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dirPath.isEmpty()) {
        appSettings.setValue("lastShaderDir", dirPath);
        shaderDirLineEdit->setText(dirPath);
    }
}

void PropWindow::selectScreenshotDirectory() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastScreenshotDir", "").toString();
    QString dirPath = QFileDialog::getExistingDirectory(
        this, "Select Screenshot Directory", lastDir, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dirPath.isEmpty()) {
        appSettings.setValue("lastScreenshotDir", dirPath);
        screenshotDirLineEdit->setText(dirPath);
    }
}

void PropWindow::restoreDefaults() {
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Restore Defaults",
                                  "Are you sure you want to restore default settings?",
                                  QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        QString defaultPicturesDir = getDefaultPicturesDirectory();

#ifdef _WIN32
        exePathLineEdit->setText("acmx2.exe");
#else
        exePathLineEdit->setText("acmx2");
#endif
        shaderDirLineEdit->setText("");
        screenshotDirLineEdit->setText(defaultPicturesDir);

        QSettings appSettings("LostSideDead");
        appSettings.setValue("exePath", exePathLineEdit->text());
        appSettings.setValue("shaders", shaderDirLineEdit->text());
        appSettings.setValue("prefix_path", screenshotDirLineEdit->text());

        QMessageBox::information(this, "Defaults Restored", "Default settings have been restored.");
    }
}
