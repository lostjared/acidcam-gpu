#include "prop.hpp"
#include "custom_style.hpp"
#include <QFileInfo>
#include <QMainWindow>
#include <QSettings>
#include <QStandardPaths>

PropWindow::PropWindow(acmx2::Backend backend, QWidget *parent)
    : QDialog(parent), active_backend(backend) {
    init();
}

void PropWindow::init() {
    setWindowTitle(tr("%1 Properties").arg(acmx2::backend_name(active_backend)));
    setFixedSize(520, active_backend == acmx2::Backend::Acmxvk ? 440 : 300);

    QLabel *exeLabel = new QLabel("Program Executable:");
    exePathLineEdit = new QLineEdit(this);
    exePathLineEdit->setText(acmx2::default_backend_executable(active_backend));
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

    QGroupBox *shaderCompilerGroup = nullptr;
    QPushButton *shaderCompilerBrowseButton = nullptr;
    if (active_backend == acmx2::Backend::Acmxvk) {
        shaderCompilerGroup = new QGroupBox("ACMXVK Shader Compiler", this);
        auto *shaderCompilerLayout = new QVBoxLayout(shaderCompilerGroup);
        shaderCompilerComboBox = new QComboBox(this);
        shaderCompilerComboBox->addItem("Automatic glslc", "auto");
        shaderCompilerComboBox->addItem(
            "Custom glslc-compatible executable", "custom");
        shaderCompilerPathLineEdit = new QLineEdit(this);
        shaderCompilerPathLineEdit->setPlaceholderText(
            "/path/to/VulkanSDK/bin/glslc");
        shaderCompilerBrowseButton = new QPushButton("Browse");
        auto *shaderCompilerPathLayout = new QHBoxLayout();
        shaderCompilerPathLayout->addWidget(shaderCompilerPathLineEdit, 1);
        shaderCompilerPathLayout->addWidget(shaderCompilerBrowseButton);
        auto *compilerHelp = new QLabel(
            "The custom executable must accept glslc command-line options. "
            "It is used by Build, Fix Build, and live shader reload.");
        compilerHelp->setWordWrap(true);
        shaderCompilerLayout->addWidget(shaderCompilerComboBox);
        shaderCompilerLayout->addLayout(shaderCompilerPathLayout);
        shaderCompilerLayout->addWidget(compilerHelp);
    }

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
    if (shaderCompilerGroup)
        mainLayout->addWidget(shaderCompilerGroup);
    mainLayout->addStretch(1);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);

    connect(exeBrowseButton, &QPushButton::clicked, this, &PropWindow::selectExecutable);
    connect(shaderDirBrowseButton, &QPushButton::clicked, this, &PropWindow::selectShaderDirectory);
    connect(screenshotDirBrowseButton, &QPushButton::clicked, this, &PropWindow::selectScreenshotDirectory);
    if (shaderCompilerBrowseButton) {
        connect(shaderCompilerBrowseButton, &QPushButton::clicked, this,
                &PropWindow::selectShaderCompiler);
        connect(shaderCompilerComboBox,
                QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this, shaderCompilerBrowseButton](int) {
                    const bool custom =
                        shaderCompilerComboBox->currentData().toString() ==
                        QStringLiteral("custom");
                    shaderCompilerPathLineEdit->setEnabled(custom);
                    shaderCompilerBrowseButton->setEnabled(custom);
                });
    }
    connect(restoreDefaultsButton, &QPushButton::clicked, this, &PropWindow::restoreDefaults);
    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    QString defaultPicturesDir = getDefaultPicturesDirectory();
    QSettings appSettings("LostSideDead");
    const QString legacyExecutable =
        active_backend == acmx2::Backend::Acmx2
            ? appSettings.value("exePath", acmx2::default_backend_executable(
                                               acmx2::Backend::Acmx2))
                  .toString()
            : acmx2::default_backend_executable(active_backend);
    QString filePath =
        appSettings
            .value(acmx2::backend_settings_key(active_backend, "executable"),
                   legacyExecutable)
            .toString();
    const QString legacyLibrary =
        active_backend == acmx2::Backend::Acmx2
            ? appSettings.value("shaders", "").toString()
            : QString();
    QString shader =
        appSettings
            .value(acmx2::backend_settings_key(active_backend, "library"),
                   legacyLibrary)
            .toString();
    QString screenshotDir = appSettings.value("prefix_path", defaultPicturesDir).toString();
    exePathLineEdit->setText(filePath);
    shaderDirLineEdit->setText(shader);
    screenshotDirLineEdit->setText(screenshotDir);
    if (shaderCompilerComboBox) {
        const QString compilerMode =
            appSettings
                .value(acmx2::backend_settings_key(
                           acmx2::Backend::Acmxvk, "shader_compiler_mode"),
                       "auto")
                .toString();
        const int compilerIndex =
            shaderCompilerComboBox->findData(compilerMode);
        shaderCompilerComboBox->setCurrentIndex(
            compilerIndex >= 0 ? compilerIndex : 0);
        shaderCompilerPathLineEdit->setText(
            appSettings
                .value(acmx2::backend_settings_key(
                    acmx2::Backend::Acmxvk, "shader_compiler_path"))
                .toString());
        const bool custom = compilerMode == QStringLiteral("custom");
        shaderCompilerPathLineEdit->setEnabled(custom);
        shaderCompilerBrowseButton->setEnabled(custom);
    }
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

void PropWindow::selectShaderCompiler() {
    QSettings appSettings("LostSideDead");
    const QString lastDir =
        appSettings.value("lastShaderCompilerDir", "").toString();
    const QString filePath = QFileDialog::getOpenFileName(
        this, "Select glslc-compatible Shader Compiler", lastDir,
        "Executable Files (*.exe);;All Files (*)");
    if (!filePath.isEmpty()) {
        appSettings.setValue("lastShaderCompilerDir",
                             QFileInfo(filePath).absolutePath());
        shaderCompilerPathLineEdit->setText(filePath);
    }
}

void PropWindow::restoreDefaults() {
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Restore Defaults",
                                  "Are you sure you want to restore default settings?",
                                  QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        QString defaultPicturesDir = getDefaultPicturesDirectory();

        exePathLineEdit->setText(
            acmx2::default_backend_executable(active_backend));
        shaderDirLineEdit->setText("");
        screenshotDirLineEdit->setText(defaultPicturesDir);
        if (shaderCompilerComboBox) {
            shaderCompilerComboBox->setCurrentIndex(0);
            shaderCompilerPathLineEdit->clear();
        }

        QSettings appSettings("LostSideDead");
        appSettings.setValue(
            acmx2::backend_settings_key(active_backend, "executable"),
            exePathLineEdit->text());
        appSettings.setValue(
            acmx2::backend_settings_key(active_backend, "library"),
            shaderDirLineEdit->text());
        if (active_backend == acmx2::Backend::Acmx2) {
            appSettings.setValue("exePath", exePathLineEdit->text());
            appSettings.setValue("shaders", shaderDirLineEdit->text());
        }
        appSettings.setValue("prefix_path", screenshotDirLineEdit->text());
        if (shaderCompilerComboBox) {
            appSettings.setValue(
                acmx2::backend_settings_key(
                    acmx2::Backend::Acmxvk, "shader_compiler_mode"),
                "auto");
            appSettings.remove(acmx2::backend_settings_key(
                acmx2::Backend::Acmxvk, "shader_compiler_path"));
        }

        QMessageBox::information(this, "Defaults Restored", "Default settings have been restored.");
    }
}
