
#include "shaderlibrary.hpp"
#include "acmxvk-source-manifest.hpp"
#include "custom_style.hpp"
#include "shader-manifest.hpp"
#include <QDir>
#include <QFile>
#include <QSettings>
#include <QTextStream>

LibraryWindow::LibraryWindow(acmx2::Backend selectedBackend, QWidget *parent)
    : QDialog(parent), backend(selectedBackend) {
    init();
}

void LibraryWindow::init() {
    QVBoxLayout *layout = new QVBoxLayout(this);

    QLabel *instructionLabel = new QLabel("Select a folder to create a shader index file:", this);
    layout->addWidget(instructionLabel);

    folderPathEdit = new QLineEdit(this);
    folderPathEdit->setPlaceholderText("Folder path");
    layout->addWidget(folderPathEdit);

    browseButton = new QPushButton("Browse", this);
    connect(browseButton, &QPushButton::clicked, this, &LibraryWindow::onBrowseButtonClicked);
    layout->addWidget(browseButton);

    createDefaultShaderCheckBox = new QCheckBox("Create default shader", this);
    layout->addWidget(createDefaultShaderCheckBox);

    createJsonManifestCheckBox = new QCheckBox("Use library.json manifest", this);
    createJsonManifestCheckBox->setToolTip(
        "Store the shader list as JSON. Existing libraries continue to support index.txt.");
    layout->addWidget(createJsonManifestCheckBox);
    if (backend == acmx2::Backend::Acmxvk) {
        createJsonManifestCheckBox->setChecked(true);
        createJsonManifestCheckBox->setEnabled(false);
        createJsonManifestCheckBox->setToolTip(
            "ACMXVK source libraries always use library.json.");
    }

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    okButton = new QPushButton("OK", this);
    connect(okButton, &QPushButton::clicked, this, &LibraryWindow::onOkButtonClicked);
    buttonLayout->addWidget(okButton);

    cancelButton = new QPushButton("Cancel", this);
    connect(cancelButton, &QPushButton::clicked, this, &LibraryWindow::onCancelButtonClicked);
    buttonLayout->addWidget(cancelButton);

    layout->addLayout(buttonLayout);

    setLayout(layout);
    setWindowTitle("Shader Library Folder Selector");
    resize(400, 200);
    acmx2::applyCustomStyleIfEnabled(this);
}

void LibraryWindow::onBrowseButtonClicked() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastLibraryDir", "").toString();
    QString folderPath = QFileDialog::getExistingDirectory(this, "Select Folder", lastDir);
    if (!folderPath.isEmpty()) {
        appSettings.setValue("lastLibraryDir", folderPath);
        folderPathEdit->setText(folderPath + "/shaders");
    }
}

const char *defaultFile = R"(#version 330 core
in vec2 tc;
out vec4 color;

uniform sampler2D samp;

void main() {
    color = texture(samp, tc);
}
)";

const char *defaultAcmxvkFile = R"(#version 450

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D samp;

layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;

void main() {
    color = texture(samp, tc);
}
)";

void LibraryWindow::onOkButtonClicked() {
    QString folderPath = folderPathEdit->text().trimmed();

    if (folderPath.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select a folder.");
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "Confirm",
        QString("Do you want to create a shader library in the folder: %1?").arg(folderPath),
        QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        QDir dir;
        if (!dir.mkpath(folderPath)) {
            QMessageBox::critical(this, "Error",
                                  "Failed to create the shader library directory.");
            return;
        }
        path = folderPath;

        const QString defaultName =
            backend == acmx2::Backend::Acmxvk ? QStringLiteral("default.frag")
                                              : QStringLiteral("default.glsl");
        QFile file(QDir(folderPath).filePath(defaultName));
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::critical(
                this, "Error",
                QString("Failed to create %1: %2")
                    .arg(defaultName, file.errorString()));
            return;
        }
        QTextStream out(&file);
        if (createDefaultShaderCheckBox->isChecked())
            out << (backend == acmx2::Backend::Acmxvk ? defaultAcmxvkFile
                                                      : defaultFile);
        out << "\n";
        file.close();

        if (!createShaderManifest(folderPath))
            return;

        QMessageBox::information(this, "Success", "Shader library created successfully.");
        accept();
    }
}

QString LibraryWindow::getShaderPath() {
    return path;
}

void LibraryWindow::onCancelButtonClicked() {
    reject();
}

bool LibraryWindow::createShaderManifest(const QString &folderPath) {
    QString error;
    if (backend == acmx2::Backend::Acmxvk) {
        acmx2::AcmxvkSourceManifestResult result;
        if (!acmx2::create_acmxvk_source_manifest(folderPath, QString(), result,
                                                  error)) {
            QMessageBox::critical(this, "Error", error);
            return false;
        }
        return true;
    }
    const auto format = createJsonManifestCheckBox->isChecked()
                            ? acmx2::ShaderManifestFormat::Json
                            : acmx2::ShaderManifestFormat::Text;
    if (!acmx2::create_shader_manifest(folderPath, format, {"default.glsl"}, error)) {
        QMessageBox::critical(this, "Error", error);
        return false;
    }
    return true;
}
