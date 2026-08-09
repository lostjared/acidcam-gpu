
#include "shaderlibrary.hpp"
#include "custom_style.hpp"
#include "shader-manifest.hpp"
#include <QDir>
#include <QFile>
#include <QSettings>
#include <QTextStream>

LibraryWindow::LibraryWindow(QWidget *parent) : QDialog(parent) {
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
        if (!dir.mkpath(folderPath) || !createShaderManifest(folderPath))
            return;
        path = folderPath;

        if (createDefaultShaderCheckBox->isChecked()) {
            QFile file(folderPath + "/default.glsl");
            if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QTextStream out(&file);
                out << defaultFile << "\n";
                file.close();
            }
        } else {

            QFile file(folderPath + "/default.glsl");
            if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QTextStream out(&file);
                out << "\n";
                file.close();
            }
        }

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
    const auto format = createJsonManifestCheckBox->isChecked()
                            ? acmx2::ShaderManifestFormat::Json
                            : acmx2::ShaderManifestFormat::Text;
    if (!acmx2::create_shader_manifest(folderPath, format, {"default.glsl"}, error)) {
        QMessageBox::critical(this, "Error", error);
        return false;
    }
    return true;
}
