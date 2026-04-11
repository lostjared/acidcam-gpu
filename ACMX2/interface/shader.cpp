#include "shader.hpp"
#include <QSettings>

ShaderDialog::ShaderDialog(QWidget *parent) : QDialog(parent) {
    init();
}

void ShaderDialog::init() {
    QVBoxLayout *layout = new QVBoxLayout(this);

    QLabel *instructionLabel = new QLabel("Enter the name of the shader file:", this);
    layout->addWidget(instructionLabel);

    shaderNameEdit = new QLineEdit(this);
    shaderNameEdit->setPlaceholderText("Shader name (e.g., myshader.glsl)");
    layout->addWidget(shaderNameEdit);

    defaultCodeCheckBox = new QCheckBox("Include default shader code", this);
    layout->addWidget(defaultCodeCheckBox);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    okButton = new QPushButton("OK", this);
    connect(okButton, &QPushButton::clicked, this, &ShaderDialog::onOkButtonClicked);
    buttonLayout->addWidget(okButton);

    cancelButton = new QPushButton("Cancel", this);
    connect(cancelButton, &QPushButton::clicked, this, &ShaderDialog::onCancelButtonClicked);
    buttonLayout->addWidget(cancelButton);

    layout->addLayout(buttonLayout);

    setLayout(layout);
    setWindowTitle("Create New Shader");
    resize(400, 150);
    QString style = "QMainWindow, QDialog { background-color: black; border: 3px solid red; }"
                    "* { color: red; font-weight: bold; } "
                    "QPushButton { border: 1px solid red; background-color: #110000; padding: 5px; }"
                    "QPushButton:hover { background-color: red; color: black; }";

    QSettings appSettings("LostSideDead");
    if (appSettings.value("useCustomStyle", true).toBool()) {
        setStyleSheet(style);
    }
}

void ShaderDialog::onOkButtonClicked() {
    QString shaderName = shaderNameEdit->text().trimmed();
    if (shaderName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a shader name.");
        return;
    }

    if (!shaderName.contains(".glsl")) {
        shaderName += ".glsl";
    }

    bool includeDefaultCode = defaultCodeCheckBox->isChecked();
    createShaderFile(shaderPath + "/" + shaderName, includeDefaultCode);
    QFile file(shaderPath + "/index.txt");
    if (file.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&file);
        out << shaderName << "\n";
        file.close();
    }
    QMessageBox::information(this, "Success", "Shader file created successfully.");
    accept();
}

void ShaderDialog::setShaderPath(const QString &path) {
    shaderPath = path;
}

void ShaderDialog::onCancelButtonClicked() {
    reject();
}

const char *defaultShFile = R"(#version 330 core
in vec2 tc;
out vec4 color;
uniform float time_f; // accumulated time value, affected by speed and audio when enabled
uniform sampler2D samp; // input video frame texture
uniform vec2 iResolution; // viewport resolution in pixels (width, height)
uniform vec4 iMouse; // mouse position: xy = current, zw = click start (drag)
uniform float amp; // audio amplitude scaled by sensitivity
uniform float uamp; // raw audio amplitude before sensitivity scaling
uniform float iTime; // elapsed wall-clock time in seconds
uniform int iFrame; // current frame number
uniform float iTimeDelta; // time since last frame in seconds
uniform vec4 iDate; // current date/time: (year, month, day, seconds since midnight)
uniform vec2 iMouseClick; // position of last mouse click
uniform float iFrameRate; // target frame rate
uniform vec3 iChannelResolution[4]; // resolution of each texture channel
uniform float iChannelTime[4]; // playback time for each texture channel
uniform float iSampleRate; // audio sample rate in Hz (e.g. 44100)
uniform float amp_peak; // peak absolute sample value in current audio buffer
uniform float amp_rms; // RMS energy of current audio buffer
uniform float amp_smooth; // exponentially smoothed amplitude for gradual transitions
uniform float amp_low; // bass energy (below ~300 Hz)
uniform float amp_mid; // mid-range energy (~300-3000 Hz)
uniform float amp_high; // treble energy (above ~3000 Hz)
uniform float iamp; // estimated dominant frequency in Hz via zero-crossing rate

void main(void) {
    color = texture(samp, tc);
}
    
)";

void ShaderDialog::createShaderFile(const QString &shaderName, bool includeDefaultCode) {
    QFile file(shaderName);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        if (includeDefaultCode) {
            out << defaultShFile << "\n";
            ;
        }
        file.close();
    } else {
        QMessageBox::critical(this, "Error", "Failed to create shader file.");
    }
}
