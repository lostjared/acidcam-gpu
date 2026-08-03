#include "shader.hpp"
#include "custom_style.hpp"
#include "shader-manifest.hpp"
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
    defaultCodeCheckBox->setChecked(true);
    layout->addWidget(defaultCodeCheckBox);

    cacheShaderCheckBox = new QCheckBox("Create as cache shader (_cache.glsl)", this);
    cacheShaderCheckBox->setToolTip(
        "Create a texture-cache shader starter with frame-cache and spectrum-history sampling.");
    layout->addWidget(cacheShaderCheckBox);

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
    resize(440, 180);
    acmx2::applyCustomStyleIfEnabled(this);
}

void ShaderDialog::onOkButtonClicked() {
    QString shaderName = shaderNameEdit->text().trimmed();
    if (shaderName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a shader name.");
        return;
    }

    if (shaderName.endsWith(".glsl", Qt::CaseInsensitive)) {
        shaderName.chop(5);
    }

    if (shaderName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a shader name.");
        return;
    }

    const bool createCacheShader = cacheShaderCheckBox->isChecked();
    if (createCacheShader &&
        !shaderName.endsWith("_cache", Qt::CaseInsensitive)) {
        shaderName += "_cache";
    }
    shaderName += ".glsl";

    const bool includeDefaultCode = defaultCodeCheckBox->isChecked();
    if (!createShaderFile(shaderPath + "/" + shaderName, includeDefaultCode,
                          createCacheShader)) {
        return;
    }

    QString manifestError;
    if (!acmx2::append_shader_manifest(shaderPath, shaderName, manifestError)) {
        QFile::remove(shaderPath + "/" + shaderName);
        QMessageBox::critical(this, "Error", manifestError);
        return;
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

namespace {
    constexpr const char *DEFAULT_SHADER_HEADER = R"(#version 330 core
in vec2 tc;
out vec4 color;

// Video and timing
uniform sampler2D samp; // input video frame texture
uniform float alpha; // oscillating alpha value
uniform float time_f; // accumulated time value, affected by time_speed
uniform float time_speed; // rate of change applied to time_f
uniform float iTime; // elapsed wall-clock time in seconds
uniform float iTimeDelta; // time since the previous frame in seconds
uniform int iFrame; // current frame number
uniform float iFrameRate; // target frame rate

// Viewport, mouse, date, and channel information
uniform vec2 iResolution; // viewport resolution in pixels (width, height)
uniform vec4 iMouse; // mouse position: xy = current, zw = button state
uniform vec2 iMouseClick; // position of the last mouse click
uniform vec4 iDate; // (year, month, day, seconds since midnight)
uniform vec3 iChannelResolution[4]; // resolution of each texture channel
uniform float iChannelTime[4]; // playback time of each texture channel

// Audio analysis
uniform float amp; // audio amplitude scaled by sensitivity
uniform float uamp; // raw audio amplitude before sensitivity scaling
uniform float iamp; // estimated dominant frequency in Hz
uniform float amp_peak; // peak audio energy
uniform float amp_rms; // RMS audio energy
uniform float amp_smooth; // smoothed audio amplitude
uniform float amp_low; // bass energy
uniform float amp_mid; // mid-range energy
uniform float amp_high; // treble energy
uniform float iSampleRate; // audio sample rate in Hz
uniform sampler1D spectrum; // current FFT spectrum
uniform sampler1D spectrum0; // current FFT spectrum alias
uniform sampler1DArray spectrum_history; // rolling FFT history
uniform int spectrum_history_head; // newest FFT history layer
uniform int spectrum_history_size; // number of FFT history layers

// acidcamGL-compatible values
uniform float value_alpha_r;
uniform float value_alpha_g;
uniform float value_alpha_b;
uniform float alpha_r;
uniform float alpha_g;
uniform float alpha_b;
uniform float alpha_value;
uniform float index_value;
uniform vec4 optx;
uniform vec4 random_var;
uniform float restore_black;
uniform vec4 inc_value;
uniform vec4 inc_valuex;

// MIDI controller values
uniform float slider1;
uniform float slider2;
uniform float slider3;
uniform float slider4;
)";

    constexpr const char *DEFAULT_SHADER_BODY = R"(
void main(void) {
    color = texture(samp, tc);
}
)";

    constexpr const char *CACHE_SHADER_CODE = R"(
// SIZE and USE_HISTORY_TEXTURE_ARRAY are supplied by ACMX2 at compile time.
#ifndef SIZE
#define SIZE 8
#endif

#ifndef USE_HISTORY_TEXTURE_ARRAY
#define USE_HISTORY_TEXTURE_ARRAY 0
#endif

#if USE_HISTORY_TEXTURE_ARRAY
uniform sampler2DArray history;
uniform int history_head;
#else
uniform sampler2D samp1;
uniform sampler2D samp2;
uniform sampler2D samp3;
uniform sampler2D samp4;
uniform sampler2D samp5;
uniform sampler2D samp6;
uniform sampler2D samp7;
uniform sampler2D samp8;
#endif

vec4 sampleTextureCache(int index, vec2 uv) {
    int cacheIndex = clamp(index, 0, SIZE - 1);
#if USE_HISTORY_TEXTURE_ARRAY
    int layer = (history_head + cacheIndex) % SIZE;
    return texture(history, vec3(uv, float(layer)));
#else
    cacheIndex = min(cacheIndex, 7);
    if (cacheIndex == 0) return texture(samp1, uv);
    if (cacheIndex == 1) return texture(samp2, uv);
    if (cacheIndex == 2) return texture(samp3, uv);
    if (cacheIndex == 3) return texture(samp4, uv);
    if (cacheIndex == 4) return texture(samp5, uv);
    if (cacheIndex == 5) return texture(samp6, uv);
    if (cacheIndex == 6) return texture(samp7, uv);
    return texture(samp8, uv);
#endif
}

float sampleSpectrumHistory(int age, float frequency) {
    if (spectrum_history_size <= 0) {
        return texture(spectrum, clamp(frequency, 0.0, 1.0)).r;
    }

    int historySize = max(spectrum_history_size, 1);
    int layer = (spectrum_history_head - (max(age, 0) % historySize) +
                 historySize) %
                historySize;
    return texture(spectrum_history,
                   vec2(clamp(frequency, 0.0, 1.0), float(layer)))
        .r;
}

void main(void) {
    vec4 liveFrame = texture(samp, tc);
    vec4 cachedFrame = sampleTextureCache(SIZE - 1, tc);
    float liveEnergy = texture(spectrum, 0.08).r;
    float historyEnergy = sampleSpectrumHistory(1, 0.08);
    float cacheMix = clamp(0.25 + historyEnergy * 0.5, 0.0, 0.75);

    color = mix(liveFrame, cachedFrame, cacheMix);
    color.rgb *= 1.0 + liveEnergy * 0.2;
}
)";
} // namespace

bool ShaderDialog::createShaderFile(const QString &shaderName,
                                    bool includeDefaultCode,
                                    bool createCacheShader) {
    QFile file(shaderName);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        if (includeDefaultCode) {
            out << DEFAULT_SHADER_HEADER;
            out << (createCacheShader ? CACHE_SHADER_CODE : DEFAULT_SHADER_BODY);
            out << "\n";
        }
        file.close();
        return true;
    } else {
        QMessageBox::critical(this, "Error", "Failed to create shader file.");
        return false;
    }
}
