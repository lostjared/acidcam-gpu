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

    QLabel *shaderTypeLabel = new QLabel("Shader type:", this);
    layout->addWidget(shaderTypeLabel);

    shaderTypeComboBox = new QComboBox(this);
    shaderTypeComboBox->addItem("Fragment shader (.glsl)");
    shaderTypeComboBox->addItem("Compute shader (.comp)");
    layout->addWidget(shaderTypeComboBox);

    defaultCodeCheckBox = new QCheckBox("Include default shader code", this);
    defaultCodeCheckBox->setChecked(true);
    layout->addWidget(defaultCodeCheckBox);

    cacheShaderCheckBox = new QCheckBox("Create as cache shader (_cache.glsl)", this);
    cacheShaderCheckBox->setToolTip(
        "Create a texture-cache shader starter with frame-cache and spectrum-history sampling.");
    layout->addWidget(cacheShaderCheckBox);

    connect(shaderTypeComboBox,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const bool createComputeShader = index == 1;
                cacheShaderCheckBox->setText(
                    createComputeShader
                        ? "Create as cache shader (_cache.comp)"
                        : "Create as cache shader (_cache.glsl)");
                shaderNameEdit->setPlaceholderText(
                    createComputeShader
                        ? "Shader name (e.g., myshader.comp)"
                        : "Shader name (e.g., myshader.glsl)");
            });

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
    resize(440, 230);
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
    } else if (shaderName.endsWith(".comp", Qt::CaseInsensitive)) {
        shaderName.chop(5);
    }

    if (shaderName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a shader name.");
        return;
    }

    const bool createComputeShader = shaderTypeComboBox->currentIndex() == 1;
    const bool createCacheShader = cacheShaderCheckBox->isChecked();
    if (createCacheShader &&
        !shaderName.endsWith("_cache", Qt::CaseInsensitive)) {
        shaderName += "_cache";
    }
    shaderName += createComputeShader ? ".comp" : ".glsl";

    const bool includeDefaultCode = defaultCodeCheckBox->isChecked();
    if (!createShaderFile(shaderPath + "/" + shaderName, includeDefaultCode,
                          createCacheShader, createComputeShader)) {
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
    constexpr const char *DEFAULT_SHADER_CODE = R"(#version 330 core
in vec2 tc;
out vec4 color;

uniform sampler2D samp;

void main() {
    color = texture(samp, tc);
}
)";

    constexpr const char *CACHE_SHADER_CODE = R"(#version 330 core
in vec2 tc;
out vec4 color;

uniform sampler2D samp;
uniform sampler2DArray history;
uniform int history_head;
#ifndef SIZE
#define SIZE 8
#endif
#ifndef CACHE_HISTORY_LAYER
#define CACHE_HISTORY_LAYER(index) ((history_head + (index)) % SIZE)
#endif

uniform sampler1D spectrum0;
uniform sampler1DArray spectrum_history;
uniform int spectrum_history_head;
uniform int spectrum_history_size;
#ifndef SPECTRUM_HISTORY_LAYER
#define SPECTRUM_HISTORY_LAYER(index) ((spectrum_history_head - ((index) % max(spectrum_history_size, 1)) + max(spectrum_history_size, 1)) % max(spectrum_history_size, 1))
#endif

vec4 sample_cache(int index, vec2 uv) {
    int cache_index = clamp(index, 0, SIZE - 1);
    return texture(history,
                   vec3(uv, float(CACHE_HISTORY_LAYER(cache_index))));
}

float sample_spectrum_history(int index, float frequency) {
    int maximum_age = max(min(SIZE, spectrum_history_size) - 1, 0);
    int spectrum_age = min(max(index, 0) + 1, maximum_age);
    return texture(spectrum_history,
                   vec2(clamp(frequency, 0.0, 1.0),
                        float(SPECTRUM_HISTORY_LAYER(spectrum_age))))
        .r;
}

void main() {
    vec4 live_frame = texture(samp, tc);
    vec4 cached_frame = sample_cache(SIZE - 1, tc);
    float live_energy = texture(spectrum0, 0.08).r;
    float history_energy = sample_spectrum_history(0, 0.08);
    float cache_mix = clamp(0.25 + history_energy * 0.5, 0.0, 0.75);

    color = mix(live_frame, cached_frame, cache_mix);
    color.rgb *= 1.0 + live_energy * 0.2;
}
)";

    constexpr const char *COMPUTE_SHADER_CODE = R"(#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = 0) writeonly uniform image2D outputImage;

uniform sampler2D samp;
uniform vec2 iResolution;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(iResolution.x) || pixel.y >= int(iResolution.y))
        return;

    vec2 uv = (vec2(pixel) + 0.5) / iResolution;
    imageStore(outputImage, pixel, texture(samp, uv));
}
)";

    constexpr const char *COMPUTE_CACHE_SHADER_CODE = R"(#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = 0) writeonly uniform image2D outputImage;

uniform sampler2D samp;
uniform vec2 iResolution;

#ifndef SIZE
#define SIZE 8
#endif
#ifndef USE_HISTORY_TEXTURE_ARRAY
#define USE_HISTORY_TEXTURE_ARRAY 0
#endif

#if USE_HISTORY_TEXTURE_ARRAY
uniform sampler2DArray history;
uniform int history_head;

vec4 sample_oldest_frame(ivec2 pixel) {
    return texelFetch(history, ivec3(pixel, history_head), 0);
}
#else
uniform sampler2D textures[SIZE];

vec4 sample_oldest_frame(ivec2 pixel) {
    return texelFetch(textures[0], pixel, 0);
}
#endif

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(iResolution.x) || pixel.y >= int(iResolution.y))
        return;

    vec4 live_frame = texelFetch(samp, pixel, 0);
    vec4 cached_frame = sample_oldest_frame(pixel);
    imageStore(outputImage, pixel, mix(live_frame, cached_frame, 0.5));
}
)";
} // namespace

bool ShaderDialog::createShaderFile(const QString &shaderName,
                                    bool includeDefaultCode,
                                    bool createCacheShader,
                                    bool createComputeShader) {
    QFile file(shaderName);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        if (includeDefaultCode) {
            if (createComputeShader) {
                out << (createCacheShader ? COMPUTE_CACHE_SHADER_CODE
                                          : COMPUTE_SHADER_CODE);
            } else {
                out << (createCacheShader ? CACHE_SHADER_CODE : DEFAULT_SHADER_CODE);
            }
            out << "\n";
        }
        file.close();
        return true;
    } else {
        QMessageBox::critical(this, "Error", "Failed to create shader file.");
        return false;
    }
}
