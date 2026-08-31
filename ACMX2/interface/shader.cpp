#include "shader.hpp"
#include "acmxvk-source-manifest.hpp"
#include "custom_style.hpp"
#include "shader-manifest.hpp"
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSettings>

ShaderDialog::ShaderDialog(acmx2::Backend selectedBackend, QWidget *parent)
    : QDialog(parent), backend(selectedBackend) {
    init();
}

void ShaderDialog::init() {
    QVBoxLayout *layout = new QVBoxLayout(this);

    QLabel *instructionLabel = new QLabel("Enter the name of the shader file:", this);
    layout->addWidget(instructionLabel);

    shaderNameEdit = new QLineEdit(this);
    shaderNameEdit->setPlaceholderText(
        backend == acmx2::Backend::Acmxvk
            ? "Shader name (e.g., myshader.frag)"
            : "Shader name (e.g., myshader.glsl)");
    layout->addWidget(shaderNameEdit);

    QLabel *shaderTypeLabel = new QLabel("Shader type:", this);
    layout->addWidget(shaderTypeLabel);

    shaderTypeComboBox = new QComboBox(this);
    shaderTypeComboBox->addItem(
        backend == acmx2::Backend::Acmxvk ? "Fragment shader (.frag)"
                                          : "Fragment shader (.glsl)");
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
                        : (backend == acmx2::Backend::Acmxvk
                               ? "Create as cache shader (_cache.frag)"
                               : "Create as cache shader (_cache.glsl)"));
                shaderNameEdit->setPlaceholderText(
                    createComputeShader
                        ? "Shader name (e.g., myshader.comp)"
                        : (backend == acmx2::Backend::Acmxvk
                               ? "Shader name (e.g., myshader.frag)"
                               : "Shader name (e.g., myshader.glsl)"));
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

    if (shaderName.endsWith(".glsl", Qt::CaseInsensitive) ||
        shaderName.endsWith(".frag", Qt::CaseInsensitive)) {
        shaderName.chop(5);
    } else if (shaderName.endsWith(".comp", Qt::CaseInsensitive)) {
        shaderName.chop(5);
    }

    if (shaderName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a shader name.");
        return;
    }

    static const QRegularExpression safeName(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9_.-]*$"));
    if (!safeName.match(shaderName).hasMatch() || shaderName == "." ||
        shaderName == "..") {
        QMessageBox::warning(
            this, "Warning",
            "Shader names may contain only letters, numbers, '.', '_', and '-'.");
        return;
    }

    const bool createComputeShader = shaderTypeComboBox->currentIndex() == 1;
    const bool createCacheShader = cacheShaderCheckBox->isChecked();
    if (createCacheShader &&
        !shaderName.endsWith("_cache", Qt::CaseInsensitive)) {
        shaderName += "_cache";
    }
    shaderName += createComputeShader
                      ? ".comp"
                      : (backend == acmx2::Backend::Acmxvk ? ".frag" : ".glsl");

    QString relativeShaderName = shaderName;
    if (backend == acmx2::Backend::Acmxvk && createComputeShader)
        relativeShaderName = QStringLiteral("compute/") + shaderName;
    const QString absoluteShaderName =
        QDir(shaderPath).filePath(relativeShaderName);
    if (QFileInfo::exists(absoluteShaderName)) {
        QMessageBox::warning(
            this, "Shader Already Exists",
            QString("The shader already exists and was not changed:\n%1")
                .arg(absoluteShaderName));
        return;
    }
    if (!QDir().mkpath(QFileInfo(absoluteShaderName).absolutePath())) {
        QMessageBox::critical(this, "Error",
                              "Failed to create the shader directory.");
        return;
    }

    const bool includeDefaultCode = defaultCodeCheckBox->isChecked();
    if (!createShaderFile(absoluteShaderName, includeDefaultCode,
                          createCacheShader, createComputeShader)) {
        return;
    }

    QString manifestError;
    bool manifestCreated = false;
    if (backend == acmx2::Backend::Acmxvk) {
        acmx2::AcmxvkSourceManifestResult result;
        manifestCreated = acmx2::create_acmxvk_source_manifest(
            shaderPath, QString(), result, manifestError);
    } else {
        manifestCreated = acmx2::append_shader_manifest(
            shaderPath, relativeShaderName, manifestError);
    }
    if (!manifestCreated) {
        QFile::remove(absoluteShaderName);
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
    imageStore(outputImage, pixel, textureLod(samp, uv, 0.0));
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

vec4 sample_oldest_frame(vec2 uv) {
    return textureLod(history, vec3(uv, float(history_head)), 0.0);
}
#else
uniform sampler2D textures[SIZE];

vec4 sample_oldest_frame(vec2 uv) {
    return textureLod(textures[0], uv, 0.0);
}
#endif

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(iResolution.x) || pixel.y >= int(iResolution.y))
        return;

    vec2 uv = (vec2(pixel) + 0.5) / iResolution;
    vec4 live_frame = textureLod(samp, uv, 0.0);
    vec4 cached_frame = sample_oldest_frame(uv);
    imageStore(outputImage, pixel, mix(live_frame, cached_frame, 0.5));
}
)";

    constexpr const char *ACMXVK_FRAGMENT_SHADER_CODE = R"(#version 450

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

#define iResolution ext.u0.zw
#define time_f ext.u2.y

void main() {
    color = texture(samp, tc);
}
)";

    constexpr const char *ACMXVK_FRAGMENT_CACHE_SHADER_CODE = R"(#version 450

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D samp;
layout(set = 0, binding = 2) uniform sampler2DArray history;

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

#define history_head int(ext.u3.x + 0.5)
#define history_size max(int(ext.u3.y), 1)

void main() {
    vec4 live_frame = texture(samp, tc);
    int cache_layer = history_head % history_size;
    vec4 cached_frame = texture(history, vec3(tc, float(cache_layer)));
    color = mix(live_frame, cached_frame, 0.5);
}
)";

    constexpr const char *ACMXVK_COMPUTE_SHADER_CODE = R"(#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 5, rgba8) writeonly uniform image2D outputImage;
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
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, size)))
        return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    imageStore(outputImage, pixel, textureLod(samp, uv, 0.0));
}
)";

    constexpr const char *ACMXVK_COMPUTE_CACHE_SHADER_CODE = R"(#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 5, rgba8) writeonly uniform image2D outputImage;
layout(set = 0, binding = 0) uniform sampler2D samp;
layout(set = 0, binding = 2) uniform sampler2DArray history;

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

#define history_head int(ext.u3.x + 0.5)
#define history_size max(int(ext.u3.y), 1)

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, size)))
        return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec4 live_frame = textureLod(samp, uv, 0.0);
    int cache_layer = history_head % history_size;
    vec4 cached_frame =
        textureLod(history, vec3(uv, float(cache_layer)), 0.0);
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
            if (backend == acmx2::Backend::Acmxvk) {
                if (createComputeShader) {
                    out << (createCacheShader
                                ? ACMXVK_COMPUTE_CACHE_SHADER_CODE
                                : ACMXVK_COMPUTE_SHADER_CODE);
                } else {
                    out << (createCacheShader
                                ? ACMXVK_FRAGMENT_CACHE_SHADER_CODE
                                : ACMXVK_FRAGMENT_SHADER_CODE);
                }
            } else if (createComputeShader) {
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
