#include "uniform-reference.hpp"
#include "custom_style.hpp"

#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHash>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QSplitter>
#include <QVBoxLayout>

namespace {
    struct UniformInfo {
        const char *name;
        const char *type;
        const char *category;
        const char *availability;
        const char *description;
        const char *example;
    };

    constexpr UniformInfo UNIFORMS[] = {
        {"samp", "sampler2D", "Core", "Always available",
         "The current video, camera, image, or previous multipass output. Texture coordinates normally use tc in the 0.0 to 1.0 range.",
         "vec4 source = texture(samp, tc);"},
        {"alpha", "float", "Core", "Always available",
         "ACMX2's animated compatibility value. It can be used as a phase, blend amount after clamping, or animation control.",
         "float blend = clamp(alpha, 0.0, 1.0);"},
        {"iTime", "float", "Time", "Always available",
         "Elapsed wall-clock time in seconds since rendering started.",
         "float wave = sin(iTime);"},
        {"time_f", "float", "Time", "Always available",
         "Controllable shader time. It follows ACMX2 time pause, stepping, speed, and audio-reactive time controls.",
         "vec2 uv = tc + 0.02 * sin(time_f);"},
        {"time_speed", "float", "Time", "Always available",
         "The current rate used to advance time_f.",
         "float speed = max(time_speed, 0.001);"},
        {"iFrame", "int", "Time", "Always available",
         "The number of rendered frames since startup, wrapped before the signed integer limit.",
         "float alternate = float(iFrame % 2);"},
        {"iTimeDelta", "float", "Time", "Always available",
         "Elapsed seconds since the preceding rendered frame.",
         "float frame_motion = velocity * iTimeDelta;"},
        {"iFrameRate", "float", "Time", "Always available",
         "The current target or reported output frame rate.",
         "float frame_seconds = 1.0 / max(iFrameRate, 1.0);"},
        {"iResolution", "vec2", "Viewport", "Always available",
         "Current render width and height in pixels. Divide x by y when correcting texture coordinates for aspect ratio.",
         "vec2 p = (tc - 0.5) * vec2(iResolution.x / iResolution.y, 1.0);"},
        {"iMouse", "vec4", "Input", "Always available",
         "Mouse state in render coordinates: xy is the current position, z is 1 while the left button is held, and w is 1 while the right button is held.",
         "vec2 mouse_uv = iMouse.xy / max(iResolution, vec2(1.0));"},
        {"iMouseClick", "vec2", "Input", "Always available",
         "Position in render pixels where the most recent left click began.",
         "vec2 click_uv = iMouseClick / max(iResolution, vec2(1.0));"},
        {"iDate", "vec4", "Time", "Always available",
         "Local date and time as year, month, day, and seconds since midnight.",
         "float day_phase = iDate.w / 86400.0;"},
        {"iChannelTime[4]", "float", "Compatibility", "Reserved Shadertoy compatibility",
         "Per-channel playback times. ACMX2 resolves these names for compatibility; channels without a separate clock remain at their default value.",
         "float channel_time = iChannelTime[0];"},
        {"iChannelResolution[4]", "vec3", "Compatibility", "Reserved Shadertoy compatibility",
         "Per-channel texture dimensions. Channels without separately supplied dimensions remain at their default value.",
         "vec2 channel_size = iChannelResolution[0].xy;"},
        {"mv_matrix", "mat4", "Transform", "Primarily used by the vertex stage",
         "The active model-view transform. The engine sets it for its 2D and 3D render paths.",
         "vec4 view_position = mv_matrix * position;"},
        {"proj_matrix", "mat4", "Transform", "Primarily used by the vertex stage",
         "The active projection transform used with mv_matrix.",
         "gl_Position = proj_matrix * mv_matrix * position;"},

        {"history", "sampler2DArray", "Frame cache", "Cache shader with texture-array cache enabled",
         "All cached video frames in one 2D array texture. Select a layer with the third texture coordinate; this is the preferred cache access method.",
         "vec4 old_frame = texture(history, vec3(tc, float(CACHE_HISTORY_LAYER(index))));"},
        {"history_head", "int", "Frame cache", "Used with history",
         "Physical array layer corresponding to logical cache index zero. Map an index with (history_head + index) % SIZE.",
         "int layer = (history_head + index) % SIZE;"},
        {"textures[SIZE]", "sampler2D", "Frame cache", "Cache shader using sampler-array mode",
         "Scalable array of individual cached-frame samplers. history is preferred when texture-array cache mode is enabled.",
         "vec4 old_frame = texture(textures[index], tc);"},
        {"samp1", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 0. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp1, tc);"},
        {"samp2", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 1. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp2, tc);"},
        {"samp3", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 2. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp3, tc);"},
        {"samp4", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 3. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp4, tc);"},
        {"samp5", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 4. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp5, tc);"},
        {"samp6", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 5. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp6, tc);"},
        {"samp7", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 6. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp7, tc);"},
        {"samp8", "sampler2D", "Frame cache", "Legacy cache binding",
         "Legacy cached-frame sampler for logical cache slot 7. Prefer history for new cache shaders.",
         "vec4 old_frame = texture(samp8, tc);"},

        {"amp", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Current overall audio amplitude after ACMX2 sensitivity and startup-envelope scaling.",
         "float pulse = 1.0 + amp * 0.2;"},
        {"uamp", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Compatibility value associated with the unscaled audio amplitude path.",
         "float raw_audio = uamp;"},
        {"iamp", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Estimated dominant audio frequency in hertz.",
         "float normalized_frequency = iamp / max(iSampleRate, 1.0);"},
        {"amp_peak", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Peak audio energy for sharp transients such as kicks and claps, scaled by sensitivity.",
         "float flash = smoothstep(0.7, 1.0, amp_peak);"},
        {"amp_rms", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Root-mean-square audio energy, useful as a steadier loudness estimate.",
         "float loudness = amp_rms;"},
        {"amp_smooth", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Smoothed amplitude that avoids abrupt frame-to-frame changes.",
         "float scale = 1.0 + amp_smooth * 0.1;"},
        {"amp_low", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Low-frequency or bass-band energy, scaled by sensitivity.",
         "float bass = amp_low;"},
        {"amp_mid", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "Mid-frequency energy, scaled by sensitivity.",
         "float middle = amp_mid;"},
        {"amp_high", "float", "Audio", "Requires an audio-enabled build and audio processing",
         "High-frequency or treble-band energy, scaled by sensitivity.",
         "float treble = amp_high;"},
        {"iSampleRate", "float", "Audio", "Available in audio-enabled builds",
         "Audio capture sample rate in hertz, commonly 44100 or 48000.",
         "float nyquist = iSampleRate * 0.5;"},
        {"spectrum", "sampler1D", "Audio", "Requires an audio-enabled build and audio processing",
         "Current FFT magnitudes. Coordinate 0 is DC and 1 is the Nyquist frequency; read energy from the red channel.",
         "float bass = texture(spectrum, 0.03).r;"},
        {"spectrum0", "sampler1D", "Audio", "Requires an audio-enabled build and audio processing",
         "Alias of spectrum for the current FFT frame. It is the preferred live-spectrum name in cache shaders.",
         "float treble = texture(spectrum0, 0.75).r;"},
        {"spectrum_history", "sampler1DArray", "Audio history", "Requires audio buffers enabled",
         "Rolling FFT history held in a 1D array texture. Sample it with vec2(frequency, physical layer) and read the red channel.",
         "float old_bass = texture(spectrum_history, vec2(0.03, float(layer))).r;"},
        {"spectrum_history_head", "int", "Audio history", "Used with spectrum_history",
         "Physical spectrum-history layer containing the newest FFT frame.",
         "int layer = SPECTRUM_HISTORY_LAYER(age);"},
        {"spectrum_history_size", "int", "Audio history", "Used with spectrum_history",
         "Runtime number of allocated FFT-history layers. Protect modulo operations with max(size, 1).",
         "int count = max(spectrum_history_size, 1);"},

        {"value_alpha_r", "float", "acidcamGL compatibility", "Always available",
         "Animated red-channel compatibility value. It is also exposed as alpha_r.",
         "color.r *= value_alpha_r;"},
        {"value_alpha_g", "float", "acidcamGL compatibility", "Always available",
         "Animated green-channel compatibility value. It is also exposed as alpha_g.",
         "color.g *= value_alpha_g;"},
        {"value_alpha_b", "float", "acidcamGL compatibility", "Always available",
         "Animated blue-channel compatibility value. It is also exposed as alpha_b.",
         "color.b *= value_alpha_b;"},
        {"alpha_r", "float", "acidcamGL compatibility", "Always available",
         "Alias of value_alpha_r for legacy shaders.",
         "color.r *= alpha_r;"},
        {"alpha_g", "float", "acidcamGL compatibility", "Always available",
         "Alias of value_alpha_g for legacy shaders.",
         "color.g *= alpha_g;"},
        {"alpha_b", "float", "acidcamGL compatibility", "Always available",
         "Alias of value_alpha_b for legacy shaders.",
         "color.b *= alpha_b;"},
        {"alpha_value", "float", "acidcamGL compatibility", "Always available",
         "Alias of the engine's animated alpha compatibility value.",
         "float phase = alpha_value;"},
        {"index_value", "float", "acidcamGL compatibility", "Always available",
         "Zero-based index of the active shader in the loaded library.",
         "float shader_seed = index_value;"},
        {"optx", "vec4", "acidcamGL compatibility", "Always available",
         "Legacy option vector maintained by ACMX2; its default components are 0.5.",
         "vec2 option_pair = optx.xy;"},
        {"random_var", "vec4", "acidcamGL compatibility", "Always available",
         "Four engine-generated random values refreshed with the legacy uniform animation state.",
         "vec2 jitter = random_var.xy / 255.0;"},
        {"restore_black", "float", "acidcamGL compatibility", "Always available",
         "Legacy restore-black flag represented as 0.0 or 1.0.",
         "color.rgb *= 1.0 - restore_black;"},
        {"inc_value", "vec4", "acidcamGL compatibility", "Always available",
         "Primary legacy increment vector maintained for acidcamGL shader compatibility.",
         "vec2 offset = inc_value.xy;"},
        {"inc_valuex", "vec4", "acidcamGL compatibility", "Always available",
         "Secondary legacy increment vector maintained for acidcamGL shader compatibility.",
         "vec2 offset = inc_valuex.xy;"},

        {"slider1", "float", "MIDI", "Requires a MIDI-enabled build",
         "Value from the MIDI control mapped to Slider 1, normalized to the 0.0 to 1.0 range.",
         "float amount = slider1;"},
        {"slider2", "float", "MIDI", "Requires a MIDI-enabled build",
         "Value from the MIDI control mapped to Slider 2, normalized to the 0.0 to 1.0 range.",
         "float amount = slider2;"},
        {"slider3", "float", "MIDI", "Requires a MIDI-enabled build",
         "Value from the MIDI control mapped to Slider 3, normalized to the 0.0 to 1.0 range.",
         "float amount = slider3;"},
        {"slider4", "float", "MIDI", "Requires a MIDI-enabled build",
         "Value from the MIDI control mapped to Slider 4, normalized to the 0.0 to 1.0 range.",
         "float amount = slider4;"},
    };

    QString acmxvkDefine(const UniformInfo &uniform) {
        static const QHash<QString, QString> DEFINES = {
            {QStringLiteral("alpha"), QStringLiteral("ext.u0.x")},
            {QStringLiteral("iTime"), QStringLiteral("ext.u0.y")},
            {QStringLiteral("iResolution"), QStringLiteral("ext.u0.zw")},
            {QStringLiteral("iMouse"), QStringLiteral("ext.mouse")},
            {QStringLiteral("iTimeDelta"), QStringLiteral("ext.u1.x")},
            {QStringLiteral("amp"), QStringLiteral("ext.u1.y")},
            {QStringLiteral("iamp"), QStringLiteral("ext.u1.z")},
            {QStringLiteral("iFrameRate"), QStringLiteral("ext.u1.w")},
            {QStringLiteral("iFrame"), QStringLiteral("int(ext.u2.x)")},
            {QStringLiteral("time_f"), QStringLiteral("ext.u2.y")},
            {QStringLiteral("iSampleRate"), QStringLiteral("ext.u2.z")},
            {QStringLiteral("amp_peak"), QStringLiteral("ext.u2.w")},
            {QStringLiteral("history_head"), QStringLiteral("int(ext.u3.x)")},
            {QStringLiteral("amp_rms"), QStringLiteral("ext.u3.z")},
            {QStringLiteral("amp_smooth"), QStringLiteral("ext.u3.w")},
            {QStringLiteral("amp_low"), QStringLiteral("ext.audio_bands.x")},
            {QStringLiteral("amp_mid"), QStringLiteral("ext.audio_bands.y")},
            {QStringLiteral("amp_high"), QStringLiteral("ext.audio_bands.z")},
            {QStringLiteral("spectrum_history_head"),
             QStringLiteral("int(ext.audio_history.x)")},
            {QStringLiteral("spectrum_history_size"),
             QStringLiteral("int(ext.audio_history.y)")},
            {QStringLiteral("alpha_value"),
             QStringLiteral("ext.custom_uniforms[0].y")},
            {QStringLiteral("alpha_r"),
             QStringLiteral("ext.custom_uniforms[0].z")},
            {QStringLiteral("alpha_g"),
             QStringLiteral("ext.custom_uniforms[0].w")},
            {QStringLiteral("alpha_b"),
             QStringLiteral("ext.custom_uniforms[1].x")},
            {QStringLiteral("value_alpha_r"),
             QStringLiteral("ext.custom_uniforms[1].y")},
            {QStringLiteral("value_alpha_g"),
             QStringLiteral("ext.custom_uniforms[1].z")},
            {QStringLiteral("value_alpha_b"),
             QStringLiteral("ext.custom_uniforms[1].w")},
            {QStringLiteral("index_value"),
             QStringLiteral("ext.custom_uniforms[2].x")},
            {QStringLiteral("restore_black"),
             QStringLiteral("ext.custom_uniforms[2].y")},
            {QStringLiteral("time_speed"),
             QStringLiteral("ext.custom_uniforms[3].y")},
            {QStringLiteral("slider1"),
             QStringLiteral("ext.custom_uniforms[5].x")},
            {QStringLiteral("slider2"),
             QStringLiteral("ext.custom_uniforms[5].y")},
            {QStringLiteral("slider3"),
             QStringLiteral("ext.custom_uniforms[5].z")},
            {QStringLiteral("slider4"),
             QStringLiteral("ext.custom_uniforms[5].w")},
        };
        const QString name = QString::fromLatin1(uniform.name);
        const auto define = DEFINES.constFind(name);
        if (define == DEFINES.constEnd())
            return QString();
        return QStringLiteral("#define %1 %2").arg(name, define.value());
    }

    QString detailsFor(const UniformInfo &uniform, acmx2::Backend backend) {
        QString declaration =
            QStringLiteral("Declaration:\nuniform %1 %2;")
                .arg(QString::fromLatin1(uniform.type),
                     QString::fromLatin1(uniform.name));
        if (backend == acmx2::Backend::Acmxvk) {
            const QString define = acmxvkDefine(uniform);
            if (!define.isEmpty()) {
                declaration =
                    QStringLiteral("ACMXVK alias (copy after the required "
                                   "SpriteExtended block):\n%1")
                        .arg(define);
            } else {
                declaration = QStringLiteral(
                                  "ACMXVK resource declaration:\nuniform %1 %2;\n\n"
                                  "This value is not an alias within SpriteExtended, "
                                  "so it does not use a #define.")
                                  .arg(QString::fromLatin1(uniform.type),
                                       QString::fromLatin1(uniform.name));
            }
        }
        return QStringLiteral("%1\n\nType: %2\nCategory: %3\nAvailability: %4\n\n%5\n\n%6\n\nExample:\n%7")
            .arg(QString::fromLatin1(uniform.name),
                 QString::fromLatin1(uniform.type),
                 QString::fromLatin1(uniform.category),
                 QString::fromLatin1(uniform.availability),
                 declaration,
                 QString::fromLatin1(uniform.description),
                 QString::fromLatin1(uniform.example));
    }
} // namespace

UniformReferenceDialog::UniformReferenceDialog(acmx2::Backend backend,
                                               QWidget *parent)
    : QDialog(parent), activeBackend(backend) {
    setWindowTitle(tr("Built-in Uniform Reference"));
    resize(900, 580);
    setModal(false);

    auto *mainLayout = new QVBoxLayout(this);
    auto *introLabel = new QLabel(
        tr("Select a uniform to see its GLSL type, availability, and usage."), this);
    introLabel->setWordWrap(true);
    mainLayout->addWidget(introLabel);

    searchEdit = new QLineEdit(this);
    searchEdit->setPlaceholderText(tr("Filter uniforms..."));
    searchEdit->setClearButtonEnabled(true);
    mainLayout->addWidget(searchEdit);

    auto *splitter = new QSplitter(Qt::Horizontal, this);
    uniformList = new QListWidget(splitter);
    uniformList->setMinimumWidth(260);
    descriptionView = new QPlainTextEdit(splitter);
    descriptionView->setReadOnly(true);
    descriptionView->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    splitter->addWidget(uniformList);
    splitter->addWidget(descriptionView);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    mainLayout->addWidget(splitter, 1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    mainLayout->addWidget(buttons);

    connect(searchEdit, &QLineEdit::textChanged, this,
            &UniformReferenceDialog::filterUniforms);
    connect(uniformList, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem *current, QListWidgetItem *) {
                showUniformDetails(current);
            });
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::close);

    populateUniforms();
    acmx2::applyCustomStyleIfEnabled(this);
}

void UniformReferenceDialog::populateUniforms() {
    uniformList->clear();
    for (const UniformInfo &uniform : UNIFORMS) {
        auto *item = new QListWidgetItem(QString::fromLatin1(uniform.name), uniformList);
        item->setData(Qt::UserRole, detailsFor(uniform, activeBackend));
        item->setData(Qt::UserRole + 1,
                      QStringLiteral("%1 %2 %3")
                          .arg(QString::fromLatin1(uniform.name),
                               QString::fromLatin1(uniform.type),
                               QString::fromLatin1(uniform.category)));
        item->setToolTip(QString::fromLatin1(uniform.category));
    }
    if (uniformList->count() > 0)
        uniformList->setCurrentRow(0);
}

void UniformReferenceDialog::setBackend(acmx2::Backend backend) {
    if (activeBackend == backend)
        return;
    activeBackend = backend;
    populateUniforms();
    filterUniforms(searchEdit->text());
}

void UniformReferenceDialog::filterUniforms(const QString &text) {
    const QString filter = text.trimmed();
    QListWidgetItem *firstVisible = nullptr;
    for (int index = 0; index < uniformList->count(); ++index) {
        QListWidgetItem *item = uniformList->item(index);
        const bool hidden = !item->data(Qt::UserRole + 1)
                                 .toString()
                                 .contains(filter, Qt::CaseInsensitive);
        item->setHidden(hidden);
        if (!hidden && !firstVisible)
            firstVisible = item;
    }
    if (firstVisible)
        uniformList->setCurrentItem(firstVisible);
    else
        descriptionView->clear();
}

void UniformReferenceDialog::showUniformDetails(QListWidgetItem *current) {
    descriptionView->setPlainText(current ? current->data(Qt::UserRole).toString()
                                          : QString());
}
