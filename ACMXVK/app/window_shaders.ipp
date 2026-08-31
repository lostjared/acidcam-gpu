        // Shader discovery, custom uniforms, interface IPC, and playlists.
        void loadShaders() {
            if (!options.fragment_shader.empty() ||
                !options.compute_shader.empty()) {
                const bool compute = !options.compute_shader.empty();
                const fs::path shader = fs::absolute(
                                            compute ? options.compute_shader : options.fragment_shader)
                                            .lexically_normal();
                const std::string label =
                    compute ? "compute shader" : "fragment shader";
                if (shader.extension() != ".spv" ||
                    !fs::is_regular_file(shader)) {
                    throw std::runtime_error(
                        label + " is not a readable .spv file: " +
                        shader.string());
                }
                input::validate_spirv_file(shader, label);
                const mxvk::ShaderModuleInfo module_info =
                    mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
                const mxvk::ShaderStage expected_stage =
                    compute ? mxvk::ShaderStage::Compute
                            : mxvk::ShaderStage::Fragment;
                if (module_info.stage != expected_stage) {
                    throw std::runtime_error(
                        label + " SPIR-V entry point has the wrong shader stage: " +
                        shader.string());
                }
                recordShaderResources(module_info, "shader");
                shaders.push_back(shader);
                return;
            }
            if (options.shader_directory.empty()) {
                return;
            }

            shader_library_directory =
                fs::absolute(options.shader_directory).lexically_normal();
            const ShaderManifest manifest =
                loadShaderManifest(shader_library_directory);
            shader_manifest_path = manifest.path;
            custom_uniforms = manifest.custom_uniforms;
            applyCustomUniformOverrides();
            for (const std::string &entry : manifest.entries) {
                const fs::path shader =
                    resolveShaderManifestEntry(shader_library_directory, entry);
                if (!shader.empty()) {
                    input::validate_spirv_file(shader,
                                               "shader manifest entry");
                    const mxvk::ShaderModuleInfo module_info =
                        mxvk::inspect_spirv(mxvk::load_spv(shader.string()));
                    recordShaderResources(module_info, "shader library");
                    shaders.push_back(shader);
                }
            }
            std::sort(shaders.begin(), shaders.end(), [](const fs::path &left, const fs::path &right) {
                std::string left_text = left.generic_string();
                std::string right_text = right.generic_string();
                std::transform(left_text.begin(), left_text.end(), left_text.begin(),
                               [](unsigned char character) {
                                   return static_cast<char>(std::tolower(character));
                               });
                std::transform(right_text.begin(), right_text.end(), right_text.begin(),
                               [](unsigned char character) {
                                   return static_cast<char>(std::tolower(character));
                               });
                return left_text < right_text;
            });
            if (shaders.empty()) {
                throw std::runtime_error("shader manifest contains no readable SPIR-V files: " +
                                         shader_manifest_path.string());
            }
            std::cout << "acmxvk: loaded " << shaders.size() << " shaders from "
                      << shader_manifest_path.string() << '\n';
            printCustomUniforms();

            if (!options.shader_file.empty()) {
                const auto selected = std::find_if(
                    shaders.begin(), shaders.end(), [&](const fs::path &path) {
                        fs::path requested(options.shader_file);
                        if (requested.extension() != ".spv") {
                            requested.replace_extension(".spv");
                        }
                        return path.filename() == requested.filename() ||
                               path.lexically_relative(shader_library_directory) == requested;
                    });
                if (selected == shaders.end()) {
                    throw std::runtime_error("shader file is not listed in the manifest: " +
                                             options.shader_file);
                }
                shader_index = static_cast<std::size_t>(std::distance(shaders.begin(), selected));
            } else {
                const int count = static_cast<int>(shaders.size());
                const int wrapped_index = ((options.shader_index % count) + count) % count;
                shader_index = static_cast<std::size_t>(wrapped_index);
            }
        }

        void applyCustomUniformOverrides() {
            for (const std::string &override_text :
                 options.custom_uniform_overrides) {
                const std::size_t separator = override_text.find('=');
                if (separator == std::string::npos || separator == 0 ||
                    separator + 1 >= override_text.size()) {
                    throw std::runtime_error(
                        "--uniform requires name=value: " + override_text);
                }
                const std::string name = trim(override_text.substr(0, separator));
                const double value = parseNumber(
                    trim(override_text.substr(separator + 1)), "--uniform");
                const auto match = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &uniform) {
                        return uniform.name == name;
                    });
                if (match == custom_uniforms.end()) {
                    throw std::runtime_error(
                        "custom uniform is not defined in library.json: " + name);
                }
                match->value = std::clamp(value, match->minimum, match->maximum);
            }

            custom_uniform_values.clear();
            custom_uniform_values.reserve(custom_uniforms.size());
            for (const ShaderManifest::CustomUniform &uniform : custom_uniforms) {
                custom_uniform_values.push_back(static_cast<float>(uniform.value));
            }
        }

        void printCustomUniforms() const {
            if (custom_uniforms.empty()) {
                return;
            }
            constexpr std::string_view COMPONENTS = "xyzw";
            std::cout << "acmxvk: custom uniforms (binding 1):\n";
            for (std::size_t index = 0; index < custom_uniforms.size(); ++index) {
                const ShaderManifest::CustomUniform &uniform = custom_uniforms[index];
                std::cout << "  " << uniform.name << '=' << uniform.value
                          << " -> custom_uniforms[" << (index / 4) << "]."
                          << COMPONENTS[index % 4] << '\n';
            }
        }

        [[nodiscard]] std::string currentShader() const {
            return shaders.empty() ? std::string{} : shaders[shader_index].string();
        }

        [[nodiscard]] bool historyCacheEnabled() const {
            return options.enable_texture_cache || shader_history_required;
        }

        void recordShaderResources(const mxvk::ShaderModuleInfo &module_info,
                                   std::string_view source) {
            if (module_info.usesHistoryTexture &&
                !shader_history_required) {
                shader_history_required = true;
                std::cout << "acmxvk: enabled shared history for " << source
                          << " binding 2\n";
            }
            if (module_info.usesSpectrumTexture &&
                !shader_spectrum_required) {
                shader_spectrum_required = true;
                std::cout << "acmxvk: enabled spectrum descriptor for " << source
                          << " binding 3\n";
            }
            if (module_info.usesSpectrumHistoryTexture &&
                !shader_spectrum_history_required) {
                shader_spectrum_history_required = true;
                if (options.audio_buffers == 0) {
                    options.audio_buffers = 8;
                }
                std::cout << "acmxvk: enabled " << options.audio_buffers
                          << " spectrum-history layers for " << source
                          << " binding 4\n";
            }
        }

        [[nodiscard]] std::uint32_t spectrumBinCount() const {
#ifdef AUDIO_ENABLED
            return audio::AudioEngine::spectrum_bin_count();
#else
            return COMPATIBILITY_SPECTRUM_BIN_COUNT;
#endif
        }

        [[nodiscard]] bool spectrumTextureEnabledForShaders() const {
#ifdef AUDIO_ENABLED
            return true;
#else
            return shader_spectrum_required;
#endif
        }

        [[nodiscard]] bool spectrumHistoryEnabledForShaders() const {
            return options.audio_buffers > 0;
        }

        void initialize_interface_control() {
            if (!options.interface_shm) {
                return;
            }

            if (!interface_client.open()) {
                return;
            }

            InterfaceState state;
            if (!interface_client.read(state)) {
                std::cerr << "acmxvk: interface control protocol does not match "
                             "this build\n";
                interface_client.close();
                return;
            }
            interface_last_sequence = state.sequence;
            apply_interface_multipass_state(state.multipass);
            apply_interface_playback_state(state.playback, false);
            apply_interface_overlay_state(state.overlay, false);
            apply_interface_gpu_filter_state(state.gpu_filters, false);
            interface_last_audio_file_sequence =
                state.audio_file.request_sequence;
            interface_last_reload_sequence = state.reload.request_sequence;
            std::cout << "acmxvk: interface live shader, multipass, playback, "
                         "overlay, GPU-filter, and audio-file control enabled\n";
        }

        void sync_interface_control() {
            InterfaceState state;
            if (!interface_client.read(state) ||
                state.sequence == interface_last_sequence) {
                return;
            }
            interface_last_sequence = state.sequence;
            apply_interface_shader_selection(state.selected_shader_name);
            apply_interface_multipass_state(state.multipass);
            apply_interface_uniform_values(state.uniform_values);
            apply_interface_playback_state(state.playback, true);
            apply_interface_overlay_state(state.overlay, true);
            apply_interface_gpu_filter_state(state.gpu_filters, true);
            if (state.audio_file.request_sequence !=
                interface_last_audio_file_sequence) {
                interface_last_audio_file_sequence =
                    state.audio_file.request_sequence;
                apply_interface_audio_file_state(state.audio_file);
            }
            if (state.reload.request_sequence !=
                interface_last_reload_sequence) {
                interface_last_reload_sequence = state.reload.request_sequence;
                apply_interface_shader_reload(state.reload);
            }
        }

        void apply_interface_playback_state(
            const InterfacePlaybackState &requested, bool announce) {
            if (options.repeat != requested.repeat) {
                options.repeat = requested.repeat;
                if (announce) {
                    std::cout << "acmxvk: interface video repeat "
                              << (options.repeat ? "enabled" : "disabled")
                              << '\n';
                }
            }
            if (options.normalized_time != requested.normalized_time) {
                options.normalized_time = requested.normalized_time;
                if (announce) {
                    std::cout << "acmxvk: interface normalized time "
                              << (options.normalized_time ? "enabled"
                                                          : "disabled")
                              << '\n';
                }
            }
        }

        void apply_interface_overlay_state(const InterfaceOverlayState &requested,
                                           bool announce) {
            if (options.display_filter != requested.display_filter) {
                options.display_filter = requested.display_filter;
                if (announce) {
                    std::cout << "acmxvk: interface display-filter overlay "
                              << (options.display_filter ? "enabled"
                                                         : "disabled")
                              << '\n';
                }
            }

            try {
                input::validate_string(requested.watermark_text,
                                       input::StringKind::DisplayText,
                                       "interface watermark", true);
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: rejected interface watermark: "
                          << error.what() << '\n';
                return;
            }

            const bool was_enabled =
                watermark_enabled && !options.watermark_text.empty();
            const bool requested_enabled =
                requested.watermark_enabled &&
                !requested.watermark_text.empty();
            const bool changed =
                watermark_enabled != requested_enabled ||
                options.watermark_text != requested.watermark_text ||
                options.watermark_color != requested.watermark_color;
            if (!changed) {
                return;
            }

            options.watermark_text = requested.watermark_text;
            options.watermark_color = requested.watermark_color;
            watermark_enabled = requested_enabled;
            if (!was_enabled && watermark_enabled) {
                counter_disabled = true;
            }
            if (announce) {
                std::cout << "acmxvk: interface watermark "
                          << (watermark_enabled ? "enabled" : "disabled");
                if (watermark_enabled) {
                    std::cout << " (color="
                              << static_cast<int>(options.watermark_color[0])
                              << ','
                              << static_cast<int>(options.watermark_color[1])
                              << ','
                              << static_cast<int>(options.watermark_color[2])
                              << ')';
                }
                std::cout << '\n';
            }
        }

        void apply_interface_gpu_filter_state(
            const InterfaceGpuFilterState &requested, bool announce) {
#ifdef ACMXVK_WITH_CUDA
            const bool requested_enabled =
                requested.enabled && !requested.filter_indices.empty();
            const bool currently_enabled = gpu_filter_engine != nullptr;
            const std::vector<int> effective_indices =
                requested_enabled ? requested.filter_indices
                                  : std::vector<int>{};
            if (requested_enabled == currently_enabled &&
                options.gpu_filter_indices == effective_indices &&
                (!requested_enabled ||
                 options.gpu_frame_buffer_size ==
                     requested.frame_buffer_size)) {
                return;
            }

            if (requested.enabled && requested.filter_indices.empty()) {
                std::cerr << "acmxvk: rejected enabled interface GPU-filter "
                             "state without any filter indices\n";
                return;
            }

            std::unique_ptr<gpu::FilterEngine> replacement;
            if (requested_enabled) {
                try {
                    replacement = std::make_unique<gpu::FilterEngine>(
                        requested.filter_indices,
                        requested.frame_buffer_size);
                } catch (const std::exception &error) {
                    std::cerr
                        << "acmxvk: rejected interface GPU-filter state: "
                        << error.what() << '\n';
                    return;
                }
            }

            gpu_filter_engine = std::move(replacement);
            options.gpu_filter_indices = effective_indices;
            if (requested_enabled) {
                options.gpu_frame_buffer_size = requested.frame_buffer_size;
            }

            if (frame_sprite != nullptr &&
                source_kind == SourceKind::Graphic && !graphic_rgba.empty()) {
                uploadInputFrame(graphic_rgba);
                if (history_initialized) {
                    updateHistoryFrame(graphic_rgba);
                    history_delay_counter = 0;
                }
            }

            if (announce) {
                std::cout << "acmxvk: interface CUDA filter chain "
                          << (requested_enabled ? "enabled" : "disabled");
                if (requested_enabled) {
                    std::cout << " (" << requested.filter_indices.size()
                              << " filters, " << requested.frame_buffer_size
                              << " history frames)";
                }
                std::cout << '\n';
            }
#else
            if (announce && requested.enabled) {
                std::cerr << "acmxvk: ignored interface GPU-filter state: this "
                             "build does not include acidcam-gpu\n";
            }
#endif
        }

        void apply_interface_audio_file_state(
            const InterfaceAudioFileState &requested) {
#ifdef AUDIO_ENABLED
            if (file_audio_source == nullptr || audio_engine == nullptr) {
                std::cerr
                    << "acmxvk: ignored live audio-file change because this "
                       "process was not started in audio-file mode\n";
                return;
            }
            if (requested.path.empty()) {
                std::cerr
                    << "acmxvk: rejected empty interface audio-file request\n";
                return;
            }

            auto replacement = std::make_unique<audio::FileAudioSource>();
            try {
                if (!replacement->open(requested.path)) {
                    std::cerr << "acmxvk: could not switch file audio to: "
                              << requested.path << '\n';
                    return;
                }
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: rejected interface audio-file request: "
                          << error.what() << '\n';
                return;
            }

            replacement->set_repeat(requested.repeat);
            if (requested.pass_through &&
                !replacement->enable_output(
                    requested.output_device,
                    static_cast<float>(options.audio_pass_through_gain))) {
                std::cerr
                    << "acmxvk: live audio-file output could not be opened; "
                       "continuing with visual reactivity only\n";
            }

            file_audio_source->stop_output();
            file_audio_source = std::move(replacement);
            options.audio_file = requested.path;
            options.audio_output_device = requested.output_device;
            options.audio_pass_through = requested.pass_through;
            options.audio_trunc = requested.trunc;
            options.audio_repeat = requested.repeat;
            audio_engine->reset();
            resetAudioWarmup();
            std::cout << "acmxvk: switched file audio to: "
                      << file_audio_source->path() << " (repeat="
                      << (options.audio_repeat ? "on" : "off")
                      << ", trunc=" << (options.audio_trunc ? "on" : "off")
                      << ", pass-through="
                      << (options.audio_pass_through ? "on" : "off")
                      << ")\n";
#else
            static_cast<void>(requested);
            std::cerr << "acmxvk: ignored interface audio-file request: this "
                         "build does not include audio support\n";
#endif
        }

        void apply_interface_shader_reload(
            const InterfaceReloadState &requested) {
            if (requested.path.empty()) {
                std::cerr
                    << "acmxvk: rejected empty interface shader reload\n";
                return;
            }

            try {
                input::validate_string(requested.path,
                                       input::StringKind::Path,
                                       "interface shader reload path");
            } catch (const std::exception &error) {
                std::cerr << "acmxvk: rejected interface shader reload: "
                          << error.what() << '\n';
                return;
            }

            std::error_code error;
            const fs::path requested_path =
                fs::weakly_canonical(requested.path, error);
            if (error || requested_path.empty() ||
                !fs::is_regular_file(requested_path)) {
                std::cerr << "acmxvk: interface shader reload file is not "
                             "readable: "
                          << requested.path << '\n';
                return;
            }

            const auto shader_match = std::find_if(
                shaders.begin(), shaders.end(),
                [&](const fs::path &shader) {
                    std::error_code shader_error;
                    const fs::path canonical_shader =
                        fs::weakly_canonical(shader, shader_error);
                    return !shader_error &&
                           canonical_shader == requested_path;
                });
            if (shader_match == shaders.end()) {
                std::cerr << "acmxvk: interface shader reload is outside the "
                             "active runtime library: "
                          << requested_path.string() << '\n';
                return;
            }

            mxvk::ShaderModuleInfo module_info;
            try {
                input::validate_spirv_file(requested_path,
                                           "interface shader reload");
                module_info = mxvk::inspect_spirv(
                    mxvk::load_spv(requested_path.string()));
            } catch (const std::exception &reload_error) {
                std::cerr << "acmxvk: rejected compiled shader reload: "
                          << reload_error.what() << '\n';
                return;
            }

            const bool history_before = shader_history_required;
            const bool spectrum_before = shader_spectrum_required;
            const bool spectrum_history_before =
                shader_spectrum_history_required;
            recordShaderResources(module_info, "live shader reload");
            const bool resources_grew =
                history_before != shader_history_required ||
                spectrum_before != shader_spectrum_required ||
                spectrum_history_before !=
                    shader_spectrum_history_required;

            const std::vector<fs::path> active_pipeline =
                activeShaderPipeline();
            const bool active =
                std::find(active_pipeline.begin(), active_pipeline.end(),
                          *shader_match) != active_pipeline.end();
            if (active && frame_sprite != nullptr) {
                if (model_effect_shader == *shader_match) {
                    model_effect_shader.clear();
                }
                if (resources_grew) {
                    initializeSprite();
                } else {
                    beginCrossfade();
                    applyShaderPipeline();
                }
                std::cout << "acmxvk: live reloaded active "
                          << (module_info.stage == mxvk::ShaderStage::Compute
                                  ? "compute"
                                  : "fragment")
                          << " shader: " << requested_path.string() << '\n';
            } else {
                std::cout << "acmxvk: live compiled shader ready for its next "
                             "use: "
                          << requested_path.string() << '\n';
            }
        }

        void apply_interface_multipass_state(
            const InterfaceMultipassState &requested) {
            std::vector<fs::path> requested_passes;
            if (requested.enabled) {
                if (requested.shader_names.empty()) {
                    std::cerr << "acmxvk: rejected enabled interface multipass "
                                 "state without any shader passes\n";
                    return;
                }
                requested_passes.reserve(requested.shader_names.size());
                for (const std::string &name : requested.shader_names) {
                    const fs::path requested_path(name);
                    const bool has_parent_reference = std::any_of(
                        requested_path.begin(), requested_path.end(),
                        [](const fs::path &part) { return part == ".."; });
                    if (requested_path.is_absolute() || has_parent_reference) {
                        std::cerr << "acmxvk: rejected unsafe interface "
                                     "multipass shader name: "
                                  << name << '\n';
                        return;
                    }
                    const fs::path shader = findShader(name);
                    if (shader.empty()) {
                        std::cerr << "acmxvk: interface multipass shader is not "
                                     "in the active library: "
                                  << name << '\n';
                        return;
                    }
                    requested_passes.push_back(shader);
                }
            }

            const bool requested_enabled =
                requested.enabled && !requested_passes.empty();
            if (multipass_enabled == requested_enabled &&
                configured_passes == requested_passes) {
                return;
            }
            if (frame_sprite != nullptr && shader_locked) {
                std::cerr << "acmxvk: interface multipass update ignored while "
                             "shader switching is locked\n";
                return;
            }

            if (frame_sprite != nullptr) {
                beginCrossfade();
            }
            configured_passes = std::move(requested_passes);
            multipass_enabled = requested_enabled;
            if (frame_sprite != nullptr) {
                applyShaderPipeline();
                resetShaderTime();
                autopilot_counter = 0;
            }

            if (multipass_enabled) {
                std::cout << "acmxvk: interface multipass enabled ("
                          << configured_passes.size() << " passes)";
                for (const fs::path &shader : configured_passes) {
                    std::cout << "\n  " << shader.filename().string();
                }
                std::cout << '\n';
            } else {
                std::cout << "acmxvk: interface multipass disabled\n";
            }
        }

        void apply_interface_shader_selection(
            const std::string &requested_name) {
            if (requested_name.empty()) {
                return;
            }

            const fs::path requested(requested_name);
            const bool has_parent_reference =
                std::any_of(requested.begin(), requested.end(),
                            [](const fs::path &part) { return part == ".."; });
            if (requested.is_absolute() || has_parent_reference) {
                std::cerr << "acmxvk: rejected unsafe interface shader name: "
                          << requested_name << '\n';
                return;
            }

            const fs::path shader = findShader(requested_name);
            const auto match = std::find(shaders.begin(), shaders.end(), shader);
            if (shader.empty() || match == shaders.end()) {
                std::cerr << "acmxvk: interface shader is not in the active "
                             "library: "
                          << requested_name << '\n';
                return;
            }

            const std::size_t next_index =
                static_cast<std::size_t>(std::distance(shaders.begin(), match));
            if (next_index == shader_index) {
                return;
            }
            if (shader_locked || frame_sprite == nullptr) {
                std::cerr << "acmxvk: interface shader selection ignored while "
                             "shader switching is locked\n";
                return;
            }

            beginCrossfade();
            shader_index = next_index;
            applyShaderPipeline();
            resetShaderTime();
            autopilot_counter = 0;
            std::cout << "acmxvk: interface selected " << activeShaderRole()
                      << ' ' << (shader_index + 1) << '/' << shaders.size()
                      << ": " << currentShader() << '\n';
        }

        void apply_interface_uniform_values(
            const std::vector<InterfaceUniformValue> &uniform_values) {
            if (uniform_values.empty()) {
                return;
            }

            std::size_t changed_count = 0;
            std::size_t ignored_count = 0;
            for (const InterfaceUniformValue &incoming : uniform_values) {
                if (!isValidCustomUniformName(incoming.name) ||
                    !std::isfinite(incoming.value)) {
                    ++ignored_count;
                    continue;
                }
                const auto match = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &uniform) {
                        return uniform.name == incoming.name;
                    });
                if (match == custom_uniforms.end()) {
                    ++ignored_count;
                    continue;
                }
                const std::size_t index = static_cast<std::size_t>(
                    std::distance(custom_uniforms.begin(), match));
                if (index >= custom_uniform_values.size()) {
                    ++ignored_count;
                    continue;
                }
                const float value = static_cast<float>(std::clamp(
                    static_cast<double>(incoming.value), match->minimum,
                    match->maximum));
                if (custom_uniform_values[index] == value) {
                    continue;
                }
                custom_uniform_values[index] = value;
                ++changed_count;
            }

            if (changed_count > 0) {
                uploadCustomUniforms();
                std::cout << "acmxvk: interface updated " << changed_count
                          << " custom uniform(s)\n";
            }
            if (ignored_count > 0) {
                std::cerr << "acmxvk: interface ignored " << ignored_count
                          << " unknown or invalid custom uniform(s)\n";
            }
        }

        [[nodiscard]] fs::path findShader(std::string name) const {
            name = trim(std::move(name));
            if (name.empty()) {
                return {};
            }

            fs::path requested(name);
            if (requested.extension() != ".spv") {
                requested += ".spv";
            }
            const auto match = std::find_if(shaders.begin(), shaders.end(),
                                            [&](const fs::path &shader) {
                                                return shader.filename() == requested.filename() ||
                                                       (!shader_library_directory.empty() &&
                                                        shader.lexically_relative(
                                                            shader_library_directory) ==
                                                            requested);
                                            });
            return match == shaders.end() ? fs::path{} : *match;
        }

        void loadShaderPasses() {
            for (const int index : options.shader_pass_indices) {
                if (index < 0 || index >= static_cast<int>(shaders.size())) {
                    throw std::runtime_error("shader pass index is out of range: " +
                                             std::to_string(index));
                }
                configured_passes.push_back(shaders[static_cast<std::size_t>(index)]);
            }
            for (const std::string &name : options.shader_pass_files) {
                const fs::path shader = findShader(name);
                if (shader.empty()) {
                    throw std::runtime_error("shader pass file is not listed in the manifest: " +
                                             name);
                }
                configured_passes.push_back(shader);
            }
            multipass_enabled = !configured_passes.empty();
        }

        void loadPlaylist() {
            if (options.playlist_file.empty()) {
                return;
            }

            input::validate_file_size(options.playlist_file,
                                      "shader playlist");
            std::ifstream playlist_input(options.playlist_file);
            if (!playlist_input) {
                throw std::runtime_error("unable to open playlist: " + options.playlist_file);
            }

            PlaylistNode *current_node = nullptr;
            std::vector<fs::path> default_entries;
            std::string line;
            std::size_t line_number = 1;
            std::size_t entry_count = 0;
            while (input::read_bounded_line(
                playlist_input, line, "shader playlist", line_number++)) {
                line = trim(std::move(line));
                if (line.empty() || line.front() == '#') {
                    continue;
                }
                if (line.size() >= 2 && line.front() == '[' && line.back() == ']') {
                    if (playlist.size() >= input::MAX_PLAYLIST_NODES) {
                        throw std::runtime_error(
                            "shader playlist contains too many nodes");
                    }
                    std::string node_name =
                        trim(line.substr(1, line.size() - 2));
                    input::validate_string(node_name,
                                           input::StringKind::DisplayText,
                                           "shader playlist node");
                    playlist.push_back({std::move(node_name), {}});
                    current_node = &playlist.back();
                    continue;
                }
                if (line.front() == '[' || line.back() == ']') {
                    throw std::runtime_error(
                        "malformed shader playlist node at line " +
                        std::to_string(line_number - 1));
                }
                if (++entry_count > input::MAX_PLAYLIST_ENTRIES) {
                    throw std::runtime_error(
                        "shader playlist contains too many entries");
                }
                input::validate_string(line, input::StringKind::Path,
                                       "shader playlist entry");

                const fs::path shader = findShader(line);
                if (shader.empty()) {
                    std::cerr << "acmxvk: playlist shader not found: " << line << '\n';
                    continue;
                }
                if (current_node != nullptr) {
                    current_node->shaders.push_back(shader);
                } else {
                    default_entries.push_back(shader);
                }
            }

            playlist.erase(std::remove_if(playlist.begin(), playlist.end(),
                                          [](const PlaylistNode &node) {
                                              return node.shaders.empty();
                                          }),
                           playlist.end());
            if (!default_entries.empty()) {
                playlist.insert(playlist.begin(), {"Default", std::move(default_entries)});
            }
            if (playlist.empty()) {
                throw std::runtime_error("playlist contains no shaders available in the SPIR-V library");
            }
            playlist_enabled = options.enable_playlist;

            std::size_t shader_count = 0;
            for (const PlaylistNode &node : playlist) {
                shader_count += node.shaders.size();
            }
            std::cout << "acmxvk: playlist loaded " << shader_count << " shaders in "
                      << playlist.size() << " nodes from " << options.playlist_file << '\n';
            logSelectedPlaylistNode("selected");
        }

        [[nodiscard]] std::string spriteVertexShader() const {
            const fs::path resource =
                findResource(options, "shaders/sprite.vert.spv");
            if (!resource.empty()) {
                return resource.string();
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_SPRITE_VERTEX_SHADER)) {
                return ACMXVK_INSTALL_SPRITE_VERTEX_SHADER;
            }
            return ACMXVK_BUILD_SPRITE_VERTEX_SHADER;
        }

        [[nodiscard]] std::string echoCacheShader() const {
            const fs::path resource =
                findResource(options, "shaders/echo_cache.frag.spv");
            if (!resource.empty()) {
                return resource.string();
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_ECHO_CACHE_SHADER)) {
                return ACMXVK_INSTALL_ECHO_CACHE_SHADER;
            }
            return ACMXVK_BUILD_ECHO_CACHE_SHADER;
        }

        [[nodiscard]] fs::path flipShader() const {
            const fs::path resource =
                findResource(options, "shaders/flip.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_FLIP_SHADER)) {
                return ACMXVK_INSTALL_FLIP_SHADER;
            }
            return ACMXVK_BUILD_FLIP_SHADER;
        }

        [[nodiscard]] fs::path passthroughShader() const {
            const fs::path resource =
                findResource(options, "shaders/passthrough.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_PASSTHROUGH_SHADER)) {
                return ACMXVK_INSTALL_PASSTHROUGH_SHADER;
            }
            return ACMXVK_BUILD_PASSTHROUGH_SHADER;
        }

        [[nodiscard]] fs::path humanCompositeShader() const {
            const fs::path resource =
                findResource(options, "shaders/human_composite.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER)) {
                return ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER;
            }
            return ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER;
        }

        [[nodiscard]] fs::path crossfadeShader() const {
            const std::string filename =
                std::string(CROSSFADE_NAMES[crossfade_shader_index]) +
                ".frag.spv";
            const fs::path resource =
                findResource(options, fs::path("shaders/xfade") / filename);
            if (!resource.empty()) {
                return resource;
            }
            const fs::path installed =
                fs::path(ACMXVK_INSTALL_CROSSFADE_DIRECTORY) / filename;
            if (fs::is_regular_file(installed)) {
                return installed;
            }
            const fs::path built =
                fs::path(ACMXVK_BUILD_CROSSFADE_DIRECTORY) / filename;
            if (fs::is_regular_file(built)) {
                return built;
            }
            throw std::runtime_error("crossfade shader was not found: " +
                                     filename);
        }

        [[nodiscard]] fs::path modelVertexShader() const {
            const fs::path resource =
                findResource(options, "shaders/model.vert.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_MODEL_VERTEX_SHADER)) {
                return ACMXVK_INSTALL_MODEL_VERTEX_SHADER;
            }
            return ACMXVK_BUILD_MODEL_VERTEX_SHADER;
        }

        [[nodiscard]] fs::path modelFragmentShader() const {
            const fs::path resource =
                findResource(options, "shaders/model.frag.spv");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER)) {
                return ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER;
            }
            return ACMXVK_BUILD_MODEL_FRAGMENT_SHADER;
        }

        [[nodiscard]] fs::path defaultModel() const {
            const fs::path resource = findResource(options, "models/cube.obj");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_DEFAULT_MODEL)) {
                return ACMXVK_INSTALL_DEFAULT_MODEL;
            }
            return ACMXVK_BUILD_DEFAULT_MODEL;
        }

        [[nodiscard]] fs::path overlayFont() const {
            const fs::path resource = findResource(options, "data/font.ttf");
            if (!resource.empty()) {
                return resource;
            }
            if (fs::is_regular_file(ACMXVK_INSTALL_OVERLAY_FONT)) {
                return ACMXVK_INSTALL_OVERLAY_FONT;
            }
            return ACMXVK_BUILD_OVERLAY_FONT;
        }
