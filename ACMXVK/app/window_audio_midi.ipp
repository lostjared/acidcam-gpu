#ifdef AUDIO_ENABLED
        std::unique_ptr<audio::AudioEngine> audio_engine;
        std::unique_ptr<audio::FileAudioSource> file_audio_source;
        float audio_warmup_envelope = 0.0F;
        bool audio_warmup_started = false;
        std::chrono::steady_clock::time_point audio_warmup_last_tick{};

        // Audio, MIDI, custom controls, and media-clock coordination.
        void resetAudioWarmup() {
            audio_warmup_envelope = 0.0F;
            audio_warmup_started = false;
            if (options.audio_warm_rate <= 0.0) {
                std::cout << "acmxvk: audio shader warmup disabled\n";
            } else {
                std::cout << "acmxvk: audio shader warmup "
                          << options.audio_warm_rate << "/second (~"
                          << 1.0 / options.audio_warm_rate
                          << " seconds to full strength)\n";
            }
        }

        [[nodiscard]] float updateAudioWarmup(
            std::chrono::steady_clock::time_point now) {
            if (options.audio_warm_rate <= 0.0) {
                audio_warmup_envelope = 1.0F;
                return audio_warmup_envelope;
            }
            if (!audio_warmup_started) {
                audio_warmup_started = true;
                audio_warmup_last_tick = now;
                return audio_warmup_envelope;
            }

            const float delta = std::max(
                std::chrono::duration<float>(now - audio_warmup_last_tick).count(),
                0.0F);
            audio_warmup_last_tick = now;
            audio_warmup_envelope = std::min(
                audio_warmup_envelope +
                    delta * static_cast<float>(options.audio_warm_rate),
                1.0F);
            return audio_warmup_envelope;
        }
#endif

        void initializeDnn() {
#ifdef ACMXVK_WITH_DNN
            if (!options.human_model.empty()) {
                human_segmenter =
                    std::make_unique<dnn::HumanSegmenter>(options.human_model);
                std::cout << "acmxvk: PP-HumanSeg enabled: "
                          << options.human_model << " ("
                          << (options.human_background
                                  ? "background-only shader composition"
                                  : "foreground isolation")
                          << ", automatic CPU/CUDA backend selection)\n";
            }
            if (!options.edge_model.empty()) {
                edge_detector =
                    std::make_unique<dnn::EdgeDetector>(options.edge_model);
                std::cout << "acmxvk: DexiNed edge detection enabled: "
                          << options.edge_model
                          << " (automatic CPU/CUDA backend selection)\n";
            }
            if (!options.onnx_configuration.empty()) {
                generic_onnx_processor =
                    std::make_unique<dnn::GenericOnnxProcessor>(
                        options.onnx_configuration);
                std::cout << "acmxvk: generic ONNX processing enabled: "
                          << options.onnx_configuration
                          << " (automatic CPU/CUDA backend selection)\n";
            }
#endif
        }

        void initializeGpuFilters() {
#ifdef ACMXVK_WITH_CUDA
            if (options.gpu_filter_indices.empty()) {
                return;
            }
            gpu_filter_engine = std::make_unique<gpu::FilterEngine>(
                options.gpu_filter_indices, options.gpu_frame_buffer_size);
#endif
        }

        void selectGpuFilter(int direction) {
#ifdef ACMXVK_WITH_CUDA
            if (gpu_filter_engine != nullptr &&
                gpu_filter_engine->select_relative_filter(direction) &&
                source_kind == SourceKind::Graphic && !graphic_rgba.empty()) {
                uploadInputFrame(graphic_rgba);
                if (history_initialized) {
                    updateHistoryFrame(graphic_rgba);
                    history_delay_counter = 0;
                }
            }
#else
            static_cast<void>(direction);
#endif
        }

        void openMidi() {
#ifdef MIDI_ENABLED
            if (!options.midi_device_specified && !options.midi_monitor &&
                options.midi_map_file.empty() && midi_cc_mappings.empty()) {
                return;
            }
            midi_input = std::make_unique<midi::MidiInput>();
            const int port = options.midi_device_specified ? options.midi_device : 0;
            if (!midi_input->open(port)) {
                throw std::runtime_error("could not open MIDI input port " +
                                         std::to_string(port));
            }
#endif
        }

        void configureMidiMappings() {
#ifdef MIDI_ENABLED
            if (!options.midi_map_file.empty()) {
                midi_action_mappings =
                    midi::load_mapping_file(options.midi_map_file);
                midi_knob_states.resize(midi_action_mappings.size());
                std::cout << "acmxvk: loaded " << midi_action_mappings.size()
                          << " MIDI mapping(s) from " << options.midi_map_file
                          << '\n';

                for (int slider = 0; slider < 4; ++slider) {
                    const int action = 600 + slider * 2;
                    const bool mapped = std::any_of(
                        midi_action_mappings.begin(),
                        midi_action_mappings.end(),
                        [&](const midi::MidiMapping &mapping) {
                            return mapping.primary_action == action &&
                                   mapping.secondary_action == action + 1;
                        });
                    if (!mapped) {
                        continue;
                    }
                    const std::string name =
                        "slider" + std::to_string(slider + 1);
                    const auto uniform = std::find_if(
                        custom_uniforms.begin(), custom_uniforms.end(),
                        [&](const ShaderManifest::CustomUniform &candidate) {
                            return candidate.name == name;
                        });
                    if (uniform == custom_uniforms.end()) {
                        std::cerr << "acmxvk: MIDI " << name
                                  << " mapping has no matching custom uniform in "
                                     "library.json\n";
                        continue;
                    }
                    midi_slider_uniform_indices[slider] = static_cast<int>(
                        std::distance(custom_uniforms.begin(), uniform));
                    std::cout << "acmxvk: MIDI Slider " << (slider + 1)
                              << " -> " << name << " [" << uniform->minimum
                              << ", " << uniform->maximum << "]\n";
                }

                std::size_t active_mappings = 0;
                for (const midi::MidiMapping &mapping :
                     midi_action_mappings) {
                    if (isMidiMappingSupported(mapping)) {
                        ++active_mappings;
                    } else if (options.midi_monitor) {
                        std::cerr
                            << "acmxvk: MIDI map action unavailable in this build: "
                            << mapping.primary_action << ':'
                            << mapping.secondary_action << '\n';
                    }
                }
                std::cout << "acmxvk: MIDI map has " << active_mappings
                          << " active mapping(s)";
                if (active_mappings != midi_action_mappings.size()) {
                    std::cout << " and "
                              << (midi_action_mappings.size() - active_mappings)
                              << " mapping(s) reserved for unported ACMX2 controls";
                }
                std::cout << '\n';
            }

            for (const std::string &mapping_text : options.midi_cc_mappings) {
                const std::size_t equals = mapping_text.find('=');
                if (equals == std::string::npos || equals == 0 ||
                    equals + 1 >= mapping_text.size() ||
                    mapping_text.find('=', equals + 1) != std::string::npos) {
                    throw std::runtime_error(
                        "--midi-cc requires [channel:]CC=uniform: " +
                        mapping_text);
                }

                const std::string source = trim(mapping_text.substr(0, equals));
                const std::string uniform_name =
                    trim(mapping_text.substr(equals + 1));
                if (!isValidCustomUniformName(uniform_name)) {
                    throw std::runtime_error(
                        "--midi-cc contains an invalid uniform name: " +
                        uniform_name);
                }

                int channel = -1;
                int controller = 0;
                const std::size_t colon = source.find(':');
                if (colon == std::string::npos) {
                    controller = parseInteger(source, "--midi-cc");
                } else {
                    if (colon == 0 || colon + 1 >= source.size() ||
                        source.find(':', colon + 1) != std::string::npos) {
                        throw std::runtime_error(
                            "--midi-cc requires [channel:]CC=uniform: " +
                            mapping_text);
                    }
                    channel = parseInteger(
                        std::string_view(source).substr(0, colon), "--midi-cc");
                    controller = parseInteger(
                        std::string_view(source).substr(colon + 1), "--midi-cc");
                    if (channel < 1 || channel > 16) {
                        throw std::runtime_error(
                            "--midi-cc channel must be between 1 and 16");
                    }
                    --channel;
                }
                if (controller < 0 || controller > 127) {
                    throw std::runtime_error(
                        "--midi-cc controller must be between 0 and 127");
                }

                const auto uniform = std::find_if(
                    custom_uniforms.begin(), custom_uniforms.end(),
                    [&](const ShaderManifest::CustomUniform &candidate) {
                        return candidate.name == uniform_name;
                    });
                if (uniform == custom_uniforms.end()) {
                    throw std::runtime_error(
                        "--midi-cc target is not defined in library.json: " +
                        uniform_name);
                }
                const std::size_t uniform_index = static_cast<std::size_t>(
                    std::distance(custom_uniforms.begin(), uniform));
                const auto duplicate = std::find_if(
                    midi_cc_mappings.begin(), midi_cc_mappings.end(),
                    [&](const MidiCcMapping &mapping) {
                        return mapping.uniform_index == uniform_index;
                    });
                if (duplicate != midi_cc_mappings.end()) {
                    throw std::runtime_error(
                        "custom uniform has more than one --midi-cc mapping: " +
                        uniform_name);
                }

                midi_cc_mappings.push_back(
                    {channel, controller, uniform_index, uniform_name});
                std::cout << "acmxvk: MIDI "
                          << (channel < 0
                                  ? std::string("any channel")
                                  : "channel " + std::to_string(channel + 1))
                          << " CC " << controller << " -> " << uniform_name
                          << " [" << uniform->minimum << ", "
                          << uniform->maximum << "]\n";
            }
#endif
        }

#ifdef MIDI_ENABLED
        [[nodiscard]] bool applyMidiCc(const midi::MidiMessage &message) {
            if (message.bytes.size() < 3 ||
                (message.bytes[0] & 0xF0U) != 0xB0U) {
                return false;
            }
            const int channel = message.bytes[0] & 0x0FU;
            const int controller = message.bytes[1] & 0x7FU;
            const int value = message.bytes[2] & 0x7FU;
            bool changed = false;
            for (const MidiCcMapping &mapping : midi_cc_mappings) {
                if (mapping.controller != controller ||
                    (mapping.channel >= 0 && mapping.channel != channel)) {
                    continue;
                }
                const ShaderManifest::CustomUniform &uniform =
                    custom_uniforms[mapping.uniform_index];
                const double normalized = static_cast<double>(value) / 127.0;
                const float mapped = static_cast<float>(
                    uniform.minimum + normalized *
                                          (uniform.maximum - uniform.minimum));
                custom_uniform_values[mapping.uniform_index] = mapped;
                changed = true;
                if (options.midi_monitor) {
                    std::cout << "acmxvk: MIDI CC " << controller << " -> "
                              << mapping.uniform_name << '=' << mapped << '\n';
                }
            }
            return changed;
        }

        [[nodiscard]] SDL_Keycode midiActionKey(int action) const {
            switch (action) {
            case 262:
#ifdef ACMXVK_WITH_CUDA
                return gpu_filter_engine != nullptr ? SDLK_RIGHT : SDLK_UNKNOWN;
#else
                return SDLK_UNKNOWN;
#endif
            case 263:
#ifdef ACMXVK_WITH_CUDA
                return gpu_filter_engine != nullptr ? SDLK_LEFT : SDLK_UNKNOWN;
#else
                return SDLK_UNKNOWN;
#endif
            case 264:
                return SDLK_DOWN;
            case 265:
                return SDLK_UP;
            case 266:
            case 504:
                return SDLK_PAGEUP;
            case 267:
            case 505:
                return SDLK_PAGEDOWN;
            case 268:
#ifdef AUDIO_ENABLED
                return SDLK_HOME;
#else
                return SDLK_UNKNOWN;
#endif
            case 269:
#ifdef AUDIO_ENABLED
                return SDLK_END;
#else
                return SDLK_UNKNOWN;
#endif
            case 260:
                return SDLK_INSERT;
            case 261:
                return SDLK_DELETE;
            case 500:
                return SDLK_U;
            case 501:
                return SDLK_I;
            case 298:
                return SDLK_F9;
            case 32:
                return SDLK_SPACE;
            case 44:
                return options.enable_3d ? SDLK_COMMA : SDLK_UNKNOWN;
            case 46:
                return options.enable_3d ? SDLK_PERIOD : SDLK_UNKNOWN;
            case 51:
                return options.enable_3d ? SDLK_3 : SDLK_UNKNOWN;
            case 52:
                return SDLK_4;
            case 53:
                return SDLK_5;
            case 54:
                return SDLK_6;
            case 67:
                return options.enable_3d ? SDLK_C : SDLK_UNKNOWN;
            case 79:
                return options.enable_3d ? SDLK_O : SDLK_UNKNOWN;
            case 91:
                return options.enable_3d ? SDLK_MINUS : SDLK_UNKNOWN;
            case 93:
                return options.enable_3d ? SDLK_EQUALS : SDLK_UNKNOWN;
            case 69:
                return options.watermark_text.empty() ? SDLK_UNKNOWN : SDLK_E;
            case 74:
                return SDLK_J;
            case 78:
                return SDLK_N;
            case 75:
                return SDLK_K;
            case 76:
                return SDLK_L;
            case 77:
                return SDLK_M;
            case 80:
                return SDLK_P;
            case 81:
#ifdef AUDIO_ENABLED
                return SDLK_Q;
#else
                return SDLK_UNKNOWN;
#endif
            case 70:
                return SDLK_F;
            case 73:
                return SDLK_I;
            case 84:
                return SDLK_T;
            case 85:
                return SDLK_U;
            case 86:
                return options.enable_3d ? SDLK_V : SDLK_UNKNOWN;
            case 88:
                return options.enable_3d ? SDLK_X : SDLK_UNKNOWN;
            case 89:
                return SDLK_Y;
            case 90:
                return SDLK_Z;
            default:
                return SDLK_UNKNOWN;
            }
        }

        [[nodiscard]] bool isMidiSliderMapping(
            const midi::MidiMapping &mapping) const {
            return mapping.primary_action >= 600 &&
                   mapping.primary_action <= 606 &&
                   mapping.primary_action % 2 == 0 &&
                   mapping.secondary_action == mapping.primary_action + 1;
        }

        [[nodiscard]] static bool usesMidiDeltaDirection(
            const midi::MidiMapping &mapping) {
            return mapping.primary_action == 506 ||
                   mapping.primary_action == 508 ||
                   mapping.primary_action == 512;
        }

        [[nodiscard]] bool isMidiModelAction(int action) const {
            return options.enable_3d && action >= 506 && action <= 515;
        }

        [[nodiscard]] bool isMidiMappingSupported(
            const midi::MidiMapping &mapping) const {
            if (isMidiSliderMapping(mapping)) {
                const int slider = (mapping.primary_action - 600) / 2;
                return midi_slider_uniform_indices[slider] >= 0;
            }
            if (mapping.secondary_action == 0) {
                return isMidiModelAction(mapping.primary_action) ||
                       midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
            }
            const bool primary_supported =
                isMidiModelAction(mapping.primary_action) ||
                midiActionKey(mapping.primary_action) != SDLK_UNKNOWN;
            const bool secondary_supported =
                isMidiModelAction(mapping.secondary_action) ||
                midiActionKey(mapping.secondary_action) != SDLK_UNKNOWN;
            return primary_supported && secondary_supported;
        }

        [[nodiscard]] std::string_view midiActionName(int action) const {
            switch (action) {
            case 262:
                return "select next CUDA filter";
            case 263:
                return "select previous CUDA filter";
            case 264:
                return "next shader or playlist node";
            case 265:
                return "previous shader or playlist node";
            case 266:
            case 504:
                return "increase shader time speed";
            case 267:
            case 505:
                return "decrease shader time speed";
            case 268:
                return "toggle audio delta-time scaling";
            case 269:
                return "toggle spectrum sensitivity scaling";
            case 260:
                return "increase audio sensitivity";
            case 261:
                return "decrease audio sensitivity";
            case 500:
                return "step shader time forward";
            case 501:
                return "step shader time backward";
            case 506:
                return "rotate model X forward";
            case 507:
                return "rotate model X backward";
            case 508:
                return "rotate model Y forward";
            case 509:
                return "rotate model Y backward";
            case 510:
                return "increase 3D manual rotation speed";
            case 511:
                return "decrease 3D manual rotation speed";
            case 512:
                return "rotate model Z forward";
            case 513:
                return "rotate model Z backward";
            case 514:
                return "increase model scale";
            case 515:
                return "decrease model scale";
            case 298:
                return "toggle runtime HUD";
            case 32:
                return "toggle shader bypass";
            case 44:
                return "decrease 3D view rotation speed";
            case 46:
                return "increase 3D view rotation speed";
            case 51:
                return "toggle 2D/3D rendering";
            case 52:
                return "take TIFF snapshot";
            case 53:
                return "take WebP snapshot";
            case 54:
                return "take raw RGBA snapshot";
            case 67:
                return "toggle 3D wave effect";
            case 79:
                return "toggle 3D scale oscillation";
            case 91:
                return "decrease model scale";
            case 93:
                return "increase model scale";
            case 69:
                return "toggle watermark";
            case 74:
                return "toggle random autopilot";
            case 78:
                return "toggle random autopilot crossfade";
            case 75:
                return "toggle shader lock";
            case 76:
                return "toggle rendering freeze";
            case 77:
                return "toggle multipass";
            case 80:
                return "toggle playlist or input pause";
            case 81:
                return "toggle audio-reactive shader time";
            case 70:
                return "toggle fullscreen";
            case 73:
                return "step shader time backward";
            case 84:
                return "toggle shader time";
            case 85:
                return "step shader time forward";
            case 86:
                return "toggle 3D view rotation";
            case 88:
                return "reset model view";
            case 89:
                return "toggle sequential autopilot";
            case 90:
                return "take screenshot";
            default:
                return "unsupported action";
            }
        }

        void dispatchMidiModelAction(int action) {
            if (!model_initialized || !isMidiModelAction(action)) {
                return;
            }

            const auto rotate = [](float &degrees, float amount) {
                degrees = std::fmod(degrees + amount, 360.0F);
                if (degrees < 0.0F) {
                    degrees += 360.0F;
                }
            };
            switch (action) {
            case 506:
                rotate(model_rotation_x_degrees,
                       model_camera_rotation_speed * 0.3F);
                break;
            case 507:
                rotate(model_rotation_x_degrees,
                       model_camera_rotation_speed * -0.33F);
                break;
            case 508:
                rotate(model_rotation_y_degrees,
                       model_camera_rotation_speed * 0.3F);
                break;
            case 509:
                rotate(model_rotation_y_degrees,
                       model_camera_rotation_speed * -0.3F);
                break;
            case 510:
                model_camera_rotation_speed = std::clamp(
                    model_camera_rotation_speed + 0.5F, 0.5F, 50.0F);
                std::cout << "acmxvk: 3D manual rotation speed "
                          << model_camera_rotation_speed << '\n';
                break;
            case 511:
                model_camera_rotation_speed = std::clamp(
                    model_camera_rotation_speed - 0.5F, 0.5F, 50.0F);
                std::cout << "acmxvk: 3D manual rotation speed "
                          << model_camera_rotation_speed << '\n';
                break;
            case 512:
                rotate(model_rotation_z_degrees,
                       model_camera_rotation_speed * 0.3F);
                break;
            case 513:
                rotate(model_rotation_z_degrees,
                       model_camera_rotation_speed * -0.3F);
                break;
            case 514:
                adjustModelScale(0.05F);
                break;
            case 515:
                adjustModelScale(-0.05F);
                break;
            default:
                break;
            }
        }

        void dispatchMidiAction(int action) {
            if (isMidiModelAction(action)) {
                if (options.midi_monitor) {
                    std::cout << "acmxvk: MIDI action: "
                              << midiActionName(action) << '\n';
                }
                dispatchMidiModelAction(action);
                return;
            }
            const SDL_Keycode key = midiActionKey(action);
            if (key == SDLK_UNKNOWN) {
                return;
            }
            if (options.midi_monitor) {
                std::cout << "acmxvk: MIDI action: " << midiActionName(action)
                          << '\n';
            }
            SDL_Event midi_event{};
            midi_event.type = SDL_EVENT_KEY_DOWN;
            midi_event.key.type = SDL_EVENT_KEY_DOWN;
            midi_event.key.key = key;
            midi_event.key.mod =
                action == 91 || action == 93 ? SDL_KMOD_SHIFT
                                             : SDL_KMOD_NONE;
            midi_event.key.repeat = false;
            event(midi_event);
        }

        [[nodiscard]] bool setMidiUniform(std::size_t uniform_index, int value,
                                          std::string_view label) {
            if (uniform_index >= custom_uniforms.size() ||
                uniform_index >= custom_uniform_values.size()) {
                return false;
            }
            const ShaderManifest::CustomUniform &uniform =
                custom_uniforms[uniform_index];
            const double normalized = static_cast<double>(value) / 127.0;
            const float mapped = static_cast<float>(
                uniform.minimum +
                normalized * (uniform.maximum - uniform.minimum));
            custom_uniform_values[uniform_index] = mapped;
            if (options.midi_monitor) {
                std::cout << "acmxvk: MIDI " << label << " -> "
                          << uniform.name << '=' << mapped << '\n';
            }
            return true;
        }

        [[nodiscard]] bool applyMidiMap(const midi::MidiMessage &message) {
            if (message.bytes.size() < 3) {
                return false;
            }
            bool changed = false;
            for (std::size_t index = 0; index < midi_action_mappings.size();
                 ++index) {
                const midi::MidiMapping &mapping = midi_action_mappings[index];
                if (message.bytes[0] != mapping.status ||
                    message.bytes[1] != mapping.data1) {
                    continue;
                }
                const int value = message.bytes[2] & 0x7FU;
                if (mapping.secondary_action == 0) {
                    if (message.bytes[2] == mapping.data2) {
                        dispatchMidiAction(mapping.primary_action);
                    }
                    continue;
                }

                if (isMidiSliderMapping(mapping)) {
                    const int slider = (mapping.primary_action - 600) / 2;
                    const int uniform_index =
                        midi_slider_uniform_indices[slider];
                    if (uniform_index >= 0) {
                        changed =
                            setMidiUniform(
                                static_cast<std::size_t>(uniform_index), value,
                                "Slider " + std::to_string(slider + 1)) ||
                            changed;
                    }
                    continue;
                }

                MidiKnobState &state = midi_knob_states[index];
                if (usesMidiDeltaDirection(mapping) &&
                    value != state.previous_value) {
                    state.direction_action =
                        value > state.previous_value
                            ? mapping.primary_action
                            : mapping.secondary_action;
                }
                state.previous_value = value;
                state.value = value;
                state.active = value != 64;
                if (!state.active) {
                    state.frame_counter = 0;
                }
            }
            return changed;
        }

        void dispatchMidiKnobs() {
            for (std::size_t index = 0; index < midi_action_mappings.size();
                 ++index) {
                const midi::MidiMapping &mapping = midi_action_mappings[index];
                MidiKnobState &state = midi_knob_states[index];
                if (!state.active || mapping.secondary_action == 0 ||
                    isMidiSliderMapping(mapping) ||
                    !isMidiMappingSupported(mapping)) {
                    continue;
                }

                const int distance = std::abs(state.value - 64);
                const int frame_skip =
                    std::max(1, 17 - (distance * 16 / 63));
                if (++state.frame_counter < frame_skip) {
                    continue;
                }
                state.frame_counter = 0;
                int action = state.value > 64
                                 ? mapping.primary_action
                                 : mapping.secondary_action;
                if (usesMidiDeltaDirection(mapping)) {
                    action = state.direction_action;
                    if (action == 0) {
                        continue;
                    }
                }
                dispatchMidiAction(action);
            }
        }
#endif

        void uploadCustomUniforms() {
            if (frame_sprite != nullptr) {
                frame_sprite->setCustomUniforms(custom_uniform_values);
            }
            for (mxvk::VK_Sprite *sprite : post_process_sprites) {
                sprite->setCustomUniforms(custom_uniform_values);
            }
        }

        void pollMidi() {
#ifdef MIDI_ENABLED
            if (midi_input == nullptr || !midi_input->is_open()) {
                return;
            }
            const std::vector<midi::MidiMessage> messages =
                midi_input->poll_messages();
            bool custom_uniforms_changed = false;
            for (const midi::MidiMessage &message : messages) {
                custom_uniforms_changed =
                    applyMidiCc(message) || custom_uniforms_changed;
                custom_uniforms_changed =
                    applyMidiMap(message) || custom_uniforms_changed;
            }
            dispatchMidiKnobs();
            if (custom_uniforms_changed) {
                uploadCustomUniforms();
            }
            if (options.midi_monitor) {
                for (const midi::MidiMessage &message : messages) {
                    std::ostringstream text;
                    text << "acmxvk: MIDI #" << message.sequence << " +"
                         << std::fixed << std::setprecision(6)
                         << message.delta_seconds << "s [";
                    for (std::size_t index = 0; index < message.bytes.size();
                         ++index) {
                        if (index > 0) {
                            text << ' ';
                        }
                        text << std::hex << std::uppercase << std::setfill('0')
                             << std::setw(2)
                             << static_cast<unsigned int>(message.bytes[index]);
                    }
                    text << ']';
                    std::cout << text.str() << '\n';
                }
            }
            const std::uint64_t dropped = midi_input->dropped_message_count();
            if (dropped != observed_midi_drops) {
                std::cerr << "acmxvk: MIDI queue dropped " << dropped
                          << " message(s) total\n";
                observed_midi_drops = dropped;
            }
#endif
        }

        void openAudio() {
            if (!options.enable_audio) {
                return;
            }
#ifdef AUDIO_ENABLED
            audio_engine = std::make_unique<audio::AudioEngine>();
            audio_engine->set_sensitivity(
                static_cast<float>(options.audio_sensitivity));
            if (!options.audio_file.empty()) {
                file_audio_source = std::make_unique<audio::FileAudioSource>();
                if (!file_audio_source->open(options.audio_file)) {
                    if (options.use_source_audio) {
                        std::cerr
                            << "acmxvk: source video has no decodable audio "
                               "track; continuing with silent audio-reactive "
                               "values";
                        if (options.audio_pass_through) {
                            std::cerr << " and pass-through disabled";
                        }
                        std::cerr << '\n';
                        file_audio_source.reset();
                        return;
                    }
                    throw std::runtime_error("could not decode --audio-file: " +
                                             options.audio_file);
                }
                if (options.use_source_audio) {
                    std::cout << "acmxvk: source video audio drives shader "
                                 "reactivity\n";
                }
                file_audio_source->set_repeat(options.audio_repeat);
                if (options.audio_pass_through &&
                    !file_audio_source->enable_output(
                        options.audio_output_device,
                        static_cast<float>(options.audio_pass_through_gain))) {
                    std::cerr << "acmxvk: file audio output could not be "
                                 "initialized; continuing with silent analysis\n";
                }
                resetAudioWarmup();
                return;
            }
            const audio::AudioStreamConfig config{
                static_cast<unsigned int>(options.audio_channels),
                static_cast<float>(options.audio_sensitivity),
                options.audio_input_device,
                options.audio_output_device,
                options.audio_pass_through,
                static_cast<float>(options.audio_pass_through_gain),
                static_cast<float>(options.audio_recording_gain),
            };
            if (!audio_engine->open(config)) {
                std::cerr << "acmxvk: audio input could not be initialized; "
                             "continuing with zero-valued audio metrics\n";
                audio_engine.reset();
            } else {
                resetAudioWarmup();
            }
#endif
        }

        void start_requested_audio_recording() {
            if (options.record_audio_file.empty()) {
                return;
            }
#ifdef AUDIO_ENABLED
            if (audio_engine == nullptr || file_audio_source != nullptr ||
                !audio_engine->is_open()) {
                throw std::runtime_error(
                    "--record-audio requires an active live audio input");
            }
            if (!options.output_file.empty() && !options.png_output) {
                return;
            }
            if (!audio_engine->is_recording() &&
                !audio_engine->start_recording()) {
                throw std::runtime_error(
                    "could not start standalone microphone recording");
            }
#endif
        }

        void adjustAudioSensitivity(float amount) {
#ifdef AUDIO_ENABLED
            if (audioSourceOpen()) {
                audio_engine->set_sensitivity(audio_engine->sensitivity() + amount);
                options.audio_sensitivity = audio_engine->sensitivity();
                std::cout << "acmxvk: audio sensitivity "
                          << options.audio_sensitivity << '\n';
                return;
            }
#else
            static_cast<void>(amount);
#endif
            std::cout << "acmxvk: audio input is not active\n";
        }

        [[nodiscard]] bool audioSourceOpen() const {
#ifdef AUDIO_ENABLED
            return audio_engine != nullptr &&
                   (audio_engine->is_open() ||
                    (file_audio_source != nullptr && file_audio_source->is_open()));
#else
            return false;
#endif
        }

        void startLiveAudioRecordingIfNeeded() {
#ifdef AUDIO_ENABLED
            if (audio_engine == nullptr || file_audio_source != nullptr ||
                !audio_engine->is_open() || audio_engine->is_recording() ||
                !writer.is_open() || options.png_output ||
                (options.copy_audio && !options.mute_output &&
                 options.record_audio_file.empty())) {
                return;
            }
            if (!audio_engine->start_recording()) {
                std::cerr << "acmxvk: could not start live audio recording; "
                             "continuing with video-only output\n";
            }
#endif
        }

        void startMediaTimelineIfReady() {
            if (media_timeline_started || !source_frame_received) {
                return;
            }
            media_timeline_started = true;
            hud_session_start = std::chrono::steady_clock::now();
            hud_fps_last_tick = hud_session_start;
            hud_fps_frame_count = 0;
            source_playback_clock_start = hud_session_start;
            source_playback_pause_start = {};
            source_playback_paused_duration = {};
            source_playback_clock_paused = false;
#ifdef AUDIO_ENABLED
            resetAudioWarmup();
#endif
            startLiveAudioRecordingIfNeeded();
            std::cout << "acmxvk: media timeline started on first source frame\n";
        }

        void setSourcePlaybackClockPaused(bool paused) {
            if (!options.use_source_fps || source_kind != SourceKind::Video ||
                !media_timeline_started ||
                paused == source_playback_clock_paused) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (paused) {
                source_playback_pause_start = now;
            } else {
                source_playback_paused_duration +=
                    now - source_playback_pause_start;
            }
            source_playback_clock_paused = paused;
        }

        [[nodiscard]] bool mediaClockSeconds(double &seconds) const {
#ifdef AUDIO_ENABLED
            if (file_audio_source != nullptr &&
                file_audio_source->has_output_clock()) {
                seconds = file_audio_source->playback_time();
                return true;
            }
            if ((!options.copy_audio || options.mute_output) && writer.is_open() &&
                audio_engine != nullptr && file_audio_source == nullptr &&
                audio_engine->is_recording()) {
                seconds = audio_engine->recording_time();
                return true;
            }
#endif
            if (source_kind == SourceKind::Camera &&
                media_timeline_started) {
                seconds = hudWallElapsedSeconds();
                return true;
            }
            if (options.use_source_fps && source_kind == SourceKind::Video &&
                media_timeline_started) {
                const auto clock_end = source_playback_clock_paused
                                           ? source_playback_pause_start
                                           : std::chrono::steady_clock::now();
                const auto active_time =
                    clock_end - source_playback_clock_start -
                    source_playback_paused_duration;
                seconds = std::max(
                    0.0, std::chrono::duration<double>(active_time).count());
                return true;
            }
            seconds = 0.0;
            return false;
        }
