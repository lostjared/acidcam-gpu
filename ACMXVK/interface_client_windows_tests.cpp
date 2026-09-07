#include "app/interface_client.hpp"
#include "interface_control.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

int main() {
#ifdef _WIN32
    HANDLE mutex = ::CreateMutexW(
        nullptr, FALSE, acmxvk::ipc::SHADER_SELECTION_MUTEX_NAME_WINDOWS);
    if (mutex == nullptr) {
        std::cerr << "CreateMutexW failed: " << ::GetLastError() << '\n';
        return 1;
    }

    HANDLE mapping = ::CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        static_cast<DWORD>(sizeof(acmxvk::ipc::ShaderSelectionData)),
        acmxvk::ipc::SHADER_SELECTION_MAPPING_NAME_WINDOWS);
    if (mapping == nullptr) {
        std::cerr << "CreateFileMappingW failed: " << ::GetLastError() << '\n';
        ::CloseHandle(mutex);
        return 1;
    }

    void *view = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0,
                                 sizeof(acmxvk::ipc::ShaderSelectionData));
    if (view == nullptr) {
        std::cerr << "MapViewOfFile failed: " << ::GetLastError() << '\n';
        ::CloseHandle(mapping);
        ::CloseHandle(mutex);
        return 1;
    }

    auto *selection =
        static_cast<acmxvk::ipc::ShaderSelectionData *>(view);
    {
        acmxvk::ipc::InterfaceLock lock(mutex);
        if (!lock) {
            std::cerr << "Could not lock test mapping\n";
            ::UnmapViewOfFile(view);
            ::CloseHandle(mapping);
            ::CloseHandle(mutex);
            return 1;
        }
        *selection = acmxvk::ipc::ShaderSelectionData{};
        selection->sequence = 42;
        selection->shader_pass_enabled = 1;
        selection->shader_pass_count = 2;
        std::strcpy(selection->selected_shader_name, "selected.frag.spv");
        std::strcpy(selection->shader_pass_names[0], "first.comp.spv");
        std::strcpy(selection->shader_pass_names[1], "second.frag.spv");
        selection->custom_uniform_count = 1;
        std::strcpy(selection->custom_uniform_names[0], "slider1");
        selection->custom_uniform_values[0] = 0.75F;
        selection->audio_file_sequence = 7;
        std::strcpy(selection->audio_file_path, "C:/audio/test.wav");
        selection->audio_pass_through = 1;
        selection->dream_enabled = 1;
        selection->dream_fp16 = 1;
        selection->dream_gpu_filter_first = 1;
        selection->dream_iterations = 3;
        selection->dream_maximum_dimension = 768;
        selection->dream_channel = 12;
        selection->dream_octaves = 2;
        selection->dream_jitter = 4;
        selection->dream_smoothing = 2;
        selection->dream_strength = 0.125F;
        selection->dream_feedback = 0.8F;
        selection->dream_zoom = 1.02F;
        selection->dream_rotation = -0.25F;
        selection->dream_octave_scale = 1.6F;
        std::strcpy(selection->dream_model_path, "C:/models/dream.pt");
        std::strcpy(selection->dream_layer, "relu4_2");
    }

    acmxvk::InterfaceClient client;
    acmxvk::InterfaceState state;
    const bool opened = client.open();
    const bool read = opened && client.read(state);
    const bool valid =
        read && state.sequence == 42 &&
        state.selected_shader_name == "selected.frag.spv" &&
        state.multipass.enabled && state.multipass.shader_names.size() == 2 &&
        state.multipass.shader_names[0] == "first.comp.spv" &&
        state.multipass.shader_names[1] == "second.frag.spv" &&
        state.uniform_values.size() == 1 &&
        state.uniform_values[0].name == "slider1" &&
        state.uniform_values[0].value == 0.75F &&
        state.audio_file.request_sequence == 7 &&
        state.audio_file.path == "C:/audio/test.wav" &&
        state.audio_file.pass_through && state.deep_dream.enabled &&
        state.deep_dream.fp16 && state.deep_dream.gpu_filter_first &&
        state.deep_dream.iterations == 3 &&
        state.deep_dream.maximum_dimension == 768 &&
        state.deep_dream.channel == 12 && state.deep_dream.octaves == 2 &&
        state.deep_dream.jitter == 4 && state.deep_dream.smoothing == 2 &&
        state.deep_dream.strength == 0.125F &&
        state.deep_dream.feedback == 0.8F &&
        state.deep_dream.zoom == 1.02F &&
        state.deep_dream.rotation == -0.25F &&
        state.deep_dream.octave_scale == 1.6F &&
        state.deep_dream.model_path == "C:/models/dream.pt" &&
        state.deep_dream.layer == "relu4_2";

    client.close();
    ::UnmapViewOfFile(view);
    ::CloseHandle(mapping);
    ::CloseHandle(mutex);
    if (!valid) {
        std::cerr << "Windows interface-control round trip failed\n";
        return 1;
    }
    std::cout << "ACMXVK Windows interface-control test passed\n";
#endif
    return 0;
}
