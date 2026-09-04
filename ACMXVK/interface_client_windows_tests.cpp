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
        state.audio_file.pass_through;

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
