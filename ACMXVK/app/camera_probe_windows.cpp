#include "camera_probe.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dshow.h>
#include <dvdmedia.h>
#include <olectl.h>
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#ifndef MEDIASUBTYPE_I420
static const GUID MEDIASUBTYPE_I420 = {
    0x30323449, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
#endif

namespace acmxvk {
    namespace {
        template <typename T>
        struct ComReleaser {
            void operator()(T *value) const {
                if (value != nullptr) {
                    value->Release();
                }
            }
        };

        template <typename T>
        using ComPtr = std::unique_ptr<T, ComReleaser<T>>;

        class ComScope {
          public:
            ComScope()
                : result(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)),
                  should_uninitialize(SUCCEEDED(result)) {
            }

            ~ComScope() {
                if (should_uninitialize) {
                    CoUninitialize();
                }
            }

            [[nodiscard]] bool ready() const {
                return SUCCEEDED(result) || result == RPC_E_CHANGED_MODE;
            }

          private:
            HRESULT result;
            bool should_uninitialize;
        };

        struct CameraDevice {
            int index = -1;
            std::string name;
            ComPtr<IMoniker> moniker;
        };

        [[nodiscard]] std::string wide_to_utf8(const wchar_t *value) {
            if (value == nullptr || *value == L'\0') {
                return {};
            }
            const int source_length = static_cast<int>(wcslen(value));
            const int length = WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS, value, source_length, nullptr,
                0, nullptr, nullptr);
            if (length <= 0) {
                return {};
            }
            std::string result(static_cast<std::size_t>(length), '\0');
            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value,
                                    source_length, result.data(), length,
                                    nullptr, nullptr) != length) {
                return {};
            }
            return result;
        }

        [[nodiscard]] std::string camera_name(IPropertyBag *property_bag) {
            if (property_bag == nullptr) {
                return {};
            }
            for (const wchar_t *property : {L"FriendlyName", L"Description"}) {
                VARIANT value;
                VariantInit(&value);
                const HRESULT result =
                    property_bag->Read(property, &value, nullptr);
                std::string name;
                if (SUCCEEDED(result) && value.vt == VT_BSTR) {
                    name = wide_to_utf8(value.bstrVal);
                }
                VariantClear(&value);
                if (!name.empty()) {
                    return name;
                }
            }
            return {};
        }

        [[nodiscard]] std::vector<CameraDevice> enumerate_devices() {
            ICreateDevEnum *device_enumerator_raw = nullptr;
            if (FAILED(CoCreateInstance(
                    CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                    IID_ICreateDevEnum,
                    reinterpret_cast<void **>(&device_enumerator_raw)))) {
                return {};
            }
            ComPtr<ICreateDevEnum> device_enumerator(device_enumerator_raw);

            IEnumMoniker *enumerator_raw = nullptr;
            if (device_enumerator->CreateClassEnumerator(
                    CLSID_VideoInputDeviceCategory, &enumerator_raw, 0) !=
                    S_OK ||
                enumerator_raw == nullptr) {
                return {};
            }
            ComPtr<IEnumMoniker> enumerator(enumerator_raw);

            std::vector<CameraDevice> devices;
            IMoniker *moniker_raw = nullptr;
            ULONG fetched = 0;
            int index = 0;
            while (enumerator->Next(1, &moniker_raw, &fetched) == S_OK) {
                ComPtr<IMoniker> moniker(moniker_raw);
                moniker_raw = nullptr;
                std::string name;
                IPropertyBag *property_bag_raw = nullptr;
                if (SUCCEEDED(moniker->BindToStorage(
                        nullptr, nullptr, IID_IPropertyBag,
                        reinterpret_cast<void **>(&property_bag_raw)))) {
                    ComPtr<IPropertyBag> property_bag(property_bag_raw);
                    name = camera_name(property_bag.get());
                }
                if (name.empty()) {
                    name = "Camera " + std::to_string(index);
                }
                devices.push_back({index, std::move(name), std::move(moniker)});
                ++index;
            }
            return devices;
        }

        void free_media_type(AM_MEDIA_TYPE *media_type) {
            if (media_type == nullptr) {
                return;
            }
            if (media_type->pbFormat != nullptr) {
                CoTaskMemFree(media_type->pbFormat);
            }
            if (media_type->pUnk != nullptr) {
                media_type->pUnk->Release();
            }
            CoTaskMemFree(media_type);
        }

        [[nodiscard]] ComPtr<IAMStreamConfig>
        stream_config(IMoniker *moniker) {
            IBaseFilter *filter_raw = nullptr;
            if (moniker == nullptr ||
                FAILED(moniker->BindToObject(
                    nullptr, nullptr, IID_IBaseFilter,
                    reinterpret_cast<void **>(&filter_raw)))) {
                return {};
            }
            ComPtr<IBaseFilter> filter(filter_raw);
            IEnumPins *pins_raw = nullptr;
            if (FAILED(filter->EnumPins(&pins_raw)) || pins_raw == nullptr) {
                return {};
            }
            ComPtr<IEnumPins> pins(pins_raw);
            IPin *pin_raw = nullptr;
            ULONG fetched = 0;
            while (pins->Next(1, &pin_raw, &fetched) == S_OK) {
                ComPtr<IPin> pin(pin_raw);
                pin_raw = nullptr;
                PIN_DIRECTION direction = PINDIR_INPUT;
                if (FAILED(pin->QueryDirection(&direction)) ||
                    direction != PINDIR_OUTPUT) {
                    continue;
                }
                IAMStreamConfig *config_raw = nullptr;
                if (SUCCEEDED(pin->QueryInterface(
                        IID_IAMStreamConfig,
                        reinterpret_cast<void **>(&config_raw))) &&
                    config_raw != nullptr) {
                    return ComPtr<IAMStreamConfig>(config_raw);
                }
            }
            return {};
        }

        [[nodiscard]] std::string subtype_name(const GUID &subtype) {
            if (subtype == MEDIASUBTYPE_YUY2)
                return "YUY2";
            if (subtype == MEDIASUBTYPE_UYVY)
                return "UYVY";
            if (subtype == MEDIASUBTYPE_YV12)
                return "YV12";
            if (subtype == MEDIASUBTYPE_NV12)
                return "NV12";
            if (subtype == MEDIASUBTYPE_I420)
                return "I420";
            if (subtype == MEDIASUBTYPE_MJPG)
                return "MJPG";
            if (subtype == MEDIASUBTYPE_RGB24)
                return "RGB24";
            if (subtype == MEDIASUBTYPE_RGB32)
                return "RGB32";
            return "OTHER";
        }

        [[nodiscard]] bool video_format(const AM_MEDIA_TYPE *media_type,
                                        int &width, int &height,
                                        double &frame_rate) {
            if (media_type == nullptr || media_type->pbFormat == nullptr) {
                return false;
            }
            const BITMAPINFOHEADER *bitmap = nullptr;
            REFERENCE_TIME interval = 0;
            if (media_type->formattype == FORMAT_VideoInfo &&
                media_type->cbFormat >= sizeof(VIDEOINFOHEADER)) {
                const auto *info = reinterpret_cast<const VIDEOINFOHEADER *>(
                    media_type->pbFormat);
                bitmap = &info->bmiHeader;
                interval = info->AvgTimePerFrame;
            } else if (media_type->formattype == FORMAT_VideoInfo2 &&
                       media_type->cbFormat >= sizeof(VIDEOINFOHEADER2)) {
                const auto *info = reinterpret_cast<const VIDEOINFOHEADER2 *>(
                    media_type->pbFormat);
                bitmap = &info->bmiHeader;
                interval = info->AvgTimePerFrame;
            }
            if (bitmap == nullptr || bitmap->biWidth <= 0 ||
                bitmap->biHeight == 0) {
                return false;
            }
            width = bitmap->biWidth;
            height = std::abs(bitmap->biHeight);
            frame_rate = interval > 0
                             ? 10000000.0 / static_cast<double>(interval)
                             : 0.0;
            return true;
        }

        void append_frame_rate(std::vector<double> &rates, double value) {
            if (!std::isfinite(value) || value <= 0.0) {
                return;
            }
            if (std::none_of(rates.begin(), rates.end(), [value](double rate) {
                    return std::abs(rate - value) < 0.05;
                })) {
                rates.push_back(value);
            }
        }
    } // namespace

    [[nodiscard]] bool listCameraDevices(std::ostream &output,
                                         std::ostream &error) {
        ComScope com_scope;
        if (!com_scope.ready()) {
            error << "acmxvk: unable to initialize COM for camera probing\n";
            return false;
        }
        const std::vector<CameraDevice> devices = enumerate_devices();
        for (const CameraDevice &device : devices) {
            output << device.index << '\t' << device.name << '\n';
        }
        if (devices.empty()) {
            error << "acmxvk: no DirectShow capture devices found\n";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool probeCameraDevice(int device_index,
                                         std::ostream &output,
                                         std::ostream &error) {
        ComScope com_scope;
        if (!com_scope.ready()) {
            error << "acmxvk: unable to initialize COM for camera probing\n";
            return false;
        }
        std::vector<CameraDevice> devices = enumerate_devices();
        const auto device = std::find_if(
            devices.begin(), devices.end(), [device_index](const CameraDevice &entry) {
                return entry.index == device_index;
            });
        if (device == devices.end()) {
            error << "acmxvk: DirectShow camera " << device_index
                  << " was not found\n";
            return false;
        }
        const ComPtr<IAMStreamConfig> config =
            stream_config(device->moniker.get());
        if (!config) {
            error << "acmxvk: camera " << device_index
                  << " exposes no DirectShow capture configuration\n";
            return false;
        }

        int capability_count = 0;
        int capability_size = 0;
        if (FAILED(config->GetNumberOfCapabilities(&capability_count,
                                                   &capability_size)) ||
            capability_count <= 0 || capability_size <= 0) {
            error << "acmxvk: camera " << device_index
                  << " exposes no DirectShow capture formats\n";
            return false;
        }

        using ResolutionMap =
            std::map<std::pair<int, int>, std::vector<double>>;
        std::map<std::string, ResolutionMap> formats;
        std::vector<BYTE> capability_buffer(
            static_cast<std::size_t>(capability_size));
        for (int index = 0; index < capability_count; ++index) {
            AM_MEDIA_TYPE *media_type = nullptr;
            if (FAILED(config->GetStreamCaps(index, &media_type,
                                             capability_buffer.data())) ||
                media_type == nullptr) {
                continue;
            }
            const std::unique_ptr<AM_MEDIA_TYPE, decltype(&free_media_type)>
                media_type_guard(media_type, &free_media_type);
            int width = 0;
            int height = 0;
            double frame_rate = 0.0;
            if (!video_format(media_type, width, height, frame_rate)) {
                continue;
            }
            std::vector<double> &rates =
                formats[subtype_name(media_type->subtype)][{width, height}];
            append_frame_rate(rates, frame_rate);
            if (capability_size >=
                static_cast<int>(sizeof(VIDEO_STREAM_CONFIG_CAPS))) {
                const auto *caps =
                    reinterpret_cast<const VIDEO_STREAM_CONFIG_CAPS *>(
                        capability_buffer.data());
                if (caps->MinFrameInterval > 0) {
                    append_frame_rate(
                        rates, 10000000.0 /
                                   static_cast<double>(caps->MinFrameInterval));
                }
                if (caps->MaxFrameInterval > 0) {
                    append_frame_rate(
                        rates, 10000000.0 /
                                   static_cast<double>(caps->MaxFrameInterval));
                }
            }
        }

        output << "Device " << device_index << ": DirectShow\n"
               << "  Driver : DirectShow\n"
               << "  Card   : " << device->name << '\n'
               << "  Bus    : Windows\n";
        for (auto &[format, resolutions] : formats) {
            output << "\n  Format: " << format << '\n';
            for (auto &[resolution, rates] : resolutions) {
                std::sort(rates.begin(), rates.end());
                output << "    " << resolution.first << 'x'
                       << resolution.second;
                bool first = true;
                for (const double rate : rates) {
                    output << (first ? " @ " : ", ") << std::fixed
                           << std::setprecision(1) << rate << " fps";
                    first = false;
                }
                output << '\n';
            }
        }
        if (formats.empty()) {
            error << "acmxvk: camera " << device_index
                  << " exposes no DirectShow capture formats\n";
            return false;
        }
        return true;
    }
} // namespace acmxvk
