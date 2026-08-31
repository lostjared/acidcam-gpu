#include "snapshot_writer.hpp"

#include <mxvk/mxvk_png.hpp>

#ifdef ACMXVK_WITH_WEBP
#include <webp/encode.h>
#endif
#ifdef ACMXVK_WITH_TIFF
#include <tiffio.h>
#endif

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace acmxvk {
    SnapshotWriter::~SnapshotWriter() {
        stop();
    }

    bool SnapshotWriter::start() {
        std::lock_guard<std::mutex> lock(mutex);
        if (worker.joinable()) {
            return true;
        }
        stopping = false;
        try {
            worker = std::thread(&SnapshotWriter::workerLoop, this);
        } catch (const std::exception &error) {
            std::cerr << "acmxvk: could not start snapshot worker: "
                      << error.what() << '\n';
            return false;
        }
        return true;
    }

    void SnapshotWriter::stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!worker.joinable()) {
                return;
            }
            stopping = true;
        }
        condition.notify_one();
        worker.join();
    }

    bool SnapshotWriter::queueFull() {
        std::lock_guard<std::mutex> lock(mutex);
        return jobs_in_flight >= QUEUE_CAPACITY;
    }

    void SnapshotWriter::enqueue(SnapshotJob job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push_back(std::move(job));
            ++jobs_in_flight;
        }
        condition.notify_one();
    }

    void SnapshotWriter::savePng(const fs::path &path, std::uint8_t *rgba,
                                 int width, int height) {
        if (!mxvk::SavePNG_RGBA(path.string().c_str(), rgba, width, height)) {
            throw std::runtime_error("unable to write PNG frame: " +
                                     path.string());
        }
    }

    void SnapshotWriter::saveRaw(const fs::path &path,
                                 const std::vector<std::uint8_t> &rgba,
                                 std::uint32_t width, std::uint32_t height) {
        if (width == 0U || height == 0U) {
            throw std::runtime_error(
                "invalid image dimensions for raw RGBA snapshot: " +
                path.string());
        }

        const std::uint64_t byte_count =
            static_cast<std::uint64_t>(width) *
            static_cast<std::uint64_t>(height) * 4U;
        if (byte_count > rgba.size() ||
            byte_count > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error(
                "invalid pixel buffer for raw RGBA snapshot: " + path.string());
        }

        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("unable to open raw RGBA snapshot: " +
                                     path.string());
        }
        output.write(reinterpret_cast<const char *>(rgba.data()),
                     static_cast<std::streamsize>(byte_count));
        if (!output) {
            throw std::runtime_error("unable to write raw RGBA snapshot: " +
                                     path.string());
        }
    }

#ifdef ACMXVK_WITH_WEBP
    void SnapshotWriter::saveWebP(const fs::path &path,
                                  const std::uint8_t *rgba, int width,
                                  int height) {
        if (rgba == nullptr || width <= 0 || height <= 0 ||
            width > std::numeric_limits<int>::max() / 4) {
            throw std::runtime_error(
                "invalid image dimensions for WebP snapshot: " + path.string());
        }

        std::uint8_t *encoded_pixels = nullptr;
        const std::size_t encoded_size = WebPEncodeLosslessRGBA(
            rgba, width, height, width * 4, &encoded_pixels);
        const std::unique_ptr<std::uint8_t, decltype(&WebPFree)> encoded_data(
            encoded_pixels, &WebPFree);
        if (encoded_size == 0 || encoded_data == nullptr) {
            throw std::runtime_error("unable to encode WebP snapshot: " +
                                     path.string());
        }

        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("unable to open WebP snapshot: " +
                                     path.string());
        }
        output.write(reinterpret_cast<const char *>(encoded_data.get()),
                     static_cast<std::streamsize>(encoded_size));
        if (!output) {
            throw std::runtime_error("unable to write WebP snapshot: " +
                                     path.string());
        }
    }
#endif

#ifdef ACMXVK_WITH_TIFF
    void SnapshotWriter::saveTiff(const fs::path &path,
                                  const std::uint8_t *rgba, int width,
                                  int height) {
        if (rgba == nullptr || width <= 0 || height <= 0 ||
            width > std::numeric_limits<int>::max() / 4) {
            throw std::runtime_error(
                "invalid image dimensions for TIFF snapshot: " + path.string());
        }

        const std::unique_ptr<TIFF, decltype(&TIFFClose)> output(
            TIFFOpen(path.string().c_str(), "w"), &TIFFClose);
        if (output == nullptr) {
            throw std::runtime_error("unable to open TIFF snapshot: " +
                                     path.string());
        }

        const std::uint16_t extra_sample = EXTRASAMPLE_UNASSALPHA;
        const bool configured =
            TIFFSetField(output.get(), TIFFTAG_IMAGEWIDTH,
                         static_cast<std::uint32_t>(width)) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_IMAGELENGTH,
                         static_cast<std::uint32_t>(height)) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_SAMPLESPERPIXEL, 4) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_BITSPERSAMPLE, 8) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_ORIENTATION,
                         ORIENTATION_TOPLEFT) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_PLANARCONFIG,
                         PLANARCONFIG_CONTIG) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB) !=
                0 &&
            TIFFSetField(output.get(), TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT) !=
                0 &&
            TIFFSetField(output.get(), TIFFTAG_COMPRESSION, COMPRESSION_LZW) !=
                0 &&
            TIFFSetField(output.get(), TIFFTAG_ROWSPERSTRIP,
                         TIFFDefaultStripSize(output.get(), 0)) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_EXTRASAMPLES, 1,
                         &extra_sample) != 0 &&
            TIFFSetField(output.get(), TIFFTAG_IMAGEDESCRIPTION,
                         "ACMXVK processed snapshot: 8-bit RGBA TIFF") != 0;
        if (!configured) {
            throw std::runtime_error("unable to configure TIFF snapshot: " +
                                     path.string());
        }

        const std::size_t row_bytes = static_cast<std::size_t>(width) * 4U;
        for (int row = 0; row < height; ++row) {
            auto *row_pixels = const_cast<std::uint8_t *>(
                rgba + static_cast<std::size_t>(row) * row_bytes);
            if (TIFFWriteScanline(output.get(), row_pixels,
                                  static_cast<std::uint32_t>(row), 0) < 0) {
                throw std::runtime_error("unable to write TIFF snapshot: " +
                                         path.string());
            }
        }
    }
#endif

    std::string_view SnapshotWriter::formatName(SnapshotFormat format) noexcept {
        switch (format) {
        case SnapshotFormat::WebP:
            return "WebP";
        case SnapshotFormat::Tiff:
            return "TIFF";
        case SnapshotFormat::Raw:
            return "raw RGBA";
        case SnapshotFormat::Png:
            return "PNG";
        }
        return "snapshot";
    }

    void SnapshotWriter::workerLoop() noexcept {
        while (true) {
            SnapshotJob job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock,
                               [&] { return stopping || !jobs.empty(); });
                if (stopping && jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }

            try {
                if (job.format == SnapshotFormat::Raw) {
                    saveRaw(job.path, job.rgba, job.width, job.height);
                } else if (job.format == SnapshotFormat::Tiff) {
#ifdef ACMXVK_WITH_TIFF
                    saveTiff(job.path, job.rgba.data(),
                             static_cast<int>(job.width),
                             static_cast<int>(job.height));
#else
                    throw std::runtime_error(
                        "TIFF snapshot support is not compiled in");
#endif
                } else if (job.format == SnapshotFormat::WebP) {
#ifdef ACMXVK_WITH_WEBP
                    saveWebP(job.path, job.rgba.data(),
                             static_cast<int>(job.width),
                             static_cast<int>(job.height));
#else
                    throw std::runtime_error(
                        "WebP snapshot support is not compiled in");
#endif
                } else {
                    savePng(job.path, job.rgba.data(),
                            static_cast<int>(job.width),
                            static_cast<int>(job.height));
                }
                std::ostringstream message;
                message << "acmxvk: took " << formatName(job.format)
                        << " snapshot: " << job.path.string() << '\n';
                std::cout << message.str();
            } catch (const std::exception &error) {
                std::ostringstream message;
                message << "acmxvk: snapshot failed: " << error.what() << '\n';
                std::cerr << message.str();
            } catch (...) {
                std::cerr
                    << "acmxvk: snapshot failed with an unknown error\n";
            }

            std::lock_guard<std::mutex> lock(mutex);
            if (jobs_in_flight > 0) {
                --jobs_in_flight;
            }
        }
    }
} // namespace acmxvk
