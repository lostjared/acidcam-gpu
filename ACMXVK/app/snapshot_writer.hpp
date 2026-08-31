#ifndef ACMXVK_APP_SNAPSHOT_WRITER_HPP
#define ACMXVK_APP_SNAPSHOT_WRITER_HPP

#include "options.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace acmxvk {
    enum class SnapshotFormat { Png,
                                WebP,
                                Tiff,
                                Raw };

    struct SnapshotJob {
        fs::path path;
        std::vector<std::uint8_t> rgba;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        SnapshotFormat format = SnapshotFormat::Png;
    };

    class SnapshotWriter {
      public:
        SnapshotWriter() = default;
        ~SnapshotWriter();
        SnapshotWriter(const SnapshotWriter &) = delete;
        SnapshotWriter &operator=(const SnapshotWriter &) = delete;
        SnapshotWriter(SnapshotWriter &&) = delete;
        SnapshotWriter &operator=(SnapshotWriter &&) = delete;

        [[nodiscard]] bool start();
        void stop() noexcept;
        [[nodiscard]] bool queueFull();
        void enqueue(SnapshotJob job);

        static void savePng(const fs::path &path, std::uint8_t *rgba,
                            int width, int height);
        [[nodiscard]] static std::string_view
        formatName(SnapshotFormat format) noexcept;
        [[nodiscard]] static std::string_view
        extension(SnapshotFormat format) noexcept;

      private:
        static constexpr std::size_t QUEUE_CAPACITY = 4;

        static void saveRaw(const fs::path &path,
                            const std::vector<std::uint8_t> &rgba,
                            std::uint32_t width, std::uint32_t height);
#ifdef ACMXVK_WITH_WEBP
        static void saveWebP(const fs::path &path, const std::uint8_t *rgba,
                             int width, int height);
#endif
#ifdef ACMXVK_WITH_TIFF
        static void saveTiff(const fs::path &path, const std::uint8_t *rgba,
                             int width, int height);
#endif
        void workerLoop() noexcept;

        std::deque<SnapshotJob> jobs;
        std::mutex mutex;
        std::condition_variable condition;
        std::thread worker;
        std::size_t jobs_in_flight = 0;
        bool stopping = false;
    };
} // namespace acmxvk

#endif
