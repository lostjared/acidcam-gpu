#ifndef __FILE_AUDIO__H_
#define __FILE_AUDIO__H_

#include <string>
#include <vector>

namespace acmx2::audio {
    class AudioAnalyzer;
}

/**
 * @file file_audio.hpp
 * @brief File-based audio input for audio-reactive shaders.
 *
 * Provides an alternative audio source that reads from a media file
 * (WAV, MP3, AAC, FLAC, OGG, or video containers with an audio track)
 * instead of a live microphone via RtAudio.  The decoded audio drives
 * the same AudioAnalyzer and 1-D FFT spectrum texture used by live input.
 *
 * Typical usage:
 * @code
 *   if (file_audio_open("music.mp3")) {
 *       // each frame:
 *       file_audio_process_frame(60.0, analyzer);
 *   }
 *   file_audio_close();
 * @endcode
 *
 * Requires FFmpeg libraries (libavformat, libavcodec, libswresample).
 * Compiled only when the CMake option @c AUDIO=ON is set.
 */

/**
 * @brief Open and fully decode an audio file to mono float PCM at 44 100 Hz.
 *
 * Uses FFmpeg to open @p filepath, locate the first audio stream,
 * decode it, and resample all samples into an internal buffer.
 * After this call the FFmpeg decoder contexts are freed; only the
 * sample buffer remains in memory.
 *
 * @param filepath Path to an audio or video file containing an audio track.
 * @return @c true on success, @c false on any decoder or I/O error.
 */
bool file_audio_open(const std::string &filepath);

/**
 * @brief Return the successfully decoded source tracks in playback order.
 *
 * A single audio file returns one path. An M3U source returns each usable
 * playlist entry with relative paths resolved against the playlist directory.
 */
std::vector<std::string> file_audio_source_paths();

/**
 * @brief Enable or disable looping of the currently decoded audio file.
 * @param enabled When true, playback restarts at the first sample at EOF.
 */
void file_audio_set_repeat(bool enabled);

/**
 * @brief Configure real-time playback of the decoded file through an output device.
 *
 * The stream is opened immediately and starts when file_audio_process_frame()
 * first advances the file, keeping audible playback aligned with visual analysis.
 *
 * @param output_device RtAudio device ID, or -1 for the default output.
 * @return @c true when the output stream was configured successfully.
 */
bool file_audio_enable_output(int output_device);

/**
 * @brief Check whether real-time output playback can provide the master clock.
 *
 * This remains true from output configuration until the decoded audio reaches
 * its end. Before the stream starts, its timestamp is zero.
 */
bool file_audio_has_output_clock();

/**
 * @brief Return the current output-device playback timestamp in seconds.
 *
 * @return Playback position measured by the RtAudio callback, or 0 when no
 *         output clock is available.
 */
double file_audio_playback_time();

/**
 * @brief Advance playback by one video frame and update the audio analyzer.
 *
 * Without output playback, advances by exactly one video-frame duration
 * using a fractional sample accumulator. With output playback, analyzes the
 * sample window at the output device's current timestamp.
 *
 * @param video_fps Video frame rate — determines how many audio samples
 *                  are consumed per call.
 * @param analyzer Shared analyzer that also services live audio input.
 */
void file_audio_process_frame(double video_fps, acmx2::audio::AudioAnalyzer &analyzer);

/**
 * @brief Check whether file audio playback is still active.
 * @return @c true if a file is open and unplayed samples remain.
 */
bool file_audio_is_active();

/**
 * @brief Close the file audio decoder and release all resources.
 *
 * Stops file playback, frees any remaining FFmpeg contexts (safe to call
 * even if already closed), and releases the decoded sample buffer.
 */
void file_audio_close();

#endif
