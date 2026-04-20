#ifndef __FILE_AUDIO__H_
#define __FILE_AUDIO__H_

#include <string>

/**
 * @file file_audio.hpp
 * @brief File-based audio input for audio-reactive shaders.
 *
 * Provides an alternative audio source that reads from a media file
 * (WAV, MP3, AAC, FLAC, OGG, or video containers with an audio track)
 * instead of a live microphone via RtAudio.  The decoded audio drives
 * the same global reactivity variables (gAmplitude, gPeak, gRMS, etc.)
 * and 1-D FFT spectrum texture used by GLSL shaders.
 *
 * Typical usage:
 * @code
 *   if (file_audio_open("music.mp3")) {
 *       // each frame:
 *       file_audio_process_frame(60.0);
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
 * @brief Advance playback by one video frame and update audio globals.
 *
 * Consumes @c 44100/video_fps samples from the internal buffer and
 * computes amplitude, peak, RMS, smoothed amplitude, 3-band energy
 * (low / mid / high), and dominant frequency.  The sample window is
 * also pushed to the FFT ring buffer via push_audio_buffer().
 *
 * @param video_fps Video frame rate — determines how many audio samples
 *                  are consumed per call.
 */
void file_audio_process_frame(double video_fps);

/**
 * @brief Check whether file audio playback is still active.
 * @return @c true if a file is open and unplayed samples remain.
 */
bool file_audio_is_active();

/**
 * @brief Close the file audio decoder and release all resources.
 *
 * Frees any remaining FFmpeg contexts (safe to call even if already
 * closed) and releases the decoded sample buffer.
 */
void file_audio_close();

#endif
