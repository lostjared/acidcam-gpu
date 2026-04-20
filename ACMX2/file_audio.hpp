#ifndef __FILE_AUDIO__H_
#define __FILE_AUDIO__H_

#include <string>

/// Open an audio file (WAV, MP3, AAC, etc.) via FFmpeg and decode to float PCM.
/// Returns true on success.
bool file_audio_open(const std::string &filepath);

/// Advance playback by one video frame and update audio globals
/// (gAmplitude, gPeak, gRMS, gSmooth, gLow, gMid, gHigh, gFrequency)
/// plus push samples to the FFT buffer.
/// @param video_fps The video frame rate — determines how many audio samples to consume per call.
void file_audio_process_frame(double video_fps);

/// Returns true if file audio is currently open and has samples remaining.
bool file_audio_is_active();

/// Close the file audio decoder and free resources.
void file_audio_close();

#endif
