#ifndef __AUDIO__H_
#define __AUDIO__H_

#include <RtAudio.h>
#include <string>
#include <vector>

int init_audio(unsigned int channels, float sense, int in_device, int out_device);
void list_audio_devices();
void close_audio();
float get_amp();
float get_sense();
void set_sense(float s);
float get_freq();
float get_amp_peak();
float get_amp_rms();
float get_amp_smooth();
float get_amp_low();
float get_amp_mid();
float get_amp_high();
void set_output(bool o);

bool start_audio_recording(const std::string &filepath);
void stop_audio_recording();
bool is_audio_recording();
/**
 * @brief Duration in seconds of the most recent (or in-progress) WAV recording.
 *
 * Computed from the bytes already written to the recording file divided by
 * `sample_rate * channels * sizeof(int16_t)`.  Returns 0 if no recording
 * has been started.  Used by the muxer to correct A/V drift when the
 * webcam capture rate is lower than the configured encoder FPS.
 */
double get_audio_recorded_duration_seconds();
void set_record_gain(float gain);
float get_record_gain();

/**
 * @defgroup audio_fft Audio FFT Spectrum
 * @brief Real-time FFT spectrum analysis of the live audio input.
 *
 * These functions expose a frequency-domain representation of the most
 * recent audio buffer captured by the RtAudio callback.  The pipeline is:
 *
 * 1. **Capture** — The audio callback copies its latest input samples into
 *    a double-buffered ring (front/back swap) so the render thread never
 *    blocks the audio thread.
 * 2. **FFT** — `compute_audio_fft()` runs a radix-2 Cooley–Tukey FFT on
 *    the captured samples and stores the resulting magnitude spectrum.
 * 3. **Upload** — The caller (typically `SpectrumTexture::update()`) reads
 *    the magnitude vector via `get_fft_magnitudes()` and uploads it to a
 *    GL_TEXTURE_1D for shaders to sample.
 *
 * ### Why a 1D texture?
 * A 1D texture is the natural fit for a frequency spectrum: each texel
 * index maps to one FFT bin (frequency), and the texel value stores the
 * magnitude (energy) at that frequency.  In GLSL the shader declares
 * `uniform sampler1D spectrum;` and looks up a frequency band with
 * `texture(spectrum, normalised_frequency)`.
 *
 * ### Thread safety
 * - `push_audio_buffer()` is called **only** from the audio callback
 *   thread.  It writes to the *back* buffer and then atomically swaps
 *   a flag so the front buffer is always a complete, consistent snapshot.
 * - `compute_audio_fft()` and `get_fft_magnitudes()` are called **only**
 *   from the render (main) thread, so no lock is needed on the output
 *   magnitude vector.
 * @{
 */

/// Number of complex samples fed into the FFT.  Must be a power of two.
/// 512 samples at 44 100 Hz ≈ 11.6 ms window → 256 unique frequency bins
/// with ~86 Hz resolution per bin.
constexpr int FFT_SIZE = 512;

/**
 * @brief Store the latest PCM samples from the audio callback for FFT processing.
 *
 * Called inside `audioCallback()` on the audio thread.  The samples are
 * written into a *back* buffer; once the copy finishes, an atomic flag is
 * flipped so the render thread sees the new data on its next call to
 * `compute_audio_fft()`.
 *
 * Only channel 0 (mono down-mix) is captured — stereo input is reduced
 * to a single channel by picking `in[i * input_channels]`.
 *
 * @param samples  Pointer to interleaved float PCM input from RtAudio.
 * @param count    Number of *frames* (not individual samples) in the buffer.
 * @param channels Number of interleaved channels per frame.
 */
void push_audio_buffer(const float *samples, unsigned int count, unsigned int channels);

/**
 * @brief Run a radix-2 Cooley–Tukey FFT on the most recent audio snapshot.
 *
 * Reads the front buffer populated by `push_audio_buffer()`, applies a
 * Hann window to reduce spectral leakage, computes an in-place FFT, and
 * writes the magnitude of each positive-frequency bin into an internal
 * `std::vector<float>` of length `FFT_SIZE / 2`.
 *
 * The magnitude is computed as:
 * \f[
 *   M_k = \frac{2}{N}\,\sqrt{\operatorname{Re}(X_k)^2 + \operatorname{Im}(X_k)^2}
 * \f]
 * where \f$N\f$ = `FFT_SIZE` and \f$k \in [0, N/2)\f$.
 *
 * Call this once per frame on the render thread, *before* uploading to
 * the spectrum texture.
 */
void compute_audio_fft();

/**
 * @brief Return the most recently computed FFT magnitude spectrum.
 *
 * The returned vector has `FFT_SIZE / 2` elements (256 bins by default).
 * Index 0 is the DC component and index 255 corresponds to the Nyquist
 * frequency (≈ 22 050 Hz at 44 100 Hz sample rate).
 *
 * @return Const reference to the internal magnitude vector.  The data
 *         remains valid until the next `compute_audio_fft()` call.
 */
const std::vector<float> &get_fft_magnitudes();

/**
 * @brief Return the FFT bin count (number of usable frequency bins).
 *
 * This equals `FFT_SIZE / 2` because the FFT of a real signal is
 * symmetric — only the first half carries unique information.
 *
 * @return Number of magnitude values returned by `get_fft_magnitudes()`.
 */
int get_fft_bin_count();

/** @} */

#endif
