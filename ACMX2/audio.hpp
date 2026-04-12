#ifndef __AUDIO__H_
#define __AUDIO__H_

#include <RtAudio.h>
#include <string>

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
void set_record_gain(float gain);
float get_record_gain();

#endif
