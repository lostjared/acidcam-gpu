#!/bin/bash

pactl load-module module-null-sink sink_name=VirtualAudio sink_properties=device.description="Virtual_Audio"
