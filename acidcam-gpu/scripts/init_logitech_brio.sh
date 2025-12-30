#!/bin/bash



# 1. Disable Dynamic Framerate (prevents FPS drops in low light)
v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0

# 2. Set Auto Exposure to Manual Mode (Value 1 = Manual)
v4l2-ctl -d /dev/video0 -c auto_exposure=1

# 3. Set a fixed Exposure Time 
# (156 is roughly 1/60th of a second, perfect for 60fps)
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=156

# 4. Set Power Line Frequency to 60Hz (prevents light flicker in US/NTSC)
v4l2-ctl -d /dev/video0 -c power_line_frequency=2

# 5. Optional: Boost Gain if the image is too dark after manual exposure
# Higher gain = brighter image but more noise. 0 to 255.
v4l2-ctl -d /dev/video0 -c gain=120

echo "Logitech Brio settings applied. FPS should now be stable."
