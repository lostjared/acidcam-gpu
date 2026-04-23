#!/bin/sh

mkdir -p build && cd build
cmake .. -DWITH_CUDA=ON -DAUDIO=ON -DMIDI=ON
make -j$(nproc)
sudo make install
cd ..


