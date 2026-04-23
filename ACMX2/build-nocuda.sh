#!/bin/sh

mkdir -p build-nocuda && cd build-nocuda
cmake .. -DWITH_CUDA=OFF -DAUDIO=ON -DMIDI=ON
make -j$(nproc)
sudo make install
cd ..

