#!/bin/sh

git clone https://github.com/lostjared/libmx2.git
cd libmx2/libmx
mkdir build && cd build
cmake .. -DEXAMPLES=OFF -DOPENGL=ON
make -j$(nproc)
sudo make install
cd ../../../
git clone https://github.com/lostjared/acidcam-gpu.git
cd acidcam-gpu/MXWrite
mkdir build1 && cd build1
cmake .. && make -j$(nrpoc) && sudo make install
cd ../..
cd acidcam-gpu
mkdir build && cd build
cmake .. 
make -j$(nproc) && sudo make install
cd ../../
cd ACMX2
mkdir build && cd build
cmake .. -DAUDIO=ON
make -j$(nproc) && sudo make install
cd ../interface
mkdir build && cd build
qmake6 ..
make -j $(nproc)
cp -rf ../data/ .
cd ../../
echo "completed..."
