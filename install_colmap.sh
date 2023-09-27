# https_proxy=http://127.0.0.1:7890 git clone https://github.com/colmap/colmap.git
# cd colmap
# git checkout dev
# mkdir build
# cd build
# cmake .. -GNinja
# ninja
# sudo ninja install

sudo apt-get install libatlas-base-dev libsuitesparse-dev
https_proxy=http://127.0.0.1:7890 git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j
sudo make install