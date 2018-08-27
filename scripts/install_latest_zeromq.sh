sudo apt-get update
sudo apt-get install libtool pkg-config build-essential autoconf automake
mkdir zeromq_build
cd zeromq_build
mkdir sodium_build
cd sodium_build
wget "https://download.libsodium.org/libsodium/releases/libsodium-1.0.16.tar.gz"
tar -zxvf "libsodium-1.0.16.tar.gz"
cd libsodium-1.0.16
./configure
make
sudo make install
sudo ldconfig
cd ..
wget "https://github.com/zeromq/libzmq/releases/download/v4.2.3/zeromq-4.2.3.tar.gz"
tar -zxvf "zeromq-4.2.3.tar.gz"
cd zeromq-4.2.3
./configure
make
sudo make install
sudo ldconfig

