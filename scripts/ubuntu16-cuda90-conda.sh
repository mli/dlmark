#
# script to install cuda 9.0 and conda on ubuntu 16

set -e

sudo apt-get update 
sudo apt-get install -y build-essential libgfortran3

https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo sh cuda_9.0.176_384.81_linux-run --silent --driver --toolkit
rm cuda_9.0.176_384.81_linux-run

echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64" >>~/.bashrc
sudo nvidia-smi -pm 1

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh

echo "export PATH=\"/home/ubuntu/miniconda3/bin:\$PATH\"" >>~/.bashrc

