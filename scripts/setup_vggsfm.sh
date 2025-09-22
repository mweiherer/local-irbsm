#!/bin/bash -l

# Small script to download and install VGGSfM.


cd extern
git clone https://github.com/facebookresearch/vggsfm.git
cd vggsfm

source install.sh
python -m pip install -e .

cd ../..