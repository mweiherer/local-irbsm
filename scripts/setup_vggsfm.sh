# Small script to download and install VGGSfM.


mkdir extern && cd extern

git clone https://github.com/facebookresearch/vggsfm.git
cd vggsfm

source install.sh
python -m pip install -e .

conda deactivate 
conda activate local-irbsm

cd ../..
