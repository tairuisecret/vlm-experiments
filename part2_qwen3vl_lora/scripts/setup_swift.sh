set -x
export CONDA_PKGS_DIRS=/data3/gzh/conda_pkgs TMPDIR=/mnt/SSD1/gzh/tmp PIP_NO_CACHE_DIR=1
source /home/gzh/anaconda3/etc/profile.d/conda.sh
ENV=/mnt/SSD1/gzh/envs/swift
[ -x $ENV/bin/python ] || conda create -y -p $ENV python=3.10 || exit 1
conda activate $ENV
pip install torch==2.6.0 torchvision==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple || exit 2
pip install "ms-swift[llm]" qwen-vl-utils decord -i https://pypi.tuna.tsinghua.edu.cn/simple || exit 3
python -c "import swift,torch;print(\"OK\",swift.__version__,torch.__version__,torch.cuda.is_available())"
echo SETUP_DONE
