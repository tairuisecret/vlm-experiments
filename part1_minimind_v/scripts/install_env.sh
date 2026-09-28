set -x
export CONDA_PKGS_DIRS=/data3/gzh/conda_pkgs TMPDIR=/data3/gzh/tmp PIP_NO_CACHE_DIR=1
mkdir -p $CONDA_PKGS_DIRS $TMPDIR
source /home/gzh/anaconda3/etc/profile.d/conda.sh
ENV=/data3/gzh/anaconda3/envs/minimind-v
[ -x $ENV/bin/python ] || conda create -y -p $ENV python=3.10 || exit 1
conda activate $ENV
cd /data3/gzh/project/minimind-v
pip install torch==2.6.0 torchvision==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple || exit 2
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || exit 3
python -c "import torch,transformers;print(\"OK\",torch.__version__,torch.cuda.is_available(),transformers.__version__)"
echo ALL_DONE
