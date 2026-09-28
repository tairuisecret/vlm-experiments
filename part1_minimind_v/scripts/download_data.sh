set -x
cd /data3/gzh/project/minimind-v
export TMPDIR=/data3/gzh/tmp PIP_NO_CACHE_DIR=1 MODELSCOPE_CACHE=/data3/gzh/modelscope_cache
PY=/data3/gzh/anaconda3/envs/minimind-v/bin
$PY/pip install -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
$PY/modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
$PY/modelscope download --model gongjy/minimind-3v-pytorch llm_768.pth --local_dir ./out
for f in sft_i2t.parquet pretrain_i2t.parquet; do
  for t in 1 2 3 4 5; do
    wget -c -q --show-progress --progress=dot:giga https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/$f -P ./dataset && break
    sleep 10
  done
done
ls -la dataset out model/siglip2-base-p32-256-ve
sha256sum dataset/*.parquet
echo DL_DONE
