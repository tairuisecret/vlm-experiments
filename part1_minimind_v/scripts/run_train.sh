#!/bin/bash
# Pretrain -> SFT，双卡 DDP（GPU 4,5），README 默认超参
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5} TMPDIR=/data3/gzh/tmp OMP_NUM_THREADS=8
PY=/data3/gzh/anaconda3/envs/minimind-v/bin
NP=$(echo $CUDA_VISIBLE_DEVICES | tr , "\n" | wc -l)
cd /data3/gzh/project/minimind-v/trainer
echo "[$(date)] pretrain start"
$PY/torchrun --nproc_per_node $NP --master_port 29517 train_pretrain_vlm.py --epochs 2 --from_weight llm \
  > ../logs/pretrain.log 2>&1 || { echo "[$(date)] pretrain FAILED"; exit 1; }
echo "[$(date)] pretrain done"
$PY/torchrun --nproc_per_node $NP --master_port 29517 train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm \
  > ../logs/sft.log 2>&1 || { echo "[$(date)] sft FAILED"; exit 2; }
echo "[$(date)] sft done, ALL_DONE"
