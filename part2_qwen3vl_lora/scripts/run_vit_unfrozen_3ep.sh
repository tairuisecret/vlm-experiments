export CUDA_VISIBLE_DEVICES=0 MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
M=/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct
swift sft --model $M \
  --dataset data/train.jsonl --val_dataset data/val.jsonl \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 \
  --freeze_vit false --vit_lr 1e-5 \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --torch_dtype bfloat16 --num_train_epochs 3 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --max_length 2048 \
  --eval_steps 150 --save_steps 625 --logging_steps 10 --save_total_limit 4 \
  --output_dir /mnt/SSD1/gzh/vlm/out_vit3ep --dataloader_num_workers 2 || exit 1
echo VIT3EP_TRAIN_DONE
# 按 epoch 边界逐个评测，用下游指标选最优（而非只看 loss）
R=$(ls -d out_vit3ep/v*/ | head -1)
for CK in 625 1250 1875; do
  D=${R}checkpoint-$CK
  [ -d "$D" ] || continue
  echo "=== 评测 epoch $((CK/625)) : $D ==="
  swift infer --adapters $D --val_dataset data/val.jsonl --infer_backend pt \
    --max_new_tokens 256 --temperature 0 --max_batch_size 16 \
    --result_path clean_vit3ep_ep$((CK/625)).jsonl || echo "eval failed for $CK"
done
echo VIT3EP_ALL_DONE
