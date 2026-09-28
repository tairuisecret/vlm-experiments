export CUDA_VISIBLE_DEVICES=6,2 NPROC_PER_NODE=2 MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
M=/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct
swift sft --model $M --dataset data/train.jsonl --val_dataset data/val.jsonl \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --freeze_vit true \
  --torch_dtype bfloat16 --num_train_epochs 3 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --max_length 2048 \
  --eval_steps 150 --save_steps 150 --logging_steps 10 --save_total_limit 15 \
  --output_dir /mnt/SSD1/gzh/vlm/out_r16_3ep --dataloader_num_workers 4 || exit 1
echo TRAIN_3EP_DONE
# 评测：最优 + 最终 checkpoint
BEST=$(python pick_ckpt.py 16_3ep | head -1); LAST=$(python pick_ckpt.py 16_3ep | tail -1)
export CUDA_VISIBLE_DEVICES=6
echo "best=$BEST last=$LAST"
swift infer --adapters $BEST --val_dataset data/val.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path clean_r16_3ep_best.jsonl
[ "$BEST" != "$LAST" ] && swift infer --adapters $LAST --val_dataset data/val.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path clean_r16_3ep_last.jsonl
echo EVAL_3EP_DONE
