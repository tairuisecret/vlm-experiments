export CUDA_VISIBLE_DEVICES=6 MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
M=/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct
swift sft --model $M \
  --dataset data/train.jsonl --val_dataset data/val.jsonl \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 \
  --freeze_vit false --vit_lr 1e-5 \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --torch_dtype bfloat16 --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --max_length 2048 \
  --eval_steps 150 --save_steps 150 --logging_steps 10 --save_total_limit 3 \
  --output_dir /mnt/SSD1/gzh/vlm/out_vit --dataloader_num_workers 2 || exit 1
echo VIT_TRAIN_DONE
LAST=$(ls -dt out_vit/v*/checkpoint-* | head -1)
echo "评测: $LAST"
swift infer --adapters $LAST --val_dataset data/val.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path clean_vit.jsonl
echo VIT_ALL_DONE
