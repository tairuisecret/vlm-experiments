export CUDA_VISIBLE_DEVICES=1,6 NPROC_PER_NODE=2 MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
M=/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct
for R in 16 8 32; do
  echo "===== TRAIN rank=$R ====="
  swift sft --model $M --dataset data/train.jsonl --val_dataset data/val.jsonl \
    --tuner_type lora --lora_rank $R --lora_alpha $((R*2)) --freeze_vit true \
    --torch_dtype bfloat16 --num_train_epochs 1 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --warmup_ratio 0.05 --max_length 2048 \
    --eval_steps 100 --save_steps 100 --logging_steps 10 --save_total_limit 10 \
    --output_dir /mnt/SSD1/gzh/vlm/out_r$R --dataloader_num_workers 4 || exit 1
done
echo ALL_TRAIN_DONE
