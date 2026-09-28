export CUDA_VISIBLE_DEVICES=6 MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
CKPT=$(ls -d out_r16_3ep/v*/checkpoint-1875 | head -1)
swift rlhf --rlhf_type grpo \
  --model /mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct \
  --adapters $CKPT \
  --dataset data_grpo/train.jsonl \
  --external_plugins /mnt/SSD1/gzh/vlm/reward_latex.py \
  --reward_funcs latex_sim latex_format \
  --num_generations 4 --max_completion_length 256 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate 1e-6 --beta 0.04 --temperature 1.0 \
  --torch_dtype bfloat16 --max_length 2048 --use_vllm false \
  --num_train_epochs 1 --logging_steps 5 --save_steps 100 --save_total_limit 5 \
  --output_dir /mnt/SSD1/gzh/vlm/out_grpo --dataloader_num_workers 2 || exit 1
echo GRPO_TRAIN_DONE
LAST=$(ls -dt out_grpo/v*/checkpoint-* | head -1)
echo "评测 checkpoint: $LAST"
swift infer --adapters $LAST --val_dataset data/val.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path clean_grpo.jsonl
echo GRPO_ALL_DONE
