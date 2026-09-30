export CUDA_VISIBLE_DEVICES=$GPU MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
python prep_ood.py || exit 1
M=/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct
BEST=$(ls -d out_vit3ep/v*/checkpoint-1875 | head -1)
echo "=== OOD 基线 ==="
swift infer --model $M --val_dataset data/ood.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path ood_base.jsonl || exit 2
echo "=== OOD 最终模型 ==="
swift infer --adapters $BEST --val_dataset data/ood.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path ood_vit3ep.jsonl || exit 3
echo OOD_DONE
