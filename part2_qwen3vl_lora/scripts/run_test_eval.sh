export CUDA_VISIBLE_DEVICES=$GPU MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
python prep_test.py || exit 1
M=/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct
BEST=$(ls -d out_vit3ep/v*/checkpoint-1875 | head -1)
FROZEN=$(ls -d out_r16_3ep/v*/checkpoint-1875 | head -1)
echo "=== 基线（未微调）==="
swift infer --model $M --val_dataset data/test.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path test_base.jsonl || exit 2
echo "=== 冻结 ViT 3ep ==="
swift infer --adapters $FROZEN --val_dataset data/test.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path test_frozen3ep.jsonl || exit 3
echo "=== 解冻 ViT 3ep（最终模型）==="
swift infer --adapters $BEST --val_dataset data/test.jsonl --infer_backend pt \
  --max_new_tokens 256 --temperature 0 --max_batch_size 16 --result_path test_vit3ep.jsonl || exit 4
echo TEST_DONE
