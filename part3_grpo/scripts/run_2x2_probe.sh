export CUDA_VISIBLE_DEVICES=3 MODELSCOPE_CACHE=/mnt/SSD1/gzh/modelscope TMPDIR=/mnt/SSD1/gzh/tmp
export PATH=/mnt/SSD1/gzh/envs/swift/bin:$PATH
cd /mnt/SSD1/gzh/vlm
head -200 data/train.jsonl > data/train_probe.jsonl
head -200 data/val.jsonl   > data/val_probe.jsonl
SFT=out_r16_3ep/v0-20260926-025757/checkpoint-1875
GRPO=out_grpo/v0-20260928-041208/checkpoint-500
for M in sft grpo; do
  [ $M = sft ] && CK=$SFT || CK=$GRPO
  for D in train val; do
    for MODE in greedy sample; do
      [ $MODE = greedy ] && T=0 || T=1.0
      OUT=probe_${M}_${D}_${MODE}.jsonl
      [ -f $OUT ] && continue
      echo "=== $M / $D / $MODE (temperature=$T) ==="
      swift infer --adapters $CK --val_dataset data/${D}_probe.jsonl --infer_backend pt \
        --max_new_tokens 256 --temperature $T --max_batch_size 16 --result_path $OUT || exit 1
    done
  done
done
echo PROBE_DONE
