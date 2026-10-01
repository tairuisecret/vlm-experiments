export CUDA_VISIBLE_DEVICES=6 TMPDIR=/mnt/SSD1/gzh/tmp
cd /mnt/SSD1/gzh/video
rm -f ablation_results.jsonl
# 实验一：固定分辨率，扫抽帧数（1 = 单帧静态图对照）
/mnt/SSD1/gzh/envs/swift/bin/python video_ablation.py   --n_q 100 --frames 1 2 4 8 16 32 --out ablation_results.jsonl 2>&1 | grep -E '^\{|题可用|Error|error'
# 实验二：固定 8 帧，扫每帧分辨率（用 --frames 8 占位，再单独跑各分辨率）
for PX in 25088 50176 200704 401408; do
  /mnt/SSD1/gzh/envs/swift/bin/python video_ablation.py     --n_q 100 --frames 8 --base_pixels $PX --out ablation_results.jsonl 2>&1 | grep -E '^\{|Error|error'
done
echo VIDEO_ABLATION_ALL_DONE
