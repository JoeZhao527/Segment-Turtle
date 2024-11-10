python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_01 \
  --focus_crop_ratio=0.1

python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_04 \
  --focus_crop_ratio=0.4

python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_07 \
  --focus_crop_ratio=0.7

python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_10 \
  --focus_crop_ratio=1.0

python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_20 \
  --focus_crop_ratio=2.0

python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_50 \
  --focus_crop_ratio=5.0

python focused_unet_evaluate.py \
  --background_path=./evaluate_unet/sem_seg_predictions.json \
  --output_dir=./focused_unet_results/r_100 \
  --focus_crop_ratio=10.0
