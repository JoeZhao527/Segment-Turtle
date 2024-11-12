python mask_rcnn_train.py \
  --output_dir=./output_mask_rcnn \
  --dev

python dual_prop_rcnn.py \
  --output_dir=./output_dual_prop_rcnn \
  --dev

python unet_train.py \
  --output_dir=./output_unet \
  --dev

python focused_unet_evaluate.py \
  --output_dir=./focused_unet_results \
  --model_path=./output_unet/model_best.pth \
  --background_path=./output_unet/sem_seg_predictions.json \
  --focus_crop_ratio=0.1 \
  --dev
