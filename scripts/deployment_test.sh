python mask_rcnn_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_mask_rcnn_dev \
  --dev

python dual_prop_rcnn.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_dual_prop_rcnn_dev \
  --intersection_thresh=0.6 \
  --dev

python unet_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_unet_dev \
  --dev

python focused_unet_evaluate.py \
  --data_dir=./turtles-data/data \
  --output_dir=./focused_unet_results_dev \
  --model_path=./output_unet_dev/model_best.pth \
  --background_path=./output_unet_dev/sem_seg_predictions.json \
  --focus_crop_ratio=0.1 \
  --dev
