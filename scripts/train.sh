python mask_rcnn_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_mask_rcnn

python dual_prop_rcnn.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_dual_prop_rcnn \
  --intersection_thresh=0.6

python unet_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_unet

python focused_unet_evaluate.py \
  --data_dir=./turtles-data/data \
  --output_dir=./focused_unet_results \
  --model_path=./output_unet/model_best.pth \
  --background_path=./output_unet/sem_seg_predictions.json \
  --focus_crop_ratio=0.1
