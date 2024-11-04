rm -rf ./output_mask_rcnn
rm -rf ./output_dual_prop_rcnn
rm -rf ./output_dual_prop_dual_rcnn

CUDA_VISIBLE_DEVICES=5 python mask_rcnn_train.py --output_dir ./output_mask_rcnn > mask_rcnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python dual_prop_rcnn.py --output_dir ./output_dual_prop_rcnn > dual_prop_rcnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python dual_prop_dual_rcnn.py --output_dir ./output_dual_prop_dual_rcnn > dual_prop_dual_rcnn.log 2>&1 &

wait