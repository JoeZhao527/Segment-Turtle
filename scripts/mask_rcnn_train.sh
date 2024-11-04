rm -rf ./output_mask_rcnn
rm -rf ./output_dual_prop_rcnn
rm -rf ./output_dual_prop_dual_rcnn

CUDA_VISIBLE_DEVICES=0 python mask_rcnn_train.py --output_dir ./output_mask_rcnn > mask_rcnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python dual_prop_rcnn.py --output_dir ./output_dual_prop_rcnn > dual_prop_rcnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python dual_prop_dual_rcnn.py --output_dir ./output_dual_prop_dual_rcnn > dual_prop_dual_rcnn.log 2>&1 &

wait