python dual_prop_evaluate.py \
    --model_path output_dual_prop_rcnn/model_best.pth \
    --score_thresh 0.2 \
    --output_dir evaluate_dual_prop_rcnn

python dual_prop_evaluate.py \
    --model_path output_dual_prop_rcnn/model_best.pth \
    --score_thresh 0.4 \
    --output_dir evaluate_dual_prop_rcnn

python dual_prop_evaluate.py \
    --model_path output_dual_prop_rcnn/model_best.pth \
    --score_thresh 0.6 \
    --output_dir evaluate_dual_prop_rcnn

python dual_prop_evaluate.py \
    --model_path output_dual_prop_rcnn/model_best.pth \
    --score_thresh 0.8 \
    --output_dir evaluate_dual_prop_rcnn


# python dual_prop_evaluate.py \
#   --model_path output_dual_prop_rcnn/model_best.pth \
#   --score_thresh 0.2 \
#   --output_dir debug_dual_prop_rcnn \
#   --dev