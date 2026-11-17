set -x
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
# Single-GPU evaluation for a checkpoint trained with the ROI object head.
# Usage:
#   bash exp/eval_hico_single_gpu.sh [checkpoint_path]
# Default checkpoint:
#   exps/BC-HOI-HICO-single/checkpoint_best.pth
#
# If you evaluate a checkpoint WITHOUT the ROI head, remove
# --use_roi_obj_head / --roi_size / --roi_query_num / --roi_train_boxes below.

CKPT=${1:-exps/BC-HOI-HICO-single/checkpoint_best.pth}
EXP_DIR=exps/BC-HOI-HICO-single-eval

python main.py \
        --eval \
        --resume ${CKPT} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --batch_size 1 \
        --num_workers 0 \
        --output_dir ${EXP_DIR} \
        --json_file ${EXP_DIR}/results.json \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --model_name blip4hoi_clip_qformer_attn_opt_loss \
        --use_roi_obj_head \
        --roi_size 4 \
        --roi_query_num 1 \
        --roi_train_boxes pred \
        --use_swanlab \
        --swanlab_project bc-hoi \
        --swanlab_experiment BC-HOI-HICO-single-eval \
        --swanlab_mode local \
        --swanlab_logdir ${EXP_DIR}/swanlab

# SwanLab: local mode needs no login; for the cloud use
# --swanlab_mode online after `swanlab login`. Remove --use_swanlab to disable.

# Default test set:
#   data/hico_20160224_det/annotations/test_hico.json
#
# Zero-shot variants (append one of these to the command above):
#   --zero_shot_type unseen_object --fix_clip --del_unseen   # UO
#   --zero_shot_type unseen_verb   --fix_clip --del_unseen   # UV
#   --zero_shot_type rare_first    --fix_clip --del_unseen   # UC (rare)
#   --zero_shot_type non_rare_first --fix_clip --del_unseen  # UC (non-rare)
# Known-Object evaluation:
#   --KO
