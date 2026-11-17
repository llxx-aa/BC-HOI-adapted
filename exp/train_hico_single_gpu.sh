set -x
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
EXP_DIR=exps/BC-HOI-HICO-single

# Single-GPU variant of train_hico.sh.
# The original script uses torch.distributed.launch with 8 processes, which
# does not work on a single-GPU Windows machine. Activate the DL env first:
#     conda activate DL
# Then run from the repository root:
#     bash exp/train_hico_single_gpu.sh
python main.py \
        --pretrained params/detr-r50-pre-2branch-hico.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 70 \
        --lr_drop 40 \
        --batch_size 2 \
        --num_workers 0 \
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
        --swanlab_experiment BC-HOI-HICO-single \
        --swanlab_mode local \
        --swanlab_logdir ${EXP_DIR}/swanlab \
        --swanlab_log_interval 10

# SwanLab notes:
#   --swanlab_mode local    : local dashboard, no login needed (default here)
#   --swanlab_mode online   : upload to SwanLab cloud, run `swanlab login` first
#   --swanlab_mode offline  : write logs only
#   --swanlab_mode disabled : no-op
#   --swanlab_experiment    : run name shown in SwanLab
# View local results with:
#   swanlab watch exps/BC-HOI-HICO-single/swanlab

# ROI object head notes:
#   --roi_train_boxes gt  : add the GT-box ROI auxiliary loss
#   --roi_query_num 4     : multi-query ablation (mean pooling)
#   --roi_pos_embed sine  : fixed 2D sine encoding instead of learned
#   --roi_no_global_cls   : ablation without the global ViT cls token
#   --roi_unfreeze_qformer: stage C, train the Q-Former body
#   --roi_use_lora        : reserved, currently not implemented

# For inference add:
#        --eval
# For UO/UV/UC evaluation add:
#        --zero_shot_type unseen_object --fix_clip --del_unseen
