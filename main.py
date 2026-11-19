import os

# Keep every downloaded pretrained weight inside the project directory.
# TORCH_HOME controls the timm/torch.hub cache; HF_HOME/HF_HUB_CACHE control
# the HuggingFace cache. Externally set variables still take precedence.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(_PROJECT_ROOT, "weights", "torch"))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, "weights", "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_PROJECT_ROOT, "weights", "huggingface", "hub"))

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
import util.swanlab_logger as swanlab_logger
# Original: from datasets import build_dataset,build_caption
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os
import sys
# Original: import loralib as lora

# SwanLab/rich prints emoji; the default GBK console on Chinese Windows
# raises UnicodeEncodeError. Reconfigure stdout/stderr to UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and _stream.encoding.lower() not in ('utf-8', 'utf8'):
            _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass


"""
BC-HOI 训练/评测入口。

关键参数：
- backbone/enc_layers/dec_layers：CNN 骨干、编码器层数、检测+交互解码器层数；
- with_clip_label：交互分类是否使用文本嵌入余弦相似度（而不是固定 600 类线性层）；
- with_obj_clip_label：物体分类是否使用文本嵌入（开放词汇物体头）；
- zero_shot_type：default / rare_first / non_rare_first / unseen_object / unseen_verb；
- del_unseen：训练时是否从文本矩阵里删掉 unseen 类别；
- KO：评测时是否使用 Known-Object 设置。
"""


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_clip', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of stage1 decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--with_mimic', action='store_true',
                        help="Use clip feature mimic")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_hoi', default=1, type=float,
                        help="Hoi class coefficient")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--hoi_loss_coef', default=2, type=float)
    parser.add_argument('--mimic_loss_coef', default=20, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # clip
    parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                        help='Use smaller learning rate to finetune clip weights')
    # 交互分类：用文本嵌入的余弦相似度，而不是固定的 600 类 FC。
    # 这是 BC-HOI 在 open vocabulary 设定下能泛化的关键开关。
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify HOI')
    parser.add_argument('--early_stop_mimic', action='store_true', help='stop mimic after step')
    # 物体分类：用物体名文本嵌入做余弦分类。
    # 打开后，模型才真正具备"输出未见物体类别"的能力。
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    parser.add_argument('--fix_clip', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)

    # ============================================================
    # ROI object head (ROI-align + Q-Former new queries)
    # See ROI_QFormer_OBJ_HEAD_PLAN.md.
    # ============================================================
    parser.add_argument('--use_roi_obj_head', action='store_true',
                        help='Use ROI-align + Q-Former object head instead of the query-feature head')
    parser.add_argument('--roi_size', default=4, type=int,
                        help='ROI-align output grid size r (r x r tokens per box)')
    parser.add_argument('--roi_query_num', default=1, type=int,
                        help='Number of new Q-Former queries m per object box')
    parser.add_argument('--roi_train_boxes', default='pred', choices=('pred', 'gt'),
                        help="'pred': ROI uses predicted object boxes (end-to-end). "
                             "'gt': additionally adds a GT-box ROI auxiliary classification loss "
                             "(the matcher still needs predicted-box logits).")
    parser.add_argument('--roi_freeze_qformer', action='store_true', default=True,
                        help='Freeze the Q-Former body (default: True)')
    parser.add_argument('--roi_unfreeze_qformer', dest='roi_freeze_qformer',
                        action='store_false',
                        help='Allow the Q-Former body to be trained (stage C)')
    parser.add_argument('--roi_use_lora', action='store_true',
                        help='Reserved for Q-Former cross-attention LoRA (not implemented yet)')
    parser.add_argument('--roi_pos_embed', default='learned', choices=('learned', 'sine'),
                        help='Positional encoding inside each ROI')
    parser.add_argument('--roi_use_global_cls', action='store_true', default=True,
                        help='Concatenate the global ViT cls token to each ROI (default: True)')
    parser.add_argument('--roi_no_global_cls', dest='roi_use_global_cls', action='store_false',
                        help='Ablation: do not concatenate the global cls token')
    parser.add_argument('--roi_logit_scale_init', default=2.659, type=float,
                        help='Initial log temperature for ROI cosine logits (log(1/0.07))')
    parser.add_argument('--roi_lr', default=1e-4, type=float,
                        help='Learning rate for the new ROI head parameters')
    parser.add_argument('--roi_gt_loss_coef', default=1.0, type=float,
                        help='Weight of the optional GT-box ROI auxiliary loss')

    # ============================================================
    # SwanLab experiment logging (optional)
    # ============================================================
    parser.add_argument('--use_swanlab', action='store_true',
                        help='Log losses/accuracy/mAP to SwanLab')
    parser.add_argument('--swanlab_project', default='bc-hoi', type=str)
    parser.add_argument('--swanlab_experiment', default='', type=str)
    parser.add_argument('--swanlab_mode', default='online',
                        choices=('online', 'local', 'offline', 'disabled', 'cloud'),
                        help='SwanLab mode; online needs `swanlab login` or SWANLAB_API_KEY')
    parser.add_argument('--swanlab_logdir', default='', type=str,
                        help='Optional local log directory for SwanLab')
    parser.add_argument('--swanlab_log_interval', default=10, type=int,
                        help='Log per-iteration metrics every N iterations')

    # zero shot type
    # 选择评测划分：
    #   unseen_object  = UO
    #   unseen_verb    = UV
    #   rare_first / non_rare_first = RF-UC / NF-UC
    parser.add_argument('--zero_shot_type', default='default',
                        help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
    # 训练时是否把 unseen 类别从文本矩阵里删掉（严格零样本设置常用）。
    parser.add_argument('--del_unseen', action='store_true', help='')
    parser.add_argument('--model_name', default="", type=str)
    parser.add_argument('--human_idx', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=117)

    parser.add_argument('--KO', action='store_true', help='')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # Original: SwanLab was initialized before build_model().
    # Build model/weights first, then initialize SwanLab, so that any
    # SwanLab internal safe.block/error cannot interrupt the huge download.
    # if utils.is_main_process():
    #     swanlab_logger.init_swanlab(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # Initialize SwanLab only after model/weights are loaded. The BLIP-2/OPT
    # download happens inside build_model() and must not be interrupted by
    # SwanLab's internal exception handling.
    if utils.is_main_process():
        swanlab_logger.init_swanlab(args)

    print('****************')
    print(model)
    print('****************')

    # del model.blip_model_opt.opt_model

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)
    if utils.is_main_process():
        swanlab_logger.log({'model/n_parameters': n_parameters}, step=0)
    #
    # n_parameters_real = sum(p.numel() for name, p in model.named_parameters() if 'opt' not in name)
    # print('number of params:', n_parameters_real)



    for name, p in model.named_parameters():
        # print(name,p.requires_grad)
        if 'eval_visual_projection' in name:
            p.requires_grad = False
        if 'blip_model' in name:
            p.requires_grad = False
        # ROI object head stage C: optionally let the Q-Former body train.
        # (The default --roi_freeze_qformer keeps the original frozen behavior.)
        if args.use_roi_obj_head and not args.roi_freeze_qformer and 'Qformer' in name:
            p.requires_grad = True

    if args.fix_clip:
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name:
                p.requires_grad = False

    if args.ft_clip_with_small_lr:
        if args.with_obj_clip_label and args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n and 'opt_proj' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('visual_projection' in n or 'obj_visual_projection' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'opt_proj' in n and p.requires_grad],
                    "lr": 1e-5,
                },
            ]
        elif args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'visual_projection' in n and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_obj_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'obj_visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'obj_visual_projection' in n and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        else:
            raise

    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

    # New ROI head parameters get their own group. They are first removed from
    # the groups above so the optimizer does not receive duplicate parameters.
    if getattr(args, 'use_roi_obj_head', False):
        roi_params = [p for n, p in model_without_ddp.named_parameters()
                      if 'roi' in n.lower() and p.requires_grad]
        if roi_params:
            roi_ids = {id(p) for p in roi_params}
            for group in param_dicts:
                group['params'] = [p for p in group['params'] if id(p) not in roi_ids]
            param_dicts.append({'params': roi_params, 'lr': args.roi_lr})

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)



    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)



    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # Original: model_without_ddp.load_state_dict(checkpoint['model'])
        # strict=False so that checkpoints without newly added parameters
        # (e.g. a future ROI head) can still be resumed.
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    if args.eval:
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
        if utils.is_main_process():
            swanlab_logger.log(
                {f'test/{k}': v for k, v in test_stats.items()}, step=0)
        swanlab_logger.finish()
        return

    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, log_interval=args.swanlab_log_interval)
        if utils.is_main_process():
            epoch_step = epoch * len(data_loader_train)
            train_log = {f'train/{k}': v for k, v in train_stats.items()}
            for error_key in ('obj_class_error', 'hoi_class_error'):
                if error_key in train_stats:
                    train_log[f'train/{error_key[:-6]}_acc'] = \
                        100.0 - train_stats[error_key]
            swanlab_logger.log(train_log, step=epoch_step)
            swanlab_logger.log({'epoch': epoch}, step=epoch_step)
        lr_scheduler.step()
        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if epoch < args.lr_drop and epoch % 5 != 0:  ## eval every 5 epoch before lr_drop
            continue

        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args, epoch= epoch)
        if utils.is_main_process():
            epoch_step = epoch * len(data_loader_train)
            swanlab_logger.log(
                {f'test/{k}': v for k, v in test_stats.items()}, step=epoch_step)
        if args.dataset_file == 'hico':
            if args.zero_shot_type != 'default':
                if args.zero_shot_type == 'unseen_object':
                    performance = test_stats['mAP unseen-obj']
                else:
                    performance = test_stats['mAP unseen']
            else:
                performance = test_stats['mAP']
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']
        elif args.dataset_file == 'hoia':
            performance = test_stats['mAP']

        if utils.is_main_process():
            swanlab_logger.log(
                {'eval/performance': performance, 'eval/best_performance': best_performance},
                step=epoch * len(data_loader_train))

        if performance >= best_performance:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    swanlab_logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
