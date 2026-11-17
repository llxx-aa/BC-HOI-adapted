import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
import util.swanlab_logger as swanlab_logger
from datasets.hico_eval_triplet import HICOEvaluator
from datasets.hico_eval_triplet_ko import HICOEvaluator as HICOEvaluator_KO
from datasets.vcoco_eval import VCOCOEvaluator


"""
训练与评测循环。

训练一轮的核心步骤：
1. 把图像和标签搬到 GPU；
2. 前向模型，得到框、物体/交互 logits、LSG 损失；
3. 用 SetCriterionHOI 计算匈牙利匹配后的各项损失；
4. 按 weight_dict 加权求和，反向传播。

评测时：
1. 前向模型（is_training=False）；
2. PostProcessHOITriplet 把 logits 解码成三元组；
3. 用 HICO/V-COCO evaluator 计算 mAP。
"""


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    log_interval: int = 10):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'):
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for it, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        # llava_captions 是 GPT-4o/QwenVL 生成的描述，只用于 LSG；
        # filename 用于评测时找原图。两者都不是张量，单独拿出来。
        not_tensor_targets = [{k: v for k, v in t.items() if k == 'filename' or k == "llava_captions"} for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != "llava_captions"} for t in targets]

        # 模型内部会同时返回主损失项（框/物体/交互）和 opt_loss（LSG）。
        outputs = model(samples,targets = targets, not_tensor_targets = not_tensor_targets, epoch = epoch)
        loss_dict = criterion(outputs, targets = targets)
        weight_dict = criterion.weight_dict
        # 按权重求和；opt_loss 的权重也在 weight_dict 里配置。
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(loss_value)
        # sys.exit()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'):
            metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # SwanLab: per-iteration losses, learning rate and stage errors.
        # Throttled by --swanlab_log_interval to avoid flooding the dashboard.
        if utils.is_main_process():
            iter_metrics = {
                'loss': loss_value,
                'lr': optimizer.param_groups[0]["lr"],
            }
            for key, value in loss_dict_reduced_scaled.items():
                iter_metrics['loss_scaled/' + key] = value
            for key, value in loss_dict_reduced_unscaled.items():
                iter_metrics['raw/' + key] = value
            for error_key in ('obj_class_error', 'hoi_class_error', 'class_error'):
                if error_key in loss_dict_reduced:
                    error_value = loss_dict_reduced[error_key]
                    iter_metrics['stage/' + error_key] = error_value
                    if error_key.endswith('_error'):
                        # BC-HOI reports classification *error* (%); also log
                        # the corresponding accuracy (100 - error).
                        iter_metrics['stage/' + error_key[:-6] + '_acc'] = \
                            100.0 - error_value
            swanlab_logger.log_iter(
                iter_metrics, epoch=epoch, iteration=it,
                loader_len=len(data_loader), interval=log_interval)

        # break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args,epoch = -1):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    counter = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        not_tensor_targets = [{k: v for k, v in t.items() if k == 'filename' or k == "llava_captions"} for t in targets]
        # The model needs target['blip_inputs'] on the same device as BLIP-2.
        # (Previously targets stayed on CPU, which made evaluation on GPU fail.)
        model_targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]
        # 推理时 LSG 不需要文本描述，模型内部用 "_" 占位。
        outputs = model(samples,targets = model_targets, is_training=False, not_tensor_targets =  not_tensor_targets,epoch= epoch)
        orig_target_sizes = torch.stack([t["orig_size"] for t in model_targets], dim=0)
        # 把 [bs, Nq, C] 的 logits 解码成三元组列表（人框、物框、物体类、交互类）。
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # Drop the (large) 224x224 BLIP input before gathering/evaluating.
        eval_targets = [{k: v for k, v in t.items() if k != 'blip_inputs'} for t in targets]
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(eval_targets)))))

        # counter += 1


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]


    if dataset_file == 'hico':
        if args.KO == True:
            # Known-Object 评测：只在"图里确实有该物体"的图上算 AP。
            evaluator = HICOEvaluator_KO(preds, gts, data_loader.dataset.rare_triplets,
                                      data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
        else:
            # Default 评测：在完整测试集上算 AP（包含没有该物体的背景图）。
            evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                      data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats
