import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label

from models.backbone import build_backbone
from models.matcher import build_matcher
from .gen import build_gen

from lavis.models import load_model_and_preprocess


"""
BC-HOI 主模型（论文方法的核心实现）。

整体数据流可以对照论文 Figure 3：

    图像
      ├─ CNN 骨干 ─┐
      │             ├─ Transformer Encoder ─► memory
      ├─ BLIP-2 ViT ┘
      │
      ├─ 检测解码器：人体查询 Qh + 物体查询 Qo
      │       └─► 人体/物体框、物体类别、交互查询
      │
      ├─ 交互解码器：用 (h_hs + o_hs)/2 作为查询
      │       └─► 交互特征 inter_hs、交互注意力图 attn_map_inter
      │
      └─ ABG：把 attn_map_inter 变成 BLIP-2 的 attention bias
              └─► BLIP-2 输出实例级视觉特征
                      └─► 与检测器交互特征融合 → 文本分类头

另外：
- hoi_prompt_qformer / hoi_prompt_clip 是复制出来的 cls token；
- embed2qformer_mask / embed2clip_mask 是论文里的 FC 适配层；
- blip_model_opt.forward_with_prompt 同时完成 ABG 的 VLM 前向和 LSG 的逐 token 损失。
"""


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class GEN_VLKT(nn.Module):
    """
    BC-HOI 的一阶段 HOI 检测器。

    设计要点（对应论文）：
    1. 检测解码器负责"人"和"物"两组查询；
    2. 交互解码器负责判断这一对人-物的关系；
    3. ABG 用交互解码器的注意力图去引导 BLIP-2；
    4. LSG 用 BLIP-2 里 OPT 的逐 token 生成损失反过来监督检测器；
    5. 交互类别用文本嵌入的余弦相似度分类，不使用固定的 600 类线性层。
    """

    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # 检测解码器的两组可学习查询：人体查询、物体查询。
        # 这两组参数在所有图像之间共享，匈牙利匹配决定每个查询负责哪个实例。
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        # 额外的一组全局查询，用于提取整图上下文（见 gen.py 的 global_token）。
        self.query_embed_global = nn.Embedding(32, hidden_dim)
        # 位置引导嵌入：给检测查询补充空间先验。
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        # 框回归头：每个查询输出归一化的 (cx, cy, w, h)。
        # 注意：换物体分类头不会影响这两个框回归头。
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 把骨干特征投影到 Transformer 的隐藏维度。
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers
        # 下面几个 Linear 是论文里"把 VLM 特征对齐到检测器维度"的适配层。
        # inter_adapter/obj_adapter 作用在文本嵌入上；prompt2inter 作用在 Q-Former 输出上。
        self.inter_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.obj_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.blip_ln = nn.Linear(768,256)
        self.blip_norm = nn.LayerNorm(hidden_dim)
        self.prompt2inter = nn.Linear(768, 256)
        self.inter_norm = nn.LayerNorm(256)
        self.opt_ln = nn.Linear(2560, 256)


        # 余弦分类的可学习温度，对应 1/tau；初始值为 log(1/0.07)。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        # 用冻结的 BLIP-2 文本编码器预计算三类文本嵌入：
        #   clip_label       : 全部 600 个 HOI 类别的文本向量（评测用）
        #   train_clip_label : 去掉 unseen 后的训练类别文本向量
        #   obj_clip_label   : 80 个物体类别的文本向量
        clip_label, obj_clip_label, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_BLIP(hoi_text_label, obj_text_label, unseen_index)
        num_obj_classes = len(obj_text) - 1  # del nothing

        self.clip_label = clip_label
        self.train_clip_label = train_clip_label
        self.obj_clip_label = obj_clip_label
        # The BLIP-2 feature extractor projects text features to 256 dims,
        # while --clip_embed_dim defaults to 512 (the original CLIP setting).
        # Derive the real dimension from the computed text embeddings so the
        # projection layers below always match (otherwise __init__ raises a
        # shape-mismatch error when assigning the pretrained text weights).
        text_dim = train_clip_label.shape[-1]

        # 交互分类头：先把交互特征过一层 MLP，再和文本嵌入做余弦相似度。
        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # with_clip_label=True 时，交互分类不使用固定的 600 类线性层，
        # 而是把"视觉特征 vs 文本嵌入"的余弦相似度当 logits。
        # 这就是论文能在未见组合上泛化的关键之一。
        if args.with_clip_label:
            # Original: self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
            self.visual_projection = nn.Linear(text_dim, len(hoi_text))
            self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                # Original: self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600)
                self.eval_visual_projection = nn.Linear(text_dim, 600)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            # Original: self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))
            self.hoi_class_embedding = nn.Linear(text_dim, len(hoi_text))

        # 物体分类头的两种模式：
        #   with_obj_clip_label=True  : 物体类别也用文本嵌入余弦相似度（开放词汇）
        #   with_obj_clip_label=False : 退回固定的 80 类线性层（闭集）
        # 注意：物体框仍然由上面的 hum_bbox_embed/obj_bbox_embed 输出，
        # 换分类头不会改变框坐标的预测方式。
        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            # Original: self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
            self.obj_visual_projection = nn.Linear(text_dim, num_obj_classes + 1)
            self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.hidden_dim = hidden_dim
        self.reset_parameters()
        # Nq = 64：64 个交互查询；global_token = 32：32 个全局查询。
        # 这两组数字与论文实现一致。
        self.num_sp_token = 64
        self.global_token = 32
        # 关键：复制 Q-Former 的 query token 和 ViT 的 cls token，
        # 并把 requires_grad_(False)，即它们是冻结的"空白画布"。
        # 64 个副本靠不同的 attention bias 区分，而不是靠各自学习参数。
        cls_token_qformer = self.blip_model_opt.query_tokens.clone().detach().requires_grad_(False)
        self.hoi_prompt_qformer = nn.Parameter(torch.cat([cls_token_qformer[:,:1,:].repeat(1,self.global_token, 1),cls_token_qformer.repeat(1, self.num_sp_token // cls_token_qformer.shape[1], 1)],dim =1))
        # self.hoi_prompt_qformer = cls_token_qformer[:,:1,:].repeat(1, self.num_sp_token, 1)
        cls_token_clip = self.blip_model_opt.visual_encoder.cls_token.clone().detach().requires_grad_(False)
        self.hoi_prompt_clip = cls_token_clip.repeat(1, self.global_token+self.num_sp_token // cls_token_clip.shape[1], 1)
        # 论文里的 FC 适配层：把交互注意力图（256 维空间）映射成
        # ViT/Q-Former 可以接受的 attention bias。
        # 这些 FC 是可学习的，所以偏置的尺度由训练决定，而不是手工调。
        self.embed2qformer_mask = nn.Linear(256, 256)
        self.embed2clip_mask = nn.Linear(256, 256)

        self.embed2qformer_mask_global = nn.Linear(256, 256)
        self.embed2clip_mask_global = nn.Linear(256, 256)

        # ============================================================
        # ROI object head (optional): ViT patch grid -> ROI-align ->
        # ROI positional encoding -> new Q-Former queries -> cosine text logits.
        # Original object head above is kept unchanged for ablation.
        # ============================================================
        self.use_roi_obj_head = getattr(args, 'use_roi_obj_head', False)
        if self.use_roi_obj_head:
            self.roi_size = args.roi_size
            self.roi_query_num = args.roi_query_num
            self.roi_use_global_cls = getattr(args, 'roi_use_global_cls', True)
            self.roi_train_boxes = getattr(args, 'roi_train_boxes', 'pred')
            self.roi_use_lora = getattr(args, 'roi_use_lora', False)
            self.roi_vision_dim = self.blip_model_opt.visual_encoder.num_features
            qformer_dim = self.blip_model_opt.Qformer.config.hidden_size

            # Copy the first m pretrained Q-Former query tokens.
            pretrained_queries = self.blip_model_opt.query_tokens
            if self.roi_query_num <= pretrained_queries.shape[1]:
                roi_queries = pretrained_queries[:, :self.roi_query_num, :]
            else:
                repeat = (self.roi_query_num + pretrained_queries.shape[1] - 1) \
                         // pretrained_queries.shape[1]
                roi_queries = pretrained_queries.repeat(1, repeat, 1)[:, :self.roi_query_num, :]
            self.obj_roi_queries = nn.Parameter(roi_queries.clone().detach())

            # Project the 768-d Q-Former output back to the text-embedding space.
            self.roi_proj = nn.Linear(qformer_dim, text_dim)

            # Positional encoding inside each ROI (r x r tokens).
            if args.roi_pos_embed == 'learned':
                self.roi_pos_embed = nn.Parameter(
                    torch.zeros(1, self.roi_size * self.roi_size, self.roi_vision_dim))
                nn.init.normal_(self.roi_pos_embed, std=0.02)
            else:
                self.register_buffer(
                    'roi_pos_embed',
                    self._build_sine_pos(self.roi_size, self.roi_vision_dim))

            self.roi_logit_scale = nn.Parameter(
                torch.ones([]) * args.roi_logit_scale_init)
            if self.roi_use_lora:
                print('[GEN_VLKT] WARNING: --roi_use_lora is reserved but not implemented '
                      'in this version; the flag is ignored.')



    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_BLIP(self, hoi_text_label, obj_text_label, unseen_index):
        """
        加载 BLIP-2，并预计算交互/物体的文本嵌入。

        这里加载两个模型：
        - blip2_feature_extractor：只用来做文本编码（得到 clip_label 等）；
        - blip2_opt：完整 VLM，训练时用于 ABG/LSG，推理时仍保留视觉塔。

        注意：文本嵌入是在 no_grad 下算的，文本编码器始终冻结。
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model, _, txt_processors= load_model_and_preprocess(name="blip2_feature_extractor",
                                                                              model_type="pretrain_vitL",
                                                                              is_eval=True,
                                                                              device=device)
        self.txt_processors = txt_processors["eval"]

        blip_model_opt, _,_= load_model_and_preprocess(name="blip2_opt",
                                                       model_type="pretrain_opt2.7b",
                                                       is_eval=True,
                                                       device=device)
        self.blip_model_opt = blip_model_opt

        self.seen = []
        self.unseen = []

        # 全部 600 个 HOI 类别的文本嵌入（评测时用，包含 unseen）。
        text_inputs = {"text_input":[self.txt_processors(hoi_text_label[id]) for id in hoi_text_label.keys()]}
        # 如果开启了 del_unseen，就把 unseen 类别从训练用的文本矩阵里删掉。
        # 这样训练时不会直接监督 unseen 类别，但评测时仍可用完整文本矩阵。
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    self.unseen.append(idx)
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
                    self.seen.append(idx)
        else:
            hoi_text_label_del = hoi_text_label.copy()
            self.seen = list(range(len(hoi_text_label)))
        text_inputs_del =  {"text_input":[self.txt_processors(hoi_text_label[id]) for id in hoi_text_label_del.keys()]}

        # 物体类别的文本嵌入，例如 "a photo of a horse"。
        obj_text_inputs = {"text_input":[self.txt_processors(obj_text[1]) for obj_text in obj_text_label]}
        with torch.no_grad():
            text_embedding = blip_model.extract_features(text_inputs,mode = "text",output_mode = "simple")[:,0,:]
            text_embedding_del = blip_model.extract_features(text_inputs_del,mode = "text",output_mode = "simple")[:,0,:]
            obj_text_embedding = blip_model.extract_features(obj_text_inputs,mode = "text",output_mode = "simple")[:,0,:]

        # 文本嵌入已经缓存，卸载只用于文本编码的模型，节省显存。
        del blip_model

        return text_embedding.float(), obj_text_embedding.float(), \
               hoi_text_label_del, obj_text_inputs, text_embedding_del.float()

    # ================================================================
    # ROI object head helpers
    # ================================================================
    @staticmethod
    def _build_sine_pos(size, dim):
        """Fixed 2D sine positional encoding for an r x r ROI grid."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing='ij',
        )
        pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        half = dim // 2
        quarter = half // 2
        omega = torch.arange(quarter, dtype=torch.float32) / max(quarter - 1, 1)
        omega = 1.0 / (10000 ** omega)
        out = torch.zeros(size * size, dim, dtype=torch.float32)
        out[:, :quarter] = torch.sin(pos[:, 0:1] * omega)
        out[:, quarter:half] = torch.cos(pos[:, 0:1] * omega)
        out[:, half:half + quarter] = torch.sin(pos[:, 1:2] * omega)
        out[:, half + quarter:half + 2 * quarter] = torch.cos(pos[:, 1:2] * omega)
        return out.unsqueeze(0)

    def _roi_features_from_boxes(self, image_embeds, boxes_cxcywh, batch_idx, clip_label_norm):
        """
        ROI-align a batch of boxes on the ViT patch grid, run the new Q-Former
        queries, and return cosine logits against the object text embeddings.

        image_embeds:   [bs, 1 + grid*grid, C] (post ln_vision, no prompt tokens)
        boxes_cxcywh:   [N, 4] normalized (cx, cy, w, h)
        batch_idx:      [N] image index for each box
        clip_label_norm:[num_obj + 1, text_dim]
        returns:        [N, num_obj + 1]
        """
        bs, seq_len, channels = image_embeds.shape
        n_patches = seq_len - 1
        grid = int(round(n_patches ** 0.5))
        if grid * grid != n_patches:
            raise ValueError('unexpected ViT patch count: %d' % n_patches)

        # [bs, C, grid, grid], float32 for roi_align
        patch_grid = image_embeds[:, 1:1 + grid * grid, :].reshape(
            bs, grid, grid, channels).permute(0, 3, 1, 2).float()

        boxes = boxes_cxcywh.float()
        cx, cy, bw, bh = boxes.unbind(dim=1)
        x1 = (cx - bw / 2).clamp(min=0.0, max=1.0)
        y1 = (cy - bh / 2).clamp(min=0.0, max=1.0)
        x2 = (cx + bw / 2).clamp(min=0.0, max=1.0)
        y2 = (cy + bh / 2).clamp(min=0.0, max=1.0)
        # Avoid degenerate boxes for roi_align.
        x2 = torch.maximum(x2, x1 + 1e-4)
        y2 = torch.maximum(y2, y1 + 1e-4)
        rois = torch.stack([batch_idx.float(), x1, y1, x2, y2], dim=1)

        roi_feats = roi_align(
            patch_grid, rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=float(grid),
            sampling_ratio=2,
            aligned=True,
        )
        roi_tokens = roi_feats.flatten(2).transpose(1, 2)  # [N, r*r, C]

        pos = self.roi_pos_embed[:, :self.roi_size * self.roi_size, :]
        roi_tokens = roi_tokens + pos.to(device=roi_tokens.device, dtype=roi_tokens.dtype)

        if self.roi_use_global_cls:
            cls = image_embeds[:, 0, :].float()
            cls = cls.to(device=roi_tokens.device, dtype=roi_tokens.dtype)[batch_idx]
            roi_tokens = torch.cat([roi_tokens, cls.unsqueeze(1)], dim=1)

        queries = self.obj_roi_queries.expand(roi_tokens.shape[0], -1, -1)
        query_output = self.blip_model_opt.Qformer.bert(
            query_embeds=queries,
            encoder_hidden_states=roi_tokens,
            encoder_attention_mask=None,
            return_dict=True,
            with_attn_bias=False,
        )
        e_roi = query_output.last_hidden_state
        if e_roi.shape[1] > 1:
            e_roi = e_roi.mean(dim=1)
        else:
            e_roi = e_roi[:, 0, :]

        z_obj = F.normalize(self.roi_proj(e_roi), dim=-1)
        text = clip_label_norm.to(device=z_obj.device, dtype=z_obj.dtype)
        return self.roi_logit_scale.exp() * (z_obj @ text.T)

    def _roi_object_logits(self, image_embeds, outputs_obj_coord, clip_label_norm):
        """ROI logits for every decoder layer: [layers, bs, Nq, num_obj+1]."""
        layers, bs, nq, _ = outputs_obj_coord.shape
        batch_idx = torch.arange(bs, device=outputs_obj_coord.device).repeat_interleave(nq)
        layer_logits = []
        for layer in range(layers):
            boxes = outputs_obj_coord[layer].reshape(bs * nq, 4)
            logits = self._roi_features_from_boxes(
                image_embeds, boxes, batch_idx, clip_label_norm)
            layer_logits.append(logits.reshape(bs, nq, -1))
        return torch.stack(layer_logits, dim=0)

    def _roi_gt_logits(self, image_embeds, targets, clip_label_norm):
        """ROI logits for all GT object boxes, flattened in target order."""
        boxes, batch_idx = [], []
        for i, target in enumerate(targets):
            gt_boxes = target['obj_boxes']
            boxes.append(gt_boxes)
            batch_idx.append(torch.full(
                (gt_boxes.shape[0],), i, dtype=torch.long, device=gt_boxes.device))
        if len(boxes) == 0 or sum(b.shape[0] for b in boxes) == 0:
            return image_embeds.new_zeros((0, clip_label_norm.shape[0]))
        boxes = torch.cat(boxes, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        return self._roi_features_from_boxes(
            image_embeds, boxes, batch_idx, clip_label_norm)

    def forward(self, samples: NestedTensor,targets = None, is_training=True,not_tensor_targets = None,epoch = 0):
        """
        一次完整的前向：
        1. CNN 骨干 + Transformer 编码器；
        2. 检测解码器输出人体/物体查询；
        3. 交互解码器输出交互特征和注意力图；
        4. 把交互注意力图变成 ABG 的 attention bias；
        5. BLIP-2 输出实例级特征，同时计算 LSG 的 token 损失；
        6. 融合特征，用文本嵌入做交互/物体分类。
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # 取骨干最后一层特征；mask 用于忽略 padding 区域。
        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape
        assert mask is not None

        # BLIP-2 的输入是单独的 224×224 图像（数据集里预处理成 blip_inputs）。
        # image_encode(mode="only_qformer") 只跑视觉塔，得到 Q-Former 的全局查询特征。
        blip_image_inputs ={"image":torch.cat([t['blip_inputs'].unsqueeze(0) for t in targets],dim = 0)}
        with torch.no_grad():
            # ROI object head also needs the post-ln_vision ViT patch grid.
            # Original line kept for reference:
            # blip_image_embed = self.blip_model_opt.image_encode(blip_image_inputs,mode = 'only_qformer')
            blip_image_embeds, blip_image_embed = self.blip_model_opt.image_encode(
                blip_image_inputs, mode='clip+qformer')
        blip_image_embed = self.blip_norm(self.blip_ln(blip_image_embed)).permute(1, 0, 2)

        # Transformer 内部同时完成：
        # - instance_decoder：人体/物体检测；
        # - interaction_decoder：交互关系；
        # - 返回交互解码器的 cross-attention 图 attn_map_inter。
        # 这里对应论文的"检测解码器 → 交互解码器"两级结构。
        h_hs, o_hs, inter_hs,global_hs,attn_map_inter,attn_map_global = self.transformer(self.input_proj(src), mask,
                                                                        self.query_embed_h.weight,
                                                                        self.query_embed_o.weight,
                                                                        self.query_embed_global.weight,
                                                                        self.pos_guided_embedd.weight,
                                                                        pos[-1],
                                                                        blip_image_embed,
                                                                        )

        # 框坐标由独立的 box FFN 输出，和后面的分类头无关。
        # 输出是归一化的 (cx, cy, w, h)，sigmoid 保证在 [0,1]。
        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        # ============================================================
        # ABG：构造 attention bias
        # ============================================================
        # attn_map_inter 的 shape 是 [bs, Nq, 32*32]：
        # 交互解码器已经对 CNN 特征做了 cross-attention，并上采样到 32×32。
        # 下面把 256 维的空间图通过 FC 适配成 ViT/Q-Former 的 attention bias。
        ### generate mask
        num_patches = 16 * 16
        num_sp_token = self.num_sp_token
        hoi_prompt_qformer = self.hoi_prompt_qformer.repeat(bs, 1, 1)
        hoi_attn_bias = attn_map_inter  # bs q 256
        hoi_attn_bias_qformer_tmp  = self.embed2qformer_mask(hoi_attn_bias)  # bs q 1+256+64
        hoi_attn_bias_clip_tmp = self.embed2clip_mask(hoi_attn_bias)# bs q 1+256+64

        # 全局查询的注意力图，用来保留整图上下文（对应论文里的 global token）。
        hoi_attn_bias_clip_global_tmp = self.embed2clip_mask_global(attn_map_global)
        hoi_attn_bias_qformer_global_tmp = self.embed2qformer_mask_global(attn_map_global)
        global_token = self.global_token

        # ---- Q-Former 的 bias 矩阵 ----
        # 形状：[bs, 32(global) + 64(HOI), 1(cls) + 256(patch) + 32 + 64]
        # 每行是一组 query，每列是一个 key。
        ### qformer mask
        hoi_attn_bias_qformer = torch.zeros([bs,global_token+num_sp_token, 1 + num_patches + global_token + num_sp_token])
        # 64 个 HOI 查询：对 256 个 patch 的注意力由 attn_map_inter 提供。
        hoi_attn_bias_qformer[:,global_token:,1:1+num_patches] = hoi_attn_bias_qformer_tmp
        # 32 个全局查询：对 256 个 patch 的注意力由 attn_map_global 提供。
        hoi_attn_bias_qformer[:, :global_token, 1:1 + num_patches] = hoi_attn_bias_qformer_global_tmp
        # 关键 mask：global/HOI 查询之间只允许"自己看自己"（对角线 0，其余 -10000）。
        # 这对应论文 Figure 4 的 mask matrix：防止 64 个 cls 分身互相混合。
        matrix_0 = (1 - torch.eye(global_token + num_sp_token, global_token + num_sp_token)) * (-10000.0)
        matrix_0 = matrix_0[None, :, :].repeat(bs, 1, 1)
        hoi_attn_bias_qformer[:, :, 1 + num_patches:] = matrix_0


        # ---- ViT 的 bias 矩阵 ----
        # 形状：[bs, 1+256+32+64, 1+256+32+64]，是方阵。
        hoi_prompt_clip = self.hoi_prompt_clip.repeat(bs, 1, 1)
        clip_prompt_mask = torch.zeros([bs, 1 + num_patches + global_token +num_sp_token, 1 + num_patches + global_token + num_sp_token])
        # 每个 HOI cls 分身对 patch 的注意力由 attn_map_inter 提供。
        clip_prompt_mask[:, -num_sp_token:,1:1 + num_patches] = hoi_attn_bias_clip_tmp
        # 原始 cls 和 patch 不能看见新增的 global/HOI token（保持原 ViT 感受野）。
        clip_prompt_mask[:, :1 + num_patches, -num_sp_token-global_token:] = -10000.0
        # 全局 cls 分身对 patch 的注意力由 attn_map_global 提供。
        clip_prompt_mask[:, -num_sp_token-global_token:-num_sp_token, 1:1 + num_patches] = hoi_attn_bias_clip_global_tmp
        # 新增的 global/HOI token 之间只允许自注意力（同论文 Figure 4 的 mask）。
        matrix_2 = (1 - torch.eye(global_token+num_sp_token, global_token+num_sp_token)) * (-10000.0)
        matrix_2 = matrix_2[None, :, :].repeat(bs, 1, 1)
        clip_prompt_mask[:, 1 + num_patches:,1 + num_patches:] = matrix_2
        # ViT 有 16 个注意力头，所以复制 16 份 bias。
        hoi_attn_bias_clip = clip_prompt_mask[:, None, :, :].repeat(1, 16, 1, 1)

        # Original: hoi_attn_bias_qformer = hoi_attn_bias_qformer.to('cuda')
        # Original: hoi_attn_bias_clip = hoi_attn_bias_clip.to('cuda')
        hoi_attn_bias_qformer = hoi_attn_bias_qformer.to(src.device)
        hoi_attn_bias_clip = hoi_attn_bias_clip.to(src.device)

        # ============================================================
        # BLIP-2 前向 + LSG 损失
        # ============================================================
        # 训练时提供 llava_captions（GPT-4o/QwenVL 生成的描述），
        # forward_with_prompt 会让 OPT 逐 token 生成描述并计算 CE。
        # 推理时用 "_" 占位，不计算 LSG 损失。
        if is_training:
            blip_image_inputs = {"image": torch.cat([t['blip_inputs'].unsqueeze(0) for t in targets], dim=0),
                                 "text_input": [t['llava_captions'] for t in not_tensor_targets]}
        else:
            blip_image_inputs = {"image": torch.cat([t['blip_inputs'].unsqueeze(0) for t in targets], dim=0),
                                 "text_input": ["_"] * bs}

        opt_loss, blip_prompt_embed, opt_in = self.blip_model_opt.forward_with_prompt(blip_image_inputs,
                                                                                      vis_prompt=hoi_prompt_qformer,
                                                                                      image_atts=hoi_attn_bias_qformer,
                                                                                      clip_prompt=hoi_prompt_clip,
                                                                                      clip_image_atts=hoi_attn_bias_clip,
                                                                                      io=global_token,
                                                                                      loss_mask = True)

        # OPT 的输出特征（2560 维）投影到 256 维，供后面的融合使用。
        opt_in = self.opt_ln(opt_in).transpose(0, 1)

        # ============================================================
        # 特征融合：检测器交互特征 + BLIP-2 实例特征 + OPT 特征
        # ============================================================
        # 第 0 个 epoch 先不融合（warm-up），让检测器先稳定；
        # 之后把 Q-Former 输出投影并与交互特征相加，再过一层 Transformer。
        warm_up = True
        if warm_up and epoch == 0:
            inter_hs = inter_hs
        else:
            inter_hs = self.inter_norm(
                inter_hs + self.prompt2inter(blip_prompt_embed)[None, :, :, :].repeat(self.args.dec_layers, 1, 1, 1))
            inter_hs = self.transformer.opt_forward(inter_hs, opt_in) + inter_hs

        # 文本嵌入本身也过一个小残差适配器，便于和视觉特征对齐。
        train_clip_label = self.inter_adapter(self.train_clip_label) + self.train_clip_label
        clip_label = self.inter_adapter(self.clip_label) + self.clip_label
        obj_clip_label = self.obj_adapter(self.obj_clip_label) + self.obj_clip_label

        clip_label_norm = clip_label / clip_label.norm(dim=-1, keepdim=True)
        train_clip_label_norm = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
        obj_clip_label_norm = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)

        # 物体分类：文本嵌入模式 or 固定线性层模式。
        # 两种模式都只作用于 o_hs（检测解码器的物体查询），
        # 框坐标仍然来自前面的 outputs_obj_coord。
        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            # outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
            outputs_obj_class = obj_logit_scale *o_hs @ obj_clip_label_norm.T
        else:
            outputs_obj_class = self.obj_class_embed(o_hs)

        # ROI object head: override the object logits when enabled.
        # The original object head above is kept unchanged so that runs with
        # --use_roi_obj_head unset reproduce the original BC-HOI behavior.
        if getattr(self.args, 'use_roi_obj_head', False):
            outputs_obj_class = self._roi_object_logits(
                blip_image_embeds, outputs_obj_coord, obj_clip_label_norm)

        # 交互分类：同样优先使用文本嵌入的余弦相似度。
        # 评测 unseen 组合时，用完整的 clip_label；
        # 训练 seen 类别时，用去掉 unseen 的 train_clip_label。
        categories_balance = True

        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                # outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                if categories_balance:
                    # 对 unseen 类别的 logits 乘 0.8，做简单的类别平衡。
                    outputs_hoi_class = inter_hs @ clip_label_norm.T
                    outputs_hoi_class[..., self.unseen] = outputs_hoi_class[..., self.unseen] * 0.8
                    outputs_hoi_class = logit_scale * outputs_hoi_class
                else:
                    outputs_hoi_class = logit_scale *inter_hs @ clip_label_norm.T

            else:
                # outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                outputs_hoi_class = logit_scale *inter_hs @ train_clip_label_norm.T
        else:
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)

        # 输出给 SetCriterionHOI：框、物体 logits、交互 logits、LSG 损失。
        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        out['opt_loss'] = opt_loss

        # Optional GT-box ROI auxiliary loss (roi_train_boxes=gt).
        # The main object logits still come from predicted boxes because the
        # Hungarian matcher consumes pred_obj_logits before GT-to-query
        # assignment is known; the GT loss is applied to matched GT boxes here.
        if getattr(self.args, 'use_roi_obj_head', False) and is_training \
                and getattr(self.args, 'roi_train_boxes', 'pred') == 'gt' \
                and targets is not None:
            out['roi_logits_gt'] = self._roi_gt_logits(
                blip_image_embeds, targets, obj_clip_label_norm)

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord)


        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord):

        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1]}
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):
    """
    BC-HOI 的总损失。

    包含三部分：
    1. 物体分类损失 loss_obj_labels（交叉熵，含 background 类）；
    2. 交互分类损失 loss_hoi_labels（多标签 focal loss）；
    3. 框回归损失 loss_sub_obj_boxes（L1 + GIoU）。

    另外，forward 里会把模型输出的 opt_loss（LSG 的逐 token CE）合并进来。
    所以总损失是：
        L = L_obj + L_hoi + L_box + L_lsg
    """

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, _ = clip.load(args.clip_model, device=device)
        self.alpha = args.alpha
        self.args = args

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        """
        物体分类交叉熵。

        src_logits 的形状是 [bs, Nq, C+1]，最后一类是 background。
        只有匈牙利匹配上的查询才被赋予真实类别，其余查询强制预测 background。
        """
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        # 默认所有查询都是 background（索引 num_obj_classes）。
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 匹配上的查询填入真实物体类别。
        target_classes[idx] = target_classes_o

        # empty_weight 给 background 一个较低的权重（eos_coef），缓解正负样本不平衡。
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_obj_labels_roi_gt(self, outputs, targets, indices):
        """
        Optional GT-box ROI auxiliary classification loss.

        ROI_QFormer_OBJ_HEAD_PLAN.md's `roi_train_boxes=gt` cannot replace
        `pred_obj_logits` for every query directly: the Hungarian matcher needs
        per-query logits before GT-to-query assignment, while GT object boxes
        only exist for the matched pairs. We therefore keep the main ROI logits
        predicted-box based and add this GT-box supervision for matched pairs.
        """
        src_logits = outputs['roi_logits_gt']
        if len(indices) == 0 or src_logits.numel() == 0:
            return {'loss_obj_ce_roi_gt': src_logits.sum() * 0.0}

        tgt_idx = torch.cat([J for (_, J) in indices])
        target_classes_o = torch.cat(
            [t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        tgt_boxes = torch.cat(
            [t['obj_boxes'][J] for t, (_, J) in zip(targets, indices)])

        # Skip HOI pairs without an object box (pure action classes).
        valid = (tgt_boxes != 0).any(dim=1)
        logits = src_logits[tgt_idx][valid]
        labels = target_classes_o[valid]
        if labels.numel() == 0:
            return {'loss_obj_ce_roi_gt': src_logits.sum() * 0.0}
        return {'loss_obj_ce_roi_gt': F.cross_entropy(logits, labels)}

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        """
        交互分类损失（多标签）。

        一个查询可以同时对应多个 HOI 类别（例如同一个人既 riding 又 holding），
        所以这里用 sigmoid + focal loss，而不是 softmax 互斥分类。
        注意：论文的文本分类头输出的 logits 仍然逐类独立，
        只是训练目标按多标签处理。
        """
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        """
        人体框 + 物体框的回归损失。

        L1 负责绝对坐标，GIoU 负责重叠面积。
        如果某个 GT 没有物体框（纯动作类），物体框损失会被 mask 掉。
        """
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses


    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        # 多标签 focal loss：正负样本分别按 (1-p)^2 / p^2 加权。
        # 这样模型会更关注难分样本，缓解大量简单负样本主导梯度的问题。
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        """
        先用匈牙利匹配把预测和 GT 对应起来，再计算各项损失。
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # Optional GT-box ROI auxiliary object classification loss.
        if 'roi_logits_gt' in outputs:
            losses.update(self.loss_obj_labels_roi_gt(outputs, targets, indices))

        # LSG：BLIP-2 里 OPT 的逐 token 生成损失。
        # 它不在 loss_map 里，而是由模型 forward 直接带回。
        if "opt_loss" in outputs.keys():
            losses.update(outputs["opt_loss"])

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = GEN_VLKT(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['opt_loss'] = 0.1
    # Optional GT-box ROI auxiliary loss (only set when roi_train_boxes=gt).
    if getattr(args, 'use_roi_obj_head', False) and \
            getattr(args, 'roi_train_boxes', 'pred') == 'gt':
        weight_dict['loss_obj_ce_roi_gt'] = getattr(args, 'roi_gt_loss_coef', 1.0)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors
