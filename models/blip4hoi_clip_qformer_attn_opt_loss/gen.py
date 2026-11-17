import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


"""
BC-HOI 使用的 Transformer 主干。

和标准 DETR 的区别：
1. 输入是 CNN 特征 + BLIP-2 视觉特征的早期融合（open_src）；
2. 检测解码器有 H、O、G 三组查询（人体、物体、全局）；
3. 交互解码器会把每层的 cross-attention 图返回出来，供 ABG 使用；
4. opt_forward 用一层额外的 decoder 把 OPT 的语言特征融合进交互特征。
"""


class GEN(nn.Module):
    """
    检测解码器 → 交互解码器 的两级结构。

    数据流：
        src（CNN 特征） + blip_image_embed（VLM 特征）
            └─► Transformer Encoder ─► memory
                    ├─► instance_decoder（H/O/G 查询）
                    └─► interaction_decoder（H+O 作为查询）
                            └─► 返回交互特征 + 每层注意力图
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before)
        instance_decoder_norm = nn.LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayerAttnMap(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoderAttnMap(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)

        opt_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        opt_decoder_norm = nn.LayerNorm(d_model)
        self.opt_decoder = TransformerDecoder(opt_decoder_layer,
                                              1,
                                              opt_decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.trans_attn_inter = nn.Linear(32 * 32, 16 * 16)
        self.trans_attn_global = nn.Linear(32 * 32, 16 * 16)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed_h, query_embed_o,query_embed_g, pos_guided_embed, pos_embed, blip_image_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]
        global_token = 32

        # ============================================================
        # Early Fusion：沿 token 维拼接 CNN 特征和 VLM 特征
        # ============================================================
        # src 是 CNN 特征，形状 [h*w, bs, C]；
        # blip_image_embed 是 Q-Former 的 32 个全局查询，形状 [32, bs, C]。
        # 注意：源码这里拼接的是"VLM 的 Q-Former 输出"，不是 BLIP-2 的 patch 特征。
        # 论文正文的 early fusion 更接近"CNN 特征 + ViT 特征"，
        # 这一版实现用的是 Q-Former 全局 token，具体以代码为准。
        open_src = torch.cat([src, blip_image_embed], dim=0)
        open_pos_embed = torch.cat([pos_embed, torch.zeros_like(blip_image_embed)], dim=0)
        blip_memory_mask = torch.zeros([bs, blip_image_embed.shape[0]]).to(device=mask.device)

        # 三组查询：人体、物体、全局。
        # 每组查询都是可学习的、跨图像共享的参数。
        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        query_embed_g = query_embed_g.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o,query_embed_g), dim=0)

        mask = mask.flatten(1)
        open_mask = torch.cat([mask, blip_memory_mask.bool()], dim=1)

        # 编码器：对拼接后的 token 做自注意力，得到 memory。
        ins_tgt = torch.zeros_like(ins_query_embed)
        memory = self.encoder(open_src, src_key_padding_mask=open_mask, pos=open_pos_embed)

        # 检测解码器：H/O/G 三组查询分别做 cross-attention。
        ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=open_mask,
                                                      pos=open_pos_embed, query_pos=ins_query_embed)
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:-global_token, :]
        g_hs = ins_hs[:,:,-global_token:,:]

        # add
        # 交互查询 = 人体查询 + 物体查询（逐元素平均），
        # 再拼接全局查询，得到"人-物对 + 全图上下文"的查询。
        ins_guided_embed = (h_hs + o_hs) / 2.0
        ins_guided_embed = torch.cat([ins_guided_embed,g_hs],dim=2)
        ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)

        # 交互解码器：返回特征 + 每层 cross-attention 图。
        inter_tgt = torch.zeros_like(ins_guided_embed[0])
        inter_hs, attn_map_list = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=open_mask,
                                                           pos=open_pos_embed, query_pos=ins_guided_embed)

        # ============================================================
        # 注意力图后处理（ABG 的输入）
        # ============================================================
        # attn_map_list 的形状：[num_layers, bs, Nq+32, num_keys]。
        # 1) 对所有层取平均；
        # 2) 只取 CNN 特征那段 key（前 h*w 个）；
        # 3) reshape 成二维图并上采样到 32×32；
        # 4) 用 trans_attn_inter 把 1024 维映射到 256 维（16×16）。
        # 这个 256 维向量就是论文里的 A_ho，后面再经 FC 变成 attention bias。
        attn_map = attn_map_list.mean(dim=0)
        attn_map = attn_map[:, :-global_token, :h * w].view(bs, num_queries, h, w)
        attn_map = F.interpolate(attn_map, [32, 32])
        attn_map = attn_map.view(bs, num_queries, 32 * 32)
        attn_map_inter = self.trans_attn_inter(attn_map)

        # 全局查询的注意力图，单独处理后用于保留整图上下文。
        attn_map_global = attn_map_list.mean(dim=0)
        attn_map_global = attn_map_global[:, -global_token:, :h * w].view(bs, global_token, h, w)
        attn_map_global = F.interpolate(attn_map_global, [32, 32])
        attn_map_global = attn_map_global.view(bs, global_token, 32 * 32)
        attn_map_global = self.trans_attn_inter(attn_map_global)

        global_hs = inter_hs[:,-global_token:,:,:]
        inter_hs = inter_hs[:,:-global_token,:,:]
        inter_hs = inter_hs.transpose(1, 2)
        global_hs = global_hs.transpose(1, 2)

        return h_hs, o_hs, inter_hs,global_hs,attn_map_inter,attn_map_global

    def opt_forward(self,inter_hs,opt_in):
        """
        用一层 Transformer decoder 把 OPT 的语言特征融合进交互特征。

        inter_hs 作为 query_pos，opt_in 作为 memory（K/V），
        输出仍然是交互特征的形状。对应论文 Figure 3 里的 Fusion Adapter。
        """
        inter_hs = inter_hs.transpose(1, 2)
        layer_num, cls_num, bs, dim = inter_hs.shape
        #############
        text_tgt = torch.zeros_like(inter_hs[0])
        text_memory_mask = torch.zeros([bs, opt_in.shape[0]]).to(device=inter_hs.device)
        inter_hs = self.opt_decoder(text_tgt, opt_in,memory_key_padding_mask=text_memory_mask,
                                                          pos=torch.zeros_like(opt_in), query_pos=inter_hs)
        inter_hs = inter_hs.transpose(1,2)
        return inter_hs

class TransformerDecoderLayerAttnMap(nn.Module):
    """
    与普通 TransformerDecoderLayer 相同，但额外返回 cross-attention 图。

    普通 nn.MultiheadAttention 默认只返回输出；
    这里使用默认的 need_weights=True，从而拿到 [bs, num_queries, num_keys] 的注意力权重。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2,attn_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt,attn_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2,attn_map = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt,attn_map

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderAttnMap(nn.Module):
    """
    把每层 decoder layer 返回的 attention map 堆叠起来，
    供 GEN.forward 做跨层平均。
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        attn_map_list = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output,attn_map = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            attn_map_list.append(attn_map)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate),torch.stack(attn_map_list)

        return output,torch.stack(attn_map_list)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
