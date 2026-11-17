# BC-HOI 源码阅读指南

> 这份指南配合源码里的中文注释使用。目标是帮你把论文里的模块（EF / ABG / LSG / 文本分类头）和代码里的具体行对应起来。

---

## 1. 建议阅读顺序

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `main.py` | 训练入口、关键开关（`with_clip_label`、`with_obj_clip_label`、`zero_shot_type`、`del_unseen`） |
| 2 | `engine.py` | 一个 epoch 里数据怎么流动、损失怎么累加、评测怎么做 |
| 3 | `datasets/hico.py` | 一张图怎么变成 `target`：框、HOI 标签、`blip_inputs`、`llava_captions` |
| 4 | `models/blip4hoi_clip_qformer_attn_opt_loss/gen_vlkt.py` | **主模型**：ABG 的 mask 构造、BLIP-2 前向、文本分类、损失 |
| 5 | `models/blip4hoi_clip_qformer_attn_opt_loss/gen.py` | 检测解码器 → 交互解码器；注意力图怎么返回、怎么上采样、怎么融合 OPT 特征 |
| 6 | `LAVIS-modified/.../blip2_opt.py` | `forward_with_prompt`：ABG 的 VLM 前向 + LSG 的逐 token 损失 |
| 7 | `LAVIS-modified/.../Qformer.py` | attention bias 在哪里加到 softmax 之前 |
| 8 | `models/matcher.py` | 匈牙利匹配的代价函数 |
| 9 | `models/backbone.py` | ResNet 骨干输出 |

---

## 2. 论文模块 → 代码位置

| 论文模块 | 代码位置 | 说明 |
|---|---|---|
| **Early Fusion** | `gen.py` 的 `GEN.forward` 中 `open_src = torch.cat([src, blip_image_embed], dim=0)` | 把 CNN 特征和 VLM 特征沿 token 维拼接后送进 Transformer 编码器 |
| **检测解码器** | `gen.py` 的 `self.instance_decoder` | 输入 H/O/G 三组查询，输出人体/物体/全局特征 |
| **交互解码器** | `gen.py` 的 `self.interaction_decoder` | 用 `(h_hs + o_hs)/2` 作为查询，返回交互特征 + 注意力图 |
| **A_ho 注意力图** | `gen.py` 中 `attn_map_list.mean(dim=0)` 之后的几行 | 跨层平均、只取 CNN 特征段、上采样到 32×32、FC 到 256 维 |
| **ABG 的 attention bias** | `gen_vlkt.py` 的 `forward` 中 `### generate mask` 一段 | 构造 Q-Former 和 ViT 的 bias 矩阵、0/-10000 mask |
| **ABG 的注入点** | `Qformer.py` 的 `BertSelfAttention.forward` 中 `attention_scores = attention_scores + attention_mask` | bias 在 softmax 之前加到注意力分数上 |
| **LSG 的逐 token 损失** | `blip2_opt.py` 的 `forward_with_prompt` | 词性加权、`-100` mask、OPT 的 CE loss |
| **交互文本分类头** | `gen_vlkt.py` 的 `with_clip_label` 分支 | 视觉特征与 600 个 HOI 文本嵌入算余弦相似度 |
| **物体文本分类头** | `gen_vlkt.py` 的 `with_obj_clip_label` 分支 | 视觉特征与 80 个物体文本嵌入算余弦相似度 |
| **框回归头** | `gen_vlkt.py` 的 `hum_bbox_embed` / `obj_bbox_embed` | 独立的 FFN，输出归一化 `(cx, cy, w, h)` |
| **匈牙利匹配** | `matcher.py` 的 `HungarianMatcherHOI.forward` | 交互类别 + 物体类别 + L1 + GIoU |
| **总损失** | `gen_vlkt.py` 的 `SetCriterionHOI` | 物体 CE + 交互 focal + 框 L1/GIoU + LSG |

---

## 3. 关键张量形状

以 `hidden_dim=256`、`Nq=64`、`global_token=32`、HICO-DET 为例：

| 变量 | 形状 | 含义 |
|---|---|---|
| `src` | `[bs, C, h, w]` | CNN 骨干最后一层特征 |
| `query_embed_h` / `query_embed_o` | `[Nq, D]` | 人体/物体查询，跨图像共享 |
| `query_embed_global` | `[32, D]` | 全局查询 |
| `open_src` | `[h*w+32, bs, D]` | 早期融合后的 token |
| `memory` | `[h*w+32, bs, D]` | 编码器输出 |
| `h_hs` / `o_hs` | `[layers, bs, Nq, D]` | 检测解码器输出 |
| `inter_hs` | `[layers, bs, Nq, D]` | 交互解码器输出 |
| `attn_map_list` | `[layers, bs, Nq+32, keys]` | 每层 cross-attention |
| `attn_map_inter` | `[bs, Nq, 256]` | 上采样并压缩后的交互注意力图 |
| `hoi_attn_bias_qformer` | `[bs, 32+64, 1+256+32+64]` | Q-Former 的 attention bias |
| `hoi_attn_bias_clip` | `[bs, 16, 353, 353]` | ViT 的 attention bias（16 个头） |
| `pred_sub_boxes` | `[bs, Nq, 4]` | 人框 `(cx, cy, w, h)` |
| `pred_obj_boxes` | `[bs, Nq, 4]` | 物框 `(cx, cy, w, h)` |
| `pred_obj_logits` | `[bs, Nq, 81]` | 80 类 + background |
| `pred_hoi_logits` | `[bs, Nq, 600]` | 600 个 HOI 类别 |

---

## 4. 一个 batch 在代码里的完整路径

```
main.py
  └─ engine.train_one_epoch
       ├─ dataset.__getitem__：图像 + target（框、标签、blip_inputs、llava_captions）
       ├─ model(samples, targets, not_tensor_targets, epoch)
       │    ├─ backbone → src
       │    ├─ blip_model_opt.image_encode → blip_image_embed
       │    ├─ transformer → h_hs/o_hs/inter_hs/attn_map_inter
       │    ├─ 构造 ABG mask
       │    ├─ blip_model_opt.forward_with_prompt → opt_loss + VLM 特征
       │    ├─ 融合 inter_hs + VLM 特征
       │    └─ 文本余弦分类 → pred_*_logits
       ├─ criterion(outputs, targets)
       │    ├─ Hungarian matcher
       │    ├─ 物体 CE + 交互 focal + 框 L1/GIoU
       │    └─ 合并 opt_loss（LSG）
       └─ backward + optimizer.step
```

---

## 5. 调试建议

### 5.1 先确认开关

- `--with_clip_label`：交互分类用文本嵌入；
- `--with_obj_clip_label`：物体分类用文本嵌入；
- `--del_unseen`：训练时是否从文本矩阵删掉 unseen；
- `--zero_shot_type`：决定评测划分和 unseen 索引。

如果 `with_obj_clip_label` 没开，物体头就是固定 80 类线性层，不会有开放词汇能力。

### 5.2 打印关键形状

建议在 `gen_vlkt.py` 的 `forward` 里临时打印：

```python
print("src", src.shape)
print("attn_map_inter", attn_map_inter.shape)
print("qformer bias", hoi_attn_bias_qformer.shape)
print("clip bias", hoi_attn_bias_clip.shape)
print("pred_obj_logits", outputs_obj_class.shape)
print("pred_hoi_logits", outputs_hoi_class.shape)
```

### 5.3 检查 mask 的数值

- `0`：不改变原注意力；
- `-10000`：屏蔽该 key；
- 如果 bias 的绝对值太小（例如 < 0.01），ABG 对注意力的影响会很弱，可以检查 `embed2qformer_mask` 的输出尺度。

### 5.4 检查 warm-up

`gen_vlkt.py` 里 `if warm_up and epoch == 0: inter_hs = inter_hs` 表示第 0 个 epoch 不融合 VLM 特征。调试时如果发现 VLM 没起作用，先确认 epoch 是否已经大于 0。

---

## 6. 阅读时需要注意的实现细节

1. **Early Fusion 的实际输入**：代码里 `open_src` 拼接的是 CNN 特征 + Q-Former 的 32 个全局查询，不是 ViT 的 256 个 patch。论文正文的表述更抽象，具体以代码为准。
2. **`blip_ln` 的 768 是硬编码**：如果换成不同维度的 VLM，需要同步修改。
3. **`forward_pre` 分支**：`TransformerDecoderLayerAttnMap.forward_pre` 里对 `multihead_attn` 的返回值多取了一个 `[0]`，如果启用 `pre_norm=True`，建议先检查这一分支。
4. **大量注释掉的实验分支**：代码里保留了 LoRA、LLaVA caption、V-COCO 等分支，阅读时以当前启用的分支为准。
5. **`num_sp_token=64`、`global_token=32` 是硬编码**：与论文设置一致；改配置时要同步改 mask 构造。

---

## 7. 一句话总结

> 先看 `main.py` 的开关，再看 `engine.py` 的循环，然后沿着 `dataset → gen_vlkt.forward → gen.py → blip2_opt.forward_with_prompt → Qformer` 这条线读。ABG 的关键在 `gen_vlkt.py` 的 mask 构造和 `Qformer.py` 的 `attention_scores + attention_mask`；LSG 的关键在 `forward_with_prompt` 的 token 级 CE。
