# BC-HOI 接入 ROI-align + Q-Former 物体头的修改方案

> 目标：在 BC-HOI 的基础上，用"ViT patch 网格 → ROI-align → 相对位置编码 → 拼接全局 cls → Q-Former 新查询 → 768 维输出 → 文本余弦分类"替换原有的物体分类头。
>
> 本文档只是**修改方案**，不是代码补丁。所有改动都建议在新模型目录中完成，保留原始 BC-HOI 作为对照。

---

## 0. 总体改动

### 0.1 改动前

当前 `blip4hoi_clip_qformer_attn_opt_loss` 变体的物体分类路径是：

```
检测解码器 o_hs
   │
   ├─► obj_bbox_embed ─► 物体框
   │
   └─► obj_class_fc ─► normalize
                           │
                           ▼
                   与 obj_clip_label 余弦 → 物体 logits
```

### 0.2 改动后

```
检测解码器 o_hs
   │
   ├─► obj_bbox_embed ─► 物体框（不变）
   │
   └─► ROI 物体头（新增）
           │
           ├─ 用物体框从 ViT patch 网格做 ROI-align
           ├─ 加相对位置编码
           ├─ 拼全局 cls
           ├─ Q-Former 新查询 → E_roi ∈ R^{768}
           └─ 与 obj_clip_label 余弦 → 物体 logits
```

### 0.3 不改的部分

- 框回归头 `hum_bbox_embed` / `obj_bbox_embed`；
- 检测解码器、交互解码器；
- ABG 的 attention bias 构造；
- LSG 的 token 级损失；
- 匈牙利匹配、`SetCriterionHOI` 的框/交互损失；
- `PostProcessHOITriplet` 和评测脚本。

---

## 1. 文件级修改清单

| 文件 | 位置 | 要做什么 |
|---|---|---|
| `main.py` | `get_args_parser` | 新增 ROI 物体头开关与超参 |
| `exp/train_hico.sh` | 启动参数 | 打开新开关 |
| `datasets/hico.py` | `__getitem__` | 补充/确认 224×224 BLIP 坐标系下的物体框 |
| `LAVIS-modified/lavis/models/blip2_models/blip2_opt.py` | `forward_with_prompt` 或新增方法 | 暴露 `ln_vision` 后的 ViT patch 网格与 cls |
| `LAVIS-modified/lavis/models/blip2_models/Qformer.py` | `BertModel.forward` | 一般不用改，用新的 `query_embeds` 调用即可 |
| `models/blip4hoi_clip_qformer_attn_opt_loss/gen_vlkt.py` | `__init__` | 新增 ROI 查询、位置编码、温度等参数 |
| 同上 | `forward` | 计算 ROI 特征并替换 `pred_obj_logits` |
| `models/__init__.py` | `build_model` | 如果复制新模型目录，注册新模型名 |
| `engine.py` | 训练/评测 | 接口不变则不用改 |

---

## 2. 新增参数

在 `main.py` 中新增：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_roi_obj_head` | `False` | 是否启用 ROI 物体头 |
| `--roi_size` | `4` | ROI-align 输出网格边长 `r` |
| `--roi_query_num` | `1` | 每个物体的 Q-Former 查询数 `m` |
| `--roi_train_boxes` | `gt` | 训练时用真值框 `gt` 还是预测框 `pred` |
| `--roi_freeze_qformer` | `True` | 是否冻结 Q-Former 主体 |
| `--roi_use_lora` | `False` | 是否给 Q-Former cross-attention 加 LoRA |
| `--roi_pos_embed` | `learned` | `learned` 或 `sine` |
| `--roi_logit_scale_init` | `4.0` | 余弦温度初值 |

`--roi_train_boxes` 很关键：

- `gt`：训练时用真值物体框，先单独验证分类头；
- `pred`：训练时用检测器预测框，端到端更真实，但会受框误差影响。

建议先跑 `gt`，再跑 `pred`。

---

## 3. 坐标对齐（最容易出错）

ROI-align 必须知道框在 **BLIP-2 的 224×224 输入**里的位置，而不是原图坐标。

在 `datasets/hico.py` 中：

```
img_0, target_0 = self._transforms[0](img, target)
img, target = self._transforms[1](img_0, target_0)
blip_inputs = self.vis_processors(img_0)
```

`blip_inputs` 是对 `img_0` 再缩放到 224 的结果。因此需要：

1. 确认 `target['boxes']` 的格式是归一化 `cxcywh` 还是像素坐标；
2. 如果是归一化框，直接映射到 patch 网格：

$$
x_p = c_x \cdot W_p,\qquad y_p = c_y \cdot H_p
$$

3. 如果是像素框，先除以 `img_0` 的宽高，再映射；
4. 建议在 dataset 里额外保存一份 `target['blip_boxes']`，避免在模型里反复推断坐标系。

---

## 4. 暴露 ViT patch 网格

在 `blip2_opt.py` 的 `forward_with_prompt` 中，当前是：

```
image_embeds = self.ln_vision(
    self.visual_encoder(image, clip_image_atts=clip_image_atts, clip_prompt=clip_prompt))
```

`image_embeds` 的形状是 `[bs, 1+Np, d]`：

- 第 0 个 token 是 cls；
- 后面 Np 个是 patch；
- BLIP-2 ViT-g/14、224 输入时：`Np=16×16=256`，`d=1408`。

建议新增一个方法（或给 `forward_with_prompt` 增加返回值）：

```
extract_vision_grid(image, clip_image_atts, clip_prompt)
    → image_embeds   # [bs, 1+256, 1408]
```

在模型侧 reshape：

```
patch_grid = image_embeds[:, 1:, :].reshape(bs, 16, 16, 1408)
cls_token  = image_embeds[:, 0, :]        # [bs, 1408]
```

注意：使用 **`ln_vision` 之后**的视觉特征，与 Q-Former 的输入保持一致。

---

## 5. ROI 物体头

### 5.1 模块参数

在 `gen_vlkt.py` 的 `__init__` 中新增：

```
self.roi_pos_embed    = nn.Parameter(...)   # [1, r*r, 1408] 或 2D sine
self.obj_roi_queries  = nn.Parameter(...)   # [1, m, 768]
self.roi_logit_scale  = nn.Parameter(...)   # 或复用 obj_logit_scale
```

初始化建议：

- `obj_roi_queries` 从预训练 Q-Former 的 `query_tokens` 复制前 `m` 个；
- `roi_pos_embed` 用小的随机值或正弦编码；
- `roi_logit_scale` 初始化为 `log(1/0.07)` 附近。

### 5.2 ROI-align

对每个物体框：

1. 框映射到 16×16 patch 坐标；
2. 用 `torchvision.ops.roi_align`：
   - 输入：`patch_grid.permute(0,3,1,2)` → `[bs,1408,16,16]`
   - `output_size=(r,r)`
   - 归一化坐标时 `spatial_scale=16`
   - `sampling_ratio=2`
   - `aligned=True`
3. 输出 `[N_boxes, 1408, r, r]`，展平成 `[N_boxes, r*r, 1408]`。

如果不同图的物体数量不同：

- 用 batch index 拼成一个 `roi_align` 调用；
- 或逐图循环后拼接。

### 5.3 相对位置编码

```
roi_tokens = roi_tokens + self.roi_pos_embed[:, :r*r, :]
```

形状仍为 `[N_boxes, r*r, 1408]`。

### 5.4 拼接全局 cls

```
cls = cls_token[batch_idx].unsqueeze(1)      # [N_boxes,1,1408]
roi_tokens = torch.cat([roi_tokens, cls], 1) # [N_boxes,r*r+1,1408]
```

### 5.5 Q-Former 新查询

复用 `self.blip_model_opt.Qformer.bert`：

```
query_embeds = self.obj_roi_queries.expand(N_boxes, -1, -1)  # [N_boxes,m,768]
query_output = Qformer.bert(
    query_embeds=query_embeds,
    encoder_hidden_states=roi_tokens,
    encoder_attention_mask=None,
    return_dict=True,
    with_attn_bias=False,
)
E_roi = query_output.last_hidden_state                     # [N_boxes,m,768]
```

- Q-Former 的 K/V 投影会把 1408 映射到 768，输入 1408 维是正确的；
- `m=1` 时直接取 `[:,0,:]`；
- `m>1` 时做 mean pooling 或 attention pooling。

### 5.6 余弦分类

```
z_obj = normalize(E_roi)                     # [N_boxes,768]
T_obj = normalize(self.obj_adapter(self.obj_clip_label) + self.obj_clip_label)
logits_obj = self.roi_logit_scale.exp() * z_obj @ T_obj.T
```

注意 `obj_clip_label` 的行顺序要和原物体类别（含 background）一致。

---

## 6. 接入 `GEN_VLKT.forward`

当前流程：

```
outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()
...
if with_obj_clip_label:
    outputs_obj_class = ...
else:
    outputs_obj_class = self.obj_class_embed(o_hs)
```

修改方案：

1. 保留 `outputs_obj_coord`；
2. 在它后面计算 ROI 物体 logits；
3. 用 ROI logits 替换 `outputs_obj_class`；
4. `out` 字典里的 `pred_obj_logits` 改为 ROI logits；
5. matcher、`loss_obj_labels`、`PostProcessHOITriplet` 不需要改，因为它们只依赖 `pred_obj_logits`。

训练时：

- `roi_train_boxes=gt`：用 `targets['obj_boxes']` 生成 ROI；
- `roi_train_boxes=pred`：用 `outputs_obj_coord` 生成 ROI。

推理时统一用 `outputs_obj_coord`。

---

## 7. 训练策略

### 7.1 三段式

| 阶段 | 冻结 | 训练 |
|---|---|---|
| A | ViT、Q-Former、文本编码器、检测器、交互分支 | `obj_roi_queries`、`roi_pos_embed`、`roi_logit_scale` |
| B | ViT、Q-Former、文本编码器 | 上述 + 检测解码器最后 1 层 |
| C（可选） | ViT、文本编码器 | 上述 + Q-Former cross-attention LoRA |

阶段 A 最稳，先验证 ROI 头本身有没有用。

### 7.2 损失

最简单的接法：

$$
\mathcal{L}
= \mathcal{L}_{hoi}
+ \mathcal{L}_{obj}
+ \mathcal{L}_{lsg}
$$

其中：

- `L_hoi`：BC-HOI 原有的检测 + 交互损失；
- `L_obj`：ROI 头的物体交叉熵；
- `L_lsg`：LLM 的逐 token 生成损失。

如果保留原来的物体头，可以做成混合：

$$
\text{logits}_{obj}
= \alpha \cdot \text{logits}_{orig}
+ (1-\alpha) \cdot \text{logits}_{roi}
$$

但第一版建议**直接替换**，消融更干净。

### 7.3 优化器分组

`main.py` 现有的 `param_dicts` 按 `backbone`、`visual_projection`、`opt_proj` 分组。新增参数需要单独分一组：

```
roi_params = [p for n,p in model.named_parameters() if 'roi' in n or 'obj_roi' in n]
```

建议学习率：`1e-4`；Q-Former LoRA 用 `1e-5`。

### 7.4 Checkpoint

BC-HOI 原权重没有新增参数，加载时使用 `strict=False`：

- 新参数随机初始化；
- 旧参数从 BC-HOI checkpoint 加载；
- 记录哪些 key 缺失/多余。

---

## 8. 评测

`PostProcessHOITriplet` 不需要改：它只读取 `pred_obj_logits` 和框。

需要确认：

1. `obj_clip_label` 的类别顺序和 `num_obj_classes+1` 一致；
2. background 的索引仍是最后一类；
3. UO 评测时 `hico_unseen_index` 的逻辑不受影响；
4. ROI 头输出维度必须是 81（80 类 + background）。

---

## 9. 张量形状速查

| 变量 | 形状 |
|---|---|
| `o_hs` | `[layers, bs, Nq, 256]` |
| `outputs_obj_coord` | `[layers, bs, Nq, 4]` |
| `patch_grid` | `[bs, 16, 16, 1408]` |
| ROI 输出 | `[N_boxes, 1408, r, r]` |
| ROI token | `[N_boxes, r*r, 1408]` |
| 拼 cls 后 | `[N_boxes, r*r+1, 1408]` |
| `E_roi` | `[N_boxes, m, 768]` |
| `z_obj` | `[N_boxes, 768]` |
| `logits_obj` | `[N_boxes, 81]` |

---

## 10. 常见坑

1. **坐标系错**：框是原图坐标，却拿去 224×224 的 patch 网格做 ROI-align。必须先归一化。
2. **`roi_align` 的 `spatial_scale` 错**：归一化坐标用 `16`，像素坐标用 `16/224`。
3. **batch index 错**：`roi_align` 的 boxes 第一列必须是 batch 索引，不是坐标。
4. **Q-Former 输入维度错**：K/V 是 1408 维，Q-Former 内部会投影到 768，不要提前投影。
5. **`encoder_attention_mask` 形状错**：`with_attn_bias=False` 时传 `None` 或 `[bs, seq]`，不要传 4D bias。
6. **背景类**：`obj_clip_label` 有 81 行（80 类 + background），ROI logits 也要 81 维。
7. **DDP unused params**：新参数如果某些 batch 没用上，保持 `find_unused_parameters=True`。
8. **显存**：`Nq=64 × batch` 的 ROI Q-Former 很吃显存；先冻结 Q-Former，必要时只对 top-K 物体框计算 ROI。

---

## 11. 最小可行实施顺序（MVP）

1. 在 `blip2_opt.py` 暴露 `image_embeds`（patch 网格 + cls）。
2. 在 `gen_vlkt.py` 新增 ROI-align + 位置编码 + 全局 cls，先用 `gt` 框。
3. 新增 `obj_roi_queries`，调 Q-Former 得到 `E_roi`。
4. 用 `E_roi` 与 `obj_clip_label` 算余弦，替换 `pred_obj_logits`。
5. 跑一个小 batch，打印所有形状，确认 `logits_obj` 是 `[N_boxes,81]`。
6. 用 `gt` 框训练，观察 UO mAP。
7. 切到 `pred` 框，端到端训练。
8. 如果提升不够，再加 Q-Former LoRA 或 CLIP 蒸馏。

---

## 12. 建议的消融矩阵

| 编号 | 配置 | 目的 |
|---|---|---|
| A0 | BC-HOI 原版 | 基线 |
| A1 | ROI Q-Former，m=1，r=4，GT 框 | 验证 ROI 头本身 |
| A2 | + 相对位置编码 | 位置编码的收益 |
| A3 | + 全局 cls | 上下文的收益 |
| A4 | 预测框替代 GT 框 | 端到端真实设定 |
| A5 | m=4 | 多查询的收益 |
| A6 | + Q-Former LoRA | 分布偏移的缓解 |
| A7 | + CLIP 蒸馏 | 对齐质量 |

每组至少 3 个随机种子，报告 UO / UV / UC / seen 四组指标。

---

## 13. 一句话总结

> 改动集中在三处：**`blip2_opt.py` 暴露 ViT patch 网格**、**`gen_vlkt.py` 增加 ROI-align + 新 Q-Former 查询**、**用 ROI logits 替换 `pred_obj_logits`**。框回归、交互分支、LSG、matcher、评测接口都可以保持不变。最容易出错的是**框坐标必须和 224×224 的 BLIP 输入对齐**。
