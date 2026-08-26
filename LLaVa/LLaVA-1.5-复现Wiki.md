# LLaVA-1.5 简易复现 · 完整解读 Wiki

> 从零手写一个"缩小版" LLaVA-1.5，跑通 **图片 → 视觉编码 → 投影对齐 → 语言模型 → 自回归生成** 的完整闭环。
> 本文档面向"想真正搞懂 LLaVA 内部发生了什么"的读者：每一步都给出**数据流向、张量维度变化、对应论文细节**。

---

## 1. 项目概览

### 1.1 我们复现了什么

| 真实 LLaVA-1.5 | 本项目（玩具版） | 对应关系 |
|---|---|---|
| CLIP ViT-L/14（24 层, dim 1024, 336px, patch14 → 576 patches） | `CLIPViT`（**4 层**, dim 1024, 336px, patch14 → 576 patches） | 层数砍掉，其余保真 |
| MLP-2x 投影器（Linear→GELU→Linear, 1024→4096） | `MLPProjector`（完全相同） | **零缩小** |
| Vicuna-7B（32 层, dim 4096, SwiGLU 11008） | `LLaMA`（**4 层**, dim 4096, SwiGLU 11008） | 层数砍掉，内部机制保真 |
| sentencepiece 词表 32000 | 手写字符级词表 **100** | 大幅缩小 |
| 两阶段训练（对齐 + SFT） | 单阶段端到端 SFT | 简化（原因见 §6.5） |

**原则**：只减少"重复的部分"（层数、词表），**不改变任何架构细节**（RoPE、RMSNorm、SwiGLU、因果掩码、视觉 token 插入、答案 mask）。

### 1.2 项目文件结构

```
LLaVa/
├── tokenizer.py   # 字符级 tokenizer：文本 <-> token id
├── vision.py      # CLIP 风格 ViT：图片 -> 576 个视觉 token
├── projector.py   # MLP-2x：1024 维视觉特征 -> 4096 维语言空间
├── llm.py         # LLaMA 风格解码器：RoPE + RMSNorm + SwiGLU
├── model.py       # 组装 LLaVA：视觉 token 插入文本序列
├── data.py        # 合成数据（几何图形）+ 模板 + labels
├── train.py       # 训练循环（冻结 ViT）
├── eval.py        # 自回归生成测试
└── LLaVA-1.5-复现Wiki.md  # 本文档
```

### 1.3 依赖

- Python 3.10+
- PyTorch 2.x（CPU 可跑，GPU 更快）
- numpy

---

## 2. 整体架构与数据流总览

```
┌─────────────────────────────── 图片路径 ───────────────────────────────┐
│   images (B, 3, 336, 336)                                             │
│      │  vision.py: CLIPViT（patch14 -> 24×24=576 patches）            │
│      ▼                                                                 │
│   (B, 577, 1024)  ← 576 patch + 1 CLS，再丢掉 CLS                     │
│      │  x[:, 1:]                                                       │
│      ▼                                                                 │
│   (B, 576, 1024)                                                       │
│      │  projector.py: MLP-2x                                           │
│      ▼                                                                 │
│   (B, 576, 4096)  ← 视觉 token，维度已对齐 LLM 词嵌入                  │
└──────────────┬─────────────────────────────────────────────────────────┘
               │
┌──────────────┴─────────────────── 文本路径 ────────────────────────────┐
│   input_ids (B, T)  ← 含 1 个 <image> 占位符                           │
│      │  llm.token_emb                                                   │
│      ▼                                                                 │
│   (B, T, 4096)                                                         │
└──────────────┬─────────────────────────────────────────────────────────┘
               │
               ▼   model.py：扫描序列，把 <image> 展开成 576 个视觉 token
        (B, 576+T-1, 4096)  ← 一条"图片token在前 + 文本token在后"的长序列
               │
               ▼   llm.py：4 层 DecoderLayer（自注意力 + 因果掩码 + RoPE）
        (B, 576+T-1, 100)  ← 每个位置预测"下一个字符"
               │
               ▼   shift + cross_entropy（只算答案部分，ignore -100）
              loss  →  backward  →  更新投影器 + LLM（ViT 冻结）
```

---

## 3. 逐模块深度解读

---

### 3.1 `tokenizer.py` — 字符级 Tokenizer

**作用**：文本和 token id 之间的翻译器。LLM 只能吃 id，必须先把字符串转成 id 序列；输出时再转回字符串。

**词表设计（id 分配顺序决定了后面所有维度）**：

| id 范围 | 内容 |
|---|---|
| 0~4 | 特殊 token：`<pad>` `<bos>` `<eos>` `<image>` `<unk>` |
| 5~99 | 可打印 ASCII（`chr(32)`~`chr(126)`，95 个） |

- `<image>`：**不是真文本**，是占位符。`model.py` 会把它的位置替换成 576 个视觉向量。
- `<bos>`/`<eos>`：序列起止标记。
- `<pad>`：batch 内对齐长度用。

**数据变化示例**：

```
"<bos>red circle<eos>"
  → [1, 87, 88, 89, 32, 98, 88, 102, 108, 101, 2]
      │  └─ r ──┘   └──── "circle" 的 6 个字符 ────┘  │
    <bos>                                        <eos>
  （'r'=114, 偏移-32 得下标 82，再 +5 个特殊token → 87）
```

**`encode` 逻辑**：逐字符扫描，优先匹配特殊 token（整体占 1 个 id），其余按单字符映射；未收录字符回落 `<unk>`。

**`decode` 逻辑**：id → 字符，特殊 token 原样还原；`-100`（loss mask 专用值）安全跳过。

---

### 3.2 `vision.py` — CLIP 风格 ViT 视觉编码器

**对应 LLaVA-1.5 的 Vision Tower**（真身 CLIP ViT-L/14）。职责：**把图片变成一串"视觉 token"**。

**数据流**：

```
x (B, 3, 336, 336)
  │ PatchEmbed: Conv2d(3, 1024, kernel=14, stride=14)
  ▼ (B, 1024, 24, 24) → flatten(2) → (B, 576, 1024)
  │ 每个 14×14 像素块 → 1 个 1024 维向量
  │ 序列头部拼 CLS token → (B, 577, 1024)
  │ + 可学习位置编码 pos_embed(1, 577, 1024) → (B, 577, 1024)
  │ 4 × Block：x = x + Attn(LN(x))；x = x + MLP(LN(x))
  ▼ (B, 577, 1024)
  │ ln_out（最终 LayerNorm）
  ▼ return x[:, 1:]   ← 丢掉 CLS，只留 576 个 patch 特征（LLaVA-1.5 独有！）
(B, 576, 1024)
```

**关键细节**：

1. **PatchEmbed 用 Conv2d**：核大小 = 步长 = patch，一步完成"切块 + 线性映射"。
2. **CLS token**：可学习的全局代表，与所有 patch 做注意力；CLIP 用它做分类。**但 LLaVA-1.5 输出时把它丢掉**——它用 576 个 patch 的"网格特征"，这是与 LLaVA-1.0（用 CLS）的标志性区别。
3. **pre-norm + 残差**：每个 Block 先 LayerNorm 再进注意力/MLP，输出加回残差。**有 bias**（CLIP 风格）。
4. **dropout**：注意力权重 dropout 0.1（作用于 softmax 后）、MLP 输出 dropout 0.1（作用于加残差前）。真实 CLIP 默认 0.0，玩具训练用 0.1 更稳。

---

### 3.3 `projector.py` — 视觉-语言投影器（Vision Projector）

**对应 LLaVA-1.5 的 MLP-2x**，是 1.5 与 1.0 的区别（1.0 只有一层线性）。职责：把 1024 维视觉向量**翻译**成 LLM 能懂的 4096 维"伪词嵌入"。

```
(B, 576, 1024) → Linear(1024→4096) → GELU → Linear(4096→4096) → (B, 576, 4096)
```

- **逐 token 独立**：576 个视觉 token 各自过同一个 MLP，互不干扰。
- 参数约 2100 万，**与真实实现零差距**（这步没有缩小）。

---

### 3.4 `llm.py` — LLaMA 风格解码器

**对应 LLaVA-1.5 的 Language Model**（真身 Vicuna-7B）。职责：读序列、预测下一个 token。与 ViT 最大的不同：**Decoder 架构（因果）+ LLaMA 三件套（RMSNorm、RoPE、SwiGLU，全无 bias）**。

**数据流**：

```
input_ids (B, N) ──token_emb──▶ (B, N, 4096)
   （或直接输入 inputs_embeds，LLaVA 走这条路）
  │ 4 × DecoderLayer：
  │    x = x + Attn(RMSNorm(x), RoPE(q,k), 因果掩码)
  │    x = x + SwiGLU(RMSNorm(x))
  ▼ (B, N, 4096)
  │ RMSNorm → llm_head(4096→100)
  ▼ logits (B, N, 100)   ← 每个位置一个"下一个字符"的 100 维打分
```

**三个核心机制**：

**① RMSNorm（只缩放，不居中）**

```
LayerNorm:  y = (x - μ) / sqrt(σ² + ε) * γ + β
RMSNorm:    y = x / RMS(x) * γ,  RMS(x) = sqrt(mean(x²) + ε)
```

LLaMA 认为"居中"不重要，"缩放"才关键；去掉均值省算力且不掉效果。没有 bias。

**② RoPE（旋转位置编码）**

把 Q/K 按位置**旋转**，让"位置差"编码进点积，注意力天然随距离衰减，且能外推。

实现分三步：
1. `precompute_rope_cache`：算好所有位置的 cos/sin 查找表 `(max_seq, head_dim)`。关键步骤 `freqs = torch.cat((freqs, freqs), dim=-1)`——因为 `rotate_half` 按"前半/后半"配对，前后两半必须用同一组角度。
2. `rotate_half`：构造 `(-x₂, x₁)`，配合旋转矩阵 `[[cos,-sin],[sin,cos]]`。
3. `apply_rope`：`q' = q·cos + rotate_half(q)·sin`，对 q 和 k 做、v 不做（v 是内容，位置只影响 q·k）。

**③ SwiGLU（门控 FFN）**

```
普通 FFN:   y = W₂ · GELU(W₁ x)
SwiGLU:    y = W_down · [ silu(W_gate x) ⊙ (W_up x) ]
```

`silu(x) = x·sigmoid(x)` 是"软开关"（0~1 之间），逐元素乘在候选值上，网络学会"选择性放行"。比普通 FFN 多一个 `gate` 矩阵（3 个 dim×hidden）。

**因果掩码**：`torch.triu(-inf, diagonal=1)`，位置 i 只能看 j ≤ i，不能偷看未来。

---

### 3.5 `model.py` — 组装完整的 LLaVA（全项目核心）

**职责**：把 ViT + 投影器 + LLM 拼起来，实现论文里最关键的操作——**视觉 token 插进文本序列**。

**数据流**：

```
images (B, 3, 336, 336)  +  input_ids (B, T)
  │  visual = projector(vision_tower(images))   → (B, 576, 4096)
  │  text   = llm.token_emb(input_ids)          → (B, T, 4096)
  │
  │  扫描序列：遇到 <image> 就用整段 visual[b] 替换（1 个占位符 → 576 个向量）
  ▼
embeds (B, 576+T-1, 4096)   ← 每条样本长度不同，右侧 pad 到 batch 内最长
  │
  ▼  llm(inputs_embeds=embeds)
logits (B, max_len, 100)
  │
  ▼  仅训练时：labels 同步展开（视觉/问题/pad 位置填 -100）
shift_logits = logits[:, :-1]      (B, max_len-1, 100)
shift_labels = labels[:, 1:]       (B, max_len-1)
loss = cross_entropy(..., ignore_index=-100)
```

**训练 / 生成双模式**：`labels=None` 时只返回 logits（生成用）；有 labels 时返回 `(logits, loss)`。

**三个容易错的细节**：
1. **展开是对齐的**：embedding 和 labels 必须按**同样的展开规则**处理，否则错位；
2. **pad 用 pad_id**：input_ids 填充用 `tok.pad_id`（0），labels 填充用 `-100`——两者不能混；
3. **labels 展开只发生在训练时**：视觉部分填 `-100`，答案部分保留真实 id。

---

### 3.6 `data.py` — 合成数据 + LLaVA 模板 + labels

**对应 LLaVA 的数据管线**（真实版是 LLaVA-665K 指令数据；我们合成几何图形数据）。

**图片生成**（numpy 画图，无需任何图片库）：

| 属性 | 取值 | 生成方式 |
|---|---|---|
| 形状 | circle / square / triangle | 掩码布尔数组（§数据里 `r` 控制大小） |
| 颜色 | red / green / blue | RGB 三通道涂色 |
| 位置 | left / center / right | 圆心/中心左右偏移 `size//4` |

三角形公式 `(yy - yc) <= (r - abs(xx - xc))`：顶点在 `(xc, yc-r)`、底边在 `(xc±r, yc)` 的等腰三角形。

**Vicuna 模板**（LLaVA-1.5 的对话格式，`<image>` 放 USER 消息开头）：

```
<bos><image>USER: What is the shape and color? ASSISTANT: red circle<eos>
```

**labels 生成**：答案部分 = 真实 token id，其余 = `-100`（被 CE 忽略）。

**collate_fn**：
- images 直接 `torch.stack` → `(B, 3, 336, 336)`；
- input_ids 右侧 pad 到 batch 最长，填充 `pad_id`；
- labels 右侧 pad 到同长度，填充 `-100`。

---

### 3.7 `train.py` — 训练循环

**对应 LLaVA-1.5 的 Stage 2 端到端微调**（我们跳过 Stage 1，原因见 §6.5）。

**关键设计**：

```python
for p in model.vision_tower.parameters():
    p.requires_grad = False          # LLaVA-1.5：冻结视觉编码器！
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr)
```

- **ViT 全程冻结**（论文明确说明），只有投影器 + LLM 更新——这也是真实 LLaVA 显存可控的原因；
- 每个 step：`forward → loss.backward() → optimizer.step()`；
- 达到总步数后保存 `llava_toy.pt`。

**超参数注意**：参考代码初始 `lr=1e-3`。真·LLaVA-1.5 的 Stage 2 用 `2e-5`。对本玩具，`lr=1e-3` 偏大（loss 会震荡，见 §7 踩坑记录），建议 `1e-4` 并增大步数。

---

### 3.8 `eval.py` — 自回归生成测试

**职责**：用训练好的权重，喂一张**新图**，看模型能不能自己"看图说话"。

**数据流**：

```
image (3, 336, 336) + 固定 prompt: <bos><image>USER: What is the shape and color? ASSISTANT:
  │ 循环（每步一次完整 forward）：
  │    logits = model(图, 当前序列)[0, -1]   ← 只看最后位置的分布
  │    next_id = argmax(logits)              ← 贪心采样
  │    拼到序列末尾
  ▼ 直到 <eos> 或 max_new 步
输出完整文本 → 提取 ASSISTANT: 之后的内容 → 与真实答案比对
```

**注意**：这是朴素的自回归（无 KV-cache），每生成一个字符都重算整条 576+ 序列，慢但概念清晰。真实推理会缓存 K/V 加速。

---

## 4. 维度变化速查表

| 阶段 | 张量 | 形状 |
|---|---|---|
| 输入图片 | `images` | `(B, 3, 336, 336)` |
| ViT 前 | patch embedding 后 | `(B, 576, 1024)` |
| ViT 后（丢 CLS） | `visual` | `(B, 576, 1024)` |
| 投影后 | 视觉 token | `(B, 576, 4096)` |
| 文本嵌入 | `text` | `(B, T, 4096)` |
| **展开拼接** | **`embeds`** | **`(B, 576+T-1, 4096)`** |
| LLM 输出 | `logits` | `(B, 576+T-1, 100)` |
| loss 输入 | `shift_logits` / `shift_labels` | `(B·(N-1), 100)` / `(B·(N-1),)` |

---

## 5. 核心机制专题

### 5.1 next-token prediction 与 shift

**铁律**：位置 `i` 的 logits 预测位置 `i+1` 的 token。

```
logits:   L[0]  L[1]  ...  L[N-2]  L[N-1]     ← 丢掉 L[N-1]（无下一 token 可监督）
           ↓     ↓          ↓
labels:  lab[1] lab[2] ... lab[N-1]            ← 丢掉 lab[0]（它是输入，不是输出）
```

- 丢掉 `L[N-1]`：它的"下一个 token"是序列末尾之后，不存在/是 padding；
- 丢掉 `lab[0]`：第一个 token（通常是 `<bos>`）是给定输入，**从来没有 logits 预测过它**——不是"因为 bos"，而是"因为它是第一个"。

### 5.2 loss masking（`-100`）

LLaVA 的 SFT 只对 **ASSISTANT 之后的答案** 算 loss。实现：labels 非答案位置填 `-100`，`cross_entropy(ignore_index=-100)` 自动跳过。图片、问题、padding 的 token 都不产生梯度。

### 5.3 `F.cross_entropy(logits_2d, labels_1d)` 是怎么算的

**逐行独立计算，行与行无交互**。`logits` 的第 `i` 行（100 维打分）对应 `labels` 的第 `i` 个值（真实字符 id）：

```
对每个位置 i：
  p = softmax(logits[i])[labels[i]]     # 取"真值字符"的预测概率
  CE = -log(p)                          # 预测越准，loss 越小
最终 loss = 所有有效位置 CE 的平均      # -100 的位置跳过
```

展平是"按行序"的，`reshape` 不改内存顺序，所以二维/一维配对天然对齐。

### 5.4 为什么冻结 ViT

1. **保护预训练权重**：CLIP 的特征已经很通用，微调可能破坏它（LLaVA-1.5 论文发现训 ViT 反而不稳/掉点）；
2. **省显存**：ViT 也参与反向会多出大量激活值。

### 5.5 为什么我们跳过 Stage 1（特征对齐）

- Stage 1 与 Stage 2 的 **loss 函数完全相同**（都是"只算答案的 next-token CE"）；
- Stage 1 存在是为了**保护预训练的 LLM**（随机投影器的噪声会冲乱它）——但我们是**随机初始化**，没有需要保护的知识；
- 所以单阶段端到端对这个玩具完全够用。

---

## 6. 踩坑记录（学习价值极高）

这些坑是"测试通过但实际错误"的典型，复习时值得重看：

| # | 坑 | 表现 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | `CHARS = range(32, 117)` | "跑通了"但 `u~z` 全部变 `<unk>` | ASCII 范围少了 `u~z` | `range(32, 127)` |
| 2 | encode 漏 `ids.append` | 特殊 token 静默消失 | 只推进指针没写 id | 补上 append |
| 3 | Block 丢残差连接 | 训练不收敛 | 改写时丢了 `x + ...` | 补回残差 |
| 4 | `num_patches = (img//patch)*(img%patch)` | pos_embed 变成 `(1,1,1024)` 但**不报错** | 乘了余数而非平方 | `(img//patch)**2` |
| 5 | 测试图尺寸 ≠ img_size | 广播维度报错 | pos_embed 与输入不匹配 | 两者统一 |
| 6 | RoPE 缺 `cat(freqs,freqs)` | `q(128) * cos(64)` 维度报错 | rotate_half 按前后半配对需相同角度 | 复制一份 |
| 7 | `loss, _ = model(...)` 解包反了 | `backward` 报"只支持标量" | forward 返回 `(logits, loss)` | 改成 `_, loss = model(...)` |
| 8 | labels 切片 `len(input_ids)` | **不报错但 labels 错位** | list 切片赋值不要求长度相等 | 用 `len(ans_ids)` |
| 9 | collate 填充 `-100` 给 input_ids | 训练时 embedding 越界 | `-100` 是 mask 值不是 token | 用 `tok.pad_id` |
| 10 | 三角形公式占位 | 画出斜线 | 几何公式没写 | `(yy-yc) <= (r - abs(xx-xc))` |

**核心教训**：张量 shape 对了 ≠ 逻辑对了。很多 bug 在"恰好等长/恰好不触发 pad"时被掩盖。

---

## 7. 训练结果参考

一次 `lr=1e-3, 30 步` 的真实输出：

```
step    0 loss = 4.4641    ← 接近理论初始值 -ln(1/100) ≈ 4.605
step    5 loss = 6.1347
step   10 loss = 11.8486   ← 尖峰：lr 过大，震荡
step   15 loss = 7.1824
step   20 loss = 3.9254
step   25 loss = 3.4117
finished, final loss = 3.5357
```

**判读**：链路通了（loss 能下降），但 `lr=1e-3` 对这个任务偏大（尖峰明显）。对"形状+颜色"这种简单任务，收敛后 loss 应远低于 3.5。**建议 `lr=1e-4` + 200 步**，观察 loss 平滑下降再进 eval。

---

## 8. 与真实 LLaVA-1.5 的对应关系

| 论文/官方组件 | 本项目文件 | 保真度 |
|---|---|---|
| Vision Tower (CLIP ViT-L/14) | `vision.py` | 结构保真，层数 24→4 |
| Vision Projector (MLP-2x) | `projector.py` | 100% |
| Language Model (Vicuna-7B) | `llm.py` | 结构保真，层数 32→4，词表 32000→100 |
| 视觉 token 插入 | `model.py` 展开逻辑 | 100% |
| 答案 loss masking | `model.py` shift + `-100` | 100% |
| 对话模板 (Vicuna) | `data.py` `build_sample` | 简化（无 system prompt） |
| Stage 1 对齐 | 省略 | 见 §5.5 |
| Stage 2 SFT | `train.py` | 100%（ViT 冻结） |
| 推理 | `eval.py` | 100%（无 KV-cache 优化） |

---

## 9. 已知待修正 / 可改进项

- [ ] `eval.py` 默认 `ckpt=""` 会加载失败，需传入训练产出的 `llava_toy.pt`
- [ ] `eval.py` 中 `map_location='cuda'` 在 CPU 机器上需改为 `'cpu'`
- [ ] `eval.py` 答案比对顺序（`{shape} {color}`）与 `data.py`（`{color} {shape}`）不一致
- [ ] 训练超参：`lr` 建议 `1e-4`，步数建议 100+
- [ ] 可加：KV-cache 推理加速、beam search、更真实的图片数据（COCO 子集）

---

## 10. 扩展阅读方向

1. LLaVA 论文：*Visual Instruction Tuning*（1.0 提出视觉 token 插入）
2. LLaVA-1.5 论文：*Improved Baselines with Visual Instruction Tuning*（patch 特征、MLP-2x、两阶段）
3. RoPE 原论文：*RoFormer: Enhanced Transformer with Rotary Position Embedding*
4. SwiGLU：*GLU Variants Improve Transformer*
5. 官方仓库：`haotian-liu/LLaVA`（看 `llava/model/language_model/llava_llama.py` 的展开逻辑与本项目对比）
