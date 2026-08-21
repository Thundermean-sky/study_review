# Vision Transformer (ViT) 学习笔记

> 论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*（Dosovitskiy et al., ICLR 2021）
> 本仓库：`modules/transformers.py` 中从零实现的 ViT-Base/16，`test.py` 为最小验证脚本。
> 本文目标：把「论文思想 → 张量流动 → 代码实现」三者对齐，形成一份可复习、可给别人看的 wiki。

---

## 目录

- [Vision Transformer (ViT) 学习笔记](#vision-transformer-vit-学习笔记)
  - [目录](#目录)
  - [1. 一句话概括](#1-一句话概括)
  - [2. 背景：为什么要把 Transformer 搬到 CV](#2-背景为什么要把-transformer-搬到-cv)
  - [3. 整体架构与数据流](#3-整体架构与数据流)
  - [4. 模块拆解](#4-模块拆解)
    - [4.1 Patch Embedding：把图片变成"句子"](#41-patch-embedding把图片变成句子)
    - [4.2 CLS Token 与位置编码](#42-cls-token-与位置编码)
      - [CLS Token（分类向量）](#cls-token分类向量)
      - [位置编码](#位置编码)
      - [初始化](#初始化)
    - [4.3 Multi-Head Self-Attention](#43-multi-head-self-attention)
      - [逐行拆解（ViT-Base：`heads=12`，`head_dim = 768/12 = 64`）](#逐行拆解vit-baseheads12head_dim--76812--64)
    - [4.4 FFN（MLP Block）](#44-ffnmlp-block)
    - [4.5 Encoder Block：Pre-Norm + 残差](#45-encoder-blockpre-norm--残差)
    - [4.6 分类头](#46-分类头)
  - [5. 模型规格与参数量](#5-模型规格与参数量)
  - [6. 归纳偏置（Inductive Bias）的讨论](#6-归纳偏置inductive-bias的讨论)
  - [7. 错误点复习](#7-错误点复习)
  - [8. 如何运行](#8-如何运行)

---

## 1. 一句话概括

**ViT 把一张图切成固定大小的 patch，每个 patch 当作一个"词"，然后原封不动地丢给 NLP 里的标准 Transformer Encoder 做分类。**

它的贡献不在模型结构，而在于证明了一件事：

> 只要预训练数据量足够大，**纯 Transformer 不需要任何卷积的归纳偏置，也能在图像分类上超过最好的 CNN**。

论文原话是 *"...reliance on CNNs is not necessary"*。这句话是 ViT 全部价值的落脚点。

---

## 2. 背景：为什么要把 Transformer 搬到 CV

- 2017 年之后，Transformer 在 NLP 里靠"大模型 + 大数据预训练"一路碾压（BERT / GPT），并且**看不到明显的性能瓶颈**——参数量和数据量加上去，效果就继续涨。
- CV 这边当时的主流仍然是 CNN。此前也有人尝试把 self-attention 引入视觉，但要么是"CNN + attention 混搭"，要么是设计特殊的局部 attention 算子，都需要改造硬件友好性，难以规模化。
- ViT 的做法非常"偷懒"，也正因为偷懒才重要：**几乎不改 Transformer，只改输入**。这样 NLP 那套成熟的扩展经验（scaling law、预训练范式、工程优化）可以直接复用。

核心难点只有一个：Transformer 的复杂度是序列长度的平方 `O(N²)`。
如果把每个像素当作一个 token，224×224 就是 50176 个 token，attention 矩阵直接爆炸。

**ViT 的解法就是 patch 化**：16×16 的像素块合成一个 token，序列长度从 50176 降到 `196 = (224/16)²`，一下子变得可算。这就是标题 "An Image is Worth 16x16 Words" 的含义。

---

## 3. 整体架构与数据流

整体流程（对应代码 `EncoderBlock.forward`）：

```
输入图像 → PatchEmbed → 拼接 CLS Token → 加位置编码
        → N × Transformer Encoder Block → LayerNorm
        → 取出 CLS 位置的向量 → Linear 分类头 → logits
```

以 ViT-Base/16、输入 224×224 为例，张量形状变化如下：

| 阶段 | 操作 | 输出形状 |
| --- | --- | --- |
| 输入 | — | `(B, 3, 224, 224)` |
| Patch 切分 | `Conv2d(3→768, k=16, s=16)` | `(B, 768, 14, 14)` |
| 拉平 | `.flatten(2).transpose(1,2)` | `(B, 196, 768)` |
| 归一化 | `LayerNorm(768)` | `(B, 196, 768)` |
| 拼 CLS | `cat([cls_token, x], dim=1)` | `(B, 197, 768)` |
| 加位置编码 | `x + pos_embed` | `(B, 197, 768)` |
| ×12 Encoder Block | Attention + FFN | `(B, 197, 768)`（形状不变） |
| 最终 LayerNorm | `LayerNorm(768)` | `(B, 197, 768)` |
| 取 CLS | `x[:, 0]` | `(B, 768)` |
| 分类头 | `Linear(768→1000)` | `(B, 1000)` |

注意一个关键性质：**Encoder 部分从头到尾形状不变，都是 `(B, 197, 768)`**。所以 Block 可以无脑堆叠，这也是 Transformer 好 scale 的工程原因之一。

对应主干代码：

```146:161:modules/transformers.py
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.Block(x)

        x = self.norm(x)

        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

---

## 4. 模块拆解

### 4.1 Patch Embedding：把图片变成"句子"

**要解决的问题**：Transformer 接受的是 `(B, N, D)` 的序列，而图片是 `(B, C, H, W)` 的二维网格。需要一个映射：

$$
x \in \mathbb{R}^{H \times W \times C} \longrightarrow x_p \in \mathbb{R}^{N \times (P^2 \cdot C)} \xrightarrow{\ E\ } \mathbb{R}^{N \times D}
$$

其中 `P = 16`，`N = H·W / P² = 196`，`D = 768`。

**论文的描述**是"把每个 patch 展平成 768 维向量（`16 × 16 × 3 = 768`），再乘一个可学习的投影矩阵 `E`"。

**实现上的技巧**：这个「切块 + 展平 + 线性投影」在数学上完全等价于一次 **kernel_size = stride = patch_size 的卷积**。因为卷积核不重叠地滑过图像，每滑一次就是对一个 patch 做一次线性变换。所以代码里一行卷积就搞定：

```22:24:modules/transformers.py
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()
```

```26:35:modules/transformers.py
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size is {H}x{W}, Not match image size {self.img_size[0]}x{self.img_size[1]}"

        x = self.proj(x).flatten(2).transpose(1, 2)

        x = self.norm(x)

        return x
```

拆解 `self.proj(x).flatten(2).transpose(1, 2)`：

1. `proj(x)`：`(B,3,224,224) → (B,768,14,14)`，14×14 就是 patch 的网格排布；
2. `.flatten(2)`：从第 2 维开始拉平，`(B,768,14,14) → (B,768,196)`；
3. `.transpose(1,2)`：换成序列格式，`(B,768,196) → (B,196,768)`，即 **196 个 token，每个 768 维**。

> **注意**：这里用卷积只是为了实现高效，**它不是"用 CNN 提特征"**。这个卷积没有重叠、没有层级、没有池化，它只是一次分块线性投影。ViT 的"无卷积归纳偏置"这一说法并不矛盾。
>
> 另外，`assert` 把输入尺寸写死为 224，说明**位置编码是与固定长度绑定的**，换分辨率必须对 `pos_embed` 做插值（见 §6）。

---

### 4.2 CLS Token 与位置编码

```122:127:modules/transformers.py
        # 类别编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patch + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(proj_drop)
```

#### CLS Token（分类向量）

- 形状 `(1, 1, 768)`，前向时 `expand` 到 batch 维，拼在序列**最前面**，序列长度 196 → 197。
- 思想直接来自 BERT 的 `[CLS]`：它本身不携带任何图像信息，但因为 self-attention 是全局的，**每一层它都会去"查询"所有 patch**，最终变成整张图的全局聚合表示。
- 这样做的好处是：分类头只需要接一个向量，而不用对 196 个 patch 做平均池化，同时避免了"偏向某些 patch"的问题。
- 论文里也对比过 **CLS token vs. Global Average Pooling(GAP)**，结论是**两者效果相当**，只是最优学习率不同；用 CLS 纯粹是为了尽量贴近原始 Transformer。

#### 位置编码

- 形状 `(1, 197, 768)`，与整个序列（含 CLS）逐元素**相加**。
- **为什么必须要**：self-attention 本身是置换不变的（permutation-invariant）——打乱 token 顺序，输出只是跟着换位置，模型无法感知"哪两个 patch 相邻"。而图像的空间结构恰恰是核心信息，所以位置信息必须显式注入。
- 这里用的是**可学习的 1D 位置编码**（learnable 1-D positional embedding），不是 NLP 里常见的 sin/cos 固定编码。
- 论文附录 D.4 做过消融：**不加位置编码掉点明显（约 3%）；而 1D / 2D / 相对位置编码之间差异很小**。作者的解释是：patch 数量只有 196，序列很短，模型自己就能从 1D 索引里学出 2D 邻接关系。论文里的可视化也显示，学出来的位置编码确实呈现出行列相似性的二维结构。

#### 初始化

两者都先用 `torch.zeros` 占位，**但随后被截断正态分布重新初始化**，并不是真的以全零参与训练：

```141:144:modules/transformers.py
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(_init_vit_weight)
```

`std=0.02` 也是沿用 BERT 的习惯。

---

### 4.3 Multi-Head Self-Attention

这是 Transformer 的核心，公式：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

```38:51:modules/transformers.py
class Attention(nn.Module):
    def __init__(self, embed_dim, head, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.heads = head
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // head
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)
```

```53:68:modules/transformers.py
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_weight = attn.softmax(dim=-1)

        x = (attn_weight @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x
```

#### 逐行拆解（ViT-Base：`heads=12`，`head_dim = 768/12 = 64`）

| 步骤 | 形状 |
| --- | --- |
| `x` | `(B, 197, 768)` |
| `self.qkv(x)` | `(B, 197, 2304)` |
| `.reshape(B, N, 3, 12, 64)` | `(B, 197, 3, 12, 64)` |
| `.permute(2, 0, 3, 1, 4)` | `(3, B, 12, 197, 64)` |
| `q / k / v` 各自 | `(B, 12, 197, 64)` |
| `q @ k.transpose(-2,-1)` | `(B, 12, 197, 197)` ← **注意力矩阵** |
| `attn_weight @ v` | `(B, 12, 197, 64)` |
| `.transpose(1,2)` | `(B, 197, 12, 64)` |
| `.reshape(B, N, C)` | `(B, 197, 768)` ← 多头拼接 |
| `self.proj` | `(B, 197, 768)` |

几个要点：

1. **QKV 三合一**：`nn.Linear(768, 768*3)` 一次算完 Q、K、V，再 reshape 切分。相比三个独立 Linear，这样做**减少 kernel launch 次数、提高 GPU 矩阵乘的并行度**，是标准工程优化。数学上完全等价。
2. **多头的本质**：768 维被切成 12 份 64 维，每个头在自己的子空间里独立算 attention，最后拼回来。这让模型能同时关注不同类型的关系（有的头关注邻近 patch，有的关注全局/同类物体）。论文的注意力距离可视化显示：**浅层的头既有局部也有全局的，深层的头几乎全是全局的**——这说明 ViT 自己学出了类似 CNN 的层级感受野，但底层就已经能看全局，这是 CNN 做不到的。
3. **缩放因子 `scale = head_dim^-0.5`**：`q · k` 是 64 个数相加，方差随维度线性增长；不除 `√d_k` 会让 softmax 输入过大，梯度趋近于 0（饱和）。
4. **`.transpose(1,2)` 之后必须 `reshape`**：把 `(B, 12, 197, 64)` 变回 `(B, 197, 768)`，这一步就是"多头拼接"。顺序不能反，否则语义错乱。
5. **复杂度**：attention 矩阵是 `197 × 197`，复杂度 `O(N² · D)`。这也是后续 Swin Transformer 等工作要做窗口注意力的原因。

---

### 4.4 FFN（MLP Block）

```71:84:modules/transformers.py
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output=None, dropout=0.):
        super().__init__()
        output = output or embed_dim

        self.fc_1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.act = nn.GELU()

        self.fc_2 = nn.Linear(hidden_dim, output)

    def forward(self, x):
        return self.dropout(self.fc_2(self.dropout(self.act(self.fc_1(x)))))
```

- 结构就是 `Linear → GELU → Dropout → Linear → Dropout`，先升维再降维，ViT-Base 的 `ffn_ratio=4`，即 `768 → 3072 → 768`。
- **为什么要先放大再缩小**：Attention 本身是各 token 之间的**线性加权组合**，缺少 token 内部的非线性变换能力。FFN 是逐位置（position-wise）独立作用的，中间的宽层 + GELU 提供了模型主要的非线性拟合容量。可以理解成"Attention 负责交换信息，FFN 负责加工信息"。
- 参数量上，FFN 约占每个 Block 的 **2/3**（见 §5），是 Transformer 真正的"存储器"。
- 激活函数用 **GELU** 而非 ReLU：GELU 平滑可导，在 Transformer 系列上经验效果更好。

---

### 4.5 Encoder Block：Pre-Norm + 残差

```87:106:modules/transformers.py
class Block(nn.Module):
    def __init__(self, embed_dim, heads, ffn_ratio, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mlp_dropout=0., norm=nn.LayerNorm):
        super(Block, self).__init__()

        self.norm_1 = norm(embed_dim)

        self.attn = Attention(embed_dim, heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)

        self.norm_2 = norm(embed_dim)

        self.mlp = FeedForward(embed_dim, embed_dim * ffn_ratio, output=embed_dim, dropout=mlp_dropout)

    def forward(self, x):
        x = self.attn(self.norm_1(x)) + x

        x = x + self.mlp(self.norm_2(x))

        return x
```

一个 Block 就两个子层：

$$
z' = \text{MSA}(\text{LN}(z)) + z
$$

$$
z'' = \text{MLP}(\text{LN}(z')) + z'
$$

两个设计点值得强调：

- **Pre-Norm（LN 在子层之前）**：这是 ViT 与原始 Transformer（Post-Norm）的一个重要差别。Pre-Norm 让残差分支成为一条"干净的恒等通路"，梯度可以无衰减地传到底层，**深层网络训练稳定得多，甚至可以不用 learning rate warmup**。这也是现代大模型（GPT / LLaMA）的通用选择。
- **只用 Encoder，没有 Decoder**：分类任务不需要自回归生成，也不需要 causal mask，所以 `Attention.forward` 里的 `mask` 参数在 ViT 中始终为 `None`（本实现中该参数实际未被使用）。

堆叠方式（ViT-Base 是 12 层）：

```129:133:modules/transformers.py
        self.Block = nn.Sequential(*[
            Block(embed_dim, heads, ffn_ratio, qkv_bias, qk_scale, attn_drop, proj_drop, mlp_dropout=0.,
                  norm=norm_layer)
            for _ in range(depth)
        ])
```

---

### 4.6 分类头

```135:139:modules/transformers.py
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()

        # 分类头
        self.head = nn.Linear(embed_dim, num_class) if num_class else nn.Identity()
```

- 最后一层 Block 之后先做一次 **LayerNorm**（Pre-Norm 架构必须在出口补一个 LN，否则输出未归一化）。
- 然后 `x[:, 0]` 取出 **CLS token 对应的 768 维向量**，作为整图表示。
- 接 `Linear(768 → num_classes)` 得到 logits。

关于 `pre_logits`：论文中**预训练阶段**的分类头是「一个隐藏层的 MLP + tanh」，**微调阶段**才简化为单个线性层。本实现直接用 `nn.Identity()` 占位，即只做微调形态的单层分类头。

---

## 5. 模型规格与参数量

论文给出的三档配置：

| 模型 | Layers (depth) | Hidden `D` | MLP size | Heads | Params |
| --- | --- | --- | --- | --- | --- |
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

命名规则 `ViT-B/16`：B = Base，16 = patch size。**patch 越小 → 序列越长 → 计算量越大、效果越好**（ViT-L/16 比 ViT-L/32 强）。

本仓库的工厂函数正好对应 ViT-B/16：

```175:193:modules/transformers.py
def vit_base_patch16_224(pretrained=False, num_classes=1000):
    model = EncoderBlock(
        img_size=224,
        in_c=3,
        num_class=num_classes,
        embed_dim=768,
        heads=12,
        patch=16,
        depth=12,
        ffn_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        norm=nn.LayerNorm,
        distilled=False,
        embed_layer=PatchEmbed,
    )
    return model
```

**参数量粗算**（ViT-Base）：

| 部分 | 计算 | 参数量 |
| --- | --- | --- |
| Patch Embed | `16 × 16 × 3 × 768 + 768` | ≈ 0.59M |
| CLS + Pos | `768 + 197 × 768` | ≈ 0.15M |
| 单个 Block · QKV | `768 × 2304 + 2304` | ≈ 1.77M |
| 单个 Block · Proj | `768 × 768 + 768` | ≈ 0.59M |
| 单个 Block · FFN | `768 × 3072 + 3072 × 768 + bias` | ≈ 4.72M |
| **单个 Block 合计** | | **≈ 7.09M** |
| 12 个 Block | `7.09M × 12` | ≈ 85.1M |
| 分类头 | `768 × 1000 + 1000` | ≈ 0.77M |
| **总计** | | **≈ 86M** ✅ |

可以看到：**参数几乎全在 12 个 Encoder Block 里，其中 FFN 占了约 2/3**。Patch Embedding 和位置编码加起来不到 1%。

## 6. 归纳偏置（Inductive Bias）的讨论

这是理解 ViT 的思想内核，值得单独一节。

**CNN 内建的先验**：
- **局部性（locality）**：卷积核只看邻域；
- **平移等变性（translation equivariance）**：同一个核在全图共享权重；
- **层级结构**：通过堆叠和下采样逐步扩大感受野。

这些先验在小数据上是巨大优势——它们等于免费告诉了模型"图像的规律长什么样"。

**ViT 几乎丢掉了全部这些先验**：
- 只有 patch 切分 和 微调时的位置编码插值 保留了一点点 2D 信息；
- self-attention 是**全局**的，第 1 层就能让任意两个 patch 交互；
- 位置关系不是硬编码的，而是**从零学出来的**。

这带来了一个权衡：

> 少了先验 → 需要更多数据来"补课"；
> 但一旦数据补够了 → 模型能学到比人工先验更优的模式，上限更高。

论文的注意力距离分析证实了这一点：ViT 底层就有一部分注意力头在做全局关注，这是 CNN 结构上不可能做到的。

论文还提出了 **Hybrid（混合）架构**作为折中：用 ResNet 的特征图代替原始图像做 patch embedding。实验显示，在小计算量下 Hybrid 优于纯 ViT，但**随着模型变大，二者差距消失**——再次说明大数据下先验并非必需。

---

## 7. 错误点复习

| 点 | ❌ 错误说法 | ✅ 正确说法 |
| --- | --- | --- |
| 卷积输出形状 | `(B, 768, 114, 14)` | `(B, 768, 14, 14)`，`14 = 224 / 16` |
| cls / pos 初始化 | 初始化为全 0 | 先 `zeros` 占位，随后 `trunc_normal_(std=0.02)` 重新初始化 |
| 预训练数据 | 在 ImageNet 上预训练就能超 CNN | 需要 **ImageNet-21k(14M) / JFT-300M** 级别；只用 ImageNet-1k 时 ViT 弱于 ResNet |
| PatchEmbed 的卷积 | ViT 用 CNN 提特征 | 那只是分块线性投影的高效实现，无重叠、无层级，不构成卷积归纳偏置 |
| LayerNorm 位置 | 和原始 Transformer 一样在子层之后 | ViT 用 **Pre-Norm**，LN 在 Attention/FFN **之前** |
| PatchEmbed 里的 LN | 论文标配 | 原论文/timm 默认**没有**这个 LN；本实现传入了 `norm_layer` 所以启用了它 |

---

## 8. 如何运行

```bash
python test.py
```

`test.py` 做的事：构造 `vit_base_patch16_224(num_classes=1000)`，切到 `eval()` 模式，喂一个随机张量验证前向能跑通。

```12:17:test.py
# ---------- 方式一：随机张量，快速验证架构能否跑通 ----------
# 模型要求输入 (B, C, H, W)，且 H=W=img_size(224)
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():          # 不计算梯度，省显存/内存
    out = model(dummy)
print("随机张量 -> 输出 shape:", tuple(out.shape))   # 期望 (1, 1000)
```

预期输出：

```
随机张量 -> 输出 shape: (1, 1000)
```

文件里注释掉的第二部分是真实图片的完整预处理流程（Resize → ToTensor → Normalize → unsqueeze 加 batch 维 → Top-5 预测），取消注释并填上图片路径即可使用。注意此处**权重是随机初始化的**，预测结果没有意义，仅用于验证流程。

依赖：

```
torch
torchvision
pillow
```

---
